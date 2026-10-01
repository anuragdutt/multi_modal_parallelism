"""torchrun entry point: one (mode, swing, cuts, batch, ctx, variant) measurement, optional correctness check.

Example (inside the container):
  torchrun --standalone --nproc_per_node=4 -m harness.run --mode split22 --swing 0.2 --cuts neutral \
      --batch 16 --ctx 8192 --variant eager --check --out results/raw/dev.csv
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
import uuid
from pathlib import Path

import torch
import torch.distributed as dist

from . import csvio
from .config import FalconH1Dims
from .correctness import compare, cross_rank_max_abs, passes
from .dist import barrier_sync, describe_comm, destroy, init_groups
from .layer import ParallelHybridLayer
from .sharding import neutral_cuts, plan_for, swing_width, symmetric_cuts
from .state import alloc_and_fill, full_states_for_reference, input_hidden, make_decode_ctx
from .timing import OpTimer, time_eager, time_graph
from .weights import load_layer_from_safetensors, random_layer, shard_to_device


def parse_cuts(spec: str, w: int, world: int) -> tuple[tuple[int, ...], str]:
    if spec in ("neutral", "model"):
        return neutral_cuts(w, world), spec  # "model" resolved by the caller when a byte model is available
    if spec.startswith("sym:"):
        return symmetric_cuts(w, int(spec[4:]), world), "grid"
    return tuple(int(x) for x in spec.split(",")), "manual"


def gpu_env(rank: int) -> dict:
    try:
        q = subprocess.run(
            ["nvidia-smi", "--query-gpu=clocks.sm,temperature.gpu", "--format=csv,noheader,nounits", "-i", str(rank)],
            capture_output=True, text=True, timeout=5,
        ).stdout.strip().split(",")
        return {"sm_clock_mhz": int(q[0]), "temp_c": int(q[1])}
    except Exception:
        return {"sm_clock_mhz": -1, "temp_c": -1}


def vllm_commit() -> str:
    try:
        import vllm

        root = Path(vllm.__file__).resolve().parents[1]
        return subprocess.run(["git", "-C", str(root), "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip() or vllm.__version__
    except Exception:
        return "unknown"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True)
    ap.add_argument("--swing", type=float, default=0.0)
    ap.add_argument("--cuts", default="neutral")
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--ctx", type=int, default=512)
    ap.add_argument("--variant", choices=["eager", "graph", "both"], default="eager")
    ap.add_argument("--kv-block", type=int, default=16)
    ap.add_argument("--model-dir", default=os.environ.get("MMP_MODEL_DIR", ""))
    ap.add_argument("--dims-yaml", default="configs/falcon_h1_7b.yaml")
    ap.add_argument("--random-weights", action="store_true")
    ap.add_argument("--layer-idx", type=int, default=0)
    ap.add_argument("--n-warm", type=int, default=20)
    ap.add_argument("--n-iter", type=int, default=100)
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--out", default="results/raw/dev.csv")
    ap.add_argument("--correctness-out", default="results/raw/correctness.csv")
    ap.add_argument("--tag", default="")
    ap.add_argument("--allow-shared-gpu", action="store_true")
    args = ap.parse_args()

    groups = init_groups()
    rank, world, device = groups.rank, groups.world_size, groups.device
    if args.model_dir:
        dims = FalconH1Dims.from_hf(args.model_dir)
    else:
        dims = FalconH1Dims.from_yaml(args.dims_yaml)
    w_cols = dims.intermediate // world
    w = swing_width(args.swing, w_cols) if args.mode == "split22" and args.swing > 0 else 0
    cuts, cuts_src = parse_cuts(args.cuts, w, world)
    plan = plan_for(args.mode, rank, world, dims, args.swing, cuts if w > 0 else None)

    if args.random_weights or not args.model_dir:
        full = random_layer(dims, args.seed)
    else:
        full = load_layer_from_safetensors(args.model_dir, args.layer_idx)
    rw = shard_to_device(full, plan, dims, device)
    ctx = make_decode_ctx(args.batch, args.ctx, args.kv_block, device)
    st = alloc_and_fill(plan, dims, ctx, args.seed, device)
    h = input_hidden(args.seed, args.batch, dims.hidden, device)
    layer = ParallelHybridLayer(plan, rw, dims, groups, st, ctx, device)
    barrier_sync()

    run_id = uuid.uuid4().hex[:8]
    ts = time.strftime("%Y-%m-%dT%H:%M:%S")
    comm_tp, comm_pair = describe_comm(groups.tp), describe_comm(groups.pair)
    commit = vllm_commit()
    base = dict(run_id=run_id, ts=ts, image_tag=os.environ.get("MMP_IMAGE_TAG", ""), vllm_commit=commit,
                mode=args.mode, S=args.swing, w=w, cuts=",".join(map(str, cuts)) if w > 0 else "",
                cuts_src=cuts_src if w > 0 else "", batch=args.batch, ctx=args.ctx, kv_block=args.kv_block,
                comm_tp=json.dumps(comm_tp), comm_pair=json.dumps(comm_pair), tag=args.tag)

    if args.check:
        # one step on fresh states; compare against the fp32 reference computed on rank 0
        out = layer.step(h, OpTimer(enabled=False)).clone()
        torch.cuda.synchronize()
        xr = cross_rank_max_abs(out, world)
        if rank == 0:
            kc, vc, conv, ssm = full_states_for_reference(dims, args.batch, args.ctx, args.seed, device)
            from .reference import decode_step

            ref = decode_step(full, dims, h, kc, vc, conv, ssm, args.ctx)
            m = compare(out, ref)
            ok = passes(m) and xr == 0.0
            row = dict(base, ref="fp32", cross_rank_max_abs=xr, **{k: m[k] for k in ("max_abs", "max_rel", "mean_rel", "cos")}, **{"pass": ok})
            csvio.append_rows(args.correctness_out, csvio.CORRECTNESS_COLUMNS, [row])
            print(f"[check] mode={args.mode} S={args.swing} cuts={cuts_src} B={args.batch} c={args.ctx} "
                  f"max_abs={m['max_abs']:.4g} mean_rel={m['mean_rel']:.3e} cos={m['cos']:.6f} cross_rank={xr:.3g} pass={ok}")
            del kc, vc, conv, ssm, ref
        # refill states so timing starts from the same point as every other run
        st2 = alloc_and_fill(plan, dims, ctx, args.seed, device)
        for name in ("kv", "key_cache", "value_cache", "conv_state", "ssm_state"):
            src, dst = getattr(st2, name), getattr(st, name)
            if src is not None and dst is not None and name in ("kv", "conv_state", "ssm_state"):
                dst.copy_(src)
        del st2
        barrier_sync()

    variants = ["eager", "graph"] if args.variant == "both" else [args.variant]
    rows: list[dict] = []
    for rep in range(args.repeats):
        for variant in variants:
            env = gpu_env(groups.local_rank)
            if variant == "eager":
                stats, wall = time_eager(layer, h, args.n_warm, args.n_iter, groups)
            else:
                stats, wall = time_graph(layer, h, args.n_warm, args.n_iter, groups)
            for op, s in stats.items():
                rows.append(dict(base, variant=variant, repeat=rep, rank=rank, op=op, **s, **env))
            rows.append(dict(base, variant=variant, repeat=rep, rank=rank, op="step_wall", n=args.n_iter,
                             median_ms=wall, p10_ms=wall, p90_ms=wall, mean_ms=wall, **env))
            barrier_sync()
    gathered: list[list[dict]] = [None] * world  # type: ignore
    if world > 1:
        dist.all_gather_object(gathered, rows)
    else:
        gathered = [rows]
    if rank == 0:
        all_rows = [r for rr in gathered for r in rr]
        # max over ranks of the per-rank step medians
        for variant in variants:
            for rep in range(args.repeats):
                steps = [r for r in all_rows if r["op"] == "step" and r["variant"] == variant and r["repeat"] == rep]
                walls = [r for r in all_rows if r["op"] == "step_wall" and r["variant"] == variant and r["repeat"] == rep]
                if steps:
                    m = max(steps, key=lambda r: r["median_ms"])
                    all_rows.append(dict(m, rank=-1, op="step_max"))
                if walls:
                    m = max(walls, key=lambda r: r["median_ms"])
                    all_rows.append(dict(m, rank=-1, op="step_wall_max"))
        csvio.append_rows(args.out, csvio.RUN_COLUMNS, all_rows)
        for variant in variants:
            sm = [r for r in all_rows if r["op"] == "step_max" and r["variant"] == variant]
            if sm:
                print(f"[time] mode={args.mode} S={args.swing} cuts={cuts_src or '-'} B={args.batch} c={args.ctx} "
                      f"variant={variant} step_max={min(r['median_ms'] for r in sm):.3f} ms "
                      f"(per-rank medians: {[round(r['median_ms'], 3) for r in all_rows if r['op'] == 'step' and r['variant'] == variant and r['repeat'] == 0]})")
    barrier_sync()
    destroy()


if __name__ == "__main__":
    main()
