"""Measurement core shared by run.py (one configuration) and sweep.py (many)."""
from __future__ import annotations

import json
import os
import subprocess
import time
import uuid
from pathlib import Path

import torch
import torch.distributed as dist

from . import csvio
from .config import FalconH1Dims, RunSpec
from .correctness import compare, cross_rank_max_abs, passes
from .dist import Groups, barrier_sync, describe_comm
from .layer import ParallelHybridLayer
from .model import BwTables, choose_k, predict_rank
from .sharding import neutral_k, plan_for, swing_width
from .state import alloc_and_fill, full_states_for_reference, input_hidden, make_decode_ctx
from .timing import OpTimer, time_eager, time_graph
from .weights import shard_to_device


def gpu_env(local_rank: int) -> dict:
    try:
        q = subprocess.run(
            ["nvidia-smi", "--query-gpu=clocks.sm,temperature.gpu,power.draw,utilization.gpu", "--format=csv,noheader,nounits", "-i", str(local_rank)],
            capture_output=True, text=True, timeout=5,
        ).stdout.strip().split(",")
        return {"sm_clock_mhz": int(q[0]), "temp_c": int(q[1]), "power_w": float(q[2]), "util_pct": int(q[3])}
    except Exception:
        return {"sm_clock_mhz": -1, "temp_c": -1, "power_w": -1.0, "util_pct": -1}


def vllm_commit() -> str:
    try:
        import vllm

        root = Path(vllm.__file__).resolve().parents[1]
        r = subprocess.run(["git", "-C", str(root), "rev-parse", "--short", "HEAD"], capture_output=True, text=True)
        return r.stdout.strip() or vllm.__version__
    except Exception:
        return "unknown"


def resolve_k(spec: RunSpec, dims: FalconH1Dims, world: int, bw: BwTables | None) -> tuple[int, int | None, str]:
    """Return (w, k, cuts_src) for a spec; 'model' needs a BwTables, otherwise falls back to neutral."""
    if not spec.mode.startswith("split") or spec.swing <= 0:
        return 0, None, ""
    w = swing_width(spec.swing, dims.intermediate // world)
    src = spec.cuts_src
    if src == "neutral":
        return w, neutral_k(w), "neutral"
    if src == "model":
        if bw is None:
            return w, neutral_k(w), "neutral(no-bw)"
        k, _ = choose_k(dims, spec.batch, spec.ctx, spec.swing, bw, "graph" if spec.variant != "eager" else "eager", spec.kv_block)
        return w, k, "model"
    assert spec.k is not None, "grid/manual needs k"
    return w, int(spec.k), src


def measure(spec: RunSpec, groups: Groups, dims: FalconH1Dims, full: dict[str, torch.Tensor], out_csv: str,
            correctness_csv: str, check: bool, bw: BwTables | None = None, image_tag: str = "") -> list[dict]:
    rank, world, device = groups.rank, groups.world_size, groups.device
    w, k, cuts_src = resolve_k(spec, dims, world, bw)
    plan = plan_for(spec.mode, rank, world, dims, spec.swing, k)
    rw = shard_to_device(full, plan, dims, device)
    ctx = make_decode_ctx(spec.batch, spec.ctx, spec.kv_block, device)
    st = alloc_and_fill(plan, dims, ctx, spec.seed, device)
    h = input_hidden(spec.seed, spec.batch, dims.hidden, device)
    layer = ParallelHybridLayer(plan, rw, dims, groups, st, ctx, device, collectives=spec.collectives)
    barrier_sync()

    run_id = uuid.uuid4().hex[:8]
    ts = time.strftime("%Y-%m-%dT%H:%M:%S")
    base = dict(run_id=run_id, ts=ts, image_tag=image_tag, vllm_commit=vllm_commit(), mode=spec.mode, S=spec.swing,
                w=w, cuts=(k if k is not None else ""), cuts_src=cuts_src, batch=spec.batch, ctx=spec.ctx,
                kv_block=spec.kv_block, comm_tp=json.dumps(describe_comm(groups.tp)) if spec.collectives else "none",
                comm_pair=json.dumps(describe_comm(groups.pair)), tag=spec.tag)

    if check and spec.collectives:
        out = layer.step(h, OpTimer(enabled=False)).clone()
        torch.cuda.synchronize()
        xr = cross_rank_max_abs(out, world)
        if rank == 0:
            from .reference import decode_step

            kc, vc, conv, ssm = full_states_for_reference(dims, spec.batch, spec.ctx, spec.seed, device)
            ref = decode_step(full, dims, h, kc, vc, conv, ssm, spec.ctx)
            m = compare(out, ref)
            ok = passes(m) and xr == 0.0
            row = dict(base, ref="fp32", cross_rank_max_abs=xr, **{k: m[k] for k in ("max_abs", "max_rel", "mean_rel", "cos")})
            row["pass"] = ok
            csvio.append_rows(correctness_csv, csvio.CORRECTNESS_COLUMNS, [row])
            print(f"[check] mode={spec.mode} S={spec.swing} cuts={cuts_src or '-'} B={spec.batch} c={spec.ctx} "
                  f"max_abs={m['max_abs']:.4g} (ref max {m['ref_max_abs']:.3g}) mean_rel={m['mean_rel']:.3e} "
                  f"cos={m['cos']:.6f} cross_rank={xr:.3g} pass={ok}", flush=True)
            del kc, vc, conv, ssm, ref
        # restore the initial states so timing starts from the same point as every other run
        st2 = alloc_and_fill(plan, dims, ctx, spec.seed, device)
        for name in ("kv", "conv_state", "ssm_state"):
            src, dst = getattr(st2, name), getattr(st, name)
            if src is not None and dst is not None:
                dst.copy_(src)
        del st2
        barrier_sync()

    pred = predict_rank(plan, dims, spec.batch, spec.ctx, bw, "eager", spec.kv_block) if bw is not None else {}
    pred_g = predict_rank(plan, dims, spec.batch, spec.ctx, bw, "graph", spec.kv_block) if bw is not None else {}
    variants = ["eager", "graph"] if spec.variant == "both" else [spec.variant]
    rows: list[dict] = []
    for rep in range(spec.repeats):
        for variant in variants:
            env = gpu_env(groups.local_rank)
            if variant == "eager":
                stats, wall = time_eager(layer, h, spec.n_warm, spec.n_iter, groups)
                p = pred
            else:
                stats, wall = time_graph(layer, h, spec.n_warm, spec.n_iter, groups)
                p = pred_g
            rep_id = rep + spec.repeat_offset
            for op, s in stats.items():
                rows.append(dict(base, variant=variant, repeat=rep_id, rank=rank, op=op, **s, **env,
                                 ms_pred=p.get(op, ""), bytes_pred=""))
            rows.append(dict(base, variant=variant, repeat=rep_id, rank=rank, op="step_wall", n=spec.n_iter,
                             median_ms=wall, p10_ms=wall, p90_ms=wall, mean_ms=wall, **env))
            barrier_sync()
    gathered: list = [None] * world
    if world > 1:
        dist.all_gather_object(gathered, rows)
    else:
        gathered = [rows]
    all_rows: list[dict] = []
    if rank == 0:
        all_rows = [r for rr in gathered for r in rr]
        for variant in variants:
            for rep in range(spec.repeat_offset, spec.repeat_offset + spec.repeats):
                steps = [r for r in all_rows if r["op"] == "step" and r["variant"] == variant and r["repeat"] == rep]
                walls = [r for r in all_rows if r["op"] == "step_wall" and r["variant"] == variant and r["repeat"] == rep]
                if steps:
                    all_rows.append(dict(max(steps, key=lambda r: r["median_ms"]), rank=-1, op="step_max"))
                if walls:
                    all_rows.append(dict(max(walls, key=lambda r: r["median_ms"]), rank=-1, op="step_wall_max"))
        csvio.append_rows(out_csv, csvio.RUN_COLUMNS, all_rows)
        for variant in variants:
            sm = [r for r in all_rows if r["op"] == "step_max" and r["variant"] == variant]
            per = [round(r["median_ms"], 3) for r in all_rows if r["op"] == "step" and r["variant"] == variant and r["repeat"] == spec.repeat_offset]
            if sm:
                print(f"[time] mode={spec.mode} S={spec.swing} cuts={cuts_src or '-'} B={spec.batch} c={spec.ctx} "
                      f"variant={variant} step_max={min(r['median_ms'] for r in sm):.3f} ms per-rank={per}", flush=True)
    # free GPU memory before the next configuration
    del layer, st, rw, h, ctx
    torch.cuda.empty_cache()
    barrier_sync()
    return all_rows
