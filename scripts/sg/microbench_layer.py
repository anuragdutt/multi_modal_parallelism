#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Per-layer cost model for Mamba-1 mixer layouts, measured on the serving node.

Purpose: before implementing the one-collective SSM/gate split (see
docs/sg_one_collective_design.md) produce a *measured* prediction of its
per-layer time against stock tensor parallelism, on the actual GPUs and
interconnect, for a decode step of B sequences and for a prefill chunk of T
tokens.

Everything is timed inside CUDA graphs, as vLLM runs decode:
  compute primitives (per rank): the GEMMs of in_proj / x_proj / dt_proj /
    out_proj at the sharded and the replicated widths, the conv-state update,
    selective_state_update at the channel shard of every layout (with and
    without the fused gate), silu, the elementwise product, and the prefill
    kernels (causal_conv1d_fn, selective_scan_fn) for a T-token chunk;
  collectives, through vLLM's own GroupCoordinator (pynccl under capture, so
    exactly the runtime path): the two 4-rank all-reduces of stock TP
    ([B, dt_rank + 2*d_state] and [B, d_model]), the uneven all-gather of the
    [y | silu(z)] channel slices for N+M role splits, the zero-padded
    all-reduce alternative, and the old SG 2+2 pair collectives (2-rank
    all-reduce, send/recv, broadcast).

The primitives are composed into per-layer times per layout (slowest role plus
the collectives) and into a request-mix throughput proxy for ISL 1024 /
OSL 128.  Shapes: Mamba-2.8B and Falcon-Mamba-7B.

Run inside the mmp-sg container on 4 GPUs:
  torchrun --nproc_per_node=4 /workspace/scripts/sg/microbench_layer.py \
      --out /workspace/results/sg/microbench_layer.json
"""
from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import statistics
import sys
import time

import torch
import torch.nn.functional as F

MODELS = {
    "mamba-2.8b": dict(d_model=2560, d_inner=5120, d_state=16, dt_rank=160, conv=4, n_layer=64),
    "falcon-mamba-7b": dict(d_model=4096, d_inner=8192, d_state=16, dt_rank=256, conv=4, n_layer=64),
}
SPLITS = [(3, 1), (2, 2), (1, 3)]  # (n_ssm, n_gate), TP = 4
DT = torch.bfloat16


def shard_sizes(n_total: int, n_parts: int) -> list[int]:
    base, rem = divmod(n_total, n_parts)
    return [base + (1 if i < rem else 0) for i in range(n_parts)]


class Bench:
    """Times fn() inside a CUDA graph: median over reps of (graph of `iters` calls)/iters, in us."""

    def __init__(self, device, groups, iters=16, reps=25):
        self.device, self.groups, self.iters, self.reps = device, groups, iters, reps

    def time(self, fn) -> float:
        from vllm.distributed.parallel_state import GraphCaptureContext

        for _ in range(3):
            fn()
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        ctx = GraphCaptureContext(torch.cuda.Stream(device=self.device))
        with contextlib.ExitStack() as es:
            for grp in self.groups:
                es.enter_context(grp.graph_capture(ctx))
            with torch.cuda.graph(g, stream=ctx.stream):
                for _ in range(self.iters):
                    fn()
        torch.cuda.synchronize()
        for _ in range(2):
            g.replay()
        torch.cuda.synchronize()
        ts = []
        for _ in range(self.reps):
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            s.record()
            g.replay()
            e.record()
            e.synchronize()
            ts.append(s.elapsed_time(e) * 1000.0 / self.iters)
        return statistics.median(ts)


# ----------------------------------------------------------------------------- primitives
def p_gemm(b: Bench, dev, M, in_f, out_f):
    x = torch.randn(M, in_f, device=dev, dtype=DT)
    W = torch.randn(out_f, in_f, device=dev, dtype=DT) * 0.02
    return b.time(lambda: F.linear(x, W))


def p_mul(b: Bench, dev, M, C):
    y = torch.randn(M, C, device=dev, dtype=DT)
    g = torch.randn(M, C, device=dev, dtype=DT)
    return b.time(lambda: y * g)


def p_silu(b: Bench, dev, M, C):
    z = torch.randn(M, C, device=dev, dtype=DT)
    return b.time(lambda: F.silu(z))


def p_conv_update(b: Bench, dev, B, C, k):
    from vllm.model_executor.layers.mamba.ops.causal_conv1d import causal_conv1d_update

    x = torch.randn(B, C, device=dev, dtype=DT)
    state = torch.zeros(B + 1, C, k - 1, device=dev, dtype=DT)
    w = torch.randn(C, k, device=dev, dtype=DT)
    bias = torch.randn(C, device=dev, dtype=DT)
    idx = torch.arange(1, B + 1, device=dev, dtype=torch.int32)
    return b.time(lambda: causal_conv1d_update(x, state, w, bias, activation="silu", conv_state_indices=idx))


def p_scan_update(b: Bench, dev, B, C, N, with_z):
    from vllm.model_executor.layers.mamba.ops.mamba_ssm import selective_state_update

    state = torch.zeros(B + 1, C, N, device=dev, dtype=torch.float32)
    x = torch.randn(B, C, device=dev, dtype=DT)
    dt = torch.randn(B, C, device=dev, dtype=DT)
    A = -torch.rand(C, N, device=dev, dtype=torch.float32)
    Bm = torch.randn(B, N, device=dev, dtype=DT)
    Cm = torch.randn(B, N, device=dev, dtype=DT)
    D = torch.randn(C, device=dev, dtype=torch.float32)
    dt_bias = torch.randn(C, device=dev, dtype=torch.float32)
    z = torch.randn(B, C, device=dev, dtype=DT) if with_z else None
    idx = torch.arange(1, B + 1, device=dev, dtype=torch.int32)
    out = torch.empty_like(x)
    return b.time(
        lambda: selective_state_update(
            state, x, dt, A, Bm, Cm, D, dt_bias, z=z, dt_softplus=True,
            state_batch_indices=idx, dst_state_batch_indices=idx, out=out,
        )
    )


def p_conv_prefill(b: Bench, dev, T, C, k):
    from vllm.model_executor.layers.mamba.ops.causal_conv1d import causal_conv1d_fn

    x = torch.randn(C, T, device=dev, dtype=DT)
    w = torch.randn(C, k, device=dev, dtype=DT)
    bias = torch.randn(C, device=dev, dtype=DT)
    states = torch.zeros(2, C, k - 1, device=dev, dtype=DT)
    qsl = torch.tensor([0, T], device=dev, dtype=torch.int32)
    cidx = torch.tensor([1], device=dev, dtype=torch.int32)
    has_init = torch.tensor([False], device=dev)
    return b.time(
        lambda: causal_conv1d_fn(
            x, w, bias, conv_states=states, query_start_loc=qsl, cache_indices=cidx,
            has_initial_state=has_init, activation="silu",
        )
    )


def p_scan_prefill(b: Bench, dev, T, C, N, with_z):
    from vllm.model_executor.layers.mamba.ops.mamba_ssm import selective_scan_fn

    u = torch.randn(C, T, device=dev, dtype=DT)
    delta = torch.randn(C, T, device=dev, dtype=DT)
    A = -torch.rand(C, N, device=dev, dtype=torch.float32)
    Bm = torch.randn(1, N, T, device=dev, dtype=DT)
    Cm = torch.randn(1, N, T, device=dev, dtype=DT)
    D = torch.randn(C, device=dev, dtype=torch.float32)
    dt_bias = torch.randn(C, device=dev, dtype=torch.float32)
    z = torch.randn(C, T, device=dev, dtype=DT) if with_z else None
    states = torch.zeros(2, C, N, device=dev, dtype=torch.float32)
    qsl = torch.tensor([0, T], device=dev, dtype=torch.int32)
    cidx = torch.tensor([1], device=dev, dtype=torch.int32)
    has_init = torch.tensor([False], device=dev)
    return b.time(
        lambda: selective_scan_fn(
            u, states, delta, A, Bm, Cm, D, z, dt_bias, delta_softplus=True,
            query_start_loc=qsl, cache_indices=cidx, has_initial_state=has_init,
        )
    )


def c_allreduce(b: Bench, dev, group, shape):
    x = torch.randn(*shape, device=dev, dtype=DT)
    return b.time(lambda: group.all_reduce(x))


def c_allgatherv(b: Bench, dev, group, sizes, M):
    """Channel-major slices [c_r, M] gathered along dim 0 -> [sum(sizes), M]."""
    x = torch.randn(sizes[group.rank_in_group], M, device=dev, dtype=DT)
    return b.time(lambda: group.all_gatherv(x, dim=0, sizes=sizes))


def c_sendrecv(b: Bench, dev, pair, shape, src, dst):
    if pair.rank_in_group == src:
        x = torch.randn(*shape, device=dev, dtype=DT)
        return b.time(lambda: pair.send(x, dst))
    return b.time(lambda: pair.recv(torch.Size(shape), DT, src))


def c_broadcast(b: Bench, dev, pair, shape, src):
    x = torch.randn(*shape, device=dev, dtype=DT)
    return b.time(lambda: pair.device_communicator.broadcast(x, src))


def safe(fn, *a, **k):
    try:
        return fn(*a, **k)
    except Exception as e:  # noqa: BLE001
        return {"error": f"{type(e).__name__}: {e}"[:300]}


# ----------------------------------------------------------------------------- composition
def val(x):
    return None if (x is None or isinstance(x, dict)) else float(x)


def add(*xs):
    """Sum of measured values; None if any is missing."""
    out = 0.0
    for x in xs:
        v = val(x)
        if v is None:
            return None
        out += v
    return out


def compose(m: dict, prim: dict, M: int, phase: str) -> dict:
    """Per-layer time (us) of each layout for M tokens of `phase` ('decode' or 'prefill')."""
    d_model, d_inner, N, R_dt = m["d_model"], m["d_inner"], m["d_state"], m["dt_rank"]
    R = R_dt + 2 * N
    G = prim["gemm"][phase]
    gemm = lambda i, o: G.get(f"{i}x{o}")
    if phase == "decode":
        conv = lambda c: prim["conv_update"].get(str(c))
        scan = lambda c, z: prim["scan_update"].get(f"{c}:{'z' if z else 'noz'}")
    else:
        conv = lambda c: prim["conv_prefill"].get(str(c))
        scan = lambda c, z: prim["scan_prefill"].get(f"{c}:{'z' if z else 'noz'}")
    mul = prim["mul"][phase].get(str(d_inner))
    silu = lambda c: prim["silu"][phase].get(str(c))
    C = prim["coll"][phase]
    out = {}

    c4 = d_inner // 4
    comp = add(gemm(d_model, 2 * c4), conv(c4), gemm(c4, R), gemm(R_dt, c4), scan(c4, True), gemm(c4, d_model))
    coll = add(C.get("ar4_small"), C.get("ar4_big"))
    out["tp4"] = dict(compute=comp, coll=coll, layer=add(comp, coll), roles=dict(all=comp))

    for n_ssm, n_gate in SPLITS:
        cs = shard_sizes(d_inner, n_ssm)[0]
        cg = shard_sizes(d_inner, n_gate)[0]
        ssm = add(gemm(d_model, d_inner), conv(d_inner), gemm(d_inner, R), gemm(R_dt, cs), scan(cs, False), mul, gemm(d_inner, d_model))
        gate = add(gemm(d_model, cg), silu(cg), mul, gemm(d_inner, d_model))
        worst = None if (ssm is None or gate is None) else max(ssm, gate)
        ag = C.get(f"agv_{n_ssm}+{n_gate}")
        out[f"split_{n_ssm}+{n_gate}"] = dict(compute=worst, coll=val(ag), layer=add(worst, ag), roles=dict(ssm=ssm, gate=gate))
        ar0 = C.get("ar4_zeropad")
        out[f"split_{n_ssm}+{n_gate}_zeropadAR"] = dict(compute=worst, coll=val(ar0), layer=add(worst, ar0), roles=dict(ssm=ssm, gate=gate))

    c2 = d_inner // 2
    ssm = add(gemm(d_model, c2), conv(c2), gemm(c2, R), C.get("ar2_small"), gemm(R_dt, c2), C.get("recv_c2"), scan(c2, True), gemm(c2, d_model), C.get("ar2_big"), C.get("bcast_big"))
    gate = add(gemm(d_model, c2), C.get("send_c2"), C.get("bcast_big"))
    worst = None if (ssm is None or gate is None) else max(ssm, gate)
    out["oldsg_2+2"] = dict(compute=None, coll=None, layer=worst, roles=dict(ssm=ssm, gate=gate))
    return out


def fmt(v, nd=1):
    return "-" if v is None else f"{v:.{nd}f}"


def tables(model_name: str, m: dict, batches: list[int], T: int, res: dict) -> str:
    L = m["n_layer"]
    layouts = list(res["decode"][str(batches[0])].keys())
    s = [f"\n### {model_name}: per-layer time, us (slowest role + collectives), measured in CUDA graphs\n"]
    s.append("| layout | " + " | ".join(f"decode B={b}" for b in batches) + f" | prefill T={T} |")
    s.append("|---|" + "---|" * (len(batches) + 1))
    for lay in layouts:
        row = [lay]
        for b in batches:
            row.append(fmt(res["decode"][str(b)][lay]["layer"]))
        row.append(fmt(res["prefill"][str(T)][lay]["layer"]))
        s.append("| " + " | ".join(row) + " |")
    s.append(f"\n### {model_name}: ratio to tp4 (<1 = faster than stock TP)\n")
    s.append("| layout | " + " | ".join(f"B={b}" for b in batches) + f" | prefill | request-mix tok/s ratio (ISL 1024, OSL 128) at B: " + ", ".join(str(b) for b in batches) + " |")
    s.append("|---|" + "---|" * (len(batches) + 2))
    for lay in layouts:
        row = [lay]
        mix = []
        for b in batches:
            a = res["decode"][str(b)][lay]["layer"]
            t = res["decode"][str(b)]["tp4"]["layer"]
            row.append(fmt(None if (a is None or t is None) else a / t, 2))
            # request mix: B prefills of T tokens (one chunk each) + 127 decode steps at B
            pa = res["prefill"][str(T)][lay]["layer"]
            pt = res["prefill"][str(T)]["tp4"]["layer"]
            if None in (a, t, pa, pt):
                mix.append("-")
            else:
                ta = L * (b * pa + 127 * a)
                tt = L * (b * pt + 127 * t)
                mix.append(f"{tt / ta:.2f}")
        pa = res["prefill"][str(T)][lay]["layer"]
        pt = res["prefill"][str(T)]["tp4"]["layer"]
        row.append(fmt(None if (pa is None or pt is None) else pa / pt, 2))
        row.append(", ".join(mix))
        s.append("| " + " | ".join(row) + " |")
    return "\n".join(s)


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--models", default=",".join(MODELS))
    ap.add_argument("--batches", default="1,8,32,128,256")
    ap.add_argument("--prefill-T", type=int, default=1024)
    ap.add_argument("--iters", type=int, default=16)
    ap.add_argument("--reps", type=int, default=25)
    args = ap.parse_args()
    batches = [int(x) for x in args.batches.split(",")]
    T = args.prefill_T

    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    local = int(os.environ["LOCAL_RANK"])
    assert world == 4, "this benchmark models TP=4 layouts"
    torch.cuda.set_device(local)
    dev = torch.device("cuda", local)

    from vllm.distributed import (
        get_tp_group, get_world_group, init_distributed_environment, initialize_model_parallel,
    )
    from vllm.distributed.parallel_state import (
        destroy_distributed_environment, destroy_model_parallel, init_model_parallel_group,
    )

    from vllm.config import VllmConfig, set_current_vllm_config

    init_distributed_environment(world_size=world, rank=rank, distributed_init_method="env://", local_rank=local)
    # initialize_model_parallel() and the group coordinators read the current vLLM config
    # (parallel sizes, custom-allreduce switch); a default config is what vLLM's own tests use.
    with set_current_vllm_config(VllmConfig()):
        initialize_model_parallel(tensor_model_parallel_size=world)
        _run(args, batches, T, rank, world, local, dev, t0=time.time())


def _run(args, batches, T, rank, world, local, dev, t0):
    from vllm.distributed import get_tp_group, get_world_group
    from vllm.distributed.parallel_state import (
        destroy_distributed_environment, destroy_model_parallel, init_model_parallel_group,
    )

    tp = get_tp_group()
    # old SG groups: gate ranks {0,1}, SSM ranks {2,3}; pair = (gate i, ssm i); branch = the two SSM ranks
    pair = init_model_parallel_group([[0, 2], [1, 3]], get_world_group().local_rank, "nccl", group_name="sg_pair")
    branch = init_model_parallel_group([[0, 1], [2, 3]], get_world_group().local_rank, "nccl", group_name="sg_branch")

    comp = Bench(dev, [], args.iters, args.reps)
    coll = Bench(dev, [tp, pair, branch], args.iters, args.reps)
    log = lambda *a: print(f"[rank{rank}]", *a, flush=True) if rank == 0 else None

    results = {
        "meta": dict(
            host=os.uname().nodename, gpu=torch.cuda.get_device_name(dev), torch=torch.__version__,
            world=world, iters=args.iters, reps=args.reps, batches=batches, prefill_T=T,
            custom_allreduce_tp=getattr(getattr(tp.device_communicator, "ca_comm", None), "disabled", None),
            custom_allreduce_pair=getattr(getattr(pair.device_communicator, "ca_comm", None), "disabled", None),
            date=time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        ),
        "models": {},
    }

    for name in args.models.split(","):
        m = MODELS[name]
        d_model, d_inner, N, R_dt, k = m["d_model"], m["d_inner"], m["d_state"], m["dt_rank"], m["conv"]
        R = R_dt + 2 * N
        c4, c2 = d_inner // 4, d_inner // 2
        widths = sorted({c4, c2, d_inner} | {shard_sizes(d_inner, n)[0] for n, _ in SPLITS} | {shard_sizes(d_inner, g)[0] for _, g in SPLITS})
        prim = {"gemm": {"decode": {}, "prefill": {}}, "mul": {"decode": {}, "prefill": {}}, "silu": {"decode": {}, "prefill": {}},
                "conv_update": {}, "scan_update": {}, "conv_prefill": {}, "scan_prefill": {}, "coll": {"decode": {}, "prefill": {}}}
        gemm_shapes = {(d_model, 2 * c4), (c4, R), (R_dt, c4), (c4, d_model),          # tp4
                       (d_model, d_inner), (d_inner, R), (d_inner, d_model),           # split replicated parts
                       (d_model, c2), (c2, R), (R_dt, c2), (c2, d_model)}             # old sg
        for n_ssm, n_gate in SPLITS:
            gemm_shapes.add((R_dt, shard_sizes(d_inner, n_ssm)[0]))
            gemm_shapes.add((d_model, shard_sizes(d_inner, n_gate)[0]))

        # --- decode primitives per batch (keyed by M inside); store per-M dicts
        prim_by_M = {}
        for M in batches + [T]:
            phase = "prefill" if M == T else "decode"
            key = str(M)
            pm = {"gemm": {}, "mul": {}, "silu": {}}
            for (i, o) in sorted(gemm_shapes):
                pm["gemm"][f"{i}x{o}"] = safe(p_gemm, comp, dev, M, i, o)
            pm["mul"][str(d_inner)] = safe(p_mul, comp, dev, M, d_inner)
            for w in widths:
                pm["silu"][str(w)] = safe(p_silu, comp, dev, M, w)
            if phase == "decode":
                pm["conv_update"] = {str(w): safe(p_conv_update, comp, dev, M, w, k) for w in widths}
                pm["scan_update"] = {}
                for w in widths:
                    for z in (True, False):
                        pm["scan_update"][f"{w}:{'z' if z else 'noz'}"] = safe(p_scan_update, comp, dev, M, w, N, z)
            else:
                pm["conv_prefill"] = {str(w): safe(p_conv_prefill, comp, dev, M, w, k) for w in widths}
                pm["scan_prefill"] = {}
                for w in widths:
                    for z in (True, False):
                        pm["scan_prefill"][f"{w}:{'z' if z else 'noz'}"] = safe(p_scan_prefill, comp, dev, M, w, N, z)
            # --- collectives (all ranks in lockstep)
            cc = {}
            cc["ar4_small"] = safe(c_allreduce, coll, dev, tp, (M, R))
            cc["ar4_big"] = safe(c_allreduce, coll, dev, tp, (M, d_model))
            cc["ar4_zeropad"] = safe(c_allreduce, coll, dev, tp, (2 * d_inner, M))
            for n_ssm, n_gate in SPLITS:
                sizes = shard_sizes(d_inner, n_ssm) + shard_sizes(d_inner, n_gate)
                cc[f"agv_{n_ssm}+{n_gate}"] = safe(c_allgatherv, coll, dev, tp, sizes, M)
            cc["ar2_small"] = safe(c_allreduce, coll, dev, branch, (M, R))
            cc["ar2_big"] = safe(c_allreduce, coll, dev, branch, (M, d_model))
            sr = safe(c_sendrecv, coll, dev, pair, (c2, M), 0, 1)
            cc["send_c2" if pair.rank_in_group == 0 else "recv_c2"] = sr
            cc["bcast_big"] = safe(c_broadcast, coll, dev, pair, (M, d_model), 1)
            pm["coll"] = cc
            prim_by_M[key] = pm
            log(f"{name}: M={M} ({phase}) primitives done, {time.time() - t0:.0f}s elapsed")

        # gather collective timings across ranks (send/recv live on different ranks); take max per key
        gathered = [None] * world
        torch.distributed.all_gather_object(gathered, {k: v["coll"] for k, v in prim_by_M.items()}, group=get_world_group().cpu_group)
        for key in prim_by_M:
            merged = {}
            for r in range(world):
                for ck, cv in gathered[r][key].items():
                    v = val(cv)
                    if v is None:
                        merged.setdefault(ck, cv)
                        continue
                    merged[ck] = v if val(merged.get(ck)) is None else max(merged[ck], v)
            prim_by_M[key]["coll"] = merged

        # compose
        res = {"decode": {}, "prefill": {}}
        for M in batches:
            pm = prim_by_M[str(M)]
            prim_view = {"gemm": {"decode": pm["gemm"]}, "mul": {"decode": pm["mul"]}, "silu": {"decode": pm["silu"]},
                         "conv_update": pm["conv_update"], "scan_update": pm["scan_update"], "coll": {"decode": pm["coll"]}}
            res["decode"][str(M)] = compose(m, prim_view, M, "decode")
        pm = prim_by_M[str(T)]
        prim_view = {"gemm": {"prefill": pm["gemm"]}, "mul": {"prefill": pm["mul"]}, "silu": {"prefill": pm["silu"]},
                     "conv_prefill": pm["conv_prefill"], "scan_prefill": pm["scan_prefill"], "coll": {"prefill": pm["coll"]}}
        res["prefill"][str(T)] = compose(m, prim_view, T, "prefill")
        results["models"][name] = dict(shape=m, primitives=prim_by_M, composed=res)
        if rank == 0:
            print(tables(name, m, batches, T, res), flush=True)

    if rank == 0:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(results, f, indent=1)
        print(f"\nwrote {args.out} ({time.time() - t0:.0f}s)", flush=True)
    torch.distributed.barrier()
    destroy_model_parallel()
    destroy_distributed_environment()


if __name__ == "__main__":
    main()
