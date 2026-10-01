"""Achieved-bandwidth curves per kernel class and collective latency tables (4 ranks, torchrun).

  torchrun --standalone --nproc_per_node=4 -m harness.microbench --out results/raw/microbench.csv
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import time

import torch
import torch.distributed as dist

from . import csvio
from . import kernels as K
from .dist import barrier_sync, destroy, init_groups
from .state import make_decode_ctx
from .timing import _stats

COLUMNS = ["kernel", "params_json", "bytes", "median_ms", "p10_ms", "p90_ms", "GBps", "rank", "variant"]


def _time(fn, n_warm: int = 10, n_iter: int = 50, graph: bool = False) -> dict[str, float]:
    if graph:
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(2):
                fn()
        torch.cuda.current_stream().wait_stream(s)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            fn()
        run = g.replay
    else:
        run = fn
    for _ in range(n_warm):
        run()
    torch.cuda.synchronize()
    ms = []
    for _ in range(n_iter):
        a = torch.cuda.Event(enable_timing=True)
        b = torch.cuda.Event(enable_timing=True)
        a.record()
        run()
        b.record()
        b.synchronize()
        ms.append(a.elapsed_time(b))
    return _stats(ms)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/raw/microbench.csv")
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    g = init_groups()
    dev, rank = g.device, g.rank
    bf = torch.bfloat16
    rows: list[dict] = []

    def rec(kernel: str, params: dict, nbytes: int, st: dict, variant: str) -> None:
        rows.append({"kernel": kernel, "params_json": json.dumps(params), "bytes": nbytes, "median_ms": st["median_ms"],
                     "p10_ms": st["p10_ms"], "p90_ms": st["p90_ms"], "GBps": nbytes / (st["median_ms"] * 1e-3) / 1e9,
                     "rank": rank, "variant": variant})

    # (a) streaming read bandwidth
    big = torch.ones(512 * 1024 * 1024, device=dev, dtype=bf)  # 1 GiB
    st = _time(lambda: big.sum())
    rec("stream", {"bytes": big.numel() * 2}, big.numel() * 2, st, "eager")
    del big

    # (b) weight-streaming GEMV: [T, 3072] x W[rows, 3072]^T
    Ts = [1, 4, 16, 64] if not args.quick else [1, 16]
    for rows_ in [384, 768, 1280, 1536, 2054, 3072, 3596, 6144, 6680, 12288]:
        W = torch.randn(rows_, 3072, device=dev, dtype=bf)
        for T in Ts:
            x = torch.randn(T, 3072, device=dev, dtype=bf)
            st = _time(lambda: K.gemm(x, W))
            rec("gemv", {"rows": rows_, "T": T}, W.numel() * 2 + x.numel() * 2, st, "eager")
        del W

    # (c) paged FA2 decode
    for (nq, nkv) in [(3, 1), (6, 1), (12, 2)]:
        for kvb in ([16, 256] if not args.quick else [16]):
            for B in ([1, 4, 16, 64] if not args.quick else [1, 16]):
                for c in ([512, 2048, 8192, 32768] if not args.quick else [512, 8192]):
                    ctx = make_decode_ctx(B, c, kvb, dev)
                    nb = B * ctx.nblk + 1
                    kv, kc, vc, kind = K.alloc_kv(nb, kvb, nkv, 128, dev)
                    kv.normal_()
                    q = torch.randn(B, nq, 128, device=dev, dtype=bf)
                    out = torch.empty_like(q)
                    fn = lambda: K.attn_decode(q, kc, vc, ctx.cu_q, ctx.seqused_k, ctx.max_k, ctx.block_table, out)  # noqa: E731
                    st = _time(fn)
                    nbytes = 2 * B * (c + 1) * nkv * 128 * 2
                    rec("attn", {"nq": nq, "nkv": nkv, "kv_block": kvb, "B": B, "c": c, "layout": kind}, nbytes, st, "eager")
                    del kv, kc, vc, q, out
                    torch.cuda.empty_cache()

    # (d) selective_state_update and (e) conv update
    for heads in [6, 12, 24]:
        for B in ([1, 4, 16, 64, 256] if not args.quick else [1, 16, 64]):
            state = torch.randn(B + 1, heads, 128, 256, device=dev, dtype=bf)
            x = torch.randn(B, heads, 128, device=dev, dtype=bf)
            dt = torch.rand(B, heads, 128, device=dev, dtype=bf)
            A = -torch.rand(heads, 128, 256, device=dev)
            Bm = torch.randn(B, 1, 256, device=dev, dtype=bf)
            Cm = torch.randn(B, 1, 256, device=dev, dtype=bf)
            D = torch.ones(heads, 128, device=dev)
            dtb = torch.zeros(heads, 128, device=dev)
            idx = torch.arange(1, B + 1, device=dev, dtype=torch.int32)
            y = torch.empty_like(x)
            fn = lambda: K.ssm_update(state, x, dt, A, Bm, Cm, D, dtb, idx, y)  # noqa: E731
            nbytes = 2 * B * heads * 128 * 256 * 2
            for variant in ("eager", "graph"):
                st = _time(fn, graph=(variant == "graph"))
                rec("ssu", {"heads": heads, "B": B}, nbytes, st, variant)
            conv_rows = heads * 128 + 512
            cs = torch.randn(B + 1, 3, conv_rows, device=dev, dtype=bf)
            cw = torch.randn(conv_rows, 4, device=dev, dtype=bf)
            cb = torch.zeros(conv_rows, device=dev, dtype=bf)
            xc = torch.randn(B, conv_rows, device=dev, dtype=bf)
            csT = cs.transpose(-1, -2)
            fn2 = lambda: K.conv_update(xc, csT, cw, cb, idx)  # noqa: E731
            nbytes = 2 * B * 3 * conv_rows * 2
            st = _time(fn2)
            rec("conv", {"rows": conv_rows, "B": B}, nbytes, st, "eager")
            del state, x, dt, A, Bm, Cm, cs, xc
            torch.cuda.empty_cache()

    # (f) all-reduce on the TP group and the pair group
    for group_name, grp in (("tp", g.tp), ("pair", g.pair)):
        if grp is None:
            continue
        for T in [1, 4, 16, 64]:
            for shape, dtype in (((T, 3072), bf), ((T, 1), torch.float32)):
                x = torch.randn(*shape, device=dev, dtype=dtype)
                fn = lambda: grp.all_reduce(x)  # noqa: E731
                nbytes = x.numel() * x.element_size()
                barrier_sync()
                st = _time(fn, n_warm=20, n_iter=100)
                rec("allreduce", {"group": group_name, "group_size": grp.world_size, "T": T, "dtype": str(dtype)}, nbytes, st, "eager")
                # graph variant
                try:
                    from .dist import capture_context

                    with capture_context(g) as ctx:
                        stream = ctx.stream
                        with torch.cuda.stream(stream):
                            for _ in range(2):
                                grp.all_reduce(x)
                        stream.synchronize()
                        cg = torch.cuda.CUDAGraph()
                        with torch.cuda.graph(cg, stream=stream):
                            grp.all_reduce(x)
                    torch.cuda.synchronize()
                    barrier_sync()
                    st = _time(cg.replay, n_warm=20, n_iter=100)
                    rec("allreduce", {"group": group_name, "group_size": grp.world_size, "T": T, "dtype": str(dtype)}, nbytes, st, "graph")
                except Exception as e:  # pragma: no cover
                    if rank == 0:
                        print(f"[microbench] graph capture of all_reduce failed for {group_name} {shape}: {e}")
                barrier_sync()

    gathered = [None] * g.world_size
    dist.all_gather_object(gathered, rows)
    if rank == 0:
        allrows = [r for rr in gathered for r in rr]
        csvio.append_rows(args.out, COLUMNS, allrows)
        print(f"[microbench] wrote {len(allrows)} rows to {args.out}")
    barrier_sync()
    destroy()


if __name__ == "__main__":
    main()
