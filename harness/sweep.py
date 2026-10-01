"""In-process sweep driver (torchrun, 4 ranks): loads weights once and iterates a YAML grid, interleaving
repeats across modes to defeat clock drift.

  torchrun --standalone --nproc_per_node=4 -m harness.sweep --config configs/sweep_stage1.yaml \
      --model-dir <snapshot> --out results/raw/sweep_<date>.csv
"""
from __future__ import annotations

import argparse
import os
import time

import yaml

from .config import RunSpec
from .dist import barrier_sync, destroy, init_groups
from .measure import measure
from .model import BwTables
from .run import add_common_args, load_dims_and_weights
from .sharding import swing_width


def expand(cfg: dict) -> list[RunSpec]:
    specs: list[RunSpec] = []
    cells = [(b, c) for b in cfg["batches"] for c in cfg["contexts"]]
    variant = cfg.get("variant", "both")
    kv_block = cfg.get("kv_block", 16)
    n_warm, n_iter = cfg.get("n_warm", 20), cfg.get("n_iter", 100)
    base = dict(variant=variant, kv_block=kv_block, n_warm=n_warm, n_iter=n_iter, repeats=1)
    for mode in cfg["modes"]:
        for b, c in cells:
            specs.append(RunSpec(mode=mode, batch=b, ctx=c, **base))
    for S in cfg.get("swing", []):
        w = swing_width(S, cfg.get("intermediate", 12288) // 4)
        for b, c in cells:
            for src in cfg.get("cut_sources", ["neutral", "model"]):
                specs.append(RunSpec(mode="split22", swing=S, cuts_src=src, batch=b, ctx=c, **base))
            step = cfg.get("grid_step", 128)
            for n_a in range(0, 2 * w + 1, step):
                specs.append(RunSpec(mode="split22", swing=S, cuts=(n_a,), cuts_src="grid", batch=b, ctx=c, **base))
    return specs


def main() -> None:
    ap = argparse.ArgumentParser()
    add_common_args(ap)
    ap.add_argument("--config", default="configs/sweep_stage1.yaml")
    ap.add_argument("--repeats", type=int, default=None)
    ap.add_argument("--check", action="store_true", help="also run the correctness check for every configuration")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    specs = expand(cfg)
    repeats = args.repeats or cfg.get("repeats", 3)
    groups = init_groups()
    dims, full = load_dims_and_weights(args)
    bw = BwTables.from_csv(args.bw) if os.path.exists(args.bw) else None
    if groups.rank == 0:
        print(f"[sweep] {len(specs)} configurations x {repeats} repeats -> {args.out}", flush=True)
    t0 = time.time()
    done = 0
    for rep in range(repeats):
        for spec in specs:
            spec_r = RunSpec(**{**spec.__dict__, "tag": f"rep{rep}", "repeat_offset": rep})
            try:
                measure(spec_r, groups, dims, full, args.out, args.correctness_out, args.check and rep == 0, bw,
                        os.environ.get("MMP_IMAGE_TAG", ""))
            except Exception as e:  # keep the sweep going; record the failure
                if groups.rank == 0:
                    print(f"[sweep] FAILED {spec_r}: {type(e).__name__}: {e}", flush=True)
                barrier_sync()
            done += 1
            if groups.rank == 0 and done % 10 == 0:
                el = time.time() - t0
                print(f"[sweep] {done}/{len(specs) * repeats} done, {el / 60:.1f} min elapsed", flush=True)
    barrier_sync()
    destroy()


if __name__ == "__main__":
    main()
