"""torchrun entry point: one (mode, swing, cuts, batch, ctx, variant) measurement, optional correctness check.

  torchrun --standalone --nproc_per_node=4 -m harness.run --mode split22 --swing 0.2 --k neutral \
      --batch 16 --ctx 8192 --variant eager --check --out results/raw/dev.csv
"""
from __future__ import annotations

import argparse
import os

from .config import FalconH1Dims, RunSpec
from .dist import barrier_sync, destroy, init_groups
from .measure import measure
from .model import BwTables
from .weights import load_layer_from_safetensors, random_layer


def load_dims_and_weights(args) -> tuple[FalconH1Dims, dict]:
    if args.model_dir:
        dims = FalconH1Dims.from_hf(args.model_dir)
    else:
        dims = FalconH1Dims.from_yaml(args.dims_yaml)
    if args.random_weights or not args.model_dir:
        full = random_layer(dims, args.seed)
    else:
        full = load_layer_from_safetensors(args.model_dir, args.layer_idx)
    return dims, full


def add_common_args(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--model-dir", default=os.environ.get("MMP_MODEL_DIR", ""))
    ap.add_argument("--dims-yaml", default="configs/falcon_h1_7b.yaml")
    ap.add_argument("--random-weights", action="store_true")
    ap.add_argument("--layer-idx", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--bw", default="results/raw/microbench.csv", help="microbench CSV for the byte model (optional)")
    ap.add_argument("--out", default="results/raw/dev.csv")
    ap.add_argument("--correctness-out", default="results/raw/correctness.csv")
    ap.add_argument("--tag", default="")


def main() -> None:
    ap = argparse.ArgumentParser()
    add_common_args(ap)
    ap.add_argument("--mode", required=True)
    ap.add_argument("--swing", type=float, default=0.0)
    ap.add_argument("--k", default="neutral", help="swing split point: neutral | model | <int in [0, 2w]>")
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--ctx", type=int, default=512)
    ap.add_argument("--variant", choices=["eager", "graph", "both"], default="eager")
    ap.add_argument("--kv-block", type=int, default=16)
    ap.add_argument("--n-warm", type=int, default=20)
    ap.add_argument("--n-iter", type=int, default=100)
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()

    if args.k in ("neutral", "model"):
        k, src = None, args.k
    else:
        k, src = int(args.k), "manual"
    spec = RunSpec(mode=args.mode, swing=args.swing, k=k, cuts_src=src, batch=args.batch, ctx=args.ctx,
                   variant=args.variant, kv_block=args.kv_block, n_warm=args.n_warm, n_iter=args.n_iter,
                   repeats=args.repeats, layer_idx=args.layer_idx, seed=args.seed, tag=args.tag)
    groups = init_groups()
    dims, full = load_dims_and_weights(args)
    bw = BwTables.from_csv(args.bw) if os.path.exists(args.bw) else None
    measure(spec, groups, dims, full, args.out, args.correctness_out, args.check, bw, os.environ.get("MMP_IMAGE_TAG", ""))
    barrier_sync()
    destroy()


if __name__ == "__main__":
    main()
