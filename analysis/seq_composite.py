"""Compose per-layer-type measurements of a sequential hybrid into a full-step comparison.

  python -m analysis.seq_composite results/nemotron3_nano/raw/sweep_full.csv --counts mamba=23,moe=23,attention=6
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--counts", default="mamba=23,moe=23,attention=6")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    counts = {k: int(v) for k, v in (kv.split("=") for kv in args.counts.split(","))}
    df = pd.read_csv(args.csv)
    g = df[(df.variant == "graph") & (df.op == "step_max")].copy()
    g[["kind", "layout"]] = g["mode"].str.split("/", expand=True)
    t = g.groupby(["kind", "layout", "batch", "ctx"])["median_ms"].min().reset_index()
    cells = sorted(set(zip(t.batch, t.ctx)))
    lines = ["# Sequential-hybrid composite (graph variant, min over repeats)", "",
             "| batch | ctx | " + " | ".join(f"{k}/{l} ms" for k, l in [("mamba", "tp4"), ("attention", "tp4"), ("attention", "dp4"), ("moe", "tp4"), ("moe", "ep4")]) +
             " | attn dp/tp | moe ep/tp | all-tp step | recipe step | speedup |", "|" + "---|" * 12]
    for B, c in cells:
        def val(kind, layout):
            s = t[(t.kind == kind) & (t.layout == layout) & (t.batch == B) & (t.ctx == c)]["median_ms"]
            return float(s.iloc[0]) if len(s) else float("nan")
        m, a_tp, a_dp, e_tp, e_ep = val("mamba", "tp4"), val("attention", "tp4"), val("attention", "dp4"), val("moe", "tp4"), val("moe", "ep4")
        all_tp = counts["mamba"] * m + counts["attention"] * a_tp + counts["moe"] * e_tp
        recipe = counts["mamba"] * m + counts["attention"] * min(a_tp, a_dp) + counts["moe"] * min(e_tp, e_ep)
        lines.append(f"| {B} | {c} | {m:.3f} | {a_tp:.3f} | {a_dp:.3f} | {e_tp:.3f} | {e_ep:.3f} | {a_tp / a_dp:.2f}x | {e_tp / e_ep:.2f}x | {all_tp:.2f} | {recipe:.2f} | {all_tp / recipe:.2f}x |")
    text = "\n".join(lines) + "\n"
    print(text)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text)


if __name__ == "__main__":
    main()
