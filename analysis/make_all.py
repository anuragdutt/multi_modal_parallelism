"""Produce every stage-1 figure and the summary from results/raw.

  python -m analysis.make_all --raw results/raw --out results
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from .load import best_swing, imbalance, load_runs, step_table  # noqa: E402

BASE_MODES = ["tp4", "tp4_fused", "tp4_streams", "split22"]


def plot_layer_time(step_g: pd.DataFrame, step_e: pd.DataFrame, figs: Path) -> None:
    for B in sorted(step_g["batch"].unique()):
        fig, ax = plt.subplots(figsize=(6.5, 4.2))
        for label in BASE_MODES:
            for st, ls, src in ((step_g, "-", "graph"), (step_e, "--", "eager")):
                s = st[(st["label"] == label) & (st["batch"] == B)].sort_values("ctx")
                if len(s):
                    ax.plot(s["ctx"], s["step_ms"], ls, marker="o", label=f"{label} ({src})" if src == "graph" else None,
                            color=f"C{BASE_MODES.index(label)}")
        for src, color, name in (("model", "C4", "swing best S<=0.4, model cuts"), ("grid", "C5", "swing best S<=0.4, empirical cuts")):
            bs = best_swing(step_g[step_g["batch"] == B], src)
            if len(bs):
                bs = bs.sort_values("ctx")
                ax.plot(bs["ctx"], bs["step_ms"], "-", marker="s", color=color, label=name)
        ax.set_xscale("log", base=2)
        ax.set_xlabel("context length (tokens)")
        ax.set_ylabel("layer decode step (ms), max over ranks")
        ax.set_title(f"Falcon-H1-7B layer, 4x A6000, batch {B} (solid: CUDA graph, dashed: eager)")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(figs / f"layer_time_vs_ctx_B{B}.png", dpi=150)
        plt.close(fig)


def plot_imbalance(df: pd.DataFrame, figs: Path) -> None:
    imb = imbalance(df, "eager")
    rows = []
    for (B, c), g in imb.groupby(["batch", "ctx"]):
        base = g[(g["label"] == "split22")]["imbalance"]
        sw_m = g[(g["S"] > 0) & (g["cuts_src"].astype(str).str.startswith("model"))]["imbalance"]
        sw_g = g[(g["S"] > 0) & (g["cuts_src"].astype(str).str.startswith("grid"))]["imbalance"]
        rows.append({"batch": B, "ctx": c, "split22": base.min() if len(base) else float("nan"),
                     "swing_model": sw_m.min() if len(sw_m) else float("nan"), "swing_grid": sw_g.min() if len(sw_g) else float("nan")})
    t = pd.DataFrame(rows)
    if not len(t):
        return
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), sharey=True)
    for ax, col in zip(axes, ["split22", "swing_model", "swing_grid"]):
        piv = t.pivot(index="batch", columns="ctx", values=col)
        im = ax.imshow(piv.values, cmap="viridis", vmin=1.0, vmax=max(1.5, float(t[["split22", "swing_model", "swing_grid"]].max().max())))
        ax.set_xticks(range(len(piv.columns)), piv.columns)
        ax.set_yticks(range(len(piv.index)), piv.index)
        ax.set_xlabel("context")
        ax.set_title(col)
        for i in range(piv.shape[0]):
            for j in range(piv.shape[1]):
                ax.text(j, i, f"{piv.values[i, j]:.2f}", ha="center", va="center", color="w", fontsize=8)
    axes[0].set_ylabel("batch")
    fig.colorbar(im, ax=axes, label="max/mean per-rank compute")
    fig.savefig(figs / "imbalance.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_collectives(df: pd.DataFrame, figs: Path) -> None:
    e = df[(df["variant"] == "eager") & (df["rank"] >= 0) & (df["op"] != "step") & (df["op"] != "step_wall")]
    cells = [(1, 512), (16, 512), (16, 8192), (16, 32768), (64, 8192)]
    fig, axes = plt.subplots(1, len(cells), figsize=(3.2 * len(cells), 3.8), sharey=False)
    for ax, (B, c) in zip(axes, cells):
        s = e[(e["batch"] == B) & (e["ctx"] == c) & (e["label"].isin(BASE_MODES))]
        if not len(s):
            ax.set_visible(False)
            continue
        s = s.assign(kind=s["op"].str.startswith("ar.").map({True: "collectives", False: "compute"}))
        agg = s.groupby(["label", "repeat", "rank", "kind"])["median_ms"].sum().reset_index()
        agg = agg.groupby(["label", "kind"])["median_ms"].median().unstack("kind").reindex(BASE_MODES)
        agg.plot.bar(stacked=True, ax=ax, legend=(ax is axes[0]))
        ax.set_title(f"B={B}, c={c}")
        ax.set_ylabel("ms per layer (median rank)")
        ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(figs / "collective_share.png", dpi=150)
    plt.close(fig)


def plot_pred_vs_meas(df: pd.DataFrame, figs: Path) -> None:
    s = df[(df["rank"] >= 0) & (df["ms_pred"].notna()) & (df["ms_pred"] != "")].copy()
    if not len(s):
        return
    s["ms_pred"] = s["ms_pred"].astype(float)
    s = s[s["ms_pred"] > 0]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    for ax, variant in zip(axes, ["eager", "graph"]):
        v = s[s["variant"] == variant]
        ops = v[v["op"] != "step"]
        st = v[v["op"] == "step"]
        ax.scatter(ops["ms_pred"], ops["median_ms"], s=6, alpha=0.4, label="per op")
        ax.scatter(st["ms_pred"], st["median_ms"], s=14, alpha=0.7, color="C3", label="per step")
        lim = [1e-3, max(1.0, float(v["median_ms"].max()) * 1.2)]
        ax.plot(lim, lim, "k--", lw=0.8)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("predicted ms")
        ax.set_ylabel("measured ms")
        err = ((v["median_ms"] - v["ms_pred"]).abs() / v["ms_pred"]).replace([float("inf")], float("nan")).dropna()
        ax.set_title(f"{variant}: median rel err {err.median():.1%}, p90 {err.quantile(0.9):.1%}")
        ax.legend()
    fig.tight_layout()
    fig.savefig(figs / "pred_vs_meas.png", dpi=150)
    plt.close(fig)


def summarize(step_g: pd.DataFrame, df: pd.DataFrame, out: Path) -> None:
    lines = ["# Stage-1 summary", ""]
    tp4 = step_g[step_g["label"] == "tp4"].set_index(["batch", "ctx"])["step_ms"]
    lines.append("## Speedup of each mode over tp4 (CUDA-graph variant, min over repeats; cells with CoV > 5% flagged *)")
    lines.append("")
    lines.append("| batch | ctx | tp4 ms | tp4_fused | tp4_streams | split22 | swing best (S, cuts) | swing speedup | max CoV |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    bs = best_swing(step_g, None)
    for (B, c), t in tp4.sort_index().items():
        def sp(label: str) -> str:
            s = step_g[(step_g["label"] == label) & (step_g["batch"] == B) & (step_g["ctx"] == c)]["step_ms"]
            return f"{t / s.min():.2f}x" if len(s) else "-"

        b = bs[(bs["batch"] == B) & (bs["ctx"] == c)]
        bstr = f"S={b.iloc[0]['S']:.1f}, {b.iloc[0]['cuts_src']}" if len(b) else "-"
        bsp = f"{t / b.iloc[0]['step_ms']:.2f}x" if len(b) else "-"
        cv = step_g[(step_g["batch"] == B) & (step_g["ctx"] == c)]["cov"].max()
        flag = "*" if cv > 0.05 else ""
        lines.append(f"| {B} | {c} | {t:.3f}{flag} | {sp('tp4_fused')} | {sp('tp4_streams')} | {sp('split22')} | {bstr} | {bsp} | {cv:.2f} |")
    corr = Path(out.parent / "raw" / "correctness.csv")
    if corr.exists():
        c = pd.read_csv(corr)
        lines += ["", "## Correctness (vs fp32 reference)", "", f"{int(c['pass'].sum())} / {len(c)} checks passed; "
                  f"worst mean_rel {c['mean_rel'].max():.2e}, worst cos {c['cos'].min():.6f}, max cross-rank diff {c['cross_rank_max_abs'].max():.3g}"]
    # go/no-go
    verdict = []
    for B in (16, 64):
        for c in (8192, 32768):
            t = tp4.get((B, c))
            b = bs[(bs["batch"] == B) & (bs["ctx"] == c) & (bs["cuts_src"].astype(str).str.startswith("model"))]
            if t is not None and len(b):
                verdict.append((B, c, t / b.iloc[0]["step_ms"]))
    lines += ["", "## Go / no-go", ""]
    if verdict:
        lines.append("Swing (model cuts, S<=0.4) over tp4 at the decision cells: " + ", ".join(f"B={B},c={c}: {x:.2f}x" for B, c, x in verdict))
        lines.append("GO" if all(x >= 1.2 for _, _, x in verdict) else "NO-GO on the >=1.2x criterion (see table)")
    else:
        lines.append("decision cells not measured yet")
    out.write_text("\n".join(lines) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="results/raw")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()
    out = Path(args.out)
    figs = out / "figs"
    figs.mkdir(parents=True, exist_ok=True)
    (out / "summary").mkdir(parents=True, exist_ok=True)
    df = load_runs(args.raw)
    step_g = step_table(df, "graph")
    step_e = step_table(df, "eager")
    if not len(step_g):
        step_g = step_e
    plot_layer_time(step_g, step_e, figs)
    plot_imbalance(df, figs)
    plot_collectives(df, figs)
    plot_pred_vs_meas(df, figs)
    summarize(step_g, df, out / "summary" / "summary.md")
    print((out / "summary" / "summary.md").read_text())


if __name__ == "__main__":
    main()
