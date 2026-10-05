"""Build every figure in blog/figs from the committed results (no GPU, no raw CSVs needed).

  python blog/make_figs.py            # from the repo root

Result plots are parsed from results/*/summary/summary.md (the CUDA-graph, min-over-repeats tables that
analysis/make_all.py writes) and results/tp2_reference.csv. The schematics are drawn as plain SVG so they
render on GitHub and in any Markdown preview without Mermaid.
"""
from __future__ import annotations

import csv
import re
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FIGS = ROOT / "blog" / "figs"
FIGS.mkdir(parents=True, exist_ok=True)

MODELS = [  # (label, results dir)
    ("Falcon-H1-7B\n2 KV heads, 24 Mamba heads", "full"),
    ("Falcon-H1-3B\n2 KV heads, 10 q heads (uneven tp4)", "h1_3b"),
    ("Falcon-H1-34B\n4 KV heads, 32 Mamba heads", "h1_34b"),
    ("Falcon-40B (parallel residual)\n8 KV heads, no SSM", "falcon40b"),
]
BATCHES = [1, 4, 16, 64]
CTXS = [512, 2048, 8192, 32768]


# ----------------------------------------------------------------------------------------------- data

def parse_summary(path: Path) -> dict[tuple[int, int], dict]:
    """Rows of the speedup table: (batch, ctx) -> {tp4_ms, fused, streams, split22, cov, flagged}."""
    out = {}
    for line in path.read_text().splitlines():
        m = re.match(r"\|\s*(\d+)\s*\|\s*(\d+)\s*\|\s*([\d.]+)(\*?)\s*\|(.*)", line)
        if not m:
            continue
        b, c, tp4, flag, rest = int(m[1]), int(m[2]), float(m[3]), m[4] == "*", m[5]
        cells = [x.strip() for x in rest.split("|")]

        def x(s: str) -> float:
            return float(s.rstrip("x")) if s not in ("-", "") else float("nan")

        out[(b, c)] = {"tp4_ms": tp4, "fused": x(cells[0]), "streams": x(cells[1]), "split22": x(cells[2]),
                       "cov": float(cells[-2]) if len(cells) >= 7 else float("nan"), "flagged": flag}
    return out


def tp2_reference() -> dict[tuple[str, int], float]:
    d = defaultdict(list)
    with open(ROOT / "results" / "tp2_reference.csv") as f:
        for r in csv.DictReader(f):
            if r["op"] == "step_max" and r["variant"] == "graph":
                d[(r["tag"], int(r["ctx"]))].append(float(r["median_ms"]))
    return {k: min(v) for k, v in d.items()}


# -------------------------------------------------------------------------------------------- figures

def fig_speedup_heatmaps(summaries: dict[str, dict]) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.9))
    vmin, vmax = 0.6, 1.45
    cmap = plt.get_cmap("RdBu")
    norm = matplotlib.colors.TwoSlopeNorm(vmin=vmin, vcenter=1.0, vmax=vmax)
    for ax, (label, key) in zip(axes, MODELS):
        s = summaries[key]
        grid = np.array([[s[(b, c)]["split22"] for c in CTXS] for b in BATCHES])
        ax.imshow(grid, cmap=cmap, norm=norm)
        for i, b in enumerate(BATCHES):
            for j, c in enumerate(CTXS):
                v = s[(b, c)]
                txt = f"{v['split22']:.2f}x" + ("*" if v["flagged"] else "")
                ax.text(j, i, txt, ha="center", va="center", fontsize=9,
                        color="white" if abs(v["split22"] - 1) > 0.22 else "black")
        ax.set_xticks(range(4), [f"{c // 1024}K" if c >= 1024 else str(c) for c in CTXS])
        ax.set_yticks(range(4), BATCHES)
        ax.set_xlabel("context (tokens)")
        ax.set_title(label, fontsize=9.5)
    axes[0].set_ylabel("decode batch")
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    fig.colorbar(sm, ax=axes, fraction=0.02, pad=0.02, label="split22 speedup over tp4")
    fig.suptitle("Branch split (2 attention ranks + 2 Mamba ranks) vs tensor parallelism, one decoder layer, "
                 "4x A6000 PCIe, CUDA-graph step, min over 3 repeats  (* = repeat CoV > 5%)", fontsize=10, y=1.02)
    fig.savefig(FIGS / "split22_speedup_heatmaps.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def fig_pred_vs_meas(summaries: dict[str, dict]) -> None:
    """Pre-registered recipe predictions (doc section 23) against the measured split22 speedups."""
    cells = [(1, 512), (16, 512), (16, 8192), (64, 8192), (64, 32768)]
    pred = {  # doc/stage1_elastic_mixer_parallelism_plan.md section 23, split22 (attention 2 + mamba 2 / mlp 2)
        "full": [1.35, 1.29, 1.44, 1.31, 1.12],
        "h1_3b": [1.39, 1.25, 1.47, 1.37, 1.14],
        "h1_34b": [1.18, 1.12, 1.09, 0.91, 0.66],
        "falcon40b": [0.73, 0.74, 0.81, 1.02, 0.68],
    }
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.6), sharey=True)
    w = 0.38
    for ax, (label, key) in zip(axes, MODELS):
        meas = [summaries[key][c]["split22"] for c in cells]
        xs = np.arange(len(cells))
        ax.bar(xs - w / 2, pred[key], w, label="recipe prediction (before the run)", color="C7")
        ax.bar(xs + w / 2, meas, w, label="measured", color="C3")
        ax.axhline(1.0, color="k", lw=0.8, ls="--")
        ax.set_xticks(xs, [f"B{b}\nc{c // 1024}K" if c >= 1024 else f"B{b}\nc{c}" for b, c in cells], fontsize=8)
        ax.set_title(label.split("\n")[0], fontsize=10)
        ax.set_ylim(0.3, 1.6)
        ax.grid(True, axis="y", alpha=0.3)
    axes[0].set_ylabel("split22 speedup over tp4")
    axes[3].legend(fontsize=8, loc="upper left")
    fig.suptitle("Recipe predictions vs measurement at the five pre-registered cells "
                 "(predictions scored with eager kernel curves; the sign is what was at stake)", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIGS / "recipe_pred_vs_meas.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def fig_phase_balance() -> None:
    """Per-rank compute with collectives removed (doc section 22; 7B, batch 16, graph variant)."""
    ctx = ["512", "8192"]
    data = {"tp4 rank (both branches + MLP/4)": [0.265, 0.368],
            "split22 attention rank (6 q heads, 1 KV head + MLP/4)": [0.174, 0.278],
            "split22 Mamba rank (12 heads + MLP/4)": [0.262, 0.262]}
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    xs = np.arange(2)
    w = 0.26
    for i, (k, v) in enumerate(data.items()):
        bars = ax.bar(xs + (i - 1) * w, v, w, label=k, color=["C0", "C1", "C2"][i])
        for b_, val in zip(bars, v):
            ax.text(b_.get_x() + b_.get_width() / 2, val + 0.006, f"{val:.3f}", ha="center", fontsize=8)
    ax.set_xticks(xs, [f"context {c}" for c in ctx])
    ax.set_ylabel("ms per layer, compute only (no collectives)")
    ax.set_title("Falcon-H1-7B, batch 16: the split is Mamba-bound at short context, attention-bound at long context",
                 fontsize=9)
    ax.set_ylim(0, 0.43)
    ax.legend(fontsize=7.5, loc="upper left")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGS / "phase_balance_no_collectives.png", dpi=160)
    plt.close(fig)


def fig_tp2_reference(summaries: dict[str, dict]) -> None:
    tp2 = tp2_reference()
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.4), sharey=True)
    for ax, (name, tag, key) in zip(axes, [("Falcon-H1-3B", "tp2_3b", "h1_3b"), ("Falcon-H1-7B", "tp2_7b", "full")]):
        ctxs = [512, 8192, 32768]
        t2 = [tp2[(tag, c)] for c in ctxs]
        t4 = [summaries[key][(16, c)]["tp4_ms"] for c in ctxs]
        sp = [summaries[key][(16, c)]["tp4_ms"] / summaries[key][(16, c)]["split22"] for c in ctxs]
        xs = np.arange(3)
        w = 0.27
        for i, (v, lab, col) in enumerate([(t2, "tp2 on 2 GPUs (vLLM's cap for the 3B)", "C7"),
                                           (t4, "tp4 on 4 GPUs (uneven heads for the 3B)", "C0"),
                                           (sp, "split22 on 4 GPUs", "C3")]):
            bars = ax.bar(xs + (i - 1) * w, v, w, label=lab, color=col)
            for b_, val in zip(bars, v):
                ax.text(b_.get_x() + b_.get_width() / 2, val + 0.01, f"{val:.3f}", ha="center", fontsize=7)
        ax.set_xticks(xs, [f"c={c}" for c in ctxs])
        ax.set_title(f"{name}, batch 16, graph step", fontsize=10)
        ax.grid(True, axis="y", alpha=0.3)
    axes[0].set_ylabel("ms per layer (max over ranks)")
    axes[0].legend(fontsize=7.5)
    fig.tight_layout()
    fig.savefig(FIGS / "tp2_vs_split22.png", dpi=160)
    plt.close(fig)


# ----------------------------------------------------------------------------------------- SVG helper

class Svg:
    FONT = "Helvetica, Arial, 'DejaVu Sans', sans-serif"

    def __init__(self, w: int, h: int):
        self.w, self.h = w, h
        self.parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}" '
                      f'font-family="{self.FONT}" font-size="12">',
                      '<defs><marker id="ar" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" '
                      'orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="#333"/></marker></defs>',
                      f'<rect width="{w}" height="{h}" fill="white"/>']

    def box(self, x, y, w, h, lines, fill="#eef3fb", stroke="#3b5b8c", fs=12, bold_first=False, rx=6, dash=False):
        sd = ' stroke-dasharray="5,3"' if dash else ""
        self.parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" stroke="{stroke}" '
                          f'stroke-width="1.4"{sd}/>')
        if isinstance(lines, str):
            lines = [lines]
        n = len(lines)
        y0 = y + h / 2 - (n - 1) * (fs + 3) / 2
        for i, ln in enumerate(lines):
            fw = ' font-weight="bold"' if (bold_first and i == 0) else ""
            self.parts.append(f'<text x="{x + w / 2}" y="{y0 + i * (fs + 3)}" text-anchor="middle" '
                              f'dominant-baseline="middle" font-size="{fs}"{fw}>{esc(ln)}</text>')

    def text(self, x, y, s, fs=12, anchor="start", bold=False, color="#222", italic=False):
        fw = ' font-weight="bold"' if bold else ""
        fi = ' font-style="italic"' if italic else ""
        self.parts.append(f'<text x="{x}" y="{y}" font-size="{fs}" text-anchor="{anchor}" fill="{color}"{fw}{fi}>'
                          f'{esc(s)}</text>')

    def arrow(self, x1, y1, x2, y2, label=None, color="#333", dash=False, lw=1.4):
        sd = ' stroke-dasharray="5,3"' if dash else ""
        self.parts.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="{lw}" '
                          f'marker-end="url(#ar)"{sd}/>')
        if label:
            self.text((x1 + x2) / 2 + 4, (y1 + y2) / 2 - 4, label, fs=10, color="#444")

    def line(self, x1, y1, x2, y2, color="#333", dash=False, lw=1.2):
        sd = ' stroke-dasharray="5,3"' if dash else ""
        self.parts.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="{lw}"{sd}/>')

    def rect(self, x, y, w, h, fill, stroke="none", label=None, fs=11, color="#222"):
        self.parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" fill="{fill}" stroke="{stroke}"/>')
        if label:
            self.parts.append(f'<text x="{x + w / 2}" y="{y + h / 2}" text-anchor="middle" dominant-baseline="middle" '
                              f'font-size="{fs}" fill="{color}">{esc(label)}</text>')

    def save(self, name: str) -> None:
        self.parts.append("</svg>")
        (FIGS / name).write_text("\n".join(self.parts))


def esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


# -------------------------------------------------------------------------------------- schematics

def svg_layer_dataflow() -> None:
    """Falcon-H1 decoder layer as vLLM runs it under TP (4 collectives) vs the branch split (2 + 1 pair)."""
    s = Svg(1000, 575)
    AR, ARS, PAIR = "#f9d6d6", "#c0392b", "#fde9c8"

    def column(x0, title, subtitle, variant):
        s.text(x0 + 220, 24, title, fs=14, anchor="middle", bold=True)
        s.text(x0 + 220, 42, subtitle, fs=11, anchor="middle", color="#555", italic=True)
        cx = x0 + 220
        s.box(cx - 70, 60, 140, 30, "input RMSNorm", fill="#f4f4f4", stroke="#777")
        s.arrow(cx, 90, cx - 110, 130)
        s.arrow(cx, 90, cx + 110, 130)
        ax, mx = cx - 190, cx + 30
        ranks_a, ranks_m = ("ranks 0-3", "ranks 0-3") if variant == "tp" else ("ranks 0,1", "ranks 2,3")
        s.text(ax, 104, ranks_a, fs=10, color="#c0392b", bold=True)
        s.text(ax, 122, "x * attn_in", fs=10, color="#555")
        s.text(mx + 160, 104, ranks_m, fs=10, color="#c0392b", bold=True, anchor="end")
        s.text(mx + 160, 122, "x * ssm_in", fs=10, color="#555", anchor="end")
        # attention branch
        s.box(ax, 130, 160, 30, "qkv_proj (col-parallel)")
        s.arrow(ax + 80, 160, ax + 80, 174)
        s.box(ax, 174, 160, 44, ["paged FA2 decode", "KV head(s) of this rank"])
        s.arrow(ax + 80, 218, ax + 80, 232)
        s.box(ax, 232, 160, 30, "o_proj (row-parallel)")
        # mamba branch
        s.box(mx, 130, 160, 30, "in_proj [z|x|B|C|dt]")
        s.arrow(mx + 80, 160, mx + 80, 174)
        s.box(mx, 174, 160, 44, ["conv1d update +", "selective_state_update"])
        s.arrow(mx + 80, 218, mx + 80, 232)
        if variant == "tp":
            s.box(mx, 232, 160, 30, "gated RMSNorm: all-reduce [T,1]", fill=AR, stroke=ARS, fs=10.5)
        else:
            s.box(mx, 232, 160, 30, "gated RMSNorm: pair all-reduce", fill=PAIR, stroke="#c77d00", fs=10.5)
        s.arrow(mx + 80, 262, mx + 80, 276)
        s.box(mx, 276, 160, 30, "out_proj (row-parallel)")
        if variant == "tp":
            s.arrow(ax + 80, 262, ax + 80, 276)
            s.box(ax, 276, 160, 30, "all-reduce (4 ranks)", fill=AR, stroke=ARS)
            s.arrow(mx + 80, 306, mx + 80, 320)
            s.box(mx, 320, 160, 30, "all-reduce (4 ranks)", fill=AR, stroke=ARS)
            s.arrow(ax + 80, 306, cx - 20, 366)
            s.arrow(mx + 80, 350, cx + 20, 366)
            s.box(cx - 90, 366, 180, 30, "attn*attn_out + ssm*ssm_out + h", fill="#f4f4f4", stroke="#777", fs=10.5)
        else:
            s.arrow(ax + 80, 262, cx - 20, 366)
            s.arrow(mx + 80, 306, cx + 20, 366)
            s.box(cx - 90, 366, 180, 30, ["one all-reduce (4 ranks) of", "attn*attn_out + ssm*ssm_out"], fill=AR,
                  stroke=ARS, fs=10.5)
        s.arrow(cx, 396, cx, 410)
        s.box(cx - 70, 410, 140, 30, "pre-FF RMSNorm", fill="#f4f4f4", stroke="#777")
        s.arrow(cx, 440, cx, 454)
        s.box(cx - 90, 454, 180, 30, "gate/up (col) - SiLU - down (row)")
        s.arrow(cx, 484, cx, 498)
        s.box(cx - 70, 498, 140, 30, "all-reduce (4 ranks)", fill=AR, stroke=ARS)
        s.arrow(cx, 528, cx, 542)
        s.text(cx, 558, "+ residual", fs=11, anchor="middle", color="#555")

    column(20, "vLLM today: tensor parallelism (tp4)", "every rank runs both branches; 4 four-rank collectives per layer",
           "tp")
    column(540, "Branch split (split22)", "ranks 0,1 run attention, ranks 2,3 run Mamba; 2 four-rank + 1 pair collective",
           "split")
    s.line(510, 20, 510, 565, color="#bbb", dash=True)
    s.save("layer_dataflow.svg")


def svg_rank_layouts() -> None:
    """What each rank owns under tp4 and split22 for Falcon-H1-7B, with the per-step byte consequences."""
    s = Svg(1000, 430)
    s.text(500, 24, "Falcon-H1-7B, 4 ranks: what each rank owns and reads per decode step", fs=14, anchor="middle",
           bold=True)
    cols = [120 + i * 215 for i in range(4)]
    for i, x in enumerate(cols):
        s.text(x + 100, 50, f"rank {i}", fs=12, anchor="middle", bold=True)
    # tp4 row
    s.text(14, 80, "tp4", fs=13, bold=True)
    s.text(14, 96, "(vLLM)", fs=10, color="#555")
    for i, x in enumerate(cols):
        kv = i // 2
        s.box(x, 62, 200, 118, [f"q heads {3 * i}-{3 * i + 2}",
                                f"KV head {kv}  (replica {i % 2 + 1} of 2)",
                                f"Mamba heads {6 * i}-{6 * i + 5}, B/C replicated",
                                "MLP columns 1/4",
                                "weights 79.4 MB",
                                "reads full KV head: 512 B x B x c"], fs=10.5)
    s.text(500, 200, "aggregate KV traffic = 2x what the model needs: both replicas of a head read all of it every step; "
                     "4 four-rank all-reduces per layer", fs=10.5, anchor="middle", color="#c0392b")
    # split22 row
    s.text(14, 250, "split22", fs=13, bold=True)
    for i, x in enumerate(cols):
        if i < 2:
            s.box(x, 232, 200, 118, ["attention group", f"q heads {6 * i}-{6 * i + 5}", f"KV head {i}  (no replica)",
                                     "MLP columns 1/4", "weights 67.6 MB", "reads 512 B x B x c of KV, no SSM state"],
                  fill="#e8f4ea", stroke="#2e7d32", fs=10.5, bold_first=True)
        else:
            j = i - 2
            s.box(x, 232, 200, 118, ["Mamba group", f"Mamba heads {12 * j}-{12 * j + 11}", "B/C replicated on the pair",
                                     "MLP columns 1/4", "weights 88.2 MB", "reads 1.57 MB x B of state, no KV"],
                  fill="#fff3e0", stroke="#ef6c00", fs=10.5, bold_first=True)
    s.text(500, 372, "each rank carries one branch plus its MLP slice; KV traffic matches the model; "
                     "2 four-rank all-reduces + 1 pair all-reduce per layer", fs=10.5, anchor="middle", color="#2e7d32")
    s.text(500, 400, "Why this cannot be a static win everywhere: the attention branch grows with context, the Mamba branch "
                     "does not, so the slower group changes with (batch, context).", fs=10.5, anchor="middle",
           color="#444", italic=True)
    s.save("rank_layouts.svg")


def svg_barrier_timeline() -> None:
    """Why swing MLP shards cannot help: each all-reduce is a barrier, so the step is a sum of per-phase maxima."""
    s = Svg(1000, 540)
    s.text(500, 24, "Why moving MLP columns cannot fix branch imbalance (short context, Mamba-bound)", fs=14,
           anchor="middle", bold=True)
    ATT, SSM, ARC, IDLE, MLP = "#a5d6a7", "#ffcc80", "#ef9a9a", "#eeeeee", "#90caf9"
    x0, scale = 150, 5.2  # px per unit

    def row(y, label, segs):
        s.text(x0 - 8, y + 15, label, fs=11, anchor="end")
        x = x0
        for width, fill, txt in segs:
            s.rect(x, y, width * scale, 22, fill, stroke="#777", label=txt, fs=9.5)
            x += width * scale
        return x

    def panel(y, title, attn_branch, ssm_branch, attn_mlp, ssm_mlp, note):
        s.text(x0, y - 6, title, fs=12, bold=True)
        bmax = max(attn_branch, ssm_branch)
        mmax = max(attn_mlp, ssm_mlp)
        ar = 10
        e1 = row(y + 6, "attention ranks", [(attn_branch, ATT, "attention branch"), (bmax - attn_branch, IDLE, "idle" if bmax - attn_branch > 4 else ""),
                                           (ar, ARC, "AR"), (attn_mlp, MLP, "MLP"), (mmax - attn_mlp, IDLE, "idle" if mmax - attn_mlp > 4 else ""),
                                           (ar, ARC, "AR")])
        row(y + 32, "Mamba ranks", [(ssm_branch, SSM, "Mamba branch"), (bmax - ssm_branch, IDLE, "idle" if bmax - ssm_branch > 4 else ""),
                                    (ar, ARC, "AR"), (ssm_mlp, MLP, "MLP"), (mmax - ssm_mlp, IDLE, "idle" if mmax - ssm_mlp > 4 else ""),
                                    (ar, ARC, "AR")])
        total = bmax + mmax + 2 * ar
        s.line(x0, y + 62, x0 + total * scale, y + 62, color="#333")
        s.line(x0 + total * scale, y + 2, x0 + total * scale, y + 62, color="#333", dash=True)
        s.text(x0, y + 78, f"step = max(branch) + AR + max(MLP) + AR = {total} units", fs=10.5, bold=True)
        s.text(x0, y + 96, note, fs=10.5, color="#444", italic=True)
        return e1

    panel(60, "(a) neutral split: MLP evenly sharded", 40, 70, 30, 30,
          "attention ranks idle for 30 units in the branch phase; both all-reduces are barriers")
    panel(210, "(b) swing MLP shards (H3): move MLP columns to the idle attention ranks", 40, 70, 45, 15,
          "the idle time sits before the first barrier, the extra MLP work sits after it: max(MLP) rises, the step gets longer")
    panel(360, "(c) what works: move Mamba-branch work that does not need the recurrence (gate rows z) to attention ranks",
          40 + 14, 70 - 14, 30, 30,
          "balancing inside the phase lowers max(branch); the z result is consumed late (gated norm), so its P2P send is off the critical path")
    s.text(500, 520, "Measured on the 7B (batch 16, context 512): with collectives removed, swing columns on the attention ranks balance "
                     "compute (0.233 vs 0.263 ms), with collectives present no swing setting ever beat the neutral split.",
           fs=10, anchor="middle", color="#444")
    s.save("barrier_timeline.svg")


def svg_recipe_pipeline() -> None:
    s = Svg(1000, 330)
    s.text(500, 24, "The recipe (harness/synth.py): from a model's layer structure to ranked rank-group layouts", fs=14,
           anchor="middle", bold=True)
    y = 60
    s.box(20, y, 150, 90, ["HF config.json", "heads, KV heads,", "SSM heads/groups,", "experts, layer order"],
          fill="#f4f4f4", stroke="#777", fs=10.5, bold_first=True)
    s.arrow(170, y + 45, 196, y + 45)
    s.box(196, y, 170, 90, ["template", "falcon_h1 (attn || mamba)", "parallel_residual (attn || mlp)",
                            "sequential_hybrid (layer types)"], fs=10.5, bold_first=True)
    s.arrow(366, y + 45, 392, y + 45)
    s.box(392, y, 200, 90, ["enumerate candidates", "R1 group <= unit count (no replication)",
                            "R2 summed branches: one all-reduce", "R3 balance inside each phase"],
          fill="#fff8e1", stroke="#f9a825", fs=10.5, bold_first=True)
    s.arrow(592, y + 45, 618, y + 45)
    s.box(618, y, 180, 90, ["score each candidate", "bytes per rank per phase /", "microbench bandwidth curve",
                            "+ measured all-reduce cost"], fill="#e3f2fd", stroke="#1565c0", fs=10.5, bold_first=True)
    s.arrow(798, y + 45, 824, y + 45)
    s.box(824, y, 160, 90, ["ranked layouts", "tp4 / tp4_fused /", "split22 / split13 / ...", "step = sum of phase maxima"],
          fill="#e8f5e9", stroke="#2e7d32", fs=10.5, bold_first=True)
    s.arrow(904, y + 90, 904, y + 120)
    s.box(560, y + 120, 424, 60, ["the harness measures the top candidates against tp4 on vLLM's kernels",
                                  "(same weights, same kernels, same NCCL groups; fp32 reference check first)"],
          fill="#f4f4f4", stroke="#777", fs=10.5)
    s.text(20, 215, "Rules are measured effects, not assumptions:", fs=11.5, bold=True)
    s.text(20, 236, "R1  replicated KV heads cost bandwidth every step (7B: 2 KV heads at 4 ranks, both replicas read the head)", fs=10.5)
    s.text(20, 254, "R2  o_proj and out_proj partials are summed anyway, so one all-reduce of the scaled sum replaces two (tp4_fused)", fs=10.5)
    s.text(20, 272, "R3  every all-reduce is a barrier: the step is sum over phases of the slowest group, so work moves only within a phase", fs=10.5)
    s.text(20, 300, "Homogeneous TP is just the candidate where every group is the whole world; the recipe says 'tp4' for Falcon-40B.", fs=10.5,
           italic=True, color="#444")
    s.save("recipe_pipeline.svg")


def svg_harness_stack() -> None:
    s = Svg(1000, 300)
    s.text(500, 24, "Measurement stack (stage 1): a 4-rank layer harness on vLLM's own kernels and collectives", fs=14,
           anchor="middle", bold=True)
    s.box(20, 50, 300, 60, ["docker image mmp:stage1", "vllm/vllm-openai:v0.30.0 + editable clone of v0.30.0",
                            "(compiled ops copied into the tree)"], fill="#f4f4f4", stroke="#777", fs=10.5, bold_first=True)
    s.box(340, 50, 320, 60, ["harness/ (torchrun, 4 ranks)", "sharding.py: who owns which heads/columns",
                             "layer.py: one decoder layer per mode"], fs=10.5, bold_first=True)
    s.box(680, 50, 300, 60, ["vLLM kernels, unmodified", "paged FA2 decode, causal_conv1d_update,",
                             "selective_state_update, rms_norm_gated, silu_and_mul"], fill="#e3f2fd", stroke="#1565c0",
          fs=10.5, bold_first=True)
    s.box(20, 130, 300, 60, ["vLLM distributed groups", "init_distributed_environment + initialize_model_parallel",
                             "4-rank TP group, 2-rank pair groups (pynccl)"], fill="#e3f2fd", stroke="#1565c0", fs=10.5,
          bold_first=True)
    s.box(340, 130, 320, 60, ["measure.py", "20 warm + 100 timed steps, CUDA events per op (eager)",
                              "+ CUDA-graph capture of the whole step (headline)"], fs=10.5, bold_first=True)
    s.box(680, 130, 300, 60, ["correctness.py / reference.py", "fp32 pure-torch layer; every mode must match",
                              "(mean rel err <= 1.5e-2, cos >= 0.9999) before a sweep"], fill="#e8f5e9", stroke="#2e7d32",
          fs=10.5, bold_first=True)
    s.box(20, 210, 460, 60, ["outputs: results/<model>/raw/*.csv (per op, per rank, per repeat)",
                             "analysis/make_all.py: min over repeats + CoV flags, figures, summary.md"],
          fill="#fff8e1", stroke="#f9a825", fs=10.5, bold_first=True)
    s.box(500, 210, 480, 60, ["hardware: guppy, 4x RTX A6000 48 GB, PCIe Gen4 only (no NVLink), shared box",
                              "custom all-reduce disabled by vLLM on this topology, NCCL 2.30.7 for every collective"],
          fill="#f4f4f4", stroke="#777", fs=10.5, bold_first=True)
    s.save("harness_stack.svg")


def main() -> None:
    summaries = {key: parse_summary(ROOT / "results" / key / "summary" / "summary.md") for _, key in MODELS}
    fig_speedup_heatmaps(summaries)
    fig_pred_vs_meas(summaries)
    fig_phase_balance()
    fig_tp2_reference(summaries)
    svg_layer_dataflow()
    svg_rank_layouts()
    svg_barrier_timeline()
    svg_recipe_pipeline()
    svg_harness_stack()
    for p in sorted(FIGS.iterdir()):
        print(p.relative_to(ROOT), p.stat().st_size)


if __name__ == "__main__":
    main()
