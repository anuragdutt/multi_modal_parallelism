"""Load raw sweep CSVs into tidy frames."""
from __future__ import annotations

import glob
from pathlib import Path

import pandas as pd

MODE_ORDER = ["tp4", "tp4_fused", "tp4_streams", "split22", "split22_swing"]


def load_runs(raw_dir: str | Path) -> pd.DataFrame:
    files = sorted(glob.glob(str(Path(raw_dir) / "sweep*.csv")) + glob.glob(str(Path(raw_dir) / "dev*.csv")))
    if not files:
        raise FileNotFoundError(f"no sweep/dev CSVs in {raw_dir}")
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df["S"] = df["S"].fillna(0.0).astype(float)
    df["label"] = df.apply(lambda r: f"split22_swing{r['S']:.1f}" if r["mode"] == "split22" and r["S"] > 0 else r["mode"], axis=1)
    return df


def step_table(df: pd.DataFrame, variant: str, agg: str = "min") -> pd.DataFrame:
    """Aggregate over repeats of the max-over-ranks step time per (label, cuts_src, batch, ctx).

    Default is the minimum over repeats: the box is shared and foreign GPU load inflates some repeats, so the
    least-contaminated repeat is the best estimate of the undisturbed step. `cov` flags cells where repeats
    disagree (contamination or thermal drift)."""
    s = df[(df["op"] == "step_max") & (df["variant"] == variant)]
    g = s.groupby(["label", "mode", "S", "cuts_src", "batch", "ctx"], dropna=False)["median_ms"]
    out = (g.min() if agg == "min" else g.median()).reset_index().rename(columns={"median_ms": "step_ms"})
    out["cov"] = (g.std() / g.mean()).reset_index(drop=True)
    out["n_rep"] = g.count().reset_index(drop=True)
    return out


def best_swing(step: pd.DataFrame, src: str | None = None, max_S: float = 0.4) -> pd.DataFrame:
    """Best swing configuration (lowest step) per (batch, ctx), optionally restricted to a cuts source."""
    s = step[(step["mode"] == "split22") & (step["S"] > 0) & (step["S"] <= max_S)]
    if src is not None:
        s = s[s["cuts_src"].astype(str).str.startswith(src)]
    idx = s.groupby(["batch", "ctx"])["step_ms"].idxmin()
    return s.loc[idx].reset_index(drop=True)


def imbalance(df: pd.DataFrame, variant: str = "eager") -> pd.DataFrame:
    """max/mean over ranks of per-rank compute time (sum of non-collective ops) per configuration."""
    e = df[(df["variant"] == variant) & (df["rank"] >= 0) & (~df["op"].str.startswith("ar.")) & (~df["op"].isin(["step", "step_wall"]))]
    per_rank = e.groupby(["run_id", "label", "S", "cuts_src", "batch", "ctx", "repeat", "rank"])["median_ms"].sum().reset_index()
    g = per_rank.groupby(["run_id", "label", "S", "cuts_src", "batch", "ctx", "repeat"])["median_ms"]
    out = (g.max() / g.mean()).reset_index().rename(columns={"median_ms": "imbalance"})
    return out.groupby(["label", "S", "cuts_src", "batch", "ctx"])["imbalance"].median().reset_index()
