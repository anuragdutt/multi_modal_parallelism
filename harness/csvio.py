"""CSV writers with fixed schemas."""
from __future__ import annotations

import csv
import os
from pathlib import Path

RUN_COLUMNS = [
    "run_id", "ts", "image_tag", "vllm_commit", "mode", "S", "w", "cuts", "cuts_src", "batch", "ctx", "kv_block",
    "variant", "repeat", "rank", "op", "n", "median_ms", "p10_ms", "p90_ms", "mean_ms", "bytes_pred", "ms_pred",
    "comm_tp", "comm_pair", "sm_clock_mhz", "temp_c", "tag",
]
CORRECTNESS_COLUMNS = [
    "run_id", "ts", "mode", "S", "cuts", "batch", "ctx", "ref", "max_abs", "max_rel", "mean_rel", "cos",
    "cross_rank_max_abs", "pass",
]


def append_rows(path: str | Path, columns: list[str], rows: list[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists() or path.stat().st_size == 0
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        if new:
            w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in columns})
