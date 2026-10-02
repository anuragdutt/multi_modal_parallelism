"""Per-rank byte model, achieved-bandwidth tables from the microbench, and the swing cut chooser."""
from __future__ import annotations

import csv
import json
import math
from bisect import bisect_left
from dataclasses import dataclass, field
from pathlib import Path

from .config import FalconH1Dims
from .sharding import RankPlan, k_grid, plan_for, swing_width

BF16 = 2
LAUNCH_US = {"eager": 4.0, "graph": 1.0}


@dataclass
class Curve:
    """bytes -> GB/s (log-interpolated, flat beyond the ends); or bytes -> us when `is_time`."""
    xs: list[float] = field(default_factory=list)
    ys: list[float] = field(default_factory=list)
    is_time: bool = False

    def add(self, x: float, y: float) -> None:
        self.xs.append(x)
        self.ys.append(y)

    def finalize(self) -> "Curve":
        pairs = sorted(zip(self.xs, self.ys))
        self.xs = [p[0] for p in pairs]
        self.ys = [p[1] for p in pairs]
        return self

    def at(self, x: float) -> float:
        if not self.xs:
            return float("nan")
        if x <= self.xs[0]:
            return self.ys[0]
        if x >= self.xs[-1]:
            return self.ys[-1]
        i = bisect_left(self.xs, x)
        x0, x1, y0, y1 = self.xs[i - 1], self.xs[i], self.ys[i - 1], self.ys[i]
        t = (math.log(x) - math.log(x0)) / (math.log(x1) - math.log(x0))
        return y0 + t * (y1 - y0)

    def time_ms(self, nbytes: float) -> float:
        if self.is_time:
            return self.at(nbytes) / 1e3
        gbps = self.at(nbytes)
        return nbytes / (gbps * 1e9) * 1e3 if gbps > 0 else float("nan")


@dataclass
class BwTables:
    gemv: Curve
    attn: dict[int, Curve]  # by kv_block
    ssu: Curve
    conv: Curve
    ar: dict[tuple[int, str], Curve]  # (group_size, variant) -> us curve
    stream: float = 0.0  # peak streaming GB/s

    @classmethod
    def from_csv(cls, path: str | Path) -> "BwTables":
        rows = list(csv.DictReader(open(path)))
        gemv_by: dict[str, Curve] = {}
        conv_by: dict[str, Curve] = {}
        ssu_by: dict[str, Curve] = {}
        attn_by: dict[tuple[int, str], Curve] = {}
        ar: dict[tuple[int, str], Curve] = {}
        stream = 0.0
        for r in rows:
            if r["rank"] not in ("0", "-1"):
                continue
            k, p = r["kernel"], json.loads(r["params_json"])
            b, gbps, ms = float(r["bytes"]), float(r["GBps"]), float(r["median_ms"])
            if k == "stream":
                stream = max(stream, gbps)
            elif k == "gemv":
                gemv_by.setdefault(r["variant"], Curve()).add(b, gbps)
            elif k == "attn":
                attn_by.setdefault((int(p["kv_block"]), r["variant"]), Curve()).add(b, gbps)
            elif k == "ssu":
                ssu_by.setdefault(r["variant"], Curve()).add(b, gbps)
            elif k == "conv":
                conv_by.setdefault(r["variant"], Curve()).add(b, gbps)
            elif k == "allreduce":
                key = (int(p["group_size"]), r["variant"])
                ar.setdefault(key, Curve(is_time=True)).add(b, ms * 1e3)
        # graph-captured curves exclude launch overhead; prefer them wherever both exist
        def pick(by: dict[str, Curve]) -> Curve:
            return by.get("graph") or by.get("eager") or Curve()

        gemv, ssu, conv = pick(gemv_by), pick(ssu_by), pick(conv_by)
        attn: dict[int, Curve] = {}
        for (kvb, var), cur in attn_by.items():
            if var == "graph" or kvb not in attn:
                attn[kvb] = cur
        for c in [gemv, ssu, conv, *attn.values(), *ar.values()]:
            c.finalize()
        return cls(gemv=gemv, attn=attn, ssu=ssu, conv=conv, ar=ar, stream=stream)


def bytes_per_op(plan: RankPlan, dims: FalconH1Dims, B: int, c: int) -> dict[str, int]:
    """Bytes moved per op on this rank for one decode step (weights read, KV read, states read+write)."""
    H, hd = dims.hidden, dims.head_dim
    out: dict[str, int] = {"norm_in": 3 * B * H * BF16, "norm_ff": 3 * B * H * BF16, "residual": 3 * B * H * BF16}
    if plan.attn is not None:
        a = plan.attn
        nq, nkv = len(a.q_heads), len(a.kv_heads)
        out["attn.qkv"] = (nq + 2 * nkv) * hd * H * BF16 + B * H * BF16
        out["attn.rope"] = 2 * B * (nq + nkv) * hd * BF16
        out["attn.kvwrite"] = 2 * B * nkv * hd * BF16
        out["attn.fa"] = 2 * B * (c + 1) * nkv * hd * BF16 + B * nq * hd * BF16
        out["attn.oproj"] = H * nq * hd * BF16
    if plan.ssm is not None:
        s = plan.ssm
        nhr, p, N = len(s.heads), dims.mamba_head_dim, dims.d_state
        in_rows = sum(s.local_in_rows)
        conv_rows = sum(r.stop - r.start for r in s.conv_rows)
        out["ssm.inproj"] = in_rows * H * BF16 + B * H * BF16
        out["ssm.conv"] = 2 * B * (dims.d_conv - 1) * conv_rows * BF16 + conv_rows * dims.d_conv * BF16
        out["ssm.ssu"] = 2 * B * nhr * p * N * BF16 + 3 * B * nhr * p * BF16
        out["ssm.norm"] = 4 * B * nhr * p * BF16
        out["ssm.outproj"] = H * nhr * p * BF16
    n = plan.mlp.active_rows
    out["mlp.gateup"] = 2 * n * H * BF16 + 2 * B * H * BF16
    out["mlp.act"] = 3 * B * n * BF16
    out["mlp.down"] = H * n * BF16
    return out


COLLECTIVES = {
    "tp1": [],
    "tp4": [("ar.attn", "tp"), ("ar.norm", "norm"), ("ar.ssm", "tp"), ("ar.mlp", "tp")],
    "tp4_fused": [("ar.norm", "norm"), ("ar.mixer", "tp"), ("ar.mlp", "tp")],
    "tp4_streams": [("ar.norm", "norm"), ("ar.mixer", "tp"), ("ar.mlp", "tp")],
    "split22": [("ar.norm", "norm"), ("ar.mixer", "tp"), ("ar.mlp", "tp")],
}

CLASS = {
    "attn.qkv": "gemv", "attn.oproj": "gemv", "ssm.inproj": "gemv", "ssm.outproj": "gemv",
    "mlp.gateup": "gemv", "mlp.down": "gemv",
    "attn.fa": "attn", "ssm.ssu": "ssu", "ssm.conv": "conv",
}


def predict_rank(plan: RankPlan, dims: FalconH1Dims, B: int, c: int, bw: BwTables, variant: str,
                 kv_block: int = 16) -> dict[str, float]:
    """Per-op predicted ms on this rank, plus 'compute' (sum of non-collective ops) and 'step'."""
    byt = bytes_per_op(plan, dims, B, c)
    launch = LAUNCH_US[variant] / 1e3
    pred: dict[str, float] = {}
    for op, nb in byt.items():
        cls = CLASS.get(op)
        if cls == "gemv":
            t = bw.gemv.time_ms(nb)
        elif cls == "attn":
            curve = bw.attn.get(kv_block) or next(iter(bw.attn.values()), None)
            t = curve.time_ms(nb) if curve else nb / (bw.stream * 1e6)
        elif cls == "ssu":
            t = bw.ssu.time_ms(nb)
        elif cls == "conv":
            t = bw.conv.time_ms(nb)
        else:
            t = nb / (bw.stream * 1e6) if bw.stream else 0.0
        pred[op] = t + launch
    # branch chains for streams: overlap attention and ssm
    attn_ops = [k for k in pred if k.startswith("attn.")]
    ssm_ops = [k for k in pred if k.startswith("ssm.")]
    if plan.mode == "tp4_streams":
        chain = max(sum(pred[k] for k in attn_ops), sum(pred[k] for k in ssm_ops))
        compute = chain + sum(v for k, v in pred.items() if not (k.startswith("attn.") or k.startswith("ssm.")))
    else:
        compute = sum(pred.values())
    pred["compute"] = compute
    msg_tp = B * dims.hidden * BF16
    msg_norm = B * 4
    coll = 0.0
    for name, kind in COLLECTIVES[plan.mode]:
        if kind == "norm":
            if plan.ssm is None:
                continue  # attention ranks of split22 do not join the norm reduction
            gs = len(plan.ssm.norm_ranks)
            if gs <= 1:
                continue
            curve = bw.ar.get((gs, variant)) or bw.ar.get((gs, "eager"))
            t = curve.time_ms(msg_norm) if curve else 0.02
        else:
            curve = bw.ar.get((plan.world, variant)) or bw.ar.get((plan.world, "eager"))
            t = curve.time_ms(msg_tp) if curve else 0.02
        pred[name] = t
        coll += t
    pred["collectives"] = coll
    pred["step"] = compute + coll
    return pred


def predict_step(dims: FalconH1Dims, mode: str, S: float, k: int | None, B: int, c: int,
                 bw: BwTables, variant: str, world: int = 4, kv_block: int = 16) -> tuple[float, list[float]]:
    per_rank = []
    for r in range(world):
        plan = plan_for(mode, r, world, dims, S, k)
        per_rank.append(predict_rank(plan, dims, B, c, bw, variant, kv_block)["step"])
    return max(per_rank), per_rank


def choose_k(dims: FalconH1Dims, B: int, c: int, S: float, bw: BwTables, variant: str,
             kv_block: int = 16, step: int = 128) -> tuple[int, float]:
    """Minimise the predicted max-rank step time over the split point k."""
    w = swing_width(S, dims.intermediate // 4)
    best, best_t = w, float("inf")
    for k in k_grid(w, step):
        t, _ = predict_step(dims, "split22", S, k, B, c, bw, variant, 4, kv_block)
        if t < best_t:
            best, best_t = k, t
    return best, best_t


def total_hbm_bytes(dims: FalconH1Dims, mode: str, S: float, k: int | None, B: int, c: int, world: int = 4) -> int:
    return sum(sum(bytes_per_op(plan_for(mode, r, world, dims, S, k), dims, B, c).values()) for r in range(world))
