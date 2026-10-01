"""Compare a mode's layer output against the fp32 reference and across ranks."""
from __future__ import annotations

import torch
import torch.distributed as dist


def compare(out: torch.Tensor, ref: torch.Tensor) -> dict[str, float]:
    o, r = out.float(), ref.float()
    diff = (o - r).abs()
    denom = r.abs().clamp_min(1e-6)
    cos = torch.nn.functional.cosine_similarity(o.flatten(), r.flatten(), dim=0).item()
    return {
        "max_abs": diff.max().item(),
        "max_rel": (diff / denom).max().item(),
        "mean_rel": (diff.sum() / r.abs().sum()).item(),
        "cos": cos,
        "ref_max_abs": r.abs().max().item(),
    }


def passes(m: dict[str, float], mean_rel: float = 2e-3, max_abs_frac: float = 2e-2, cos: float = 0.9999) -> bool:
    return m["mean_rel"] <= mean_rel and m["max_abs"] <= max_abs_frac * m["ref_max_abs"] and m["cos"] >= cos


def cross_rank_max_abs(out: torch.Tensor, world: int) -> float:
    """All ranks must hold identical outputs after the final all-reduce."""
    if world == 1:
        return 0.0
    gathered = [torch.empty_like(out) for _ in range(world)]
    dist.all_gather(gathered, out.contiguous())
    base = gathered[0].float()
    return max((g.float() - base).abs().max().item() for g in gathered[1:])
