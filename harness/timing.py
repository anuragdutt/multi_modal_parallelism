"""CUDA-event op timer, eager and CUDA-graph step timing."""
from __future__ import annotations

import statistics
import time

import torch

from .dist import Groups, barrier_sync, capture_context


class OpTimer:
    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self._open: dict[str, torch.cuda.Event] = {}
        self.pairs: dict[str, list[tuple[torch.cuda.Event, torch.cuda.Event]]] = {}

    def begin(self, name: str) -> None:
        if not self.enabled:
            return
        e = torch.cuda.Event(enable_timing=True)
        e.record(torch.cuda.current_stream())
        self._open[name] = e

    def end(self, name: str) -> None:
        if not self.enabled:
            return
        e = torch.cuda.Event(enable_timing=True)
        e.record(torch.cuda.current_stream())
        self.pairs.setdefault(name, []).append((self._open.pop(name), e))

    def reset(self) -> None:
        self._open.clear()
        self.pairs.clear()

    def summarize(self) -> dict[str, dict[str, float]]:
        torch.cuda.synchronize()
        out = {}
        for name, pairs in self.pairs.items():
            ms = [s.elapsed_time(e) for s, e in pairs]
            out[name] = _stats(ms)
        return out


NULL = OpTimer(enabled=False)


def _stats(ms: list[float]) -> dict[str, float]:
    ms_sorted = sorted(ms)
    n = len(ms_sorted)
    return {
        "n": n,
        "median_ms": statistics.median(ms_sorted),
        "p10_ms": ms_sorted[max(0, int(0.1 * n) - 1)] if n > 1 else ms_sorted[0],
        "p90_ms": ms_sorted[min(n - 1, int(0.9 * n))],
        "mean_ms": statistics.fmean(ms_sorted),
    }


def time_eager(layer, h: torch.Tensor, n_warm: int, n_iter: int, groups: Groups) -> tuple[dict, float]:
    """Returns (op stats incl. 'step', wall ms per step)."""
    for _ in range(n_warm):
        layer.step(h, NULL)
    barrier_sync()
    t = OpTimer(enabled=True)
    t0 = time.perf_counter()
    for _ in range(n_iter):
        t.begin("step")
        layer.step(h, t)
        t.end("step")
    torch.cuda.synchronize()
    wall = (time.perf_counter() - t0) * 1e3 / n_iter
    return t.summarize(), wall


def time_graph(layer, h: torch.Tensor, n_warm: int, n_iter: int, groups: Groups) -> tuple[dict, float]:
    """Capture one full step (collectives included) and replay it."""
    with capture_context(groups) as ctx:
        stream = ctx.stream
        with torch.cuda.stream(stream):
            for _ in range(2):
                layer.step(h, NULL)
        stream.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, stream=stream):
            layer.step(h, NULL)
    torch.cuda.synchronize()
    for _ in range(n_warm):
        g.replay()
    barrier_sync()
    starts, ends = [], []
    t0 = time.perf_counter()
    for _ in range(n_iter):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        g.replay()
        e.record()
        starts.append(s)
        ends.append(e)
    torch.cuda.synchronize()
    wall = (time.perf_counter() - t0) * 1e3 / n_iter
    ms = [s.elapsed_time(e) for s, e in zip(starts, ends)]
    return {"step": _stats(ms)}, wall
