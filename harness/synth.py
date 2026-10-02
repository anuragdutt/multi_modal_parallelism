"""Architecture-aware parallelism synthesizer (the "recipe").

Given a model's layer structure, enumerate rank-group assignments for the parallel branches of each phase,
score them with the byte model (measured bandwidth curves and collective costs), and print the candidates.

  python -m harness.synth --model configs/falcon_h1_7b.yaml --template falcon_h1 --world 4 \
      --bw results/raw/microbench.csv --cells 1:512 16:512 16:8192 64:8192 64:32768

Templates
  falcon_h1          : phase 1 = attention || mamba2 (summed), phase 2 = MLP (after the mixer all-reduce)
  parallel_residual  : phase 1 = attention || mlp (summed into the residual), no second phase (Falcon-40B)
  sequential_hybrid  : one phase per layer type; per-type choice of TP, data-parallel (sequence split) or a
                       smaller group; expert layers by expert parallelism (Nemotron 3 Nano, Qwen3.5)

Rules encoded
  R1 replication: an operator's group never exceeds its unit count (kv heads, ssm heads/groups, experts).
  R2 summed branches share one all-reduce; internal reductions (gated norm) run on the branch group only.
  R3 phase balance: the step is the sum over phases of the slowest group's time, plus collectives.
"""
from __future__ import annotations

import argparse
import itertools
import math
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from .model import BF16, LAUNCH_US, BwTables, Curve

FLAT_GBPS = 450.0  # fallback bandwidth when no microbench is given
AR_US = {(4, "graph"): 40.0, (2, "graph"): 25.0, (4, "eager"): 80.0, (2, "eager"): 60.0}


@dataclass
class Op:
    """One operator of a branch: weight bytes (fixed), per-sequence state bytes, per-token cache bytes."""
    name: str
    cls: str  # gemv | attn | ssu | conv | misc
    weight: int = 0
    state_per_seq: int = 0  # read+write per step per sequence (recurrent state, conv state)
    cache_per_tok: int = 0  # read per step per sequence per token of context (KV)
    units: int = 10**9  # replication-free parallel units for this op

    def bytes(self, B: int, c: int, g: int) -> int:
        """Bytes on one rank of a group of size g (op sharded g ways; replicated beyond `units`)."""
        eff = min(g, self.units)
        return self.weight // g + (self.state_per_seq * B) // g + (self.cache_per_tok * B * c) // eff


@dataclass
class Branch:
    name: str
    ops: list[Op]
    cap: int  # max group size without replication (R1)
    divisors: list[int] = field(default_factory=list)  # allowed group sizes (head divisibility)
    internal_reduction: bool = False  # e.g. gated RMSNorm across the branch group

    def bytes(self, B: int, c: int, g: int) -> dict[str, int]:
        return {op.name: op.bytes(B, c, g) for op in self.ops}


@dataclass
class Phase:
    name: str
    branches: list[Branch]
    summed: bool  # outputs of the branches are summed -> one all-reduce for the phase


@dataclass
class LayerSpec:
    model: str
    hidden: int
    phases: list[Phase]
    layers: int = 1


# ----------------------------------------------------------------------------- templates
def falcon_h1_spec(d: dict) -> LayerSpec:
    H, hd = d["hidden"], d["head_dim"]
    nq, nkv, nh, p, N, G = d["n_q"], d["n_kv"], d["n_mamba_heads"], d["mamba_head_dim"], d["d_state"], d["n_groups"]
    d_ssm = d["d_ssm"]
    attn = Branch("attention", [
        Op("attn.qkv", "gemv", weight=(nq + 2 * nkv) * hd * H * BF16, units=nkv),
        Op("attn.fa", "attn", cache_per_tok=2 * nkv * hd * BF16, units=nkv),
        Op("attn.oproj", "gemv", weight=H * nq * hd * BF16),
    ], cap=max(nkv, 1), divisors=[g for g in range(1, 65) if nq % g == 0])
    in_rows = 2 * d_ssm + 2 * G * N + nh
    conv_rows = d_ssm + 2 * G * N
    mamba = Branch("mamba2", [
        Op("ssm.inproj", "gemv", weight=in_rows * H * BF16),
        Op("ssm.conv", "conv", state_per_seq=2 * (d["d_conv"] - 1) * conv_rows * BF16),
        Op("ssm.ssu", "ssu", state_per_seq=2 * nh * p * N * BF16),
        Op("ssm.outproj", "gemv", weight=H * d_ssm * BF16),
    ], cap=nh, divisors=[g for g in range(1, 65) if nh % g == 0 and (G % g == 0 or g % G == 0)],
        internal_reduction=(d.get("mamba_rms_norm", True)))
    mlp = Branch("mlp", [Op("mlp", "gemv", weight=3 * d["intermediate"] * H * BF16)], cap=10**9)
    return LayerSpec(d["name"], H, [Phase("mixer", [attn, mamba], summed=True), Phase("mlp", [mlp], summed=True)], d.get("layers", 1))


def parallel_residual_spec(d: dict) -> LayerSpec:
    H, hd, nq, nkv = d["hidden"], d["head_dim"], d["n_q"], d["n_kv"]
    attn = Branch("attention", [
        Op("attn.qkv", "gemv", weight=(nq + 2 * nkv) * hd * H * BF16, units=nkv),
        Op("attn.fa", "attn", cache_per_tok=2 * nkv * hd * BF16, units=nkv),
        Op("attn.oproj", "gemv", weight=H * nq * hd * BF16),
    ], cap=max(nkv, 1), divisors=[g for g in range(1, 65) if nq % g == 0])
    nmat = 3 if d.get("gated_mlp", False) else 2
    mlp = Branch("mlp", [Op("mlp", "gemv", weight=nmat * d["intermediate"] * H * BF16)], cap=10**9)
    return LayerSpec(d["name"], H, [Phase("block", [attn, mlp], summed=True)], d.get("layers", 1))


# ----------------------------------------------------------------------------- scoring
def op_time_ms(op: Op, nbytes: int, bw: BwTables | None, variant: str) -> float:
    launch = LAUNCH_US[variant] / 1e3
    if bw is None:
        return nbytes / (FLAT_GBPS * 1e6) + launch
    if op.cls == "gemv":
        return bw.gemv.time_ms(nbytes) + launch
    if op.cls == "attn":
        curve = bw.attn.get(16) or next(iter(bw.attn.values()), None)
        return (curve.time_ms(nbytes) if curve else nbytes / (bw.stream * 1e6)) + launch
    if op.cls == "ssu":
        return bw.ssu.time_ms(nbytes) + launch
    if op.cls == "conv":
        return bw.conv.time_ms(nbytes) + launch
    return nbytes / ((bw.stream or FLAT_GBPS) * 1e6) + launch


def ar_time_ms(group: int, msg_bytes: int, bw: BwTables | None, variant: str) -> float:
    if bw is not None:
        curve = bw.ar.get((group, variant)) or bw.ar.get((group, "eager"))
        if curve is not None:
            return curve.time_ms(msg_bytes)
    return AR_US.get((group, variant), 40.0) / 1e3


@dataclass
class Candidate:
    name: str
    groups: dict[str, int]  # branch -> group size (homogeneous: all = world)
    homogeneous: bool

    def describe(self) -> str:
        return "+".join(f"{b}:{g}" for b, g in self.groups.items())


def enumerate_candidates(phase: Phase, world: int) -> list[Candidate]:
    cands = [Candidate("tp", {b.name: world for b in phase.branches}, True)]
    if len(phase.branches) < 2:
        return cands
    sizes = []
    for b in phase.branches:
        allowed = [g for g in range(1, world) if g <= b.cap and (not b.divisors or g in b.divisors)]
        sizes.append(allowed)
    for combo in itertools.product(*sizes):
        if sum(combo) == world:
            cands.append(Candidate("split" + "".join(map(str, combo)), dict(zip([b.name for b in phase.branches], combo)), False))
    return cands


def score(spec: LayerSpec, cand_per_phase: list[Candidate], B: int, c: int, world: int, bw: BwTables | None,
          variant: str, fused_tp: bool = True) -> tuple[float, dict]:
    total, detail = 0.0, {}
    msg = B * spec.hidden * BF16
    for phase, cand in zip(spec.phases, cand_per_phase):
        times = {}
        for b in phase.branches:
            g = cand.groups[b.name]
            t = sum(op_time_ms(op, nb, bw, variant) for op, nb in zip(b.ops, b.bytes(B, c, g).values()))
            if b.internal_reduction and g > 1:
                t += ar_time_ms(g, B * 4, bw, variant)
            times[b.name] = t
        if cand.homogeneous:
            phase_t = sum(times.values())  # every rank runs every branch
            n_ar = 1 if (fused_tp or len(phase.branches) == 1) else len(phase.branches)
        else:
            phase_t = max(times.values())
            n_ar = 1
        coll = n_ar * ar_time_ms(world, msg, bw, variant) if world > 1 else 0.0
        total += phase_t + coll
        detail[phase.name] = {"branch_ms": times, "collectives_ms": coll, "layout": cand.describe()}
    return total, detail


def run(spec: LayerSpec, world: int, bw: BwTables | None, cells: list[tuple[int, int]], variant: str = "graph") -> None:
    print(f"\n=== {spec.model}: world={world}, variant={variant} ===")
    phase_cands = [enumerate_candidates(p, world) for p in spec.phases]
    combos = list(itertools.product(*phase_cands))
    header = "layout".ljust(34) + "".join(f"{f'B{B} c{c}':>14s}" for B, c in cells)
    print(header)
    rows = []
    for combo in combos:
        name = " | ".join(c.describe() for c in combo)
        vals = [score(spec, list(combo), B, c, world, bw, variant)[0] for B, c in cells]
        rows.append((name, vals))
    base = rows[0][1]
    for name, vals in rows:
        print(name.ljust(34) + "".join(f"{v:9.3f}ms{b / v:4.2f}x" for v, b in zip(vals, base)))
    print("(numbers: predicted layer step in ms, then speedup over the homogeneous tp layout; R1 caps already applied)")


def load_dims(path: str) -> dict:
    d = yaml.safe_load(Path(path).read_text())
    d.setdefault("name", Path(path).stem)
    return d


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="dims yaml")
    ap.add_argument("--template", choices=["falcon_h1", "parallel_residual"], required=True)
    ap.add_argument("--world", type=int, default=4)
    ap.add_argument("--bw", default="results/raw/microbench.csv")
    ap.add_argument("--variant", default="graph")
    ap.add_argument("--cells", nargs="*", default=["1:512", "16:512", "16:8192", "64:8192", "64:32768"])
    args = ap.parse_args()
    d = load_dims(args.model)
    spec = falcon_h1_spec(d) if args.template == "falcon_h1" else parallel_residual_spec(d)
    bw = BwTables.from_csv(args.bw) if Path(args.bw).exists() else None
    cells = [tuple(int(x) for x in s.split(":")) for s in args.cells]
    run(spec, args.world, bw, cells, args.variant)


if __name__ == "__main__":
    main()
