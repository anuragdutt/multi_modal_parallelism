"""Exact per-rank index math for every parallel mode. Pure python, no torch dependency except for
`build_mup_vector`, which is imported lazily so the module can be unit-tested on CPU.

Conventions (Falcon-H1 checkpoint layout):
  q_proj [n_q*hd, H], k_proj/v_proj [n_kv*hd, H], o_proj [H, n_q*hd]
  in_proj [2*d_ssm + 2*G*N + nh, H] with rows [z | x | B | C | dt]
  conv1d.weight [d_ssm + 2*G*N, 1, d_conv] with rows [x | B | C]
  out_proj [H, d_ssm]; norm.weight [d_ssm]; A_log/D/dt_bias [nh]
  gate_proj/up_proj [I, H]; down_proj [H, I]
Mamba head j owns z/x columns [j*p, (j+1)*p), dt/A/D/dt_bias[j], norm[j*p:(j+1)*p], out_proj columns [j*p,(j+1)*p).
Query head i uses KV head i // (n_q // n_kv). Mamba head j uses group j // (nh // G).
"""
from __future__ import annotations

from dataclasses import dataclass

from .config import FalconH1Dims

SWING_GRANULE = 64


@dataclass(frozen=True)
class AttnShard:
    q_heads: range
    kv_heads: range  # usually one head; several when n_kv > group size
    kv_replicas: int  # how many ranks of the group hold the same kv head(s)
    q_rows: slice
    k_rows: slice
    v_rows: slice
    o_cols: slice
    group_rank: int
    group_size: int


@dataclass(frozen=True)
class SsmShard:
    heads: range
    groups: range  # Mamba B/C groups held locally (one, possibly replicated, when G < group size)
    z_rows: slice
    x_rows: slice
    b_rows: slice
    c_rows: slice
    dt_rows: slice
    conv_rows: tuple[slice, slice, slice]  # x part, B part, C part of conv1d rows
    out_cols: slice
    norm_cols: slice
    group_rank: int
    group_size: int
    norm_ranks: tuple[int, ...]  # world ranks sharing the gated-norm reduction (incl. self)

    @property
    def local_in_rows(self) -> tuple[int, int, int, int, int]:
        n = len(self.heads)
        return (
            self.z_rows.stop - self.z_rows.start,
            self.x_rows.stop - self.x_rows.start,
            self.b_rows.stop - self.b_rows.start,
            self.c_rows.stop - self.c_rows.start,
            n,
        )


@dataclass(frozen=True)
class MlpShard:
    base_cols: slice  # the rank's 1/world slice of the intermediate dim
    fixed_cols: slice  # base minus its swing piece
    w: int  # swing piece width (columns); 0 -> no swing
    swing_pieces: tuple[slice, ...]  # global column ranges W_0..W_{world-1}
    cuts: tuple[int, ...]  # (c_0=0 <= ... <= c_world = world*w) over the concatenated swing space
    group_rank: int
    group_size: int

    @property
    def my_swing(self) -> tuple[int, int]:
        return self.cuts[self.group_rank], self.cuts[self.group_rank + 1]

    def swing_global_cols(self, start: int, stop: int) -> list[slice]:
        """Map a [start, stop) range of the concatenated swing space to global intermediate columns."""
        out: list[slice] = []
        w = self.w
        t = start
        while t < stop:
            piece, off = divmod(t, w)
            n = min(stop - t, w - off)
            g = self.swing_pieces[piece]
            out.append(slice(g.start + off, g.start + off + n))
            t += n
        return out

    @property
    def active_cols(self) -> int:
        a, b = self.my_swing
        return (self.fixed_cols.stop - self.fixed_cols.start) + (b - a)


@dataclass(frozen=True)
class RankPlan:
    mode: str
    rank: int
    world: int
    attn: AttnShard | None
    ssm: SsmShard | None
    mlp: MlpShard
    attn_ranks: tuple[int, ...]
    ssm_ranks: tuple[int, ...]
    combine: str  # allreduce_each | sum_then_allreduce | none

    @property
    def has_attn(self) -> bool:
        return self.attn is not None

    @property
    def has_ssm(self) -> bool:
        return self.ssm is not None


def _check(cond: bool, msg: str) -> None:
    if not cond:
        raise ValueError(msg)


def attn_shard(dims: FalconH1Dims, group_rank: int, group_size: int) -> AttnShard:
    hd = dims.head_dim
    _check(dims.n_q % group_size == 0, f"attention heads {dims.n_q} not divisible by group {group_size}")
    nqr = dims.n_q // group_size
    q_heads = range(group_rank * nqr, (group_rank + 1) * nqr)
    q_per_kv = dims.n_q // dims.n_kv
    if dims.n_kv >= group_size:
        _check(dims.n_kv % group_size == 0, "kv heads not divisible by group size")
        nkvr = dims.n_kv // group_size
        kv_heads = range(group_rank * nkvr, (group_rank + 1) * nkvr)
        replicas = 1
    else:
        _check(group_size % dims.n_kv == 0, "group size not a multiple of kv heads")
        replicas = group_size // dims.n_kv
        kv_heads = range(group_rank // replicas, group_rank // replicas + 1)
    # consistency with GQA: every q head of this rank must map into kv_heads
    for i in q_heads:
        _check(i // q_per_kv in kv_heads, f"q head {i} maps to kv head {i // q_per_kv}, not in {list(kv_heads)}")
    return AttnShard(
        q_heads=q_heads,
        kv_heads=kv_heads,
        kv_replicas=replicas,
        q_rows=slice(q_heads.start * hd, q_heads.stop * hd),
        k_rows=slice(kv_heads.start * hd, kv_heads.stop * hd),
        v_rows=slice(kv_heads.start * hd, kv_heads.stop * hd),
        o_cols=slice(q_heads.start * hd, q_heads.stop * hd),
        group_rank=group_rank,
        group_size=group_size,
    )


def ssm_shard(dims: FalconH1Dims, group_rank: int, group_size: int, group_world_ranks: tuple[int, ...]) -> SsmShard:
    p, N, G, nh = dims.mamba_head_dim, dims.d_state, dims.n_groups, dims.n_mamba_heads
    _check(nh % group_size == 0, f"mamba heads {nh} not divisible by group {group_size}")
    nhr = nh // group_size
    heads = range(group_rank * nhr, (group_rank + 1) * nhr)
    heads_per_group = nh // G
    if G >= group_size:
        _check(G % group_size == 0, "n_groups not divisible by group size")
        ngr = G // group_size
        groups = range(group_rank * ngr, (group_rank + 1) * ngr)
        norm_ranks = (group_world_ranks[group_rank],)
    else:
        _check(group_size % G == 0, "group size not a multiple of n_groups")
        per_group_ranks = group_size // G
        g = group_rank // per_group_ranks
        groups = range(g, g + 1)
        first = g * per_group_ranks
        norm_ranks = tuple(group_world_ranks[first : first + per_group_ranks])
    for j in heads:
        _check(j // heads_per_group in groups, f"head {j} belongs to group {j // heads_per_group}, not {list(groups)}")
    z0, x0 = heads.start * p, dims.d_ssm + heads.start * p
    b0 = 2 * dims.d_ssm + groups.start * N
    c0 = 2 * dims.d_ssm + G * N + groups.start * N
    dt0 = 2 * dims.d_ssm + 2 * G * N + heads.start
    ng = len(groups)
    return SsmShard(
        heads=heads,
        groups=groups,
        z_rows=slice(z0, z0 + nhr * p),
        x_rows=slice(x0, x0 + nhr * p),
        b_rows=slice(b0, b0 + ng * N),
        c_rows=slice(c0, c0 + ng * N),
        dt_rows=slice(dt0, dt0 + nhr),
        conv_rows=(
            slice(heads.start * p, heads.stop * p),
            slice(dims.d_ssm + groups.start * N, dims.d_ssm + groups.stop * N),
            slice(dims.d_ssm + G * N + groups.start * N, dims.d_ssm + G * N + groups.stop * N),
        ),
        out_cols=slice(heads.start * p, heads.stop * p),
        norm_cols=slice(heads.start * p, heads.stop * p),
        group_rank=group_rank,
        group_size=group_size,
        norm_ranks=norm_ranks,
    )


def swing_width(S: float, cols_per_rank: int, granule: int = SWING_GRANULE) -> int:
    if S <= 0:
        return 0
    w = granule * round(S * cols_per_rank / granule)
    return max(granule, min(w, cols_per_rank))


def neutral_cuts(w: int, world: int) -> tuple[int, ...]:
    return tuple(w * r for r in range(world + 1))


def symmetric_cuts(w: int, n_attn: int, world: int = 4) -> tuple[int, ...]:
    """Attention ranks (0,1) each take n_attn swing columns, SSM ranks (2,3) take the rest, evenly."""
    _check(world == 4, "symmetric_cuts assumes 2+2")
    _check(0 <= n_attn <= 2 * w, f"n_attn {n_attn} outside [0, {2 * w}]")
    rest = 4 * w - 2 * n_attn
    return (0, n_attn, 2 * n_attn, 2 * n_attn + rest // 2, 4 * w)


def mlp_shard(dims: FalconH1Dims, rank: int, world: int, swing: float, cuts: tuple[int, ...] | None) -> MlpShard:
    _check(dims.intermediate % world == 0, "intermediate not divisible by world")
    cpr = dims.intermediate // world
    base = slice(rank * cpr, (rank + 1) * cpr)
    w = swing_width(swing, cpr) if world > 1 else 0
    pieces = tuple(slice((r + 1) * cpr - w, (r + 1) * cpr) for r in range(world))
    if cuts is None:
        cuts = neutral_cuts(w, world)
    _check(len(cuts) == world + 1 and cuts[0] == 0 and cuts[-1] == world * w, f"bad cuts {cuts} for w={w}")
    _check(all(cuts[i] <= cuts[i + 1] for i in range(world)), f"cuts not monotone: {cuts}")
    _check(all(c % SWING_GRANULE == 0 for c in cuts) or w == 0, f"cuts must be multiples of {SWING_GRANULE}")
    return MlpShard(
        base_cols=base,
        fixed_cols=slice(base.start, base.stop - w),
        w=w,
        swing_pieces=pieces,
        cuts=tuple(cuts),
        group_rank=rank,
        group_size=world,
    )


def plan_for(
    mode: str,
    rank: int,
    world: int,
    dims: FalconH1Dims,
    swing: float = 0.0,
    cuts: tuple[int, ...] | None = None,
) -> RankPlan:
    all_ranks = tuple(range(world))
    if mode == "tp1":
        _check(world == 1, "tp1 needs world size 1")
        return RankPlan(
            mode=mode, rank=0, world=1,
            attn=attn_shard(dims, 0, 1),
            ssm=ssm_shard(dims, 0, 1, (0,)),
            mlp=mlp_shard(dims, 0, 1, 0.0, None),
            attn_ranks=(0,), ssm_ranks=(0,), combine="none",
        )
    if mode in ("tp4", "tp4_fused", "tp4_streams"):
        _check(swing == 0.0, "swing only applies to split modes")
        return RankPlan(
            mode=mode, rank=rank, world=world,
            attn=attn_shard(dims, rank, world),
            ssm=ssm_shard(dims, rank, world, all_ranks),
            mlp=mlp_shard(dims, rank, world, 0.0, None),
            attn_ranks=all_ranks, ssm_ranks=all_ranks,
            combine="allreduce_each" if mode == "tp4" else "sum_then_allreduce",
        )
    if mode == "split22":
        _check(world == 4, "split22 needs world size 4")
        attn_ranks, ssm_ranks = (0, 1), (2, 3)
        is_attn = rank in attn_ranks
        return RankPlan(
            mode=mode, rank=rank, world=world,
            attn=attn_shard(dims, rank, 2) if is_attn else None,
            ssm=None if is_attn else ssm_shard(dims, rank - 2, 2, ssm_ranks),
            mlp=mlp_shard(dims, rank, world, swing, cuts),
            attn_ranks=attn_ranks, ssm_ranks=ssm_ranks, combine="sum_then_allreduce",
        )
    raise ValueError(f"unknown mode {mode}")


def mup_blocks(dims: FalconH1Dims, shard: SsmShard) -> list[tuple[int, float]]:
    """(length, multiplier) per block of the local in_proj output [z | x | B | C | dt]."""
    z, x, b, c, dt = shard.local_in_rows
    m = dims.ssm_mults
    return [(z, m[0]), (x, m[1]), (b, m[2]), (c, m[3]), (dt, m[4])]


def build_mup_vector(dims: FalconH1Dims, shard: SsmShard):
    import torch

    blocks = mup_blocks(dims, shard)
    vec = torch.ones(1, sum(n for n, _ in blocks), dtype=torch.float32)
    off = 0
    for n, mult in blocks:
        vec[:, off : off + n] *= mult
        off += n
    return vec


def kv_replication_map(dims: FalconH1Dims, group_size: int) -> dict[int, tuple[int, ...]]:
    """kv head -> group ranks holding it."""
    out: dict[int, list[int]] = {}
    for r in range(group_size):
        for h in attn_shard(dims, r, group_size).kv_heads:
            out.setdefault(h, []).append(r)
    return {h: tuple(v) for h, v in out.items()}
