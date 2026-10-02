"""Parallel-residual Transformer template (Falcon-40B style):  h + attention(ln_attn(h)) + mlp(ln_mlp(h)).

The attention and the MLP are summed branches of one phase, so the recipe's candidates are rank-group splits of
the two branches with a single all-reduce of partial sums (vLLM's own Falcon path already fuses the reduction).
Modes: tp4 (homogeneous), tp4_streams (branches on two CUDA streams per rank), split13 / split22 (attention on
the first a ranks, MLP columns on the remaining world - a ranks).
"""
from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F

from . import csvio
from . import kernels as K
from .config import RunSpec
from .correctness import compare, cross_rank_max_abs, passes
from .dist import Groups, barrier_sync, describe_comm
from .layer import AttentionBranch
from .measure import gpu_env, vllm_commit
from .reference import apply_rope_neox, rope_cos_sin
from .sharding import AttnShard, _even_split, attn_shard
from .state import DecodeCtx, RankStates, alloc_and_fill, full_kv_head, input_hidden, make_decode_ctx
from .timing import OpTimer, time_eager, time_graph

PR_KEYS = (
    "ln_attn.weight", "ln_attn.bias", "ln_mlp.weight", "ln_mlp.bias",
    "self_attention.query_key_value.weight", "self_attention.dense.weight",
    "mlp.dense_h_to_4h.weight", "mlp.dense_4h_to_h.weight",
)


@dataclass(frozen=True)
class PRDims:
    hidden: int
    intermediate: int
    n_q: int
    n_kv: int
    head_dim: int
    ln_eps: float
    rope_theta: float
    max_pos: int
    key_mult: float = 1.0
    attn_out: float = 1.0
    attn_in: float = 1.0
    name: str = "falcon_40b"

    @property
    def q_per_kv(self) -> int:
        return self.n_q // self.n_kv

    @classmethod
    def from_hf(cls, model_dir: str | Path) -> "PRDims":
        cfg = json.loads((Path(model_dir) / "config.json").read_text())
        hidden = cfg["hidden_size"]
        n_q = cfg["num_attention_heads"]
        n_kv = cfg.get("num_kv_heads") or cfg.get("n_head_kv") or 1
        assert cfg.get("parallel_attn", False) and cfg.get("new_decoder_architecture", False), "template expects the new parallel Falcon decoder"
        return cls(hidden=hidden, intermediate=cfg.get("ffn_hidden_size") or 4 * hidden, n_q=n_q, n_kv=n_kv,
                   head_dim=hidden // n_q, ln_eps=cfg.get("layer_norm_epsilon", 1e-5),
                   rope_theta=cfg.get("rope_theta") or 10000.0, max_pos=cfg.get("max_position_embeddings", 2048),
                   name=Path(model_dir).parts[-3].replace("models--", "") if len(Path(model_dir).parts) > 3 else "pr")


def load_layer_pr(model_dir: str | Path, layer_idx: int) -> dict[str, torch.Tensor]:
    from safetensors import safe_open

    model_dir = Path(model_dir)
    index = json.loads((model_dir / "model.safetensors.index.json").read_text())["weight_map"]
    prefix = f"transformer.h.{layer_idx}."
    by_file: dict[str, list[str]] = {}
    for k in PR_KEYS:
        by_file.setdefault(index[prefix + k], []).append(prefix + k)
    out: dict[str, torch.Tensor] = {}
    for fname, names in by_file.items():
        with safe_open(str(model_dir / fname), framework="pt", device="cpu") as f:
            for n in names:
                out[n[len(prefix):]] = f.get_tensor(n)
    return out


def random_layer_pr(dims: PRDims, seed: int = 0) -> dict[str, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    H, I, hd = dims.hidden, dims.intermediate, dims.head_dim

    def lin(r: int, c: int) -> torch.Tensor:
        return (torch.randn(r, c, generator=g) * 0.02).to(torch.bfloat16)

    return {
        "ln_attn.weight": torch.ones(H, dtype=torch.bfloat16), "ln_attn.bias": torch.zeros(H, dtype=torch.bfloat16),
        "ln_mlp.weight": torch.ones(H, dtype=torch.bfloat16), "ln_mlp.bias": torch.zeros(H, dtype=torch.bfloat16),
        "self_attention.query_key_value.weight": lin((dims.n_q + 2 * dims.n_kv) * hd, H),
        "self_attention.dense.weight": lin(H, dims.n_q * hd),
        "mlp.dense_h_to_4h.weight": lin(I, H), "mlp.dense_4h_to_h.weight": lin(H, I),
    }


# ----------------------------------------------------------------------------- sharding
@dataclass(frozen=True)
class PRRankPlan:
    mode: str
    rank: int
    world: int
    attn: AttnShard | None
    mlp_cols: slice | None
    attn_ranks: tuple[int, ...]
    mlp_ranks: tuple[int, ...]
    ssm = None  # so the shared state allocator skips SSM buffers

    @property
    def has_attn(self) -> bool:
        return self.attn is not None

    @property
    def has_mlp(self) -> bool:
        return self.mlp_cols is not None


def plan_for_pr(mode: str, rank: int, world: int, dims: PRDims) -> PRRankPlan:
    if mode in ("tp4", "tp4_streams"):
        cols = _even_split(dims.intermediate, world)[rank]
        return PRRankPlan(mode, rank, world, attn_shard(dims, rank, world), slice(cols.start, cols.stop),
                          tuple(range(world)), tuple(range(world)))
    if mode.startswith("split"):
        a = int(mode[5])
        m = int(mode[6])
        assert a + m == world, f"{mode} does not match world {world}"
        attn_ranks, mlp_ranks = tuple(range(a)), tuple(range(a, world))
        if rank < a:
            return PRRankPlan(mode, rank, world, attn_shard(dims, rank, a), None, attn_ranks, mlp_ranks)
        cols = _even_split(dims.intermediate, m)[rank - a]
        return PRRankPlan(mode, rank, world, None, slice(cols.start, cols.stop), attn_ranks, mlp_ranks)
    raise ValueError(mode)


@dataclass
class PRRankWeights:
    wqkv: torch.Tensor | None = None
    wo: torch.Tensor | None = None
    q_rows: int = 0
    kv_rows: int = 0
    w_up: torch.Tensor | None = None  # [n, H]
    w_downT: torch.Tensor | None = None  # [n, H]
    ln_attn_w: torch.Tensor | None = None
    ln_attn_b: torch.Tensor | None = None
    ln_mlp_w: torch.Tensor | None = None
    ln_mlp_b: torch.Tensor | None = None


def qkv_row_index(dims: PRDims, shard: AttnShard) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Row indices into the fused [n_kv * (q_per_kv + 2) * hd, H] QKV weight for this shard's q, k, v."""
    hd, qpk = dims.head_dim, dims.q_per_kv
    block = (qpk + 2) * hd
    q_idx, k_idx, v_idx = [], [], []
    for i in shard.q_heads:
        g, j = divmod(i, qpk)
        q_idx.extend(range(g * block + j * hd, g * block + (j + 1) * hd))
    for h in shard.kv_heads:
        k_idx.extend(range(h * block + qpk * hd, h * block + (qpk + 1) * hd))
        v_idx.extend(range(h * block + (qpk + 1) * hd, h * block + (qpk + 2) * hd))
    return torch.tensor(q_idx), torch.tensor(k_idx), torch.tensor(v_idx)


def shard_pr(full: dict[str, torch.Tensor], plan: PRRankPlan, dims: PRDims, device: torch.device) -> PRRankWeights:
    bf = torch.bfloat16
    rw = PRRankWeights()
    if plan.attn is not None:
        q_idx, k_idx, v_idx = qkv_row_index(dims, plan.attn)
        qkv = full["self_attention.query_key_value.weight"]
        rw.wqkv = torch.cat([qkv[q_idx], qkv[k_idx], qkv[v_idx]], 0).to(device, bf).contiguous()
        rw.wo = full["self_attention.dense.weight"][:, plan.attn.o_cols].to(device, bf).contiguous()
        rw.q_rows, rw.kv_rows = len(q_idx), len(k_idx)
        rw.ln_attn_w = full["ln_attn.weight"].to(device, bf)
        rw.ln_attn_b = full["ln_attn.bias"].to(device, bf)
    if plan.mlp_cols is not None:
        c = plan.mlp_cols
        rw.w_up = full["mlp.dense_h_to_4h.weight"][c].to(device, bf).contiguous()
        rw.w_downT = full["mlp.dense_4h_to_h.weight"][:, c].T.to(device, bf).contiguous()
        rw.ln_mlp_w = full["ln_mlp.weight"].to(device, bf)
        rw.ln_mlp_b = full["ln_mlp.bias"].to(device, bf)
    return rw


# ----------------------------------------------------------------------------- layer
class GeluMLPBranch:
    name = "mlp"

    def __init__(self, w: PRRankWeights):
        self.w = w

    def forward(self, y: torch.Tensor, t: OpTimer) -> torch.Tensor:
        t.begin("mlp.up")
        u = K.gemm(y, self.w.w_up)
        t.end("mlp.up")
        t.begin("mlp.act")
        a = F.gelu(u)
        t.end("mlp.act")
        t.begin("mlp.down")
        part = K.gemm_T(a, self.w.w_downT)
        t.end("mlp.down")
        return part


class PRLayer:
    def __init__(self, plan: PRRankPlan, w: PRRankWeights, dims: PRDims, groups: Groups, st: RankStates,
                 ctx: DecodeCtx, device: torch.device, collectives: bool = True):
        self.plan, self.w, self.dims, self.groups, self.ctx, self.device = plan, w, dims, groups, ctx, device
        self.collectives = collectives
        self.attn = AttentionBranch(plan, w, dims, st, ctx, device) if plan.attn is not None else None
        self.mlp = GeluMLPBranch(w) if plan.mlp_cols is not None else None
        self.out = torch.empty(ctx.B, dims.hidden, device=device, dtype=torch.bfloat16)
        self.mode = plan.mode
        if self.mode == "tp4_streams":
            self.s_attn = torch.cuda.Stream(device=device)
            self.s_mlp = torch.cuda.Stream(device=device)

    def _ar(self, x: torch.Tensor, t: OpTimer) -> torch.Tensor:
        if self.groups.world_size == 1 or not self.collectives:
            return x
        t.begin("ar.mixer")
        y = self.groups.tp.all_reduce(x)
        t.end("ar.mixer")
        return y

    def _attn(self, h: torch.Tensor, t: OpTimer) -> torch.Tensor:
        t.begin("ln_attn")
        x = F.layer_norm(h, (self.dims.hidden,), self.w.ln_attn_w, self.w.ln_attn_b, self.dims.ln_eps)
        t.end("ln_attn")
        return self.attn.decode(x, t)

    def _mlp(self, h: torch.Tensor, t: OpTimer) -> torch.Tensor:
        t.begin("ln_mlp")
        x = F.layer_norm(h, (self.dims.hidden,), self.w.ln_mlp_w, self.w.ln_mlp_b, self.dims.ln_eps)
        t.end("ln_mlp")
        return self.mlp.forward(x, t)

    def step(self, h: torch.Tensor, t: OpTimer) -> torch.Tensor:
        mode = self.mode
        if mode == "tp4":
            part = self._attn(h, t) + self._mlp(h, t)
        elif mode == "tp4_streams":
            cur = torch.cuda.current_stream(self.device)
            self.s_attn.wait_stream(cur)
            self.s_mlp.wait_stream(cur)
            with torch.cuda.stream(self.s_attn):
                oa = self._attn(h, t)
            with torch.cuda.stream(self.s_mlp):
                om = self._mlp(h, t)
            cur.wait_stream(self.s_attn)
            cur.wait_stream(self.s_mlp)
            part = oa + om
        else:
            part = self._attn(h, t) if self.attn is not None else self._mlp(h, t)
        mix = self._ar(part, t)
        t.begin("residual")
        self.out.copy_(h + mix)
        t.end("residual")
        return self.out


# ----------------------------------------------------------------------------- reference
@torch.no_grad()
def reference_pr(full: dict[str, torch.Tensor], dims: PRDims, h: torch.Tensor, k_cache: torch.Tensor,
                 v_cache: torch.Tensor, pos: int) -> torch.Tensor:
    dev = h.device
    f32 = lambda t: full[t].to(dev, torch.float32)  # noqa: E731
    B, H, hd, nq, nkv, qpk = h.shape[0], dims.hidden, dims.head_dim, dims.n_q, dims.n_kv, dims.q_per_kv
    h = h.float()
    xa = F.layer_norm(h, (H,), f32("ln_attn.weight"), f32("ln_attn.bias"), dims.ln_eps)
    xm = F.layer_norm(h, (H,), f32("ln_mlp.weight"), f32("ln_mlp.bias"), dims.ln_eps)
    qkv = (xa @ f32("self_attention.query_key_value.weight").T).view(B, nkv, qpk + 2, hd)
    q = qkv[:, :, :qpk].reshape(B, nq, hd)
    k = qkv[:, :, qpk]  # [B, nkv, hd]
    v = qkv[:, :, qpk + 1]
    positions = torch.full((B,), pos, device=dev, dtype=torch.int64)
    cos, sin = rope_cos_sin(positions, hd, dims.rope_theta)
    q, k = apply_rope_neox(q, cos, sin), apply_rope_neox(k, cos, sin)
    Kc = torch.cat([k_cache.float(), k[:, None]], 1)
    Vc = torch.cat([v_cache.float(), v[:, None]], 1)
    Kq = Kc.repeat_interleave(qpk, dim=2)
    Vq = Vc.repeat_interleave(qpk, dim=2)
    scores = torch.einsum("bqd,bkqd->bqk", q, Kq) / (hd ** 0.5)
    o = torch.einsum("bqk,bkqd->bqd", torch.softmax(scores, -1), Vq).reshape(B, nq * hd)
    attn = o @ f32("self_attention.dense.weight").T
    mlp = F.gelu(xm @ f32("mlp.dense_h_to_4h.weight").T) @ f32("mlp.dense_4h_to_h.weight").T
    return h + attn + mlp


def full_kv_for_reference(dims: PRDims, B: int, c: int, seed: int, device: torch.device):
    ks, vs = [], []
    for hh in range(dims.n_kv):
        k, v = full_kv_head(seed, hh, B, c, dims.head_dim, device)
        ks.append(k)
        vs.append(v)
    return torch.stack(ks, 2), torch.stack(vs, 2)


# ----------------------------------------------------------------------------- measurement
def measure_pr(spec: RunSpec, groups: Groups, dims: PRDims, full: dict[str, torch.Tensor], out_csv: str,
               correctness_csv: str, check: bool, image_tag: str = "") -> list[dict]:
    rank, world, device = groups.rank, groups.world_size, groups.device
    plan = plan_for_pr(spec.mode, rank, world, dims)
    rw = shard_pr(full, plan, dims, device)
    ctx = make_decode_ctx(spec.batch, spec.ctx, spec.kv_block, device)
    st = alloc_and_fill(plan, dims, ctx, spec.seed, device)
    h = input_hidden(spec.seed, spec.batch, dims.hidden, device)
    layer = PRLayer(plan, rw, dims, groups, st, ctx, device, collectives=spec.collectives)
    barrier_sync()
    run_id, ts = uuid.uuid4().hex[:8], time.strftime("%Y-%m-%dT%H:%M:%S")
    base = dict(run_id=run_id, ts=ts, image_tag=image_tag, vllm_commit=vllm_commit(), mode=spec.mode, S=0.0, w=0,
                cuts="", cuts_src="", batch=spec.batch, ctx=spec.ctx, kv_block=spec.kv_block,
                comm_tp=json.dumps(describe_comm(groups.tp)) if spec.collectives else "none",
                comm_pair=json.dumps(describe_comm(groups.pair)), tag=spec.tag)
    if check and spec.collectives:
        out = layer.step(h, OpTimer(enabled=False)).clone()
        torch.cuda.synchronize()
        xr = cross_rank_max_abs(out, world)
        if rank == 0:
            kc, vc = full_kv_for_reference(dims, spec.batch, spec.ctx, spec.seed, device)
            ref = reference_pr(full, dims, h, kc, vc, spec.ctx)
            m = compare(out, ref)
            ok = passes(m) and xr == 0.0
            row = dict(base, ref="fp32", cross_rank_max_abs=xr, **{k: m[k] for k in ("max_abs", "max_rel", "mean_rel", "cos")})
            row["pass"] = ok
            csvio.append_rows(correctness_csv, csvio.CORRECTNESS_COLUMNS, [row])
            print(f"[check] mode={spec.mode} B={spec.batch} c={spec.ctx} max_abs={m['max_abs']:.4g} (ref max {m['ref_max_abs']:.3g}) "
                  f"mean_rel={m['mean_rel']:.3e} cos={m['cos']:.6f} cross_rank={xr:.3g} pass={ok}", flush=True)
            del kc, vc, ref
        st2 = alloc_and_fill(plan, dims, ctx, spec.seed, device)
        if st.kv is not None:
            st.kv.copy_(st2.kv)
        del st2
        barrier_sync()
    variants = ["eager", "graph"] if spec.variant == "both" else [spec.variant]
    rows: list[dict] = []
    for rep in range(spec.repeats):
        for variant in variants:
            env = gpu_env(groups.local_rank)
            if variant == "eager":
                stats, wall = time_eager(layer, h, spec.n_warm, spec.n_iter, groups)
            else:
                stats, wall = time_graph(layer, h, spec.n_warm, spec.n_iter, groups)
            rep_id = rep + spec.repeat_offset
            for op, s in stats.items():
                rows.append(dict(base, variant=variant, repeat=rep_id, rank=rank, op=op, **s, **env))
            rows.append(dict(base, variant=variant, repeat=rep_id, rank=rank, op="step_wall", n=spec.n_iter,
                             median_ms=wall, p10_ms=wall, p90_ms=wall, mean_ms=wall, **env))
            barrier_sync()
    gathered: list = [None] * world
    if world > 1:
        dist.all_gather_object(gathered, rows)
    else:
        gathered = [rows]
    all_rows: list[dict] = []
    if rank == 0:
        all_rows = [r for rr in gathered for r in rr]
        for variant in variants:
            for rep in range(spec.repeat_offset, spec.repeat_offset + spec.repeats):
                steps = [r for r in all_rows if r["op"] == "step" and r["variant"] == variant and r["repeat"] == rep]
                walls = [r for r in all_rows if r["op"] == "step_wall" and r["variant"] == variant and r["repeat"] == rep]
                if steps:
                    all_rows.append(dict(max(steps, key=lambda r: r["median_ms"]), rank=-1, op="step_max"))
                if walls:
                    all_rows.append(dict(max(walls, key=lambda r: r["median_ms"]), rank=-1, op="step_wall_max"))
        csvio.append_rows(out_csv, csvio.RUN_COLUMNS, all_rows)
        for variant in variants:
            sm = [r for r in all_rows if r["op"] == "step_max" and r["variant"] == variant]
            per = [round(r["median_ms"], 3) for r in all_rows if r["op"] == "step" and r["variant"] == variant and r["repeat"] == spec.repeat_offset]
            if sm:
                print(f"[time] mode={spec.mode} B={spec.batch} c={spec.ctx} variant={variant} step_max={min(r['median_ms'] for r in sm):.3f} ms per-rank={per}", flush=True)
    del layer, st, rw, h, ctx
    torch.cuda.empty_cache()
    barrier_sync()
    return all_rows
