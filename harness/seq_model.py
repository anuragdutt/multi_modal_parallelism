"""Sequential-hybrid template (Nemotron 3 Nano / Nemotron-H style): each layer is h + mixer(norm(h)) with one of
three mixers: Mamba-2, GQA attention without positional encoding, or a sigmoid-routed MoE with a shared expert.

Layer types and layouts measured:
  mamba     : tp4 (heads sharded, grouped norm local since n_groups >= world)
  attention : tp4 (query heads sharded, KV head replicated on world/n_kv ranks)  vs  dp4 (sequences split,
              weights replicated, each rank reads only its own sequences' KV once, all-gather of outputs)
  moe       : tp4 (every expert column-sharded on every rank)  vs  ep4 (experts partitioned; a rank computes only
              the experts it owns), shared expert column-sharded in both; partial sums all-reduced.
Routing is computed once on the host from the fixed input so CUDA-graph capture sees static index tensors.
"""
from __future__ import annotations

import json
import math
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F

from . import csvio
from . import kernels as K
from .config import FalconH1Dims, RunSpec
from .correctness import compare, cross_rank_max_abs, passes
from .dist import Groups, barrier_sync, describe_comm
from .layer import AttentionBranch, SsmBranch
from .measure import gpu_env, vllm_commit
from .sharding import AttnShard, RankPlan, _even_split, attn_shard, mlp_shard, ssm_shard
from .state import DecodeCtx, RankStates, alloc_and_fill, full_conv_state, full_kv_head, full_ssm_head, input_hidden, make_decode_ctx
from .timing import OpTimer, time_eager, time_graph
from .weights import RankWeights, shard_to_device


# ----------------------------------------------------------------------------- dims and loading
@dataclass(frozen=True)
class SeqDims:
    hidden: int
    layers: int
    pattern: str
    n_q: int
    n_kv: int
    head_dim: int
    n_mamba_heads: int
    mamba_head_dim: int
    d_ssm: int
    d_state: int
    n_groups: int
    d_conv: int
    n_experts: int
    top_k: int
    moe_intermediate: int
    shared_intermediate: int
    routed_scaling: float
    norm_topk: bool
    rms_eps: float
    use_rope: bool = False
    rope_theta: float = 10000.0
    max_pos: int = 262144
    key_mult: float = 1.0
    attn_out: float = 1.0
    attn_in: float = 1.0

    @property
    def counts(self) -> dict[str, int]:
        return {"mamba": self.pattern.count("M"), "moe": self.pattern.count("E"), "attention": self.pattern.count("*")}

    def first_layer(self, kind: str) -> int:
        return self.pattern.index({"mamba": "M", "moe": "E", "attention": "*"}[kind])

    @classmethod
    def from_hf(cls, model_dir: str | Path) -> "SeqDims":
        cfg = json.loads((Path(model_dir) / "config.json").read_text())
        nh = cfg["mamba_num_heads"]
        return cls(hidden=cfg["hidden_size"], layers=cfg["num_hidden_layers"], pattern=cfg["hybrid_override_pattern"],
                   n_q=cfg["num_attention_heads"], n_kv=cfg["num_key_value_heads"], head_dim=cfg["head_dim"],
                   n_mamba_heads=nh, mamba_head_dim=cfg["mamba_head_dim"], d_ssm=nh * cfg["mamba_head_dim"],
                   d_state=cfg["ssm_state_size"], n_groups=cfg["n_groups"], d_conv=cfg["conv_kernel"],
                   n_experts=cfg["n_routed_experts"], top_k=cfg["num_experts_per_tok"],
                   moe_intermediate=cfg["moe_intermediate_size"], shared_intermediate=cfg["moe_shared_expert_intermediate_size"],
                   routed_scaling=cfg.get("routed_scaling_factor", 1.0), norm_topk=cfg.get("norm_topk_prob", True),
                   rms_eps=cfg.get("layer_norm_epsilon", cfg.get("rms_norm_eps", 1e-5)), rope_theta=cfg.get("rope_theta", 10000.0))

    def as_h1(self) -> FalconH1Dims:
        """Falcon-H1-shaped dims with unit multipliers so the shared Mamba-2/attention code applies."""
        return FalconH1Dims(hidden=self.hidden, intermediate=64, n_q=self.n_q, n_kv=self.n_kv, head_dim=self.head_dim,
                            n_mamba_heads=self.n_mamba_heads, mamba_head_dim=self.mamba_head_dim, d_ssm=self.d_ssm,
                            d_state=self.d_state, n_groups=self.n_groups, d_conv=self.d_conv, chunk=128, rms_eps=self.rms_eps,
                            rope_theta=self.rope_theta, max_pos=self.max_pos, attn_in=1.0, attn_out=1.0, key_mult=1.0,
                            ssm_in=1.0, ssm_out=1.0, ssm_mults=(1.0, 1.0, 1.0, 1.0, 1.0), mlp_mults=(1.0, 1.0))


def load_seq_layer(model_dir: str | Path, dims: SeqDims, kind: str, layer_idx: int | None = None) -> dict[str, torch.Tensor]:
    from safetensors import safe_open

    model_dir = Path(model_dir)
    L = dims.first_layer(kind) if layer_idx is None else layer_idx
    index = json.loads((model_dir / "model.safetensors.index.json").read_text())["weight_map"]
    prefix = f"backbone.layers.{L}."
    names = [k for k in index if k.startswith(prefix)]
    by_file: dict[str, list[str]] = {}
    for n in names:
        by_file.setdefault(index[n], []).append(n)
    raw: dict[str, torch.Tensor] = {}
    for fname, ns in by_file.items():
        with safe_open(str(model_dir / fname), framework="pt", device="cpu") as f:
            for n in ns:
                raw[n[len(prefix):]] = f.get_tensor(n)
    if kind == "mamba":  # remap to the Falcon-H1 key names the shared sharding code expects
        H = dims.hidden
        return {
            "mamba.in_proj.weight": raw["mixer.in_proj.weight"], "mamba.conv1d.weight": raw["mixer.conv1d.weight"],
            "mamba.conv1d.bias": raw["mixer.conv1d.bias"], "mamba.A_log": raw["mixer.A_log"], "mamba.D": raw["mixer.D"],
            "mamba.dt_bias": raw["mixer.dt_bias"], "mamba.norm.weight": raw["mixer.norm.weight"],
            "mamba.out_proj.weight": raw["mixer.out_proj.weight"], "input_layernorm.weight": raw["norm.weight"],
            # dummies for the shared loader (unused by the Mamba layer)
            "self_attn.q_proj.weight": torch.zeros(dims.n_q * dims.head_dim, H, dtype=torch.bfloat16),
            "self_attn.k_proj.weight": torch.zeros(dims.n_kv * dims.head_dim, H, dtype=torch.bfloat16),
            "self_attn.v_proj.weight": torch.zeros(dims.n_kv * dims.head_dim, H, dtype=torch.bfloat16),
            "self_attn.o_proj.weight": torch.zeros(H, dims.n_q * dims.head_dim, dtype=torch.bfloat16),
            "feed_forward.gate_proj.weight": torch.zeros(64, H, dtype=torch.bfloat16),
            "feed_forward.up_proj.weight": torch.zeros(64, H, dtype=torch.bfloat16),
            "feed_forward.down_proj.weight": torch.zeros(H, 64, dtype=torch.bfloat16),
            "pre_ff_layernorm.weight": torch.ones(H, dtype=torch.bfloat16),
        }
    if kind == "attention":
        return {"q": raw["mixer.q_proj.weight"], "k": raw["mixer.k_proj.weight"], "v": raw["mixer.v_proj.weight"],
                "o": raw["mixer.o_proj.weight"], "norm": raw["norm.weight"]}
    # moe
    E = dims.n_experts
    up = torch.stack([raw[f"mixer.experts.{e}.up_proj.weight"] for e in range(E)])  # [E, I, H]
    down = torch.stack([raw[f"mixer.experts.{e}.down_proj.weight"] for e in range(E)])  # [E, H, I]
    return {"up": up, "down": down, "gate_w": raw["mixer.gate.weight"], "gate_bias": raw["mixer.gate.e_score_correction_bias"],
            "shared_up": raw["mixer.shared_experts.up_proj.weight"], "shared_down": raw["mixer.shared_experts.down_proj.weight"],
            "norm": raw["norm.weight"]}


# ----------------------------------------------------------------------------- shared pieces
def rms(x: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    out = torch.empty_like(x)
    return K.rms_norm(x, w, eps, out)


def route(dims: SeqDims, x: torch.Tensor, gate_w: torch.Tensor, gate_bias: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Nemotron sigmoid router with bias-corrected choice; returns (topk_idx [T,k] long, topk_w [T,k] fp32)."""
    logits = x.float() @ gate_w.float().T
    scores = torch.sigmoid(logits)
    choice = scores + gate_bias.float()[None]
    idx = torch.topk(choice, dims.top_k, dim=-1, sorted=False)[1]
    w = torch.gather(scores, 1, idx)
    if dims.norm_topk:
        w = w / (w.sum(-1, keepdim=True) + 1e-20)
    return idx, w * dims.routed_scaling


# ----------------------------------------------------------------------------- the three layers
class MambaLayer:
    """h + out_proj(ssm(norm(h))) under tensor parallelism; reuses the Falcon-H1 SSM branch with unit multipliers."""

    def __init__(self, dims: SeqDims, full: dict, groups: Groups, ctx: DecodeCtx, seed: int, device: torch.device, collectives: bool):
        h1 = dims.as_h1()
        world, rank = groups.world_size, groups.rank
        self.plan = RankPlan(mode="tp4", rank=rank, world=world, attn=None,
                             ssm=ssm_shard(h1, rank, world, tuple(range(world))), mlp=mlp_shard(h1, rank, world, 0.0, None),
                             attn_ranks=(), ssm_ranks=tuple(range(world)), combine="sum_then_allreduce")
        self.w = shard_to_device(full, self.plan, h1, device)
        self.st = alloc_and_fill(self.plan, h1, ctx, seed, device)
        ng = groups.group_for_ranks(self.plan.ssm.norm_ranks) if collectives else None
        self.branch = SsmBranch(self.plan, self.w, h1, self.st, ctx, device, ng)
        self.groups, self.dims, self.collectives = groups, dims, collectives
        self.out = torch.empty(ctx.B, dims.hidden, device=device, dtype=torch.bfloat16)

    def step(self, h: torch.Tensor, t: OpTimer) -> torch.Tensor:
        t.begin("norm_in")
        x = rms(h, self.w.in_norm, self.dims.rms_eps)
        t.end("norm_in")
        part = self.branch.decode(x, t)
        if self.groups.world_size > 1 and self.collectives:
            t.begin("ar.mixer")
            part = self.groups.tp.all_reduce(part)
            t.end("ar.mixer")
        self.out.copy_(h + part)
        return self.out


class AttentionLayer:
    """h + o_proj(attn(norm(h))). tp4: heads sharded + all-reduce. dp4: sequences split + all-gather."""

    def __init__(self, dims: SeqDims, full: dict, groups: Groups, ctx: DecodeCtx, seed: int, device: torch.device,
                 mode: str, collectives: bool):
        self.dims, self.groups, self.mode, self.collectives, self.device = dims, groups, mode, collectives, device
        world, rank, B = groups.world_size, groups.rank, ctx.B
        h1 = dims.as_h1()
        bf = torch.bfloat16
        self.norm_w = full["norm"].to(device, bf)
        if mode == "dp4":
            self.Bl = math.ceil(B / world)
            self.b0 = min(rank * self.Bl, B)
            self.nloc = max(0, min(self.Bl, B - self.b0))
            shard = attn_shard(h1, 0, 1)  # every head
            self.ctx = make_decode_ctx(self.Bl, ctx.c, ctx.kv_block, device)
        else:
            shard = attn_shard(h1, rank, world)
            self.Bl, self.b0, self.nloc = B, 0, B
            self.ctx = ctx
        plan = RankPlan(mode=mode, rank=rank, world=world, attn=shard, ssm=None, mlp=mlp_shard(h1, rank, world, 0.0, None),
                        attn_ranks=tuple(range(world)), ssm_ranks=(), combine="sum_then_allreduce")
        rw = RankWeights(wqkv=None, wo=None)
        rw.wqkv = torch.cat([full["q"][shard.q_rows], full["k"][shard.k_rows], full["v"][shard.v_rows]], 0).to(device, bf).contiguous()
        rw.wo = full["o"][:, shard.o_cols].to(device, bf).contiguous()
        rw.q_rows, rw.kv_rows = shard.q_rows.stop - shard.q_rows.start, shard.k_rows.stop - shard.k_rows.start
        self.w = rw
        self.plan = plan
        # KV cache: tp -> all B sequences for the local kv head(s); dp -> this rank's sequence slice, all kv heads
        self.st = alloc_and_fill(plan, h1, self.ctx, seed, device) if mode != "dp4" else self._fill_dp(h1, seed, device)
        self.branch = AttentionBranch(plan, rw, h1, self.st, self.ctx, device)
        self.branch.use_rope = dims.use_rope
        self.out = torch.empty(B, dims.hidden, device=device, dtype=bf)
        self.gather_buf = torch.empty(world * self.Bl, dims.hidden, device=device, dtype=bf)
        self.local_buf = torch.zeros(self.Bl, dims.hidden, device=device, dtype=bf)

    def _fill_dp(self, h1: FalconH1Dims, seed: int, device: torch.device) -> RankStates:
        """Allocate the DP rank's cache for its sequence slice with the same deterministic content as tp4."""
        B_total = self.out.shape[0] if hasattr(self, "out") else None
        st = RankStates()
        nb = self.Bl * self.ctx.nblk + 1
        st.kv, st.key_cache, st.value_cache, st.kv_kind = K.alloc_kv(nb, self.ctx.kv_block, self.dims.n_kv, self.dims.head_dim, device)
        Bg = self.b0 + self.Bl  # generate up to the last global sequence this rank holds
        for hh in range(self.dims.n_kv):
            k, v = full_kv_head(seed, hh, Bg, self.ctx.c, self.dims.head_dim, device)
            for bl in range(self.nloc):
                b = self.b0 + bl
                for bi, blk in enumerate(self.ctx.block_table[bl].tolist()):
                    lo = bi * self.ctx.kv_block
                    if lo >= self.ctx.c:
                        break
                    n = min(self.ctx.kv_block, self.ctx.c - lo)
                    st.key_cache[blk, :n, hh, :] = k[b, lo : lo + n]
                    st.value_cache[blk, :n, hh, :] = v[b, lo : lo + n]
            del k, v
        return st

    def step(self, h: torch.Tensor, t: OpTimer) -> torch.Tensor:
        t.begin("norm_in")
        x = rms(h, self.norm_w, self.dims.rms_eps)
        t.end("norm_in")
        if self.mode == "dp4":
            xl = self.local_buf
            xl.zero_()
            if self.nloc > 0:
                xl[: self.nloc].copy_(x[self.b0 : self.b0 + self.nloc])
            part = self.branch.decode(xl, t)  # [Bl, H]
            if self.groups.world_size > 1 and self.collectives:
                t.begin("ag.attn")
                dist.all_gather_into_tensor(self.gather_buf, part.contiguous())
                t.end("ag.attn")
                full_out = self.gather_buf[: h.shape[0]]  # ranks hold consecutive slices; drop the padding
            else:
                full_out = part[: h.shape[0]]
            self.out.copy_(h + full_out)
            return self.out
        part = self.branch.decode(x, t)
        if self.groups.world_size > 1 and self.collectives:
            t.begin("ar.mixer")
            part = self.groups.tp.all_reduce(part)
            t.end("ar.mixer")
        self.out.copy_(h + part)
        return self.out


class MoELayer:
    """h + shared(norm(h)) + sum_k w_k expert_k(norm(h)) with relu^2 experts. tp4: every expert column-sharded;
    ep4: experts partitioned across ranks. Routing precomputed on the host for the fixed input."""

    def __init__(self, dims: SeqDims, full: dict, groups: Groups, ctx: DecodeCtx, h: torch.Tensor, device: torch.device,
                 mode: str, collectives: bool):
        self.dims, self.groups, self.mode, self.collectives = dims, groups, mode, collectives
        world, rank = groups.world_size, groups.rank
        bf = torch.bfloat16
        E, I, H = dims.n_experts, dims.moe_intermediate, dims.hidden
        self.norm_w = full["norm"].to(device, bf)
        # shared expert: column-sharded in both modes
        sc = _even_split(dims.shared_intermediate, world)[rank]
        self.sh_up = full["shared_up"][sc.start : sc.stop].to(device, bf).contiguous()
        self.sh_downT = full["shared_down"][:, sc.start : sc.stop].T.to(device, bf).contiguous()
        # routing on the host from the fixed input
        x = rms(h, self.norm_w, dims.rms_eps)
        idx, w = route(dims, x, full["gate_w"].to(device), full["gate_bias"].to(device))
        self.plan_experts: list[tuple[int, torch.Tensor, torch.Tensor]] = []
        if mode == "ep4":
            owned = _even_split(E, world)[rank]
            mine = set(range(owned.start, owned.stop))
        else:
            mine = set(range(E))
        for e in sorted(mine):
            tok, slot = torch.nonzero(idx == e, as_tuple=True)
            if tok.numel() == 0:
                continue
            self.plan_experts.append((e, tok, w[tok, slot].to(bf)[:, None]))
        touched = [e for e, _, _ in self.plan_experts]
        if mode == "ep4":
            self.up = {e: full["up"][e].to(device, bf).contiguous() for e in touched}  # [I, H]
            self.downT = {e: full["down"][e].T.to(device, bf).contiguous() for e in touched}  # [I, H]
        else:
            cols = _even_split(I, world)[rank]
            self.up = {e: full["up"][e][cols.start : cols.stop].to(device, bf).contiguous() for e in touched}
            self.downT = {e: full["down"][e][:, cols.start : cols.stop].T.to(device, bf).contiguous() for e in touched}
        self.out = torch.empty_like(h)
        self.n_touched_total = int(torch.unique(idx).numel())

    def step(self, h: torch.Tensor, t: OpTimer) -> torch.Tensor:
        t.begin("norm_in")
        x = rms(h, self.norm_w, self.dims.rms_eps)
        t.end("norm_in")
        t.begin("moe.shared")
        part = K.gemm_T(F.relu(K.gemm(x, self.sh_up)) ** 2, self.sh_downT)
        t.end("moe.shared")
        t.begin("moe.experts")
        for e, tok, wt in self.plan_experts:
            xe = x[tok]
            ye = K.gemm_T(F.relu(K.gemm(xe, self.up[e])) ** 2, self.downT[e]) * wt
            part.index_add_(0, tok, ye)
        t.end("moe.experts")
        if self.groups.world_size > 1 and self.collectives:
            t.begin("ar.mixer")
            part = self.groups.tp.all_reduce(part)
            t.end("ar.mixer")
        self.out.copy_(h + part)
        return self.out


# ----------------------------------------------------------------------------- references
@torch.no_grad()
def reference_attention(dims: SeqDims, full: dict, h: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor) -> torch.Tensor:
    dev = h.device
    f32 = lambda k: full[k].to(dev, torch.float32)  # noqa: E731
    B, nq, nkv, hd = h.shape[0], dims.n_q, dims.n_kv, dims.head_dim
    hf = h.float()
    x = hf * torch.rsqrt(hf.pow(2).mean(-1, keepdim=True) + dims.rms_eps) * f32("norm")
    q = (x @ f32("q").T).view(B, nq, hd)
    k = (x @ f32("k").T).view(B, nkv, hd)
    v = (x @ f32("v").T).view(B, nkv, hd)
    Kc = torch.cat([k_cache.float(), k[:, None]], 1)
    Vc = torch.cat([v_cache.float(), v[:, None]], 1)
    rep = nq // nkv
    scores = torch.einsum("bqd,bkqd->bqk", q, Kc.repeat_interleave(rep, 2)) / math.sqrt(hd)
    o = torch.einsum("bqk,bkqd->bqd", torch.softmax(scores, -1), Vc.repeat_interleave(rep, 2)).reshape(B, nq * hd)
    return hf + o @ f32("o").T


@torch.no_grad()
def reference_moe(dims: SeqDims, full: dict, h: torch.Tensor) -> torch.Tensor:
    dev = h.device
    hf = h.float()
    x = hf * torch.rsqrt(hf.pow(2).mean(-1, keepdim=True) + dims.rms_eps) * full["norm"].to(dev).float()
    idx, w = route(dims, x, full["gate_w"].to(dev), full["gate_bias"].to(dev))
    y = F.relu(x @ full["shared_up"].to(dev).float().T) ** 2 @ full["shared_down"].to(dev).float().T
    for e in torch.unique(idx).tolist():
        tok, slot = torch.nonzero(idx == e, as_tuple=True)
        ye = F.relu(x[tok] @ full["up"][e].to(dev).float().T) ** 2 @ full["down"][e].to(dev).float().T
        y.index_add_(0, tok, ye * w[tok, slot][:, None])
    return hf + y


@torch.no_grad()
def reference_mamba(dims: SeqDims, full: dict, h: torch.Tensor, conv_state: torch.Tensor, ssm_state: torch.Tensor) -> torch.Tensor:
    from .reference import decode_step

    h1 = dims.as_h1()
    # reuse the Falcon-H1 reference with attention and MLP contributions removed via zero weights (dummies above);
    # unit multipliers make the mixer the plain Mamba-2 block; the dummy MLP (zeros) contributes nothing.
    B, hd = h.shape[0], dims.head_dim
    kz = torch.zeros(B, 0, dims.n_kv, hd, device=h.device, dtype=torch.bfloat16)
    out = decode_step(full, h1, h, kz, kz, conv_state, ssm_state, 0)
    return out


# ----------------------------------------------------------------------------- measurement
def measure_seq(spec: RunSpec, kind: str, groups: Groups, dims: SeqDims, full: dict, out_csv: str,
                correctness_csv: str, check: bool, image_tag: str = "") -> list[dict]:
    rank, world, device = groups.rank, groups.world_size, groups.device
    ctx = make_decode_ctx(spec.batch, spec.ctx, spec.kv_block, device)
    h = input_hidden(spec.seed, spec.batch, dims.hidden, device)
    if kind == "mamba":
        layer = MambaLayer(dims, full, groups, ctx, spec.seed, device, spec.collectives)
    elif kind == "attention":
        layer = AttentionLayer(dims, full, groups, ctx, spec.seed, device, spec.mode, spec.collectives)
    else:
        layer = MoELayer(dims, full, groups, ctx, h, device, spec.mode, spec.collectives)
    barrier_sync()
    base = dict(run_id=uuid.uuid4().hex[:8], ts=time.strftime("%Y-%m-%dT%H:%M:%S"), image_tag=image_tag, vllm_commit=vllm_commit(),
                mode=f"{kind}/{spec.mode}", S=0.0, w=0, cuts="", cuts_src="", batch=spec.batch, ctx=spec.ctx, kv_block=spec.kv_block,
                comm_tp=json.dumps(describe_comm(groups.tp)) if spec.collectives else "none",
                comm_pair=json.dumps(describe_comm(groups.pair)), tag=spec.tag)
    if check and spec.collectives:
        out = layer.step(h, OpTimer(enabled=False)).clone()
        torch.cuda.synchronize()
        xr = cross_rank_max_abs(out, world)
        if rank == 0:
            if kind == "attention":
                ks = [full_kv_head(spec.seed, hh, spec.batch, spec.ctx, dims.head_dim, device) for hh in range(dims.n_kv)]
                ref = reference_attention(dims, full, h, torch.stack([k for k, _ in ks], 2), torch.stack([v for _, v in ks], 2))
            elif kind == "moe":
                ref = reference_moe(dims, full, h)
            else:
                h1 = dims.as_h1()
                conv = full_conv_state(spec.seed, h1, spec.batch, device)
                ssm = torch.stack([full_ssm_head(spec.seed, j, spec.batch, dims.mamba_head_dim, dims.d_state, device) for j in range(dims.n_mamba_heads)], 1)
                ref = reference_mamba(dims, full, h, conv, ssm)
            m = compare(out, ref)
            ok = passes(m) and xr == 0.0
            row = dict(base, ref="fp32", cross_rank_max_abs=xr, **{k: m[k] for k in ("max_abs", "max_rel", "mean_rel", "cos")})
            row["pass"] = ok
            csvio.append_rows(correctness_csv, csvio.CORRECTNESS_COLUMNS, [row])
            print(f"[check] {kind}/{spec.mode} B={spec.batch} c={spec.ctx} max_abs={m['max_abs']:.4g} (ref max {m['ref_max_abs']:.3g}) "
                  f"mean_rel={m['mean_rel']:.3e} cos={m['cos']:.6f} cross_rank={xr:.3g} pass={ok}", flush=True)
        # refresh states that the check step mutated
        if kind == "mamba":
            st2 = alloc_and_fill(layer.plan, dims.as_h1(), ctx, spec.seed, device)
            layer.st.conv_state.copy_(st2.conv_state)
            layer.st.ssm_state.copy_(st2.ssm_state)
        barrier_sync()
    variants = ["eager", "graph"] if spec.variant == "both" else [spec.variant]
    rows: list[dict] = []
    for rep in range(spec.repeats):
        for variant in variants:
            env = gpu_env(groups.local_rank)
            stats, wall = (time_eager if variant == "eager" else time_graph)(layer, h, spec.n_warm, spec.n_iter, groups)
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
                if steps:
                    all_rows.append(dict(max(steps, key=lambda r: r["median_ms"]), rank=-1, op="step_max"))
        csvio.append_rows(out_csv, csvio.RUN_COLUMNS, all_rows)
        for variant in variants:
            sm = [r for r in all_rows if r["op"] == "step_max" and r["variant"] == variant]
            per = [round(r["median_ms"], 3) for r in all_rows if r["op"] == "step" and r["variant"] == variant and r["repeat"] == spec.repeat_offset]
            if sm:
                print(f"[time] {kind}/{spec.mode} B={spec.batch} c={spec.ctx} variant={variant} step_max={min(r['median_ms'] for r in sm):.3f} ms per-rank={per}", flush=True)
    del layer, h, ctx
    torch.cuda.empty_cache()
    barrier_sync()
    return all_rows
