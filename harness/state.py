"""Decode-step metadata and per-rank state buffers (paged KV, conv state, SSM state) with deterministic
content keyed by (sequence, GLOBAL head, position) so every mode's shard sees identical values and the
fp32 reference can rebuild the full states on rank 0."""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from .config import FalconH1Dims
from .sharding import RankPlan
from . import kernels as K


@dataclass
class DecodeCtx:
    B: int
    c: int
    kv_block: int
    nblk: int
    positions: torch.Tensor  # [B] int64 = c
    cu_q: torch.Tensor  # [B+1] int32
    seqused_k: torch.Tensor  # [B] int32 = c+1
    block_table: torch.Tensor  # [B, nblk] int32
    slot_mapping: torch.Tensor  # [B] int64
    state_idx: torch.Tensor  # [B] int32, values 1..B (line 0 is the null block)
    max_k: int


def make_decode_ctx(B: int, c: int, kv_block: int, device: torch.device) -> DecodeCtx:
    nblk = math.ceil((c + 1) / kv_block)
    bt = torch.arange(B * nblk, device=device, dtype=torch.int32).view(B, nblk)
    slot = (bt[:, c // kv_block].to(torch.int64) * kv_block + (c % kv_block))
    return DecodeCtx(
        B=B, c=c, kv_block=kv_block, nblk=nblk,
        positions=torch.full((B,), c, device=device, dtype=torch.int64),
        cu_q=torch.arange(B + 1, device=device, dtype=torch.int32),
        seqused_k=torch.full((B,), c + 1, device=device, dtype=torch.int32),
        block_table=bt, slot_mapping=slot,
        state_idx=torch.arange(1, B + 1, device=device, dtype=torch.int32),  # line 0 is vLLM's null block
        max_k=c + 1,
    )


def _gen(seed: int, device: torch.device) -> torch.Generator:
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    return g


def full_kv_head(seed: int, h: int, B: int, c: int, hd: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Keys/values of global kv head h for positions [0, c): two [B, c, hd] bf16 tensors."""
    g = _gen(seed * 7919 + 100003 * h + 1, device)
    k = (torch.randn(B, c, hd, generator=g, device=device) * 0.5).to(torch.bfloat16)
    v = (torch.randn(B, c, hd, generator=g, device=device) * 0.5).to(torch.bfloat16)
    return k, v


def full_ssm_head(seed: int, j: int, B: int, p: int, N: int, device: torch.device) -> torch.Tensor:
    g = _gen(seed * 7919 + 200003 * j + 2, device)
    return (torch.randn(B, p, N, generator=g, device=device) * 0.1).to(torch.bfloat16)


def full_conv_state(seed: int, dims: FalconH1Dims, B: int, device: torch.device) -> torch.Tensor:
    """[B, d_conv-1, conv_rows] bf16, oldest position first."""
    g = _gen(seed * 7919 + 300007, device)
    return (torch.randn(B, dims.d_conv - 1, dims.conv_rows, generator=g, device=device) * 0.5).to(torch.bfloat16)


def input_hidden(seed: int, B: int, H: int, device: torch.device) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed * 7919 + 400009)
    return torch.randn(B, H, generator=g).to(device=device, dtype=torch.bfloat16)


@dataclass
class RankStates:
    kv: torch.Tensor | None = None  # the raw paged buffer
    key_cache: torch.Tensor | None = None
    value_cache: torch.Tensor | None = None
    kv_kind: str = ""
    conv_state: torch.Tensor | None = None  # [B+1, d_conv-1, conv_rows_local]
    ssm_state: torch.Tensor | None = None  # [B+1, nhr, p, N]

    def bytes(self) -> dict[str, int]:
        nb = lambda t: 0 if t is None else t.numel() * t.element_size()  # noqa: E731
        return {"kv": nb(self.kv), "conv": nb(self.conv_state), "ssm": nb(self.ssm_state)}


def alloc_and_fill(plan: RankPlan, dims: FalconH1Dims, ctx: DecodeCtx, seed: int, device: torch.device) -> RankStates:
    st = RankStates()
    B, c = ctx.B, ctx.c
    if plan.attn is not None:
        a = plan.attn
        nkv_local = len(a.kv_heads)
        nb = B * ctx.nblk + 1
        st.kv, st.key_cache, st.value_cache, st.kv_kind = K.alloc_kv(nb, ctx.kv_block, nkv_local, dims.head_dim, device)
        # fill positions [0, c) of each sequence's blocks for the local kv heads
        for li, h in enumerate(a.kv_heads):
            k, v = full_kv_head(seed, h, B, c, dims.head_dim, device)
            for b in range(B):
                blocks = ctx.block_table[b].tolist()
                for bi, blk in enumerate(blocks):
                    lo = bi * ctx.kv_block
                    if lo >= c:
                        break
                    n = min(ctx.kv_block, c - lo)
                    st.key_cache[blk, :n, li, :] = k[b, lo : lo + n]
                    st.value_cache[blk, :n, li, :] = v[b, lo : lo + n]
            del k, v
    if plan.ssm is not None:
        s = plan.ssm
        nhr, p, N = len(s.heads), dims.mamba_head_dim, dims.d_state
        conv_rows_local = sum(r.stop - r.start for r in s.conv_rows)
        st.conv_state = torch.zeros(B + 1, dims.d_conv - 1, conv_rows_local, device=device, dtype=torch.bfloat16)
        full_conv = full_conv_state(seed, dims, B, device)
        st.conv_state[1 : B + 1] = torch.cat([full_conv[:, :, r] for r in s.conv_rows], -1)
        st.ssm_state = torch.zeros(B + 1, nhr, p, N, device=device, dtype=torch.bfloat16)
        for li, j in enumerate(s.heads):
            st.ssm_state[1 : B + 1, li] = full_ssm_head(seed, j, B, p, N, device)
    return st


def full_states_for_reference(dims: FalconH1Dims, B: int, c: int, seed: int, device: torch.device):
    """Full unsharded initial states on one device for reference.decode_step."""
    ks, vs = [], []
    for h in range(dims.n_kv):
        k, v = full_kv_head(seed, h, B, c, dims.head_dim, device)
        ks.append(k)
        vs.append(v)
    k_cache = torch.stack(ks, 2)  # [B, c, nkv, hd]
    v_cache = torch.stack(vs, 2)
    conv = full_conv_state(seed, dims, B, device)
    ssm = torch.stack([full_ssm_head(seed, j, B, dims.mamba_head_dim, dims.d_state, device) for j in range(dims.n_mamba_heads)], 1)
    return k_cache, v_cache, conv, ssm
