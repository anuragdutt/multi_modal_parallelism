"""The only module that imports vLLM kernels and custom ops. Every wrapper is a thin, named call so the
timer can attribute time to it and the byte model can reason per op. Signatures follow vLLM v0.30.0."""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

try:
    from vllm import _custom_ops as ops
    from vllm.model_executor.layers.mamba.ops.causal_conv1d import causal_conv1d_update
    from vllm.model_executor.layers.mamba.ops.layernorm_gated import rms_norm_gated as _rms_norm_gated
    from vllm.model_executor.layers.mamba.ops.mamba_ssm import selective_state_update as _ssu
except ImportError as e:  # pragma: no cover
    raise ImportError(f"vLLM kernels not importable ({e}); run inside the mmp:stage1 container") from e

try:
    from vllm.vllm_flash_attn import flash_attn_varlen_func as _fa_varlen
except ImportError:  # pragma: no cover
    from vllm.v1.attention.backends.fa_utils import flash_attn_varlen_func as _fa_varlen  # type: ignore



# ----------------------------------------------------------------------------- elementwise / norms
def rms_norm(x: torch.Tensor, w: torch.Tensor, eps: float, out: torch.Tensor) -> torch.Tensor:
    ops.rms_norm(out, x, w, eps)
    return out


def silu_and_mul(x: torch.Tensor) -> torch.Tensor:
    """x: [T, 2d] with [gate | up] halves -> silu(gate) * up, [T, d]. Direct call of vLLM's compiled op
    (the SiluAndMul CustomOp wrapper needs an engine config context, which the harness does not have)."""
    d = x.shape[-1] // 2
    out = torch.empty(x.shape[:-1] + (d,), dtype=x.dtype, device=x.device)
    torch.ops._C.silu_and_mul(out, x)
    return out


def gated_rmsnorm(y: torch.Tensor, z: torch.Tensor, w: torch.Tensor, eps: float, group) -> torch.Tensor:
    """Mamba-2 gated RMSNorm, gate applied before the norm (mamba_norm_before_gate=false).
    group=None -> vLLM's fused Triton kernel (tp=1 path); otherwise vLLM's native TP path with a [T,1]
    fp32 all-reduce of the local sum of squares over `group`."""
    if group is None:
        return _rms_norm_gated(y, w, None, z=z, eps=eps, norm_before_gate=False)
    dtype = y.dtype
    x = y * F.silu(z.to(torch.float32))
    local = x.pow(2).sum(dim=-1, keepdim=True)
    total = group.all_reduce(local)
    var = total / (group.world_size * x.shape[-1])
    x = x * torch.rsqrt(var + eps)
    return w * x.to(dtype)


# ----------------------------------------------------------------------------- rope
def rope_cos_sin_cache(head_dim: int, max_pos: int, theta: float, device: torch.device) -> torch.Tensor:
    """vLLM RotaryEmbedding cache layout: [max_pos, head_dim] = cat(cos, sin) over the rotary dim/2."""
    inv = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32, device=device) / head_dim))
    t = torch.arange(max_pos, dtype=torch.float32, device=device)
    freqs = torch.outer(t, inv)
    return torch.cat([freqs.cos(), freqs.sin()], dim=-1).to(torch.bfloat16)


def apply_rope(positions: torch.Tensor, q: torch.Tensor, k: torch.Tensor, head_dim: int, cache: torch.Tensor) -> None:
    """In place, NeoX style. q/k: [T, heads*head_dim]."""
    ops.rotary_embedding(positions, q, k, head_dim, cache, True)


# ----------------------------------------------------------------------------- attention (paged FA2)
def kv_cache_layout(num_blocks: int, block: int, nkv: int, hd: int) -> tuple[tuple[int, ...], str]:
    """Mirror vLLM's FlashAttention backend cache layout; returns (shape, kind).

    v0.30.0 stores (num_blocks, num_kv_heads, block_size, 2*head_dim) and derives
    key_cache, value_cache = kv.transpose(1, 2).split(head_dim, -1) (flash_attn.py:1233). Older releases
    exposed get_kv_cache_shape with a stacked (2, nb, bs, nkv, hd) layout; both are handled."""
    try:
        from vllm.v1.attention.backends.flash_attn import FlashAttentionBackend

        shape = tuple(FlashAttentionBackend.get_kv_cache_shape(num_blocks, block, nkv, hd))  # type: ignore[attr-defined]
    except Exception:
        shape = (num_blocks, nkv, block, 2 * hd)
    if shape[0] == 2 and len(shape) == 5:
        return shape, "stacked"  # (2, nb, bs, nkv, hd): key, value = kv.unbind(0)
    if len(shape) == 4 and shape[-1] == 2 * hd:
        return shape, "interleaved"  # (nb, nkv, bs, 2hd): key, value = kv.transpose(1,2).split(hd, -1)
    if len(shape) == 5 and shape[1] == 2:
        return shape, "stacked1"  # (nb, 2, bs, nkv, hd): key, value = kv.unbind(1)
    raise RuntimeError(f"unrecognized kv cache shape {shape}")


def alloc_kv(num_blocks: int, block: int, nkv: int, hd: int, device: torch.device, dtype=torch.bfloat16):
    shape, kind = kv_cache_layout(num_blocks, block, nkv, hd)
    kv = torch.zeros(shape, device=device, dtype=dtype)
    if kind == "stacked":
        key_cache, value_cache = kv.unbind(0)
    elif kind == "stacked1":
        key_cache, value_cache = kv.unbind(1)
    else:
        key_cache, value_cache = kv.transpose(1, 2).split(hd, dim=-1)
    return kv, key_cache, value_cache, kind


_ONE = {}


def _scale(device: torch.device) -> torch.Tensor:
    if device not in _ONE:
        _ONE[device] = torch.ones(1, device=device, dtype=torch.float32)
    return _ONE[device]


def write_kv(k: torch.Tensor, v: torch.Tensor, key_cache: torch.Tensor, value_cache: torch.Tensor, slot_mapping: torch.Tensor) -> None:
    """k, v: [T, nkv, hd]."""
    s = _scale(k.device)
    ops.reshape_and_cache_flash(k, v, key_cache, value_cache, slot_mapping, "auto", s, s)


def attn_decode(q: torch.Tensor, key_cache: torch.Tensor, value_cache: torch.Tensor, cu_q: torch.Tensor,
                seqused_k: torch.Tensor, max_k: int, block_table: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    """q: [T, nq, hd]; out: [T, nq, hd]. One query token per sequence (max_seqlen_q=1)."""
    hd = q.shape[-1]
    _fa_varlen(
        q=q, k=key_cache, v=value_cache, out=out,
        cu_seqlens_q=cu_q, max_seqlen_q=1, seqused_k=seqused_k, max_seqlen_k=max_k,
        softmax_scale=1.0 / math.sqrt(hd), causal=True, block_table=block_table, fa_version=2,
    )
    return out


# ----------------------------------------------------------------------------- mamba-2 decode
def conv_update(xBC: torch.Tensor, conv_state_T: torch.Tensor, w: torch.Tensor, b: torch.Tensor,
                idx: torch.Tensor) -> torch.Tensor:
    """xBC: [T, conv_rows]; conv_state_T: [lines, conv_rows, d_conv-1] view; w: [conv_rows, d_conv]."""
    return causal_conv1d_update(xBC, conv_state_T, w, b, "silu", conv_state_indices=idx)


def ssm_update(state: torch.Tensor, x: torch.Tensor, dt: torch.Tensor, A: torch.Tensor, B: torch.Tensor,
               C: torch.Tensor, D: torch.Tensor, dt_bias: torch.Tensor, idx: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    """Shapes follow MambaMixer2's decode call: state [lines, nh, p, N]; x/dt/out [T, nh, p]; A [nh, p, N];
    B/C [T, ng, N]; D/dt_bias [nh, p]."""
    _ssu(state, x, dt, A, B, C, D, dt_bias, dt_softplus=True, state_batch_indices=idx,
         dst_state_batch_indices=idx, out=out)
    return out


# ----------------------------------------------------------------------------- gemm wrappers
def gemm(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """x [T, in] @ w[out, in]^T."""
    return F.linear(x, w)


def gemm_T(x: torch.Tensor, wT: torch.Tensor) -> torch.Tensor:
    """x [T, k] @ wT [k, out]."""
    return x @ wT
