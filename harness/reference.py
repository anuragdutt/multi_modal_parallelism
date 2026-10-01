"""Pure-torch fp32 reference of one Falcon-H1 decoder layer decode step (no vLLM). Runs on one GPU with
the full (unsharded) weights and the full initial states, so every parallel mode can be checked against it."""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from .config import FalconH1Dims


def rmsnorm(x: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    var = x.pow(2).mean(-1, keepdim=True)
    return x * torch.rsqrt(var + eps) * w


def rope_cos_sin(positions: torch.Tensor, hd: int, theta: float) -> tuple[torch.Tensor, torch.Tensor]:
    inv = 1.0 / (theta ** (torch.arange(0, hd, 2, device=positions.device, dtype=torch.float32) / hd))
    f = positions.float()[:, None] * inv[None, :]  # [B, hd/2]
    return torch.cos(f), torch.sin(f)


def apply_rope_neox(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """x: [B, heads, hd]; GPT-NeoX style (rotate halves), matching vLLM's is_neox=True."""
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    c, s = cos[:, None, :], sin[:, None, :]
    return torch.cat([x1 * c - x2 * s, x2 * c + x1 * s], -1)


@torch.no_grad()
def decode_step(
    full: dict[str, torch.Tensor],
    dims: FalconH1Dims,
    h: torch.Tensor,  # [B, H] any dtype
    k_cache: torch.Tensor,  # [B, c, n_kv, hd] previous keys (already rotated and key-multiplied)
    v_cache: torch.Tensor,  # [B, c, n_kv, hd]
    conv_state: torch.Tensor,  # [B, d_conv-1, conv_rows] previous conv inputs, oldest first
    ssm_state: torch.Tensor,  # [B, nh, p, N]
    pos: int,
) -> torch.Tensor:
    dev = h.device
    f32 = lambda t: full[t].to(dev, torch.float32)  # noqa: E731
    B = h.shape[0]
    H, hd, nq, nkv = dims.hidden, dims.head_dim, dims.n_q, dims.n_kv
    nh, p, N, G = dims.n_mamba_heads, dims.mamba_head_dim, dims.d_state, dims.n_groups
    h = h.float()
    x = rmsnorm(h, f32("input_layernorm.weight"), dims.rms_eps)

    # ---- attention branch
    xa = x * dims.attn_in
    q = (xa @ f32("self_attn.q_proj.weight").T).view(B, nq, hd)
    k = (xa @ f32("self_attn.k_proj.weight").T).view(B, nkv, hd) * dims.key_mult
    v = (xa @ f32("self_attn.v_proj.weight").T).view(B, nkv, hd)
    positions = torch.full((B,), pos, device=dev, dtype=torch.int64)
    cos, sin = rope_cos_sin(positions, hd, dims.rope_theta)
    q, k = apply_rope_neox(q, cos, sin), apply_rope_neox(k, cos, sin)
    K = torch.cat([k_cache.float(), k[:, None]], 1)  # [B, c+1, nkv, hd]
    V = torch.cat([v_cache.float(), v[:, None]], 1)
    rep = nq // nkv
    Kq = K.repeat_interleave(rep, dim=2)  # [B, c+1, nq, hd]
    Vq = V.repeat_interleave(rep, dim=2)
    scores = torch.einsum("bqd,bkqd->bqk", q, Kq) / math.sqrt(hd)
    probs = torch.softmax(scores, -1)
    o = torch.einsum("bqk,bkqd->bqd", probs, Vq).reshape(B, nq * hd)
    attn = (o @ f32("self_attn.o_proj.weight").T) * dims.attn_out

    # ---- mamba branch
    xs = x * dims.ssm_in
    zxbcdt = xs @ f32("mamba.in_proj.weight").T
    sizes = [dims.d_ssm, dims.d_ssm, G * N, G * N, nh]
    mults = dims.ssm_mults
    off = 0
    for n, m in zip(sizes, mults):
        zxbcdt[:, off : off + n] *= m
        off += n
    z, xBC, dt = torch.split(zxbcdt, [dims.d_ssm, dims.d_ssm + 2 * G * N, nh], -1)
    cw = f32("mamba.conv1d.weight")[:, 0, :]  # [conv_rows, d_conv]
    cb = f32("mamba.conv1d.bias")
    window = torch.cat([conv_state.float(), xBC[:, None, :]], 1)  # [B, d_conv, conv_rows]
    conv = (window * cw.T[None]).sum(1) + cb  # [B, conv_rows]
    conv = F.silu(conv)
    xm, Bm, Cm = torch.split(conv, [dims.d_ssm, G * N, G * N], -1)
    xm = xm.view(B, nh, p)
    Bm = Bm.view(B, G, N)
    Cm = Cm.view(B, G, N)
    A = -torch.exp(f32("mamba.A_log"))  # [nh]
    dt = F.softplus(dt + f32("mamba.dt_bias"))  # [B, nh]
    dA = torch.exp(dt * A)  # [B, nh]
    hpg = nh // G
    gidx = torch.arange(nh, device=dev) // hpg
    Bh = Bm[:, gidx]  # [B, nh, N]
    Ch = Cm[:, gidx]
    state = ssm_state.float() * dA[:, :, None, None] + dt[:, :, None, None] * xm[:, :, :, None] * Bh[:, :, None, :]
    y = (state * Ch[:, :, None, :]).sum(-1) + f32("mamba.D")[None, :, None] * xm  # [B, nh, p]
    y = y.reshape(B, nh * p)
    y = y * F.silu(z)
    gw = dims.d_ssm // G
    yg = y.view(B, G, gw)
    yg = yg * torch.rsqrt(yg.pow(2).mean(-1, keepdim=True) + dims.rms_eps)
    y = yg.view(B, nh * p) * f32("mamba.norm.weight")
    ssm = (y @ f32("mamba.out_proj.weight").T) * dims.ssm_out

    h = h + attn + ssm

    # ---- mlp
    y2 = rmsnorm(h, f32("pre_ff_layernorm.weight"), dims.rms_eps)
    gate = F.silu((y2 @ f32("feed_forward.gate_proj.weight").T) * dims.mlp_mults[0])
    up = y2 @ f32("feed_forward.up_proj.weight").T
    mlp = ((gate * up) @ f32("feed_forward.down_proj.weight").T) * dims.mlp_mults[1]
    return h + mlp
