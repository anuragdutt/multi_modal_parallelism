"""Stage-by-stage comparison of the kernel path against the fp32 reference (single rank, tp1).

  torchrun --standalone --nproc_per_node=1 -m harness.debug --batch 2 --ctx 64
"""
from __future__ import annotations

import argparse
import math
import os

import torch
import torch.nn.functional as F

from . import kernels as K
from .config import FalconH1Dims
from .correctness import compare
from .dist import destroy, init_groups
from .layer import ParallelHybridLayer
from .reference import apply_rope_neox, decode_step, rope_cos_sin
from .sharding import plan_for
from .state import alloc_and_fill, full_states_for_reference, input_hidden, make_decode_ctx
from .timing import NULL
from .weights import load_layer_from_safetensors, random_layer, shard_to_device


def rep(name: str, a: torch.Tensor, b: torch.Tensor) -> None:
    m = compare(a, b)
    print(f"  {name:12s} max_abs={m['max_abs']:.4g} ref_max={m['ref_max_abs']:.4g} mean_rel={m['mean_rel']:.3e} cos={m['cos']:.6f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default=os.environ.get("MMP_MODEL_DIR", ""))
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--ctx", type=int, default=64)
    ap.add_argument("--random-weights", action="store_true")
    args = ap.parse_args()
    g = init_groups()
    dev = g.device
    dims = FalconH1Dims.from_hf(args.model_dir) if args.model_dir else FalconH1Dims.from_yaml("configs/falcon_h1_7b.yaml")
    full = random_layer(dims, 0) if (args.random_weights or not args.model_dir) else load_layer_from_safetensors(args.model_dir, 0)
    plan = plan_for("tp1", 0, 1, dims)
    rw = shard_to_device(full, plan, dims, dev)
    ctx = make_decode_ctx(args.batch, args.ctx, 16, dev)
    st = alloc_and_fill(plan, dims, ctx, 0, dev)
    h = input_hidden(0, args.batch, dims.hidden, dev)
    layer = ParallelHybridLayer(plan, rw, dims, g, st, ctx, dev)
    inter: dict = {}
    kc, vc, conv, ssm = full_states_for_reference(dims, args.batch, args.ctx, 0, dev)
    ref_out = decode_step(full, dims, h, kc, vc, conv, ssm, args.ctx, intermediates=inter)
    print(f"dims: rope_theta={dims.rope_theta} key_mult={dims.key_mult} attn_in/out={dims.attn_in}/{dims.attn_out} ssm_in/out={dims.ssm_in}/{dims.ssm_out}")

    # ---- kernel path, stage by stage (fresh states identical to the reference's)
    B = args.batch
    x = K.rms_norm(h, rw.in_norm, dims.rms_eps, torch.empty_like(h))
    rep("norm_in", x, inter["x"])
    # attention sub-stages
    a = layer.attn
    xa = x * dims.attn_in
    qkv = K.gemm(xa, rw.wqkv)
    q, k, v = torch.split(qkv, [rw.q_rows, rw.kv_rows, rw.kv_rows], dim=-1)
    k = k * dims.key_mult
    rep("q(pre-rope)", q.view(B, dims.n_q, dims.head_dim), inter["q"] * 0 + _unrope(inter["q"], ctx, dims))
    q2, k2 = q.contiguous(), k.contiguous()
    K.apply_rope(ctx.positions, q2, k2, dims.head_dim, a.rope_cache)
    rep("q(rope)", q2.view(B, dims.n_q, dims.head_dim), inter["q"])
    rep("k(rope)", k2.view(B, dims.n_kv, dims.head_dim), inter["k"])
    attn_k = a.decode(xa, NULL)
    rep("attn_out", attn_k, inter["attn"])
    # manual attention from the kernel's own cache to isolate FA vs cache fill
    o_k = a.out.view(B, dims.n_q * dims.head_dim)
    rep("attn_o", o_k, inter["o"])
    # ssm sub-stages
    s = layer.ssm
    xs = x * dims.ssm_in
    zx = K.gemm(xs, rw.win) * rw.mup
    rep("zxbcdt", zx, inter["zxbcdt"])
    st_fresh = alloc_and_fill(plan, dims, ctx, 0, dev)
    s.st.conv_state.copy_(st_fresh.conv_state)
    s.st.ssm_state.copy_(st_fresh.ssm_state)
    ssm_k = s.decode(xs, NULL)
    rep("ssm_out", ssm_k, inter["ssm"])
    rep("y_ssm(pre-outproj)", s.y.view(B, -1), inter["y_ssm"] * 0 + _pre_norm_y(inter, dims, B))
    h1 = h + attn_k + ssm_k
    rep("h1", h1, inter["h1"])
    y2 = K.rms_norm(h1, rw.ff_norm, dims.rms_eps, torch.empty_like(h1))
    mlp_k = layer.mlp.forward(y2, NULL)
    rep("mlp", mlp_k, inter["mlp"])
    # full step on fresh states
    st_fresh = alloc_and_fill(plan, dims, ctx, 0, dev)
    for n in ("kv", "conv_state", "ssm_state"):
        getattr(st, n).copy_(getattr(st_fresh, n))
    out = layer.step(h, NULL)
    rep("final", out, ref_out)
    destroy()


def _unrope(q_roped: torch.Tensor, ctx, dims) -> torch.Tensor:
    cos, sin = rope_cos_sin(ctx.positions, dims.head_dim, dims.rope_theta)
    return apply_rope_neox(q_roped, cos, -sin)


def _pre_norm_y(inter: dict, dims, B: int) -> torch.Tensor:
    # the reference's y_ssm is post-norm; return it (comparison is only indicative)
    return inter["y_ssm"]


if __name__ == "__main__":
    main()
