#!/usr/bin/env bash
# Run inside the mmp-dev container: import checks (GPU needed), kernel calls, and the 4-rank group self-test.
set -euo pipefail
cd /workspace
python3 - <<'PY'
import os, torch, vllm
from vllm import _custom_ops as ops
p = os.path.dirname(vllm.__file__)
assert p == "/opt/vllm/vllm", p
print("vllm", vllm.__version__, "at", p, "| torch", torch.__version__, "| cuda", torch.version.cuda, "| gpus", torch.cuda.device_count())
assert hasattr(torch.ops._C, "rms_norm"), "stable-libtorch _C ops not loaded"
import vllm.vllm_flash_attn as fa
assert fa.FA2_AVAILABLE, "FA2 extension missing"
from vllm.model_executor.layers.mamba.ops.mamba_ssm import selective_state_update
from vllm.model_executor.layers.mamba.ops.causal_conv1d import causal_conv1d_update
from vllm.model_executor.layers.mamba.ops.layernorm_gated import rms_norm_gated
from vllm.v1.attention.backends.flash_attn import FlashAttentionBackend
print("kv cache shape (nb=2, bs=16, nkv=1, hd=128):", FlashAttentionBackend.get_kv_cache_shape(2, 16, 1, 128))
import inspect
print("flash_attn_varlen_func:", inspect.signature(fa.flash_attn_varlen_func))
# tiny kernel calls
x = torch.randn(4, 3072, device="cuda", dtype=torch.bfloat16); w = torch.ones(3072, device="cuda", dtype=torch.bfloat16); out = torch.empty_like(x)
ops.rms_norm(out, x, w, 1e-5); print("rms_norm ok", out.shape)
st = torch.zeros(5, 6, 128, 256, device="cuda", dtype=torch.bfloat16)
xs = torch.randn(4, 6, 128, device="cuda", dtype=torch.bfloat16); dt = torch.rand(4, 6, 128, device="cuda", dtype=torch.bfloat16)
A = -torch.ones(6, 128, 256, device="cuda"); B = torch.randn(4, 1, 256, device="cuda", dtype=torch.bfloat16); C = torch.randn(4, 1, 256, device="cuda", dtype=torch.bfloat16)
D = torch.ones(6, 128, device="cuda"); dtb = torch.zeros(6, 128, device="cuda"); idx = torch.arange(4, device="cuda", dtype=torch.int32); y = torch.empty_like(xs)
selective_state_update(st, xs, dt, A, B, C, D, dtb, dt_softplus=True, state_batch_indices=idx, dst_state_batch_indices=idx, out=y); print("selective_state_update ok", y.shape)
cs = torch.zeros(5, 3, 1280, device="cuda", dtype=torch.bfloat16); cw = torch.randn(1280, 4, device="cuda", dtype=torch.bfloat16); cb = torch.zeros(1280, device="cuda", dtype=torch.bfloat16)
xc = torch.randn(4, 1280, device="cuda", dtype=torch.bfloat16)
yc = causal_conv1d_update(xc, cs.transpose(-1, -2), cw, cb, "silu", conv_state_indices=idx); print("causal_conv1d_update ok", yc.shape)
z = torch.randn(4, 768, device="cuda", dtype=torch.bfloat16); yy = torch.randn(4, 768, device="cuda", dtype=torch.bfloat16); nw = torch.ones(768, device="cuda", dtype=torch.bfloat16)
print("rms_norm_gated ok", rms_norm_gated(yy, nw, None, z=z, eps=1e-5, norm_before_gate=False).shape)
PY
torchrun --standalone --nnodes=1 --nproc_per_node=4 -m harness.dist --selftest
