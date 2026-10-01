"""Probe causal_conv1d_update's state ordering and layout handling with a tiny known input."""
import torch, torch.nn.functional as F
from vllm.model_executor.layers.mamba.ops.causal_conv1d import causal_conv1d_update
torch.manual_seed(0)
dim, W, B = 8, 4, 2
w = torch.randn(dim, W, device="cuda", dtype=torch.bfloat16)
b = torch.zeros(dim, device="cuda", dtype=torch.bfloat16)
x = torch.randn(B, dim, device="cuda", dtype=torch.bfloat16)
s = torch.randn(B, W - 1, dim, device="cuda", dtype=torch.bfloat16)  # SD layout (lines, state_len, dim)
idx = torch.arange(B, device="cuda", dtype=torch.int32)

def ref(state_sd, x, oldest_first=True, tap_reverse=False):
    st = state_sd.float()
    if not oldest_first:
        st = st.flip(1)
    win = torch.cat([st, x.float()[:, None]], 1)  # (B, W, dim)
    ww = w.float().T  # (W, dim)
    if tap_reverse:
        ww = ww.flip(0)
    return F.silu((win * ww[None]).sum(1) + b.float())

sT = s.clone().transpose(-1, -2)
out_sd = causal_conv1d_update(x.clone(), sT, w, b, "silu", conv_state_indices=idx)
sDS = s.clone().transpose(-1, -2).contiguous()
out_ds = causal_conv1d_update(x.clone(), sDS, w, b, "silu", conv_state_indices=idx)
for name, o in (("SD-view", out_sd), ("DS-contig", out_ds)):
    for of in (True, False):
        for tr in (False, True):
            r = ref(s, x, of, tr)
            print(f"{name:10s} oldest_first={of!s:5s} tap_reverse={tr!s:5s} max_abs={(o.float() - r).abs().max().item():.4f}")
print("SD vs DS kernel outputs max diff:", (out_sd.float() - out_ds.float()).abs().max().item())
print("state after update (SD view, line0, first 2 dims):", sT[0].float()[:2])
print("original state line0 (dim x state_len):", s[0].float().T[:2])
print("x line0:", x[0].float()[:2])
