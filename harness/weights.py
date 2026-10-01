"""Weight loading (safetensors or random) and per-rank sharding to device."""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import torch

from .config import FalconH1Dims
from .sharding import RankPlan, build_mup_vector

KEYS = (
    "self_attn.q_proj.weight", "self_attn.k_proj.weight", "self_attn.v_proj.weight", "self_attn.o_proj.weight",
    "mamba.in_proj.weight", "mamba.conv1d.weight", "mamba.conv1d.bias", "mamba.A_log", "mamba.D", "mamba.dt_bias",
    "mamba.norm.weight", "mamba.out_proj.weight",
    "feed_forward.gate_proj.weight", "feed_forward.up_proj.weight", "feed_forward.down_proj.weight",
    "input_layernorm.weight", "pre_ff_layernorm.weight",
)


def load_layer_from_safetensors(model_dir: str | Path, layer_idx: int) -> dict[str, torch.Tensor]:
    from safetensors import safe_open

    model_dir = Path(model_dir)
    index = json.loads((model_dir / "model.safetensors.index.json").read_text())["weight_map"]
    prefix = f"model.layers.{layer_idx}."
    wanted = {prefix + k: k for k in KEYS}
    by_file: dict[str, list[str]] = {}
    for full, short in wanted.items():
        if full not in index:
            raise KeyError(f"{full} not in checkpoint index")
        by_file.setdefault(index[full], []).append(full)
    out: dict[str, torch.Tensor] = {}
    for fname, names in by_file.items():
        with safe_open(str(model_dir / fname), framework="pt", device="cpu") as f:
            for n in names:
                out[wanted[n]] = f.get_tensor(n)
    return out


def random_layer(dims: FalconH1Dims, seed: int = 0) -> dict[str, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    H, I, hd = dims.hidden, dims.intermediate, dims.head_dim

    def lin(rows: int, cols: int) -> torch.Tensor:
        return (torch.randn(rows, cols, generator=g) * 0.02).to(torch.bfloat16)

    nh = dims.n_mamba_heads
    w = {
        "self_attn.q_proj.weight": lin(dims.n_q * hd, H),
        "self_attn.k_proj.weight": lin(dims.n_kv * hd, H),
        "self_attn.v_proj.weight": lin(dims.n_kv * hd, H),
        "self_attn.o_proj.weight": lin(H, dims.n_q * hd),
        "mamba.in_proj.weight": lin(dims.in_proj_rows, H),
        "mamba.conv1d.weight": (torch.rand(dims.conv_rows, 1, dims.d_conv, generator=g) - 0.5).to(torch.bfloat16),
        "mamba.conv1d.bias": (torch.rand(dims.conv_rows, generator=g) * 0.1).to(torch.bfloat16),
        "mamba.A_log": torch.log(1 + 15 * torch.rand(nh, generator=g)),
        "mamba.D": torch.ones(nh),
        "mamba.dt_bias": torch.log(torch.expm1(1e-3 + 0.099 * torch.rand(nh, generator=g))),
        "mamba.norm.weight": torch.ones(dims.d_ssm, dtype=torch.bfloat16),
        "mamba.out_proj.weight": lin(H, dims.d_ssm),
        "feed_forward.gate_proj.weight": lin(I, H),
        "feed_forward.up_proj.weight": lin(I, H),
        "feed_forward.down_proj.weight": lin(H, I),
        "input_layernorm.weight": torch.ones(H, dtype=torch.bfloat16),
        "pre_ff_layernorm.weight": torch.ones(H, dtype=torch.bfloat16),
    }
    return w


@dataclass
class RankWeights:
    # attention (None on SSM-only ranks)
    wqkv: torch.Tensor | None  # [q+k+v rows, H]
    wo: torch.Tensor | None  # [H, q cols]
    q_rows: int = 0
    kv_rows: int = 0
    # mamba (None on attention-only ranks)
    win: torch.Tensor | None = None  # [z+x+B+C+dt rows, H]
    conv_w: torch.Tensor | None = None  # [conv_rows_local, d_conv]
    conv_b: torch.Tensor | None = None  # [conv_rows_local]
    A: torch.Tensor | None = None  # [nhr] fp32, = -exp(A_log)
    D: torch.Tensor | None = None  # [nhr] fp32
    dt_bias: torch.Tensor | None = None  # [nhr] fp32
    norm_w: torch.Tensor | None = None  # [nhr*p]
    wout: torch.Tensor | None = None  # [H, nhr*p]
    mup: torch.Tensor | None = None  # [1, in rows] bf16
    # mlp pool, row r of each matrix is global intermediate column col_order[r]
    mlp_gate: torch.Tensor | None = None  # [n_fixed + 2w, H]
    mlp_up: torch.Tensor | None = None  # [n_fixed + 2w, H]
    mlp_downT: torch.Tensor | None = None  # [n_fixed + 2w, H] (down_proj columns, transposed)
    n_fixed: int = 0
    n_pool: int = 0
    # norms
    in_norm: torch.Tensor | None = None
    ff_norm: torch.Tensor | None = None

    def resident_bytes(self) -> dict[str, int]:
        def nb(t: torch.Tensor | None) -> int:
            return 0 if t is None else t.numel() * t.element_size()

        return {
            "attn": nb(self.wqkv) + nb(self.wo),
            "mamba": nb(self.win) + nb(self.conv_w) + nb(self.conv_b) + nb(self.wout) + nb(self.norm_w),
            "mlp_pool": nb(self.mlp_gate) + nb(self.mlp_up) + nb(self.mlp_downT),
        }


def shard_to_device(full: dict[str, torch.Tensor], plan: RankPlan, dims: FalconH1Dims, device: torch.device) -> RankWeights:
    bf = torch.bfloat16
    rw = RankWeights(wqkv=None, wo=None)
    if plan.attn is not None:
        a = plan.attn
        q = full["self_attn.q_proj.weight"][a.q_rows]
        k = full["self_attn.k_proj.weight"][a.k_rows]
        v = full["self_attn.v_proj.weight"][a.v_rows]
        rw.wqkv = torch.cat([q, k, v], 0).to(device, bf).contiguous()
        rw.wo = full["self_attn.o_proj.weight"][:, a.o_cols].to(device, bf).contiguous()
        rw.q_rows, rw.kv_rows = q.shape[0], k.shape[0]
    if plan.ssm is not None:
        s = plan.ssm
        ip = full["mamba.in_proj.weight"]
        rw.win = torch.cat([ip[s.z_rows], ip[s.x_rows], ip[s.b_rows], ip[s.c_rows], ip[s.dt_rows]], 0).to(device, bf).contiguous()
        cw = full["mamba.conv1d.weight"][:, 0, :]
        cb = full["mamba.conv1d.bias"]
        rows = s.conv_rows
        rw.conv_w = torch.cat([cw[rows[0]], cw[rows[1]], cw[rows[2]]], 0).to(device, bf).contiguous()
        rw.conv_b = torch.cat([cb[rows[0]], cb[rows[1]], cb[rows[2]]], 0).to(device, bf).contiguous()
        rw.A = (-torch.exp(full["mamba.A_log"][s.heads].float())).to(device)
        rw.D = full["mamba.D"][s.heads].float().to(device)
        rw.dt_bias = full["mamba.dt_bias"][s.heads].float().to(device)
        rw.norm_w = full["mamba.norm.weight"][s.norm_cols].to(device, bf).contiguous()
        rw.wout = full["mamba.out_proj.weight"][:, s.out_cols].to(device, bf).contiguous()
        rw.mup = build_mup_vector(dims, s).to(device, bf)
    m = plan.mlp
    gp, up, dp = full["feed_forward.gate_proj.weight"], full["feed_forward.up_proj.weight"], full["feed_forward.down_proj.weight"]
    idx = torch.tensor(m.col_order, dtype=torch.long)
    rw.mlp_gate = gp[idx].to(device, bf).contiguous()
    rw.mlp_up = up[idx].to(device, bf).contiguous()
    rw.mlp_downT = dp[:, idx].T.to(device, bf).contiguous()
    rw.n_fixed = m.n_fixed
    rw.n_pool = len(m.col_order)
    rw.in_norm = full["input_layernorm.weight"].to(device, bf).contiguous()
    rw.ff_norm = full["pre_ff_layernorm.weight"].to(device, bf).contiguous()
    return rw


def layer_bytes(dims: FalconH1Dims) -> dict[str, int]:
    """Full-layer bf16 weight bytes by component (for the byte model and sanity checks)."""
    H, I, hd = dims.hidden, dims.intermediate, dims.head_dim
    attn = 2 * (dims.n_q * hd * H + 2 * dims.n_kv * hd * H + H * dims.n_q * hd)
    mamba = 2 * (dims.in_proj_rows * H + H * dims.d_ssm + dims.conv_rows * dims.d_conv + dims.conv_rows + dims.d_ssm)
    mlp = 2 * (3 * I * H)
    return {"attn": attn, "mamba": mamba, "mlp": mlp, "total": attn + mamba + mlp}
