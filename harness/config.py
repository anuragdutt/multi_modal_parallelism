"""Model dimensions and run specifications."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

Mode = Literal["tp1", "tp4", "tp4_fused", "tp4_streams", "split22"]
MODES: tuple[str, ...] = ("tp1", "tp4", "tp4_fused", "tp4_streams", "split22")


@dataclass(frozen=True)
class FalconH1Dims:
    hidden: int
    intermediate: int
    n_q: int
    n_kv: int
    head_dim: int
    n_mamba_heads: int
    mamba_head_dim: int
    d_ssm: int
    d_state: int
    n_groups: int
    d_conv: int
    chunk: int
    rms_eps: float
    rope_theta: float
    max_pos: int
    attn_in: float
    attn_out: float
    key_mult: float
    ssm_in: float
    ssm_out: float
    ssm_mults: tuple[float, float, float, float, float]
    mlp_mults: tuple[float, float]

    @property
    def in_proj_rows(self) -> int:
        return 2 * self.d_ssm + 2 * self.n_groups * self.d_state + self.n_mamba_heads

    @property
    def conv_rows(self) -> int:
        return self.d_ssm + 2 * self.n_groups * self.d_state

    @classmethod
    def from_hf(cls, model_dir: str | Path) -> "FalconH1Dims":
        cfg = json.loads((Path(model_dir) / "config.json").read_text())
        hidden = cfg["hidden_size"]
        d_ssm = cfg.get("mamba_d_ssm") or int(cfg.get("mamba_expand", 2) * hidden)
        n_heads = cfg["mamba_n_heads"]
        head_dim = cfg.get("head_dim") or hidden // cfg["num_attention_heads"]
        return cls(
            hidden=hidden,
            intermediate=cfg["intermediate_size"],
            n_q=cfg["num_attention_heads"],
            n_kv=cfg["num_key_value_heads"],
            head_dim=head_dim,
            n_mamba_heads=n_heads,
            mamba_head_dim=cfg.get("mamba_d_head") or d_ssm // n_heads,
            d_ssm=d_ssm,
            d_state=cfg["mamba_d_state"],
            n_groups=cfg.get("mamba_n_groups", 1),
            d_conv=cfg.get("mamba_d_conv", 4),
            chunk=cfg.get("mamba_chunk_size", 256),
            rms_eps=cfg.get("rms_norm_eps", 1e-5),
            rope_theta=cfg.get("rope_theta", 1e4),
            max_pos=cfg.get("max_position_embeddings", 8192),
            attn_in=cfg.get("attention_in_multiplier", 1.0),
            attn_out=cfg.get("attention_out_multiplier", 1.0),
            key_mult=cfg.get("key_multiplier", 1.0),
            ssm_in=cfg.get("ssm_in_multiplier", 1.0),
            ssm_out=cfg.get("ssm_out_multiplier", 1.0),
            ssm_mults=tuple(cfg.get("ssm_multipliers", [1.0] * 5)),
            mlp_mults=tuple(cfg.get("mlp_multipliers", [1.0, 1.0])),
        )

    @classmethod
    def from_yaml(cls, path: str | Path) -> "FalconH1Dims":
        import yaml

        d = yaml.safe_load(Path(path).read_text())
        d["ssm_mults"] = tuple(d["ssm_mults"])
        d["mlp_mults"] = tuple(d["mlp_mults"])
        return cls(**d)


@dataclass(frozen=True)
class RunSpec:
    mode: str
    swing: float = 0.0
    k: int | None = None  # swing split point in [0, 2w]; None -> resolved from cuts_src
    cuts_src: str = "neutral"  # neutral | model | grid | manual
    batch: int = 1
    ctx: int = 512
    variant: Literal["eager", "graph"] = "eager"
    kv_block: int = 16
    n_warm: int = 20
    n_iter: int = 100
    repeats: int = 3
    repeat_offset: int = 0  # sweep.py sets this so outer repeats get distinct 'repeat' values
    layer_idx: int = 0
    seed: int = 0
    tag: str = ""
