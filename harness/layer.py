"""One Falcon-H1 decoder layer assembled from vLLM kernels, with the per-mode order of kernels and
collectives. All buffers are static so a step can be captured into a CUDA graph."""
from __future__ import annotations

import torch
import torch.nn.functional as F

from . import kernels as K
from .config import FalconH1Dims
from .dist import Groups
from .sharding import RankPlan
from .state import DecodeCtx, RankStates
from .timing import OpTimer
from .weights import RankWeights


class AttentionBranch:
    name = "attn"

    def __init__(self, plan: RankPlan, w: RankWeights, dims: FalconH1Dims, st: RankStates, ctx: DecodeCtx, device: torch.device):
        a = plan.attn
        assert a is not None
        self.plan, self.w, self.dims, self.st, self.ctx = plan, w, dims, st, ctx
        self.nq, self.nkv, self.hd = len(a.q_heads), len(a.kv_heads), dims.head_dim
        self.rope_cache = K.rope_cos_sin_cache(self.hd, max(dims.max_pos, ctx.c + 2), dims.rope_theta, device)
        self.out = torch.empty(ctx.B, self.nq, self.hd, device=device, dtype=torch.bfloat16)

    def decode(self, x: torch.Tensor, t: OpTimer) -> torch.Tensor:
        """x: [B, H] already scaled by attn_in. Returns o_proj partial scaled by attn_out."""
        w, ctx = self.w, self.ctx
        t.begin("attn.qkv")
        qkv = K.gemm(x, w.wqkv)
        q, k, v = torch.split(qkv, [w.q_rows, w.kv_rows, w.kv_rows], dim=-1)
        k = k * self.dims.key_mult
        t.end("attn.qkv")
        t.begin("attn.rope")
        q = q.contiguous()
        k = k.contiguous()
        K.apply_rope(ctx.positions, q, k, self.hd, self.rope_cache)
        t.end("attn.rope")
        t.begin("attn.kvwrite")
        K.write_kv(k.view(ctx.B, self.nkv, self.hd), v.contiguous().view(ctx.B, self.nkv, self.hd),
                   self.st.key_cache, self.st.value_cache, ctx.slot_mapping)
        t.end("attn.kvwrite")
        t.begin("attn.fa")
        K.attn_decode(q.view(ctx.B, self.nq, self.hd), self.st.key_cache, self.st.value_cache, ctx.cu_q,
                      ctx.seqused_k, ctx.max_k, ctx.block_table, self.out)
        t.end("attn.fa")
        t.begin("attn.oproj")
        o = K.gemm(self.out.view(ctx.B, self.nq * self.hd), w.wo) * self.dims.attn_out
        t.end("attn.oproj")
        return o


class SsmBranch:
    name = "ssm"

    def __init__(self, plan: RankPlan, w: RankWeights, dims: FalconH1Dims, st: RankStates, ctx: DecodeCtx,
                 device: torch.device, norm_group):
        s = plan.ssm
        assert s is not None
        self.plan, self.w, self.dims, self.st, self.ctx = plan, w, dims, st, ctx
        self.norm_group = norm_group
        z, x, b, c, dt = s.local_in_rows
        self.sizes = (z, x + b + c, dt)
        self.xbc_sizes = (x, b, c)
        self.nhr, self.p, self.N = len(s.heads), dims.mamba_head_dim, dims.d_state
        self.ng = len(s.groups)
        # expanded parameter views, exactly as MambaMixer2 builds them
        self.A_exp = w.A[:, None, None].expand(self.nhr, self.p, self.N)
        self.D_exp = w.D[:, None].expand(self.nhr, self.p)
        self.dtb_exp = w.dt_bias[:, None].expand(self.nhr, self.p)
        self.y = torch.empty(ctx.B, self.nhr, self.p, device=device, dtype=torch.bfloat16)
        self.conv_state_T = st.conv_state.transpose(-1, -2)  # [lines, rows, d_conv-1] view

    def decode(self, x: torch.Tensor, t: OpTimer) -> torch.Tensor:
        w, ctx = self.w, self.ctx
        t.begin("ssm.inproj")
        zxbcdt = K.gemm(x, w.win) * w.mup
        z, xBC, dt = torch.split(zxbcdt, list(self.sizes), dim=-1)
        t.end("ssm.inproj")
        t.begin("ssm.conv")
        xBC = K.conv_update(xBC.contiguous(), self.conv_state_T, w.conv_w, w.conv_b, ctx.state_idx)
        t.end("ssm.conv")
        xs, Bm, Cm = torch.split(xBC, list(self.xbc_sizes), dim=-1)
        t.begin("ssm.ssu")
        K.ssm_update(
            self.st.ssm_state,
            xs.view(ctx.B, self.nhr, self.p),
            dt[:, :, None].expand(ctx.B, self.nhr, self.p),
            self.A_exp,
            Bm.view(ctx.B, self.ng, self.N),
            Cm.view(ctx.B, self.ng, self.N),
            self.D_exp, self.dtb_exp, ctx.state_idx, self.y,
        )
        t.end("ssm.ssu")
        t.begin("ssm.norm")
        y = K.gated_rmsnorm(self.y.view(ctx.B, self.nhr * self.p), z, w.norm_w, self.dims.rms_eps, self.norm_group)
        t.end("ssm.norm")
        t.begin("ssm.outproj")
        o = K.gemm(y, w.wout) * self.dims.ssm_out
        t.end("ssm.outproj")
        return o


class SwingMLP:
    """One contiguous GEMM chain per rank; the per-step assignment only changes the active row count."""

    def __init__(self, plan: RankPlan, w: RankWeights, dims: FalconH1Dims):
        self.plan, self.w, self.dims = plan, w, dims
        self.m = plan.mlp
        self.k = plan.mlp.k
        self.gm, self.dm = dims.mlp_mults

    def set_k(self, k: int) -> None:
        self.k = k

    @property
    def n_active(self) -> int:
        m = self.m
        if m.w == 0:
            return m.n_fixed
        return m.n_fixed + (self.k if m.role == "attn" else 2 * m.w - self.k)

    def forward(self, y: torch.Tensor, t: OpTimer) -> torch.Tensor:
        w, n = self.w, self.n_active
        t.begin("mlp.gateup")
        g = K.gemm(y, w.mlp_gate[:n])
        u = K.gemm(y, w.mlp_up[:n])
        t.end("mlp.gateup")
        t.begin("mlp.act")
        h = F.silu(g * self.gm) * u
        t.end("mlp.act")
        t.begin("mlp.down")
        part = K.gemm_T(h, w.mlp_downT[:n])
        t.end("mlp.down")
        return part * self.dm


class _LocalOnly:
    """Stands in for a GroupCoordinator in the no-collectives diagnostic: keeps the native norm path."""

    def __init__(self, world_size: int):
        self.world_size = world_size

    def all_reduce(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.world_size


class ParallelHybridLayer:
    def __init__(self, plan: RankPlan, w: RankWeights, dims: FalconH1Dims, groups: Groups, st: RankStates,
                 ctx: DecodeCtx, device: torch.device, collectives: bool = True):
        self.plan, self.w, self.dims, self.groups, self.ctx = plan, w, dims, groups, ctx
        self.collectives = collectives
        self.device = device
        self.B, self.H = ctx.B, dims.hidden
        self.attn = AttentionBranch(plan, w, dims, st, ctx, device) if plan.attn is not None else None
        norm_group = groups.group_for_ranks(plan.ssm.norm_ranks) if (plan.ssm is not None and collectives) else None
        if plan.ssm is not None and not collectives and len(plan.ssm.norm_ranks) > 1:
            norm_group = _LocalOnly(len(plan.ssm.norm_ranks))  # diagnostic: same arithmetic path, no reduction
        self.ssm = SsmBranch(plan, w, dims, st, ctx, device, norm_group) if plan.ssm is not None else None
        self.mlp = SwingMLP(plan, w, dims)
        self.xn = torch.empty(ctx.B, dims.hidden, device=device, dtype=torch.bfloat16)
        self.yn = torch.empty(ctx.B, dims.hidden, device=device, dtype=torch.bfloat16)
        self.out = torch.empty(ctx.B, dims.hidden, device=device, dtype=torch.bfloat16)
        self.tp = groups.tp
        self.mode = plan.mode
        if self.mode == "tp4_streams":
            self.s_attn = torch.cuda.Stream(device=device)
            self.s_ssm = torch.cuda.Stream(device=device)

    def _ar(self, x: torch.Tensor, name: str, t: OpTimer) -> torch.Tensor:
        if self.groups.world_size == 1 or not self.collectives:
            return x
        t.begin(name)
        y = self.tp.all_reduce(x)
        t.end(name)
        return y

    def step(self, h: torch.Tensor, t: OpTimer) -> torch.Tensor:
        dims, B = self.dims, self.B
        t.begin("norm_in")
        x = K.rms_norm(h, self.w.in_norm, dims.rms_eps, self.xn)
        t.end("norm_in")
        mode = self.mode
        if mode == "tp1":
            mix = self.attn.decode(x * dims.attn_in, t) + self.ssm.decode(x * dims.ssm_in, t)
        elif mode == "tp4":
            oa = self._ar(self.attn.decode(x * dims.attn_in, t), "ar.attn", t)
            os_ = self._ar(self.ssm.decode(x * dims.ssm_in, t), "ar.ssm", t)
            mix = oa + os_
        elif mode == "tp4_fused":
            part = self.attn.decode(x * dims.attn_in, t) + self.ssm.decode(x * dims.ssm_in, t)
            mix = self._ar(part, "ar.mixer", t)
        elif mode == "tp4_streams":
            cur = torch.cuda.current_stream(self.device)
            self.s_attn.wait_stream(cur)
            self.s_ssm.wait_stream(cur)
            with torch.cuda.stream(self.s_attn):
                oa = self.attn.decode(x * dims.attn_in, t)
            with torch.cuda.stream(self.s_ssm):
                os_ = self.ssm.decode(x * dims.ssm_in, t)
            cur.wait_stream(self.s_attn)
            cur.wait_stream(self.s_ssm)
            mix = self._ar(oa + os_, "ar.mixer", t)
        elif mode == "split22":
            if self.attn is not None:
                part = self.attn.decode(x * dims.attn_in, t)
            else:
                part = self.ssm.decode(x * dims.ssm_in, t)
            mix = self._ar(part, "ar.mixer", t)
        else:
            raise ValueError(mode)
        t.begin("residual")
        h1 = h + mix
        t.end("residual")
        t.begin("norm_ff")
        y = K.rms_norm(h1, self.w.ff_norm, dims.rms_eps, self.yn)
        t.end("norm_ff")
        part = self.mlp.forward(y, t)
        part = self._ar(part, "ar.mlp", t)
        t.begin("residual2")
        self.out.copy_(h1 + part)
        t.end("residual2")
        return self.out
