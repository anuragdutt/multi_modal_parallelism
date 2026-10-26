# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import NamedTuple

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.parameter import Parameter

from vllm.config import CacheConfig, ModelConfig, get_current_vllm_config
from vllm.distributed.parallel_state import (
    get_mamba_sg_branch_group,
    get_mamba_sg_pair_group,
)
from vllm.forward_context import ForwardContext, get_forward_context
from vllm.model_executor.custom_op import PluggableLayer
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.mamba.abstract import MambaBase
from vllm.model_executor.layers.mamba.mamba_utils import (
    MambaStateDtypeCalculator,
    MambaStateShapeCalculator,
    is_conv_state_dim_first,
)
from vllm.model_executor.layers.mamba.ops.causal_conv1d import (
    causal_conv1d_fn,
    causal_conv1d_update,
)
from vllm.model_executor.layers.mamba.ops.mamba_ssm import selective_scan_fn
from vllm.model_executor.layers.mamba.ops.ssu_dispatch import selective_state_update
from vllm.model_executor.utils import set_weight_attrs
from vllm.platforms import current_platform
from vllm.utils.torch_utils import (
    LayerNameType,
    _encode_layer_name,
    _resolve_layer_name,
    direct_register_custom_op,
)
from vllm.v1.attention.backend import AttentionMetadata
from vllm.v1.attention.backends.mamba1_attn import Mamba1AttentionMetadata
from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum


# Adapted from transformers.models.mamba.modeling_mamba.MambaMixer
# --8<-- [start:mamba_mixer]
@PluggableLayer.register("mamba_mixerSSM")
class MambaMixerSSM(MambaBase, PluggableLayer):
    """
    Compute ∆, A, B, C, and D the state space parameters and compute
    the `contextualized_states`. A, D are input independent
    (see Mamba paper [1] Section 3.5.2 "Interpretation of A"
    for why A isn't selective) ∆, B, C are input-dependent
    (this is a key difference between Mamba and the linear time
    invariant S4, and is why Mamba is called
    **selective** state spaces)
    """

    # --8<-- [end:mamba_mixer]

    def __init__(
        self,
        hidden_size: int,
        ssm_state_size: int,
        conv_kernel_size: int,
        intermediate_size: int,
        time_step_rank: int,
        use_conv_bias: bool,
        use_bias: bool,
        use_rms_norm: bool,
        rms_norm_has_weight: bool = True,
        rms_norm_eps: float = 1e-5,
        activation="silu",
        is_lora_enabled: bool = False,
        model_config: ModelConfig | None = None,
        cache_config: CacheConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()
        if activation != "silu":
            raise ValueError("Mamba SG currently supports only silu activation")
        if is_lora_enabled:
            raise NotImplementedError("Mamba SG does not support LoRA")

        self.time_step_rank = time_step_rank
        self.ssm_state_size = ssm_state_size
        self.use_rms_norm = use_rms_norm
        self.activation = activation
        self.is_lora_enabled = is_lora_enabled
        self.conv_kernel_size = conv_kernel_size
        self.intermediate_size = intermediate_size
        self.hidden_size = hidden_size
        self.branch_group = get_mamba_sg_branch_group()
        self.pair_group = get_mamba_sg_pair_group()
        self.branch_rank = self.branch_group.rank_in_group
        self.branch_size = self.branch_group.world_size
        if intermediate_size % self.branch_size:
            raise ValueError(
                f"intermediate_size={intermediate_size} is not divisible by "
                f"Mamba SG branch size {self.branch_size}"
            )
        self.local_intermediate_size = intermediate_size // self.branch_size

        # These are branch-local tensor-parallel shards. They intentionally do
        # not use vLLM's global TP linear layers, because gate ranks execute a
        # different collective sequence.
        self.conv1d = nn.Linear(
            conv_kernel_size, self.local_intermediate_size, bias=use_conv_bias
        )
        self.conv1d.weight.data = self.conv1d.weight.data.unsqueeze(1)
        self.in_proj = nn.Linear(hidden_size, self.local_intermediate_size, bias=use_bias)
        self.x_proj = nn.Linear(
            self.local_intermediate_size,
            time_step_rank + ssm_state_size * 2,
            bias=False,
        )
        self.dt_proj = nn.Linear(time_step_rank, self.local_intermediate_size, bias=True)
        self.A = nn.Parameter(
            torch.empty(self.local_intermediate_size, ssm_state_size, dtype=torch.float32)
        )
        self.D = nn.Parameter(torch.ones(self.local_intermediate_size))
        self.out_proj = nn.Linear(self.local_intermediate_size, hidden_size, bias=use_bias)

        self._install_weight_loaders()

        self.dt_layernorm = (
            RMSNorm(time_step_rank, eps=rms_norm_eps, has_weight=rms_norm_has_weight)
            if use_rms_norm
            else None
        )
        self.b_layernorm = (
            RMSNorm(ssm_state_size, eps=rms_norm_eps, has_weight=rms_norm_has_weight)
            if use_rms_norm
            else None
        )
        self.c_layernorm = (
            RMSNorm(ssm_state_size, eps=rms_norm_eps, has_weight=rms_norm_has_weight)
            if use_rms_norm
            else None
        )

        compilation_config = get_current_vllm_config().compilation_config
        if prefix in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        compilation_config.static_forward_context[prefix] = self
        self.kv_cache = (torch.tensor([]), torch.tensor([]))
        self.model_config = model_config
        self.cache_config = cache_config
        self.prefix = prefix

    def _local_offset(self) -> int:
        return self.branch_rank * self.local_intermediate_size

    @staticmethod
    def _set_loader(param: Parameter | None, loader) -> None:
        if param is not None:
            set_weight_attrs(param, {"weight_loader": loader})

    def _install_weight_loaders(self) -> None:
        offset = self._local_offset()
        local = self.local_intermediate_size
        intermediate = self.intermediate_size

        def copy_exact(param: Parameter, value: torch.Tensor, name: str) -> None:
            if tuple(param.shape) != tuple(value.shape):
                raise ValueError(
                    f"Cannot load {name}: expected {tuple(param.shape)}, "
                    f"got {tuple(value.shape)}"
                )
            param.data.copy_(value)

        def conv_weight(param: Parameter, loaded: torch.Tensor) -> None:
            copy_exact(param, loaded.narrow(0, offset, local), "conv1d.weight")

        def row_vector(name: str):
            def loader(param: Parameter, loaded: torch.Tensor) -> None:
                copy_exact(param, loaded.narrow(0, offset, local), name)
            return loader

        def in_weight(param: Parameter, loaded: torch.Tensor) -> None:
            if loaded.shape[0] != intermediate * 2:
                raise ValueError("Expected fused Mamba in_proj weight with 2I rows")
            copy_exact(param, loaded.narrow(0, offset, local), "SSM in_proj.weight")

        def in_bias(param: Parameter, loaded: torch.Tensor) -> None:
            if loaded.shape[0] != intermediate * 2:
                raise ValueError("Expected fused Mamba in_proj bias with 2I entries")
            copy_exact(param, loaded.narrow(0, offset, local), "SSM in_proj.bias")

        def x_weight(param: Parameter, loaded: torch.Tensor) -> None:
            copy_exact(param, loaded.narrow(1, offset, local), "x_proj.weight")

        def out_weight(param: Parameter, loaded: torch.Tensor) -> None:
            copy_exact(param, loaded.narrow(1, offset, local), "out_proj.weight")

        def full_bias(name: str):
            def loader(param: Parameter, loaded: torch.Tensor) -> None:
                copy_exact(param, loaded, name)
            return loader

        def a_weight(param: Parameter, loaded: torch.Tensor) -> None:
            shard = loaded.narrow(0, offset, local)
            copy_exact(param, -torch.exp(shard.float()), "A_log")

        self._set_loader(self.conv1d.weight, conv_weight)
        self._set_loader(self.conv1d.bias, row_vector("conv1d.bias"))
        self._set_loader(self.in_proj.weight, in_weight)
        self._set_loader(self.in_proj.bias, in_bias)
        self._set_loader(self.x_proj.weight, x_weight)
        self._set_loader(self.dt_proj.weight, row_vector("dt_proj.weight"))
        self._set_loader(self.dt_proj.bias, row_vector("dt_proj.bias"))
        self._set_loader(self.A, a_weight)
        self._set_loader(self.D, row_vector("D"))
        self._set_loader(self.out_proj.weight, out_weight)
        self._set_loader(self.out_proj.bias, full_bias("out_proj.bias"))

    def _row_parallel_output(self, inputs: torch.Tensor) -> torch.Tensor:
        bias = self.out_proj.bias if self.branch_rank == 0 else None
        local_output = F.linear(inputs, self.out_proj.weight, bias)
        return self.branch_group.all_reduce(local_output)

    def _ssm_transform(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # LoRA kernel requires contiguous tensor.
        # ROCm: Non-contiguous tensors cause incorrect GEMM
        # results when batch > 1.
        if self.is_lora_enabled or current_platform.is_rocm():
            x = x.contiguous()
        ssm_params = self.branch_group.all_reduce(self.x_proj(x))
        time_step, B, C = torch.split(
            ssm_params,
            [self.time_step_rank, self.ssm_state_size, self.ssm_state_size],
            dim=-1,
        )
        if self.use_rms_norm:
            assert self.dt_layernorm is not None
            assert self.b_layernorm is not None
            assert self.c_layernorm is not None
            time_step = self.dt_layernorm(time_step.contiguous())
            B = self.b_layernorm(B.contiguous())
            C = self.c_layernorm(C.contiguous())

        # ROCm: tensor from split is non-contiguous, causing incorrect
        # GEMM results in dt_proj.
        if current_platform.is_rocm():
            time_step = time_step.contiguous()

        # Match ColumnParallelLinear(skip_bias_add=True): the recurrent
        # kernels add dt_proj.bias exactly once via _time_proj_bias().
        discrete_time_step = F.linear(
            time_step, self.dt_proj.weight, None
        ).transpose(-2, -1)
        return discrete_time_step, B, C

    def forward(self, hidden_states: torch.Tensor, output: torch.Tensor):
        torch.ops.vllm.mamba_mixer_ssm(
            hidden_states,
            output,
            _encode_layer_name(self.prefix),
        )

    def forward_impl(self, hidden_states: torch.Tensor, output: torch.Tensor):
        """
        Run the Mamba-1 SSM pipeline.

        Steps
        -----
        1. Apply the gated-MLP linear projection to the raw input.
        2. Pass the projected sequence through the convolutional mixing layer.
        3. Feed the result into the State-Space Model (SSM) blocks.
        4. Perform the recurrence y ← SSM(A, B, C, Δ)(x)
           to produce contextual representations.
        5. Project the contextualised sequence back
           to the output embedding dimension.

        Batch handling
        --------------
        Prefill and decode tokens are processed by dedicated CUDA
        kernels for both the convolutional (conv1d) and SSM stages.
        In the case of a mixed batch (containing both prefill and
        decode tokens), both sets of kernels are executed independently
        and their outputs are concatenated before the final output projection.
        """

        forward_context: ForwardContext = get_forward_context()
        attn_metadata_raw = forward_context.attn_metadata

        assert self.cache_config is not None
        mamba_block_size = self.cache_config.mamba_block_size
        is_mamba_cache_all = self.cache_config.mamba_cache_mode == "all"

        attn_metadata: AttentionMetadata | None = None
        if attn_metadata_raw is not None:
            assert isinstance(attn_metadata_raw, dict)
            attn_metadata = attn_metadata_raw[self.prefix]
            assert isinstance(attn_metadata, Mamba1AttentionMetadata)
            query_start_loc_p = attn_metadata.query_start_loc_p
            state_indices_tensor_p = attn_metadata.state_indices_tensor_p
            state_indices_tensor_d = attn_metadata.state_indices_tensor_d
            conv_state = (
                self.kv_cache[0]
                if is_conv_state_dim_first()
                else self.kv_cache[0].transpose(-1, -2)
            )
            ssm_state = self.kv_cache[1]
            has_initial_states_p = attn_metadata.has_initial_states_p
            cu_chunk_seqlen_p = attn_metadata.cu_chunk_seqlen_p
            last_chunk_indices_p = attn_metadata.last_chunk_indices_p

        # 1. Gated MLP's linear projection
        hidden_states_BC = self.in_proj(hidden_states).transpose(-2, -1)
        gate = self.pair_group.recv(
            hidden_states_BC.size(), hidden_states_BC.dtype, src=0
        )

        conv_weights = self.conv1d.weight.view(
            self.conv1d.weight.size(0), self.conv1d.weight.size(2)
        )

        if attn_metadata is None:
            # V1 profile run. Keep the communication sequence identical on
            # both roles even though the recurrent kernels are skipped.
            hidden_states_BC = hidden_states_BC.contiguous()
            profiled_output = self._row_parallel_output(
                hidden_states_BC.transpose(-2, -1)
            )
            output.copy_(profiled_output)
            self.pair_group.broadcast(output, src=1)
            return

        num_prefill_tokens = attn_metadata.num_prefill_tokens  # token count
        num_decode_tokens = attn_metadata.num_decode_tokens
        num_prefills = attn_metadata.num_prefills  # request count
        num_decodes = attn_metadata.num_decode_tokens  # token count (=request)
        has_prefill = num_prefill_tokens > 0
        has_decode = num_decode_tokens > 0
        num_actual_tokens = num_prefill_tokens + num_decode_tokens

        prefill_decode_split = split_batch_to_prefill_and_decode(
            hidden_states_BC,
            num_prefill_tokens,
            num_decode_tokens,
        )
        hidden_states_BC_p = prefill_decode_split.hidden_states_BC_p
        hidden_states_BC_d = prefill_decode_split.hidden_states_BC_d
        gate_d, gate_p = torch.split(
            gate[..., :num_actual_tokens],
            [num_decode_tokens, num_prefill_tokens],
            dim=-1,
        )

        if is_mamba_cache_all:
            block_idx_last_computed_token_d, block_idx_last_computed_token_p = (
                torch.split(
                    attn_metadata.block_idx_last_computed_token,
                    [num_decodes, num_prefills],
                    dim=0,
                )
            )
            block_idx_last_scheduled_token_d, block_idx_last_scheduled_token_p = (
                torch.split(
                    attn_metadata.block_idx_last_scheduled_token,
                    [num_decodes, num_prefills],
                    dim=0,
                )
            )

            block_idx_first_scheduled_token_p = (
                attn_metadata.block_idx_first_scheduled_token_p
            )
            num_computed_tokens_p = attn_metadata.num_computed_tokens_p
        else:
            block_idx_last_computed_token_d = None
            block_idx_last_computed_token_p = None
            block_idx_last_scheduled_token_d = None
            block_idx_last_scheduled_token_p = None
            block_idx_first_scheduled_token_p = None
            num_computed_tokens_p = None

        ssm_outputs = []

        if has_prefill:
            # 2. Convolution sequence transformation
            conv_out_p = causal_conv1d_fn(
                hidden_states_BC_p,
                conv_weights,
                self.conv1d.bias,
                activation=self.activation,
                conv_states=conv_state,
                has_initial_state=has_initial_states_p,
                cache_indices=state_indices_tensor_p,
                query_start_loc=query_start_loc_p,
                block_idx_first_scheduled_token=block_idx_first_scheduled_token_p,
                block_idx_last_scheduled_token=block_idx_last_scheduled_token_p,
                initial_state_idx=block_idx_last_computed_token_p,
                num_computed_tokens=num_computed_tokens_p,
                block_size_to_align=mamba_block_size,
                metadata=attn_metadata,
            )
            # 3. State Space Model sequence transformations.
            discrete_time_step_p, B_p, C_p = self._ssm_transform(
                conv_out_p.transpose(-2, -1)
            )
            time_proj_bias = self._time_proj_bias()

            # 4. Perform the recurrence y ← SSM(A, B, C, Δ)(x)
            scan_out_p = selective_scan_fn(
                conv_out_p,
                ssm_state,
                discrete_time_step_p,
                self.A,
                B_p.transpose(-2, -1),
                C_p.transpose(-2, -1),
                self.D.float(),
                gate_p,
                time_proj_bias,
                delta_softplus=True,
                cache_indices=state_indices_tensor_p,
                has_initial_state=has_initial_states_p,
                query_start_loc=query_start_loc_p,
                block_size=mamba_block_size,
                block_idx_first_scheduled_token=block_idx_first_scheduled_token_p,
                block_idx_last_scheduled_token=block_idx_last_scheduled_token_p,
                initial_state_idx=block_idx_last_computed_token_p,
                cu_chunk_seqlen=cu_chunk_seqlen_p,
                last_chunk_indices=last_chunk_indices_p,
            )
            ssm_outputs.append(scan_out_p)

        if has_decode:
            # state_indices_tensor_d is assigned when attn_metadata is not None,
            # and has_decode is only True when attn_metadata is not None
            assert state_indices_tensor_d is not None
            if is_mamba_cache_all:
                state_indices_tensor_d_input = state_indices_tensor_d.gather(
                    1, block_idx_last_computed_token_d.unsqueeze(1)
                ).squeeze(1)
                state_indices_tensor_d_output = state_indices_tensor_d.gather(
                    1, block_idx_last_scheduled_token_d.unsqueeze(1)
                ).squeeze(1)
            else:
                state_indices_tensor_d_input = state_indices_tensor_d
                state_indices_tensor_d_output = state_indices_tensor_d
            # 2. Convolution sequence transformation
            conv_out_d = causal_conv1d_update(
                hidden_states_BC_d.transpose(0, 1),
                conv_state,
                conv_weights,
                self.conv1d.bias,
                self.activation,
                conv_state_indices=state_indices_tensor_d,
                block_idx_last_scheduled_token=block_idx_last_scheduled_token_d,
                initial_state_idx=block_idx_last_computed_token_d,
            ).transpose(0, 1)

            # 3. State Space Model sequence transformation.
            discrete_time_step_d, B_d, C_d = self._ssm_transform(
                conv_out_d.transpose(-2, -1)
            )
            time_proj_bias = self._time_proj_bias()

            # 4. Perform the recurrence y ← SSM(A, B, C, Δ)(x)
            scan_outputs_d = torch.empty_like(hidden_states_BC_d.transpose(0, 1))
            selective_state_update(
                ssm_state,
                conv_out_d.transpose(0, 1),
                discrete_time_step_d.transpose(0, 1),
                self.A,
                B_d,
                C_d,
                self.D,
                time_proj_bias,
                z=gate_d.transpose(0, 1),
                dt_softplus=True,
                state_batch_indices=state_indices_tensor_d_input,
                dst_state_batch_indices=state_indices_tensor_d_output,
                out=scan_outputs_d,
            )
            scan_outputs_d = scan_outputs_d.transpose(0, 1)

            ssm_outputs.insert(0, scan_outputs_d)

        scan_outputs_combined = (
            ssm_outputs[0] if len(ssm_outputs) == 1 else torch.cat(ssm_outputs, dim=-1)
        )

        # 5. The SSM branch owns the shared output projection. Reduce its
        # column shards within the SSM group, then return the complete hidden
        # states to the paired gate rank for the next layer.
        projected = scan_outputs_combined.transpose(-2, -1)
        out = self._row_parallel_output(projected)
        output[:num_actual_tokens] = out
        self.pair_group.broadcast(output, src=1)

    def get_state_dtype(self) -> tuple[torch.dtype]:
        assert self.model_config is not None
        assert self.cache_config is not None
        return MambaStateDtypeCalculator.mamba1_state_dtype(
            self.model_config.dtype,
            self.cache_config.mamba_cache_dtype,
            self.cache_config.mamba_ssm_cache_dtype,
        )

    def get_state_shape(self) -> tuple[tuple[int, ...], tuple[int, ...]]:
        return MambaStateShapeCalculator.mamba1_state_shape(
            tp_world_size=self.branch_size,
            intermediate_size=self.intermediate_size,
            state_size=self.ssm_state_size,
            conv_kernel=self.conv_kernel_size,
        )

    @property
    def mamba_type(self) -> MambaAttentionBackendEnum:
        return MambaAttentionBackendEnum.MAMBA1

    def _time_proj_bias(self) -> torch.Tensor | None:
        if hasattr(self.dt_proj, "bias") and self.dt_proj.bias is not None:
            return self.dt_proj.bias.float()
        return None


class PrefillDecodeSplit(NamedTuple):
    hidden_states_BC_p: torch.Tensor
    hidden_states_BC_d: torch.Tensor


def split_batch_to_prefill_and_decode(
    hidden_states_BC: torch.Tensor,
    num_prefill_tokens: int,
    num_decode_tokens: int,
) -> PrefillDecodeSplit:
    num_actual_tokens = num_prefill_tokens + num_decode_tokens

    # In v1, decode tokens come first, then prefill tokens.
    hidden_states_BC_d, hidden_states_BC_p = torch.split(
        hidden_states_BC[..., :num_actual_tokens],
        [num_decode_tokens, num_prefill_tokens],
        dim=-1,
    )

    return PrefillDecodeSplit(
        hidden_states_BC_p=hidden_states_BC_p,
        hidden_states_BC_d=hidden_states_BC_d
    )


def mamba_mixer_ssm(
    hidden_states: torch.Tensor,
    output: torch.Tensor,
    layer_name: LayerNameType,
) -> None:
    layer_name = _resolve_layer_name(layer_name)
    forward_context: ForwardContext = get_forward_context()
    self = forward_context.no_compile_layers[layer_name]
    self.forward_impl(hidden_states=hidden_states, output=output)


def mamba_mixer_ssm_fake(
    hidden_states: torch.Tensor,
    output: torch.Tensor,
    layer_name: LayerNameType,
) -> None:
    return


direct_register_custom_op(
    op_name="mamba_mixer_ssm",
    op_func=mamba_mixer_ssm,
    mutates_args=["output"],
    fake_impl=mamba_mixer_ssm_fake,
)
