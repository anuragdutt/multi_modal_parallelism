# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Gate branch for experimental Mamba SSM-Gate parallelism."""

import torch
from torch import nn
from torch.nn.parameter import Parameter

from vllm.config import CacheConfig, ModelConfig, get_current_vllm_config
from vllm.distributed.parallel_state import (
    get_mamba_sg_branch_group,
    get_mamba_sg_pair_group,
)
from vllm.forward_context import ForwardContext, get_forward_context
from vllm.model_executor.custom_op import PluggableLayer
from vllm.model_executor.utils import set_weight_attrs
from vllm.utils.torch_utils import (
    LayerNameType,
    _encode_layer_name,
    _resolve_layer_name,
    direct_register_custom_op,
)


@PluggableLayer.register("mamba_mixerGate")
class MambaMixerGate(PluggableLayer):
    """Compute the raw gate projection and exchange it with an SSM rank."""

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
        activation: str = "silu",
        is_lora_enabled: bool = False,
        model_config: ModelConfig | None = None,
        cache_config: CacheConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        if activation != "silu":
            raise ValueError("Mamba SG currently supports only silu activation")
        if is_lora_enabled:
            raise NotImplementedError("Mamba SG does not support LoRA")

        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
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
        self.in_proj = nn.Linear(
            hidden_size, self.local_intermediate_size, bias=use_bias
        )
        set_weight_attrs(self.in_proj.weight, {"weight_loader": self._load_weight})
        if self.in_proj.bias is not None:
            set_weight_attrs(self.in_proj.bias, {"weight_loader": self._load_bias})

        self.layer_name = f"{prefix}.gate_path" if prefix else "gate_path"
        compilation_config = get_current_vllm_config().compilation_config
        if self.layer_name in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate layer name: {self.layer_name}")
        compilation_config.static_forward_context[self.layer_name] = self

    def _load_weight(self, param: Parameter, loaded_weight: torch.Tensor) -> None:
        expected = (self.intermediate_size * 2, self.hidden_size)
        if tuple(loaded_weight.shape) != expected:
            raise ValueError(
                f"Expected fused in_proj.weight shape {expected}, got "
                f"{tuple(loaded_weight.shape)}"
            )
        offset = self.intermediate_size + self.branch_rank * self.local_intermediate_size
        param.data.copy_(loaded_weight.narrow(0, offset, self.local_intermediate_size))

    def _load_bias(self, param: Parameter, loaded_weight: torch.Tensor) -> None:
        expected = self.intermediate_size * 2
        if loaded_weight.ndim != 1 or loaded_weight.shape[0] != expected:
            raise ValueError(
                f"Expected fused in_proj.bias length {expected}, got "
                f"{tuple(loaded_weight.shape)}"
            )
        offset = self.intermediate_size + self.branch_rank * self.local_intermediate_size
        param.data.copy_(loaded_weight.narrow(0, offset, self.local_intermediate_size))

    def forward(self, hidden_states: torch.Tensor, output: torch.Tensor) -> None:
        torch.ops.vllm.mamba_mixer_gate(
            hidden_states, output, _encode_layer_name(self.layer_name)
        )

    def forward_impl(self, hidden_states: torch.Tensor, output: torch.Tensor) -> None:
        # Keep the channel-first layout used by the Mamba kernels. The paired
        # SSM rank owns the recurrence and applies this raw projection as z.
        gate = self.in_proj(hidden_states).transpose(-2, -1).contiguous()
        self.pair_group.send(gate, dst=1)
        self.pair_group.broadcast(output, src=1)


def mamba_mixer_gate(
    hidden_states: torch.Tensor,
    output: torch.Tensor,
    layer_name: LayerNameType,
) -> None:
    layer_name = _resolve_layer_name(layer_name)
    forward_context: ForwardContext = get_forward_context()
    layer = forward_context.no_compile_layers[layer_name]
    if not isinstance(layer, MambaMixerGate):
        raise TypeError(
            f"Expected MambaMixerGate for {layer_name!r}, got {type(layer).__name__}"
        )
    layer.forward_impl(hidden_states=hidden_states, output=output)


def mamba_mixer_gate_fake(
    hidden_states: torch.Tensor,
    output: torch.Tensor,
    layer_name: LayerNameType,
) -> None:
    return


direct_register_custom_op(
    op_name="mamba_mixer_gate",
    op_func=mamba_mixer_gate,
    mutates_args=["output"],
    fake_impl=mamba_mixer_gate_fake,
)
