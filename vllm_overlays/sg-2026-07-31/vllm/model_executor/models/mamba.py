# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""PyTorch MAMBA model."""

import os
from collections.abc import Iterable
from itertools import islice

import torch
from torch import nn
from transformers import MambaConfig

from vllm.compilation.decorators import support_torch_compile
from vllm.config import CacheConfig, ModelConfig, VllmConfig
from vllm.distributed.parallel_state import get_pp_group, get_tp_group
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.mamba.mamba_mixer import MambaMixer
from vllm.model_executor.layers.mamba.mamba_mixer_gate import MambaMixerGate
from vllm.model_executor.layers.mamba.mamba_mixer_ssm import MambaMixerSSM
from vllm.model_executor.layers.mamba.mamba_utils import (
    MambaStateCopyFunc,
    MambaStateCopyFuncCalculator,
    MambaStateDtypeCalculator,
    MambaStateShapeCalculator,
)
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.interfaces import (
    HasInnerState,
    IsAttentionFree,
    SupportsMambaPrefixCaching,
    SupportsPP,
)
from vllm.sequence import IntermediateTensors

from .utils import (
    AutoWeightsLoader,
    is_pp_missing_parameter,
    make_empty_intermediate_tensors_factory,
    make_layers,
    maybe_prefix,
)

from vllm.distributed.mamba_sg_parallel_state import (
    MambaSGRole,
    get_mamba_sg_role,
)

KVCache = tuple[torch.Tensor, torch.Tensor]


def _print_mamba_parallel_config(
    vllm_config: VllmConfig, *, location: str
) -> None:
    """Print Mamba SG settings from the process that builds the model."""
    parallel_config = vllm_config.parallel_config
    rank = os.environ.get("RANK", os.environ.get("LOCAL_RANK", "unknown"))
    print(
        "[Mamba SG] ",
        f"location={location}, rank={rank}, ",
        f"enabled={parallel_config.enable_mamba_sg}, ",
        f"tensor_parallel_size={parallel_config.tensor_parallel_size}, ",
        flush=True,
    )


class MambaDecoderLayer(nn.Module):
    def __init__(
        self,
        config: MambaConfig,
        model_config: ModelConfig | None = None,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        is_lora_enabled: bool | None = False,
        enable_mamba_sg: bool | None = False,
        mamba_sg_role: MambaSGRole| None = None,
        # use_gate_path: bool = False,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.enable_mamba_sg = enable_mamba_sg
        self.is_falcon_mamba = config.model_type == "falcon_mamba"
        self.is_lora_enabled = is_lora_enabled
        mixer_rms_eps = config.mixer_rms_eps if self.is_falcon_mamba else None
        mixer_cls = None

        if not enable_mamba_sg:
            mixer_cls = MambaMixer
        elif mamba_sg_role == MambaSGRole.GATE:
            mixer_cls = MambaMixerGate
        else:
            mixer_cls = MambaMixerSSM

        self.mixer = mixer_cls(
            hidden_size=config.hidden_size,
            ssm_state_size=config.state_size,
            conv_kernel_size=config.conv_kernel,
            intermediate_size=config.intermediate_size,
            time_step_rank=config.time_step_rank,
            use_conv_bias=config.use_conv_bias,
            use_bias=config.use_bias,
            use_rms_norm=self.is_falcon_mamba,
            rms_norm_has_weight=not self.is_falcon_mamba,
            rms_norm_eps=mixer_rms_eps,
            activation=config.hidden_act,
            is_lora_enabled=self.is_lora_enabled,
            model_config=model_config,
            cache_config=cache_config,
            prefix=f"{prefix}.mixer",
        )

        self.norm = RMSNorm(config.hidden_size, eps=config.layer_norm_epsilon)

    def forward(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
        **kwargs,
    ):
        if residual is None:
            residual = hidden_states
            hidden_states = self.norm(hidden_states)
        else:
            hidden_states, residual = self.norm(hidden_states, residual)

        output = torch.empty_like(hidden_states)
        self.mixer(hidden_states, output)
        return output, residual


@support_torch_compile
class MambaModel(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()

        parallel_config = vllm_config.parallel_config
        config = vllm_config.model_config.hf_config
        model_config = vllm_config.model_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config
        lora_config = vllm_config.lora_config
        is_lora_enabled = bool(lora_config)

        self.enable_mamba_sg = parallel_config.enable_mamba_sg
        if self.enable_mamba_sg:
            branch_size = parallel_config.tensor_parallel_size // 2
            gate_size = branch_size
            ssm_size = branch_size
        else:
            gate_size = parallel_config.tensor_parallel_size
            ssm_size = parallel_config.tensor_parallel_size

        self.config = config
        
        self.vocab_size = config.vocab_size

        self.embeddings = VocabParallelEmbedding(
            self.vocab_size,
            config.hidden_size,
        )

        self.mamba_sg_role = None
        if self.enable_mamba_sg:
            self.mamba_sg_role = get_mamba_sg_role(
                gate_parallel_size=gate_size,
                ssm_parallel_size=ssm_size,
            )

        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix: MambaDecoderLayer(
                config,
                model_config=model_config,
                cache_config=cache_config,
                quant_config=quant_config,
                is_lora_enabled=is_lora_enabled,
                enable_mamba_sg=self.enable_mamba_sg,
                mamba_sg_role=self.mamba_sg_role,
                prefix=prefix,
            ),
            prefix=f"{prefix}.layers",
        )

        print(f"This is model {type(self.start_layer).__name__}.")

        self.norm_f = RMSNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
            ["hidden_states", "residual"], config.hidden_size
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embeddings(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                hidden_states = self.embed_input_ids(input_ids)
            residual = None
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]

        for layer in islice(self.layers, self.start_layer, self.end_layer):
            hidden_states, residual = layer(
                positions=positions, hidden_states=hidden_states, residual=residual
            )
        if not get_pp_group().is_last_rank:
            return IntermediateTensors(
                {"hidden_states": hidden_states, "residual": residual}
            )
        hidden_states, _ = self.norm_f(hidden_states, residual)

        return hidden_states

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()
        for name, loaded_weight in weights:
            if "A_log" in name:
                name = name.replace("A_log", "A")
            # Skip loading extra bias for GPTQ models.
            if name.endswith(".bias") and name not in params_dict:
                continue
            if is_pp_missing_parameter(name, self):
                continue
            if name not in params_dict:
                if self.enable_mamba_sg:
                    # Gate-only layers intentionally omit all convolution,
                    # SSM, and recurrent-state parameters.
                    continue
                raise KeyError(name)

            param = params_dict[name]
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, loaded_weight)
            loaded_params.add(name)
        return loaded_params


class MambaForCausalLM(
    nn.Module, HasInnerState, IsAttentionFree, SupportsPP, SupportsMambaPrefixCaching
):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        config = vllm_config.model_config.hf_config

        _print_mamba_parallel_config(vllm_config, location="model_init")

        self.scheduler_config = vllm_config.scheduler_config

        super().__init__()
        self.config = config
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        self.backbone = MambaModel(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "backbone")
        )

        if config.tie_word_embeddings:
            self.lm_head = self.backbone.embeddings
        else:
            self.lm_head = ParallelLMHead(
                config.vocab_size,
                config.hidden_size,
                prefix=maybe_prefix(prefix, "lm_head"),
            )

        self.logits_processor = LogitsProcessor(config.vocab_size)

        self.make_empty_intermediate_tensors = (
            self.backbone.make_empty_intermediate_tensors
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.backbone.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs,
    ):
        hidden_states = self.backbone(
            input_ids, positions, intermediate_tensors, inputs_embeds
        )
        return hidden_states

    @classmethod
    def get_mamba_state_dtype_from_config(
        cls,
        vllm_config: "VllmConfig",
    ) -> tuple[torch.dtype, torch.dtype]:
        return MambaStateDtypeCalculator.mamba1_state_dtype(
            vllm_config.model_config.dtype,
            vllm_config.cache_config.mamba_cache_dtype,
            vllm_config.cache_config.mamba_ssm_cache_dtype,
        )

    @classmethod
    def get_mamba_state_shape_from_config(
        cls,
        vllm_config: "VllmConfig",
    ) -> tuple[tuple[int, int], tuple[int, int]]:
        parallel_config = vllm_config.parallel_config
        hf_config = vllm_config.model_config.hf_config

        _print_mamba_parallel_config(vllm_config, location="state_shape")

        return MambaStateShapeCalculator.mamba1_state_shape(
            tp_world_size=(
                parallel_config.tensor_parallel_size // 2
                if parallel_config.enable_mamba_sg
                else parallel_config.tensor_parallel_size
            ),
            intermediate_size=hf_config.intermediate_size,
            state_size=hf_config.state_size,
            conv_kernel=hf_config.conv_kernel,
        )

    @classmethod
    def get_mamba_state_copy_func(cls) -> tuple[MambaStateCopyFunc, MambaStateCopyFunc]:
        return MambaStateCopyFuncCalculator.mamba1_state_copy_func()

    def copy_inputs_before_cuda_graphs(self, input_buffers, **kwargs):
        return self.mamba_cache.copy_inputs_before_cuda_graphs(input_buffers, **kwargs)

    def get_seqlen_agnostic_capture_inputs(self, batch_size: int):
        return self.mamba_cache.get_seqlen_agnostic_capture_inputs(batch_size)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        logits = self.logits_processor(self.lm_head, hidden_states)
        return logits

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights)
