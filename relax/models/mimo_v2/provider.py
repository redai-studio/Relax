# Copyright (c) 2026 Relax Authors. All Rights Reserved.
# Adapted from Miles (RadixArk/miles, fa13e2ec), licensed under Apache-2.0.
# Copyright 2025-2026 RadixArk, Inc. Miles includes software developed from Slime,
# Copyright 2025 Zhipu AI, licensed under Apache-2.0.

from __future__ import annotations

import copy
from collections.abc import Callable
from dataclasses import dataclass, field

from megatron.bridge.models.gpt_provider import GPTModelProvider
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_decoder_block_spec
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.transformer import ModuleSpec

from .attention import MiMoV2SelfAttention, MiMoV2TEDotProductAttention


def mimo_v2_layer_spec(config, vp_stage: int | None = None) -> ModuleSpec:
    spec = get_gpt_decoder_block_spec(config, use_transformer_engine=True, vp_stage=vp_stage)
    for layer_spec in spec.layer_specs:
        layer_spec.submodules.self_attention.module = MiMoV2SelfAttention
        layer_spec.submodules.self_attention.submodules.core_attention = MiMoV2TEDotProductAttention
    return spec


@dataclass
class MiMoV2ModelProvider(GPTModelProvider):
    """GPT provider plus the MiMo-V2 attention fields; each layer is built from
    ``get_config_for_layer``."""

    transformer_layer_spec: ModuleSpec | Callable = field(default_factory=lambda: mimo_v2_layer_spec)
    # Makes TransformerBlock build each layer from get_config_for_layer.
    heterogeneous_block_specs: bool = True
    hybrid_attention_pattern: list[int] | None = None
    full_attn_num_query_groups: int = 4
    swa_num_query_groups: int = 8
    v_head_dim: int = 128
    attention_value_scale: float | None = None
    full_attention_sink: bool = False
    swa_attention_sink: bool = True

    def get_config_for_layer(self, layer_number: int) -> MiMoV2ModelProvider:
        """Shallow copy with the KV heads and attention sink of global layer
        ``layer_number`` (1-indexed)."""
        assert layer_number <= len(self.hybrid_attention_pattern), f"no attention pattern for layer {layer_number}"
        swa = self.hybrid_attention_pattern[layer_number - 1]
        config = copy.copy(self)
        config.num_query_groups = self.swa_num_query_groups if swa else self.full_attn_num_query_groups
        sink = self.swa_attention_sink if swa else self.full_attention_sink
        config.softmax_type = "learnable" if sink else "vanilla"
        return config

    def provide(self, pre_process=None, post_process=None, vp_stage=None) -> GPTModel:
        assert self.tensor_model_parallel_size <= min(self.full_attn_num_query_groups, self.swa_num_query_groups), (
            "MiMo-V2 needs TP <= the smallest per-layer KV head count"
        )
        assert self.context_parallel_size == 1, "MiMo-V2 attention does not support context parallelism yet"
        return super().provide(pre_process=pre_process, post_process=post_process, vp_stage=vp_stage)
