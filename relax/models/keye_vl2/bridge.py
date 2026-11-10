# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from typing import Any

from megatron.bridge.models.conversion.mapping_registry import MegatronMappingRegistry
from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge
from megatron.bridge.models.conversion.param_mapping import (
    AutoMapping,
    FusedExpertMapping,
    FusedGatedExpertMapping,
    QKVMapping,
    ReplicatedMapping,
    RowParallelMapping,
)
from megatron.bridge.models.qwen.qwen3_moe_bridge import Qwen3MoEBridge
from megatron.core.models.gpt.gpt_model import GPTModel

from .provider import KeyeModelProvider


@MegatronModelBridge.register_bridge(
    source="KeyeVL2MoeForConditionalGeneration", target=GPTModel, provider=KeyeModelProvider, model_type="KeyeVL2"
)
class KeyeBridge(Qwen3MoEBridge):
    def provider_bridge(self, hf_pretrained: Any) -> KeyeModelProvider:
        provider = super().provider_bridge(hf_pretrained)
        provider.hf_model_path = hf_pretrained.model_name_or_path
        provider.hf_config = hf_pretrained.config
        provider.mrope_section = [16, 24, 24]
        provider.kv_channels = hf_pretrained.config.head_dim
        provider.rotary_base = hf_pretrained.config.rope_theta
        provider.moe_ffn_hidden_size = hf_pretrained.config.moe_intermediate_size
        return provider

    def mapping_registry(self) -> MegatronMappingRegistry:
        names = {
            "embedding.word_embeddings.weight": "model.embed_tokens.weight",
            "output_layer.weight": "lm_head.weight",
            "decoder.final_layernorm.weight": "model.norm.weight",
            "decoder.layers.*.input_layernorm.weight": "model.layers.*.input_layernorm.weight",
            "decoder.layers.*.pre_mlp_layernorm.weight": "model.layers.*.post_attention_layernorm.weight",
            "decoder.layers.*.self_attention.q_layernorm.weight": "model.layers.*.self_attn.q_norm.weight",
            "decoder.layers.*.self_attention.k_layernorm.weight": "model.layers.*.self_attn.k_norm.weight",
        }
        prefix = "language_model." if getattr(self, "_multimodal", False) else ""
        mappings = [AutoMapping(prefix + k, v) for k, v in names.items()]
        mappings.append(
            RowParallelMapping(
                prefix + "decoder.layers.*.self_attention.linear_proj.weight", "model.layers.*.self_attn.o_proj.weight"
            )
        )
        mappings.append(
            ReplicatedMapping(prefix + "decoder.layers.*.mlp.router.weight", "model.layers.*.mlp.gate.weight")
        )
        for suffix in ("q_proj.weight", "k_proj.weight", "q_norm.weight", "k_norm.weight", "k_norm.bias"):
            mappings.append(
                ReplicatedMapping(
                    f"{prefix}decoder.layers.*.self_attention.core_attention.sa_indexer.{suffix}",
                    f"model.layers.*.self_attn.sa_indexer.{suffix}",
                )
            )
        mappings.extend(
            [
                QKVMapping(
                    prefix + "decoder.layers.*.self_attention.linear_qkv.weight",
                    q="model.layers.*.self_attn.q_proj.weight",
                    k="model.layers.*.self_attn.k_proj.weight",
                    v="model.layers.*.self_attn.v_proj.weight",
                ),
                FusedGatedExpertMapping(
                    prefix + "decoder.layers.*.mlp.experts.linear_fc1.weight*",
                    "model.layers.*.mlp.experts.gate_up_proj",
                    transpose_on_export=True,
                ),
                FusedExpertMapping(
                    prefix + "decoder.layers.*.mlp.experts.linear_fc2.weight*",
                    "model.layers.*.mlp.experts.down_proj",
                    transpose_on_export=True,
                ),
            ]
        )
        if prefix:
            mappings.extend([ReplicatedMapping("visual.**", "visual.**"), ReplicatedMapping("mlp_AR.**", "mlp_AR.**")])
        return MegatronMappingRegistry(*mappings)

    def build_conversion_tasks(self, hf_pretrained: Any, megatron_model: Any, weight_dtype: Any = None) -> list:
        models = megatron_model if isinstance(megatron_model, list) else [megatron_model]
        self._multimodal = any("language_model." in name for model in models for name, _ in model.named_parameters())
        tasks = super().build_conversion_tasks(hf_pretrained, megatron_model, weight_dtype=weight_dtype)
        keys = set(hf_pretrained.state.source.get_all_keys())
        consumed = {name for task in tasks for name in task.required_hf_param_names}
        ignored = set() if self._multimodal else {name for name in keys if name.startswith(("visual.", "mlp_AR."))}
        if keys - consumed - ignored:
            raise ValueError(f"Unmapped Keye language weights: {sorted(keys - consumed - ignored)}")
        return tasks
