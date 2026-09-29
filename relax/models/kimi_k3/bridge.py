# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from copy import copy

from megatron.bridge.models.conversion.mapping_registry import MegatronMappingRegistry
from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge
from megatron.bridge.models.conversion.param_mapping import ReplicatedMapping
from megatron.bridge.models.hf_pretrained.causal_lm import PreTrainedCausalLM
from megatron.bridge.models.kimi.kimi_k3_bridge import KimiK3Bridge

from .model import KimiK3VLModel
from .provider import KimiK3VLModelProvider


@MegatronModelBridge.register_bridge(
    source="KimiK3ForConditionalGeneration",
    target=KimiK3VLModel,
    provider=KimiK3VLModelProvider,
    model_type="kimi_k3",
)
class KimiK3VLBridge(KimiK3Bridge):
    """Extend upstream's K3 language conversion with trainable visual
    weights."""

    # Both visual modules have conversion mappings below. The language-only
    # bridge's passthrough would otherwise export stale source visual weights.
    _HF_PASSTHROUGH_PREFIXES = ()

    def provider_bridge(self, hf_pretrained: PreTrainedCausalLM) -> KimiK3VLModelProvider:
        provider = super().provider_bridge(hf_pretrained)
        provider.vision_config = hf_pretrained.config.vision_config
        provider.hf_model_path = hf_pretrained.model_name_or_path
        provider.trust_remote_code = bool(getattr(hf_pretrained, "trust_remote_code", False))
        provider.image_token_id = hf_pretrained.config.media_placeholder_token_id
        provider.scatter_embedding_sequence_parallel = False
        return provider

    def mapping_registry(self) -> MegatronMappingRegistry:
        mappings = []
        for original in super().mapping_registry().mappings:
            mapping = copy(original)
            mapping.megatron_param = f"language_model.{original.megatron_param}"
            mappings.append(mapping)
        mappings.extend(
            [
                ReplicatedMapping("vision_tower.**", "vision_tower.**"),
                ReplicatedMapping("mm_projector.**", "mm_projector.**"),
            ]
        )
        return MegatronMappingRegistry(*mappings)
