# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""HF Qwen2-VL conversion using Megatron Bridge's standard mapping
machinery."""

import torch
from megatron.bridge.models.conversion.mapping_registry import MegatronMappingRegistry
from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge
from megatron.bridge.models.conversion.param_mapping import AutoMapping, GatedMLPMapping, QKVMapping, ReplicatedMapping
from megatron.bridge.models.conversion.transformers_compat import rope_theta_from_hf
from megatron.bridge.models.hf_pretrained.causal_lm import PreTrainedCausalLM

from relax.models.qwen2_vl.configuration import checkpoint_prefixes, parameter_mapping_specs, validate_qwen2_vl_config
from relax.models.qwen2_vl.model import Qwen2VLModel
from relax.models.qwen2_vl.provider import Qwen2VLModelProvider


@MegatronModelBridge.register_bridge(
    source="Qwen2VLForConditionalGeneration",
    target=Qwen2VLModel,
    provider=Qwen2VLModelProvider,
    model_type="qwen2_vl",
)
class Qwen2VLBridge(MegatronModelBridge):
    def provider_bridge(self, hf_pretrained: PreTrainedCausalLM) -> Qwen2VLModelProvider:
        config = hf_pretrained.config
        text, sections = validate_qwen2_vl_config(config)
        dtype = self.dtype_from_hf(config, default=torch.float32)
        kwargs = self.hf_config_to_provider_kwargs(text)
        kwargs.update(
            normalization="RMSNorm",
            gated_linear_unit=True,
            add_qkv_bias=True,
            add_bias_linear=False,
            qk_layernorm=False,
            hidden_dropout=0.0,
            rotary_base=rope_theta_from_hf(text),
            position_embedding_type="mrope",
            mrope_section=sections,
            share_embeddings_and_output_weights=bool(config.tie_word_embeddings),
            vision_config=config.vision_config,
            image_token_id=config.image_token_id,
            video_token_id=config.video_token_id,
            vision_start_token_id=config.vision_start_token_id,
            vision_end_token_id=config.vision_end_token_id,
            params_dtype=dtype,
            fp16=dtype == torch.float16,
            bf16=dtype == torch.bfloat16,
        )
        return Qwen2VLModelProvider(**kwargs)

    def mapping_registry(self) -> MegatronMappingRegistry:
        # Full conversion supplies serialized keys; adapter-only export instead
        # needs the module names of the installed HF implementation.
        pretrained = getattr(self, "hf_pretrained", None)
        state = getattr(pretrained, "state", None)
        if state is None:
            prefixes = getattr(self, "_adapter_export_prefixes", None)
            if prefixes is None:
                from transformers import Qwen2VLForConditionalGeneration

                config = getattr(self, "hf_config", None)
                if config is None:
                    raise ValueError("Qwen2-VL adapter export requires its HF config")
                with torch.device("meta"):
                    reference = Qwen2VLForConditionalGeneration(config)
                prefixes = checkpoint_prefixes(reference.state_dict())
                self._adapter_export_prefixes = prefixes
            text, vision = prefixes
        else:
            text, vision = checkpoint_prefixes(state)
        mappings = []
        for kind, megatron, hf in parameter_mapping_specs(text, vision):
            if kind == "qkv":
                mapping = QKVMapping(megatron_param=megatron, **hf)
            elif kind == "gated_mlp":
                mapping = GatedMLPMapping(megatron_param=megatron, **hf)
            else:
                cls = ReplicatedMapping if kind == "replicated" else AutoMapping
                mapping = cls(megatron_param=megatron, hf_param=hf)
            mappings.append(mapping)
        return MegatronMappingRegistry(*mappings)
