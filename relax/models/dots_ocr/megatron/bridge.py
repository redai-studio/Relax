# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import torch
from megatron.bridge.models.conversion.mapping_registry import MegatronMappingRegistry
from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge
from megatron.bridge.models.conversion.param_mapping import AutoMapping, GatedMLPMapping, QKVMapping, ReplicatedMapping
from megatron.bridge.models.conversion.transformers_compat import rope_theta_from_hf
from megatron.bridge.models.hf_pretrained.causal_lm import PreTrainedCausalLM

from relax.models.dots_ocr.configuration import DotsVisionConfig
from relax.models.dots_ocr.megatron.model import DotsOCRModel
from relax.models.dots_ocr.megatron.provider import DotsOCRModelProvider


@MegatronModelBridge.register_bridge(
    source="DotsOCRForCausalLM",
    target=DotsOCRModel,
)
class DotsOCRBridge(MegatronModelBridge):
    def provider_bridge(self, hf_pretrained: PreTrainedCausalLM) -> DotsOCRModelProvider:
        hf_config = hf_pretrained.config
        model_dtype = self.dtype_from_hf(hf_config, default=torch.float32)
        vision_config = hf_config.vision_config
        if isinstance(vision_config, dict):
            vision_config = DotsVisionConfig(**vision_config)
        vision_config.torch_dtype = model_dtype

        head_dim = getattr(hf_config, "head_dim", hf_config.hidden_size // hf_config.num_attention_heads)
        return DotsOCRModelProvider(
            num_layers=hf_config.num_hidden_layers,
            hidden_size=hf_config.hidden_size,
            ffn_hidden_size=hf_config.intermediate_size,
            num_attention_heads=hf_config.num_attention_heads,
            num_query_groups=hf_config.num_key_value_heads,
            kv_channels=head_dim,
            init_method_std=hf_config.initializer_range,
            layernorm_epsilon=hf_config.rms_norm_eps,
            make_vocab_size_divisible_by=self.make_vocab_size_divisible_by(hf_config.vocab_size),
            rotary_base=rope_theta_from_hf(hf_config),
            share_embeddings_and_output_weights=getattr(hf_config, "tie_word_embeddings", False),
            vocab_size=hf_config.vocab_size,
            seq_length=hf_config.max_position_embeddings,
            language_max_sequence_length=hf_config.max_position_embeddings,
            fp16=(model_dtype == torch.float16),
            bf16=(model_dtype == torch.bfloat16),
            params_dtype=model_dtype,
            add_qkv_bias=getattr(hf_config, "attention_bias", True),
            vision_config=vision_config,
            image_token_id=getattr(hf_config, "image_token_id", 151665),
            video_token_id=getattr(hf_config, "video_token_id", 151656),
        )

    def mapping_registry(self) -> MegatronMappingRegistry:
        mapping_list = [
            AutoMapping(
                megatron_param="language_model.embedding.word_embeddings.weight",
                hf_param="model.embed_tokens.weight",
            ),
            AutoMapping(megatron_param="language_model.output_layer.weight", hf_param="lm_head.weight"),
            AutoMapping(
                megatron_param="language_model.decoder.final_layernorm.weight",
                hf_param="model.norm.weight",
            ),
            AutoMapping(
                megatron_param="language_model.decoder.layers.*.self_attention.linear_qkv.layer_norm_weight",
                hf_param="model.layers.*.input_layernorm.weight",
            ),
            AutoMapping(
                megatron_param="language_model.decoder.layers.*.mlp.linear_fc1.layer_norm_weight",
                hf_param="model.layers.*.post_attention_layernorm.weight",
            ),
            AutoMapping(
                megatron_param="language_model.decoder.layers.*.self_attention.linear_proj.weight",
                hf_param="model.layers.*.self_attn.o_proj.weight",
            ),
            AutoMapping(
                megatron_param="language_model.decoder.layers.*.mlp.linear_fc2.weight",
                hf_param="model.layers.*.mlp.down_proj.weight",
            ),
            ReplicatedMapping(megatron_param="vision_model.**", hf_param="vision_tower.**"),
            QKVMapping(
                megatron_param="language_model.decoder.layers.*.self_attention.linear_qkv.weight",
                q="model.layers.*.self_attn.q_proj.weight",
                k="model.layers.*.self_attn.k_proj.weight",
                v="model.layers.*.self_attn.v_proj.weight",
            ),
            QKVMapping(
                megatron_param="language_model.decoder.layers.*.self_attention.linear_qkv.bias",
                q="model.layers.*.self_attn.q_proj.bias",
                k="model.layers.*.self_attn.k_proj.bias",
                v="model.layers.*.self_attn.v_proj.bias",
            ),
            GatedMLPMapping(
                megatron_param="language_model.decoder.layers.*.mlp.linear_fc1.weight",
                gate="model.layers.*.mlp.gate_proj.weight",
                up="model.layers.*.mlp.up_proj.weight",
            ),
        ]
        return MegatronMappingRegistry(*mapping_list)
