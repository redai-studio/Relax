# Copyright (c) 2026 Relax Authors. All Rights Reserved.
# Adapted from Miles (RadixArk/miles, fa13e2ec), licensed under Apache-2.0.
# Copyright 2025-2026 RadixArk, Inc. Miles includes software developed from Slime,
# Copyright 2025 Zhipu AI, licensed under Apache-2.0.

from __future__ import annotations

import torch
from megatron.bridge.models.conversion.mapping_registry import MegatronMappingRegistry
from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge
from megatron.bridge.models.conversion.param_mapping import AutoMapping, GatedMLPMapping, ReplicatedMapping
from megatron.bridge.models.mimo_v2_flash.mimo_v2_flash_bridge import MiMoV2FlashQKVMapping
from megatron.core.models.gpt.gpt_model import GPTModel

from .attention import _qkv_group_sizes, _split_qkv
from .provider import MiMoV2ModelProvider


@MegatronModelBridge.register_bridge(
    source="MiMoV2ForCausalLM", target=GPTModel, provider=MiMoV2ModelProvider, model_type="mimo_v2"
)
class MiMoV2Bridge(MegatronModelBridge):
    """HF ``MiMoV2ForCausalLM`` (BF16, split q/k/v) <-> Megatron GPTModel with
    MiMo-V2 attention."""

    def provider_bridge(self, hf_pretrained) -> MiMoV2ModelProvider:
        hf = hf_pretrained.config
        if getattr(hf, "attention_projection_layout", "split") != "split" or getattr(hf, "quantization_config", None):
            raise ValueError("MiMo-V2 requires a BF16 checkpoint with separate q_proj, k_proj and v_proj weights.")
        for swa_key, full_key in (
            ("swa_num_attention_heads", "num_attention_heads"),
            ("swa_head_dim", "head_dim"),
            ("swa_v_head_dim", "v_head_dim"),
        ):
            assert getattr(hf, swa_key) == getattr(hf, full_key), f"{swa_key} must equal {full_key}"

        provider = super().provider_bridge(hf_pretrained)
        if getattr(hf_pretrained, "trust_remote_code", False):
            provider.hf_model_path = hf_pretrained.model_name_or_path
        vision = getattr(hf, "vision_config", None)
        provider.vision_config = vision.to_dict() if hasattr(vision, "to_dict") else vision
        provider.image_token_id = getattr(hf, "image_token_id", None)
        pattern = [int(p) for p in hf.hybrid_layer_pattern]
        provider.hybrid_attention_pattern = pattern
        # A SWA layer attends the query plus sliding_window_size previous keys, like SGLang's
        # RadixAttention (whose window excludes the query token), so rollout and training agree.
        provider.window_size = (hf.sliding_window_size, 0)
        provider.window_attn_skip_freq = pattern
        provider.rotary_base = hf.rope_theta
        provider.rotary_base_per_layer = [hf.swa_rope_theta if p else hf.rope_theta for p in pattern]
        provider.apply_rope_fusion = False
        provider.full_attn_num_query_groups = hf.num_key_value_heads
        provider.swa_num_query_groups = hf.swa_num_key_value_heads
        provider.num_query_groups = hf.num_key_value_heads
        provider.v_head_dim = hf.v_head_dim
        provider.attention_value_scale = hf.attention_value_scale
        provider.full_attention_sink = bool(hf.add_full_attention_sink_bias)
        provider.swa_attention_sink = bool(hf.add_swa_attention_sink_bias)

        provider.normalization = "RMSNorm"
        provider.layernorm_epsilon = hf.layernorm_epsilon
        provider.gated_linear_unit = True
        provider.add_bias_linear = False
        provider.position_embedding_type = "rope"
        provider.share_embeddings_and_output_weights = False
        # HF has no hidden dropout. TransformerConfig otherwise defaults to 0.1.
        provider.hidden_dropout = 0.0

        provider.moe_layer_freq = [int(f) for f in hf.moe_layer_freq]
        provider.moe_grouped_gemm = True
        provider.moe_token_dispatcher_type = "alltoall"
        provider.moe_router_enable_expert_bias = True
        provider.moe_router_bias_update_rate = 0.0
        provider.moe_router_load_balancing_type = "none"
        provider.moe_aux_loss_coeff = 0.0
        provider.moe_router_dtype = "fp32"
        if getattr(hf, "n_group", None) in (None, 1):
            provider.moe_router_num_groups = None
            provider.moe_router_group_topk = None
        provider.mtp_num_layers = None
        return provider

    def mapping_registry(self) -> MegatronMappingRegistry:
        direct = {
            "embedding.word_embeddings.weight": "model.embed_tokens.weight",
            "output_layer.weight": "lm_head.weight",
            "decoder.final_layernorm.weight": "model.norm.weight",
            "decoder.layers.*.self_attention.linear_qkv.layer_norm_weight": "model.layers.*.input_layernorm.weight",
            "decoder.layers.*.self_attention.linear_proj.weight": "model.layers.*.self_attn.o_proj.weight",
            "decoder.layers.*.self_attention.core_attention.softmax_offset": "model.layers.*.self_attn.attention_sink_bias",
            "decoder.layers.*.mlp.linear_fc1.layer_norm_weight": "model.layers.*.post_attention_layernorm.weight",
            "decoder.layers.*.pre_mlp_layernorm.weight": "model.layers.*.post_attention_layernorm.weight",
            "decoder.layers.*.mlp.linear_fc2.weight": "model.layers.*.mlp.down_proj.weight",
            "decoder.layers.*.mlp.router.weight": "model.layers.*.mlp.gate.weight",
            "decoder.layers.*.mlp.router.expert_bias": "model.layers.*.mlp.gate.e_score_correction_bias",
            "decoder.layers.*.mlp.experts.linear_fc2.weight*": "model.layers.*.mlp.experts.*.down_proj.weight",
            "decoder.layers.*.mlp.experts.local_experts.*.linear_fc2.weight": "model.layers.*.mlp.experts.*.down_proj.weight",
        }
        gated = {
            "decoder.layers.*.mlp.linear_fc1.weight": "model.layers.*.mlp",
            "decoder.layers.*.mlp.experts.linear_fc1.weight*": "model.layers.*.mlp.experts.*",
            "decoder.layers.*.mlp.experts.local_experts.*.linear_fc1.weight": "model.layers.*.mlp.experts.*",
        }
        mappings = [AutoMapping(megatron_param=m, hf_param=h) for m, h in direct.items()]
        mappings.append(ReplicatedMapping("visual.**", "visual.**"))
        mappings += [
            GatedMLPMapping(megatron_param=m, gate=f"{h}.gate_proj.weight", up=f"{h}.up_proj.weight")
            for m, h in gated.items()
        ]
        mappings.append(
            MiMoV2FlashQKVMapping(
                megatron_param="decoder.layers.*.self_attention.linear_qkv.weight",
                q="model.layers.*.self_attn.q_proj.weight",
                k="model.layers.*.self_attn.k_proj.weight",
                v="model.layers.*.self_attn.v_proj.weight",
            )
        )
        return MegatronMappingRegistry(*mappings)

    def _split_qkv_linear_out_weight(self, megatron_model, linear_out_weight: torch.Tensor) -> dict[str, torch.Tensor]:
        """Split a TP-gathered ``linear_qkv`` LoRA B into the HF q/k/v rows for
        adapter export and merge.

        The generic split sizes V like K and takes the KV heads from the model
        config; here each layer's KV heads are the global or SWA count whose
        fused row count matches the tensor.
        """
        config = (megatron_model[0] if isinstance(megatron_model, list) else megatron_model).config
        heads, qk_dim, v_dim = config.num_attention_heads, config.kv_channels, config.v_head_dim
        rows = linear_out_weight.shape[0]
        groups = next(
            (
                g
                for g in (config.full_attn_num_query_groups, config.swa_num_query_groups)
                if g * sum(_qkv_group_sizes(heads_per_group=heads // g, qk_dim=qk_dim, v_dim=v_dim)) == rows
            ),
            None,
        )
        assert groups is not None, f"linear_qkv LoRA B has {rows} rows, matching neither a global nor a SWA layer"
        query, key, value = _split_qkv(
            linear_out_weight, dim=0, groups=groups, heads_per_group=heads // groups, qk_dim=qk_dim, v_dim=v_dim
        )
        return {"q_proj": query.flatten(0, 1), "k_proj": key.flatten(0, 1), "v_proj": value.flatten(0, 1)}
