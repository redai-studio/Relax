# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Megatron Bridge conversion contract for the DeepSeek V4.1 text model."""

from collections.abc import Mapping
from types import SimpleNamespace
from typing import Any, ClassVar

import torch
from megatron.bridge.models.conversion import quantization_utils
from megatron.bridge.models.conversion.mapping_registry import MegatronMappingRegistry
from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge, WeightConversionTask
from megatron.bridge.models.conversion.param_mapping import (
    AutoMapping,
    ColumnParallelMapping,
    GatedMLPMapping,
    ReplicatedMapping,
    RowParallelMapping,
)
from megatron.bridge.models.deepseek.deepseek_v4_bridge import _HCAlphaMapping, _HCAlphaSecondaryMapping
from megatron.bridge.models.hf_pretrained.causal_lm import PreTrainedCausalLM
from megatron.core.models.gpt import GPTModel

from .configuration import V41_CONFIG_FIELDS, hf_config_dict, normalize_text_config
from .provider import DeepSeekV41ModelProvider


@MegatronModelBridge.register_bridge(
    source="DeepseekV41ForCausalLM",
    target=GPTModel,
    provider=DeepSeekV41ModelProvider,
    model_type="deepseek_v41",
)
class DeepSeekV41Bridge(MegatronModelBridge):
    """Convert trainable text weights, excluding frozen Engram tables and
    auxiliary heads."""

    # Do not let the generic builder bypass the V4.1 provider's execution gate.
    MODEL_CONFIG_CLASS: ClassVar[type[Any] | None] = None
    # A standalone checkpoint needs the frozen tables and an explicit auxiliary-head policy.
    SUPPORTS_HF_PRETRAINED_EXPORT: ClassVar[bool] = False

    def provider_bridge(self, hf_pretrained: PreTrainedCausalLM) -> DeepSeekV41ModelProvider:
        text = normalize_text_config(hf_pretrained.config)
        num_layers = text["num_hidden_layers"]
        ratios = list(text["compress_ratios"][:num_layers])
        if len(ratios) != num_layers or any(ratio not in (0, 1, 2) for ratio in ratios):
            raise ValueError("DeepSeek V4.1 requires one compression ratio (0, 1 or 2) per decoder layer.")
        text.update(
            intermediate_size=text.get("intermediate_size", text["moe_intermediate_size"]),
            kv_lora_rank=text["head_dim"],
            qk_nope_head_dim=text["head_dim"] - text["qk_rope_head_dim"],
            v_head_dim=text["head_dim"],
            num_nextn_predict_layers=0,
        )
        text["torch_dtype"] = text.get("dtype") or text.get("torch_dtype") or "bfloat16"
        provider = super().provider_bridge(SimpleNamespace(config=SimpleNamespace(**text)))
        provider.v41_hf_config = hf_config_dict(hf_pretrained.config)
        provider.v41_hf_checkpoint = getattr(hf_pretrained, "model_name_or_path", None)
        provider.experimental_attention_variant = None
        provider.multi_latent_attention = True
        provider.normalization = "RMSNorm"
        provider.qk_layernorm = True
        provider.add_bias_linear = False
        provider.add_qkv_bias = False
        provider.hidden_dropout = 0.0
        provider.attention_dropout = text.get("attention_dropout", 0.0)
        provider.rotary_percent = 1.0
        rope = text.get("rope_scaling") or text.get("rope_parameters") or {}
        main_rope = rope.get("main", {}) if "compress" in rope else rope
        compressed_rope = rope.get("compress", rope)
        provider.rope_type = "yarn"
        provider.rotary_base = float(main_rope.get("rope_theta", text["rope_theta"]))
        provider.rotary_scaling_factor = float(compressed_rope["factor"])
        provider.original_max_position_embeddings = int(compressed_rope["original_max_position_embeddings"])
        provider.beta_fast = float(compressed_rope.get("beta_fast", 32))
        provider.beta_slow = float(compressed_rope.get("beta_slow", 1))
        provider.mscale = 1.0
        provider.mscale_all_dim = 1.0
        provider.o_groups = text["o_groups"]
        provider.o_lora_rank = text["o_lora_rank"]
        provider.csa_compress_ratios = ratios
        provider.csa_window_size = text["sliding_window"]
        provider.csa_compress_rotary_base = text["compress_rope_theta"]
        provider.dsa_indexer_n_heads = text["index_n_heads"]
        provider.dsa_indexer_head_dim = text["index_head_dim"]
        provider.dsa_indexer_topk = text["index_topk"]
        provider.dsa_kernel_backend = "none"
        provider.enable_hyper_connections = True
        provider.num_residual_streams = text["hc_mult"]
        provider.mhc_sinkhorn_iterations = text["hc_sinkhorn_iters"]
        provider.use_fused_mhc = False
        provider.v41_hc_eps = text["hc_eps"]
        provider.gated_linear_unit = True
        provider.moe_grouped_gemm = True
        provider.moe_layer_freq = [1] * num_layers
        provider.moe_n_hash_layers = 0
        provider.moe_shared_expert_intermediate_size = text["moe_intermediate_size"] * text["n_shared_experts"]
        provider.moe_router_pre_softmax = False
        provider.moe_router_enable_expert_bias = True
        provider.moe_router_bias_update_rate = 0.0
        provider.moe_router_dtype = "fp32"
        provider.moe_router_load_balancing_type = "none"
        provider.moe_aux_loss_coeff = 0.0
        provider.norm_topk_prob = text["norm_topk_prob"]
        provider.activation_func_clamp_value = text["swiglu_limit"]
        provider.activation_func_clamp_shared_expert = True
        provider.bias_activation_fusion = False
        provider.mtp_num_layers = None
        for name in V41_CONFIG_FIELDS:
            value = text[name]
            setattr(provider, f"v41_{name}", tuple(value) if isinstance(value, list) else value)
        return provider

    def mapping_registry(self) -> MegatronMappingRegistry:
        mappings = [
            AutoMapping("embedding.word_embeddings.weight", "embed.weight"),
            AutoMapping("output_layer.weight", "head.weight"),
            ReplicatedMapping("decoder.final_layernorm.weight", "norm.weight"),
        ]
        mg = "decoder.layers.*."
        hf = "layers.*."
        replicated = {
            "input_layernorm.weight": "attn_norm.weight",
            "pre_mlp_layernorm.weight": "ffn_norm.weight",
            "self_attention.linear_q_down_proj.weight": "attn.wq_a.weight",
            "self_attention.q_layernorm.weight": "attn.q_norm.weight",
            "self_attention.linear_kv_proj.weight": "attn.wkv.weight",
            "self_attention.kv_layernorm.weight": "attn.kv_norm.weight",
            "self_attention.core_attention.compressor.linear_wkv.weight": "attn.compressor.wkv.weight",
            "self_attention.core_attention.compressor.linear_wgate.weight": "attn.compressor.wgate.weight",
            "self_attention.core_attention.compressor.norm.weight": "attn.compressor.norm.weight",
            "self_attention.core_attention.indexer.linear_wq_b.weight": "attn.indexer.wq_b.weight",
            "self_attention.core_attention.indexer.linear_weights_proj.weight": "attn.indexer.weights_proj.weight",
            "self_attention.core_attention.indexer.linear_wk.weight": "attn.indexer.wk.weight",
            "self_attention.core_attention.indexer.k_norm.weight": "attn.indexer.k_norm.weight",
            "mlp.router.weight": "ffn.gate.weight",
            "mlp.router.expert_bias": "ffn.gate.bias",
            "v41_engram.linear_wkv.weight": "engram.wkv.weight",
            "v41_engram.q_weight": "engram.q_weight",
            "v41_engram.k_weight": "engram.k_weight",
        }
        mappings.extend(ReplicatedMapping(mg + key, hf + value) for key, value in replicated.items())
        mappings.extend(
            [
                ColumnParallelMapping(mg + "self_attention.linear_q_up_proj.weight", hf + "attn.wq_b.weight"),
                ColumnParallelMapping(mg + "self_attention.linear_o_group_proj", hf + "attn.wo_a.weight"),
                ColumnParallelMapping(mg + "self_attention.core_attention.attn_sink", hf + "attn.attn_sink"),
                RowParallelMapping(mg + "self_attention.linear_proj.weight", hf + "attn.wo_b.weight"),
            ]
        )
        for expert_mg, expert_hf in (
            ("mlp.experts.linear_fc1.weight*", "ffn.experts.*."),
            ("mlp.experts.local_experts.*.linear_fc1.weight", "ffn.experts.*."),
            ("mlp.shared_experts.linear_fc1.weight", "ffn.shared_experts."),
        ):
            mappings.append(
                GatedMLPMapping(mg + expert_mg, gate=hf + expert_hf + "w1.weight", up=hf + expert_hf + "w3.weight")
            )
            mappings.append(
                AutoMapping(mg + expert_mg.replace("linear_fc1", "linear_fc2"), hf + expert_hf + "w2.weight")
            )
        for site, scale in (
            ("self_attention_hyper_connection", "hc_attn_scale"),
            ("mlp_hyper_connection", "hc_ffn_scale"),
        ):
            prefix = mg + site
            stem = "hc_attn" if site == "self_attention_hyper_connection" else "hc_ffn"
            mappings.extend(
                [
                    ReplicatedMapping(prefix + ".mapping_proj.weight", hf + stem + "_fn"),
                    ReplicatedMapping(prefix + ".bias", hf + stem + "_base"),
                    _HCAlphaMapping(prefix + ".alpha_pre", prefix + ".alpha_post", prefix + ".alpha_res", hf + scale),
                    _HCAlphaSecondaryMapping(prefix + ".alpha_post", hf + scale, 1),
                    _HCAlphaSecondaryMapping(prefix + ".alpha_res", hf + scale, 2),
                ]
            )
        return MegatronMappingRegistry(*mappings)

    def maybe_modify_loaded_hf_weight(
        self, hf_param: str | dict[str, str], hf_state_dict: Mapping[str, torch.Tensor]
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        if isinstance(hf_param, dict):
            return {key: self.maybe_modify_loaded_hf_weight(name, hf_state_dict) for key, name in hf_param.items()}
        if ".engram.embed." in hf_param:
            raise ValueError(
                "Frozen Engram tables require the model's CPU-backed loader, not Bridge tensor conversion."
            )
        if hf_param.endswith(".ffn.gate.bias") and hf_param not in hf_state_dict:
            text = normalize_text_config(self.hf_config)
            return torch.zeros(text["n_routed_experts"], dtype=torch.float32)
        weight = hf_state_dict[hf_param]
        if weight.dtype not in (torch.float8_e4m3fn, torch.int8, torch.uint8):
            return weight
        if not hf_param.endswith(".weight"):
            return weight.to(torch.bfloat16) if weight.dtype == torch.float8_e4m3fn else weight
        scale_key = hf_param.removesuffix(".weight") + ".scale"
        if scale_key not in hf_state_dict:
            raise ValueError(f"Missing quantization scale {scale_key!r} for {hf_param!r}.")
        scale = hf_state_dict[scale_key]
        if scale.dtype == torch.uint8:
            scale = scale.view(torch.float8_e8m0fnu)
        if weight.dtype in (torch.int8, torch.uint8):
            return quantization_utils.dequantize_mxfp4_e2m1_packed(weight, scale)
        return quantization_utils.dequantize_fp8_e4m3fn_with_scale(weight, scale, name=hf_param, block_size=32)

    def maybe_modify_converted_hf_weight(
        self,
        task: WeightConversionTask,
        converted_weights_dict: dict[str, torch.Tensor],
        hf_state_dict: Mapping[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        result: dict[str, torch.Tensor] = {}
        for name, weight in converted_weights_dict.items():
            if name.endswith(".ffn.gate.bias") and name not in hf_state_dict:
                continue
            scale_key = name.removesuffix(".weight") + ".scale"
            if task.weight_dtype is not None or not name.endswith(".weight") or scale_key not in hf_state_dict:
                result[name] = weight
                continue
            source_dtype = hf_state_dict[name].dtype
            if source_dtype not in (torch.float8_e4m3fn, torch.int8, torch.uint8):
                result[name] = weight
                continue
            source_scale = hf_state_dict[scale_key]
            scale = source_scale.view(torch.float8_e8m0fnu) if source_scale.dtype == torch.uint8 else source_scale
            if source_dtype in (torch.int8, torch.uint8):
                quantized, quantized_scale = quantization_utils.quantize_mxfp4_e2m1_like_scale(
                    weight, scale, name=name
                )
                quantized = quantized.view(source_dtype)
            else:
                quantized, quantized_scale = quantization_utils.quantize_fp8_e4m3fn_like_scale(
                    weight, scale, name=name, block_size=32
                )
            result[name] = quantized
            result[scale_key] = (
                quantized_scale.view(torch.uint8) if source_scale.dtype == torch.uint8 else quantized_scale
            )
        return result

    @classmethod
    def megatron_to_hf_config(cls, provider: DeepSeekV41ModelProvider) -> dict[str, Any]:
        """Retain V4.1 identity and source metadata; this is not a standalone
        checkpoint export."""
        config = hf_config_dict(provider.v41_hf_config)
        text = config.get("text_config", config)
        text.update(
            num_hidden_layers=provider.num_layers,
            compress_ratios=list(provider.csa_compress_ratios),
            num_nextn_predict_layers=0,
        )
        config["model_type"] = "deepseek_v41"
        config["architectures"] = ["DeepseekV41ForCausalLM"]
        return config
