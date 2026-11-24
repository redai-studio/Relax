# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from collections.abc import Mapping

import torch
from megatron.bridge.models.conversion.mapping_registry import MegatronMappingRegistry
from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge
from megatron.bridge.models.conversion.param_mapping import (
    AutoMapping,
    GatedMLPMapping,
    QKVMapping,
)
from megatron.bridge.models.hf_pretrained.causal_lm import PreTrainedCausalLM
from megatron.bridge.models.mla_provider import MLAModelProvider
from megatron.core.models.gpt.gpt_model import GPTModel
from transformers import GlmMoeDsaForCausalLM

from relax.utils.logging_utils import get_logger

from .index_sharing import index_sources


logger = get_logger(__name__)


@MegatronModelBridge.register_bridge(
    source=GlmMoeDsaForCausalLM, target=GPTModel, provider=MLAModelProvider, model_type="glm_moe_dsa"
)
class GLM5Bridge(MegatronModelBridge):
    """Megatron Bridge for GLM-5 through GLM-5.3 (MoE + MLA + DSA).

    This bridge handles conversion between HuggingFace ``GlmMoeDsaForCausalLM``
    and Megatron-Core ``GPTModel`` formats. GLM-5 and GLM-5.1 share the same
    architecture. GLM-5.2 and GLM-5.3 additionally share DSA indices across
    decoder layers; the HF configuration determines indexer ownership.

    The architecture uses Multi-Latent Attention (MLA), Dynamic Sparse Attention
    (DSA) indexer layers, and Mixture-of-Experts (MoE), with optional
    Multi-Token Prediction (MTP) layers.

    Requires ``transformers>=5.2.0``.

    Example:
        >>> from megatron.bridge import AutoBridge
        >>> bridge = AutoBridge.from_hf_pretrained("zai-org/GLM-5.1")
        >>> provider = bridge.to_megatron_provider()
    """

    def maybe_modify_loaded_hf_weight(
        self, hf_param: str | dict[str, str], hf_state_dict: Mapping[str, torch.Tensor]
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        """Restore block-scaled FP8 releases before TP sharding and BF16
        training."""
        if isinstance(hf_param, dict):
            return {key: self.maybe_modify_loaded_hf_weight(name, hf_state_dict) for key, name in hf_param.items()}
        weight = hf_state_dict[hf_param]
        if weight.dtype not in (torch.float8_e4m3fn, torch.float8_e5m2):
            return weight
        quantization = getattr(self.hf_config, "quantization_config", None) or {}
        block = quantization.get("weight_block_size")
        if quantization.get("quant_method") != "fp8" or block != [128, 128]:
            raise ValueError(f"Unsupported GLM FP8 block quantization for {hf_param}: {quantization}")
        scale_name = f"{hf_param}_scale_inv"
        if scale_name not in hf_state_dict:
            raise ValueError(f"Missing FP8 scale tensor {scale_name}")
        scale = hf_state_dict[scale_name]
        if weight.ndim != 2:
            raise ValueError(f"Expected a matrix for FP8 weight {hf_param}, got shape {tuple(weight.shape)}")
        rows, cols = weight.shape
        grid_rows, grid_cols = (rows + 127) // 128, (cols + 127) // 128
        if tuple(scale.shape) != (grid_rows, grid_cols):
            raise ValueError(f"Invalid FP8 scale shape for {hf_param}: {tuple(scale.shape)}")
        # A full tensor is loaded lazily on the import rank. Broadcast scales
        # over blocks without materializing a second full-size scale matrix.
        values = torch.nn.functional.pad(weight.float(), (0, grid_cols * 128 - cols, 0, grid_rows * 128 - rows))
        values = values.view(grid_rows, 128, grid_cols, 128)
        values.mul_(scale.to(device=values.device, dtype=torch.float32)[:, None, :, None])
        return values.view(grid_rows * 128, grid_cols * 128)[:rows, :cols].to(torch.bfloat16).contiguous()

    def provider_bridge(self, hf_pretrained: PreTrainedCausalLM) -> MLAModelProvider:
        provider = super().provider_bridge(hf_pretrained)
        hf_config = hf_pretrained.config

        # Use fused DSAMLASelfAttention spec (ported from slime) instead of
        # Megatron Core's native DSAttention. This enables:
        #   1. Context Parallelism (CP > 1)
        #   2. Fused SparseMLA kernel (no O(n^2) attention matrix)
        #   3. Query-blocked indexer scoring and cross-layer top-k sharing
        from relax.models.glm_moe_dsa.dsa_attention import get_glm5_dsa_spec, transform_glm5_layer_spec

        provider.transformer_layer_spec = get_glm5_dsa_spec
        provider.mtp_layer_spec_transform = transform_glm5_layer_spec

        # GLM-5 uses RoPE, not learned absolute position embeddings.
        provider.position_embedding_type = "rope"
        provider.rope_type = "rope"

        provider.normalization = "RMSNorm"
        provider.gated_linear_unit = True
        provider.add_bias_linear = False
        provider.share_embeddings_and_output_weights = False
        provider.qk_layernorm = True
        provider.multi_latent_attention = True
        provider.moe_grouped_gemm = True
        provider.moe_router_pre_softmax = True
        provider.moe_token_dispatcher_type = "alltoall"
        provider.moe_router_load_balancing_type = "seq_aux_loss"
        # NOTE: moe_shared_expert_overlap only works with the alltoall dispatcher.
        # When using the flex dispatcher (overridden via bridge_keys), this must
        # be False. Default to False here; enable via --moe-shared-expert-overlap
        # if using alltoall.
        provider.moe_shared_expert_overlap = False
        provider.moe_router_score_function = "sigmoid"
        provider.moe_router_enable_expert_bias = True
        provider.moe_router_dtype = "fp32"
        provider.moe_permute_fusion = True
        provider.hidden_dropout = 0.0
        provider.attention_softmax_in_fp32 = False
        provider.make_vocab_size_divisible_by = 1280

        # GLM5-specific: computed fields not in CONFIG_MAPPING.
        provider.moe_layer_freq = [0] * hf_config.first_k_dense_replace + [1] * (
            hf_config.num_hidden_layers - hf_config.first_k_dense_replace
        )
        provider.moe_shared_expert_intermediate_size = hf_config.moe_intermediate_size * hf_config.n_shared_experts

        # GLM5-specific: rotary_base is nested in rope_parameters.
        provider.rotary_base = hf_config.rope_parameters["rope_theta"]

        # GLM5 uses default rope (no YaRN scaling).
        provider.rotary_scaling_factor = 1.0
        provider.mscale = 1.0
        provider.mscale_all_dim = 1.0

        # DSA indexer params — stored on the config for DSAMLASelfAttention to
        # read. NOTE: We do NOT set ``experimental_attention_variant = "dsa"``
        # here, because that would trigger the CP=1 assertion in Megatron Core's
        # ``transformer_config.py``. The fused DSAMLASelfAttention handles DSA
        # natively without that flag.
        provider.index_head_dim = hf_config.index_head_dim
        provider.index_num_attention_heads = hf_config.index_n_heads
        provider.dsa_indexer_topk = hf_config.index_topk
        provider.dsa_indexer_loss_coeff = 0.001
        provider.dsa_indexer_use_sparse_loss = True
        provider.glm_dsa_index_sources = index_sources(hf_config)

        return provider

    def mapping_registry(self) -> MegatronMappingRegistry:
        param_mappings = {
            # Embed
            "embedding.word_embeddings.weight": "model.embed_tokens.weight",
            # LM Head
            "decoder.final_layernorm.weight": "model.norm.weight",
            "output_layer.weight": "lm_head.weight",
            # Attention layernorm
            "decoder.layers.*.self_attention.linear_qkv.layer_norm_weight": "model.layers.*.input_layernorm.weight",
            "decoder.layers.*.input_layernorm.weight": "model.layers.*.input_layernorm.weight",
            # Attention output
            "decoder.layers.*.self_attention.linear_proj.weight": "model.layers.*.self_attn.o_proj.weight",
            # Post-attention layernorm — MoE layers use pre_mlp_layernorm,
            # dense layers use layer_norm_weight.
            "decoder.layers.*.pre_mlp_layernorm.weight": "model.layers.*.post_attention_layernorm.weight",
            "decoder.layers.*.mlp.linear_fc1.layer_norm_weight": "model.layers.*.post_attention_layernorm.weight",
            # MLA weights
            "decoder.layers.*.self_attention.linear_q_down_proj.weight": "model.layers.*.self_attn.q_a_proj.weight",
            "decoder.layers.*.self_attention.linear_q_up_proj.weight": "model.layers.*.self_attn.q_b_proj.weight",
            "decoder.layers.*.self_attention.linear_q_up_proj.layer_norm_weight": "model.layers.*.self_attn.q_a_layernorm.weight",
            "decoder.layers.*.self_attention.q_layernorm.weight": "model.layers.*.self_attn.q_a_layernorm.weight",
            "decoder.layers.*.self_attention.linear_kv_down_proj.weight": "model.layers.*.self_attn.kv_a_proj_with_mqa.weight",
            "decoder.layers.*.self_attention.linear_kv_up_proj.weight": "model.layers.*.self_attn.kv_b_proj.weight",
            "decoder.layers.*.self_attention.linear_kv_up_proj.layer_norm_weight": "model.layers.*.self_attn.kv_a_layernorm.weight",
            "decoder.layers.*.self_attention.kv_layernorm.weight": "model.layers.*.self_attn.kv_a_layernorm.weight",
            # For non-MLA attention (fallback)
            "decoder.layers.*.self_attention.linear_q_proj.weight": "model.layers.*.self_attn.q_proj.weight",
            # DSA indexer — weights live directly on DSAMLASelfAttention (not
            # nested under core_attention.indexer).
            "decoder.layers.*.self_attention.wq_b.weight": "model.layers.*.self_attn.indexer.wq_b.weight",
            "decoder.layers.*.self_attention.wk.weight": "model.layers.*.self_attn.indexer.wk.weight",
            "decoder.layers.*.self_attention.k_norm.weight": "model.layers.*.self_attn.indexer.k_norm.weight",
            "decoder.layers.*.self_attention.k_norm.bias": "model.layers.*.self_attn.indexer.k_norm.bias",
            "decoder.layers.*.self_attention.weights_proj.weight": "model.layers.*.self_attn.indexer.weights_proj.weight",
            # Dense MLP
            "decoder.layers.*.mlp.linear_fc2.weight": "model.layers.*.mlp.down_proj.weight",
            # MoE router
            "decoder.layers.*.mlp.router.weight": "model.layers.*.mlp.gate.weight",
            "decoder.layers.*.mlp.router.expert_bias": "model.layers.*.mlp.gate.e_score_correction_bias",
            # MoE shared experts
            "decoder.layers.*.mlp.shared_experts.router.weight": "model.layers.*.mlp.shared_experts.gate.weight",
            "decoder.layers.*.mlp.shared_experts.linear_fc2.weight": "model.layers.*.mlp.shared_experts.down_proj.weight",
        }

        mapping_list = [AutoMapping(megatron_param=k, hf_param=v) for k, v in param_mappings.items()]

        # Attention (non-MLA fallback: combined QKV).
        mapping_list.extend(
            [
                QKVMapping(
                    megatron_param="decoder.layers.*.self_attention.linear_qkv.weight",
                    q="model.layers.*.self_attn.q_proj.weight",
                    k="model.layers.*.self_attn.k_proj.weight",
                    v="model.layers.*.self_attn.v_proj.weight",
                ),
                QKVMapping(
                    megatron_param="decoder.layers.*.self_attention.linear_qkv.bias",
                    q="model.layers.*.self_attn.q_proj.bias",
                    k="model.layers.*.self_attn.k_proj.bias",
                    v="model.layers.*.self_attn.v_proj.bias",
                ),
                # Dense MLP gate+up → fc1
                GatedMLPMapping(
                    megatron_param="decoder.layers.*.mlp.linear_fc1.weight",
                    gate="model.layers.*.mlp.gate_proj.weight",
                    up="model.layers.*.mlp.up_proj.weight",
                ),
                # Shared expert gate+up → fc1
                GatedMLPMapping(
                    megatron_param="decoder.layers.*.mlp.shared_experts.linear_fc1.weight",
                    gate="model.layers.*.mlp.shared_experts.gate_proj.weight",
                    up="model.layers.*.mlp.shared_experts.up_proj.weight",
                ),
            ]
        )

        # MoE expert weights (per-expert format: experts.N.gate_proj / up_proj /
        # down_proj).
        mapping_list.extend(
            [
                GatedMLPMapping(
                    megatron_param="decoder.layers.*.mlp.experts.linear_fc1.weight*",
                    gate="model.layers.*.mlp.experts.*.gate_proj.weight",
                    up="model.layers.*.mlp.experts.*.up_proj.weight",
                ),
                AutoMapping(
                    megatron_param="decoder.layers.*.mlp.experts.linear_fc2.weight*",
                    hf_param="model.layers.*.mlp.experts.*.down_proj.weight",
                ),
            ]
        )

        # --- MTP (Multi-Token Prediction) layer mappings ---
        # ``self.hf_config`` is set by the dispatch system before this method is
        # called. When the bridge is constructed standalone (e.g. in unit
        # tests), ``hf_config`` may be unset, in which case MTP mappings are
        # skipped — that matches the reference behavior.
        hf_config = getattr(self, "hf_config", None)
        if hf_config is None:
            logger.warning("No HF config found on bridge instance, skipping MTP mappings.")
            return MegatronMappingRegistry(*mapping_list)

        num_mtp_layers = getattr(hf_config, "num_nextn_predict_layers", 0) or 0
        num_transformer_layers = hf_config.num_hidden_layers

        # Layer-specific mappings reused for the MTP mtp_model_layer. These
        # mirror the decoder layer mappings but with an mtp_model_layer prefix.
        mtp_layer_mappings = {
            # Attention layernorm
            "self_attention.linear_qkv.layer_norm_weight": "input_layernorm.weight",
            "input_layernorm.weight": "input_layernorm.weight",
            # Attention output
            "self_attention.linear_proj.weight": "self_attn.o_proj.weight",
            # Post-attention layernorm (MoE layer uses pre_mlp_layernorm)
            "pre_mlp_layernorm.weight": "post_attention_layernorm.weight",
            # MLA weights
            "self_attention.linear_q_down_proj.weight": "self_attn.q_a_proj.weight",
            "self_attention.linear_q_up_proj.weight": "self_attn.q_b_proj.weight",
            "self_attention.linear_q_up_proj.layer_norm_weight": "self_attn.q_a_layernorm.weight",
            "self_attention.q_layernorm.weight": "self_attn.q_a_layernorm.weight",
            "self_attention.linear_kv_down_proj.weight": "self_attn.kv_a_proj_with_mqa.weight",
            "self_attention.linear_kv_up_proj.weight": "self_attn.kv_b_proj.weight",
            "self_attention.linear_kv_up_proj.layer_norm_weight": "self_attn.kv_a_layernorm.weight",
            "self_attention.kv_layernorm.weight": "self_attn.kv_a_layernorm.weight",
            # DSA indexer — weights live directly on DSAMLASelfAttention.
            "self_attention.wq_b.weight": "self_attn.indexer.wq_b.weight",
            "self_attention.wk.weight": "self_attn.indexer.wk.weight",
            "self_attention.k_norm.weight": "self_attn.indexer.k_norm.weight",
            "self_attention.k_norm.bias": "self_attn.indexer.k_norm.bias",
            "self_attention.weights_proj.weight": "self_attn.indexer.weights_proj.weight",
            # MoE router
            "mlp.router.weight": "mlp.gate.weight",
            "mlp.router.expert_bias": "mlp.gate.e_score_correction_bias",
            # MoE shared experts
            "mlp.shared_experts.router.weight": "mlp.shared_experts.gate.weight",
            "mlp.shared_experts.linear_fc2.weight": "mlp.shared_experts.down_proj.weight",
        }

        for mtp_layer in range(num_mtp_layers):
            hf_layer_idx = mtp_layer + num_transformer_layers

            # AutoMapping for layer-specific params.
            for megatron_suffix, hf_suffix in mtp_layer_mappings.items():
                mapping_list.append(
                    AutoMapping(
                        megatron_param=f"mtp.layers.{mtp_layer}.mtp_model_layer.{megatron_suffix}",
                        hf_param=f"model.layers.{hf_layer_idx}.{hf_suffix}",
                    )
                )

            # MTP-specific mappings (enorm, hnorm, eh_proj, final_layernorm).
            mapping_list.extend(
                [
                    AutoMapping(
                        megatron_param=f"mtp.layers.{mtp_layer}.enorm.weight",
                        hf_param=f"model.layers.{hf_layer_idx}.enorm.weight",
                    ),
                    AutoMapping(
                        megatron_param=f"mtp.layers.{mtp_layer}.hnorm.weight",
                        hf_param=f"model.layers.{hf_layer_idx}.hnorm.weight",
                    ),
                    AutoMapping(
                        megatron_param=f"mtp.layers.{mtp_layer}.eh_proj.weight",
                        hf_param=f"model.layers.{hf_layer_idx}.eh_proj.weight",
                    ),
                    AutoMapping(
                        megatron_param=f"mtp.layers.{mtp_layer}.final_layernorm.weight",
                        hf_param=f"model.layers.{hf_layer_idx}.shared_head.norm.weight",
                    ),
                ]
            )

            # Shared expert gate+up → fc1.
            mapping_list.append(
                GatedMLPMapping(
                    megatron_param=f"mtp.layers.{mtp_layer}.mtp_model_layer.mlp.shared_experts.linear_fc1.weight",
                    gate=f"model.layers.{hf_layer_idx}.mlp.shared_experts.gate_proj.weight",
                    up=f"model.layers.{hf_layer_idx}.mlp.shared_experts.up_proj.weight",
                )
            )

            # MoE expert weights.
            mapping_list.extend(
                [
                    GatedMLPMapping(
                        megatron_param=f"mtp.layers.{mtp_layer}.mtp_model_layer.mlp.experts.linear_fc1.weight*",
                        gate=f"model.layers.{hf_layer_idx}.mlp.experts.*.gate_proj.weight",
                        up=f"model.layers.{hf_layer_idx}.mlp.experts.*.up_proj.weight",
                    ),
                    AutoMapping(
                        megatron_param=f"mtp.layers.{mtp_layer}.mtp_model_layer.mlp.experts.linear_fc2.weight*",
                        hf_param=f"model.layers.{hf_layer_idx}.mlp.experts.*.down_proj.weight",
                    ),
                ]
            )

        return MegatronMappingRegistry(*mapping_list)
