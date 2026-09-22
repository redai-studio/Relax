# Copyright (c) 2026 Relax Authors. All Rights Reserved.


import torch
import torch.nn.functional as F
from megatron.bridge.models.conversion.mapping_registry import MegatronMappingRegistry
from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge
from megatron.bridge.models.conversion.param_mapping import AutoMapping, GatedMLPMapping, QKVMapping, ReplicatedMapping
from megatron.bridge.models.hf_pretrained.causal_lm import PreTrainedCausalLM
from transformers import Qwen3OmniMoeForConditionalGeneration

from relax.models.qwen_omni.modeling_qwen3_omni.model import Qwen3OmniMoeModel
from relax.models.qwen_omni.qwen3_omni_provider import Qwen3OmniModelProvider


@MegatronModelBridge.register_bridge(source=Qwen3OmniMoeForConditionalGeneration, target=Qwen3OmniMoeModel)
class Qwen3OmniMoEBridge(MegatronModelBridge):
    """Megatron Bridge for Qwen3-VL MoE (Mixture of Experts) Conditional
    Generation.

    This bridge handles the conversion between HuggingFace Qwen3VLMoEForConditionalGeneration
    and Megatron-Core Qwen3VL MoE model formats, including weight mappings and
    configuration translation for vision-language MoE models.

    The weight mappings handle:
    - Vision model weights (same as dense model)
    - Language model MoE layers with expert routing
    - Shared embeddings and output layers
    - QK layernorm specific to Qwen3 architecture

    This bridge works with any Qwen3VL MoE model size and automatically extracts
    the MoE configuration from the HuggingFace model.

    Example:
        >>> from megatron.bridge import AutoBridge
        >>> bridge = AutoBridge.from_hf_pretrained("Qwen/Qwen3-VL-30B-A3B-Instruct")
        >>> provider = bridge.to_megatron_provider()
    """

    # copied from https://github.com/fzyzcjy/Megatron-Bridge/blob/6b1b80cdd3f5387e378545399287bf4a21a56fe0/src/megatron/bridge/models/gpt_oss/gpt_oss_bridge.py#L54
    def __init__(self):
        super().__init__()
        self.hf_weights_cache = {}

    def provider_bridge(self, hf_pretrained: PreTrainedCausalLM) -> Qwen3OmniModelProvider:
        """Create a Qwen3OmniModelProvider from a HuggingFace pretrained MoE
        model.

        Args:
            hf_pretrained: HuggingFace pretrained VLM MoE model

        Returns:
            Qwen3OmniModelProvider configured with the HF MoE model's parameters
        """
        # to check
        # hf_pretrained.config
        hf_config = hf_pretrained.config.thinker_config
        text_config = hf_config.text_config

        # Get the model dtype from text config
        model_dtype = self.dtype_from_hf(hf_config, default=torch.float32)

        # Set vision config dtype to match the language model dtype
        # This ensures vision model parameters are initialized in the same dtype
        audio_config = hf_config.audio_config
        audio_config.torch_dtype = model_dtype
        vision_config = hf_config.vision_config
        vision_config.torch_dtype = model_dtype

        head_dim = getattr(text_config, "head_dim", text_config.hidden_size // text_config.num_attention_heads)
        provider = Qwen3OmniModelProvider(
            num_layers=text_config.num_hidden_layers,
            hidden_size=text_config.hidden_size,
            ffn_hidden_size=text_config.intermediate_size,  # Dense FFN size (for non-MoE layers if any)
            moe_ffn_hidden_size=text_config.moe_intermediate_size,  # Expert FFN size
            num_attention_heads=text_config.num_attention_heads,
            num_query_groups=text_config.num_key_value_heads,  # GQA configuration
            head_dim=head_dim,
            kv_channels=head_dim,  # Must explicitly set kv_channels for MCore TransformerConfig
            init_method_std=text_config.initializer_range,
            layernorm_epsilon=text_config.rms_norm_eps,
            gated_linear_unit=True,  # Qwen3 MoE uses gated linear units
            make_vocab_size_divisible_by=self.make_vocab_size_divisible_by(text_config.vocab_size),
            rotary_base=getattr(text_config, "rope_theta", 1000000.0),  # Default Qwen3 rope theta
            share_embeddings_and_output_weights=getattr(text_config, "tie_word_embeddings", False),
            vocab_size=text_config.vocab_size,
            seq_length=text_config.max_position_embeddings,
            fp16=(model_dtype == torch.float16),
            bf16=(model_dtype == torch.bfloat16),
            params_dtype=model_dtype,
            # Qwen3 specific parameters — match Qwen3VLMoEBridge settings
            normalization="RMSNorm",  # Qwen3 uses RMSNorm (no bias in layernorms)
            activation_func=F.silu,  # Qwen3 uses SwiGLU (silu + gated_linear_unit)
            add_qkv_bias=text_config.attention_bias,  # Qwen3 can have bias in QKV
            add_bias_linear=False,  # Qwen3 has no bias in linear layers (o_proj, MLP, router)
            hidden_dropout=0.0,  # Qwen3 uses no hidden dropout
            qk_layernorm=True,  # Qwen3 uses QK layernorm
            # MoE specific parameters
            num_moe_experts=text_config.num_experts,
            moe_router_topk=text_config.num_experts_per_tok,
            moe_grouped_gemm=True,
            moe_router_load_balancing_type="aux_loss",
            moe_aux_loss_coeff=1e-3,
            decoder_sparse_step=getattr(text_config, "decoder_sparse_step", 1),  # Default to every layer being MoE
            mlp_only_layers=getattr(text_config, "mlp_only_layers", []),  # Default to all layers using MoE
            # Vision configuration
            audio_config=audio_config,
            vision_config=vision_config,
            # Store the original HF text config for RoPE initialization
            hf_text_config=text_config,
            # Vision-Language token IDs
            bos_token_id=getattr(text_config, "bos_token_id", 151643),
            eos_token_id=getattr(text_config, "eos_token_id", 151645),
            vision_start_token_id=getattr(hf_config, "vision_start_token_id", 151652),
            vision_end_token_id=getattr(hf_config, "vision_end_token_id", 151653),
            image_token_id=getattr(hf_config, "image_token_id", 151655),
            video_token_id=getattr(hf_config, "video_token_id", 151656),
            # audio
            audio_token_id=hf_config.audio_token_id,
            audio_start_token_id=hf_config.audio_start_token_id,
            audio_end_token_id=hf_config.audio_end_token_id,
            # MRoPE configuration for multimodal position embeddings
            mrope_section=getattr(text_config, "rope_scaling", {}).get("mrope_section", [24, 20, 20]),
            position_id_per_seconds=hf_config.position_id_per_seconds,
            spatial_merge_size=vision_config.spatial_merge_size,
        )

        return provider

    def mapping_registry(self) -> MegatronMappingRegistry:
        """Return MegatronMappingRegistry containing parameter mappings for MoE
        models.

        The MoE mappings include:
        1. Standard language model mappings (embeddings, layer norms, output)
        2. Vision model mappings (same as dense model)
        3. QKV mappings with QK layernorm
        4. MoE-specific mappings:
           - Router weights for expert selection
           - Expert MLPs (multiple experts per layer)
           - Pre-MLP layernorm
        5. Deepstack visual merger mappings

        Returns:
            MegatronMappingRegistry with all MoE parameter mappings
        """
        # Language model direct mappings (same as dense model)
        # NOTE: Megatron side (left) uses param names from Qwen3OmniMoeModel (no "thinker." prefix),
        #       HF side (right) uses param names from Qwen3OmniMoeForConditionalGeneration (with "thinker." prefix).
        param_mappings = {
            # Embeddings and output layers
            "language_model.embedding.word_embeddings.weight": "thinker.model.embed_tokens.weight",
            "language_model.output_layer.weight": "thinker.lm_head.weight",
            "language_model.decoder.final_layernorm.weight": "thinker.model.norm.weight",
            # Layer normalization for attention (TE format - fused into linear)
            "language_model.decoder.layers.*.self_attention.linear_qkv.layer_norm_weight": "thinker.model.layers.*.input_layernorm.weight",
            # MoE-specific: pre-MLP layernorm
            "language_model.decoder.layers.*.pre_mlp_layernorm.weight": "thinker.model.layers.*.post_attention_layernorm.weight",
            # Dense MLP layer norm (for non-MoE layers, i.e. mlp_only_layers)
            "language_model.decoder.layers.*.mlp.linear_fc1.layer_norm_weight": "thinker.model.layers.*.post_attention_layernorm.weight",
            # Attention output projection
            "language_model.decoder.layers.*.self_attention.linear_proj.weight": "thinker.model.layers.*.self_attn.o_proj.weight",
            # QK layernorm weights (Qwen3 specific)
            "language_model.decoder.layers.*.self_attention.q_layernorm.weight": "thinker.model.layers.*.self_attn.q_norm.weight",
            "language_model.decoder.layers.*.self_attention.k_layernorm.weight": "thinker.model.layers.*.self_attn.k_norm.weight",
            # MoE router weights
            "language_model.decoder.layers.*.mlp.router.weight": "thinker.model.layers.*.mlp.gate.weight",
            # MoE router expert bias
            "language_model.decoder.layers.*.mlp.router.expert_bias": "thinker.model.layers.*.mlp.gate.e_score_correction_bias",
            # Dense MLP down projection (for non-MoE layers, i.e. mlp_only_layers)
            "language_model.decoder.layers.*.mlp.linear_fc2.weight": "thinker.model.layers.*.mlp.down_proj.weight",
            # Shared expert down projection
            "language_model.decoder.layers.*.mlp.shared_experts.linear_fc2.weight": "thinker.model.layers.*.mlp.shared_expert.down_proj.weight",
            # Shared expert gate weight
            "language_model.decoder.layers.*.mlp.shared_experts.gate_weight": "thinker.model.layers.*.mlp.shared_expert_gate.weight",
        }

        mapping_list = []

        # Convert simple 1:1 mappings to AutoMapping objects
        for megatron_param, hf_param in param_mappings.items():
            mapping_list.append(AutoMapping(megatron_param=megatron_param, hf_param=hf_param))

        # Add special mappings that require parameter transformation
        mapping_list.extend(
            [
                # Audio and vision model weights are replicated directly (HF encoders)
                ReplicatedMapping(
                    megatron_param="audio_model.**",
                    hf_param="thinker.audio_tower.**",
                ),
                ReplicatedMapping(
                    megatron_param="vision_model.**",
                    hf_param="thinker.visual.**",
                ),
                # QKV mapping: Combine separate Q, K, V matrices
                QKVMapping(
                    megatron_param="language_model.decoder.layers.*.self_attention.linear_qkv.weight",
                    q="thinker.model.layers.*.self_attn.q_proj.weight",
                    k="thinker.model.layers.*.self_attn.k_proj.weight",
                    v="thinker.model.layers.*.self_attn.v_proj.weight",
                ),
                # QKV bias mapping (if attention_bias is True)
                QKVMapping(
                    megatron_param="language_model.decoder.layers.*.self_attention.linear_qkv.bias",
                    q="thinker.model.layers.*.self_attn.q_proj.bias",
                    k="thinker.model.layers.*.self_attn.k_proj.bias",
                    v="thinker.model.layers.*.self_attn.v_proj.bias",
                ),
                # Expert mappings for TEGroupedMLP
                GatedMLPMapping(
                    megatron_param="language_model.decoder.layers.*.mlp.experts.linear_fc1.weight*",
                    gate="thinker.model.layers.*.mlp.experts.*.gate_proj.weight",
                    up="thinker.model.layers.*.mlp.experts.*.up_proj.weight",
                ),
                AutoMapping(
                    megatron_param="language_model.decoder.layers.*.mlp.experts.linear_fc2.weight*",
                    hf_param="thinker.model.layers.*.mlp.experts.*.down_proj.weight",
                ),
                # Expert mappings for SequentialMLP (used by quantization)
                GatedMLPMapping(
                    megatron_param="language_model.decoder.layers.*.mlp.experts.local_experts.*.linear_fc1.weight",
                    gate="thinker.model.layers.*.mlp.experts.*.gate_proj.weight",
                    up="thinker.model.layers.*.mlp.experts.*.up_proj.weight",
                ),
                AutoMapping(
                    megatron_param="language_model.decoder.layers.*.mlp.experts.local_experts.*.linear_fc2.weight",
                    hf_param="thinker.model.layers.*.mlp.experts.*.down_proj.weight",
                ),
                # Dense MLP gate+up (for non-MoE layers, i.e. mlp_only_layers)
                GatedMLPMapping(
                    megatron_param="language_model.decoder.layers.*.mlp.linear_fc1.weight",
                    gate="thinker.model.layers.*.mlp.gate_proj.weight",
                    up="thinker.model.layers.*.mlp.up_proj.weight",
                ),
                # Shared expert gate+up
                GatedMLPMapping(
                    megatron_param="language_model.decoder.layers.*.mlp.shared_experts.linear_fc1.weight",
                    gate="thinker.model.layers.*.mlp.shared_expert.gate_proj.weight",
                    up="thinker.model.layers.*.mlp.shared_expert.up_proj.weight",
                ),
            ]
        )

        return MegatronMappingRegistry(*mapping_list)
