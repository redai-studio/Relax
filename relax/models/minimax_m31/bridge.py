# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from argparse import Namespace
from copy import deepcopy
from dataclasses import dataclass, field, fields
from typing import Any

from megatron.bridge.models.conversion.mapping_registry import MegatronMappingRegistry
from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge
from megatron.bridge.models.conversion.param_mapping import ReplicatedMapping
from megatron.bridge.models.hf_pretrained.causal_lm import PreTrainedCausalLM
from megatron.bridge.models.minimax_m3.minimax_m3_bridge import (
    MiniMaxM3Bridge,
    MiniMaxM3VLModelProvider,
    minimax_m3_block_spec,
)
from megatron.bridge.models.minimax_m3.modeling_minimax_m3_vl import MiniMaxM3VLModel
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_block import TransformerBlockSubmodules


def _get_msa_config(config: Any) -> dict[str, Any] | None:
    text_config = getattr(config, "text_config", None)
    if isinstance(text_config, dict):
        sparse = text_config.get("sparse_attention_config")
    else:
        sparse = getattr(text_config, "sparse_attention_config", None)
    # MiniMax-M3 has three dense attention layers. Preserve its upstream provider
    # and its frozen checkpoint-only indexer mappings.
    if not sparse or not sparse.get("sparse_attention_freq") or not all(sparse["sparse_attention_freq"]):
        return None
    return sparse


@dataclass
class MiniMaxM31ModelProvider(MiniMaxM3VLModelProvider):
    msa_config: dict[str, Any] = field(default_factory=dict)
    relax_minimax_text_sft: bool = field(default=True, init=False)

    def configure_runtime(self, args: Namespace) -> None:
        if args.loss_type != "sft" or args.lora_rank <= 0 or args.multimodal_keys:
            raise ValueError("MiniMax-M3.1-Flash requires text-only LoRA SFT.")
        if args.context_parallel_size != 1 or args.pipeline_model_parallel_size != 1 or args.dynamic_context_parallel:
            raise ValueError("MiniMax-M3.1-Flash requires PP=CP=1 and static context parallelism.")
        if args.qkv_format != "thd" or args.allgather_cp:
            raise ValueError("MiniMax-M3.1-Flash requires regular THD sequence packing.")
        if args.tensor_model_parallel_size not in (1, 2, 4):
            raise ValueError("MiniMax-M3.1-Flash supports tensor parallel sizes 1, 2, and 4.")
        if args.apply_rope_fusion:
            raise ValueError("MiniMax-M3.1-Flash requires --no-rope-fusion.")


def minimax_m31_block_spec(config: MiniMaxM31ModelProvider, **kwargs: Any) -> TransformerBlockSubmodules:
    from megatron.core.extensions.transformer_engine import TEColumnParallelLinear, TENorm

    from .attention import MiniMaxM31Attention, MiniMaxRMSNorm

    spec = minimax_m3_block_spec(config, **kwargs)
    for layer_id, original in enumerate(spec.layer_specs):
        layer = deepcopy(original)
        attention = layer.submodules.self_attention
        attention.submodules.linear_qkv = TEColumnParallelLinear
        attention.submodules.q_layernorm = MiniMaxRMSNorm
        attention.submodules.k_layernorm = MiniMaxRMSNorm
        layer.submodules.input_layernorm = TENorm
        layer.submodules.self_attention = ModuleSpec(
            module=MiniMaxM31Attention, params=attention.params or {}, submodules=attention.submodules
        )
        spec.layer_specs[layer_id] = layer
    return spec


@MegatronModelBridge.register_bridge(
    source="MiniMaxM3SparseForConditionalGeneration",
    target=MiniMaxM3VLModel,
    provider=MiniMaxM3VLModelProvider,
    model_type="minimax_m3_vl",
)
class MiniMaxM31Bridge(MiniMaxM3Bridge):
    """Use the M3.1 indexers in attention and retain upstream M3 conversion."""

    def provider_bridge(self, hf_pretrained: PreTrainedCausalLM) -> MiniMaxM3VLModelProvider:
        hf_config = hf_pretrained.config
        sparse = _get_msa_config(hf_config)
        provider = super().provider_bridge(hf_pretrained)
        if sparse is None:
            return provider
        if getattr(hf_config, "quantization_config", None):
            raise ValueError(
                "MiniMax-M3.1-Flash SFT requires a BF16 checkpoint. Run scripts/tools/convert_minimax_m3_to_bf16.py."
            )
        if (
            not sparse["use_sparse_attention"]
            or sparse["sparse_score_type"] != "max"
            or not all(sparse["sparse_disable_index_value"])
            or sparse["sparse_num_index_heads"] != provider.num_query_groups
        ):
            raise ValueError("Unsupported MiniMax-M3.1-Flash sparse-attention configuration.")
        provider_values = {f.name: getattr(provider, f.name) for f in fields(MiniMaxM3VLModelProvider) if f.init}
        provider_values["lightning_indexer_layers"] = []
        provider_values["transformer_layer_spec"] = minimax_m31_block_spec
        return MiniMaxM31ModelProvider(**provider_values, msa_config=dict(sparse))

    def mapping_registry(self) -> MegatronMappingRegistry:
        upstream = super().mapping_registry()
        if _get_msa_config(self.hf_config) is None:
            return upstream
        mappings = [
            mapping for mapping in upstream.mappings if not mapping.megatron_param.startswith("lightning_indexers.")
        ]
        for megatron_name, hf_name in (
            ("index_q_proj.weight", "index_q_proj.weight"),
            ("index_k_proj.weight", "index_k_proj.weight"),
            ("index_q_norm", "index_q_norm.weight"),
            ("index_k_norm", "index_k_norm.weight"),
        ):
            mappings.append(
                ReplicatedMapping(
                    f"language_model.decoder.layers.*.self_attention.indexer.{megatron_name}",
                    f"language_model.model.layers.*.self_attn.{hf_name}",
                )
            )
        return MegatronMappingRegistry(*mappings)
