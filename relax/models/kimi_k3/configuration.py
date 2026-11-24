# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Preserve HF architecture while applying Relax runtime settings."""

from argparse import Namespace
from collections.abc import Mapping
from typing import Any


ARCHITECTURE_KEYS = {
    "num_layers",
    "moe_layer_freq",
    "q_lora_rank",
    "kv_lora_rank",
    "qk_head_dim",
    "qk_pos_emb_head_dim",
    "v_head_dim",
    "rotary_scaling_factor",
    "rotary_base",
    "moe_router_pre_softmax",
    "moe_router_enable_expert_bias",
    "moe_router_bias_update_rate",
    "moe_router_dtype",
    "moe_router_load_balancing_type",
    "moe_aux_loss_coeff",
    "moe_shared_expert_intermediate_size",
    "moe_router_topk",
    "moe_router_num_groups",
    "moe_router_group_topk",
    "moe_router_topk_scaling_factor",
    "moe_router_score_function",
    "moe_ffn_hidden_size",
    "activation_func_clamp_shared_expert",
}


def kimi_k31_dsa_config(text_config: Any) -> dict[str, Any]:
    """Validate the supported K3.1 indexer and its one-based layer layout."""
    source = getattr(text_config, "dsa_config", None)
    if not isinstance(source, Mapping):
        raise ValueError("Kimi K3.1 requires text_config.dsa_config.")
    config = dict(source)
    for name, supported in (
        ("index_rope_dim", 0),
        ("index_k_norm_type", "layernorm"),
        ("index_topk_freq", 1),
    ):
        value = config.setdefault(name, supported)
        if value != supported:
            raise ValueError(f"Kimi K3.1 requires {name}={supported!r}; got {value!r}.")
    if not getattr(text_config, "mla_use_nope", False):
        raise ValueError("Kimi K3.1 requires mla_use_nope=True.")
    for name in ("index_topk", "index_num_heads", "index_head_dim"):
        value = config.get(name)
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"Kimi K3.1 requires a positive integer {name}; got {value!r}.")

    num_layers = text_config.num_hidden_layers
    linear_config = text_config.linear_attn_config
    kda_layers = linear_config["kda_layers"]
    if len(set(kda_layers)) != len(kda_layers) or any(
        not isinstance(layer, int) or isinstance(layer, bool) or not 1 <= layer <= num_layers for layer in kda_layers
    ):
        raise ValueError("Kimi K3.1 kda_layers must contain unique one-based layer numbers.")
    mla_layers = set(range(1, num_layers + 1)) - set(kda_layers)
    dsa_layers = config.get("dsa_layers")
    if (
        not isinstance(dsa_layers, (list, tuple))
        or any(not isinstance(layer, int) or isinstance(layer, bool) for layer in dsa_layers)
        or len(set(dsa_layers)) != len(dsa_layers)
        or set(dsa_layers) != mla_layers
        or set(linear_config.get("full_attn_layers", mla_layers)) != mla_layers
    ):
        raise ValueError("Kimi K3.1 dsa_layers must contain exactly the non-KDA layer numbers (one-based).")
    config["dsa_layers"] = tuple(dsa_layers)
    return config


def configure_runtime(provider: Any, args: Namespace) -> None:
    provider.variable_seq_lengths = True
    # Keep processor/rebuild payloads in HF format; use MoonViT's host-only
    # geometry key at the training batch boundary, before any device transfer.
    args.multimodal_input_key_map = {"image_grid_thw": "grid_thws"}
    if getattr(args, "dynamic_context_parallel", False):
        raise ValueError("Kimi K3 dynamic context parallelism is not supported; use a fixed CP size.")
    if getattr(provider, "context_parallel_size", 1) > 1 and (
        getattr(args, "allgather_cp", False) or getattr(args, "cp_partition_mode", "zigzag") != "zigzag"
    ):
        raise ValueError("Kimi K3 context parallelism requires zigzag partitioning without --allgather-cp.")
    if getattr(provider, "virtual_pipeline_model_parallel_size", None) is not None:
        raise ValueError("Kimi K3 does not support virtual pipeline parallelism yet.")
