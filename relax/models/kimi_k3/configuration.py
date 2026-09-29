# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Preserve HF architecture while applying Relax runtime settings."""

from argparse import Namespace
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
