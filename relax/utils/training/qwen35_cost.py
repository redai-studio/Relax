# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Qwen3.5 dense/MoE sample costs for scheduling, not hardware FLOPs/MFU.

Count multiply-add as two operations. GDN uses three state contractions per
real token; chunk/WY overhead, padding, recomputation, communication, lookup,
normalization and elementwise operations are excluded. Head rows use all-token
execution. Costs are independent of TP/PP/EP.
"""

from argparse import Namespace
from numbers import Integral
from typing import Any


def _positive(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
        raise ValueError(f"Qwen3.5 cost requires a positive integer {name}, got {value!r}")
    return int(value)


def _layer_types(text: Any) -> list[str]:
    count = _positive(text.num_hidden_layers, "num_hidden_layers")
    types = getattr(text, "layer_types", None)
    if types is None:
        interval = _positive(getattr(text, "full_attention_interval", 4), "full_attention_interval")
        types = ["full_attention" if (i + 1) % interval == 0 else "linear_attention" for i in range(count)]
    if len(types) != count or any(t not in ("full_attention", "linear_attention") for t in types):
        raise ValueError("Qwen3.5 layer_types must cover every layer with full_attention or linear_attention")
    return list(types)


def _layer_forward(text: Any, length: int, layer_type: str) -> tuple[float, float]:
    """Return (parameterized projection/conv FLOPs, attention/state FLOPs)."""
    hidden = _positive(text.hidden_size, "hidden_size")
    if layer_type == "full_attention":
        heads = _positive(text.num_attention_heads, "num_attention_heads")
        dim = _positive(getattr(text, "head_dim", None) or hidden // heads, "head_dim")
        q = heads * dim
        kv = _positive(text.num_key_value_heads, "num_key_value_heads") * dim
        # q_proj emits both Q and the output gate; o_proj consumes Q width.
        projections = hidden * (3 * q + 2 * kv)
        attention = 2 * length * (length + 1) * q
    else:
        key_dim = _positive(text.linear_key_head_dim, "linear_key_head_dim")
        value_dim = _positive(text.linear_value_head_dim, "linear_value_head_dim")
        value_heads = _positive(text.linear_num_value_heads, "linear_num_value_heads")
        key = _positive(text.linear_num_key_heads, "linear_num_key_heads") * key_dim
        value = value_heads * value_dim
        projections = hidden * (2 * key + 3 * value + 2 * value_heads)
        projections += _positive(text.linear_conv_kernel_dim, "linear_conv_kernel_dim") * (2 * key + value)
        # k^T state prediction, rank-one state update and q^T state readout.
        attention = 6 * length * value_heads * key_dim * value_dim
    experts = getattr(text, "num_experts", 0) or 0
    if experts:
        experts = _positive(experts, "num_experts")
        topk = _positive(text.num_experts_per_tok, "num_experts_per_tok")
        if topk > experts:
            raise ValueError("num_experts_per_tok cannot exceed num_experts")
        shared = _positive(text.shared_expert_intermediate_size, "shared_expert_intermediate_size")
        projections += hidden * experts + hidden
        projections += 3 * hidden * (topk * _positive(text.moe_intermediate_size, "moe_intermediate_size") + shared)
    else:
        projections += 3 * hidden * _positive(text.intermediate_size, "intermediate_size")
    return float(2 * length * projections), float(attention)


def estimate_sample_costs(config: Any, rollout_data: dict[str, Any], args: Namespace) -> list[float]:
    """Estimate text training work for the supported Dense/MoE models."""
    if (getattr(args, "lora_rank", 0) or 0) > 0 or any(
        getattr(args, name, None) for name in ("freeze_params_name_list", "only_train_params_name_list")
    ):
        raise ValueError("Qwen3.5 FLOPs balancing does not support LoRA or custom parameter masks")
    text = getattr(config, "text_config", config)
    types = _layer_types(text)
    if getattr(text, "mlp_only_layers", None) or getattr(text, "decoder_sparse_step", 1) != 1:
        raise ValueError("Qwen3.5 FLOPs balancing requires the standard dense or all-MoE layer layout")
    lengths = rollout_data["total_lengths"]
    inputs = rollout_data.get("multimodal_train_inputs")
    if inputs is None:
        inputs = [None] * len(lengths)
    if len(inputs) != len(lengths):
        raise ValueError("multimodal_train_inputs must contain one entry per sample")
    if getattr(args, "freeze_language_model", False) or (
        getattr(args, "enable_mtp_training", False) and (getattr(args, "mtp_num_layers", 0) or 0)
    ):
        raise ValueError("FLOPs scheduling currently supports trainable text without MTP")
    if any(media is not None and (not isinstance(media, dict) or bool(media)) for media in inputs):
        raise ValueError("FLOPs scheduling currently supports text-only samples")
    hidden = _positive(text.hidden_size, "hidden_size")
    vocab = _positive(text.vocab_size, "vocab_size")
    costs = []
    for length in lengths:
        length = _positive(length, "sequence length")
        linears, attention = zip(*(_layer_forward(text, length, kind) for kind in types), strict=True)
        costs.append(3 * (sum(linears) + sum(attention) + 2 * length * hidden * vocab))
    return costs
