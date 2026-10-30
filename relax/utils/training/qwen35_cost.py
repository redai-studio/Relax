# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Qwen3.5 dense/MoE sample costs for scheduling, not hardware FLOPs/MFU.

Count multiply-add as two operations. GDN uses three state contractions per
real token; chunk/WY overhead, padding, recomputation, communication, lookup,
normalization and elementwise operations are excluded. Head rows use all-token
execution, including each enabled MTP depth. Costs are independent of TP/PP/EP.
"""

from argparse import Namespace
from collections.abc import Sequence
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


def _grids(inputs: dict[str, Any] | None) -> list[tuple[int, int, int]]:
    result = []
    inputs = inputs or {}
    for name, pixels in (("image_grid_thw", "pixel_values"), ("video_grid_thw", "pixel_values_videos")):
        value = inputs.get(name)
        if value is None:
            if inputs.get(pixels) is not None:
                raise ValueError(f"Qwen3.5 FLOPs balancing requires {name} for {pixels}")
            continue
        if hasattr(value, "device") and value.device.type != "cpu":
            raise ValueError("FLOPs balancing requires CPU grid metadata; CUDA copies are not allowed")
        if hasattr(value, "tolist"):
            value = value.tolist()
        for row in value:
            if len(row) != 3:
                raise ValueError(f"{name} must have shape [N, 3]")
            result.append(tuple(_positive(v, name) for v in row))
    return result


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


def _vision_cost(vision: Any, grids: Sequence[tuple[int, int, int]], args: Namespace) -> float:
    if not grids:
        return 0.0
    if vision is None:
        raise ValueError("Visual grid metadata requires a Qwen3.5 vision_config")
    hidden = _positive(vision.hidden_size, "vision.hidden_size")
    merge = _positive(vision.spatial_merge_size, "vision.spatial_merge_size")
    if any(h % merge or w % merge for _, h, w in grids):
        raise ValueError("Visual grid height/width must be divisible by spatial_merge_size")
    patches = sum(t * h * w for t, h, w in grids)
    # Vision attention is frame-local, not attention across all video frames.
    squares = sum(t * (h * w) ** 2 for t, h, w in grids)
    merged = patches // (merge * merge)
    patch_weights = hidden * _positive(vision.in_channels, "vision.in_channels")
    patch_weights *= _positive(vision.temporal_patch_size, "vision.temporal_patch_size")
    patch_weights *= _positive(vision.patch_size, "vision.patch_size") ** 2
    blocks = _positive(vision.depth, "vision.depth") * (
        2
        * patches
        * (4 * hidden * hidden + 2 * hidden * _positive(vision.intermediate_size, "vision.intermediate_size"))
        + 4 * squares * hidden
    )
    width = hidden * merge * merge
    merger = 2 * merged * width * (width + _positive(vision.out_hidden_size, "vision.out_hidden_size"))
    merger *= 1 + len(getattr(vision, "deepstack_visual_indexes", None) or [])
    encoder_grad = not getattr(args, "freeze_vision_model", False)
    projector_grad = not getattr(args, "freeze_vision_projection", False)
    # Merger norm is trainable with the projector, so fc1 also needs dX.
    return (
        2 * patches * patch_weights * (1 + int(encoder_grad))
        + blocks * (3 if encoder_grad else 1)
        + merger * (1 + int(encoder_grad or projector_grad) + int(projector_grad))
    )


def estimate_sample_costs(config: Any, rollout_data: dict[str, Any], args: Namespace) -> list[float]:
    """Estimate ideal training graph work, honoring independent freeze
    flags."""
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
    train_text = not getattr(args, "freeze_language_model", False)
    mtp = (getattr(args, "mtp_num_layers", 0) or 0) if getattr(args, "enable_mtp_training", False) else 0
    if mtp:
        _positive(mtp, "mtp_num_layers")
        if not train_text or types[-1] != "full_attention" or getattr(args, "pipeline_model_parallel_layout", None):
            raise ValueError(
                "Qwen3.5 MTP cost requires trainable language, a full-attention last layer and no custom PP layout"
            )
    hidden = _positive(text.hidden_size, "hidden_size")
    vocab = _positive(text.vocab_size, "vocab_size")
    costs = []
    for length, media in zip(lengths, inputs, strict=True):
        length = _positive(length, "sequence length")
        grids = _grids(media)
        media_grad = bool(grids) and not (
            getattr(args, "freeze_vision_model", False) and getattr(args, "freeze_vision_projection", False)
        )
        backward = train_text or media_grad
        linear_factor = 1 + int(backward) + int(train_text)
        linears, attention = zip(*(_layer_forward(text, length, kind) for kind in types), strict=True)
        cost = linear_factor * (sum(linears) + 2 * length * hidden * vocab)
        cost += (3 if backward else 1) * sum(attention)
        if mtp:
            mtp_linear, mtp_attention = _layer_forward(text, length, "full_attention")
            detached_head = "lm-head" in getattr(args, "mtp_detach_paths", ("embedding", "backbone", "lm-head"))
            cost += mtp * (
                3 * (mtp_linear + mtp_attention + 4 * length * hidden * hidden)
                + (4 if detached_head else 6) * length * hidden * vocab
            )
        cost += _vision_cost(getattr(config, "vision_config", None), grids, args)
        costs.append(float(cost))
    return costs
