# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Useful K3 model FLOPs, independent of TP/PP/CP/EP and kernel tiling.

Count matmul multiply-adds as two operations, plus depthwise convolutions
and the three KDA state contractions. Trainable text uses the conventional
forward + 2 * forward backward estimate. Exclude recomputation, padding,
communication, embedding lookups, AttnRes reductions, normalization and
elementwise activations/gates. This is model MFU, not hardware FLOPs.
"""

from collections.abc import Sequence
from typing import Any


def _text_forward_flops(config: Any, lengths: Sequence[int]) -> tuple[float, float]:
    """Return (parameterized linear/conv FLOPs, attention/state FLOPs)."""
    tokens = sum(lengths)
    hidden = config.hidden_size
    layers = config.num_hidden_layers
    linear = config.linear_attn_config
    kda_layers = sum(1 <= layer <= layers for layer in set(linear["kda_layers"]))
    mla_layers = layers - kda_layers

    heads, dim = linear["num_heads"], linear["head_dim"]
    width = heads * dim
    # Q/K/V, output gate, output projection; forget gate D->d->Hd; beta D->H.
    kda_weights = 5 * hidden * width + hidden * dim + dim * width + hidden * heads
    kda_weights += 3 * width * linear["short_conv_kernel_size"]

    q_dim = config.qk_nope_head_dim + config.qk_rope_head_dim
    v_dim = config.v_head_dim
    mla_heads = config.num_attention_heads
    mla_weights = hidden * config.q_lora_rank + config.q_lora_rank * mla_heads * q_dim
    mla_weights += hidden * (config.kv_lora_rank + config.qk_rope_head_dim)
    mla_weights += config.kv_lora_rank * mla_heads * (config.qk_nope_head_dim + v_dim)
    mla_weights += 2 * hidden * mla_heads * v_dim  # output gate and output projection

    moe_layers = sum(layer % config.moe_layer_freq == 0 for layer in range(config.first_k_dense_replace, layers))
    latent = config.routed_expert_hidden_size
    moe_weights = hidden * config.num_experts + 2 * hidden * latent
    moe_weights += 3 * latent * config.moe_intermediate_size * config.num_experts_per_token
    moe_weights += 3 * hidden * config.moe_intermediate_size * config.num_shared_experts
    dense_weights = 3 * hidden * config.intermediate_size
    weights = kda_layers * kda_weights + mla_layers * mla_weights
    weights += moe_layers * moe_weights + (layers - moe_layers) * dense_weights
    # Input embeddings are a lookup; only the vocabulary output is a matmul.
    weights += hidden * config.vocab_size

    # Causal MLA: QK^T and AV, each with half of the square attention matrix.
    attention = sum(length * length for length in lengths) * mla_heads * (q_dim + v_dim) * mla_layers
    # KDA: k^T S prediction, rank-one state update, q^T S readout (2*d*d each).
    attention += 6 * tokens * heads * dim * dim * kda_layers
    return 2 * tokens * weights, attention


def _vision_flops(
    config: Any,
    grids: Sequence[Sequence[int]],
    *,
    forward_only: bool,
    freeze_vision_model: bool,
    freeze_vision_projection: bool,
) -> float:
    """MoonViT-V2 then PatchMergerMLPV2, including temporal pooling."""
    if not grids:
        return 0.0
    hidden = config.vt_hidden_size
    qkv_width = config.qkv_hidden_size
    merge_h, merge_w = config.merge_kernel_size
    patch = config.patch_size
    patch_area = patch * patch if isinstance(patch, int) else patch[0] * patch[1]
    tokens = sum(t * h * w for t, h, w in grids)
    squares = sum((t * h * w) ** 2 for t, h, w in grids)
    # The encoder attends across ALL frames before temporal pooling. The
    # projector sees one spatial grid per image/video, not one per frame.
    merged_tokens = sum((h // merge_h) * (w // merge_w) for _, h, w in grids)
    patch_flops = 2 * tokens * 3 * patch_area * hidden
    blocks = config.vt_num_hidden_layers * (
        8 * tokens * hidden * qkv_width + 4 * tokens * hidden * config.vt_intermediate_size + 4 * squares * qkv_width
    )
    merged_width = merge_h * merge_w * hidden
    project1 = 2 * merged_tokens * merged_width * merged_width
    project2 = 2 * merged_tokens * merged_width * config.text_hidden_size

    encoder_grad = not forward_only and not freeze_vision_model
    projector_grad = not forward_only and not freeze_vision_projection
    # Pixels have no input gradient. A frozen projector still propagates dX
    # when its encoder trains; its parameters simply don't need dW.
    return (
        patch_flops * (1 + int(encoder_grad))
        + blocks * (3 if encoder_grad else 1)
        + project1 * (1 + int(encoder_grad) + int(projector_grad))
        + project2 * (1 + int(encoder_grad or projector_grad) + int(projector_grad))
    )


def estimate_kimi_k3_flops(
    config: Any,
    tokens_sum: int,
    batch_seqlens: Sequence[int],
    delta_time: float,
    *,
    forward_only: bool = False,
    freeze_vision_model: bool = False,
    freeze_vision_projection: bool = False,
    freeze_language_model: bool = False,
    image_grid_thw: Sequence[Sequence[int]] | None = None,
    images_seqlens: Sequence[int] | None = None,
    **kwargs: Any,
) -> float:
    """Estimate full-parameter K3 TFLOPS; frozen region flags are independent.

    ``image_grid_thw`` is the original CPU metadata, before spatial/temporal
    merging. Legacy ``images_seqlens`` callers are treated as still images.
    LoRA/custom parameter masks require a separate trainability-aware model and
    are filtered by FlopsCounter, rather than reported as full training.
    """
    del tokens_sum, kwargs
    text = getattr(config, "text_config", config)
    grids = image_grid_thw
    vision = getattr(config, "vision_config", None)
    if grids is None:
        # Only used by callers with per-image patch counts but no geometry.
        # Valid MoonViT still-image grids have dimensions divisible by merger.
        merge_h = vision.merge_kernel_size[0] if vision is not None else 1
        grids = [(1, merge_h, length // merge_h) for length in (images_seqlens or [])]

    linears, attention = _text_forward_flops(text, batch_seqlens)
    input_grad = bool(grids) and vision is not None and not (freeze_vision_model and freeze_vision_projection)
    backward = not forward_only and (not freeze_language_model or input_grad)
    linear_factor = 1 + int(backward) + int(not forward_only and not freeze_language_model)
    flops = linear_factor * linears + (3 if backward else 1) * attention
    if vision is not None:
        flops += _vision_flops(
            vision,
            grids,
            forward_only=forward_only,
            freeze_vision_model=freeze_vision_model,
            freeze_vision_projection=freeze_vision_projection,
        )
    return flops / delta_time / 1e12
