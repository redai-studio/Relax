# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Useful GLM-5.3-Flash model FLOPs for throughput reporting.

The decoder uses Slime's model-equivalent 3x-forward training convention,
including LoRA runs. Frozen DSA selection and frozen vision count forward work.
Exclude padding, replication, checkpoint replay, communication, sorting and
elementwise activations and normalization. Count attention using logical
dimensions.
"""

from collections.abc import Sequence
from typing import Any


def _causal_pool_pairs(length: int, pool_size: int) -> int:
    """Sum the number of complete causal pools across a document's queries."""
    blocks, tail = divmod(length, pool_size)
    return pool_size * blocks * (blocks - 1) // 2 + (tail + 1) * blocks


def _dsa_pairs(lengths: Sequence[int], topk: int, pool_size: int, keep_tail: bool) -> tuple[int, int]:
    attention_pairs = indexer_pairs = 0
    pool_budget = topk // pool_size
    for length in lengths:
        prefix = min(length, pool_budget * pool_size)
        selected = _causal_pool_pairs(prefix, pool_size) + (length - prefix) * pool_budget
        attention_pairs += selected * pool_size
        indexer_pairs += _causal_pool_pairs(length, pool_size)
        if keep_tail:
            blocks, tail = divmod(length, pool_size)
            attention_pairs += blocks * pool_size * (pool_size - 1) // 2 + tail * (tail + 1) // 2
    return attention_pairs, indexer_pairs


def _vision_forward_flops(config: Any, frame_lengths: Sequence[int]) -> tuple[int, int, int]:
    patches = sum(frame_lengths)
    width, out = config.hidden_size, config.out_hidden_size
    patch_weights = width * config.in_channels * config.temporal_patch_size * config.patch_size**2
    block_weights = 4 * width**2 + 3 * width * config.intermediate_size
    attention = 4 * width * sum(length**2 for length in frame_lengths)
    merged = patches // config.spatial_merge_size**2
    merger_weights = out**2 + 3 * out * config.projection_intermediate_size
    return (
        2 * patches * patch_weights + config.depth * (2 * patches * block_weights + attention),
        2 * patches * width * out,  # Strided spatial downsampling convolution.
        2 * merged * merger_weights,
    )


def estimate_glm53_flops(
    config: Any,
    tokens_sum: int,
    batch_seqlens: Sequence[int],
    delta_time: float,
    *,
    forward_only: bool = False,
    freeze_vision_model: bool = False,
    freeze_vision_projection: bool = False,
    lora_rank: int = 0,
    has_image_inputs: bool = False,
    image_grid_thw: Sequence[Sequence[int]] | None = None,
    images_seqlens: Sequence[int] | None = None,
    enable_mtp_training: bool = False,
    **_: Any,
) -> float:
    """Return whole-model TFLOPS/s for the main decoder and vision tower."""
    if enable_mtp_training and not forward_only:
        return float("nan")

    text = getattr(config, "text_config", config)
    layers = text.num_hidden_layers
    attention_types = getattr(text, "layer_types", ()) or ()
    kda_layers = attention_types.count("linear_attention")
    dsa_layers = attention_types.count("indexed_attention") + attention_types.count("deepseek_sparse_attention")
    if len(attention_types) != layers or kda_layers + dsa_layers != layers:
        return float("nan")
    mlp_types = getattr(text, "mlp_layer_types", None)
    if mlp_types is None:
        dense_layers = getattr(text, "first_k_dense_replace", min(3, layers))
        moe_layers = layers - dense_layers
    else:
        dense_layers, moe_layers = mlp_types.count("dense"), mlp_types.count("sparse")
        if len(mlp_types) != layers or dense_layers + moe_layers != layers:
            return float("nan")

    hidden = text.hidden_size
    weights = dense_layers * (3 * hidden * text.intermediate_size)
    if moe_layers:
        moe_width = text.moe_intermediate_size * (text.num_experts_per_tok + text.n_shared_experts)
        weights += moe_layers * hidden * (3 * moe_width + text.n_routed_experts)
    weights += hidden * text.vocab_size  # Embedding lookups do not contribute matmul FLOPs.
    if streams := getattr(text, "hc_mult", 0):
        weights += 2 * layers * hidden * streams * (2 * streams + streams**2)

    attention = indexer = 0
    if kda_layers:
        linear = getattr(text, "linear_attn_config", None) or {}
        heads = getattr(text, "linear_num_heads", None) or linear["num_heads"]
        dim = getattr(text, "linear_head_dim", None) or linear["head_dim"]
        conv_kernel = getattr(text, "linear_conv_kernel_dim", None) or linear["short_conv_kernel_size"]
        width = heads * dim
        # Q/K/V and output projections, two two-stage gates, beta and depthwise Q/K/V convolution.
        kda_weights = 4 * hidden * width + 2 * dim * (hidden + width) + hidden * heads
        weights += kda_layers * (kda_weights + 3 * width * conv_kernel)
        attention += kda_layers * 6 * tokens_sum * heads * dim**2  # Three KDA state contractions.
    if dsa_layers:
        heads, q_dim, v_dim = text.num_attention_heads, text.qk_head_dim, text.v_head_dim
        q_rank, kv_rank = text.q_lora_rank, text.kv_lora_rank
        rope_dim = q_dim - text.qk_nope_head_dim
        mla_weights = q_rank * (hidden + heads * q_dim) + hidden * (kv_rank + rope_dim)
        mla_weights += kv_rank * heads * (text.qk_nope_head_dim + v_dim) + hidden * heads * v_dim
        weights += dsa_layers * mla_weights

        pool_size = getattr(text, "index_kpool", 1)
        if not 1 <= pool_size <= text.index_topk:
            return float("nan")
        selected, scored = _dsa_pairs(
            batch_seqlens, text.index_topk, pool_size, getattr(text, "index_kpool_always_select_tail", True)
        )
        attention += dsa_layers * 2 * selected * heads * (q_dim + v_dim)
        index_heads, index_dim = text.index_n_heads, text.index_head_dim
        index_weights = q_rank * index_heads * index_dim + hidden * (index_dim + index_heads)
        if pool_size > 1:
            index_weights += (hidden + 1) * index_dim  # KPool gate projection and weighted key reduction.
        indexer = dsa_layers * 2 * (tokens_sum * index_weights + scored * index_heads * (index_dim + 1))

    vision = getattr(config, "vision_config", None)
    frame_lengths = [h * w for t, h, w in image_grid_thw for _ in range(t)] if image_grid_thw else images_seqlens
    if (frame_lengths and vision is None) or (has_image_inputs and not frame_lengths):
        return float("nan")
    vision_flops = 0
    if frame_lengths:
        patch_blocks, downsample, merger = _vision_forward_flops(vision, frame_lengths)
        if forward_only or (lora_rank > 0 and freeze_vision_model):
            vision_flops = patch_blocks + downsample + merger
        else:
            # Bridge freezes patch embedding and blocks separately from the merger.
            # A frozen merger still propagates gradients to the downsampling convolution.
            vision_flops = (1 if freeze_vision_model else 3) * patch_blocks
            vision_flops += 3 * downsample + (2 if freeze_vision_projection else 3) * merger
    decoder = (1 if forward_only else 3) * (2 * tokens_sum * weights + attention)
    return (decoder + indexer + vision_flops) / delta_time / 1e12
