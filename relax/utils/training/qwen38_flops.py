# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Qwen3.8 useful FLOPs, preserving the pre-refactor measurement contract."""

import re
from typing import Any

from relax.utils.logging_utils import get_logger


logger = get_logger(__name__)


def _qwen38_selected_pairs(length: int, budget: int, ratio: int) -> int:
    """Causal token pairs after compressed-block top-k and the local tail."""
    if length <= budget:
        return length * (length + 1) // 2
    cycles, tail = divmod(length - budget, ratio)
    return (
        budget * (budget + 1) // 2
        + budget * (length - budget)
        + cycles * ratio * (ratio - 1) // 2
        + tail * (tail + 1) // 2
    )


def _qwen38_indexer_pairs(length: int, budget: int, ratio: int) -> int:
    def floor_sum(value: int) -> int:
        blocks, tail = divmod(value, ratio)
        return ratio * blocks * (blocks - 1) // 2 + blocks * (tail + 1)

    cutoff = budget + ratio - 1
    return 0 if length <= cutoff else floor_sum(length) - floor_sum(cutoff)


def _estimate_qwen38_flops(
    config: Any, tokens_sum: int, batch_seqlens: list[int], delta_time: float, **kwargs: Any
) -> float:
    """Useful FLOPs for Qwen3.8 with a frozen indexer.

    Includes language fwd/bwd, frozen auxiliary forwards and PLE conv input
    gradients. PLE projections/norms/conv may be all frozen or all trainable.
    Vision and its projection may be jointly frozen or jointly trainable. GDN
    recurrence uses the conventional 22-FLOP approximation. Excludes
    recomputation, padding, communication, duplicated CP work and small
    elementwise/top-k operations. Document boundaries remain independent. MTP
    supports the one-QSA-layer recipe with optional detachment: its MoE/QSA/HC
    block, stream-wise input projections and full-token LM head are included.
    Embedding lookups are not dense vocabulary matmuls; their backward scatter-
    adds are excluded alongside other elementwise operations. Unvalidated
    training policies return zero rather than a misleading MFU.
    """
    if kwargs.get("lora_rank", 0) or kwargs.get("only_train_params_name_list"):
        logger.warning("Qwen3.8 MFU does not support LoRA or selective trainable-parameter policies.")
        return 0.0
    patterns = kwargs.get("freeze_params_name_list") or []
    frozen_indexer = any(re.search(pattern, "decoder.layers.0.self_attention.indexer.weight") for pattern in patterns)
    # Match real PLE parameter suffixes rather than a nonexistent module weight.
    # Only layer-independent, uniform policies are audited here; partial freezes
    # must not silently inherit the FLOPs of a fully frozen/trainable PLE branch.
    ple_parameters = (
        "ple_embedding.ngram_embedding.weight",
        "key_proj.weight",
        "value_proj.weight",
        "norm_key.weight",
        "norm_query.weight",
        "norm_conv.weight",
        "conv1d.weight",
    )
    frozen_ple = [
        any(re.search(pattern, f"per_layer_embedding.{name}") for pattern in patterns) for name in ple_parameters
    ]
    train_ple_branch = not any(frozen_ple[1:])
    images = kwargs.get("images_seqlens") or []
    frozen_vision = bool(kwargs.get("freeze_vision_model", False))
    frozen_vision_projection = bool(kwargs.get("freeze_vision_projection", False))
    text = getattr(config, "text_config", config)
    # Layer indices can be PP-local. Reject prefix/layer-specific exceptions to
    # the suffix policy for either the text or VL model's parameter names.
    uniform_ple_policy = all(
        any(re.search(pattern, f"{prefix}decoder.layers.{layer}.per_layer_embedding.{name}") for pattern in patterns)
        == frozen
        for prefix in ("", "language_model.")
        for layer in range(text.num_hidden_layers)
        for name, frozen in zip(ple_parameters, frozen_ple)
    )
    mtp_layers = 0
    mtp_head_factor = 6
    if kwargs.get("enable_mtp_training", False):
        mtp = getattr(text, "mtp", None)
        mtp_types = mtp.get("layer_types") if isinstance(mtp, dict) else getattr(mtp, "layer_types", None)
        detach_paths = kwargs.get("mtp_detach_paths", ("embedding", "backbone", "lm-head"))
        if detach_paths in ("none", ("none",), ["none"]):
            detach_paths = ()
        valid_detach_paths = isinstance(detach_paths, (tuple, list)) and all(
            path in ("embedding", "backbone", "lm-head") for path in detach_paths
        )
        frozen_mtp_indexer = any(
            re.search(pattern, "mtp.layers.0.mtp_model_layer.layers.0.self_attention.indexer.weight")
            for pattern in patterns
        )
        if (
            kwargs.get("mtp_num_layers") != 1
            or not valid_detach_paths
            or mtp_types not in (["full_attention"], ["qwen_sparse_attention"])
            or not frozen_mtp_indexer
            or kwargs.get("mtp_only_training", False)
        ):
            logger.warning(
                "Qwen3.8 MFU requires one QSA MTP layer, a frozen indexer, joint backbone training, "
                "and mtp_detach_paths drawn from embedding/backbone/lm-head (or none)."
            )
            return 0.0
        mtp_layers = 1
        # Detached output weights still propagate gradients into MTP hidden states.
        mtp_head_factor = 4 if "lm-head" in detach_paths else 6
    if (
        not frozen_indexer
        or not (all(frozen_ple) or train_ple_branch)
        or not uniform_ple_policy
        or (images and frozen_vision != frozen_vision_projection)
    ):
        logger.warning(
            "Qwen3.8 MFU requires jointly frozen/trainable vision and projection, a frozen indexer, "
            "and fully frozen PLE, "
            "fully trainable PLE, or only its lookup table frozen; other partial freezes are not supported."
        )
        return 0.0
    hidden, layers = text.hidden_size, text.num_hidden_layers + mtp_layers
    linear_layers = text.layer_types.count("linear_attention")
    # Qwen4ExpTextConfig rewrites the checkpoint's full_attention labels.
    qsa_layers = sum(layer in {"full_attention", "qwen_sparse_attention"} for layer in text.layer_types) + mtp_layers
    budget, ratio = text.indexer_budget, text.indexer_compress_ratio
    if budget <= 0 or ratio <= 0 or budget % ratio:
        raise ValueError(
            "Qwen3.8 FLOPs estimator requires positive indexer_budget divisible by positive compress_ratio"
        )
    key_width = text.linear_num_key_heads * text.linear_key_head_dim
    value_width = text.linear_num_value_heads * text.linear_value_head_dim
    channels, hc_rank = text.hc_count, text.hc_lowrank
    # Routed/shared expert MLPs and both routing gates.
    linear = (
        6
        * layers
        * 3
        * hidden
        * (text.num_experts_per_tok * text.moe_intermediate_size + text.shared_expert_intermediate_size)
    )
    linear += 6 * layers * hidden * (text.num_experts + 1)
    # GDN Q/K/V, output gate/projection, beta/decay, convolution and recurrence.
    linear += 6 * linear_layers * hidden * (2 * key_width + 3 * value_width + 2 * text.linear_num_value_heads)
    linear += 6 * linear_layers * (2 * key_width + value_width) * text.linear_conv_kernel_dim
    linear += 22 * linear_layers * text.linear_num_value_heads * text.linear_key_head_dim * text.linear_value_head_dim
    # QSA attention projections/gate and frozen indexer projection.
    linear += 6 * qsa_layers * hidden * (3 * text.num_attention_heads + 2 * text.num_key_value_heads) * text.head_dim
    linear += 2 * qsa_layers * hidden * (text.indexer_n_heads + text.indexer_kv_heads) * text.indexer_head_dim
    # Hyper-connections include the final output reduction and both sublayers.
    linear += 6 * (2 * layers + 1 + mtp_layers) * 2 * channels * hidden * hc_rank
    linear += 6 * (2 * layers) * channels**2 * hidden
    # MTP e_proj sees one stream; h_proj sees all hc_count residual streams.
    # Detachment precedes trainable enorm/hnorm, so both projections still need
    # dInput as well as dWeight. Joint training already backprops the main model.
    linear += 6 * mtp_layers * (channels + 1) * hidden**2
    ple_layers = len(text.ple_layer_ids)
    # Frozen lookup outputs need no dInput: trainable key/value projections cost
    # forward + dWeight (4), not the usual forward + dInput + dWeight (6).
    projection_factor = 4 if train_ple_branch else 2
    if not frozen_ple[0]:
        # A trainable lookup table also needs both projections' dInput matmuls.
        projection_factor = 6
    # Conv inputs depend on trainable hidden states even when all PLE is frozen.
    conv_factor = 6 if train_ple_branch else 4
    linear += projection_factor * ple_layers * text.ple_embed_dim * hidden * (channels + 1)
    linear += conv_factor * ple_layers * channels * hidden * text.ple_conv_kernel_size
    head_tokens = kwargs.get("lm_head_tokens", tokens_sum)
    # MTP shifts with zero fill without shortening packed sequences, and its
    # output projection runs before loss masking, even with chunked main logits.
    total = linear * tokens_sum + hidden * text.vocab_size * (
        6 * head_tokens + mtp_head_factor * mtp_layers * tokens_sum
    )
    total += (
        12
        * qsa_layers
        * text.num_attention_heads
        * text.head_dim
        * sum(_qwen38_selected_pairs(length, budget, ratio) for length in batch_seqlens)
    )
    total += (
        2
        * qsa_layers
        * text.indexer_n_heads
        * text.indexer_head_dim
        * sum(_qwen38_indexer_pairs(length, budget, ratio) for length in batch_seqlens)
    )
    if images:
        vision = config.vision_config
        width, depth, intermediate = vision.hidden_size, vision.depth, vision.intermediate_size
        merge = vision.spatial_merge_size**2
        # Pixels have no gradients: patch embedding needs forward + dWeight,
        # while subsequent trainable blocks/mergers also need dInput.
        projection_factor = 2 if frozen_vision else 4
        linear_factor = 2 if frozen_vision else 6
        attention_factor = 4 if frozen_vision else 12
        projection = projection_factor * width * vision.in_channels * vision.temporal_patch_size * vision.patch_size**2
        blocks = linear_factor * depth * (4 * width**2 + 2 * width * intermediate)
        merger = linear_factor * ((merge * width) ** 2 + merge * width * vision.out_hidden_size)
        merger *= 1 + len(getattr(vision, "deepstack_visual_indexes", []))
        total += (projection + blocks) * sum(images)
        total += attention_factor * depth * width * sum(length**2 for length in images)
        total += merger * sum(images) / merge
    return total / delta_time / 1e12
