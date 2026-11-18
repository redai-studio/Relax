# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Useful model FLOPs for Relax's DeepSeek V4.1 text-training implementation.

Count GEMMs and attention/mHC contractions, with two FLOPs per multiply-add.
Use original document lengths, excluding padding, recompute, communication,
replicated TP/CP work, lookups, quantization, norms and elementwise operations.
These are model FLOPs, not a count of the hardware instructions executed.

The model freezes router weights (but propagates dX), the retrieval indexer
(no_grad), and Engram lookup tables. Engram projections need dW but no dX.
Consequently training is NOT uniformly three times the forward estimate.
Compressed KV and retrieval scores are only built at their source layers;
candidate masking follows indexer scoring and does not reduce that work.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from numbers import Integral, Real
from typing import Any


def _get(config: Any, name: str, default: Any = None) -> Any:
    return config.get(name, default) if isinstance(config, Mapping) else getattr(config, name, default)


def _key_pairs(length: int, ratio: int, cap: int | None = None) -> int:
    """Sum min(floor(position / ratio), cap) for positions 1..length."""
    end = min(length, ratio * cap - 1) if cap is not None else length
    groups, remainder = divmod(end, ratio)
    pairs = ratio * groups * (groups - 1) // 2 + groups * (remainder + 1)
    return pairs + (length - end) * cap if cap is not None else pairs


def _forward_flops(config: Any, lengths: Sequence[int]) -> tuple[float, float, float]:
    """Return (trainable fwd+dX+dW, fwd+one backward, no_grad) forward work."""
    text = _get(config, "text_config") or config
    fields = (
        "hidden_size",
        "num_hidden_layers",
        "num_attention_heads",
        "head_dim",
        "q_lora_rank",
        "o_lora_rank",
        "o_groups",
        "moe_intermediate_size",
        "n_routed_experts",
        "num_experts_per_tok",
        "vocab_size",
        "sliding_window",
        "index_n_heads",
        "index_head_dim",
        "index_topk",
        "hc_mult",
    )
    values = [_get(text, field) for field in fields]
    if any(not isinstance(value, Integral) or value <= 0 for value in values):
        raise ValueError("V4.1 requires positive text architecture dimensions")
    (
        hidden,
        layers,
        heads,
        dim,
        q_rank,
        o_rank,
        groups,
        intermediate,
        experts,
        topk,
        vocab,
        window,
        index_heads,
        index_dim,
        index_topk,
        streams,
    ) = values
    shared = _get(text, "n_shared_experts", 0)
    if shared < 0 or topk > experts or heads % groups or _get(text, "num_key_value_heads", 1) != 1:
        raise ValueError("Unsupported V4.1 expert/head layout")
    ratios = _get(text, "compress_ratios")[:layers]
    if len(ratios) != layers or any(ratio not in (0, 1, 2) for ratio in ratios):
        raise ValueError("V4.1 requires a complete ratio-0/1/2 backbone schedule")
    kv_sources = set(_get(text, "kv_source_layer_ids"))
    index_sources = set(_get(text, "index_source_layer_ids"))
    if any(layer < 0 or layer >= layers or not ratios[layer] for layer in kv_sources | index_sources):
        raise ValueError("V4.1 source layers must belong to the compressed backbone")
    if not kv_sources.issubset(index_sources):
        raise ValueError("V4.1 KV sources require index-key projections")
    for layer, ratio in enumerate(ratios):
        if ratio and not any(source <= layer and ratios[source] == ratio for source in kv_sources):
            raise ValueError("Missing V4.1 compressed KV source")
    candidate_layer = _get(text, "candidate_source_layer_id", -1)
    if any(layer > candidate_layer >= 0 for layer in index_sources):
        blocks, block_size = _get(text, "candidate_topk_blocks"), _get(text, "candidate_block_size")
        # The newest block may be partial. A smaller budget makes the number
        # of selected keys data-dependent, which config alone cannot count.
        if blocks <= 0 or block_size <= 0 or (blocks - 1) * block_size + 1 < index_topk:
            raise ValueError("V4.1 candidate budget may return fewer than index_topk keys")

    tokens = sum(lengths)
    attention_weights = hidden * q_rank + q_rank * heads * dim + hidden * dim
    attention_weights += heads * dim * o_rank + groups * o_rank * hidden
    expert_weights = 3 * hidden * intermediate * (topk + shared)
    # Two mHC sites per layer, each mapping nD -> n^2+2n.
    hc_weights = 2 * streams * hidden * (streams * streams + 2 * streams)
    weights = layers * (attention_weights + expert_weights + hc_weights) + hidden * vocab
    trainable = 2 * tokens * weights
    # Two residual-mixing matrix contractions per layer. Norms, Sinkhorn
    # and elementwise pre/post gates follow the exclusions above.
    trainable += 4 * tokens * layers * streams * streams * hidden
    trainable += 4 * heads * dim * layers * sum(_key_pairs(length, 1, window) for length in lengths)
    partial = 2 * tokens * layers * hidden * experts  # frozen router: fwd+dX
    frozen = 0.0

    for layer, ratio in enumerate(ratios):
        if not ratio:
            continue
        trainable += 4 * heads * dim * sum(_key_pairs(length, ratio, index_topk) for length in lengths)
        compressed_tokens = sum(length // ratio for length in lengths)
        if layer in kv_sources:
            # Ratio 2 has both value and gate projections; an incomplete
            # final group is discarded separately in each document.
            trainable += 2 * compressed_tokens * ratio * hidden * dim * ratio
        if layer in index_sources:
            frozen += 2 * tokens * (q_rank * index_heads * index_dim + hidden * index_heads)
            frozen += 2 * index_heads * (index_dim + 1) * sum(_key_pairs(length, ratio) for length in lengths)
            if layer in kv_sources:
                frozen += 2 * compressed_tokens * dim * index_dim

    engram_layers = [layer for layer in _get(text, "engram_layer_ids", ()) if 0 <= layer < layers]
    if engram_layers:
        width = (
            (_get(text, "engram_max_ngram_size") - 1) * _get(text, "engram_n_heads") * _get(text, "engram_head_dim")
        )
        if width <= 0:
            raise ValueError("Invalid V4.1 Engram projection width")
        partial += 2 * tokens * len(engram_layers) * width * hidden * (streams + 1)
    return trainable, partial, frozen


def estimate_deepseek_v41_flops(
    config: Any,
    tokens_sum: int,
    batch_seqlens: Sequence[int],
    delta_time: float,
    *,
    forward_only: bool = False,
    **kwargs: Any,
) -> float:
    """Global TFLOPS/s for the text recipe; unsupported policies return NaN."""
    if (
        kwargs.get("has_image_inputs", False)
        or kwargs.get("lora_rank", 0)
        or kwargs.get("freeze_language_model", False)
        or kwargs.get("freeze_params_name_list")
        or kwargs.get("only_train_params_name_list")
        or kwargs.get("enable_mtp_training", False)
        or kwargs.get("mtp_num_layers")
        or kwargs.get("mtp_only_training", False)
        or any(
            kwargs.get(name) is not None and len(kwargs[name]) > 0
            for name in ("images_seqlens", "audio_seqlens", "image_grid_thw")
        )
    ):
        return float("nan")
    if (
        not isinstance(delta_time, Real)
        or not math.isfinite(delta_time)
        or delta_time <= 0
        or not isinstance(tokens_sum, Integral)
        or tokens_sum < 0
        or any(not isinstance(length, Integral) or length < 0 for length in batch_seqlens)
        or tokens_sum != sum(batch_seqlens)
    ):
        return float("nan")
    try:
        trainable, partial, frozen = _forward_flops(config, batch_seqlens)
    except (AttributeError, KeyError, IndexError, TypeError, ValueError):
        return float("nan")
    flops = trainable + partial + frozen if forward_only else 3 * trainable + 2 * partial + frozen
    return flops / delta_time / 1e12
