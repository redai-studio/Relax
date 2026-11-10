# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Useful model FLOPs for Gemma 4 text-only 26B MoE and 31B dense training.

Count active GEMMs and causal attention pairs, excluding padding, recompute,
embedding lookups, normalization and elementwise operations. K=V removes the
global V projection, while QK and AV are still separate attention operations.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from numbers import Integral, Real
from typing import Any


def _get(config: Any, name: str, default: Any = None) -> Any:
    return config.get(name, default) if isinstance(config, Mapping) else getattr(config, name, default)


def gemma4_text_spec(config: Any) -> tuple[Any, list[tuple[int, int, bool]]]:
    """Return text config and per-layer (head dim, KV heads, tied K/V)."""
    text = _get(config, "text_config") or config
    if _get(text, "hidden_size_per_layer_input", 0) or _get(text, "per_layer_embed_dim", 0):
        raise ValueError("Gemma4 PLE is not supported by the text FLOPs formula")
    if _get(text, "num_kv_shared_layers", 0):
        raise ValueError("Gemma4 cross-layer KV sharing is not supported")
    if _get(text, "use_bidirectional_attention") == "all":
        raise ValueError("Gemma4 bidirectional text attention is not supported")
    layers = _get(text, "num_hidden_layers")
    types = _get(text, "layer_types")
    if not layers or not types or len(types) != layers:
        raise ValueError("Gemma4 requires a complete attention layer schedule")
    overrides = _get(text, "per_layer_config", {})
    if not isinstance(overrides, (Mapping, Sequence)):
        overrides = vars(overrides)
    if isinstance(overrides, Mapping):
        # HF serializes heterogeneous layer indices with zero padding ("05").
        overrides = {int(key): value for key, value in overrides.items()}
    shapes = []
    for index, kind in enumerate(types):
        if kind not in ("sliding_attention", "full_attention"):
            raise ValueError(f"Unsupported Gemma4 attention type: {kind}")
        sliding = kind == "sliding_attention"
        tied = bool(_get(text, "attention_k_eq_v", False)) and not sliding
        if isinstance(overrides, Mapping):
            layer = overrides.get(index, {})
        elif isinstance(overrides, Sequence):
            layer = overrides[index]
        else:
            raise ValueError("Unsupported Gemma4 per_layer_config representation")
        # Resolved heterogeneous configs deliberately forbid global reads of
        # ambiguous attributes. Read the layer first, with legacy JSON fallback.
        dim = _get(layer, "head_dim")
        if dim is None:
            dim = _get(text, "head_dim") if sliding else _get(text, "global_head_dim", 512)
        kv = _get(layer, "num_key_value_heads")
        if kv is None:
            kv = _get(text, "num_global_key_value_heads") if tied else None
            if kv is None:
                kv = _get(text, "num_key_value_heads")
        if not dim or not kv or dim <= 0 or kv <= 0:
            raise ValueError("Gemma4 requires positive head dimensions and KV head counts")
        shapes.append((dim, kv, tied))
    return text, shapes


def gemma4_active_weights(config: Any) -> int:
    """Count active GEMM weights; include the output head once."""
    text, shapes = gemma4_text_spec(config)
    hidden, heads = _get(text, "hidden_size"), _get(text, "num_attention_heads")
    layers, intermediate = len(shapes), _get(text, "intermediate_size")
    vocab = _get(text, "vocab_size")
    if not all(value and value > 0 for value in (hidden, heads, intermediate, vocab)):
        raise ValueError("Missing required Gemma4 text shapes")
    weights = sum(hidden * dim * (2 * heads + kv * (1 if tied else 2)) for dim, kv, tied in shapes)
    weights += layers * 3 * hidden * intermediate + hidden * vocab
    if _get(text, "enable_moe_block", False):
        experts, top_k = _get(text, "num_experts"), _get(text, "top_k_experts")
        width = _get(text, "moe_intermediate_size")
        if not all(value and value > 0 for value in (experts, top_k, width)) or top_k > experts:
            raise ValueError("Missing or invalid Gemma4 MoE shapes")
        weights += layers * hidden * (3 * width * top_k + experts)
    return weights


def _causal_pairs(length: int, window: int | None = None) -> int:
    width = min(length, window) if window is not None else length
    return width * length - width * (width - 1) // 2


def _has_inputs(value: Any) -> bool:
    if value is None:
        return False
    try:
        return len(value) > 0
    except TypeError:
        return True


def estimate_gemma4_flops(
    config: Any,
    tokens_sum: int,
    batch_seqlens: Sequence[int],
    delta_time: float,
    *,
    dense_multiplier: int = 6,
    **kwargs: Any,
) -> float:
    """Return global training TFLOPS/s; unsupported paths return NaN."""
    if (
        dense_multiplier != 6
        or kwargs.get("lora_rank", 0)
        or kwargs.get("freeze_language_model", False)
        or kwargs.get("freeze_params_name_list")
        or kwargs.get("only_train_params_name_list")
        or kwargs.get("enable_mtp_training", False)
        or kwargs.get("mtp_only_training", False)
        or any(
            _has_inputs(value)
            for value in (kwargs.get("images_seqlens"), kwargs.get("audio_seqlens"), kwargs.get("image_grid_thw"))
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
        text, shapes = gemma4_text_spec(config)
        weights = gemma4_active_weights(config)
        types = _get(text, "layer_types")
        window = _get(text, "sliding_window")
        if "sliding_attention" in types and (not window or window <= 0):
            return float("nan")
        full_pairs = sum(_causal_pairs(length) for length in batch_seqlens)
        local_pairs = sum(_causal_pairs(length, window) for length in batch_seqlens)
        attention = sum(
            dim * (local_pairs if kind == "sliding_attention" else full_pairs)
            for kind, (dim, _, _) in zip(types, shapes)
        )
        flops = 6 * weights * tokens_sum + 12 * _get(text, "num_attention_heads") * attention
        return flops / delta_time / 1e12
    except (AttributeError, KeyError, IndexError, TypeError, ValueError):
        return float("nan")
