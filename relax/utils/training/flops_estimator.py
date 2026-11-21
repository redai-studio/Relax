# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""CPU-only absolute Qwen3.5 work estimates using Relax's training convention.

This is mathematical model work, not executed kernel FLOPs, memory capacity or
latency. Padding, recomputation, expert dispatch/imbalance and communication
are not modeled. No production scheduler consumes this API yet.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Integral
from types import SimpleNamespace
from typing import Any

from relax.utils.training.flops_counter import _estimate_qwen3_5_absolute_flops


_MISSING = object()
_MODEL_TYPES = {"qwen3_5", "qwen3_5_text", "qwen3_5_moe", "qwen3_5_moe_text"}
# Transformers 5.3 Qwen3.5 Dense/MoE: explicit text-config defaults for
# attention_bias/hidden_act; modeling always uses gated 2q and bias-free MLPs.
# These checks bound the existing formula, rather than adding new formula terms.
_TEXT_ASSUMPTIONS = {"attn_output_gate": True, "attention_bias": False, "hidden_act": "silu", "mlp_bias": False}
_TEXT_FIELDS = (
    "hidden_size",
    "vocab_size",
    "num_hidden_layers",
    "num_attention_heads",
    "num_key_value_heads",
    "head_dim",
    "linear_num_key_heads",
    "linear_key_head_dim",
    "linear_num_value_heads",
    "linear_value_head_dim",
    "linear_conv_kernel_dim",
)
_VISION_FIELDS = (
    "num_heads",
    "depth",
    "hidden_size",
    "intermediate_size",
    "out_hidden_size",
    "spatial_merge_size",
    "in_channels",
    "temporal_patch_size",
    "patch_size",
)


def _field(config: Any, name: str, default: Any = _MISSING) -> Any:
    value = config.get(name, default) if isinstance(config, Mapping) else getattr(config, name, default)
    if value is _MISSING:
        raise ValueError(f"Missing model configuration field: {name}")
    return value


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _lengths(values: Sequence[int], name: str, *, allow_empty: bool) -> tuple[int, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise TypeError(f"{name} must be a sequence of positive integers")
    result = tuple(_positive_int(value, name) for value in values)
    if not result and not allow_empty:
        raise ValueError(f"{name} must contain at least one independent attention segment")
    return result


@dataclass(frozen=True)
class SampleMeta:
    """Lengths after tokenization, before layout padding.

    Text lengths are independent attention segments, not concatenated token
    totals. Image lengths use the existing counter's pre-merge patch sequences
    (the actor records h*w per frame). Text lengths already include visual
    tokens processed by the text model; image lengths add vision-tower work
    only.
    """

    sequence_lengths: tuple[int, ...]
    image_sequence_lengths: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "sequence_lengths", _lengths(self.sequence_lengths, "sequence_lengths", allow_empty=False)
        )
        object.__setattr__(
            self,
            "image_sequence_lengths",
            _lengths(self.image_sequence_lengths, "image_sequence_lengths", allow_empty=True),
        )


def _snapshot_config(config: Any) -> SimpleNamespace:
    model_type = _field(config, "model_type")
    if model_type not in _MODEL_TYPES:
        raise ValueError(f"Unsupported model_type for RelaxFlopsEstimator: {model_type!r}")
    text = _field(config, "text_config", config)
    text_type = _field(text, "model_type", model_type)
    if text_type not in _MODEL_TYPES or ("moe" in text_type) != ("moe" in model_type):
        raise ValueError("Top-level and text model types must describe the same Qwen3.5 architecture")
    for name, supported in _TEXT_ASSUMPTIONS.items():
        # Text fields live in text_config (or the root for a text-only config).
        # Explicit wrapper duplicates must agree, not hide an unsupported flag.
        for level, source in (("text", text), ("root", config)):
            value = _field(source, name, supported)
            if type(value) is not type(supported) or value != supported:
                raise ValueError(f"Unsupported {level} {name}: expected {supported!r}, got {value!r}")
    values = {name: _positive_int(_field(text, name), name) for name in _TEXT_FIELDS}
    if values["num_attention_heads"] % values["num_key_value_heads"]:
        raise ValueError("num_attention_heads must be divisible by num_key_value_heads")
    if values["linear_num_value_heads"] % values["linear_num_key_heads"]:
        raise ValueError("linear_num_value_heads must be divisible by linear_num_key_heads")
    layers = _field(text, "layer_types", None)
    if layers is None:
        interval = _positive_int(_field(text, "full_attention_interval"), "full_attention_interval")
        layers = tuple(
            "full_attention" if (index + 1) % interval == 0 else "linear_attention"
            for index in range(values["num_hidden_layers"])
        )
    if (
        not isinstance(layers, (list, tuple))
        or len(layers) != values["num_hidden_layers"]
        or any(layer not in ("full_attention", "linear_attention") for layer in layers)
    ):
        raise ValueError("layer_types must describe every layer as full_attention or linear_attention")
    values["layer_types"] = tuple(layers)
    if _field(text, "mlp_only_layers", ()) or _field(text, "decoder_sparse_step", 1) != 1:
        raise ValueError("Mixed dense/MoE MLP placement is not covered by the existing formula")
    if _field(config, "tie_word_embeddings", False) or _field(text, "tie_word_embeddings", False):
        raise ValueError("Tied embeddings are not supported by this main-model work contract")
    if "moe" in model_type:
        for name in ("num_experts", "num_experts_per_tok", "moe_intermediate_size", "shared_expert_intermediate_size"):
            values[name] = _positive_int(_field(text, name), name)
        if values["num_experts_per_tok"] > values["num_experts"]:
            raise ValueError("num_experts_per_tok cannot exceed num_experts")
    else:
        if _field(text, "num_experts", None) is not None:
            raise ValueError("Dense model_type cannot include num_experts")
        values["intermediate_size"] = _positive_int(_field(text, "intermediate_size"), "intermediate_size")
    snapshot = SimpleNamespace(model_type=model_type, text_config=SimpleNamespace(**values))
    vision = _field(config, "vision_config", None)
    if vision is not None:
        if _field(vision, "hidden_act", "gelu_pytorch_tanh") != "gelu_pytorch_tanh":
            raise ValueError("Unsupported vision hidden_act: expected 'gelu_pytorch_tanh'")
        vision_values = {name: _positive_int(_field(vision, name), name) for name in _VISION_FIELDS}
        if vision_values["hidden_size"] % vision_values["num_heads"]:
            raise ValueError("Vision hidden_size must be divisible by num_heads")
        indexes = _field(vision, "deepstack_visual_indexes", ())
        if indexes is None:
            indexes = ()
        if not isinstance(indexes, (list, tuple)) or any(
            isinstance(index, bool) or not isinstance(index, Integral) or not 0 <= index < vision_values["depth"]
            for index in indexes
        ):
            raise ValueError("Invalid deepstack_visual_indexes")
        if len(set(indexes)) != len(indexes):
            raise ValueError("Duplicate deepstack_visual_indexes")
        vision_values["deepstack_visual_indexes"] = tuple(indexes)
        snapshot.vision_config = SimpleNamespace(**vision_values)
    return snapshot


class RelaxFlopsEstimator:
    """Estimate non-MTP main-model FLOPs under Relax's training convention.

    Supports Qwen3.5 Dense/MoE text and optional vision metadata. Only training
    scope is supported: the GDN coefficient does not establish a separately
    validated forward formula. The caller must explicitly declare the actual
    MTP training mode; model mtp/nextn metadata does not establish that mode.
    MTP-enabled training, frozen regions, LoRA and actual hardware layout are
    outside this contract. An existing vision_config is eagerly validated, even
    for text-only estimates. Caller mutations to the original config cannot
    change an initialized estimator.
    """

    def __init__(self, model_config: Any, *, mtp_training_enabled: bool, compute_scope: str = "training") -> None:
        if type(mtp_training_enabled) is not bool:
            raise ValueError("mtp_training_enabled must be an explicit bool describing the actual training mode")
        if mtp_training_enabled:
            raise ValueError("MTP-enabled training FLOPs are not supported; only non-MTP main-model work is modeled")
        if compute_scope != "training":
            raise ValueError("Only the audited existing training convention is supported")
        self._config = _snapshot_config(model_config)

    @property
    def compute_scope(self) -> str:
        return "training"

    def estimate_sample(self, sample_meta: SampleMeta) -> float:
        """Return absolute model-work FLOPs, without querying devices or
        time."""
        return self.estimate_microbatch((sample_meta,))

    def estimate_microbatch(self, sample_metas: Sequence[SampleMeta]) -> float:
        """Preserve independent attention boundaries; an empty microbatch costs
        zero."""
        lengths, images = [], []
        for meta in sample_metas:
            if not isinstance(meta, SampleMeta):
                raise TypeError("Expected SampleMeta values")
            lengths.extend(meta.sequence_lengths)
            images.extend(meta.image_sequence_lengths)
        if images and not hasattr(self._config, "vision_config"):
            raise ValueError("Image metadata requires a complete vision_config")
        try:
            value = float(_estimate_qwen3_5_absolute_flops(self._config, sum(lengths), lengths, images_seqlens=images))
        except OverflowError as error:
            raise ValueError("Estimated FLOPs exceed finite floating-point range") from error
        if not math.isfinite(value) or value < 0:
            raise ValueError("Estimated FLOPs must be finite and non-negative")
        return value
