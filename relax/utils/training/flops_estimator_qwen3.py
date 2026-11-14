# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""CPU-only absolute Qwen3 dense work estimates using Relax's training convention.

Mathematical model work, not executed kernel FLOPs, memory capacity or latency.
Covers the official Qwen3 dense family: ``model_type="qwen3"``, tied or untied
embeddings, all full-attention layers, non-gated Q projection, SwiGLU MLPs.
Coefficient conventions are shared with the frozen P2 formulas
(``flops_counter.py``); the Qwen3.5 estimator and this module are independent.
Tied embeddings do not change computation: the output projection GEMM still
executes, so embedding and lm_head are counted twice, matching the existing
untied convention (``emd_and_lm_head_N = vocab_size * hidden_size * 2``).
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from numbers import Integral
from types import SimpleNamespace
from typing import Any

from relax.utils.training.flops_estimator import SampleMeta, RelaxFlopsEstimator


_MISSING = object()
_SUPPORTED_MODEL_TYPES = {"qwen3"}
# Bound the formula to the audited Qwen3 dense structure: bias-free attention
# projections and MLPs with SiLU-gated (SwiGLU) MLPs, no sliding window.
_TEXT_ASSUMPTIONS = {"attention_bias": False, "hidden_act": "silu", "mlp_bias": False}
# Structural exclusions: anything present (beyond None/empty) means the config
# leaves the audited dense full-attention contract and is rejected instead of
# silently approximated.
_FORBIDDEN_FIELDS = (
    "full_attention_interval",
    "linear_num_value_heads",
    "linear_num_key_heads",
    "linear_value_head_dim",
    "linear_key_head_dim",
    "linear_conv_kernel_dim",
    "decoder_sparse_step",
    "mlp_only_layers",
    "num_experts",
    "num_experts_per_tok",
    "moe_intermediate_size",
    "shared_expert_intermediate_size",
    "vision_config",
)
_TEXT_FIELDS = (
    "hidden_size",
    "vocab_size",
    "num_hidden_layers",
    "num_attention_heads",
    "num_key_value_heads",
    "head_dim",
    "intermediate_size",
)


def _read(config: Any, name: str) -> Any:
    if isinstance(config, Mapping):
        return config.get(name, _MISSING)
    return getattr(config, name, _MISSING)


def _field(config: Any, name: str) -> Any:
    value = _read(config, name)
    if value is _MISSING:
        raise ValueError(f"Missing model configuration field: {name}")
    return value


def _positive_int(config: Any, name: str) -> int:
    value = _field(config, name)
    if not isinstance(value, Integral) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"Model configuration field {name} must be a positive integer, got {value!r}")
    return int(value)


def _snapshot_qwen3_config(config: Any) -> SimpleNamespace:
    model_type = _field(config, "model_type")
    if model_type not in _SUPPORTED_MODEL_TYPES:
        raise ValueError(f"Unsupported model_type for RelaxQwen3DenseEstimator: {model_type!r}")
    for name in _FORBIDDEN_FIELDS:
        value = _read(config, name)
        if value is not _MISSING and value not in (None, [], (), ""):
            raise ValueError(f"Qwen3 dense contract does not support {name}={value!r}")
    for name, expected in _TEXT_ASSUMPTIONS.items():
        value = _read(config, name)
        if value is not _MISSING and value != expected:
            raise ValueError(f"Unsupported Qwen3 dense {name}: {value!r}")
    use_sliding_window = _read(config, "use_sliding_window")
    if use_sliding_window not in (_MISSING, False):
        raise ValueError(f"Qwen3 dense contract does not support use_sliding_window={use_sliding_window!r}")
    sliding_window = _read(config, "sliding_window")
    if sliding_window not in (_MISSING, None):
        raise ValueError(f"Qwen3 dense contract does not support sliding_window={sliding_window!r}")
    # tie_word_embeddings is accepted as-is: computation is identical for tied
    # and untied weights (see module docstring); no untied masquerade needed.
    values = {name: _positive_int(config, name) for name in _TEXT_FIELDS}
    if values["num_attention_heads"] % values["num_key_value_heads"] != 0:
        raise ValueError("num_attention_heads must be divisible by num_key_value_heads")
    # layer_types: HF configs materialize an explicit list on load. A uniform
    # all-full-attention list is exactly the audited dense contract; anything
    # else (hybrid/linear layers, wrong length) is rejected, not approximated.
    layer_types = _read(config, "layer_types")
    if layer_types is not _MISSING and layer_types is not None:
        if (
            not isinstance(layer_types, (list, tuple))
            or len(layer_types) != values["num_hidden_layers"]
            or any(value != "full_attention" for value in layer_types)
        ):
            raise ValueError(
                f"Qwen3 dense contract requires uniform full_attention layers, got {layer_types!r}"
            )
    return SimpleNamespace(model_type=model_type, **values)


class RelaxQwen3DenseEstimator(RelaxFlopsEstimator):
    """Absolute FLOPs estimator for the official Qwen3 dense contract.

    Subclasses :class:`RelaxFlopsEstimator` for typing compatibility only; the
    frozen Qwen3.5 snapshot/validation is bypassed in favor of the Qwen3 dense
    contract above. The snapshot keeps the ``_config`` attribute name consumed
    by ``cost_identity``.
    """

    def __init__(self, model_config: Any, *, mtp_training_enabled: bool, compute_scope: str = "training") -> None:
        if type(mtp_training_enabled) is not bool:
            raise ValueError("mtp_training_enabled must be an explicit bool")
        if mtp_training_enabled:
            raise ValueError("MTP-enabled training FLOPs are not supported; only non-MTP main-model work is modeled")
        if compute_scope != "training":
            raise ValueError("Only the audited training compute scope is supported")
        self._config = _snapshot_qwen3_config(model_config)

    @property
    def compute_scope(self) -> str:
        return "training"

    def estimate_sample(self, sample_meta: SampleMeta) -> float:
        return self.estimate_microbatch((sample_meta,))

    def estimate_microbatch(self, sample_metas: Sequence[SampleMeta]) -> float:
        lengths: list[int] = []
        for meta in sample_metas:
            if not isinstance(meta, SampleMeta):
                raise TypeError(f"Expected SampleMeta, got {type(meta).__name__}")
            if meta.image_sequence_lengths:
                raise ValueError("Qwen3 dense text contract does not accept image_sequence_lengths")
            lengths.extend(meta.sequence_lengths)
        if not lengths:
            raise ValueError("sample_metas must contain at least one sequence")
        if any(not isinstance(value, Integral) or isinstance(value, bool) or value <= 0 for value in lengths):
            raise ValueError(f"Sequence lengths must be positive integers, got {lengths!r}")
        config = self._config
        hidden_size = config.hidden_size
        q_size = config.num_attention_heads * config.head_dim
        kv_size = config.num_key_value_heads * config.head_dim
        o_size = config.num_attention_heads * config.head_dim
        # Same integer expression family as the frozen dense convention
        # (flops_counter._estimate_qwen2_flops): all factors are exact ints, so
        # regrouping across layers is bit-exact.
        mlp_N = hidden_size * config.intermediate_size * 3 * config.num_hidden_layers
        attn_proj_N = hidden_size * (q_size + kv_size + kv_size + o_size) * config.num_hidden_layers
        emd_and_lm_head_N = config.vocab_size * hidden_size * 2
        dense_N = mlp_N + attn_proj_N + emd_and_lm_head_N
        dense_N_flops = 6 * dense_N * sum(lengths)
        seqlen_square_sum = sum(value * value for value in lengths)
        attn_qkv_flops = 6 * seqlen_square_sum * config.head_dim * config.num_attention_heads * config.num_hidden_layers
        total = dense_N_flops + attn_qkv_flops
        if not math.isfinite(total) or total < 0:
            raise ValueError(f"Non-finite or negative FLOPs estimate: {total!r}")
        return float(total)
