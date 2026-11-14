# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""CPU planning for FLOPs weights; token capacity remains an independent
bound."""

from __future__ import annotations

import hashlib
import json
import math
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from numbers import Integral, Real
from pathlib import Path
from typing import Any

from relax.utils.data.seqlen_balancing import get_seqlen_balanced_partitions
from relax.utils.training.flops_estimator import RelaxFlopsEstimator, SampleMeta


MODES = ("off", "microbatch", "dp", "both")


def scheduling_mode(args: Any) -> str:
    mode = getattr(args, "flops_scheduling", "off")
    if mode not in MODES:
        raise ValueError(f"Unknown FLOPs scheduling mode: {mode!r}")
    return mode


def build_estimator(args: Any, model_config: Any) -> RelaxFlopsEstimator | None:
    """An off caller does not inspect runtime modes or model configuration."""
    if scheduling_mode(args) == "off":
        return None
    unsupported = [
        name
        for name in (
            "enable_lora",
            "lora_rank",
            "lora_merge_mode",
            "lora_adapter_mode",
            "freeze_params_name_list",
            "only_train_params_name_list",
            "freeze_language_model",
            "freeze_vision_model",
            "freeze_vision_projection",
            "freeze_audio_model",
            "freeze_audio_projection",
            "sft_async_prepack",
            "fully_async",
            "multimodal_keys",
            "sft_wholepack",
            "use_critic",
            "partial_rollout",
            "use_dynamic_global_batch_size",
        )
        if getattr(args, name, False)
    ]
    if unsupported:
        raise ValueError(f"FLOPs scheduling does not support these runtime options: {', '.join(unsupported)}")
    if getattr(args, "train_backend", "megatron") != "megatron" or not getattr(args, "use_dynamic_batch_size", False):
        raise ValueError("FLOPs scheduling currently requires Megatron dynamic batching")
    if getattr(args, "loss_type", None) in ("dpo", "rm", "ipo", "i2i_sft", "i2v_sft") or getattr(
        args, "generation_task", None
    ) in ("i2i", "i2v"):
        raise ValueError("FLOPs scheduling currently supports ordinary text SFT/RL objectives only")
    if getattr(args, "task_type", "causal_lm") != "causal_lm":
        raise ValueError("FLOPs scheduling requires the causal_lm head; classification heads are not supported")
    if not hasattr(args, "enable_mtp_training"):
        raise ValueError("FLOPs scheduling requires the normalized args.enable_mtp_training runtime switch")
    model_type = (
        model_config.get("model_type") if hasattr(model_config, "get") else getattr(model_config, "model_type", None)
    )
    if model_type == "qwen3":
        from relax.utils.training.flops_estimator_qwen3 import RelaxQwen3DenseEstimator

        return RelaxQwen3DenseEstimator(model_config, mtp_training_enabled=args.enable_mtp_training)
    return RelaxFlopsEstimator(model_config, mtp_training_enabled=args.enable_mtp_training)


def row_weights(estimator: RelaxFlopsEstimator, lengths: Sequence[int]) -> list[float]:
    """Ordinary text rows are separate causal sequences; never merge
    identities."""
    if len(lengths) == 0:
        raise ValueError("FLOPs scheduling requires a nonempty rank batch")
    return _weights([estimator.estimate_sample(SampleMeta((length,))) for length in lengths], len(lengths))


def metadata_step_count(args: Any, payload: dict, dp_size: int) -> tuple[int, int]:
    """Check mini boundaries before any rank can enter a differently sized K
    collective."""
    size = _positive_int(payload.get("dynamic_global_batch_size", args.global_batch_size), "global batch size")
    local_size = size // dp_size
    if local_size <= 0:
        raise ValueError("FLOPs scheduling requires at least one logical sample per DP rank")
    count = len(payload["total_lengths"])
    minis = payload.get("rollout_mini_local_sample_counts")
    if minis is not None:
        if (
            not isinstance(minis, list)
            or not minis
            or any(type(v) is not int or v <= 0 for v in minis)
            or sum(minis) != count
        ):
            raise ValueError("Invalid rollout mini local boundaries")
        steps = len(minis)
    elif count <= local_size:
        steps = 1
    elif count % local_size:
        raise ValueError("Rollout rows without explicit mini boundaries must divide the local batch size")
    else:
        steps = count // local_size
    global_minis = payload.get("rollout_mini_global_sample_counts")
    if global_minis is not None and (
        not isinstance(global_minis, list)
        or len(global_minis) != steps
        or any(type(v) is not int or v <= 0 for v in global_minis)
    ):
        raise ValueError("Invalid rollout mini global boundaries")
    return steps, size


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _weights(values: Sequence[float], count: int) -> list[float]:
    if len(values) != count:
        raise ValueError("Expected exactly one FLOPs weight per row")
    result = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, Real):
            raise ValueError("FLOPs weights must be finite and positive")
        try:
            cost = float(value)
        except OverflowError as error:
            raise ValueError("FLOPs weights must be finite and positive") from error
        if not math.isfinite(cost) or cost <= 0:
            raise ValueError("FLOPs weights must be finite and positive")
        result.append(cost)
    try:
        if not math.isfinite(math.fsum(result)):
            raise ValueError("Total FLOPs must be finite")
    except OverflowError as error:
        raise ValueError("Total FLOPs must be finite") from error
    return result


@dataclass(frozen=True)
class MicrobatchPlan:
    indices: tuple[tuple[int, ...], ...]
    dummy_offsets: frozenset[int]
    logical_tokens: tuple[int, ...]
    physical_tokens: tuple[int, ...]
    flops: tuple[float, ...]
    oversized_singletons: tuple[int, ...]
    fallback_reason: str | None
    cpu_seconds: float


def plan_microbatches(
    logical_lengths: Sequence[int],
    physical_lengths: Sequence[int],
    weights: Sequence[float],
    required_microbatches: int,
    capacity: int,
    legacy_fallback: Callable[[list[int], int, int], tuple[list[list[int]], set[int]]],
) -> MicrobatchPlan:
    """KK weights at existing DP/VPP K, or delegate to the real legacy
    fallback.

    Inputs are rank-local mini rows. Physical lengths are supplied by the
    existing layout/capacity path. Oversized singletons retain the legacy
    exception, never an assertion of memory safety. Dummy slots duplicate one
    local real MB.
    """
    started = time.perf_counter()
    logical = [_positive_int(v, "logical length") for v in logical_lengths]
    physical = [_positive_int(v, "physical length") for v in physical_lengths]
    if not logical or len(logical) != len(physical) or any(p < l for p, l in zip(physical, logical, strict=True)):
        raise ValueError("Need nonempty aligned row metadata with physical lengths >= logical lengths")
    costs = _weights(weights, len(logical))
    count = _positive_int(required_microbatches, "required_microbatches")
    capacity = _positive_int(capacity, "capacity")
    parts = get_seqlen_balanced_partitions(costs, min(count, len(logical)), equal_size=False)

    def fits(groups: list[list[int]]) -> bool:
        return all(sum(physical[i] for i in g) <= capacity or len(g) == 1 for g in groups)

    reason = None
    dummy = set()
    if not fits(parts):
        reason = "weighted_kk_token_capacity"
        parts, dummy = legacy_fallback(physical, count, capacity)
    else:
        shortest = min(parts, key=lambda group: sum(physical[i] for i in group))
        while len(parts) < count:
            dummy.add(len(parts))
            parts.append(shortest.copy())
    if (
        len(parts) != count
        or not dummy <= set(range(count))
        or any(not group or any(type(i) is not int or not 0 <= i < len(logical) for i in group) for group in parts)
    ):
        raise RuntimeError("FLOPs microbatch planner produced invalid indices/count/dummy offsets")
    if sorted(i for j, group in enumerate(parts) if j not in dummy for i in group) != list(range(len(logical))):
        raise RuntimeError("FLOPs microbatches must cover each real row exactly once")
    if not fits(parts) or any(
        not any(group == parts[j] for j in range(count) if j not in dummy)
        for i, group in enumerate(parts)
        if i in dummy
    ):
        raise RuntimeError("FLOPs plan violates token capacity or dummy source contract")
    return MicrobatchPlan(
        indices=tuple(tuple(group) for group in parts),
        dummy_offsets=frozenset(dummy),
        logical_tokens=tuple(sum(logical[i] for i in group) for group in parts),
        physical_tokens=tuple(sum(physical[i] for i in group) for group in parts),
        flops=tuple(sum(costs[i] for i in group) for group in parts),
        oversized_singletons=tuple(j for j, group in enumerate(parts) if sum(physical[i] for i in group) > capacity),
        fallback_reason=reason,
        cpu_seconds=time.perf_counter() - started,
    )


def assign_dp_rows(rows: Sequence[int], weights: Sequence[float], dp_size: int) -> list[list[int]]:
    """Redistribute an already admitted set, preserving IDs and equal-count
    policy."""
    dp_size = _positive_int(dp_size, "dp_size")
    if len(set(rows)) != len(rows) or len(rows) < dp_size:
        raise ValueError("DP placement requires unique rows and at least one row per rank")
    costs = _weights(weights, len(rows))
    positions = get_seqlen_balanced_partitions(costs, dp_size, equal_size=len(rows) % dp_size == 0)
    return [[rows[i] for i in part] for part in positions]


@dataclass(frozen=True)
class SchedulingContext:
    mode: str
    estimator: RelaxFlopsEstimator
    signature: str
    control_group: Any = None


def runtime_context(args: Any, model_config: Any = None) -> SchedulingContext | None:
    if scheduling_mode(args) == "off":
        return None
    if model_config is None:
        path = Path(args.hf_checkpoint) / "config.json"
        if not path.is_file():
            raise ValueError("FLOPs scheduling requires a locally available HF config.json")
        model_config = json.loads(path.read_text(encoding="utf-8"))
    estimator = build_estimator(args, model_config)
    config = model_config.to_dict() if hasattr(model_config, "to_dict") else model_config
    data = json.dumps({"mode": scheduling_mode(args), "config": config}, sort_keys=True, default=vars)
    return SchedulingContext(scheduling_mode(args), estimator, hashlib.sha256(data.encode()).hexdigest())


@dataclass(frozen=True)
class RowPlacement:
    context: SchedulingContext

    @property
    def signature(self) -> str:
        return self.context.signature

    def __call__(self, rows: list[int], metadata: dict[int, dict], dp_size: int) -> list[list[int]]:
        started = time.perf_counter()
        if any(metadata[i].get("multimodal_train_inputs") is not None for i in rows):
            raise ValueError("FLOPs DP scheduling does not yet support visual row metadata")
        costs = row_weights(self.context.estimator, [metadata[i]["total_lengths"] for i in rows])
        groups = assign_dp_rows(rows, costs, dp_size)
        from relax.utils.logging_utils import get_logger

        by_row = dict(zip(rows, costs, strict=True))
        get_logger(__name__).info(
            "FLOPs DP placement: dp=%s noop=%s rank_flops=%s rank_rows=%s cpu_seconds=%s",
            dp_size,
            dp_size == 1,
            [sum(by_row[i] for i in group) for group in groups],
            [len(group) for group in groups],
            time.perf_counter() - started,
        )
        return groups
