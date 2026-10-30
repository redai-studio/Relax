# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""CPU sample costs for capacity-constrained dynamic microbatch balancing."""

import math
from argparse import Namespace
from collections.abc import Callable, Sequence
from typing import Any


SampleCostEstimator = Callable[[Any, dict[str, Any], Namespace], Sequence[float]]
_ESTIMATORS: dict[str, SampleCostEstimator] = {}


def register_sample_cost_estimator(model_type: str, estimator: SampleCostEstimator) -> None:
    """Register a CPU-only estimator(config, rollout_data, args).

    Return one positive finite cost per sample, in the original order. The
    estimator must not mutate data, access CUDA tensors or run collectives.
    Import registrations in each worker via RELAX_EXTRA_MODULES.
    """
    if not model_type or not callable(estimator):
        raise ValueError("A nonempty model_type and callable estimator are required")
    if model_type in _ESTIMATORS:
        raise ValueError(f"Sample cost estimator already registered for {model_type!r}")
    _ESTIMATORS[model_type] = estimator


def validate_microbatch_balance(args: Namespace) -> bool:
    """Reject modes that bypass the supported dynamic partitioner."""
    mode = getattr(args, "microbatch_balance", "tokens")
    if mode not in ("tokens", "flops"):
        raise ValueError("--microbatch-balance must be tokens or flops")
    if mode == "tokens":
        return False
    if getattr(args, "train_backend", "megatron") != "megatron":
        raise ValueError("--microbatch-balance flops requires --train-backend megatron")
    if not getattr(args, "use_dynamic_batch_size", False):
        raise ValueError("FLOPs balancing requires --use-dynamic-batch-size")
    if getattr(args, "sft_packing_strategy", "balanced") != "balanced":
        raise ValueError("FLOPs balancing requires balanced packing, not binpack")
    if getattr(args, "sft_wholepack", False):
        raise ValueError("FLOPs balancing does not support wholepack")
    if getattr(args, "fully_async", False) or getattr(args, "hybrid", False):
        raise ValueError("FLOPs balancing does not support fully-async streaming/drain scheduling")
    if getattr(args, "loss_type", None) in ("dpo", "rm"):
        raise ValueError("FLOPs balancing does not support preference pairs")
    return True


def sample_costs(args: Namespace, rollout_data: dict[str, Any]) -> list[float] | None:
    """Resolve the model estimator once per window, without changing data."""
    if not validate_microbatch_balance(args):
        return None
    if "_sft_wholepack_v1" in rollout_data:
        raise ValueError("FLOPs balancing cannot consume an existing wholepack plan")
    from relax.utils.misc import get_hf_config

    config = get_hf_config(args.hf_checkpoint)
    model_type = config.model_type
    if model_type in ("qwen3_5", "qwen3_5_text", "qwen3_5_moe", "qwen3_5_moe_text"):
        from relax.utils.training.qwen35_cost import estimate_sample_costs

        estimator = _ESTIMATORS.get(model_type, estimate_sample_costs)
    else:
        estimator = _ESTIMATORS.get(model_type)
    if estimator is None:
        raise ValueError(
            f"No FLOPs sample cost estimator for model_type={model_type!r}; "
            "register_sample_cost_estimator or use --microbatch-balance tokens"
        )
    costs = [float(value) for value in estimator(config, rollout_data, args)]
    if len(costs) != len(rollout_data["total_lengths"]):
        raise ValueError("Sample cost count must match total_lengths")
    if any(not math.isfinite(value) or value <= 0 for value in costs):
        raise ValueError("Sample costs must be positive and finite")
    return costs
