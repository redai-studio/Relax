# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""CPU sample costs for capacity-constrained dynamic microbatch balancing."""

import math
from argparse import Namespace
from collections.abc import Callable, Sequence
from typing import Any


SampleCostEstimator = Callable[[Any, dict[str, Any], Namespace], Sequence[float]]
_ESTIMATORS: dict[str, SampleCostEstimator] = {}


def configure_flops_balance(args: Namespace) -> None:
    """Validate independent objectives after the final CLI/YAML overrides."""
    validate_microbatch_balance(args)
    validate_dp_balance(args)


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


def _uses_flops(args: Namespace, attribute: str) -> bool:
    mode = getattr(args, attribute, "tokens")
    if mode not in ("tokens", "flops"):
        raise ValueError(f"--{attribute.replace('_', '-')} must be tokens or flops")
    return mode == "flops"


def _validate_flops_configuration(args: Namespace) -> None:
    if getattr(args, "train_backend", "megatron") != "megatron":
        raise ValueError("FLOPs balancing requires --train-backend megatron")
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
    auxiliary = getattr(args, "moe_aux_loss_coeff", 0.0)
    auxiliary = auxiliary if isinstance(auxiliary, (list, tuple)) else [auxiliary]
    if any(value is not None and value != 0 for value in auxiliary):
        raise ValueError("FLOPs balancing requires zero MoE auxiliary-loss coefficients to preserve the objective")
    if getattr(args, "moe_expert_capacity_factor", None) is not None:
        raise ValueError("FLOPs balancing does not support microbatch-dependent MoE token dropping")
    routing = getattr(args, "moe_router_load_balancing_type", "none")
    routing = routing if isinstance(routing, (list, tuple)) else [routing]
    if any(mode not in ("none", "aux_loss", "seq_aux_loss", "global_aux_loss") for mode in routing):
        raise ValueError("FLOPs balancing requires token-independent MoE routing; batch-coupled modes are unsupported")


def validate_microbatch_balance(args: Namespace) -> bool:
    """Validate only the rank-local microbatch objective."""
    if not _uses_flops(args, "microbatch_balance"):
        return False
    _validate_flops_configuration(args)
    return True


def validate_dp_balance(args: Namespace) -> bool:
    """Validate DP placement without changing microbatches or legacy
    enablement."""
    if not _uses_flops(args, "dp_balance"):
        return False
    _validate_flops_configuration(args)
    if getattr(args, "loss_type", None) == "sft":
        if getattr(args, "n_samples_per_prompt", 1) != 1:
            raise ValueError("FLOPs DP balancing for SFT requires n_samples_per_prompt=1")
        if getattr(args, "sft_image_preprocess_on_rank", False):
            raise ValueError("FLOPs DP balancing requires CPU media metadata before queue publication")
    elif not getattr(args, "balance_data", False):
        raise ValueError("--dp-balance flops requires --balance-data for non-SFT training")
    return True


def sample_costs(args: Namespace, rollout_data: dict[str, Any], *, for_dp: bool = False) -> list[float] | None:
    """Resolve costs only for the explicitly selected microbatch or DP
    objective."""
    validate = validate_dp_balance if for_dp else validate_microbatch_balance
    if not validate(args):
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
        option = "--dp-balance" if for_dp else "--microbatch-balance"
        raise ValueError(
            f"No FLOPs sample cost estimator for model_type={model_type!r}; "
            f"register_sample_cost_estimator or use {option} tokens"
        )
    costs = [float(value) for value in estimator(config, rollout_data, args)]
    if len(costs) != len(rollout_data["total_lengths"]):
        raise ValueError("Sample cost count must match total_lengths")
    if any(not math.isfinite(value) or value <= 0 for value in costs):
        raise ValueError("Sample costs must be positive and finite")
    return costs


def add_flops_cost_metadata(
    args: Namespace | None, batch: dict[str, Any], metadata: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Attach CPU costs before payloads enter the queue's sampling layer."""
    if args is None or not validate_dp_balance(args):
        return metadata
    costs = sample_costs(args, batch, for_dp=True)
    if costs is None or len(costs) != len(metadata):
        raise ValueError("FLOPs DP metadata requires one model cost per physical sample")
    return [{**row, "flops_cost": cost} for row, cost in zip(metadata, costs, strict=True)]
