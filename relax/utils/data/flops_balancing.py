# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""CPU workload estimation and capacity-constrained microbatch scheduling."""

import math
from argparse import Namespace
from collections.abc import Sequence
from dataclasses import dataclass
from numbers import Integral
from types import SimpleNamespace
from typing import Any

from relax.utils.data.packing import get_first_fit_partitions
from relax.utils.data.seqlen_balancing import get_seqlen_balanced_partitions
from relax.utils.logging_utils import get_logger
from relax.utils.training.flops_counter import _ESTIMATE_FUNC


logger = get_logger(__name__)

_SUPPORTED_MODELS = {"qwen2", "qwen3", "qwen2_moe", "qwen3_moe", "qwen3_5", "qwen3_5_moe"}


def _namespace(value: Any) -> Any:
    if isinstance(value, dict) and all(isinstance(key, str) for key in value):
        return SimpleNamespace(**{key: _namespace(item) for key, item in value.items()})
    return value


@dataclass(frozen=True)
class FlopsWorkload:
    """Serializable model configuration; estimates useful forward/backward
    FLOPs.

    Use the existing model formulas without calling the MFU/device-peak API.
    Full attention contributes sum(L²), not the square of packed token count.
    """

    model_type: str
    config: Any

    @classmethod
    def from_config(cls, config: Any) -> "FlopsWorkload":
        config = _namespace(config.to_dict() if hasattr(config, "to_dict") else config)
        text = getattr(config, "text_config", config)
        model_type = getattr(config, "model_type", "").removesuffix("_text")
        if model_type not in _SUPPORTED_MODELS:
            raise ValueError(f"--balance-by-flops does not support model_type={model_type!r}")
        workload = cls(model_type, text)
        try:
            workload([1])
        except (AttributeError, TypeError, ValueError, ZeroDivisionError) as exc:
            raise ValueError(f"Invalid {model_type} configuration for --balance-by-flops: {exc}") from exc
        return workload

    def __call__(self, lengths: Sequence[int]) -> list[float]:
        estimator = _ESTIMATE_FUNC[self.model_type]
        costs: dict[int, float] = {}
        for length in lengths:
            if not isinstance(length, Integral) or isinstance(length, bool) or length <= 0:
                raise ValueError(f"FLOPs scheduling requires positive integer sequence lengths, got {length!r}")
            if length not in costs:
                # delta_time=1 converts the existing TFLOP/s formula to TFLOPs.
                cost = float(estimator(self.config, length, [length], 1.0)) * 1e12
                if self.model_type in {"qwen2_moe", "qwen3_moe"}:
                    shared_width = getattr(self.config, "shared_expert_intermediate_size", 0)
                    if shared_width:
                        cost += (
                            6
                            * self.config.num_hidden_layers
                            * self.config.hidden_size
                            * (3 * shared_width + 1)
                            * length
                        )
                if not math.isfinite(cost) or cost <= 0:
                    raise ValueError("FLOPs scheduling requires finite, positive workload estimates")
                costs[length] = cost
        return [costs[length] for length in lengths]


def load_flops_workload(args: Namespace) -> FlopsWorkload:
    """Read configuration only, including before S3 weights are staged
    locally."""
    source = getattr(args, "model_source", None)
    if source is not None and source.uri.startswith("s3://"):
        from transformers import AutoConfig

        from relax.utils.s3_model_loader import read_s3_model_config

        metadata = read_s3_model_config(args)
        config = AutoConfig.for_model(
            metadata["model_type"], **{key: value for key, value in metadata.items() if key != "model_type"}
        )
    else:
        from relax.utils.misc import get_hf_config

        config = get_hf_config(args.hf_checkpoint)
    return FlopsWorkload.from_config(config)


def validate_flops_balancing(args: Namespace) -> None:
    if not getattr(args, "balance_by_flops", False):
        return
    if not args.use_dynamic_batch_size or not args.max_tokens_per_gpu or args.max_tokens_per_gpu <= 0:
        raise ValueError("--balance-by-flops requires --use-dynamic-batch-size and positive --max-tokens-per-gpu")
    incompatible = {
        "non-Megatron backend": getattr(args, "train_backend", "megatron") != "megatron",
        "--fully-async/--hybrid": args.fully_async or args.hybrid,
        "--sft-wholepack": getattr(args, "sft_wholepack", False),
        "--sft-async-prepack": getattr(args, "sft_async_prepack", False),
        "--sft-packing-strategy binpack": args.sft_packing_strategy != "balanced",
        "preference objectives": args.loss_type in {"dpo", "rm"},
        "non-causal-lm task": getattr(args, "task_type", "causal_lm") != "causal_lm",
        "MTP": bool(getattr(args, "enable_mtp_training", False) or getattr(args, "mtp_only_training", False)),
        "multimodal input": bool(getattr(args, "multimodal_keys", None)),
        "SFT n_samples_per_prompt != 1": args.loss_type == "sft" and args.n_samples_per_prompt != 1,
    }
    # Bridge can infer experts from HF even when args.num_experts is unset.
    coefficients = getattr(args, "moe_aux_loss_coeff", 0.0) or 0.0
    if not isinstance(coefficients, (tuple, list)):
        coefficients = [coefficients]
    routing = getattr(args, "moe_router_load_balancing_type", None)
    if not isinstance(routing, (tuple, list)):
        routing = [routing]
    incompatible.update(
        {
            "MoE auxiliary loss": any(value != 0 for value in coefficients),
            "MoE router z-loss": bool(getattr(args, "moe_z_loss_coeff", None)),
            "MoE Sinkhorn routing": "sinkhorn" in routing,
            "MoE expert capacity": getattr(args, "moe_expert_capacity_factor", None) is not None,
            "MoE dynamic expert bias": getattr(args, "moe_router_enable_expert_bias", False),
        }
    )
    conflicts = [name for name, active in incompatible.items() if active]
    if conflicts:
        raise ValueError("--balance-by-flops does not support: " + ", ".join(conflicts))
    args.balance_data = True


@dataclass
class FlopsMicrobatchPlan:
    lengths: list[int]
    workloads: list[float]
    capacity: int
    partitions: list[list[int]]


def _balanced_candidate(
    lengths: list[int], workloads: list[float], capacity: int, count: int
) -> list[list[int]] | None:
    oversized = [[i] for i, length in enumerate(lengths) if length > capacity]
    regular = [i for i, length in enumerate(lengths) if length <= capacity]
    if not regular:
        return oversized if len(oversized) == count else None
    regular_count = count - len(oversized)
    if not 1 <= regular_count <= len(regular):
        return None
    groups = get_seqlen_balanced_partitions([workloads[i] for i in regular], regular_count, equal_size=False)
    partitions = [[regular[i] for i in group] for group in groups]
    if any(sum(lengths[i] for i in group) > capacity for group in partitions):
        return None
    return oversized + partitions


def plan_flops_microbatches(lengths: Sequence[int], workloads: Sequence[float], capacity: int) -> FlopsMicrobatchPlan:
    lengths, workloads = list(lengths), list(workloads)
    if capacity <= 0 or len(lengths) != len(workloads):
        raise ValueError("FLOPs packing requires positive capacity and one workload per sequence")
    if any(length <= 0 for length in lengths) or any(not math.isfinite(cost) or cost <= 0 for cost in workloads):
        raise ValueError("FLOPs packing requires positive lengths and finite positive workloads")
    regular_tokens = sum(length for length in lengths if length <= capacity)
    lower_bound = sum(length > capacity for length in lengths) + (regular_tokens + capacity - 1) // capacity
    partitions = _balanced_candidate(lengths, workloads, capacity, lower_bound)
    if partitions is None:
        partitions = get_first_fit_partitions(lengths, capacity)
        logger.debug("FLOPs packing exceeded token capacity; first-fit fallback K=%s", len(partitions))
    return FlopsMicrobatchPlan(lengths, workloads, capacity, partitions)


def align_flops_microbatches(plan: FlopsMicrobatchPlan, count: int) -> tuple[list[list[int]], set[int]]:
    """Align an already feasible plan to DP/VPP K without introducing
    overflow."""
    if count < len(plan.partitions) or (count > 0 and not plan.partitions):
        raise ValueError("Aligned microbatch count must cover the feasible non-empty plan")
    real_count = min(count, len(plan.lengths))
    partitions = _balanced_candidate(plan.lengths, plan.workloads, plan.capacity, real_count)
    if partitions is None:
        partitions = [group.copy() for group in plan.partitions]
        while len(partitions) < real_count:
            splittable = [i for i, group in enumerate(partitions) if len(group) > 1]
            target = max(splittable, key=lambda i: (sum(plan.workloads[j] for j in partitions[i]), -i))
            sample = max(partitions[target], key=lambda i: (plan.workloads[i], -i))
            partitions[target].remove(sample)
            partitions.append([sample])
    dummy = set(range(len(partitions), count))
    if dummy:
        cheapest = min(partitions, key=lambda group: sum(plan.workloads[i] for i in group))
        partitions.extend(cheapest.copy() for _ in dummy)
    return partitions, dummy


def log_flops_balance(workloads: Sequence[float], partitions: Sequence[Sequence[int]]) -> None:
    loads = [sum(workloads[i] for i in group) for group in partitions]
    if loads:
        logger.debug(
            "FLOPs scheduling partitions=%s estimated_flops_min=%g mean=%g max=%g",
            len(loads),
            min(loads),
            sum(loads) / len(loads),
            max(loads),
        )
