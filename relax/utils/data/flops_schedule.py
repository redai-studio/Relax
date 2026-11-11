# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Pure CPU planning used by ordinary and prefetched SFT microbatches."""

from dataclasses import dataclass
from typing import Any

from relax.utils.data.workload_balancing import pack_by_workload
from relax.utils.training.flops_counter import SequenceFlopsEstimator


def flops_scheduling_enabled(args: Any) -> bool:
    return bool(getattr(args, "flops_microbatch_packing", False) or getattr(args, "flops_dp_balancing", False))


def unequal_flops_dp_enabled(args: Any) -> bool:
    """Use variable DP row counts only for the validated fixed-window SFT
    path."""
    if not (
        getattr(args, "flops_dp_balancing", False)
        and getattr(args, "loss_type", None) == "sft"
        and getattr(args, "task_type", "causal_lm") == "causal_lm"
        and getattr(args, "train_backend", "megatron") == "megatron"
        and getattr(args, "use_dynamic_batch_size", False)
        and getattr(args, "qkv_format", "thd") == "thd"
        and getattr(args, "n_samples_per_prompt", 1) == 1
        and getattr(args, "calculate_per_token_loss", False)
    ):
        return False
    if any(
        getattr(args, name, 1) != 1
        for name in (
            "tensor_model_parallel_size",
            "pipeline_model_parallel_size",
            "context_parallel_size",
            "expert_model_parallel_size",
            "expert_tensor_parallel_size",
        )
    ):
        return False
    if getattr(args, "virtual_pipeline_model_parallel_size", None) not in (None, 1):
        return False
    if getattr(args, "num_layers_per_virtual_pipeline_stage", None) is not None:
        return False
    if (
        any(
            getattr(args, name, False)
            for name in (
                "sft_async_prepack",
                "fully_async",
                "hybrid",
                "dynamic_context_parallel",
                "partial_rollout",
                "use_dynamic_global_batch_size",
                "sft_wholepack",
                "debug_train_only",
                "multimodal_keys",
            )
        )
        or getattr(args, "generation_task", None) == "i2i"
    ):
        return False
    batch_size = getattr(args, "global_batch_size", None)
    return (
        type(batch_size) is int
        and batch_size > 0
        and getattr(args, "rollout_batch_size", None) == batch_size
        and getattr(args, "num_steps_per_rollout", None) in (None, 1)
    )


def validate_flops_scheduling_args(args: Any) -> None:
    if not flops_scheduling_enabled(args):
        return
    if getattr(args, "loss_type", None) != "sft" or getattr(args, "task_type", "causal_lm") != "causal_lm":
        raise ValueError("FLOPs scheduling currently supports causal-LM SFT only")
    if getattr(args, "train_backend", "megatron") != "megatron":
        raise ValueError("FLOPs scheduling currently requires the Megatron backend")
    if not getattr(args, "use_dynamic_batch_size", False):
        raise ValueError("FLOPs scheduling requires --use-dynamic-batch-size")
    if getattr(args, "sft_wholepack", False):
        raise ValueError("FLOPs scheduling and --sft-wholepack are mutually exclusive")
    if (
        getattr(args, "flops_microbatch_packing", False)
        and getattr(args, "sft_packing_strategy", "balanced") != "balanced"
    ):
        raise ValueError(
            "FLOPs microbatch packing replaces the balanced strategy; disable --sft-packing-strategy binpack"
        )
    if getattr(args, "qkv_format", "thd") != "thd":
        raise ValueError("FLOPs scheduling currently requires THD packing")
    if getattr(args, "dynamic_context_parallel", False):
        raise ValueError("FLOPs scheduling does not yet support dynamic context parallelism")
    if getattr(args, "fully_async", False) and not getattr(args, "sft_async_prepack", False):
        raise ValueError("FLOPs scheduling does not support the streaming fully-async sampler")
    if getattr(args, "multimodal_keys", None):
        raise ValueError("FLOPs scheduling is text-only; multimodal inputs are not supported")
    if getattr(args, "mtp_num_layers", 0) not in (None, 0):
        raise ValueError("Disable MTP for FLOPs scheduling; its extra work is not modeled")
    for field in ("moe_aux_loss_coeff", "moe_z_loss_coeff"):
        value = getattr(args, field, 0)
        values = value if isinstance(value, (list, tuple)) else (value,)
        if any(item not in (None, 0, 0.0) for item in values):
            raise ValueError(f"FLOPs scheduling requires zero {field}; auxiliary gradients are not dummy-masked")
    if getattr(args, "moe_expert_capacity_factor", None) is not None:
        raise ValueError("FLOPs scheduling does not support expert-capacity token dropping")
    if getattr(args, "moe_router_enable_expert_bias", False):
        raise ValueError("FLOPs scheduling does not support dynamic expert-bias updates")


def get_flops_estimator(args: Any, hf_config: Any = None) -> SequenceFlopsEstimator | None:
    """Cache coefficients once per owner; never inspect the model per
    sample."""
    if not flops_scheduling_enabled(args):
        return None
    estimator = getattr(args, "_flops_workload_estimator", None)
    if estimator is None:
        if hf_config is None:
            from relax.utils.misc import get_hf_config

            hf_config = get_hf_config(args.hf_checkpoint)
        estimator = SequenceFlopsEstimator.from_config(hf_config)
        args._flops_workload_estimator = estimator
    return estimator


def validate_text_rollout(rollout_data: dict[str, Any]) -> None:
    media = rollout_data.get("multimodal_train_inputs")
    if media is not None:
        values = media if isinstance(media, (list, tuple)) else (media,)
        if any(value is not None and (not isinstance(value, dict) or len(value) > 0) for value in values):
            raise ValueError("FLOPs scheduling received multimodal data; only text workloads are modeled")


def flops_microbatch_geometry(args: Any, tp_size: int, cp_size: int) -> dict[str, Any]:
    return {
        "tp_size": tp_size,
        "cp_size": cp_size,
        "pad_multiplier": getattr(args, "data_pad_size_multiplier", 128),
        "is_vl_model": getattr(args, "is_vl_model", False) or getattr(args, "uses_unsplit_forward", False),
        "allgather_cp": getattr(args, "allgather_cp", False),
    }


@dataclass(frozen=True)
class WorkloadMicrobatchPlan:
    partitions: list[list[int]]
    dummy_offsets: frozenset[int]
    capacity_lengths: list[int]
    max_capacity: int


def expand_workload_microbatch_plan(plan: WorkloadMicrobatchPlan, count: int) -> WorkloadMicrobatchPlan:
    """Expand an already validated plan after DP/VPP agreement, on CPU only.

    Splitting legal groups preserves capacity and coverage. Once each real row
    has its own group, remaining slots repeat the shortest row with zero loss.
    No second model lookup, cost estimate or heuristic packing can fail after
    peers have agreed on the execution count.
    """
    if type(count) is not int or count < len(plan.partitions):
        raise ValueError("The agreed count must not shrink a validated plan")
    groups = [list(group) for offset, group in enumerate(plan.partitions) if offset not in plan.dummy_offsets]
    while len(groups) < min(count, len(plan.capacity_lengths)):
        index = max(range(len(groups)), key=lambda i: (len(groups[i]), -i))
        group = groups[index]
        middle = len(group) // 2
        groups[index], tail = group[:middle], group[middle:]
        groups.append(tail)
    dummy_offsets = frozenset(range(len(groups), count))
    shortest = min(groups, key=lambda group: (sum(plan.capacity_lengths[i] for i in group), group))
    groups.extend(list(shortest) for _ in dummy_offsets)
    return WorkloadMicrobatchPlan(groups, dummy_offsets, plan.capacity_lengths, plan.max_capacity)


def build_workload_microbatch_plan(
    lengths: list[int],
    estimator: SequenceFlopsEstimator,
    max_tokens_per_gpu: int,
    *,
    tp_size: int = 1,
    cp_size: int = 1,
    pad_multiplier: int = 128,
    is_vl_model: bool = False,
    allgather_cp: bool = False,
    num_microbatches: int | None = None,
) -> WorkloadMicrobatchPlan:
    """Plan real samples and optional zero-loss slots for one optimizer step.

    Capacity mirrors THD's per-sample CP padding and final TP-aligned local
    padding. It bounds the additive token representation, not total GPU memory.
    Work estimates retain the existing raw-text (unpadded) accounting
    convention.
    """
    for name, value in (
        ("max_tokens_per_gpu", max_tokens_per_gpu),
        ("tp_size", tp_size),
        ("cp_size", cp_size),
        ("pad_multiplier", pad_multiplier),
    ):
        if type(value) is not int or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if num_microbatches is not None and (type(num_microbatches) is not int or num_microbatches <= 0):
        raise ValueError("num_microbatches must be a positive integer")
    costs = estimator.estimate_per_sample(lengths)
    lengths = [int(length) for length in lengths]
    if not lengths:
        raise ValueError("A training step must contain at least one real sample")
    local_alignment = tp_size * pad_multiplier
    local_capacity = max_tokens_per_gpu // local_alignment * local_alignment
    if local_capacity == 0:
        raise ValueError("Token budget is smaller than the final THD padding alignment")
    sample_alignment = 1
    if cp_size > 1 and (is_vl_model or not allgather_cp):
        sample_alignment = 2 * cp_size * (tp_size if is_vl_model else 1)
    capacity_lengths = [((length + sample_alignment - 1) // sample_alignment) * sample_alignment for length in lengths]
    capacity = local_capacity * cp_size
    real_count = None if num_microbatches is None else min(num_microbatches, len(lengths))
    groups = pack_by_workload(capacity_lengths, costs, capacity, num_partitions=real_count)
    dummy_offsets: set[int] = set()
    if num_microbatches is not None and len(groups) < num_microbatches:
        shortest = min(groups, key=lambda group: (sum(capacity_lengths[i] for i in group), group))
        for offset in range(len(groups), num_microbatches):
            groups.append(list(shortest))
            dummy_offsets.add(offset)
    return WorkloadMicrobatchPlan(groups, frozenset(dummy_offsets), capacity_lengths, capacity)
