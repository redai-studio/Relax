# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Pack a complete optimizer step, then assign whole microbatches to DP
ranks."""

import hashlib
import json
import math
from argparse import Namespace
from collections.abc import Sequence
from dataclasses import dataclass
from numbers import Integral
from types import SimpleNamespace
from typing import Any

from relax.utils.data.packing import get_binpack_partitions
from relax.utils.data.seqlen_balancing import get_seqlen_balanced_partitions
from relax.utils.logging_utils import get_logger
from relax.utils.training.flops_counter import _ESTIMATE_FUNC


logger = get_logger(__name__)

_SUPPORTED_MODELS = {"qwen2", "qwen3", "qwen2_moe", "qwen3_moe", "qwen3_5", "qwen3_5_moe"}
PLAN_METADATA_FIELD = "_flops_schedule_v1"
STEP_PLANS_FIELD = "_flops_step_plans"


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
        config = _namespace(config)
        text = getattr(config, "text_config", config)
        model_type = getattr(config, "model_type", "").removesuffix("_text")
        if model_type not in _SUPPORTED_MODELS:
            raise ValueError(f"--balance-by-flops does not support model_type={model_type!r}")
        workload = cls(model_type, text)
        try:
            if model_type in {"qwen2_moe", "qwen3_moe"} and not hasattr(text, "num_experts"):
                text.num_experts = text.num_local_experts
            workload([1])
        except (AttributeError, TypeError, ValueError, ZeroDivisionError) as exc:
            raise ValueError(f"Invalid {model_type} configuration for --balance-by-flops: {exc}") from exc
        return workload

    def __call__(self, lengths: Sequence[int]) -> list[float]:
        estimator = _ESTIMATE_FUNC[self.model_type]
        mlp_params_delta = 0
        if self.model_type in {"qwen2_moe", "qwen3_moe"}:
            config = self.config
            sparse_step = getattr(config, "decoder_sparse_step", 1)
            dense_only = set(getattr(config, "mlp_only_layers", None) or [])
            moe_layers = sum(
                config.num_experts > 0 and layer not in dense_only and (layer + 1) % sparse_step == 0
                for layer in range(config.num_hidden_layers)
            )
            dense_layers = config.num_hidden_layers - moe_layers
            # The shared estimator assumes routed MoE in every layer.
            if dense_layers:
                routed_params = config.hidden_size * (
                    3 * config.num_experts_per_tok * config.moe_intermediate_size + config.num_experts
                )
                dense_params = 3 * config.hidden_size * config.intermediate_size
                mlp_params_delta = dense_layers * (dense_params - routed_params)
            shared_width = getattr(config, "shared_expert_intermediate_size", 0)
            if shared_width:
                mlp_params_delta += moe_layers * config.hidden_size * (3 * shared_width + 1)
        costs: dict[int, float] = {}
        for length in lengths:
            if not isinstance(length, Integral) or isinstance(length, bool) or length <= 0:
                raise ValueError(f"FLOPs scheduling requires positive integer sequence lengths, got {length!r}")
            if length not in costs:
                # delta_time=1 converts the existing TFLOP/s formula to TFLOPs.
                cost = float(estimator(self.config, length, [length], 1.0)) * 1e12
                cost += 6 * length * mlp_params_delta
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
        "--debug-train-only (bypasses the queue planner)": getattr(args, "debug_train_only", False),
        "non-polling TransferQueue (requires fixed row counts)": not getattr(args, "polling_mode", True),
        "--sft-packing-strategy binpack": args.sft_packing_strategy != "balanced",
        "preference objectives": args.loss_type in {"dpo", "rm"},
        "non-causal-lm task": getattr(args, "task_type", "causal_lm") != "causal_lm",
        "MTP": bool(getattr(args, "enable_mtp_training", False) or getattr(args, "mtp_only_training", False)),
        "multimodal input": bool(getattr(args, "multimodal_keys", None)),
        "SFT n_samples_per_prompt != 1": args.loss_type == "sft" and args.n_samples_per_prompt != 1,
        "microbatch-dependent M2PO clipping": getattr(args, "advantage_estimator", None) == "m2po",
        "microbatch-dependent direct OPD loss": bool(
            getattr(args, "use_opd", False) and getattr(args, "opd_loss_coef", 0.0)
        ),
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


def get_dp_packing_geometry(args: Namespace, task_name: str) -> dict[str, int] | None:
    if not getattr(args, "balance_by_flops", False) or task_name not in {
        "train",
        "train_critic",
        "actor_train",
        "critic_train",
        "sft_train",
    }:
        return None
    cp = getattr(args, "context_parallel_size", 1)
    tp = getattr(args, "tensor_model_parallel_size", 1)
    budget = args.max_tokens_per_gpu
    if getattr(args, "allgather_cp", False):
        alignment = tp * args.data_pad_size_multiplier
        if alignment <= 0 or budget < alignment:
            raise ValueError("allgather CP padding must fit within max_tokens_per_gpu")
        budget = budget // alignment * alignment
    unsplit = getattr(args, "is_vl_model", False) or getattr(args, "uses_unsplit_forward", False)
    return dict(
        capacity=budget * cp,
        sequence_alignment=tp * cp * 2 if cp > 1 and getattr(args, "qkv_format", "thd") == "thd" and unsplit else 1,
        microbatch_group_size=getattr(args, "flops_microbatch_group_size", 1),
        context_parallel_size=cp,
        tensor_parallel_size=tp,
        pipeline_stages=getattr(args, "pipeline_model_parallel_size", 1),
        dynamic_context_parallel=int(getattr(args, "dynamic_context_parallel", False)),
    )


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


def build_flops_schedule(
    lengths: list[int], workloads: list[float], capacity: int, dp_size: int, microbatch_group_size: int = 1
) -> tuple[list[list[list[int]]], list[set[int]]]:
    """Global FLOPs packing with DP/VPP-aligned repacking and rank placement.

    Token capacity is a hard limit except for indivisible oversized samples. If
    alignment needs more slots than samples, duplicate a local pack with zero
    loss; TransferQueue still delivers each real row exactly once.
    """
    if (
        any(
            not isinstance(value, Integral) or isinstance(value, bool) or value <= 0
            for value in [capacity, dp_size, microbatch_group_size, *lengths]
        )
        or len(lengths) < dp_size
    ):
        raise ValueError("FLOPs scheduling requires positive lengths/geometry and at least one sample per DP")
    if len(lengths) != len(workloads) or any(not math.isfinite(cost) or cost <= 0 for cost in workloads):
        raise ValueError("FLOPs scheduling requires one finite positive workload per sample")
    regular_tokens = sum(length for length in lengths if length <= capacity)
    lower_bound = sum(length > capacity for length in lengths) + (regular_tokens + capacity - 1) // capacity
    bins = _balanced_candidate(lengths, workloads, capacity, lower_bound)
    balanced = bins is not None
    if bins is None:
        bins = get_binpack_partitions(lengths, capacity)
    alignment = dp_size * microbatch_group_size
    target = (len(bins) + alignment - 1) // alignment * alignment
    # Bin-packing fills packs to capacity; rebalance it even when its count is already aligned.
    if (not balanced or len(bins) < target) and target <= len(lengths):
        aligned_bins = _balanced_candidate(lengths, workloads, capacity, target)
        if aligned_bins is not None:
            bins = aligned_bins
    while len(bins) < min(target, len(lengths)):
        _, index = max((sum(lengths[i] for i in group), j) for j, group in enumerate(bins) if len(group) > 1)
        halves: list[list[int]] = [[], []]
        totals = [0, 0]
        for i in sorted(bins[index], key=lambda i: -lengths[i]):
            half = 0 if totals[0] <= totals[1] else 1
            halves[half].append(i)
            totals[half] += lengths[i]
        bins[index] = halves[0]
        bins.append(halves[1])
    bins.extend([] for _ in range(target - len(bins)))
    weights = [sum(workloads[i] for i in group) for group in bins]
    owners = get_seqlen_balanced_partitions(weights, dp_size, equal_size=True)
    rank_bins, rank_dummy = [], []
    for owned in owners:
        dummy = {j for j, index in enumerate(owned) if not bins[index]}
        cheapest = min((bins[i] for i in owned if bins[i]), key=lambda group: sum(workloads[i] for i in group))
        rank_bins.append([bins[i] or cheapest.copy() for i in owned])
        rank_dummy.append(dummy)
    loads = [sum(weights[i] for i in group) for group in owners]
    logger.debug(
        "Global FLOPs schedule K=%s DP loads min=%g mean=%g max=%g",
        target,
        min(loads),
        sum(loads) / dp_size,
        max(loads),
    )
    return rank_bins, rank_dummy


def encode_schedule(
    rows: list[int],
    metadata: dict[int, dict[str, Any]],
    workload: FlopsWorkload,
    geometry: dict[str, int],
    dp_size: int,
    global_count: int,
    identity: tuple,
) -> tuple[list[list[int]], dict[int, dict[str, Any]]]:
    lengths = [metadata[i]["total_lengths"] for i in rows]
    costs = workload(lengths)
    pad = geometry["sequence_alignment"]
    physical_lengths = [(length + pad - 1) // pad * pad for length in lengths]
    bins, dummy = build_flops_schedule(
        physical_lengths, costs, geometry["capacity"], dp_size, geometry["microbatch_group_size"]
    )
    for rank, groups in enumerate(bins):
        order = list(range(len(groups)))
        if geometry["dynamic_context_parallel"]:
            order.sort(key=lambda j: max(lengths[i] for i in groups[j]))
        elif geometry["pipeline_stages"] == 1 and dp_size > 1:
            order.sort(key=lambda j: sum(costs[i] for i in groups[j]))
        bins[rank] = [groups[j] for j in order]
        dummy[rank] = {j for j, old in enumerate(order) if old in dummy[rank]}
    packs = [[[rows[i] for i in group] for group in rank] for rank in bins]
    digest = hashlib.sha256(json.dumps([identity, geometry, rows, lengths, costs, packs]).encode()).hexdigest()
    assignments, updates = [], {}
    for rank, groups in enumerate(packs):
        ids = [i for j, group in enumerate(groups) if j not in dummy[rank] for i in group]
        assignments.append(ids)
        for i in ids:
            updates[i] = {PLAN_METADATA_FIELD: {"sha256": digest}}
        updates[ids[0]][PLAN_METADATA_FIELD] = dict(
            sha256=digest,
            geometry=geometry,
            dp_rank=rank,
            dp_size=dp_size,
            global_count=global_count,
            rows=ids,
            lengths=[metadata[i]["total_lengths"] for i in ids],
            packs=groups,
            dummy=sorted(dummy[rank]),
        )
    return assignments, updates


def apply_schedule_metadata(
    partition: Any, updates: dict[int, dict[str, Any]] | None, geometry: dict[str, int] | None
) -> None:
    if updates is None:
        if geometry is not None:
            raise ValueError("FLOPs replay is missing its checkpoint plan")
        return
    header = next(row[PLAN_METADATA_FIELD] for row in updates.values() if "packs" in row[PLAN_METADATA_FIELD])
    if header["geometry"] != geometry:
        raise ValueError("FLOPs packing geometry changed across requests or checkpoint restore")
    current = partition.get_custom_meta(list(updates))
    partition.set_custom_meta({i: {**current[i], **update} for i, update in updates.items()})


def restore_schedule(batch_meta: Any, rollout_data: dict[str, Any]) -> None:
    """Restore the sampler's exact plan after the TQ payload/metadata
    broadcast."""
    entries = [row.get(PLAN_METADATA_FIELD) for row in batch_meta.get_all_custom_meta()]
    headers = [entry for entry in entries if entry and "packs" in entry]
    if not all(entries) or len(headers) != 1:
        raise ValueError("Missing or partial FLOPs schedule")
    header = headers[0]
    if any(entry["sha256"] != header["sha256"] for entry in entries):
        raise ValueError("Mixed FLOPs schedules")
    ids = list(batch_meta.global_indexes)
    if ids != header["rows"] or len(ids) != len(set(ids)) or list(rollout_data["total_lengths"]) != header["lengths"]:
        raise ValueError("FLOPs schedule row order/lengths differ from payload")
    if "tokens" in rollout_data and [len(tokens) for tokens in rollout_data["tokens"]] != header["lengths"]:
        raise ValueError("FLOPs schedule token lengths differ from payload")
    position = {index: i for i, index in enumerate(ids)}
    indices = [[position[index] for index in group] for group in header["packs"]]
    rollout_data[STEP_PLANS_FIELD] = [{**header, "indices": indices}]
    rollout_data["rollout_mini_local_sample_counts"] = [len(ids)]
    rollout_data["rollout_mini_global_sample_counts"] = [header["global_count"]]


def local_schedule(rollout_data: dict[str, Any]) -> tuple[list[list[int]], list[int], set[int]]:
    """Join step plans without repacking or crossing optimizer boundaries."""
    plans = rollout_data[STEP_PLANS_FIELD]
    indices, counts, dummy, offset = [], [], set(), 0
    for plan in plans:
        groups = plan["indices"]
        local_count = len(plan["rows"])
        real = [i for j, group in enumerate(groups) if j not in plan["dummy"] for i in group]
        if not groups or any(not group for group in groups) or sorted(real) != list(range(local_count)):
            raise ValueError("FLOPs plan must consume every local sample exactly once")
        if any(
            j not in range(len(groups)) or groups[j] not in [g for k, g in enumerate(groups) if k not in plan["dummy"]]
            for j in plan["dummy"]
        ):
            raise ValueError("FLOPs dummy must copy a local real microbatch")
        if list(rollout_data["total_lengths"][offset : offset + local_count]) != plan["lengths"]:
            raise ValueError("FLOPs plan lengths differ after concatenation")
        dummy.update(len(indices) + j for j in plan["dummy"])
        indices.extend([[i + offset for i in group] for group in groups])
        counts.append(len(groups))
        offset += local_count
    if not plans or offset != len(rollout_data["total_lengths"]):
        raise ValueError("FLOPs plans must cover the complete payload")
    if rollout_data.get("rollout_mini_local_sample_counts") != [len(p["rows"]) for p in plans] or rollout_data.get(
        "rollout_mini_global_sample_counts"
    ) != [p["global_count"] for p in plans]:
        raise ValueError("FLOPs plan optimizer step boundaries differ")
    return indices, counts, dummy
