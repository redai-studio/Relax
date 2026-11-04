# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Slime PR #2017's pack-first, whole-microbatch DP scheduling.

Algorithm pinned to THUDM/slime 053bdfcebb9733373cc4fc70d6c1e011fa3ee216. The
capacity determines the initial bin count; upstream FLOPs KK does not enforce a
hard token bound on each resulting bin. No local repair or search is applied
here.
"""

import hashlib
import json
import math
from argparse import Namespace
from numbers import Integral
from typing import Any

from relax.utils.data.seqlen_balancing import get_seqlen_balanced_partitions


PLAN_METADATA_FIELD = "_flops_schedule_v1"
STEP_PLANS_FIELD = "_flops_step_plans"


def get_dp_packing_geometry(args: Namespace, task_name: str) -> dict[str, Any] | None:
    """Describe one complete training step without GPU queries."""
    if not getattr(args, "balance_by_flops", False) or task_name not in (
        "train",
        "train_critic",
        "actor_train",
        "critic_train",
        "sft_train",
    ):
        return None
    from relax.utils.training.microbatch_cost import validate_flops_balance

    validate_flops_balance(args)
    cp_size = getattr(args, "context_parallel_size", 1)
    capacity = args.max_tokens_per_gpu * cp_size
    if getattr(args, "allgather_cp", False):
        padding = getattr(args, "tensor_model_parallel_size", 1) * args.data_pad_size_multiplier
        if padding <= 0 or args.max_tokens_per_gpu < padding:
            raise ValueError("allgather CP padding must fit within max_tokens_per_gpu")
        capacity = cp_size * (args.max_tokens_per_gpu // padding) * padding
    return {
        "version": 4,
        "capacity": capacity,
        "microbatch_group_size": 1,
        "context_parallel_size": cp_size,
        "pipeline_stages": getattr(args, "pipeline_model_parallel_size", 1),
    }


def validate_packing_geometry(geometry: dict[str, Any] | None) -> dict[str, Any] | None:
    """Validate a queue request; replay geometry lives in the cached plan."""
    if geometry is not None:
        fields = {
            "version",
            "capacity",
            "context_parallel_size",
            "microbatch_group_size",
            "pipeline_stages",
        }
        if not isinstance(geometry, dict) or set(geometry) != fields:
            raise ValueError("Invalid DP packing geometry fields")
        geometry = dict(geometry)
        for name in fields:
            value = geometry[name]
            if not isinstance(value, Integral) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"DP packing geometry requires positive integer {name}")
        if geometry["version"] != 4:
            raise ValueError("Unsupported FLOPs packing geometry version or pipeline size")
    return geometry


def validated_cost_weights(rows: list[int], metadata: dict[int, dict[str, Any]]) -> dict[int, float]:
    weights = {}
    for index in rows:
        try:
            value = float(metadata[index]["flops_cost"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"Missing or invalid flops_cost for sample {index}") from error
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"flops_cost must be positive and finite for sample {index}")
        weights[index] = value
    return weights


def build_flops_schedule(
    lengths: list[int], costs: list[float], capacity: int, dp_size: int, microbatch_group_size: int = 1
) -> list[list[list[int]]]:
    """Plan one complete optimizer step; indices refer to the input order."""
    if (
        any(
            not isinstance(v, Integral) or isinstance(v, bool) or v <= 0
            for v in [capacity, dp_size, microbatch_group_size, *lengths]
        )
        or len(lengths) < dp_size
    ):
        raise ValueError("FLOPs scheduling needs positive lengths/geometry and at least one sample per DP")
    if len(costs) != len(lengths) or any(not math.isfinite(c) or c <= 0 for c in costs):
        raise ValueError("FLOPs scheduling needs one positive finite cost per sample")
    count = max(1, (sum(lengths) + capacity - 1) // capacity)
    bins = (
        [[i] for i in range(len(lengths))]
        if count >= len(lengths)
        else get_seqlen_balanced_partitions(costs, count, equal_size=False)
    )
    align = dp_size * microbatch_group_size
    target = max((len(bins) + align - 1) // align * align, align)
    # Exact upstream expand_bins_by_splitting, including tie breaking/order.
    while len(bins) < target:
        candidates = [(sum(lengths[i] for i in b), j) for j, b in enumerate(bins) if len(b) > 1]
        if not candidates:
            raise ValueError(f"Cannot split {len(lengths)} samples into {target} aligned microbatches")
        _, j = max(candidates)
        halves: list[list[int]] = [[], []]
        sums = [0, 0]
        for i in sorted(bins[j], key=lambda i: -lengths[i]):
            half = 0 if sums[0] <= sums[1] else 1
            halves[half].append(i)
            sums[half] += lengths[i]
        bins[j] = halves[0]
        bins.append(halves[1])
    weights = [sum(costs[i] for i in b) for b in bins]
    owners = get_seqlen_balanced_partitions(weights, dp_size, equal_size=True)
    return [[bins[j] for j in group] for group in owners]


def encode_schedule(
    rows: list[int],
    metadata: dict[int, dict[str, Any]],
    geometry: dict[str, Any],
    dp_size: int,
    global_count: int,
    identity: tuple,
) -> tuple[list[list[int]], dict[int, dict[str, Any]]]:
    lengths = [metadata[i]["total_lengths"] for i in rows]
    weights = validated_cost_weights(rows, metadata)
    costs = [weights[i] for i in rows]
    plan = build_flops_schedule(lengths, costs, geometry["capacity"], dp_size, geometry["microbatch_group_size"])
    packs = [[[rows[i] for i in group] for group in rank] for rank in plan]
    digest = hashlib.sha256(json.dumps([identity, geometry, rows, lengths, costs, packs]).encode()).hexdigest()
    assignments, updates = [], {}
    for rank, groups in enumerate(packs):
        ids = [i for group in groups for i in group]
        assignments.append(ids)
        for i in ids:
            updates[i] = {PLAN_METADATA_FIELD: {"sha256": digest}}
        updates[ids[0]][PLAN_METADATA_FIELD] = dict(
            version=1,
            sha256=digest,
            identity=list(identity),
            geometry=geometry,
            dp_rank=rank,
            dp_size=dp_size,
            global_count=global_count,
            rows=ids,
            lengths=[metadata[i]["total_lengths"] for i in ids],
            packs=groups,
            K=len(groups),
        )
    return assignments, updates


def apply_schedule_metadata(
    partition: Any, updates: dict[int, dict[str, Any]] | None, geometry: dict[str, Any] | None
) -> None:
    """Check cached geometry and merge with current producer metadata."""
    if updates is None:
        if geometry is not None:
            raise RuntimeError("FLOPs replay is missing its checkpoint plan")
        return
    header = next(row[PLAN_METADATA_FIELD] for row in updates.values() if "packs" in row[PLAN_METADATA_FIELD])
    if header["geometry"] != geometry:
        raise ValueError("DP packing geometry changed across requests or checkpoint restore")
    current = partition.get_custom_meta(list(updates))
    partition.set_custom_meta({i: {**current[i], **update} for i, update in updates.items()})


def restore_schedule(batch_meta: Any, rollout_data: dict[str, Any]) -> None:
    """Restore exact local pack membership after payload/metadata broadcast."""
    entries = [m.get(PLAN_METADATA_FIELD) for m in batch_meta.get_all_custom_meta()]
    if not any(entries):
        return
    headers = [m for m in entries if m and "packs" in m]
    if not all(entries) or len(headers) != 1:
        raise ValueError("Partial FLOPs schedule or missing/duplicate local header")
    h = headers[0]
    if h["version"] != 1 or any(m["sha256"] != h["sha256"] for m in entries):
        raise ValueError("Mixed FLOPs schedules or unsupported version")
    ids = list(batch_meta.global_indexes)
    if ids != h["rows"] or len(ids) != len(set(ids)) or list(rollout_data["total_lengths"]) != h["lengths"]:
        raise ValueError("FLOPs schedule row order/lengths differ from payload")
    if "tokens" in rollout_data and [len(t) for t in rollout_data["tokens"]] != h["lengths"]:
        raise ValueError("FLOPs schedule token payload lengths differ")
    position = {i: j for j, i in enumerate(ids)}
    indices = [[position[i] for i in group] for group in h["packs"]]
    if len(indices) != h["K"] or not indices or any(not group for group in indices):
        raise ValueError("Invalid FLOPs microbatch count")
    if sorted(i for group in indices for i in group) != list(range(len(ids))):
        raise ValueError("FLOPs schedule must consume every local sample exactly once")
    rollout_data[STEP_PLANS_FIELD] = [{**h, "indices": indices}]
    rollout_data["rollout_mini_local_sample_counts"] = [len(ids)]
    rollout_data["rollout_mini_global_sample_counts"] = [h["global_count"]]


def local_schedule(rollout_data: dict[str, Any]) -> tuple[list[list[int]], list[int]]:
    """Flatten per-step plans while retaining explicit optimizer boundaries."""
    plans = rollout_data[STEP_PLANS_FIELD]
    indices, counts, offset = [], [], 0
    for plan in plans:
        local_count = len(plan["rows"])
        if sorted(i for group in plan["indices"] for i in group) != list(range(local_count)):
            raise ValueError("FLOPs plan local sample coverage differs")
        if len(plan["indices"]) != plan["K"]:
            raise ValueError("FLOPs plan microbatch count differs")
        if list(rollout_data["total_lengths"][offset : offset + local_count]) != plan["lengths"]:
            raise ValueError("FLOPs plan lengths differ after concatenation")
        indices.extend([[i + offset for i in group] for group in plan["indices"]])
        counts.append(plan["K"])
        offset += local_count
    if offset != len(rollout_data["total_lengths"]):
        raise ValueError("FLOPs plans must cover the complete payload")
    if rollout_data.get("rollout_mini_global_sample_counts") != [p["global_count"] for p in plans]:
        raise ValueError("FLOPs plan logical step counts differ")
    return indices, counts
