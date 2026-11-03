# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Cost-weighted DP placement without changing queue admission or replay."""

import inspect
import math
from argparse import Namespace
from numbers import Integral
from typing import Any

from transfer_queue.sampler import BaseSampler

from relax.utils.data.seqlen_balancing import get_seqlen_balanced_partitions


def get_dp_packing_geometry(args: Namespace, task_name: str) -> dict[str, Any] | None:
    """Describe a complete CP1/VPP1 training window without GPU queries."""
    if getattr(args, "dp_balance", "tokens") != "flops" or task_name not in (
        "train",
        "train_critic",
        "actor_train",
        "critic_train",
        "sft_train",
    ):
        return None
    if (
        not getattr(args, "use_dynamic_batch_size", False)
        or getattr(args, "sft_packing_strategy", "balanced") != "balanced"
        or getattr(args, "qkv_format", "thd") != "thd"
        or getattr(args, "context_parallel_size", 1) != 1
        or (getattr(args, "virtual_pipeline_model_parallel_size", None) or 1) != 1
        or getattr(args, "num_layers_per_virtual_pipeline_stage", None) is not None
        or getattr(args, "num_virtual_stages_per_pipeline_rank", None) not in (None, 1)
        or getattr(args, "dynamic_context_parallel", False)
    ):
        return None
    from relax.engine.sft.runtime import is_sft_mode, sft_tq_num_shards

    if is_sft_mode(args) and sft_tq_num_shards(args) != 1:
        return None
    if not is_sft_mode(args):
        if getattr(args, "num_steps_per_rollout", None) not in (None, 1):
            return None
        rollout_size = getattr(args, "rollout_batch_size", None)
        samples = getattr(args, "n_samples_per_prompt", None)
        global_batch_size = getattr(args, "global_batch_size", None)
        if (
            any(
                not isinstance(value, Integral) or isinstance(value, bool) or value <= 0
                for value in (rollout_size, samples, global_batch_size)
            )
            or rollout_size * samples != global_batch_size
        ):
            return None
    capacity = getattr(args, "max_tokens_per_gpu", None)
    if capacity is None:
        return None
    if getattr(args, "allgather_cp", False):
        padding = getattr(args, "tensor_model_parallel_size", 1) * args.data_pad_size_multiplier
        if padding <= 0 or capacity < padding:
            raise ValueError("DP packing requires padding to fit within max_tokens_per_gpu")
        capacity = capacity // padding * padding
    return {
        "version": 1,
        "capacity": capacity,
        "length_alignment": 1,
        "microbatch_balance": getattr(args, "microbatch_balance", "tokens"),
        "microbatch_group_size": 1,
    }


def checked_packing_geometry(
    geometries: dict[str, dict[str, Any] | None], task_name: str, geometry: dict[str, Any] | None
) -> dict[str, Any] | None:
    """Require identical scheduling geometry before returning cached rows."""
    if geometry is not None:
        fields = {"version", "capacity", "length_alignment", "microbatch_balance", "microbatch_group_size"}
        if not isinstance(geometry, dict) or set(geometry) != fields:
            raise ValueError("Invalid DP packing geometry fields")
        geometry = dict(geometry)
        for name in fields - {"microbatch_balance"}:
            value = geometry[name]
            if not isinstance(value, Integral) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"DP packing geometry requires positive integer {name}")
        if geometry["version"] != 1 or geometry["microbatch_balance"] not in ("tokens", "flops"):
            raise ValueError("Unsupported DP packing geometry version or microbatch objective")
    if task_name in geometries and geometries[task_name] != geometry:
        raise ValueError("DP packing geometry changed across requests or checkpoint restore")
    geometries[task_name] = geometry
    return geometry


def select_packing_aware_assignment(
    original: list[list[int]],
    weighted: list[list[int]],
    weights: dict[int, float],
    metadata: dict[int, dict[str, Any]],
    geometry: dict[str, Any] | None,
) -> list[list[int]]:
    """Compare existing placements by aligned global K, then rank cost."""
    if geometry is None:
        return weighted
    from relax.utils.data.data import get_minimum_num_micro_batch_size
    from relax.utils.data.packing import get_flops_packing_seed

    alignment = geometry["length_alignment"]
    capacity = geometry["capacity"]
    group_size = geometry["microbatch_group_size"]

    def objective(groups: list[list[int]]) -> tuple[int, float, float]:
        counts, loads = [], []
        for rows in groups:
            lengths = []
            for index in rows:
                value = metadata[index]["total_lengths"]
                if not isinstance(value, Integral) or isinstance(value, bool) or value <= 0:
                    raise ValueError("DP packing requires positive integer total_lengths")
                lengths.append((int(value) + alignment - 1) // alignment * alignment)
            costs = [weights[index] for index in rows]
            count = (
                len(get_flops_packing_seed(lengths, costs, capacity))
                if costs and geometry["microbatch_balance"] == "flops"
                else get_minimum_num_micro_batch_size(lengths, capacity)
            )
            counts.append(count)
            loads.append(sum(costs))
        global_k = max(1, (max(counts) + group_size - 1) // group_size * group_size)
        return global_k, max(loads), sum(load * load for load in loads)

    # min retains the original token assignment on an exact tie.
    return min((original, weighted), key=objective)


def validated_cost_weights(
    rows: list[int], metadata: dict[int, dict[str, Any]], cost_key: str = "flops_cost"
) -> dict[int, float]:
    weights = {}
    for index in rows:
        try:
            value = float(metadata[index][cost_key])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"Missing or invalid {cost_key} for sample {index}") from error
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{cost_key} must be positive and finite for sample {index}")
        weights[index] = value
    return weights


class FlopsBalancedSampler(BaseSampler):
    """Keep SFT batch selection and cache ownership in TransferQueue."""

    def __init__(self, n_samples_per_prompt: int = 1, dp_size: int = 1) -> None:
        from transfer_queue import SeqlenBalancedSampler

        if n_samples_per_prompt != 1:
            raise ValueError("FLOPs SFT DP assignment requires one sample per group")
        self.base = SeqlenBalancedSampler(n_samples_per_prompt=n_samples_per_prompt, dp_size=dp_size)
        self._packing_geometries: dict[str, dict[str, Any] | None] = {}
        if "assignment_transform" not in inspect.signature(self.base.sample).parameters:
            raise RuntimeError("FLOPs DP assignment requires the image-built TransferQueue assignment_transform patch")

    def has_cached_result(self, partition_id: str, task_name: str, sampling_config: Any = None) -> bool:
        return self.base.has_cached_result(partition_id, task_name, sampling_config)

    def sample(
        self, ready_indexes: list[int], batch_size: int, task_name: str = "", partition_id: str = "", **kwargs: Any
    ) -> tuple[list[int], list[int]]:
        partition = kwargs.get("partition")
        if partition is None:
            raise ValueError("FLOPs DP assignment requires queue metadata")
        geometry = checked_packing_geometry(self._packing_geometries, task_name, kwargs.get("packing_geometry"))

        def transform(original: list[list[int]]) -> list[list[int]]:
            rows = sorted(index for group in original for index in group)
            metadata = partition.get_custom_meta(rows)
            weights = validated_cost_weights(rows, metadata)
            groups = get_seqlen_balanced_partitions(
                [weights[index] for index in rows], self.base.dp_size, equal_size=True
            )
            weighted = [[rows[position] for position in group] for group in groups]
            return select_packing_aware_assignment(original, weighted, weights, metadata, geometry)

        return self.base.sample(
            ready_indexes, batch_size, task_name, partition_id, assignment_transform=transform, **kwargs
        )

    def clear_cache(self, partition_id: str) -> None:
        self.base.clear_cache(partition_id)

    def save_checkpoint(self) -> dict:
        return {
            "base": self.base.save_checkpoint(),
            "dp_size": self.base.dp_size,
            "cost_key": "flops_cost",
            "packing_geometries": self._packing_geometries.copy(),
        }

    def load_checkpoint(self, state: dict) -> None:
        if state.get("dp_size") != self.base.dp_size or state.get("cost_key") != "flops_cost":
            raise ValueError("FLOPs sampler checkpoint geometry or mode differs")
        self.base.load_checkpoint(state["base"])
        self._packing_geometries = state.get("packing_geometries", {}).copy()
