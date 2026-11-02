# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Cost-weighted DP placement without changing queue admission or replay."""

import inspect
import math
from typing import Any

from transfer_queue.sampler import BaseSampler

from relax.utils.data.seqlen_balancing import get_seqlen_balanced_partitions


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

        def transform(original: list[list[int]]) -> list[list[int]]:
            rows = sorted(index for group in original for index in group)
            weights = validated_cost_weights(rows, partition.get_custom_meta(rows))
            groups = get_seqlen_balanced_partitions(
                [weights[index] for index in rows], self.base.dp_size, equal_size=True
            )
            return [[rows[position] for position in group] for group in groups]

        return self.base.sample(
            ready_indexes, batch_size, task_name, partition_id, assignment_transform=transform, **kwargs
        )

    def clear_cache(self, partition_id: str) -> None:
        self.base.clear_cache(partition_id)

    def save_checkpoint(self) -> dict:
        return {"base": self.base.save_checkpoint(), "dp_size": self.base.dp_size, "cost_key": "flops_cost"}

    def load_checkpoint(self, state: dict) -> None:
        if state.get("dp_size") != self.base.dp_size or state.get("cost_key") != "flops_cost":
            raise ValueError("FLOPs sampler checkpoint geometry or mode differs")
        self.base.load_checkpoint(state["base"])
