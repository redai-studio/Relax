# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Global SFT microbatch planning within TransferQueue's assignment
transaction."""

import inspect
from typing import Any

from transfer_queue import SeqlenBalancedSampler
from transfer_queue.sampler import BaseSampler

from relax.utils.data.flops_balancing import FlopsWorkload, apply_schedule_metadata, encode_schedule


class FlopsBalancedSampler(BaseSampler):
    def __init__(self, dp_size: int, workload: FlopsWorkload) -> None:
        self.base = SeqlenBalancedSampler(n_samples_per_prompt=1, dp_size=dp_size)
        if "assignment_transform" not in inspect.signature(self.base.sample).parameters:
            raise RuntimeError(
                "--balance-by-flops for SFT requires the existing Docker TransferQueue assignment_transform patch"
            )
        self.workload = workload
        self.plans: dict[tuple[str, str, int], dict[int, dict[str, Any]]] = {}

    def has_cached_result(self, partition_id: str, task_name: str, sampling_config: Any = None) -> bool:
        return self.base.has_cached_result(partition_id, task_name, sampling_config)

    def sample(
        self,
        ready_indexes: list[int],
        batch_size: int,
        task_name: str = "",
        partition_id: str = "",
        *args: Any,
        **kwargs: Any,
    ) -> tuple[list[int], list[int]]:
        geometry = kwargs.get("dp_packing_geometry")
        if geometry is None:
            return self.base.sample(ready_indexes, batch_size, task_name, partition_id, *args, **kwargs)
        partition = kwargs.get("partition")
        if partition is None:
            raise ValueError("FLOPs placement requires TransferQueue partition metadata")

        key = (partition_id, task_name, kwargs["batch_index"])

        def place(original: list[list[int]]) -> list[list[int]]:
            rows = sorted(index for group in original for index in group)
            assignments, updates = encode_schedule(
                rows, partition.get_custom_meta(rows), self.workload, geometry, self.base.dp_size, len(rows), key
            )
            self.plans[key] = updates
            return assignments

        result = self.base.sample(
            ready_indexes,
            batch_size,
            task_name,
            partition_id,
            *args,
            assignment_transform=place,
            **kwargs,
        )
        if result[0]:
            apply_schedule_metadata(partition, self.plans.get(key), geometry)
        return result

    def clear_cache(self, partition_id: str) -> None:
        self.base.clear_cache(partition_id)
        for key in [key for key in self.plans if key[0] == partition_id]:
            del self.plans[key]

    def save_checkpoint(self) -> dict:
        return {**self.base.save_checkpoint(), "flops_plans": self.plans}

    def load_checkpoint(self, state: dict) -> None:
        self.base.load_checkpoint(state)
        self.plans = state.get("flops_plans", {})
