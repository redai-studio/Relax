# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""SFT FLOPs placement using TransferQueue's existing assignment
transaction."""

import inspect
from typing import Any

from transfer_queue import SeqlenBalancedSampler
from transfer_queue.sampler import BaseSampler

from relax.utils.data.flops_balancing import FlopsWorkload, log_flops_balance
from relax.utils.data.seqlen_balancing import get_seqlen_balanced_partitions


class FlopsBalancedSampler(BaseSampler):
    def __init__(self, dp_size: int, workload: FlopsWorkload) -> None:
        self.base = SeqlenBalancedSampler(n_samples_per_prompt=1, dp_size=dp_size)
        if "assignment_transform" not in inspect.signature(self.base.sample).parameters:
            raise RuntimeError(
                "--balance-by-flops for SFT requires the existing Docker TransferQueue assignment_transform patch"
            )
        self.workload = workload

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
        if "eval" in task_name or "eval" in partition_id:
            return self.base.sample(ready_indexes, batch_size, task_name, partition_id, *args, **kwargs)
        partition = kwargs.get("partition")
        if partition is None:
            raise ValueError("FLOPs placement requires TransferQueue partition metadata")

        def place(original: list[list[int]]) -> list[list[int]]:
            rows = sorted(index for group in original for index in group)
            metadata = partition.get_custom_meta(rows)
            workloads = self.workload([metadata[index]["total_lengths"] for index in rows])
            assignments = get_seqlen_balanced_partitions(workloads, self.base.dp_size, equal_size=True)
            log_flops_balance(workloads, assignments)
            return [[rows[index] for index in group] for group in assignments]

        return self.base.sample(
            ready_indexes,
            batch_size,
            task_name,
            partition_id,
            *args,
            assignment_transform=place,
            **kwargs,
        )

    def clear_cache(self, partition_id: str) -> None:
        self.base.clear_cache(partition_id)

    def save_checkpoint(self) -> dict:
        return self.base.save_checkpoint()

    def load_checkpoint(self, state: dict) -> None:
        self.base.load_checkpoint(state)
