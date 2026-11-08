# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""FLOPs assignment over TransferQueue's existing selection/cache protocol."""

import inspect
from time import perf_counter
from typing import Any

from transfer_queue.sampler import BaseSampler

from relax.utils.data.flops_balancing import guarded_dp_assignment
from relax.utils.logging_utils import get_logger
from relax.utils.training.flops_counter import SampleFlopsCost


logger = get_logger(__name__)


class FlopsBalancedSampler(BaseSampler):
    """Transform selected SFT identities before the base sampler publishes
    them."""

    def __init__(
        self,
        base: BaseSampler,
        cost_model: SampleFlopsCost,
        token_capacity: int,
        *,
        packing_strategy: str = "balanced",
        capacity_alignment: int = 1,
    ) -> None:
        if "assignment_transform" not in inspect.signature(base.sample).parameters:
            raise RuntimeError(
                "--balance-by-flops requires the Docker-pinned TransferQueue assignment_transform API; "
                "use the official Relax image."
            )
        if base.n_samples_per_prompt != 1:
            raise ValueError("FLOPs SFT assignment requires n_samples_per_prompt=1")
        self.token_capacity = token_capacity
        self.packing_strategy = packing_strategy
        self.capacity_alignment = capacity_alignment
        self.base = base
        self.cost_model = cost_model
        self.last_assignment_metrics: dict[str, Any] = {}

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

        def transform(original: list[list[int]]) -> list[list[int]]:
            started = perf_counter()
            partition = kwargs.get("partition")
            if partition is None:
                raise ValueError("FLOPs assignment requires partition metadata with total_lengths")
            identities = [identity for group in original for identity in group]
            metadata = partition.get_custom_meta(identities)
            try:
                lengths = [metadata[identity]["total_lengths"] for identity in identities]
            except KeyError as error:
                raise ValueError("FLOPs assignment requires total_lengths for every selected identity") from error
            costs = self.cost_model.estimate(lengths)
            index = {identity: position for position, identity in enumerate(identities)}
            before = [[index[identity] for identity in group] for group in original]
            proposed, self.last_assignment_metrics = guarded_dp_assignment(
                lengths,
                costs,
                before,
                self.token_capacity,
                packing_strategy=self.packing_strategy,
                capacity_alignment=self.capacity_alignment,
            )
            self.last_assignment_metrics["scheduler_seconds"] = perf_counter() - started
            logger.debug("FLOPs DP assignment: %s", self.last_assignment_metrics)
            return [[identities[position] for position in group] for group in proposed]

        return self.base.sample(
            ready_indexes,
            batch_size,
            task_name,
            partition_id,
            *args,
            assignment_transform=transform,
            **kwargs,
        )

    def has_cached_result(
        self, partition_id: str, task_name: str, sampling_config: dict[str, Any] | None = None
    ) -> bool:
        return self.base.has_cached_result(partition_id, task_name, sampling_config)

    def clear_cache(self, partition_id: str) -> None:
        self.base.clear_cache(partition_id)

    def save_checkpoint(self) -> dict:
        return self.base.save_checkpoint()

    def load_checkpoint(self, state: dict) -> None:
        self.base.load_checkpoint(state)
