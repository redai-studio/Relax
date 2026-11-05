# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""FLOPs placement while retaining TransferQueue's group and replay
protocol."""

from dataclasses import dataclass
from typing import Any

from transfer_queue import SeqlenBalancedSampler

from relax.utils.training.flops_counter import SequenceFlopsEstimator


@dataclass(frozen=True)
class _WorkloadPartitionView:
    partition: Any
    estimator: SequenceFlopsEstimator

    def get_custom_meta(self, indexes: list[int]) -> dict[int, dict[str, Any]]:
        metadata = self.partition.get_custom_meta(indexes)
        missing = [index for index in indexes if "total_lengths" not in metadata.get(index, {})]
        if missing:
            raise ValueError(f"FLOPs DP balancing requires total_lengths metadata, missing indexes: {missing}")
        weights = self.estimator.estimate_per_sample([metadata[index]["total_lengths"] for index in indexes])
        # The parent treats total_lengths as weights; leave shared metadata intact.
        return {
            index: {**metadata[index], "total_lengths": weight} for index, weight in zip(indexes, weights, strict=True)
        }


class FlopsBalancedSampler(SeqlenBalancedSampler):
    """Replace placement weights, preserving complete groups and replay caches.

    The parent chooses the exact global batch, balances whole prompt groups
    with equal group counts, and populates all DP replay/polling entries. Only
    its private metadata view changes; payloads and shared lengths stay intact.
    """

    def __init__(self, estimator: SequenceFlopsEstimator, n_samples_per_prompt: int = 1, dp_size: int = 1) -> None:
        super().__init__(n_samples_per_prompt=n_samples_per_prompt, dp_size=dp_size)
        self.estimator = estimator

    def sample(
        self,
        ready_indexes: list[int],
        batch_size: int,
        task_name: str = "",
        partition_id: str = "",
        *args: Any,
        **kwargs: Any,
    ) -> tuple[list[int], list[int]]:
        rank = kwargs.get("dp_rank", 0)
        if type(rank) is not int or not 0 <= rank < self.dp_size:
            raise ValueError(f"dp_rank must be in [0, {self.dp_size}), got {rank!r}")
        key = (partition_id, task_name, kwargs.get("batch_index", 0))
        if key not in self._balanced_cache:
            partition = kwargs.get("partition")
            if partition is None:
                raise ValueError("FLOPs DP balancing requires partition metadata; equal splitting is not a fallback")
            kwargs["partition"] = _WorkloadPartitionView(partition, self.estimator)
        return super().sample(ready_indexes, batch_size, task_name, partition_id, *args, **kwargs)

    def save_checkpoint(self) -> dict[str, Any]:
        state = super().save_checkpoint()
        state["flops_coefficients"] = [self.estimator.linear_flops, self.estimator.quadratic_flops]
        return state

    def load_checkpoint(self, state: dict[str, Any]) -> None:
        expected = [self.estimator.linear_flops, self.estimator.quadratic_flops]
        if state.get("flops_coefficients") != expected:
            raise ValueError("Cannot restore FLOPs sampler assignments with different or missing model coefficients")
        super().load_checkpoint(state)
