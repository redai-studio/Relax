# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""FLOPs placement while retaining TransferQueue's group and replay
protocol."""

from dataclasses import dataclass
from typing import Any

from transfer_queue import GRPOGroupNSampler, SeqlenBalancedSampler

from relax.utils.data.seqlen_balancing import get_seqlen_balanced_partitions
from relax.utils.data.workload_balancing import PackingInfeasibleError, pack_by_workload
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
        return {
            index: {**metadata[index], "total_lengths": weight} for index, weight in zip(indexes, weights, strict=True)
        }


class FlopsBalancedSampler(SeqlenBalancedSampler):
    """Replace placement weights, preserving complete groups and replay caches.

    Equal-count placement retains the parent's group policy. For fixed-window
    SFT, unequal placement transforms the parent's exact selected identities
    before it publishes per-rank replay entries. Shared lengths stay intact.
    """

    def __init__(
        self,
        estimator: SequenceFlopsEstimator,
        n_samples_per_prompt: int = 1,
        dp_size: int = 1,
        *,
        unequal_size: bool = False,
        max_tokens_per_gpu: int | None = None,
        pad_multiplier: int = 128,
    ) -> None:
        super().__init__(n_samples_per_prompt=n_samples_per_prompt, dp_size=dp_size)
        self.estimator = estimator
        self.unequal_size = unequal_size
        self.max_tokens_per_gpu = max_tokens_per_gpu
        self.pad_multiplier = pad_multiplier
        if max_tokens_per_gpu is not None:
            if not unequal_size:
                raise ValueError("Capacity-aware DP placement requires variable row counts")
            if type(pad_multiplier) is not int or pad_multiplier <= 0:
                raise ValueError("pad_multiplier must be a positive integer")
            if type(max_tokens_per_gpu) is not int or max_tokens_per_gpu < pad_multiplier:
                raise ValueError("max_tokens_per_gpu must be an integer at least one padding block")
        if unequal_size:
            if n_samples_per_prompt != 1:
                raise ValueError("Variable-count FLOPs placement requires n_samples_per_prompt=1")

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
        if self.unequal_size and key in self._balanced_cache:
            groups = self._balanced_cache[key]
            indexes = [index for group in groups for index in group]
            if (
                len(groups) != self.dp_size
                or any(not group for group in groups)
                or len(indexes) != batch_size * self.dp_size
                or len(set(indexes)) != len(indexes)
            ):
                raise ValueError("Cached FLOPs assignments do not match the requested complete global batch")
        if key not in self._balanced_cache:
            partition = kwargs.get("partition")
            if partition is None:
                raise ValueError("FLOPs DP balancing requires partition metadata; equal splitting is not a fallback")
            if self.unequal_size:
                if "assignment_transform" in kwargs:
                    raise ValueError("FLOPs placement owns assignment_transform")
                indexes, _ = GRPOGroupNSampler.sample(
                    self, ready_indexes, batch_size * self.dp_size, task_name, partition_id
                )
                if not indexes:
                    return [], []
                if len(indexes) != batch_size * self.dp_size or len(set(indexes)) != len(indexes):
                    raise ValueError("Variable-count FLOPs placement requires a complete unique global batch")
                metadata = partition.get_custom_meta(indexes)
                missing = [index for index in indexes if "total_lengths" not in metadata.get(index, {})]
                if missing:
                    raise ValueError(f"FLOPs DP balancing requires total_lengths metadata, missing indexes: {missing}")
                lengths = [metadata[index]["total_lengths"] for index in indexes]
                workloads = self.estimator.estimate_per_sample(lengths)
                groups = get_seqlen_balanced_partitions(workloads, self.dp_size, equal_size=False)
                if self.max_tokens_per_gpu is not None:
                    capacity = self.max_tokens_per_gpu // self.pad_multiplier * self.pad_multiplier
                    if any(sum(lengths[i] for i in group) > capacity for group in groups):
                        try:
                            groups = pack_by_workload(lengths, workloads, capacity, num_partitions=self.dp_size)
                        except PackingInfeasibleError:
                            # Local microbatch packing handles rows that cannot fit one DP pack.
                            pass
                if any(not group for group in groups):
                    raise ValueError("Variable-count FLOPs placement requires a nonempty batch on every DP rank")
                partitions = [[indexes[i] for i in group] for group in groups]
                self._balanced_cache[key] = partitions
                states = self._states.setdefault(partition_id, {}).setdefault(task_name, {})
                for rank_i, samples in enumerate(partitions):
                    states.setdefault(rank_i, {})[key[2]] = (samples, samples.copy())
                sampled = partitions[rank]
                return sampled, sampled.copy()
            else:
                kwargs["partition"] = _WorkloadPartitionView(partition, self.estimator)
        return super().sample(ready_indexes, batch_size, task_name, partition_id, *args, **kwargs)

    def save_checkpoint(self) -> dict[str, Any]:
        state = super().save_checkpoint()
        state["flops_coefficients"] = [self.estimator.linear_flops, self.estimator.quadratic_flops]
        state["flops_geometry"] = [self.dp_size, self.n_samples_per_prompt]
        state["flops_assignment_policy"] = "unequal_v1" if self.unequal_size else "equal_v1"
        state["flops_capacity"] = (
            [self.max_tokens_per_gpu, self.pad_multiplier] if self.max_tokens_per_gpu is not None else None
        )
        return state

    def load_checkpoint(self, state: dict[str, Any]) -> None:
        expected_capacity = (
            [self.max_tokens_per_gpu, self.pad_multiplier] if self.max_tokens_per_gpu is not None else None
        )
        if state.get("flops_capacity") != expected_capacity:
            raise ValueError("Cannot restore FLOPs sampler assignments with a different capacity contract")
        expected_policy = "unequal_v1" if self.unequal_size else "equal_v1"
        if state.get("flops_assignment_policy", "equal_v1") != expected_policy:
            raise ValueError("Cannot restore FLOPs sampler assignments with a different assignment policy")
        expected = [self.estimator.linear_flops, self.estimator.quadratic_flops]
        if state.get("flops_coefficients") != expected:
            raise ValueError("Cannot restore FLOPs sampler assignments with different or missing model coefficients")
        if state.get("flops_geometry") != [self.dp_size, self.n_samples_per_prompt]:
            raise ValueError("Cannot restore FLOPs sampler assignments with different or missing DP/prompt geometry")
        super().load_checkpoint(state)
