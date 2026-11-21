# Copyright 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright 2025 The TransferQueue Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import heapq
import logging
import os
from typing import Any

from transfer_queue.sampler.grpo_group_n_sampler import GRPOGroupNSampler

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("TQ_LOGGING_LEVEL", logging.WARNING))


class SeqlenBalancedSampler(GRPOGroupNSampler):
    """Sequence-length balanced sampler that extends GRPOGroupNSampler.

    This sampler first uses the GRPO group-N logic to select complete prompt
    groups (ensuring group integrity), then redistributes the selected
    samples across DP ranks using Karmarkar-Karp balanced partitioning so
    that each rank receives approximately the same total token count.

    Each DP rank independently calls ``sample()`` with its own ``dp_rank``.
    On the **first** call for a given ``(partition_id, task_name, batch_index)``,
    the sampler:

    1. Delegates to ``GRPOGroupNSampler.sample()`` with the full
       ``global_batch_size`` to select complete prompt groups.
    2. Looks up per-sample ``total_lengths`` from the partition's
       ``custom_meta`` (populated during data insertion).
    3. Runs the Karmarkar-Karp algorithm (``get_seqlen_balanced_partitions``)
       to partition samples across ``dp_size`` ranks.
    4. Caches the per-DP assignments.

    Subsequent calls for the same key return the cached assignment for the
    requested ``dp_rank``.

    Requires:
    - ``custom_meta`` for each sample must contain ``{"total_lengths": <int>}``.
    - The controller must pass ``partition=<DataPartitionStatus>`` in kwargs
      when calling the sampler.
    - ``batch_size`` passed in is the **per-DP** batch size; the sampler
      internally multiplies by ``dp_size`` to get the global batch size for
      the initial GRPO selection.
    """

    def __init__(self, n_samples_per_prompt: int = 1, dp_size: int = 1):
        super().__init__(n_samples_per_prompt=n_samples_per_prompt)
        if dp_size <= 0:
            raise ValueError(f"dp_size must be positive, got {dp_size}")
        self.dp_size = dp_size
        # Cache: (partition_id, task_name, batch_index) -> list[list[int]]
        self._balanced_cache: dict[tuple, list[list[int]]] = {}

    def sample(
        self,
        ready_indexes: list[int],
        batch_size: int,
        task_name: str = "",
        partition_id: str = "",
        *args: Any,
        **kwargs: Any,
    ) -> tuple[list[int], list[int]]:
        """Sample indices for a specific DP rank with seqlen balancing.

        Args:
            ready_indexes: List of ready global indices.
            batch_size: **Per-DP** batch size requested by this rank.
            task_name: Task identifier.
            partition_id: Partition identifier.
            **kwargs: Must include ``dp_rank``, ``batch_index``, and
                ``partition`` (the ``DataPartitionStatus`` object from the
                controller).

        Returns:
            Tuple of (sampled_indexes, consumed_indexes).
        """
        dp_rank = kwargs.get("dp_rank", 0)
        batch_index = kwargs.get("batch_index", 0)
        partition = kwargs.get("partition", None)

        cache_key = (partition_id, task_name, batch_index)

        if cache_key in self._balanced_cache:
            # Return cached assignment for this dp_rank
            partitions = self._balanced_cache[cache_key]
            if dp_rank < len(partitions):
                sampled = partitions[dp_rank]
            else:
                sampled = []
            return sampled, sampled.copy()

        # --- First call: do global sampling + balancing ---

        # Step 1: Use GRPO logic to select complete groups for the full
        # global batch (batch_size * dp_size).
        global_batch_size = batch_size * self.dp_size
        grpo_sampled, grpo_consumed = super().sample(
            ready_indexes,
            global_batch_size,
            task_name=task_name,
            partition_id=partition_id,
        )

        if not grpo_sampled:
            return [], []

        # Step 2: Get total_lengths from custom_meta
        if partition is None:
            logger.warning(
                "SeqlenBalancedSampler: no partition object provided, falling back to equal-split without balancing."
            )
            # Fallback: equal split
            chunk_size = len(grpo_sampled) // self.dp_size
            partitions = []
            for i in range(self.dp_size):
                start = i * chunk_size
                end = start + chunk_size if i < self.dp_size - 1 else len(grpo_sampled)
                partitions.append(grpo_sampled[start:end])
        else:
            custom_meta = partition.get_custom_meta(grpo_sampled)
            total_lengths = []
            for idx in grpo_sampled:
                meta = custom_meta.get(idx, {})
                tl = meta.get("total_lengths", 1)
                total_lengths.append(tl)

            # Step 3: Karmarkar-Karp balanced partitioning at the GROUP
            # level.  Each prompt group consists of ``n_samples_per_prompt``
            # consecutive samples.  We aggregate their total_lengths into a
            # single group weight so that the KK algorithm keeps groups
            # intact, preserving the invariant that every DP rank receives
            # complete groups (required by pass@k metrics and GRPO
            # advantage normalisation).
            group_size = self.n_samples_per_prompt
            num_groups = len(total_lengths) // group_size
            remainder = len(total_lengths) % group_size

            if num_groups > 0 and remainder == 0:
                # Aggregate per-group total token counts
                group_lengths = [sum(total_lengths[g * group_size : (g + 1) * group_size]) for g in range(num_groups)]
                # Balance groups across DP ranks
                balanced_group_partitions = get_seqlen_balanced_partitions(group_lengths, self.dp_size, equal_size=True)
                # Expand group indices back to sample indices
                partitions = []
                for group_indices in balanced_group_partitions:
                    sample_indices = []
                    for g in group_indices:
                        for s in range(group_size):
                            sample_indices.append(grpo_sampled[g * group_size + s])
                    partitions.append(sample_indices)
            else:
                # Fallback: no valid grouping — balance at sample level
                balanced_partitions = get_seqlen_balanced_partitions(total_lengths, self.dp_size, equal_size=False)
                partitions = [[grpo_sampled[i] for i in part_indices] for part_indices in balanced_partitions]

        # Cache the result
        self._balanced_cache[cache_key] = partitions

        # Populate the inherited _states cache for ALL dp_ranks so that
        # the controller's polling check (which looks at self.sampler._states)
        # works correctly even when ready_indexes < batch_size for later ranks
        # (because earlier ranks already consumed their portion).
        if partition_id not in self._states:
            self._states[partition_id] = {}
        if task_name not in self._states[partition_id]:
            self._states[partition_id][task_name] = {}
        states = self._states[partition_id][task_name]
        for rank_i in range(self.dp_size):
            if rank_i not in states:
                states[rank_i] = {}
            rank_sampled = partitions[rank_i] if rank_i < len(partitions) else []
            states[rank_i][batch_index] = (rank_sampled, rank_sampled.copy())

        # Return this dp_rank's portion
        sampled = partitions[dp_rank] if dp_rank < len(partitions) else []
        # All samples are consumed (without replacement)
        return sampled, sampled.copy()

    def clear_cache(self, partition_id: str):
        """Clear cached states for the given partition."""
        super().clear_cache(partition_id)
        keys_to_remove = [k for k in self._balanced_cache if k[0] == partition_id]
        for k in keys_to_remove:
            del self._balanced_cache[k]

    def save_checkpoint(self) -> dict:
        """Return sampler state including the balanced cache."""
        state = super().save_checkpoint()
        state["_balanced_cache"] = self._balanced_cache
        return state

    def load_checkpoint(self, state: dict) -> None:
        """Restore sampler state including the balanced cache."""
        super().load_checkpoint(state)
        self._balanced_cache = state["_balanced_cache"]


# Copied from https://github.com/volcengine/verl/blob/468adf22c43b744348051fccd7a5d830c6c3c36a/verl/utils/seqlen_balancing.py
def karmarkar_karp(seqlen_list: list[int], k_partitions: int, equal_size: bool):
    """Partition items into k groups with balanced sums using the Karmarkar-Karp largest differencing method.

    See: https://en.wikipedia.org/wiki/Largest_differencing_method

    Args:
        seqlen_list: List of sequence lengths (or weights) to partition.
        k_partitions: Number of partitions to create.
        equal_size: If True, enforce that all partitions have exactly the same number of items
            (requires ``len(seqlen_list) % k_partitions == 0``).

    Returns:
        A list of k partitions, where each partition is a list of original indices.
    """

    class Set:
        """A weighted set that tracks items and their cumulative sum for partitioning."""

        def __init__(self) -> None:
            self.sum = 0
            self.items: list[tuple[int, int]] = []

        def add(self, idx: int, val: int):
            self.items.append((idx, val))
            self.sum += val

        def merge(self, other):
            for idx, val in other.items:
                self.items.append((idx, val))
                self.sum += val

        def __lt__(self, other):
            if self.sum != other.sum:
                return self.sum < other.sum
            if len(self.items) != len(other.items):
                return len(self.items) < len(other.items)
            return self.items < other.items

    class State:
        """A k-way partition state used in the Karmarkar-Karp heap-based merge process."""

        def __init__(self, items: list[tuple[int, int]], k: int) -> None:
            self.k = k
            # sets should always be decreasing order
            self.sets = [Set() for _ in range(k)]
            assert len(items) in [1, k], f"{len(items)} not in [1, {k}]"
            for i, (idx, seqlen) in enumerate(items):
                self.sets[i].add(idx=idx, val=seqlen)
            self.sets = sorted(self.sets, reverse=True)

        def get_partitions(self):
            partitions = []
            for i in range(len(self.sets)):
                cur_partition = []
                for idx, _ in self.sets[i].items:
                    cur_partition.append(idx)
                partitions.append(cur_partition)
            return partitions

        def merge(self, other):
            for i in range(self.k):
                self.sets[i].merge(other.sets[self.k - 1 - i])
            self.sets = sorted(self.sets, reverse=True)

        @property
        def spread(self) -> int:
            return self.sets[0].sum - self.sets[-1].sum

        def __lt__(self, other):
            # least heap, let the state with largest spread to be popped first,
            # if the spread is the same, let the state who has the largest set
            # to be popped first.
            if self.spread != other.spread:
                return self.spread > other.spread
            return self.sets[0] > other.sets[0]

        def __repr__(self) -> str:
            repr_str = "["
            for i in range(self.k):
                if i > 0:
                    repr_str += ","
                repr_str += "{"
                for j, (_, seqlen) in enumerate(self.sets[i].items):
                    if j > 0:
                        repr_str += ","
                    repr_str += str(seqlen)
                repr_str += "}"
            repr_str += "]"
            return repr_str

    sorted_seqlen_list = sorted([(seqlen, i) for i, seqlen in enumerate(seqlen_list)])
    states_pq: list[State] = []
    if equal_size:
        assert len(seqlen_list) % k_partitions == 0, f"{len(seqlen_list)} % {k_partitions} != 0"
        for offset in range(0, len(sorted_seqlen_list), k_partitions):
            items = []
            for i in range(k_partitions):
                seqlen, idx = sorted_seqlen_list[offset + i]
                items.append((idx, seqlen))
            heapq.heappush(states_pq, State(items=items, k=k_partitions))
    else:
        for seqlen, idx in sorted_seqlen_list:
            heapq.heappush(states_pq, State(items=[(idx, seqlen)], k=k_partitions))

    while len(states_pq) > 1:
        state0 = heapq.heappop(states_pq)
        state1 = heapq.heappop(states_pq)
        # merge states
        state0.merge(state1)
        heapq.heappush(states_pq, state0)

    final_state = states_pq[0]
    partitions = final_state.get_partitions()
    if equal_size:
        for _i, partition in enumerate(partitions):
            assert len(partition) * k_partitions == len(seqlen_list), (
                f"{len(partition)} * {k_partitions} != {len(seqlen_list)}"
            )
    return partitions


def get_seqlen_balanced_partitions(seqlen_list: list[int], k_partitions: int, equal_size: bool):
    """get order of seq lengths to make partitions balanced, this is
        used in balancing sum of seqlength across dp ranks and microbatches
    Parameters:
        seqlen_list (List[int]):
            seq lengths of each items
        k_partitions (int):
            resulting number of partitions
        equal_size (bool):
            if True, number of items in each partitions must be equal.
            if False, only consider balancing the sum, each partition can have
            variable number of items
    Returns:
        partitions (List[List[int]]):
            return k_partitions list containing the index of items.
    """
    assert len(seqlen_list) >= k_partitions, f"number of items:[{len(seqlen_list)}] < k_partitions:[{k_partitions}]"

    def _check_and_sort_partitions(partitions):
        assert len(partitions) == k_partitions, f"{len(partitions)} != {k_partitions}"
        seen_idx = set()
        sorted_partitions: list[list[int]] = [[] for _ in range(k_partitions)]
        for i, partition in enumerate(partitions):
            assert len(partition) > 0, f"the {i}-th partition is empty"
            for idx in partition:
                seen_idx.add(idx)
            sorted_partitions[i] = sorted(partition)
        assert seen_idx == set(range(len(seqlen_list)))
        return sorted_partitions

    partitions = karmarkar_karp(seqlen_list=seqlen_list, k_partitions=k_partitions, equal_size=equal_size)
    return _check_and_sort_partitions(partitions)
