# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Previous/current FIFO partition state for complete scored Groups.

The resident Pipeline owns TQ task lifetimes; this module owns partition and
batch state.
"""

from __future__ import annotations

import time
from argparse import Namespace
from dataclasses import dataclass, field
from typing import Any, List, Optional, Tuple, cast

from relax.agentic.pipeline import GroupExport
from relax.agentic.profile import mark_sample_agentic_event, mark_sample_agentic_event_once
from relax.utils.types import Sample


TransferBatch = Tuple[int, Tuple[GroupExport, ...], bool]


async def _transfer_batch_to_data_system(
    *,
    args: Namespace,
    batch_samples: list[list[Sample]],
    rollout_id: int,
    data_system_client: Any,
    is_last: bool = False,
) -> None:
    """Convert complete Samples and cross the TQ boundary."""

    from relax.utils.utils import build_rollout_custom_meta, convert_samples_to_train_data

    samples = [sample for group in batch_samples for sample in group]
    for sample in samples:
        mark_sample_agentic_event(sample, "transfer_release_start_at")
    try:
        batch_samples.sort(key=lambda group: group[0].index)
        samples = [sample for group in batch_samples for sample in group]
        rollout_batch = convert_samples_to_train_data(args, samples)
        custom_meta = build_rollout_custom_meta(rollout_batch)
        await data_system_client.async_put(
            data=rollout_batch,
            partition_id=f"train_{rollout_id}",
            custom_meta=custom_meta,
            is_last=is_last,
        )
    finally:
        for sample in samples:
            mark_sample_agentic_event(sample, "transfer_release_end_at")


@dataclass
class _Partition:
    """Own one physical partition and its accepted Group count."""

    partition_id: int
    target_groups: int
    sealed: bool
    emit_eos: bool
    accepted_groups: int = 0
    pending_batch: List[GroupExport] = field(default_factory=list)

    @property
    def remaining_groups(self) -> int:
        """Derive partition debt from accepted write refs."""

        return self.target_groups - self.accepted_groups

    @property
    def ready_to_release(self) -> bool:
        """Report whether the sealed partition has no remaining or pending
        Groups."""

        return self.sealed and self.remaining_groups == 0 and not self.pending_batch


class TransferDomain:
    """Store complete Groups under previous-before-current FIFO partitioning.

    Accepted Group truth lives in each partition's count; partition debt is
    derived from that count, and the previous partition always has priority.

    TQ calls and their task lifecycle belong to ``AgenticResidentPipeline``.
    """

    def __init__(self, *, args: Namespace) -> None:
        if args.use_dynamic_global_batch_size and (
            not args.partial_rollout or args.fully_async or args.over_sampling_batch_size <= args.rollout_batch_size
        ):
            raise ValueError(
                "Agentic dynamic global batch size requires partial rollout and over-sampling "
                "(over_sampling_batch_size > rollout_batch_size), and is incompatible with fully-async mode."
            )
        self._transfer_batch_group_count = (
            args.global_batch_size // args.num_iters_per_train_update // args.n_samples_per_prompt
            if args.fully_async
            else args.rollout_batch_size
        )
        self._fully_async = args.fully_async
        self._previous_partition: Optional[_Partition] = None
        self._current_partition: Optional[_Partition] = None

    @property
    def total_debt(self) -> int:
        """Derive total quota debt from the active partition refs."""

        return sum(
            partition.remaining_groups
            for partition in (self._previous_partition, self._current_partition)
            if partition is not None
        )

    def open_partition(self, partition_id: int, target_groups: int, *, accepts_surplus: bool = False) -> None:
        """Open one physical partition with its target Group count."""

        previous = self._previous_partition
        current = self._current_partition
        if (previous is not None and previous.partition_id == partition_id) or (
            current is not None and current.partition_id == partition_id
        ):
            raise ValueError(f"physical partition already active: {partition_id}")
        if self._previous_partition is not None:
            raise RuntimeError("previous partition must be produced before opening the next partition")
        self._previous_partition = self._current_partition
        self._current_partition = _Partition(
            partition_id,
            target_groups,
            sealed=not accepts_surplus,
            emit_eos=self._fully_async and not accepts_surplus,
        )

    def accept(self, group: GroupExport) -> None:
        """Accept one Group into its physical FIFO partition."""

        previous = self._previous_partition
        current = self._current_partition
        if previous is not None and previous.remaining_groups > 0:
            partition = previous
        elif current is not None and current.remaining_groups > 0:
            partition = current
        else:
            partition = None
        if partition is None:
            raise ValueError("no physical partition needs a group")

        buffered_at = time.time()
        for sample in group.samples:
            mark_sample_agentic_event_once(sample, "transfer_buffer_enter_at", buffered_at)
        partition.pending_batch.append(group)
        partition.accepted_groups += 1

    def detach_ready_batches(self) -> Tuple[TransferBatch, ...]:
        """Detach each partition's ready snapshot after it reaches the
        waterline."""

        batches = []
        for partition in (self._previous_partition, self._current_partition):
            if partition is None or not partition.pending_batch:
                continue
            if len(partition.pending_batch) >= self._transfer_batch_group_count or (
                partition.sealed and partition.remaining_groups == 0
            ):
                batches.append(self._take_batch(partition))
        return tuple(batches)

    def finish_current_partition(
        self,
        surplus: Tuple[GroupExport, ...],
    ) -> Tuple[TransferBatch, ...]:
        """Append DP-aligned completed refs, then finalize a dynamic
        partition."""

        partition = cast(_Partition, self._current_partition)
        partition.target_groups += len(surplus)
        for group in surplus:
            self.accept(group)
        partition.sealed = True
        return self.detach_ready_batches()

    def _take_batch(self, partition: _Partition) -> TransferBatch:
        """Detach one complete batch from resident partition state."""

        batch = tuple(partition.pending_batch)
        partition.pending_batch.clear()
        for group in batch:
            for sample in group.samples:
                mark_sample_agentic_event(sample, "transfer_enqueue_at")
        is_last = partition.emit_eos and partition.sealed and partition.remaining_groups == 0
        return partition.partition_id, batch, is_last

    def flush_batches(self) -> Tuple[TransferBatch, ...]:
        """Detach incomplete batches at a step or partition boundary."""

        batches = []
        for partition in (self._previous_partition, self._current_partition):
            if partition is None:
                continue
            if partition.pending_batch:
                batches.append(self._take_batch(partition))
        return tuple(batches)

    def release_finished_partitions(self) -> None:
        """Release physical partitions after the Pipeline gathers TQ tasks."""

        previous = self._previous_partition
        current = self._current_partition
        if previous is not None and previous.ready_to_release:
            self._previous_partition = None
        if self._previous_partition is None and current is not None and current.ready_to_release:
            self._current_partition = None

    def clear(self) -> None:
        """Release resident partition refs during terminal shutdown."""

        self._previous_partition = None
        self._current_partition = None
