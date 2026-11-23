# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Previous/current FIFO partition state for complete scored Groups.

The resident Pipeline owns TQ task lifetimes; this module owns partition and
batch state.
"""

from __future__ import annotations

import ctypes
import gc
import time
from argparse import Namespace
from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional, Tuple, cast

import torch

from relax.agentic.pipeline import GroupExport
from relax.agentic.profile import mark_sample_agentic_event, mark_sample_agentic_event_once
from relax.utils.env import Envs
from relax.utils.logging_utils import get_logger
from relax.utils.types import Sample


# ---- release pixel tensors once a batch crossed the TQ boundary (2026-09-08) ----
# The pipeline keeps every Sample of the rollout (context.output_groups) for the
# end-of-rollout metrics / log. Full debug dumps need pixels and bypass release.
# Each Sample carried its full image window (~1.7 GiB per session in bf16), so the
# RolloutManager grew by that amount per finished session until the rollout ended.
# After async_put() succeeded the trainer owns the only copy that matters; replace
# pixel tensors with 0-row stand-ins that keep dtype and width. image_grid_thw is
# tiny and stays (metrics derive per-image token counts from it).
_MM_RELEASE_AFTER_TRANSFER = Envs.RELAX_MM_RELEASE_AFTER_TRANSFER
# Replacing the tensor is not enough: glibc raises its dynamic mmap threshold after
# the first large free, so later pixel blocks come from the arena and free() only
# returns them to the process, never to the OS -- RSS climbed monotonically for the
# whole 3h rollout even though every batch had been released. gc + malloc_trim(0)
# hands the arena's free pages back. The log line tells us which of the two it was.
_TRANSFER_BATCH_GROUPS = Envs.RELAX_TRANSFER_BATCH_GROUPS
_MM_RELEASE_TRIM = Envs.RELAX_MM_RELEASE_TRIM
_MM_RELEASE_LOG = Envs.RELAX_MM_RELEASE_LOG
_MM_LOG = get_logger(__name__)
try:
    _MM_LIBC = ctypes.CDLL("libc.so.6")
except Exception:
    _MM_LIBC = None


def _mm_rss_bytes() -> int:
    try:
        with open("/proc/self/statm", "rb") as _f:
            return int(_f.read().split()[1]) * 4096
    except Exception:
        return 0


def _release_transferred_pixels(samples: List[Sample]) -> int:
    if not _MM_RELEASE_AFTER_TRANSFER:
        return 0
    freed = 0
    for sample in samples:
        mm = getattr(sample, "multimodal_train_inputs", None)
        if not isinstance(mm, dict):
            continue
        for key in ("pixel_values", "pixel_values_videos"):
            v = mm.get(key)
            if isinstance(v, torch.Tensor) and v.numel() > 0:
                freed += v.numel() * v.element_size()
                mm[key] = v.new_empty((0,) + tuple(v.shape[1:]))
            del v  # Drop the final tensor reference before collecting/trimming.
    if freed > 0 and _MM_RELEASE_TRIM:
        _rss0 = _mm_rss_bytes()
        gc.collect()
        if _MM_LIBC is not None:
            try:
                _MM_LIBC.malloc_trim(0)
            except Exception:
                pass
        if _MM_RELEASE_LOG:
            _rss1 = _mm_rss_bytes()
            _MM_LOG.info(
                "mm_release samples=%d freed=%.1fGiB rss=%.1f->%.1fGiB returned=%.1fGiB",
                len(samples),
                freed / 2**30,
                _rss0 / 2**30,
                _rss1 / 2**30,
                (_rss0 - _rss1) / 2**30,
            )
    return freed


TransferBatch = Tuple[int, Tuple[GroupExport, ...], bool]


async def _transfer_batch_to_data_system(
    *,
    args: Namespace,
    batch_samples: list[list[Sample]],
    partition_id: int,
    data_system_client: Any,
    is_last: bool = False,
    on_put_started: Optional[Callable[[], None]] = None,
) -> None:
    """Convert complete Samples and cross the TQ boundary.

    ``partition_id`` is the physical TQ partition, which is not always the
    driver step that owns this write: the domain keeps a previous and a current
    partition open at once.
    """

    from relax.utils.utils import build_rollout_custom_meta, convert_samples_to_train_data

    samples = [sample for group in batch_samples for sample in group]
    for sample in samples:
        mark_sample_agentic_event(sample, "transfer_release_start_at")
    try:
        batch_samples.sort(key=lambda group: group[0].index)
        samples = [sample for group in batch_samples for sample in group]
        rollout_batch = convert_samples_to_train_data(args, samples)
        custom_meta = build_rollout_custom_meta(rollout_batch)
        if on_put_started is not None:
            on_put_started()
        await data_system_client.async_put(
            data=rollout_batch,
            partition_id=f"train_{partition_id}",
            custom_meta=custom_meta,
            is_last=is_last,
        )
        del rollout_batch
        if getattr(args, "save_debug_rollout_data", None) is None and getattr(args, "dump_details", None) is None:
            _release_transferred_pixels(samples)
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
        # In colocate (non fully_async) mode the batch count defaults to the whole
        # rollout, so no partition is ever "ready" mid-rollout: every transfer -- and
        # therefore every pixel release -- is deferred to flush_batches() at the end.
        # RolloutManager then carries all N sessions' pixel tensors (2.65 GiB/session
        # measured) until then, which is what caps us at ~256 sessions. Overriding the
        # count makes the partition flush every K groups so the peak is one batch.
        self._transfer_batch_group_count = (
            args.global_batch_size // args.num_iters_per_train_update // args.n_samples_per_prompt
            if args.fully_async
            else (_TRANSFER_BATCH_GROUPS or args.rollout_batch_size)
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

    @property
    def previous_debt(self) -> int:
        """Return debt that must be filled before another partition opens."""

        return self._previous_partition.remaining_groups if self._previous_partition is not None else 0

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

    def pending_groups(self) -> Tuple[GroupExport, ...]:
        """Report accepted Groups that no batch has detached yet.

        Read before ``clear()`` so shutdown can record what it is about to
        discard: these Groups were admitted into a partition but never crossed
        the TQ boundary.
        """

        return tuple(
            group
            for partition in (self._previous_partition, self._current_partition)
            if partition is not None
            for group in partition.pending_batch
        )

    def clear(self) -> None:
        """Release resident partition refs during terminal shutdown."""

        self._previous_partition = None
        self._current_partition = None
