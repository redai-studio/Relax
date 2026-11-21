# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Local shard copies for a locked CUDA weight publication."""

from collections.abc import Iterable, Iterator, Mapping, Sequence
from time import perf_counter
from typing import Any

import torch

from relax.utils.logging_utils import get_logger


logger = get_logger(__name__)


def _allocate_bank(size: int, *, spare: bool = False) -> torch.Tensor:
    started = perf_counter()
    bank = torch.empty(size, dtype=torch.uint8, device="cpu", pin_memory=True)
    logger.info("Pinned shard bank allocated: bytes=%d spare=%s elapsed_s=%.6f", size, spare, perf_counter() - started)
    return bank


def preallocate_spare_bank(size: int, device: torch.device) -> torch.Tensor | None:
    # CUDA's current device is thread-local. The host allocation belongs to
    # the same rank/device as the mandatory shard bank.
    with torch.cuda.device(device):
        try:
            return _allocate_bank(size, spare=True)
        except (MemoryError, RuntimeError) as exc:
            # Some pinned allocators expose CUDA/OS exhaustion as a plain RuntimeError.
            if not isinstance(exc, (MemoryError, torch.OutOfMemoryError)) and not any(
                text in str(exc).lower()
                for text in ("out of memory", "cannot allocate memory", "can't allocate memory")
            ):
                raise
            logger.warning("Could not preallocate %d pinned shard bytes: %s", size, exc)
            return None


def shard_layout(items: Iterable[tuple[str, torch.Tensor]]) -> tuple[list[tuple[str, torch.Tensor, int, int]], int]:
    """Byte offsets aligned for typed views, including scalars and empties."""
    entries = []
    size = 0
    for name, tensor in items:
        width = tensor.element_size()
        offset = (size + width - 1) // width * width
        length = tensor.numel() * width
        entries.append((name, tensor, offset, length))
        size = offset + length
    return entries, size


def _view(storage: torch.Tensor, tensor: torch.Tensor, offset: int, length: int) -> torch.Tensor:
    return storage.narrow(0, offset, length).view(tensor.dtype).view(tensor.shape)


def _copy_bank_slices(bank: torch.Tensor, slot: torch.Tensor, layout: list) -> None:
    """Copy adjacent typed views in one DMA, including their alignment
    padding."""
    src, dst, end = 0, 0, 0
    for _, host, offset, length in layout:
        if not length:
            continue
        start = host.storage_offset() * host.element_size()
        if end == dst or start - src != offset - dst:
            if end > dst:
                slot[dst:end].copy_(bank[src : src + end - dst], non_blocking=True)
            src, dst = start, offset
        end = offset + length
    if end > dst:
        slot[dst:end].copy_(bank[src : src + end - dst], non_blocking=True)


class CudaShardBackup:
    """Copy stable local shards into the next pinned bank on a private
    stream."""

    def __init__(
        self,
        entries: list[tuple[str, torch.Tensor, int, int]],
        size: int,
        spare: torch.Tensor | None,
        budget: int,
        bucket_bytes: int,
    ) -> None:
        if size > budget:
            raise ValueError("Acknowledged and pending aligned shard banks exceed their host budget")
        self.storage = spare if spare is not None and spare.numel() == size else _allocate_bank(size)
        device = entries[0][1].device
        self.stream = torch.cuda.Stream(device=device)
        self.targets = {}
        self.events = []
        self.stream.wait_stream(torch.cuda.current_stream(device))
        try:
            with torch.cuda.stream(self.stream):
                copied = 0
                for name, source, offset, length in entries:
                    if copied and copied + length > bucket_bytes:
                        self.events.append(self.stream.record_event())
                        copied = 0
                    target = _view(self.storage, source, offset, length)
                    target.copy_(source.detach(), non_blocking=True)
                    self.targets[name] = target
                    copied += length
                self.events.append(self.stream.record_event())
        except BaseException:
            # A partial enqueue must finish before its inputs or bank can be reused.
            self.stream.synchronize()
            raise

    def finish(self) -> None:
        # The last event is ordered after every shard on this copy stream.
        self.events[-1].synchronize()


class CudaShardInputs(Mapping[str, torch.Tensor]):
    """Provide privately owned CUDA inputs to the ordinary Bridge converter.

    A converter may retain a view across a yield or use an in-place mapping.
    Borrowed model storage and reusable prefetch slots therefore leave this
    boundary as independent per-input tensors on the converter's stream.
    """

    def __init__(
        self,
        shards: Mapping[str, torch.Tensor],
        device: torch.device,
        slot_bytes: int = 0,
        *,
        bank: torch.Tensor | None = None,
    ) -> None:
        self.shards = shards
        self.device = device
        self.slot_bytes = slot_bytes
        self._bank = bank
        self.active = {}
        self._slots = [None, None]
        self._slot_capacity = [0, 0]
        self._copies = None

    def __len__(self) -> int:
        return len(self.shards)

    def __iter__(self) -> Iterator[str]:
        return iter(self.shards)

    def __getitem__(self, name: str) -> torch.Tensor:
        value = self.active.get(name)
        if value is None:
            value = self.shards[name]
        if value.device == self.device:
            return value.detach().clone()
        return value.detach().to(self.device, non_blocking=True)

    def join_prefetch(self, compute: torch.cuda.Stream) -> None:
        if self._copies is not None:
            # On cancellation the lookahead may have no consumer. The bank
            # owner must fence even these copies before recycling host storage.
            compute.wait_stream(self._copies)

    def _prefetch_bucket(self, index: int, group: Sequence[Any], vanilla: Mapping[str, str], rank: int) -> tuple:
        items = [(vanilla[i.name], self.shards[vanilla[i.name]]) for i in group if i.src_rank == rank]
        # Bridge's lexical ordering can differ from the packed bank's storage
        # order. Converter lookup is by name, so coalesce adjacent bank slices.
        if self._bank is not None:
            items.sort(key=lambda item: item[1].storage_offset() * item[1].element_size())
        elif not all(t.is_pinned() for _, t in items):
            return {}, None
        layout, size = shard_layout(items)
        if not size or size > self.slot_bytes:
            return {}, None
        slot = index % 2
        capacity = self._slot_capacity[slot]
        with torch.cuda.stream(self._copies):
            buf = self._slots[slot]
            if buf is not None and capacity != buf.numel():
                self._slots[slot] = buf = None
            if buf is None:
                try:
                    self._slots[slot] = buf = torch.empty(capacity, dtype=torch.uint8, device=self.device)
                except torch.OutOfMemoryError:
                    # The original per-input H2D remains available to Bridge.
                    return {}, None
            views = {name: _view(buf, host, offset, length) for name, host, offset, length in layout}
            if self._bank is None:
                for name, host, _, _ in layout:
                    views[name].copy_(host, non_blocking=True)
            else:
                _copy_bank_slices(self._bank, buf, layout)
            return views, self._copies.record_event()

    def prefetch(self, buckets: Sequence[Sequence[Any]], vanilla: Mapping[str, str], rank: int) -> Iterator:
        if not self.slot_bytes or not buckets:
            yield from buckets
            return
        compute = torch.cuda.current_stream(self.device)
        if self._copies is None:
            self._copies = torch.cuda.Stream(device=self.device)
        # Plan each parity slot once per Bridge phase. Include at most one
        # alignment gap per input, independent of the bank's storage order.
        self._slot_capacity = [0, 0]
        for i, group in enumerate(buckets):
            local = [self.shards[vanilla[info.name]] for info in group if info.src_rank == rank]
            size = sum(t.nbytes for t in local)
            if 0 < size <= self.slot_bytes:
                alignment = sum(t.element_size() - 1 for t in local)
                capacity = min(size + alignment, self.slot_bytes)
                self._slot_capacity[i % 2] = max(self._slot_capacity[i % 2], capacity)
        current = self._prefetch_bucket(0, buckets[0], vanilla, rank)
        for i, group in enumerate(buckets):
            following = self._prefetch_bucket(i + 1, buckets[i + 1], vanilla, rank) if i + 1 < len(buckets) else None
            self.active, ready = current
            if ready is not None:
                compute.wait_event(ready)
            try:
                yield group
            finally:
                self.active = {}
                if ready is not None:
                    # Slots are allocated and retired on the copy stream.
                    # Fence the last consumer there before reuse or resize.
                    self._copies.wait_stream(compute)
            current = following
