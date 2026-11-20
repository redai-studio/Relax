# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Local shard copies for a locked CUDA weight publication."""

from collections.abc import Iterable, Iterator, Mapping, Sequence
from typing import Any

import torch


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
        self.storage = (
            spare
            if spare is not None and spare.numel() == size
            else torch.empty(size, dtype=torch.uint8, device="cpu", pin_memory=True)
        )
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

    def __init__(self, shards: Mapping[str, torch.Tensor], device: torch.device, slot_bytes: int = 0) -> None:
        self.shards = shards
        self.device = device
        self.slot_bytes = slot_bytes
        self.active = {}
        self._slots = [None, None]
        self._released = [None, None]
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

    def prefetch(self, buckets: Sequence[Sequence[Any]], vanilla: Mapping[str, str], rank: int) -> Iterator:
        if not self.slot_bytes or not buckets:
            yield from buckets
            return
        compute = torch.cuda.current_stream(self.device)
        if self._copies is None:
            self._copies = torch.cuda.Stream(device=self.device)
        copies, slots, released = self._copies, self._slots, self._released

        def post(index):
            group = buckets[index]
            items = [(vanilla[i.name], self.shards[vanilla[i.name]]) for i in group if i.src_rank == rank]
            layout, size = shard_layout(items)
            if not size or size > self.slot_bytes or not all(t.is_pinned() for _, t in items):
                return {}, None
            slot = index % 2
            views = {}
            with torch.cuda.stream(copies):
                if released[slot] is not None:
                    copies.wait_event(released[slot])
                if slots[slot] is None:
                    try:
                        slots[slot] = torch.empty(self.slot_bytes, dtype=torch.uint8, device=self.device)
                    except torch.OutOfMemoryError:
                        # Only this optional lookahead is skipped. The original
                        # per-input H2D still runs inside the converter.
                        return {}, None
                for name, host, offset, length in layout:
                    value = _view(slots[slot], host, offset, length)
                    value.copy_(host, non_blocking=True)
                    views[name] = value
                ready = copies.record_event()
            return views, ready

        current = post(0)
        for i, group in enumerate(buckets):
            following = post(i + 1) if i + 1 < len(buckets) else None
            self.active, ready = current
            if ready is not None:
                compute.wait_event(ready)
                slots[i % 2].record_stream(compute)
            try:
                yield group
            finally:
                self.active = {}
                if ready is not None:
                    released[i % 2] = compute.record_event()
            current = following
