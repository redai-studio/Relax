# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Bounded reusable pinned host buffers, independent of the replay cache.

Each device retains at most two 128 MiB slots. Oversized fallback copies are
transient and do not enlarge this persistent pool.
"""

from __future__ import annotations

import os
from threading import Lock
from typing import Any


_MIN_SLOT_BYTES = 1 << 20
_MAX_SLOT_BYTES = 128 << 20
_SLOT_COUNT = 2
_pools: dict[int, _PinnedCopies] = {}
_pool_lock = Lock()


class _PinnedCopies:
    def __init__(self, device: Any) -> None:
        import torch

        self.device = device
        self.pid = os.getpid()
        self.stream = torch.cuda.Stream(device=device)
        self.lock = Lock()
        self.slots: list[tuple[Any, Any] | None] = [None] * _SLOT_COUNT
        self.next_slot = 0

    def copy(self, values: Any, mask: Any) -> tuple[Any, Any]:
        import torch

        value_bytes = values.numel() * values.element_size()
        required = value_bytes + mask.numel()
        capacity = max(_MIN_SLOT_BYTES, 1 << (required - 1).bit_length())
        current = torch.cuda.current_stream(self.device)
        with self.lock, torch.cuda.device(self.device):
            # Oversized individual layers retain PyTorch's tracked pinned-copy
            # lifetime; they never enlarge the persistent two-slot pool.
            if capacity > _MAX_SLOT_BYTES:
                host_values, host_mask = values.pin_memory(), mask.pin_memory()
                slot_index = None
                event = torch.cuda.Event()
            else:
                slot_index = self.next_slot
                self.next_slot = (slot_index + 1) % len(self.slots)
                slot = self.slots[slot_index]
                if slot is not None:
                    buffer, event = slot
                    # This event covers DMA only, not the model's compute.
                    if not event.query():
                        event.synchronize()
                    if buffer.numel() < capacity:
                        self.slots[slot_index] = None
                        del buffer, slot
                        buffer = torch.empty(capacity, dtype=torch.uint8, pin_memory=True)
                else:
                    buffer = torch.empty(capacity, dtype=torch.uint8, pin_memory=True)
                    event = torch.cuda.Event()
                host_values = buffer[:value_bytes].view(values.dtype).view(values.shape)
                host_mask = buffer[value_bytes:required].view(torch.bool).view(mask.shape)
                host_values.copy_(values)
                host_mask.copy_(mask)
                self.slots[slot_index] = (buffer, event)
            with torch.cuda.stream(self.stream):
                try:
                    result = host_values.to(self.device, non_blocking=True)
                    result_mask = host_mask.to(self.device, non_blocking=True)
                finally:
                    event.record(self.stream)
            current.wait_event(event)
            # Outputs are caller-owned. Their allocations must outlive work on
            # the consuming stream even after this function drops its refs.
            result.record_stream(current)
            result_mask.record_stream(current)
            return result, result_mask


def copy_indexer(values: Any, mask: Any, device: Any) -> tuple[Any, Any]:
    import torch

    index = device.index if device.index is not None else torch.cuda.current_device()
    with _pool_lock:
        pool = _pools.get(index)
        if pool is not None and pool.pid != os.getpid():
            raise RuntimeError("Create replay CUDA staging after spawn, not before fork")
        if pool is None:
            pool = _PinnedCopies(torch.device("cuda", index))
            _pools[index] = pool
    return pool.copy(values, mask)
