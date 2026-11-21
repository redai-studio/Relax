# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Worker-local, device-resident reuse of deterministic frozen conditions.

Each cache belongs to one loaded model and is discarded on resume/teardown.
Weights, tokenizer, processor and precision are immutable during that lifetime;
entries cannot cross model instances or checkpoints. Adapters own dependencies
and tensor packing. The cache knows neither model names nor training
algorithms.
"""

from __future__ import annotations

import hashlib
import json
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any

import torch


def image_condition_key(data: dict[str, Any], *, version: str, dtype: torch.dtype) -> bytes:
    """Hash actual post-augmentation inputs on CPU, never filenames or GPU
    tensors.

    Target pixels are not conditions; target geometry can affect positional
    IDs. Ordered reference pixels and the final prompt include
    crop/augmentation and caption-dropout outcomes. A stochastic encoder itself
    is not cacheable.
    """
    digest = hashlib.sha256()
    metadata = [version, str(dtype), data["prompt"], data["image"].size]
    digest.update(json.dumps(metadata, ensure_ascii=False).encode())
    for image in data["edit_image"]:
        pixels = image.tobytes()
        digest.update(json.dumps([image.mode, image.size, len(pixels)]).encode())
        digest.update(pixels)
    return digest.digest()


@dataclass(frozen=True)
class ConditionRequest:
    key: bytes
    tensors: dict[str, torch.Tensor] | None


class FrozenConditionCache:
    """Byte/entry bounded LRU. Returned tensors are borrowed read-only.

    Entries own freshly encoded storage; no cross-entry views. Account backing
    storages (including views), not tensor numel. Admission never copies or
    casts tensors and never synchronizes a GPU stream. Encoding must use
    no_grad, not inference_mode, since generation backward may save these
    tensors.
    """

    def __init__(self, max_bytes: int, max_entries: int) -> None:
        if type(max_bytes) is not int or max_bytes <= 0 or type(max_entries) is not int or max_entries <= 0:
            raise ValueError("Frozen condition cache requires positive integer byte and entry limits")
        self.max_bytes, self.max_entries = max_bytes, max_entries
        self._entries: OrderedDict[bytes, tuple[dict[str, torch.Tensor], int]] = OrderedDict()
        self.resident_bytes = 0
        self.peak_bytes = 0
        self.lookups = self.local_hits = self.hits = self.encodes = self.evictions = self.oversized = 0

    def lookup(self, key: bytes) -> ConditionRequest:
        self.lookups += 1
        entry = self._entries.get(key)
        if entry is not None:
            self.local_hits += 1
            self._entries.move_to_end(key)
        return ConditionRequest(key, None if entry is None else entry[0])

    def put(self, key: bytes, tensors: dict[str, torch.Tensor]) -> None:
        # This is the adapter/cache boundary. Reject a graph instead of silently
        # detaching trainable conditions and producing incorrect gradients.
        if not tensors or any(t.requires_grad or t.grad_fn is not None for t in tensors.values()):
            raise ValueError("Frozen conditions must be nonempty tensors encoded without autograd")
        storages = {}
        for tensor in tensors.values():
            storage = tensor.untyped_storage()
            storages[(storage.device, storage.data_ptr())] = storage.nbytes()
        size = sum(storages.values())
        old = self._entries.pop(key, None)
        if old is not None:
            self.resident_bytes -= old[1]
        if size > self.max_bytes:
            self.oversized += 1
            return
        while self._entries and (
            self.resident_bytes + size > self.max_bytes or len(self._entries) >= self.max_entries
        ):
            _, (_, removed) = self._entries.popitem(last=False)
            self.resident_bytes -= removed
            self.evictions += 1
        self._entries[key] = (tensors, size)
        self.resident_bytes += size
        self.peak_bytes = max(self.peak_bytes, self.resident_bytes)

    def clear(self) -> None:
        self._entries.clear()
        self.resident_bytes = 0

    def metrics(self) -> dict[str, int]:
        return {
            "lookups": self.lookups,
            "local_hits": self.local_hits,
            "hits": self.hits,
            "encodes": self.encodes,
            "evictions": self.evictions,
            "oversized": self.oversized,
            "resident_bytes": self.resident_bytes,
            "peak_bytes": self.peak_bytes,
            "entries": len(self._entries),
        }


def agree_condition_request(request: ConditionRequest, all_hit: bool) -> ConditionRequest:
    """Every FSDP rank must execute the same conditional encoder
    collectives."""
    return request if all_hit else ConditionRequest(request.key, None)
