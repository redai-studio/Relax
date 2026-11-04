# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Model-independent, bounded storage for deterministic frozen conditions.

Only adapters decide what is frozen and construct the content identity. This
module never encodes images, samples randomness, or owns model parameters.
Returned trees are private copies: callers cannot append to a shared KV cache.
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
import time
from collections import OrderedDict
from dataclasses import asdict, dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class FrozenConditionSpec:
    encoder_identity: str
    preprocessing_version: str
    dtype: str
    feature_schema: str
    adapter_version: str

    def __post_init__(self) -> None:
        if not all(isinstance(value, str) and value for value in asdict(self).values()):
            raise ValueError("Every frozen-condition identity component must be a nonempty string")


def _digest_value(digest: Any, value: Any) -> None:
    """Length-framed, type-sensitive hashing; never transfers a CUDA tensor."""
    if isinstance(value, torch.Tensor):
        if value.device.type != "cpu":
            raise ValueError("Condition keys must be built from CPU content, not CUDA tensors")
        if value.layout != torch.strided:
            raise TypeError("Condition identity only supports dense tensors")
        _digest_value(digest, ("tensor", str(value.dtype), tuple(value.shape)))
        value = value.detach().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
    if isinstance(value, bytes):
        digest.update(b"b" + str(len(value)).encode() + b":" + value)
    elif isinstance(value, dict):
        if not all(isinstance(key, str) for key in value):
            raise TypeError("Condition tree mapping keys must be strings")
        digest.update(b"d" + str(len(value)).encode() + b":")
        for key in sorted(value):
            _digest_value(digest, key)
            _digest_value(digest, value[key])
    elif isinstance(value, (tuple, list)):
        digest.update((b"t" if isinstance(value, tuple) else b"l") + str(len(value)).encode() + b":")
        for item in value:
            _digest_value(digest, item)
    elif value is None or type(value) in (str, int, float, bool):
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError("Non-finite condition identity")
        encoded = json.dumps(value, ensure_ascii=False, allow_nan=False).encode()
        digest.update(type(value).__name__.encode() + str(len(encoded)).encode() + b":" + encoded)
    else:
        raise TypeError(f"Unsupported condition identity type: {type(value).__name__}")


def condition_cache_key(spec: FrozenConditionSpec, content: Any) -> str:
    digest = hashlib.sha256()
    _digest_value(digest, asdict(spec))
    _digest_value(digest, content)
    return digest.hexdigest()


def _walk_tensors(tree: Any):
    if isinstance(tree, torch.Tensor):
        yield tree
    elif isinstance(tree, dict):
        if not all(isinstance(key, str) for key in tree):
            raise TypeError("Condition tree mapping keys must be strings")
        for value in tree.values():
            yield from _walk_tensors(value)
    elif isinstance(tree, (tuple, list)):
        for value in tree:
            yield from _walk_tensors(value)
    elif tree is not None and type(tree) not in (str, int, float, bool):
        raise TypeError(f"Unsupported condition payload type: {type(tree).__name__}")


def _copy_tree(tree: Any, device: torch.device | None) -> Any:
    if isinstance(tree, torch.Tensor):
        # copy=True protects both same-device and cross-device consumers. Never
        # cast: cache readback must preserve the encoder's exact dtype and bits.
        # Version-tracked storage also works when a caller fills the cache in
        # inference_mode; inference tensors themselves have no version counter.
        with torch.inference_mode(False):
            return tree.detach().to(device=device or tree.device, copy=True).contiguous()
    if isinstance(tree, dict):
        return {key: _copy_tree(value, device) for key, value in tree.items()}
    if isinstance(tree, tuple):
        return tuple(_copy_tree(value, device) for value in tree)
    if isinstance(tree, list):
        return [_copy_tree(value, device) for value in tree]
    return tree


def _tree_bytes(tree: Any) -> int:
    # Charge tensor bytes, Python tree/key/scalar overhead and Tensor headers.
    # max_entries separately bounds allocator/OrderedDict overhead. This is a
    # payload accounting limit, not a claim about CUDA allocator reservation.
    size = sys.getsizeof(tree)
    if isinstance(tree, torch.Tensor):
        return size + tree.numel() * tree.element_size()
    if isinstance(tree, dict):
        return size + sum(sys.getsizeof(key) + _tree_bytes(value) for key, value in tree.items())
    if isinstance(tree, (tuple, list)):
        return size + sum(_tree_bytes(value) for value in tree)
    return size


def _structure(tree: Any) -> Any:
    if isinstance(tree, torch.Tensor):
        return ("tensor", tuple(tree.shape), tree.dtype, tree.device, tree._version)
    if isinstance(tree, dict):
        return ("dict", tuple((key, _structure(value)) for key, value in tree.items()))
    if isinstance(tree, (tuple, list)):
        return (type(tree).__name__, tuple(_structure(value) for value in tree))
    return (type(tree).__name__, tree)


@dataclass
class _Entry:
    payload: Any
    size: int
    structure: Any


class ConditionFeatureCache:
    """Single-owner byte/entry-bounded LRU. No disk or global mutable state.

    CPU and CUDA storage use the same interface. An adapter may compose two
    instances for tiers; the first implementation deliberately avoids hidden
    background transfers. get() always returns new tensors, including on GPU.
    """

    def __init__(self, max_bytes: int, storage_device: str | torch.device = "cpu", max_entries: int = 4096) -> None:
        if type(max_bytes) is not int or max_bytes < 0 or type(max_entries) is not int or max_entries <= 0:
            raise ValueError("Cache byte limit must be nonnegative and entry limit positive")
        self.max_bytes, self.max_entries = max_bytes, max_entries
        self.storage_device = torch.device(storage_device)
        if self.storage_device.type not in ("cpu", "cuda"):
            raise ValueError("Condition cache storage must be CPU or CUDA")
        self._entries: OrderedDict[str, _Entry] = OrderedDict()
        self._bytes = 0
        self._closed = False
        self._counters = dict(hits=0, misses=0, puts=0, evictions=0, oversized=0, corruptions=0)
        self._counters.update(read_tensor_bytes=0, write_tensor_bytes=0, d2h_bytes=0, h2d_bytes=0)
        self._read_seconds = self._write_seconds = 0.0

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("Condition cache is closed")

    def get(self, key: str, device: str | torch.device | None = None) -> Any | None:
        self._require_open()
        started = time.monotonic()
        entry = self._entries.get(key)
        if entry is None:
            self._counters["misses"] += 1
            return None
        try:
            if _structure(entry.payload) != entry.structure:
                raise ValueError("Stored condition tree was modified")
            target = torch.device(device) if device is not None else None
            result = _copy_tree(entry.payload, target)
        except (TypeError, ValueError, AttributeError):
            self.discard(key)
            self._counters["corruptions"] += 1
            self._counters["misses"] += 1
            return None
        self._entries.move_to_end(key)
        self._counters["hits"] += 1
        for tensor in _walk_tensors(entry.payload):
            size = tensor.numel() * tensor.element_size()
            self._counters["read_tensor_bytes"] += size
            if tensor.device.type == "cpu" and target is not None and target.type == "cuda":
                self._counters["h2d_bytes"] += size
            if tensor.device.type == "cuda" and target is not None and target.type == "cpu":
                self._counters["d2h_bytes"] += size
        self._read_seconds += time.monotonic() - started
        return result

    def contains(self, key: str) -> bool:
        """Probe without a copy, for collective agreement before any
        transfer."""
        self._require_open()
        entry = self._entries.get(key)
        if entry is None:
            return False
        try:
            valid = _structure(entry.payload) == entry.structure
        except (TypeError, ValueError, AttributeError):
            valid = False
        if not valid:
            self.discard(key)
            self._counters["corruptions"] += 1
        return valid

    def put(self, key: str, payload: Any) -> bool:
        self._require_open()
        if not isinstance(key, str) or not key:
            raise ValueError("Cache keys must be nonempty strings")
        tensors = list(_walk_tensors(payload))
        if not tensors:
            raise ValueError("A frozen-condition payload must contain at least one tensor")
        if any(t.requires_grad or t.grad_fn is not None or t.layout != torch.strided for t in tensors):
            raise ValueError("Cache only accepts dense, detached frozen features")
        size = _tree_bytes(payload) + sys.getsizeof(key)
        self.discard(key)
        if size > self.max_bytes:
            self._counters["oversized"] += 1
            return False
        started = time.monotonic()
        while self._entries and (self._bytes + size > self.max_bytes or len(self._entries) >= self.max_entries):
            _, old = self._entries.popitem(last=False)
            self._bytes -= old.size
            self._counters["evictions"] += 1
        stored = _copy_tree(payload, self.storage_device)
        self._entries[key] = _Entry(stored, size, _structure(stored))
        self._bytes += size
        self._counters["puts"] += 1
        for tensor in tensors:
            count = tensor.numel() * tensor.element_size()
            self._counters["write_tensor_bytes"] += count
            if tensor.device.type == "cuda" and self.storage_device.type == "cpu":
                self._counters["d2h_bytes"] += count
            if tensor.device.type == "cpu" and self.storage_device.type == "cuda":
                self._counters["h2d_bytes"] += count
        self._write_seconds += time.monotonic() - started
        return True

    def discard(self, key: str) -> None:
        entry = self._entries.pop(key, None)
        if entry is not None:
            self._bytes -= entry.size

    def clear(self) -> None:
        self._entries.clear()
        self._bytes = 0

    def close(self) -> None:
        self.clear()
        self._closed = True

    def keys(self) -> tuple[str, ...]:
        return tuple(self._entries)

    def stats(self) -> dict[str, Any]:
        return {
            **self._counters,
            "entries": len(self._entries),
            "bytes": self._bytes,
            "max_bytes": self.max_bytes,
            "max_entries": self.max_entries,
            "storage_device": str(self.storage_device),
            "closed": self._closed,
            "read_seconds": self._read_seconds,
            "write_seconds": self._write_seconds,
        }


def all_ranks_hit(local_hit: bool, process_group: Any = None) -> bool:
    """All ranks take the same branch before invoking sharded encoders.

    Prefer the existing Gloo group to avoid a CUDA scalar round trip. The
    explicit NCCL fallback also works in standalone FSDP runners. This tiny
    control-plane synchronization is required for collective call ordering.
    """
    import torch.distributed as dist

    if not dist.is_available() or not dist.is_initialized():
        return bool(local_hit)
    group = process_group if process_group is not None else dist.group.WORLD
    if dist.get_world_size(group) == 1:
        return bool(local_hit)
    device = torch.device("cuda", torch.cuda.current_device()) if dist.get_backend(group) == "nccl" else "cpu"
    flag = torch.tensor([int(local_hit)], dtype=torch.int32, device=device)
    dist.all_reduce(flag, op=dist.ReduceOp.MIN, group=group)
    return bool(flag.item())
