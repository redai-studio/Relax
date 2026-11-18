# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Lossless CPU tensor buffers for ART engine-to-driver Ray transport."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class TensorBuffer:
    data: object
    dtype: object
    shape: tuple[int, ...]


def pack(value: Any, *, min_bytes: int = 1048576) -> Any:
    """Expose large contiguous CPU tensors as byte arrays for Ray
    serialization.

    Distinct views sharing storage use the original transport to retain
    aliases. No dtype conversion or GPU-to-CPU transfer is performed.
    """
    import torch

    memo = {}
    owners = {}
    seen = set()

    def storage_key(item: Any) -> tuple[int, int] | None:
        if (
            isinstance(item, torch.Tensor)
            and item.device.type == "cpu"
            and item.layout == torch.strided
            and item.numel()
            and not item.is_quantized
        ):
            storage = item.untyped_storage()
            return (storage.data_ptr(), storage.nbytes())
        return None

    def inspect(item: Any) -> None:
        identity = id(item)
        if identity in seen:
            return
        seen.add(identity)
        key = storage_key(item)
        if key is not None:
            owners.setdefault(key, set()).add(identity)
        elif isinstance(item, dict):
            for child in item.values():
                inspect(child)
        elif isinstance(item, (list, tuple)):
            for child in item:
                inspect(child)

    inspect(value)

    def visit(item: Any) -> Any:
        identity = id(item)
        if identity in memo:
            return memo[identity]
        if isinstance(item, torch.Tensor):
            if (
                item.device.type == "cpu"
                and item.layout == torch.strided
                and item.is_contiguous()
                and not item.requires_grad
                and not item.is_quantized
                and not item.is_conj()
                and not item.is_neg()
                and item.dtype
                in (
                    torch.float16,
                    torch.bfloat16,
                    torch.float32,
                    torch.float64,
                    torch.uint8,
                    torch.int8,
                    torch.int16,
                    torch.int32,
                    torch.int64,
                    torch.bool,
                )
                and item.numel() > 0
                and item.numel() * item.element_size() >= min_bytes
                and len(owners.get(storage_key(item), ())) == 1
            ):
                result = TensorBuffer(item.reshape(-1).view(torch.uint8).numpy(), item.dtype, tuple(item.shape))
            else:
                result = item
        elif isinstance(item, dict):
            result = {}
            memo[identity] = result
            result.update((key, visit(child)) for key, child in item.items())
        elif isinstance(item, list):
            result = []
            memo[identity] = result
            result.extend(visit(child) for child in item)
        elif isinstance(item, tuple):
            result = tuple(visit(child) for child in item)
        else:
            return item
        memo[identity] = result
        return result

    return visit(value)


def unpack(value: Any) -> Any:
    """Restore tensors with owned writable storage, preserving repeated
    objects."""
    import numpy as np
    import torch

    memo = {}

    def visit(item: Any) -> Any:
        identity = id(item)
        if identity in memo:
            return memo[identity]
        if isinstance(item, TensorBuffer):
            if not isinstance(item.data, np.ndarray) or item.data.dtype != np.uint8 or item.data.ndim != 1:
                raise ValueError("Invalid CPU tensor buffer")
            # Ray numpy buffers are read-only. Copy once before exposing a writable tensor.
            result = torch.from_numpy(item.data.copy()).view(item.dtype).reshape(item.shape)
        elif isinstance(item, dict):
            result = {}
            memo[identity] = result
            result.update((key, visit(child)) for key, child in item.items())
        elif isinstance(item, list):
            result = []
            memo[identity] = result
            result.extend(visit(child) for child in item)
        elif isinstance(item, tuple):
            result = tuple(visit(child) for child in item)
        else:
            return item
        memo[identity] = result
        return result

    return visit(value)


def unpack_shards(shards: list[Any], *, max_workers: int = 4) -> list[Any]:
    """Bounded owned copies; results stay in shard order and failures join all
    workers."""
    if max_workers not in (1, 2, 4):
        raise ValueError("Unpack worker count must be 1, 2 or 4")
    if max_workers == 1 or len(shards) < 2:
        return [unpack(shard) for shard in shards]
    from concurrent.futures import ThreadPoolExecutor

    pool = ThreadPoolExecutor(max_workers=min(max_workers, len(shards)), thread_name_prefix="return-copy")
    futures = []
    try:
        futures = [pool.submit(unpack, shard) for shard in shards]
        return [future.result() for future in futures]
    except BaseException:
        for future in futures:
            future.cancel()
        raise
    finally:
        pool.shutdown(wait=True, cancel_futures=True)
