# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Element mappings for lossless TP concatenation, QKV and gated MLP splits."""

import math
import mmap
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .codec import MAX_CHUNK_BYTES
from .store import digest
from .traffic import emit, file_write


@dataclass(frozen=True)
class TensorMapping:
    """Map a local source element to a logical tensor without numeric
    conversion.

    ``outputs`` follow the existing converter's output order. Returned pairs
    contain positions in the supplied index array and flat destination indices.
    Padding can be cropped only by a registered vocabulary mapping.
    """

    shape: tuple[int, ...]
    tp_size: int
    partition_dim: int | None
    outputs: tuple[tuple[str, tuple[int, ...]], ...]
    kind: str = "identity"
    qkv_groups: int = 0
    q_heads_per_group: int = 0
    head_dim: int = 0
    crop_vocabulary: bool = False

    def __post_init__(self):
        if not self.shape or any(type(d) is not int or d <= 0 for d in self.shape):
            raise ValueError("Source dimensions must be positive integers")
        if type(self.tp_size) is not int or self.tp_size < 1:
            raise ValueError("Invalid source TP size")
        if self.partition_dim is not None and not 0 <= self.partition_dim < len(self.shape):
            raise ValueError("Invalid source partition dimension")
        if len({name for name, _ in self.outputs}) != len(self.outputs):
            raise ValueError("Duplicate logical destination")
        total = math.prod(self.shape) * (self.tp_size if self.partition_dim is not None else 1)
        sizes = [math.prod(shape) for _, shape in self.outputs]
        if self.kind in ("identity", "gather_gate_up"):
            if len(sizes) != 1 or sizes[0] > total or (sizes[0] != total and not self.crop_vocabulary):
                raise ValueError("Identity mapping does not cover the logical output")
            if self.kind == "gather_gate_up" and (self.partition_dim not in (None, 0) or self.shape[0] % 2):
                raise ValueError("Gathered gate/up rows must split evenly")
        elif self.kind == "gate_up":
            if self.partition_dim not in (None, 0) or len(sizes) != 2 or sizes != [total // 2] * 2:
                raise ValueError("Gated MLP mapping requires equal gate/up halves")
            if self.shape[0] % 2:
                raise ValueError("Local gate/up rows must split evenly")
        elif self.kind == "qkv":
            if min(self.qkv_groups, self.q_heads_per_group, self.head_dim) < 1:
                raise ValueError("QKV head dimensions are required")
            if self.partition_dim not in (None, 0):
                raise ValueError("QKV source must be row partitioned")
            column_size = math.prod(self.shape[1:])
            kv = self.qkv_groups * self.head_dim * column_size
            if sizes != [kv * self.q_heads_per_group, kv, kv] or sum(sizes) != total:
                raise ValueError("QKV mapping does not cover the logical outputs")
        else:
            raise ValueError("Unregistered lossless mapping")

    def destinations(self, shard: int, indices) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        indices = np.asarray(indices, dtype=np.int64)
        shards = self.tp_size if self.partition_dim is not None else 1
        if not 0 <= shard < shards or indices.ndim != 1:
            raise ValueError("Invalid source shard or index array")
        if np.any(indices < 0) or np.any(indices >= math.prod(self.shape)):
            raise ValueError("Source index outside its shard")
        positions = np.arange(len(indices), dtype=np.int64)
        if self.kind in ("gate_up", "gather_gate_up") and self.partition_dim is not None:
            # Megatron keeps [gate_rank, up_rank] in each TP shard; the
            # complete converter receives [all_gate_ranks, all_up_ranks].
            half = math.prod(self.shape) // 2
            logical = (indices // half * self.tp_size + shard) * half + indices % half
        elif self.partition_dim is not None:
            inner = math.prod(self.shape[self.partition_dim + 1 :])
            axis = self.shape[self.partition_dim]
            outer = indices // (axis * inner)
            logical = outer * (axis * self.tp_size * inner) + shard * axis * inner + indices % (axis * inner)
        else:
            logical = indices
        if self.kind in ("identity", "gather_gate_up"):
            keep = logical < math.prod(self.outputs[0][1])
            return {self.outputs[0][0]: (positions[keep], logical[keep])}
        if self.kind == "gate_up":
            half = math.prod(self.outputs[0][1])
            return {
                name: (positions[logical // half == part], logical[logical // half == part] % half)
                for part, (name, _) in enumerate(self.outputs)
            }
        kv = self.head_dim * math.prod(self.shape[1:])
        query = self.q_heads_per_group * kv
        group_size = query + 2 * kv
        group, local = logical // group_size, logical % group_size
        result = {}
        for (name, _), offset, length in zip(self.outputs, (0, query, query + kv), (query, kv, kv), strict=True):
            keep = (local >= offset) & (local < offset + length)
            result[name] = (positions[keep], group[keep] * length + local[keep] - offset)
        return result


def apply_replacements(
    mapping: TensorMapping, shard: int, offset: int, payload: bytes, width: int, staging: Path, *, sync: bool = True
) -> dict:
    """Apply ordered index/new-value records to private canonical staging
    files.

    Hard-linked base files are detached before the first write. The caller
    commits or discards this staging directory after the complete group has
    passed its frame and content checks.
    """
    if width not in (1, 2, 4, 8) or type(offset) is not int or offset < 0:
        raise ValueError("Invalid source frame dimensions")
    if len(payload) > MAX_CHUNK_BYTES or len(payload) % (4 + width):
        raise ValueError("Invalid replacement frame")
    records = np.frombuffer(payload, dtype=np.dtype([("index", "<u4"), ("value", f"V{width}")]))
    indices = records["index"].astype(np.int64)
    if np.any(indices[1:] <= indices[:-1]):
        raise ValueError("Source replacement indices must be ordered and unique")
    destinations = mapping.destinations(shard, indices + offset)
    sizes = dict(mapping.outputs)
    applied = 0
    for name, (positions, targets) in destinations.items():
        if not len(positions):
            continue
        path = staging / (digest(name.encode("utf-8")) + ".bin")
        if path.is_symlink() or path.stat().st_size != math.prod(sizes[name]) * width:
            raise ValueError(f"Canonical staging schema differs: {name}")
        if path.stat().st_nlink > 1:
            descriptor, temporary = tempfile.mkstemp(prefix="detach-", dir=staging)
            os.close(descriptor)
            try:
                shutil.copyfile(path, temporary)
                file_write(temporary, Path(temporary).stat().st_size)
                os.replace(temporary, path)
            finally:
                Path(temporary).unlink(missing_ok=True)
        with path.open("r+b") as handle, mmap.mmap(handle.fileno(), 0) as memory:
            values = np.ndarray((math.prod(sizes[name]),), dtype=f"V{width}", buffer=memory)
            values[targets] = records["value"][positions]
            del values
            if sync:
                memory.flush()
        emit("W", len(positions) * width, str(path.resolve()), "mapped_replacement")
        applied += len(positions)
    return {"elements_applied": applied, "elements_cropped": len(indices) - applied}


def sync_replacements(mapping: TensorMapping, staging: Path) -> None:
    """Drain a complete private replacement group once before publication.

    Shared mmap writes are immediately visible to subsequent local readers;
    durability is needed at the completed group boundary, not per frame. The
    caller must still discard or fully replace a failed group.
    """
    for name, _ in mapping.outputs:
        path = staging / (digest(name.encode("utf-8")) + ".bin")
        if path.is_symlink():
            raise ValueError(f"Canonical staging is a symlink: {name}")
        with path.open("r+b") as handle:
            os.fsync(handle.fileno())
