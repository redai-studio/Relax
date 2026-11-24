# Copyright (c) 2026 Relax Authors. All Rights Reserved.
# Adapted from verl, Copyright 2025 Bytedance Ltd. and/or its affiliates, Apache-2.0.

"""Carry verl's positions, typed values and per-flush manifest in one CUDA IPC
allocation."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterable, Iterator
from itertools import groupby

import torch

from .verl_codec import DeltaParam, checksum


_DTYPES = {
    name: getattr(torch, name)
    for name in (
        "bool",
        "uint8",
        "int8",
        "int16",
        "float16",
        "bfloat16",
        "int32",
        "float32",
        "int64",
        "float64",
    )
}
WIRE_ALIGNMENT = 8
_FIELDS = {"name", "shape", "dtype", "pos_start", "pos_end", "pos_width", "val_start", "val_end", "full", "count"}


def schema_digest(schema: dict[str, tuple]) -> str:
    return hashlib.sha256(json.dumps(list(schema.items()), separators=(",", ":")).encode()).hexdigest()


def _aligned(size: int) -> int:
    return (size + WIRE_ALIGNMENT - 1) // WIRE_ALIGNMENT * WIRE_ALIGNMENT


def _flush_bounds(size: int, pos_bytes: int, val_bytes: int, limit: int) -> tuple[int, int, int]:
    begin = _aligned(size)
    value_offset = _aligned(begin + pos_bytes)
    end = value_offset + val_bytes
    if end > limit:
        raise ValueError("Complete loader group exceeds the admitted wire frame budget")
    return begin, value_offset, end


def _weight_flushes(weights: Iterable, limit: int) -> Iterator[tuple[list[dict], torch.Tensor, torch.Tensor]]:
    size = 0
    for (dtype, dense), group in groupby(weights, key=lambda row: (row[2], row[3] is None)):
        entries, indices, values = [], [], []
        count = pos_bytes = 0
        for name, shape, _, positions, tensor in group:
            if tensor.dtype != dtype or tensor.ndim != 1:
                raise ValueError("Invalid verl values")
            nnz = tensor.numel()
            if not dense:
                if (
                    math.prod(shape) >= 1 << 31
                    or not tensor.is_floating_point()
                    or positions.dtype != torch.int32
                    or positions.ndim != 1
                    or positions.numel() != nnz
                    or positions.device != tensor.device
                ):
                    raise ValueError("Invalid verl int32 positions or sparse storage dtype")
                indices.append(positions)
            elif nnz != math.prod(shape):
                raise ValueError("Incomplete native FULL weight")
            stop = pos_bytes if dense else pos_bytes + 4 * nnz
            param = DeltaParam(
                name, str(dtype).removeprefix("torch."), list(shape), pos_bytes, stop, 4, count, count + nnz
            )
            entries.append(dict(vars(param), full=dense, count=nnz))
            values.append(tensor)
            count += nnz
            pos_bytes = stop
        # Admit the complete group before allocating either concatenation.
        _, _, size = _flush_bounds(size, pos_bytes, count * values[0].element_size(), limit)
        vals = torch.cat(values)
        pos = torch.cat(indices).contiguous().view(torch.uint8) if indices else vals.new_empty(0, dtype=torch.uint8)
        yield entries, pos, vals


def encode(weights: Iterable, *, max_bytes: int) -> tuple[dict, torch.Tensor]:
    return encode_flushes(_weight_flushes(weights, max_bytes), max_bytes=max_bytes)


def encode_flushes(
    groups: Iterable[tuple[list[dict], torch.Tensor, torch.Tensor]], *, max_bytes: int
) -> tuple[dict, torch.Tensor]:
    """Encode typed verl flushes.

    Each pair of packed buffers can describe many slots and must stay immutable
    until the wire frame is assembled.
    """
    slots, flushes, parts = [], [], []
    size = 0
    direct = False
    for entries, pos, vals in groups:
        start = len(slots)
        if (
            not entries
            or pos.dtype != torch.uint8
            or pos.ndim != 1
            or vals.ndim != 1
            or vals.dtype != _DTYPES.get(entries[0]["dtype"])
            or pos.device != vals.device
            or not pos.is_contiguous()
            or not vals.is_contiguous()
        ):
            raise ValueError("Invalid packed verl positions or values")
        pos_bytes = entries[-1]["pos_end"]
        if pos.numel() != pos_bytes or vals.numel() != entries[-1]["val_end"]:
            raise ValueError("Packed verl data differs from its slot offsets")
        begin, value_offset, end = _flush_bounds(size, pos_bytes, vals.nbytes, max_bytes)
        for length, block in ((begin - size, pos), (value_offset - begin - pos_bytes, vals.view(torch.uint8))):
            if length:
                parts.append(pos.new_zeros(length))
            parts.append(block)
        slots.extend(entries)
        direct = direct or entries[0].get("expert_direct", False)
        flushes.append(
            dict(
                start=start,
                stop=len(slots),
                offset=begin,
                value_offset=value_offset,
                end=end,
                encoding="dense" if entries[0]["full"] else "indices",
                checksum=checksum(pos, vals),
            )
        )
        size = end
    if not flushes:
        raise ValueError("A wire frame requires at least one tensor")
    directory = dict(format=2, bytes=size, tensors=slots, flushes=flushes)
    if direct:
        directory["expert_direct"] = True
    validate_directory(directory, max_bytes=max_bytes)
    return directory, torch.cat(parts)


class WireBucket:
    """Batch complete, independently checksummed verl flushes within the proxy
    budget."""

    def __init__(self, limit: int) -> None:
        self.limit = limit
        self.frames = []
        self.size = self.entries = 0

    def offer(self, directory: dict, raw: torch.Tensor) -> tuple | None:
        if raw.numel() > self.limit:
            raise ValueError("Single frame exceeds the wire bucket limit")
        count = len(directory["tensors"])
        emit = None
        if self.frames and (
            _aligned(self.size) + raw.numel() > self.limit
            or self.entries + count > 4096
            or self.frames[0][1].get("verify") != directory.get("verify")
        ):
            emit = self.finish()
        self.frames.append((_aligned(self.size), directory, raw))
        self.size = _aligned(self.size) + raw.numel()
        self.entries += count
        return emit

    def finish(self) -> tuple | None:
        if not self.frames:
            return None
        frames, size = self.frames, self.size
        self.frames, self.size, self.entries = [], 0, 0
        if len(frames) == 1:
            return frames[0][1:]
        slots, groups = [], []
        raw = frames[0][2].new_empty(size)
        end = 0
        for offset, directory, slab in frames:
            raw[end:offset].zero_()
            end = offset + slab.numel()
            raw[offset:end].copy_(slab)
            for f in directory["flushes"]:
                groups.append(
                    dict(
                        f,
                        start=f["start"] + len(slots),
                        stop=f["stop"] + len(slots),
                        offset=f["offset"] + offset,
                        value_offset=f["value_offset"] + offset,
                        end=f["end"] + offset,
                    )
                )
            slots.extend(directory["tensors"])
        directory = dict(format=2, bytes=size, tensors=slots, flushes=groups)
        if frames[0][1].get("verify"):
            directory["verify"] = True
        if any(slot.get("expert_direct") for slot in slots):
            directory["expert_direct"] = True
        return directory, raw


def validate_directory(meta: dict, *, max_bytes: int) -> None:
    if (
        not isinstance(meta, dict)
        or meta.keys() - {"verify", "expert_direct"} != {"format", "bytes", "tensors", "flushes"}
        or meta["format"] != 2
        or any(meta[key] is not True for key in ("verify", "expert_direct") if key in meta)
        or ("verify" in meta and "expert_direct" in meta)
    ):
        raise ValueError("Invalid verl wire directory")
    size, entries, groups = meta["bytes"], meta["tensors"], meta["flushes"]
    if meta.get("expert_direct") and type(size) is int and size == 0 and entries == [] and groups == []:
        return
    if (
        type(size) is not int
        or not 0 <= size <= max_bytes
        or not isinstance(entries, list)
        or not 0 < len(entries) <= 65536
        or not isinstance(groups, list)
        or not 0 < len(groups) <= len(entries)
    ):
        raise ValueError("Invalid verl frame bounds")
    cursor = offset = 0
    for group in groups:
        if set(group) != {"start", "stop", "offset", "value_offset", "end", "encoding", "checksum"}:
            raise ValueError("Invalid verl flush metadata")
        stop = group["stop"]
        if (
            type(stop) is not int
            or not cursor < stop <= len(entries)
            or group["start"] != cursor
            or group["offset"] != _aligned(offset)
            or type(group["checksum"]) is not int
            or group["encoding"] not in ("dense", "indices")
        ):
            raise ValueError("Invalid verl flush boundaries")
        dtype = entries[cursor]["dtype"]
        dense = group["encoding"] == "dense"
        direct = entries[cursor].get("expert_direct")
        if (direct is not None and (direct is not True or not meta.get("expert_direct") or dense)) or (
            meta.get("verify") and not dense
        ):
            raise ValueError("Invalid purpose for a verl flush")
        pos = val = 0
        for slot in entries[cursor:stop]:
            if (
                slot.keys() - {"expert_direct"} != _FIELDS
                or slot.get("expert_direct") is not direct
                or ("expert_direct" in slot and direct is not True)
                or type(slot["name"]) is not str
                or not slot["name"]
                or len(slot["name"]) > 4096
                or dtype not in _DTYPES
                or slot["dtype"] != dtype
                or slot["full"] is not dense
                or not isinstance(slot["shape"], list)
                or len(slot["shape"]) > 16
                or any(type(x) is not int or x < 0 for x in slot["shape"])
            ):
                raise ValueError("Invalid verl named slot")
            count = slot["count"]
            volume = math.prod(slot["shape"])
            if (
                type(count) is not int
                or not 0 <= count <= volume
                or (dense and count != volume)
                or (not dense and (volume >= 1 << 31 or not _DTYPES[dtype].is_floating_point))
                or slot["pos_width"] != 4
                or slot["pos_start"] != pos
                or slot["val_start"] != val
            ):
                raise ValueError("Invalid verl slot coordinates")
            pos += 0 if dense else 4 * count
            val += count
            if slot["pos_end"] != pos or slot["val_end"] != val:
                raise ValueError("Invalid verl payload offsets")
        base = _aligned(group["offset"] + pos)
        offset = base + val * _DTYPES[dtype].itemsize
        if group["value_offset"] != base or group["end"] != offset or offset > size:
            raise ValueError("Invalid verl storage ranges")
        cursor = stop
    if cursor != len(entries) or offset != size:
        raise ValueError("Incomplete verl frame directory")


def frame_flushes(metadata: dict, raw: torch.Tensor) -> Iterator[tuple[dict, torch.Tensor, torch.Tensor]]:
    """Expose the two contiguous upstream blobs directly from the proxy-owned
    IPC slab."""
    for group in metadata["flushes"]:
        params = metadata["tensors"][group["start"] : group["stop"]]
        pos = raw[group["offset"] : group["offset"] + params[-1]["pos_end"]]
        vals = raw[group["value_offset"] : group["end"]].view(_DTYPES[params[0]["dtype"]])
        spec = dict(encoding=group["encoding"], checksum=group["checksum"], params=params)
        if params[0].get("expert_direct"):
            spec["expert_direct"] = True
        yield spec, pos, vals
