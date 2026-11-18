# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""One raw-bit FULL/sparse wire shared by TCP and GPU transports.

The caller supplies final tensor coordinates. This module knows no model,
training partition, inference loader, snapshot, or update lifecycle.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Any


_DTYPES = {
    "bool": 1,
    "uint8": 1,
    "int8": 1,
    "int16": 2,
    "float16": 2,
    "bfloat16": 2,
    "int32": 4,
    "float32": 4,
    "int64": 8,
    "float64": 8,
}


def encode(weights: Iterable[tuple[str, tuple[int, ...], Any, Any, Any]], *, max_bytes: int) -> tuple[dict, Any]:
    """Pack one complete loader group; positions=None means FULL.

    Positions are sorted unique int32 coordinates supplied by the exporter.
    Values are copied into independent uint8 storage before any producer reuse.
    """
    import torch

    if type(max_bytes) is not int or max_bytes <= 0:
        raise ValueError("A positive wire frame budget is required")
    entries, pieces, size, device = [], [], 0, None
    names = set()
    for name, shape, dtype, positions, values in weights:
        dtype_name = str(dtype).removeprefix("torch.")
        if not isinstance(name, str) or not name or name in names or dtype_name not in _DTYPES:
            raise ValueError("Wire tensors require distinct names and supported storage dtypes")
        if not isinstance(shape, tuple) or any(type(dim) is not int or dim < 0 for dim in shape):
            raise ValueError("Wire tensor shapes must contain nonnegative integers")
        if values.dtype != dtype or values.ndim != 1 or not values.is_contiguous():
            raise ValueError("Wire values must be flat contiguous tensors in their actual storage dtype")
        device = values.device if device is None else device
        if device != values.device:
            raise ValueError("One wire frame must have one allocating device")
        count = values.numel()
        full = positions is None
        if full and count != math.prod(shape):
            raise ValueError("FULL values must cover the complete tensor")
        if not full and (
            positions.dtype != torch.int32
            or positions.ndim != 1
            or not positions.is_contiguous()
            or positions.numel() != count
            or positions.device != device
        ):
            raise ValueError("Sparse coordinates must be flat int32 on the values' device")
        # Eight-byte alignment also supports FULL int64/FP64 buffers.
        offset = (size + 7) // 8 * 8
        if offset > size:
            pieces.append(torch.zeros(offset - size, dtype=torch.uint8, device=device))
        position_bytes = 0 if full else count * 4
        value_offset = offset + position_bytes
        value_offset = (value_offset + _DTYPES[dtype_name] - 1) // _DTYPES[dtype_name] * _DTYPES[dtype_name]
        if not full:
            pieces.append(positions.view(torch.uint8))
        if value_offset > offset + position_bytes:
            pieces.append(torch.zeros(value_offset - offset - position_bytes, dtype=torch.uint8, device=device))
        pieces.append(values.view(torch.uint8))
        size = value_offset + count * _DTYPES[dtype_name]
        if size > max_bytes:
            raise ValueError("Complete loader group exceeds the admitted wire frame budget")
        entries.append(
            dict(
                name=name,
                shape=list(shape),
                dtype=dtype_name,
                full=full,
                count=count,
                offset=offset,
                value_offset=value_offset,
            )
        )
        names.add(name)
    if not entries:
        raise ValueError("A wire frame requires at least one tensor directory entry")
    raw = torch.cat(pieces)  # Independent immutable producer storage, including empty delta.
    metadata = dict(format=1, bytes=size, tensors=entries)
    validate_directory(metadata, max_bytes=max_bytes)
    return metadata, raw


def validate_directory(metadata: dict, *, max_bytes: int) -> None:
    """Bound allocations and disallow overlapping or ambiguous wire ranges."""
    if not isinstance(metadata, dict) or set(metadata) != {"format", "bytes", "tensors"}:
        raise ValueError("Invalid lossless wire directory")
    size, tensors = metadata["bytes"], metadata["tensors"]
    if (
        type(metadata["format"]) is not int
        or metadata["format"] != 1
        or type(size) is not int
        or not 0 <= size <= max_bytes
    ):
        raise ValueError("Unsupported wire format or payload size")
    if not isinstance(tensors, list) or not 0 < len(tensors) <= 65536:
        raise ValueError("Invalid wire tensor directory length")
    names, end = set(), 0
    for entry in tensors:
        if not isinstance(entry, dict) or set(entry) != {
            "name",
            "shape",
            "dtype",
            "full",
            "count",
            "offset",
            "value_offset",
        }:
            raise ValueError("Invalid wire tensor entry")
        name, shape, dtype = entry["name"], entry["shape"], entry["dtype"]
        if (
            not isinstance(name, str)
            or not name
            or len(name) > 4096
            or name in names
            or not isinstance(dtype, str)
            or dtype not in _DTYPES
        ):
            raise ValueError("Invalid wire name or storage dtype")
        if not isinstance(shape, list) or len(shape) > 16 or any(type(dim) is not int or dim < 0 for dim in shape):
            raise ValueError("Invalid wire shape")
        count, offset, value_offset = entry["count"], entry["offset"], entry["value_offset"]
        if type(entry["full"]) is not bool or any(type(n) is not int or n < 0 for n in (count, offset, value_offset)):
            raise ValueError("Invalid wire encoding or byte offsets")
        elements = math.prod(shape)
        if count > elements or (entry["full"] and count != elements) or (not entry["full"] and elements >= 2**31):
            raise ValueError("Wire coordinate count disagrees with tensor geometry")
        expected_offset = (end + 7) // 8 * 8
        expected_values = offset + (0 if entry["full"] else count * 4)
        expected_values = (expected_values + _DTYPES[dtype] - 1) // _DTYPES[dtype] * _DTYPES[dtype]
        if offset != expected_offset or value_offset != expected_values:
            raise ValueError("Wire ranges overlap or disagree with their canonical alignment")
        end = value_offset + count * _DTYPES[dtype]
        if end > size:
            raise ValueError("Wire tensor range exceeds the frame")
        names.add(name)
    if end != size:
        raise ValueError("Wire frame contains unclaimed trailing bytes")


def tensors(metadata: dict, raw: Any) -> Iterable[tuple[dict, Any, Any]]:
    """Yield typed views; consumers own validation, allocation and
    application."""
    import torch

    validate_directory(metadata, max_bytes=raw.numel())
    if raw.dtype != torch.uint8 or raw.ndim != 1 or not raw.is_contiguous() or raw.numel() != metadata["bytes"]:
        raise ValueError("Wire payload must be a flat uint8 tensor of the declared size")
    for entry in metadata["tensors"]:
        width = _DTYPES[entry["dtype"]]
        offset, count, value_offset = entry["offset"], entry["count"], entry["value_offset"]
        values = raw[value_offset : value_offset + count * width].view(getattr(torch, entry["dtype"]))
        positions = None if entry["full"] else raw[offset : offset + count * 4].view(torch.int32)
        yield entry, positions, values
