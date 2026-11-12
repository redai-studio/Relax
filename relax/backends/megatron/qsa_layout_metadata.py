# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Bind producer-known physical QSA boundaries across prepack and device
copies."""

from __future__ import annotations

from bisect import bisect_left
from typing import Any

from relax.utils.env import Envs


def physical_cu_field(packed: Any) -> str:
    if getattr(packed, "cu_seqlens_q_padded", None) is not None:
        return "cu_seqlens_q_padded"
    return "cu_seqlens_q"


def layout_binding(packed: Any, record: tuple[Any, ...]) -> tuple[Any, ...] | None:
    """Read metadata only; inference tensors without a version fall back."""
    if len(record) != 3 or record[0] != 1 or record[1] != physical_cu_field(packed):
        return None
    tensor = getattr(packed, record[1], None)
    if tensor is None:
        return None
    try:
        return (id(tensor), tensor._version, tuple(tensor.shape), str(tensor.dtype), str(tensor.device), record)
    except RuntimeError:
        return None


def valid_physical_layout(packed: Any) -> tuple[Any, ...] | None:
    record = getattr(packed, "qsa_physical_layout_cpu", None)
    if not isinstance(record, tuple):
        return None
    binding = layout_binding(packed, record)
    if binding is None or binding != getattr(packed, "qsa_physical_layout_binding", None):
        return None
    return record


def bind_physical_layout(packed: Any, boundaries: tuple[int, ...]) -> None:
    """The caller owns the exact CPU values used to construct the physical
    cu."""
    field = physical_cu_field(packed)
    tensor = getattr(packed, field)
    assert len(boundaries) >= 2 and boundaries[0] == 0
    assert all(isinstance(value, int) and value >= 0 for value in boundaries)
    assert all(left <= right for left, right in zip(boundaries, boundaries[1:]))
    assert tuple(tensor.shape) == (len(boundaries),)
    record = (1, field, boundaries)
    packed.qsa_physical_layout_cpu = record
    packed.qsa_physical_layout_binding = layout_binding(packed, record)


def rebind_mapped_layout(mapped: Any, record: tuple[Any, ...] | None) -> None:
    """Preserve a valid producer contract across pinning or asynchronous
    H2D."""
    for name in ("qsa_physical_layout_cpu", "qsa_physical_layout_binding", "_qsa_cpu_layout_cache"):
        if hasattr(mapped, name):
            delattr(mapped, name)
    if record is not None:
        bind_physical_layout(mapped, record[2])


def get_cpu_layout(packed: Any, batch_size: int, seq_len: int) -> tuple[int, int, tuple[int, ...]] | None:
    """Return exact n_docs/max_doc_len/offsets without accelerator scalar
    reads."""
    if not Envs.RELAX_QWEN38_QSA_CPU_LAYOUT or batch_size <= 0 or seq_len <= 0:
        return None
    if packed is None or getattr(packed, "qkv_format", None) != "thd":
        return batch_size, seq_len, tuple(index * seq_len for index in range(batch_size + 1))
    record = valid_physical_layout(packed)
    if record is None or batch_size != 1:
        return None
    cu = record[2]
    tensor = getattr(packed, record[1])
    if (
        not isinstance(cu, tuple)
        or len(cu) < 2
        or cu[0] != 0
        or cu[-1] != seq_len
        or tuple(tensor.shape) != (len(cu),)
        or not all(type(value) is int and value >= 0 for value in cu)
        or any(left > right for left, right in zip(cu, cu[1:]))
    ):
        return None
    n_docs = bisect_left(cu[:-1], seq_len)
    return n_docs, max(right - left for left, right in zip(cu, cu[1:])), cu[: n_docs + 1]
