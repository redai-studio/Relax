# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Packed QSA coordinates shared by CP selection and CuTe attention."""

from __future__ import annotations

from typing import NamedTuple

import torch
from torch import Tensor


def build_zigzag_cp_local_positions(total: int, size: int, rank: int, device: torch.device) -> Tensor:
    if size < 1 or not 0 <= rank < size or total % (2 * size):
        raise ValueError("Zigzag CP requires a valid rank and a length divisible by 2 * CP")
    half = total // (2 * size)
    offsets = torch.arange(half, device=device, dtype=torch.long)
    return torch.cat((offsets + rank * half, offsets + total - (rank + 1) * half))


def validate_bounds(bounds: tuple[int, ...], total: int, cp_size: int = 1, layout: str = "per_document") -> None:
    if (
        not isinstance(bounds, tuple)
        or len(bounds) < 2
        or bounds[0] != 0
        or bounds[-1] != total
        or any(type(value) is not int for value in bounds)
        or any(left > right for left, right in zip(bounds, bounds[1:]))
    ):
        raise ValueError("QSA requires exact monotonic producer CPU document boundaries")
    if layout not in ("per_document", "per_sequence") or cp_size < 1:
        raise ValueError("QSA requires an explicit zigzag packing layout and positive CP size")
    lengths = [right - left for left, right in zip(bounds, bounds[1:])]
    if cp_size > 1 and any(length % (2 * cp_size) for length in (lengths if layout == "per_document" else [total])):
        raise ValueError("QSA zigzag packing must be divisible by 2 * CP")


class BlockMetadata(NamedTuple):
    indices: Tensor
    counts: Tensor
    bases: Tensor
    ends: Tensor
    query_positions: Tensor


def block_metadata(ids: Tensor, bounds: tuple[int, ...], query_positions: Tensor | None = None) -> BlockMetadata:
    """Translate document-relative complete IDs plus causal tails to global
    blocks.

    Coordinates are computed on the input device. CPU boundaries originate in
    the data producer; no device tensor is read back to determine allocation
    sizes. Empty documents and lengths not divisible by four retain their own
    block space.
    """
    if ids.ndim != 3 or ids.dtype not in (torch.int32, torch.int64):
        raise ValueError("CuTe QSA requires integer [batch, local_queries, budget_blocks] IDs")
    batch, local_length, budget = ids.shape
    total = bounds[-1]
    validate_bounds(bounds, total)
    device = ids.device
    if query_positions is None:
        if local_length != total:
            raise ValueError("Local-query QSA requires explicit global query positions")
        query_positions = torch.arange(total, device=device, dtype=torch.long)
    if query_positions.shape != (local_length,) or query_positions.device != device:
        raise ValueError("QSA query positions must match the local query rows and device")
    cu = torch.tensor(bounds, dtype=torch.long, device=device)
    docs = torch.searchsorted(cu[1:], query_positions, right=True)
    positions = query_positions - cu[docs]
    # One extra block per document encodes the incomplete tail, including an
    # empty sentinel when the document ends exactly at a block boundary.
    counts_cpu = [(right - left) // 4 + 1 for left, right in zip(bounds, bounds[1:])]
    offsets_cpu = [0]
    for count in counts_cpu:
        offsets_cpu.append(offsets_cpu[-1] + count)
    offsets = torch.tensor(offsets_cpu, dtype=torch.long, device=device)
    slots = torch.arange(offsets_cpu[-1], device=device)
    block_docs = torch.searchsorted(offsets[1:], slots, right=True)
    bases = cu[block_docs] + 4 * (slots - offsets[block_docs])
    ends = torch.minimum(bases + 4, cu[block_docs + 1])
    visible = (positions + 1) // 4
    valid = (ids >= 0) & (ids < visible[None, :, None])
    # Upstream metadata consumes the first `counts` entries. Stable compaction
    # also handles arbitrary invalid slots without changing valid-ID order.
    order = torch.argsort((~valid).to(torch.int8), dim=-1, stable=True)
    compact = ids.gather(-1, order)
    compact = torch.where(valid.gather(-1, order), compact, 0)
    own = visible[None, :, None].expand(batch, -1, -1)
    indices = torch.cat((own, compact), dim=-1) + offsets[docs][None, :, None]
    counts = valid.sum(-1) + 1
    return BlockMetadata(
        indices.to(torch.int32), counts.to(torch.int32), bases.int(), ends.int(), query_positions.long()
    )
