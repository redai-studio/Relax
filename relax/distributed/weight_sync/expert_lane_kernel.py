# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Stable compaction of sparse HF coordinates to one native expert TP lane."""

from itertools import accumulate

import torch
import triton
import triton.language as tl


@triton.jit
def _lane_offsets(metadata, owners, indices, BLOCK: tl.constexpr):
    tile = tl.program_id(0)
    slot = tl.load(owners + tile)
    fields = metadata + slot * 7
    start = tl.load(fields)
    length = tl.load(fields + 1)
    axis = tl.load(fields + 2)
    low = tl.load(fields + 3)
    high = tl.load(fields + 4)
    columns = tl.load(fields + 5)
    first_tile = tl.load(fields + 6)
    offset = (tile - first_tile) * BLOCK + tl.arange(0, BLOCK)
    live = offset < length
    source = start + offset
    position = tl.load(indices + source, mask=live, other=0)
    # Match torch floor division even if a signed coordinate is outside the source shape.
    negative = (position < 0).to(tl.int32)
    row = (position + negative) // columns - negative
    column = position - row * columns
    coordinate = tl.where(axis == 0, row, column)
    local = tl.where(axis == 0, position - low * columns, row * (high - low) + column - low)
    return source, local, live & (coordinate >= low) & (coordinate < high)


@triton.jit
def _count(metadata, owners, indices, counts, BLOCK: tl.constexpr):
    _, _, keep = _lane_offsets(metadata, owners, indices, BLOCK)
    tl.store(counts + tl.program_id(0), tl.sum(keep.to(tl.int32), axis=0))


@triton.jit
def _scatter(metadata, owners, indices, words, prefix, locations, output, BLOCK: tl.constexpr):
    source, local, keep = _lane_offsets(metadata, owners, indices, BLOCK)
    dest = tl.load(prefix + tl.program_id(0)) + tl.cumsum(keep.to(tl.int32), axis=0) - 1
    word = tl.load(words + source, mask=keep, other=0)
    tl.store(locations + dest, local, mask=keep)
    tl.store(output + dest, word, mask=keep)


def compact_lane(
    indices: torch.Tensor, values: torch.Tensor, lengths: list[int], descriptors: list[tuple]
) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
    """Retain source order, including repeated coordinates and original value
    bits."""
    block = 256
    tile_ends = list(accumulate(triton.cdiv(n, block) for n in lengths))
    starts = accumulate(lengths[:-1], initial=0)
    tile_starts = [0, *tile_ends[:-1]]
    shape = [
        (start, length, *desc, tile)
        for start, length, desc, tile in zip(starts, lengths, descriptors, tile_starts, strict=True)
    ]
    device = indices.device
    metadata = torch.tensor(shape, device=device, dtype=torch.int32)
    ends = torch.tensor(tile_ends, device=device, dtype=torch.int32)
    tiles = tile_ends[-1]
    owners = torch.searchsorted(
        ends, torch.arange(tiles, dtype=torch.int32, device=device), right=True, out_int32=True
    )
    counts = torch.empty(tiles, dtype=torch.int32, device=device)
    prefix = torch.empty(tiles + 1, dtype=torch.int32, device=device)
    _count[(tiles,)](metadata, owners, indices, counts, BLOCK=block)
    prefix[:1].zero_()
    torch.cumsum(counts, 0, dtype=torch.int32, out=prefix[1:])
    # The existing wire directory needs one host count per HF input. Read all counts together.
    slot_ends = prefix.index_select(0, ends).tolist()
    locations = indices.new_empty(slot_ends[-1])
    words = values.view({2: torch.int16, 4: torch.int32, 8: torch.int64}[values.element_size()])
    output = words.new_empty(slot_ends[-1])
    if slot_ends[-1]:
        _scatter[(tiles,)](metadata, owners, indices, words, prefix, locations, output, BLOCK=block)
    return locations, output.view(values.dtype), slot_ends
