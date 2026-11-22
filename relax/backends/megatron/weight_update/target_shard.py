# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Map exact post-Bridge expert changes to four physical TP lanes."""

from __future__ import annotations

from collections import defaultdict

import torch

from relax.distributed.weight_sync.codec import encode
from relax.distributed.weight_sync.target_shard import TP_SIZE, split_axis


def partition_owner_entries(slots, entries, qualified: set[int]):
    """Partition coordinates on their source owner, coalescing work by
    geometry.

    Each input already has strictly increasing HF positions. Stable grouping
    preserves that ordering inside each destination slice, including NaN bits.
    """
    grouped = defaultdict(list)
    for (index, _, _), entry in zip(slots, entries, strict=True):
        axis = split_axis(entry.name, entry.shape)
        if index in qualified and not entry.full:
            grouped[(axis, entry.shape)].append((index, entry))
    results = {}
    for (axis, (rows, cols)), group in grouped.items():
        sizes = [item.positions.numel() for _, item in group]
        pos = torch.cat([item.positions for _, item in group])
        words = torch.cat([item.values.view(torch.int16) for _, item in group])
        if pos.numel():
            if axis == 0:
                elements = (rows // TP_SIZE) * cols
                lanes, local = pos // elements, pos % elements
            else:
                width = cols // TP_SIZE
                lanes = (pos % cols) // width
                local = (pos // cols) * width + pos % width
            entry_ids = torch.repeat_interleave(
                torch.arange(len(group), device=pos.device, dtype=torch.int32),
                torch.tensor(sizes, device=pos.device),
                output_size=pos.numel(),
            )
            keys = entry_ids * TP_SIZE + lanes
            order = torch.argsort(keys, stable=True)
            counts = torch.bincount(keys.long(), minlength=len(group) * TP_SIZE).reshape(-1, TP_SIZE).cpu().tolist()
            pos = local.index_select(0, order)
            words = words.index_select(0, order)
        else:
            counts = [[0] * TP_SIZE for _ in group]
        offset = 0
        for (index, _entry), size, per_lane in zip(group, sizes, counts, strict=True):
            results[index] = (
                per_lane,
                pos.narrow(0, offset, size),
                words.narrow(0, offset, size).view(torch.bfloat16),
            )
            offset += size
    return results


def encode_frame(chunk, max_bytes: int):
    marked = [i for i, entry in enumerate(chunk) if entry.target_lane_counts is not None]
    if not marked:
        return encode(((e.name, e.shape, e.dtype, e.positions, e.values) for e in chunk), max_bytes=max_bytes)
    directories, slabs = [], []
    offsets = [0] * len(chunk)
    for tp in range(TP_SIZE):
        items = []
        for i, entry in enumerate(chunk):
            positions, values = entry.positions, entry.values
            if entry.target_lane_counts is not None:
                size = entry.target_lane_counts[tp]
                positions = positions.narrow(0, offsets[i], size)
                values = values.narrow(0, offsets[i], size)
                offsets[i] += size
            items.append((entry.name, entry.shape, entry.dtype, positions, values))
        directory, slab = encode(items, max_bytes=max_bytes)
        directory["target_shard"] = dict(format=1, tp_rank=tp, tp_size=TP_SIZE, entries=marked)
        directories.append(directory)
        slabs.append(slab)
    return directories, slabs
