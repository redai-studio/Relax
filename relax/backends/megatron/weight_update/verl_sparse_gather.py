# Copyright (c) 2026 Relax Authors. All Rights Reserved.
# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Bitwise shard differences and batched sparse gathering from verl
delta_sharded."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass
from itertools import groupby

import torch
import torch.distributed as dist


_DTYPE_INT = {1: torch.uint8, 2: torch.int16, 4: torch.int32, 8: torch.int64}


def shard_delta_indices(
    local_new: torch.Tensor,
    local_snap: torch.Tensor,
    offset: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Byte-diff a local shard against its snapshot; return (global_positions,
    values).

    Positions are int64 indices into the *full flattened parameter* (offset +
    local index). Dtype-agnostic, bytewise (view-as-int), no arithmetic --
    matches ``bytewise_diff_mask``.
    """
    es = local_new.element_size()
    int_dtype = _DTYPE_INT.get(es)
    if int_dtype is None:
        raise ValueError(f"unsupported element size {es}")
    mask = local_new.view(int_dtype) != local_snap.view(int_dtype)
    local_idx = mask.nonzero(as_tuple=False).view(-1)
    values = local_new[local_idx]
    global_idx = local_idx.to(torch.int64) + offset
    return global_idx, values


@dataclass
class SparseGatherTraffic:
    count_exchanges: int = 0
    local_count_input_bytes: int = 0
    local_count_received_bytes: int = 0
    payload_rounds: int = 0
    local_gather_index_bytes: int = 0
    local_gather_value_bytes: int = 0
    local_padding_index_bytes: int = 0
    local_padding_value_bytes: int = 0
    peak_root_padded_bytes: int = 0


def _gather_rounds(counts: list[list[int]], capacity: int) -> Iterator[tuple[list, list]]:
    """Pack slot-major, rank-major pieces up to the per-rank capacity,
    including within a single slot."""
    used = [0] * len(counts)
    pieces = []
    for slot, sizes in enumerate(zip(*counts, strict=True)):
        for rank, size in enumerate(sizes):
            while size:
                if used[rank] == capacity:
                    yield pieces, used
                    pieces, used = [], [0] * len(counts)
                amount = min(size, capacity - used[rank])
                pieces.append((slot, rank, used[rank], amount))
                used[rank] += amount
                size -= amount
    if pieces:
        yield pieces, used


def gather_slot_entries_to_rank0(
    idx_concat: torch.Tensor,
    val_concat: torch.Tensor,
    counts: torch.Tensor,
    group: dist.ProcessGroup | None = None,
    max_round_bytes: int | None = None,
    *,
    consume: Callable[[int, torch.Tensor, torch.Tensor], None] | None = None,
    traffic: SparseGatherTraffic | None = None,
) -> list | None:
    """Gather final HF coordinates and typed values in deterministic bounded
    rounds.

    Exchange slot counts once. The total padded receive slabs for both data
    gathers fit the root's round budget. A slot may span rounds. The optional
    consumer receives those pieces in their original slot and rank order,
    without allocating a full oversized slot on the publisher.
    """
    rank = dist.get_rank(group)
    world = dist.get_world_size(group)
    dst = dist.get_global_rank(group, 0) if group is not None else 0
    counts = counts.to(idx_concat.device)
    counts_all = [torch.empty_like(counts) for _ in range(world)]
    dist.all_gather(counts_all, counts, group=group)
    counts_cpu = torch.stack(counts_all).cpu().tolist()
    if traffic is not None:
        traffic.count_exchanges += 1
        traffic.local_count_input_bytes += counts.nbytes
        traffic.local_count_received_bytes += counts.nbytes * (world - 1)

    width = idx_concat.element_size() + val_concat.element_size()
    longest = max(map(sum, counts_cpu))
    capacity = longest if max_round_bytes is None else max_round_bytes // (world * width)
    if longest and capacity < 1:
        raise ValueError("Sparse gather budget cannot hold one index and value per process-group rank")

    result = [[] for _ in range(counts.numel())] if rank == 0 and consume is None else None
    local_start = 0
    for pieces, sizes in _gather_rounds(counts_cpu, capacity):
        padded = max(sizes)
        size = sizes[rank]
        if traffic is not None:
            traffic.payload_rounds += 1
            traffic.local_gather_index_bytes += size * idx_concat.element_size()
            traffic.local_gather_value_bytes += size * val_concat.element_size()
            traffic.local_padding_index_bytes += (padded - size) * idx_concat.element_size()
            traffic.local_padding_value_bytes += (padded - size) * val_concat.element_size()
            traffic.peak_root_padded_bytes = max(traffic.peak_root_padded_bytes, padded * width * world)

        packed = []
        for data in (idx_concat, val_concat):
            local = data.narrow(0, local_start, size)
            if size < padded:
                local = torch.nn.functional.pad(local, (0, padded - size))
            outputs = [torch.empty_like(local) for _ in range(world)] if rank == 0 else None
            dist.gather(local, outputs, dst=dst, group=group)
            if rank == 0:
                # Copy only useful words into a compact slab. Sparse views
                # crossing a wire frame must not retain padded receive buffers.
                packed.append(torch.cat([outputs[r].narrow(0, start, n) for _, r, start, n in pieces]))
            del outputs
        local_start += size
        if rank == 0:
            offset = 0
            for slot, parts in groupby(pieces, key=lambda p: p[0]):
                size = sum(p[3] for p in parts)
                idx, values = (tensor.narrow(0, offset, size) for tensor in packed)
                if consume is not None:
                    consume(slot, idx, values)
                else:
                    result[slot].append((idx, values))
                offset += size

    if result is None:
        return None
    merged = []
    for parts in result:
        if not parts:
            merged.append((idx_concat[:0], val_concat[:0]))
        elif len(parts) == 1:
            merged.append(parts[0])
        else:
            merged.append(tuple(torch.cat(group) for group in zip(*parts, strict=True)))
    return merged
