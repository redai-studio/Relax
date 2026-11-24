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

"""Batched sparse gather adapted from verl 78105206 for Megatron TP/EP
groups."""

from __future__ import annotations

from bisect import bisect_left
from collections import deque
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from itertools import accumulate

import torch
import torch.distributed as dist


# These scalar types can be installed through the Relax weight-sync wire.
_VALUE_DTYPES = (
    torch.bool,
    torch.uint8,
    torch.int8,
    torch.int16,
    torch.float16,
    torch.bfloat16,
    torch.int32,
    torch.float32,
    torch.int64,
    torch.float64,
)
_WORDS = {1: torch.uint8, 2: torch.int16, 4: torch.int32, 8: torch.int64}
_Slot = tuple[torch.Tensor, torch.Tensor]


@dataclass(frozen=True)
class SparseGatherRound:
    """Group logical inputs in bytes, counting each contributor once, including
    the root.

    Payload index/value bytes exclude padding. All-gather replication, receive
    buffers and backend traffic are excluded.
    """

    operation: str
    group_ranks: tuple[int, ...]
    entries_per_rank: tuple[int, ...] = ()
    control_bytes: int = 0
    count_bytes: int = 0
    index_bytes: int = 0
    value_bytes: int = 0
    padding_bytes: int = 0


_Observer = Callable[[SparseGatherRound], None]


def _all_gather_rows(tensor: torch.Tensor, group: dist.ProcessGroup) -> list[list[int]]:
    rows = tensor.new_empty((dist.get_world_size(group), tensor.numel()))
    dist.all_gather_into_tensor(rows.view(-1), tensor, group=group)
    # Peers must derive the same variable packet sizes before the next collective.
    return rows.cpu().tolist()


def _slot_layout(
    indices: torch.Tensor,
    values: torch.Tensor,
    counts: torch.Tensor,
    budget: int,
    device: torch.device,
    ranks: tuple[int, ...],
    group: dist.ProcessGroup,
    observe: _Observer | None,
) -> list[list[int]]:
    # A fixed-size header keeps a malformed rank out of variable-shape collectives.
    valid = (
        isinstance(indices, torch.Tensor)
        and isinstance(values, torch.Tensor)
        and isinstance(counts, torch.Tensor)
        and indices.layout == values.layout == counts.layout == torch.strided
        and indices.ndim == values.ndim == counts.ndim == 1
        and indices.dtype == torch.int32
        and counts.dtype == torch.int64
        and values.dtype in _VALUE_DTYPES
        and not values.is_neg()
        and not values.is_conj()
        and indices.device == values.device == device
        and counts.device in (device, torch.device("cpu"))
        and indices.numel() == values.numel()
        and type(budget) is int
        and 0 < budget <= torch.iinfo(torch.int64).max
    )
    fields = (counts.numel(), _VALUE_DTYPES.index(values.dtype), budget, indices.numel()) if valid else (-1,) * 4
    header = torch.tensor(fields, dtype=torch.int64, device=device)
    headers = _all_gather_rows(header, group)
    world = len(ranks)
    if observe is not None:
        observe(SparseGatherRound("all_gather_header", ranks, control_bytes=header.nbytes * world))
    invalid = [r for r, row in zip(ranks, headers, strict=True) if row[0] < 0]
    if invalid:
        raise ValueError(f"Invalid sparse gather input, device, dtype or budget on global ranks {invalid}")
    if any(row[:3] != headers[0][:3] for row in headers):
        raise ValueError("Sparse gather members disagree on slot count, value dtype or byte budget")

    rows = [[] for _ in ranks]
    # Bound the device count matrix as well as the data packets. One count is
    # always exchanged when K > 0, even with a very small payload budget.
    step = max(1, budget // ((world + 1) * counts.element_size()))
    for start in range(0, counts.numel(), step):
        local = counts[start : start + step].to(device=device).contiguous()
        peers = _all_gather_rows(local, group)
        if observe is not None:
            observe(SparseGatherRound("all_gather_counts", ranks, count_bytes=local.nbytes * world))
        for row, part in zip(rows, peers, strict=True):
            row.extend(part)
    for r, row, (_, _, _, received_n) in zip(ranks, rows, headers, strict=True):
        if any(n < 0 for n in row) or sum(row) != received_n:
            raise ValueError(f"Sparse slot counts on global rank {r} do not partition its {received_n} entries")
    return rows


def _check_local_order(indices: torch.Tensor, counts: list[int], budget: int, status: torch.Tensor) -> None:
    """Check all slots in one owner's flattened input before the ready vote."""
    if not indices.numel():
        return
    invalid = indices.amin() < 0
    if max(counts) > 1:
        ends = list(accumulate(n for n in counts if n))
        # One comparison byte and at most one int64 slot boundary per entry.
        step = max(1, (budget - 2) // 9)
        lower = 0
        for start in range(1, indices.numel(), step):
            end = min(start + step, indices.numel())
            disorder = indices[start:end] <= indices[start - 1 : end - 1]
            upper = bisect_left(ends, end, lo=lower)
            if upper > lower:
                # Consecutive slots may restart at the same (or a smaller) coordinate.
                boundaries = indices.new_tensor([n - start for n in ends[lower:upper]], dtype=torch.int64)
                disorder.index_fill_(0, boundaries, False)
                del boundaries
            lower = upper
            invalid.logical_or_(disorder.any())
            # Retire this mask before allocating the next bounded packet.
            del disorder
    status.copy_(invalid)


def _outputs(
    indices: torch.Tensor, values: torch.Tensor, rows: list[list[int]], single_owner: bool
) -> tuple[list[_Slot], list[deque[_Slot]]]:
    destinations: list[deque[_Slot]] = [deque() for _ in rows]
    word_type = _WORDS[values.element_size()]
    if single_owner:
        # Each peer's flat receive bank already contains complete slots. Copy
        # each packet once and hand out views in the caller's directory order.
        empty = (indices.new_empty(0), values.new_empty(0))
        slots = [empty] * len(rows[0])
        for counts, queue in zip(rows, destinations, strict=True):
            slot_ids = [slot for slot, n in enumerate(counts) if n]
            if slot_ids:
                sizes = [counts[slot] for slot in slot_ids]
                total = sum(sizes)
                pos, val = indices.new_empty(total), values.new_empty(total)
                queue.append((pos, val.view(word_type)))
                for slot, i, v in zip(slot_ids, pos.split(sizes), val.split(sizes), strict=True):
                    slots[slot] = (i, v)
        return slots, destinations
    slots = []
    for counts in zip(*rows, strict=True):
        n = sum(counts)
        pos, val = indices.new_empty(n), values.new_empty(n)
        slots.append((pos, val))
        words = val.view(word_type)
        offset = 0
        for queue, count in zip(destinations, counts, strict=True):
            if count:
                queue.append((pos.narrow(0, offset, count), words.narrow(0, offset, count)))
                offset += count
    return slots, destinations


def _unpack(destinations: deque[_Slot], indices: torch.Tensor, values: torch.Tensor, count: int) -> None:
    offset = 0
    while offset < count:
        out_idx, out_val = destinations.popleft()
        n = min(count - offset, out_idx.numel())
        out_idx[:n].copy_(indices[offset : offset + n])
        out_val[:n].copy_(values[offset : offset + n])
        if n < out_idx.numel():
            destinations.appendleft((out_idx[n:], out_val[n:]))
        offset += n


def _gather_payloads(
    indices: torch.Tensor,
    values: torch.Tensor,
    rows: list[list[int]],
    capacity: int,
    budget: int,
    ranks: tuple[int, ...],
    group: dist.ProcessGroup,
    status: torch.Tensor,
    observe: _Observer | None,
) -> tuple[list[_Slot] | None, Exception | None, bool]:
    rank, world = dist.get_rank(group), len(ranks)
    totals = [sum(row) for row in rows]
    longest = max(totals)
    element_bytes = indices.element_size() + values.element_size()

    single_owner = all(sum(n > 0 for n in col) <= 1 for col in zip(*rows, strict=True))
    error = None
    slots = destinations = None
    try:
        if single_owner:
            _check_local_order(indices, rows[rank], budget, status)
        if rank == 0:
            slots, destinations = _outputs(indices, values, rows, single_owner)
        # A same-dtype BOOL copy can normalize storage bytes. Move integer
        # words throughout packing, transport and sorting for every dtype.
        values = values.view(_WORDS[values.element_size()])
        send_idx, send_val = indices.new_zeros(capacity), values.new_zeros(capacity)
        recv_idx = [indices.new_empty(capacity) for _ in ranks] if rank == 0 else None
        recv_val = [values.new_empty(capacity) for _ in ranks] if rank == 0 else None
    except Exception as exc:
        error = exc
        status.fill_(2)
    # Reuse the allocation vote for the order check. A 1 requires root
    # sorting and validation; a 2 aborts before the variable-size collectives.
    dist.all_reduce(status, op=dist.ReduceOp.MAX, group=group)
    if observe is not None:
        observe(SparseGatherRound("all_reduce_ready", ranks, control_bytes=status.nbytes * world))
    ready = int(status.cpu())
    if ready == 2:
        raise RuntimeError("A sparse gather member could not validate its input or allocate its buffers") from error
    needs_sort = not single_owner or ready == 1
    if ready == 1:
        status.zero_()

    for start in range(0, longest, max(capacity, 1)):
        width = min(capacity, longest - start)
        lengths = tuple(max(0, min(width, total - start)) for total in totals)
        n = lengths[rank]
        idx, val = send_idx[:width], send_val[:width]
        if error is None:
            try:
                idx[:n].copy_(indices[start : start + n])
                val[:n].copy_(values[start : start + n])
                idx[n:].zero_()
                val[n:].zero_()
            except Exception as exc:
                # All peers must complete the agreed rounds, even if a local
                # source view fails to pack. The final vote rejects this batch.
                error = exc
        idx_list = [buf[:width] for buf in recv_idx] if rank == 0 else None
        # Gloo and NCCL move the same storage bytes, including noncanonical NaNs.
        byte_values = val.view(torch.uint8)
        byte_list = [buf[:width].view(torch.uint8) for buf in recv_val] if rank == 0 else None
        dist.gather(idx, idx_list, dst=ranks[0], group=group)
        dist.gather(byte_values, byte_list, dst=ranks[0], group=group)
        if observe is not None:
            used = sum(lengths)
            observe(
                SparseGatherRound(
                    "gather_payload",
                    ranks,
                    lengths,
                    index_bytes=used * idx.element_size(),
                    value_bytes=used * val.element_size(),
                    padding_bytes=world * (idx.nbytes + byte_values.nbytes) - used * element_bytes,
                )
            )
        if destinations is not None and error is None:
            try:
                for dest, received_idx, received_val, count in zip(
                    destinations, idx_list, recv_val, lengths, strict=True
                ):
                    _unpack(dest, received_idx, received_val, count)
            except Exception as exc:
                # Finish the agreed collective sequence; peers await the final status.
                error = exc
    return slots, error, needs_sort


def _merge_pairs(n: int) -> Iterator[tuple[int, int]]:
    """Batcher odd-even mergesort comparators on n blocks, always ascending."""
    span = 1
    while span < n:
        gap = span
        while gap:
            for start in range(gap % span, n - gap, 2 * gap):
                for left in range(start, min(start + gap, n - gap)):
                    right = left + gap
                    if left // (2 * span) == right // (2 * span):
                        yield left, right
            gap //= 2
        span *= 2


def _sort_blocks(left: _Slot, right: _Slot | None = None) -> None:
    indices, values = left
    if right is not None:
        indices = torch.cat((indices, right[0]))
        values = torch.cat((values, right[1]))
    sorted_indices, order = indices.sort()
    sorted_values = values.index_select(0, order)
    split = left[0].numel()
    left[0].copy_(sorted_indices[:split])
    left[1].copy_(sorted_values[:split])
    if right is not None:
        right[0].copy_(sorted_indices[split:])
        right[1].copy_(sorted_values[split:])


def _order_slots(slots: list[_Slot], budget: int, status: torch.Tensor) -> None:
    for positions, values in slots:
        size = positions.numel()
        if not size:
            continue
        values = values.view(_WORDS[values.element_size()])
        # Two blocks can be joined and sorted together. Count their joined input,
        # sorted output and int64 permutation; keep even a single long slot bounded.
        pair_bytes = 2 * (positions.element_size() + values.element_size()) + 8
        block_size = max(1, budget // (2 * pair_bytes))
        if size <= 2 * block_size:
            if size > 1:
                _sort_blocks((positions, values))
        else:

            def block(i: int) -> _Slot:
                start = i * block_size
                return positions[start : start + block_size], values[start : start + block_size]

            nblocks = (size + block_size - 1) // block_size
            if block_size > 1:
                for i in range(nblocks):
                    _sort_blocks(block(i))
            for a, b in _merge_pairs(nblocks):
                _sort_blocks(block(a), block(b))
        status.bitwise_or_((positions[0] < 0).to(status.dtype))
        for start in range(1, size, 2 * block_size):
            end = min(start + 2 * block_size, size)
            duplicate = (positions[start:end] == positions[start - 1 : end - 1]).any()
            status.bitwise_or_(duplicate.to(status.dtype))


@torch.no_grad()
def gather_slot_entries_to_rank0(
    idx_concat: torch.Tensor,
    val_concat: torch.Tensor,
    counts: torch.Tensor,
    *,
    group: dist.ProcessGroup,
    max_round_bytes: int,
    observe: _Observer | None = None,
) -> list[_Slot] | None:
    """Gather flattened int32 HF coordinates and original values for K same-
    dtype slots.

    Counts are int64 on the host or the payload device. Every group member enters with
    the same ordered slots and budget; any member can contribute zero entries. The group
    root receives sorted, disjoint coordinates and members reject invalid inputs together.

    The budget bounds padded send plus receive storage on the root, excluding returned
    sparse tensors, group headers and backend-internal workspace. Count and sort chunks
    have a fixed minimum of one count and a two-entry compare. Only the root calls the
    optional observer with group totals, in collective order.
    """
    if group is None or dist.get_rank(group) < 0:
        raise ValueError("A live process group containing this rank is required")
    backend = dist.get_backend(group)
    if backend == dist.Backend.NCCL:
        device = group.bound_device_id or torch.device("cuda", torch.cuda.current_device())
    elif backend == dist.Backend.GLOO:
        device = torch.device("cpu")
    else:
        raise ValueError(f"Sparse gather requires a Gloo or NCCL process group, got {backend}")

    ranks = tuple(dist.get_process_group_ranks(group))
    is_root = dist.get_rank(group) == 0
    observer_error = None

    def record(event: SparseGatherRound) -> None:
        nonlocal observer_error
        if observer_error is None:
            try:
                observe(event)
            except Exception as exc:
                # Preserve the collective order even if the receipt sink fails.
                observer_error = exc

    audit = record if is_root and observe is not None else None
    rows = _slot_layout(idx_concat, val_concat, counts, max_round_bytes, device, ranks, group, audit)
    # One send buffer plus one receive buffer per group member on root.
    min_bytes = (len(ranks) + 1) * (idx_concat.element_size() + val_concat.element_size())
    longest = max(map(sum, rows))
    capacity = min(longest, max_round_bytes // min_bytes)
    if longest and not capacity:
        raise ValueError(f"Sparse gather needs at least {min_bytes} bytes for one padded entry")

    status = torch.zeros((), dtype=torch.int32, device=device)
    slots, error, needs_sort = _gather_payloads(
        idx_concat, val_concat, rows, capacity, max_round_bytes, ranks, group, status, audit
    )
    if needs_sort and slots is not None and error is None:
        try:
            _order_slots(slots, max_round_bytes, status)
        except Exception as exc:
            error = exc
    if audit is not None:
        audit(SparseGatherRound("all_reduce_result", ranks, control_bytes=status.nbytes * len(ranks)))
        if observer_error is not None:
            error = observer_error
    if error is not None:
        status.fill_(2)
    dist.all_reduce(status, op=dist.ReduceOp.MAX, group=group)
    code = int(status.cpu())
    if code == 1:
        raise ValueError("Sparse HF coordinates must be nonnegative and unique within each slot")
    if code:
        raise RuntimeError("Sparse gather failed") from error
    return slots
