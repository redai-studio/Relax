# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Sparse TP/EP collection through real Gloo and the Relax wire codec."""

from contextlib import ExitStack, nullcontext
from datetime import timedelta
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from relax.backends.megatron.weight_update import sparse_gather
from relax.backends.megatron.weight_update.sparse_gather import gather_slot_entries_to_rank0
from relax.distributed.weight_sync import codec


# BF16 bit patterns include both zeros, signaling and quiet NaNs and two payloads.
_WORDS = (0, -32768, 0x7F82, 0x7FC3, -60, 0x3F80)
_RANK_SLOTS = (
    ((8, 0), (), tuple(range(42, -1, -2)), (0,)),
    ((5,), (), tuple(range(37, 0, -2)), ()),
    ((), (), (), ()),
)


def _bits(slot: int, coordinate: int) -> int:
    return _WORDS[(coordinate + slot) % len(_WORDS)]


def _with_wire_trace(group, indices, values, counts, budget, observe):
    # Record the actual local input tensors of the real collectives. This
    # independent trace catches double-counted bytes and omitted padding.
    calls = []

    def trace(name, original):
        def call(*args, **kwargs):
            assert kwargs["group"] is group
            tensor = args[1] if name == "all_gather_into_tensor" else args[0]
            if name == "gather":
                assert (args[1] is not None) == (dist.get_rank(group) == 0)
                assert kwargs["dst"] == 1
                if args[1] is not None:
                    assert len(args[1]) == 3 and all(t.nbytes == tensor.nbytes for t in args[1])
            calls.append((name, tensor.nbytes))
            return original(*args, **kwargs)

        return call

    with ExitStack() as wrapped:
        for name in ("all_gather_into_tensor", "gather", "all_reduce"):
            wrapped.enter_context(patch.object(dist, name, trace(name, getattr(dist, name))))
        output = gather_slot_entries_to_rank0(
            indices, values, counts, group=group, max_round_bytes=budget, observe=observe
        )
    return output, calls


def _check_events(events, traces, budget):
    group_root = 1
    totals = [sum(map(len, slots)) for slots in _RANK_SLOTS]
    start = 0
    data_rounds = 0
    actual = iter(zip(*traces, strict=True))
    for event in events:
        assert event.group_ranks == (group_root, 2, 3)
        if event.operation == "gather_payload":
            index_inputs, value_inputs = next(actual), next(actual)
            assert all(name == "gather" for name, _ in (*index_inputs, *value_inputs))
            width = index_inputs[0][1] // 4
            assert all(size == width * 4 for _, size in index_inputs)
            assert all(size == width * 2 for _, size in value_inputs)
            expected = tuple(min(width, max(0, n - start)) for n in totals)
            assert event.entries_per_rank == expected
            assert (event.index_bytes, event.value_bytes) == (sum(expected) * 4, sum(expected) * 2)
            input_bytes = sum(size for _, size in (*index_inputs, *value_inputs))
            assert event.padding_bytes == input_bytes - event.index_bytes - event.value_bytes
            assert event.count_bytes == event.control_bytes == 0
            assert input_bytes + index_inputs[0][1] + value_inputs[0][1] <= budget
            data_rounds += 1
            start += width
        else:
            row = next(actual)
            name = {
                "all_gather_header": "all_gather_into_tensor",
                "all_gather_counts": "all_gather_into_tensor",
                "all_reduce_ready": "all_reduce",
                "all_reduce_result": "all_reduce",
            }[event.operation]
            assert all(op == name for op, _ in row)
            expected = sum(size for _, size in row)
            if event.operation == "all_gather_counts":
                assert event.count_bytes == expected and event.control_bytes == 0
            else:
                assert event.control_bytes == expected and event.count_bytes == 0
    assert next(actual, None) is None
    assert start == max(totals) and data_rounds > 1


def _check_layout(group):
    rank = dist.get_rank(group)
    slots = _RANK_SLOTS[rank]
    indices = torch.tensor([index for slot in slots for index in slot], dtype=torch.int32)
    words = [_bits(i, index) for i, slot in enumerate(slots) for index in slot]
    values = torch.tensor(words, dtype=torch.int16).view(torch.bfloat16)
    counts = torch.tensor(list(map(len, slots)), dtype=torch.int64)
    budget = 112  # Smaller than one HF slot, including its useful bytes alone.
    events = []
    output, calls = _with_wire_trace(group, indices, values, counts, budget, events.append)
    traces = [None] * 3 if rank == 0 else None
    dist.gather_object(calls, traces, dst=1, group=group)
    if rank:
        assert output is None and events == []
        return
    _check_events(events, traces, budget)
    directory, raw = codec.encode(
        ((str(i), (64,), torch.bfloat16, idx, val) for i, (idx, val) in enumerate(output)), max_bytes=1024
    )
    for i, (_, idx, val) in enumerate(codec.tensors(directory, raw)):
        coordinates = sorted(x for rank_slots in _RANK_SLOTS for x in rank_slots[i])
        assert idx.dtype == torch.int32 and idx.tolist() == coordinates
        assert val.dtype == torch.bfloat16 and val.view(torch.int16).tolist() == [_bits(i, x) for x in coordinates]


def _check_empty_and_invalid(group):
    rank = dist.get_rank(group)
    pos = torch.tensor([3] if rank == 2 else [], dtype=torch.int32)
    val = torch.tensor([-2147483648] if rank == 2 else [], dtype=torch.int32).view(torch.float32)
    count = torch.tensor([pos.numel()], dtype=torch.int64)

    def gather(idx=pos, values=val, counts=count, budget=96, observe=None):
        return gather_slot_entries_to_rank0(idx, values, counts, group=group, max_round_bytes=budget, observe=observe)

    # In each failure the following real collective must still be reachable.
    with pytest.raises(ValueError, match="partition"):
        gather(counts=count + int(rank == 1))
    with pytest.raises(ValueError, match="disagree"):
        gather(counts=count[:0] if rank == 0 else count)
    with pytest.raises(ValueError, match="disagree"):
        gather(values=val.view(torch.int32) if rank == 0 else val)
    with pytest.raises(ValueError, match="disagree"):
        gather(budget=120 if rank == 1 else 96)
    with pytest.raises(ValueError, match="Invalid sparse gather"):
        gather(idx=pos.to(torch.int64) if rank == 0 else pos)
    with pytest.raises(ValueError, match="at least"):
        gather(budget=8)
    with pytest.raises(ValueError, match="unique"):
        gather(torch.tensor([1], dtype=torch.int32), torch.tensor([rank], dtype=torch.int32), torch.tensor([1]))
    with pytest.raises(ValueError, match="nonnegative"):
        gather(idx=-pos)

    # Root setup can fail even when all source tensors and counts are valid.
    allocation = patch.object(sparse_gather, "_outputs", side_effect=MemoryError("root allocation"))
    with allocation if rank == 0 else nullcontext(), pytest.raises(RuntimeError, match="allocate"):
        gather()

    # If a non-root source fails while packing, its peers finish the active
    # rounds and reject the incomplete result in the same final vote.
    real_copy = torch.Tensor.copy_

    def fail_source_copy(destination, source, *args, **kwargs):
        if source.numel() and source.data_ptr() == pos.data_ptr():
            raise RuntimeError("source copy failed")
        return real_copy(destination, source, *args, **kwargs)

    packing = patch.object(torch.Tensor, "copy_", fail_source_copy)
    with packing if rank == 2 else nullcontext(), pytest.raises(RuntimeError, match="Sparse gather failed"):
        gather()

    def failed_observer(event):
        raise OSError("failed to record a receipt")

    with pytest.raises(RuntimeError, match="Sparse gather failed"):
        gather(observe=failed_observer)

    output = gather()
    if rank == 0:
        [(actual_pos, actual_val)] = output
        assert actual_pos.tolist() == [3]
        assert actual_val.view(torch.int32).tolist() == [-2147483648]
    else:
        assert output is None
    # Some same-dtype BOOL copies normalize these underlying storage bytes to 1.
    bits = torch.tensor([0xFF, 0, 0x80, 0, 0x02, 0], dtype=torch.uint8).view(torch.bool)
    output = gather(torch.tensor([2 - rank], dtype=torch.int32), bits[rank * 2 : rank * 2 + 1], torch.tensor([1]))
    if rank == 0:
        [(actual_pos, actual_val)] = output
        assert actual_pos.tolist() == [0, 1, 2]
        assert actual_val.view(torch.uint8).tolist() == [0x02, 0x80, 0xFF]
    for counts in (torch.zeros(2, dtype=torch.int64), torch.empty(0, dtype=torch.int64)):
        events = []
        output = gather(pos[:0], val[:0], counts, observe=events.append)
        if rank == 0:
            assert len(output) == counts.numel() and all(i.numel() == v.numel() == 0 for i, v in output)
            assert all(event.operation != "gather_payload" for event in events)
        else:
            assert output is None and not events


def _worker(rank, rendezvous):
    torch.set_num_threads(1)
    timeout = timedelta(seconds=20)
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=4, timeout=timeout)
    try:
        group = dist.new_group(ranks=[1, 2, 3], backend="gloo", timeout=timeout)
        if rank in (1, 2, 3):
            _check_layout(group)
            _check_empty_and_invalid(group)
            dist.destroy_process_group(group)
    finally:
        dist.destroy_process_group()


def test_sparse_gather_subgroup_bits_budget_and_collective_failures(tmp_path):
    mp.start_processes(_worker, args=((tmp_path / "gloo").as_uri(),), nprocs=4, join=True, start_method="fork")
