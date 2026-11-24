# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""The verl bytewise shard differ and batched sparse gather on CPU."""

from datetime import timedelta

import pytest
import torch
import torch.distributed as dist

from relax.backends.megatron.weight_update.verl_sparse_gather import (
    gather_slot_entries_to_rank0,
    shard_delta_indices,
)


@pytest.mark.parametrize(
    ("dtype", "word_dtype", "old_nan", "new_nan", "negative_zero"),
    [
        (torch.bfloat16, torch.int16, 0x7FC1, 0x7FC2, -(1 << 15)),
        (torch.float32, torch.int32, 0x7FC00001, 0x7FC00002, -(1 << 31)),
    ],
)
def test_verl_shard_delta_indices_compares_raw_words(dtype, word_dtype, old_nan, new_nan, negative_zero):
    previous = torch.tensor([old_nan, 0, old_nan, 17], dtype=word_dtype).view(dtype)
    current = torch.tensor([new_nan, negative_zero, old_nan, 17], dtype=word_dtype).view(dtype)
    indices, values = shard_delta_indices(current, previous, offset=11)
    assert indices.tolist() == [11, 12]
    assert values.view(word_dtype).tolist() == [new_nan, negative_zero]


def _gather_on_two_ranks(rank, path):
    dist.init_process_group("gloo", init_method=path, rank=rank, world_size=2, timeout=timedelta(seconds=15))
    group = dist.group.WORLD
    try:
        indices = torch.tensor([0, 5, 2] if rank == 0 else [8], dtype=torch.int32)
        raw_words = torch.tensor([0x7FC2, -32768, 0x3F80] if rank == 0 else [0x4000], dtype=torch.int16)
        counts = torch.tensor([2, 0, 1] if rank == 0 else [1, 0, 0], dtype=torch.int64)
        for budget in (None, 12):
            result = gather_slot_entries_to_rank0(indices, raw_words.view(torch.bfloat16), counts, group, budget)
            if rank == 1:
                assert result is None
                continue
            expected = [([0, 5, 8], [0x7FC2, -32768, 0x4000]), ([], []), ([2], [0x3F80])]
            for (idx, values), (wanted_idx, wanted_words) in zip(result, expected, strict=True):
                assert idx.dtype == torch.int32
                assert idx.tolist() == wanted_idx
                assert values.view(torch.int16).tolist() == wanted_words

        empty = gather_slot_entries_to_rank0(indices[:0], raw_words[:0].view(torch.bfloat16), counts * 0, group)
        if rank == 0:
            assert len(empty) == 3 and all(idx.numel() == val.numel() == 0 for idx, val in empty)
        else:
            assert empty is None
    finally:
        dist.destroy_process_group()


def test_verl_sparse_gather_preserves_raw_words_with_empty_slots_and_bounded_subrounds(tmp_path):
    torch.multiprocessing.start_processes(
        _gather_on_two_ranks, args=((tmp_path / "sparse-ranks").as_uri(),), nprocs=2, start_method="fork"
    )
