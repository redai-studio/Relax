# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Lossless bitset conversion, including unaligned rows and reserved tail
slots."""

import pytest
import torch

from relax.models.qwen3_8.ops.qsa_mixed_ops.bits_to_ids import bits_to_ids


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA and Triton")
@pytest.mark.parametrize("position_dtype", [torch.int32, torch.int64])
def test_bits_to_ids_preserves_sparse_blocks_and_empty_rows(position_dtype):
    # Five-byte rows exercise byte strides that are not divisible by four.
    bits = torch.tensor(
        [[0, 0, 0, 0, 0], [3, 0, 0, 0, 0], [129, 0, 0, 0, 0], [2, 1, 0, 0, 0], [1, 0, 0, 128, 1]],
        dtype=torch.uint8,
        device="cuda",
    )
    positions = torch.tensor([0, 7, 31, 35, 131], dtype=position_dtype, device="cuda")
    expected = torch.tensor(
        [[-1, -1, -1, -1], [0, 1, -1, -1], [0, 7, -1, -1], [1, 8, -1, -1], [0, 31, 32, -1]],
        dtype=torch.int32,
        device="cuda",
    )
    torch.testing.assert_close(bits_to_ids(bits, positions, 4, 4), expected, rtol=0, atol=0)
    assert bits_to_ids(bits[:0], positions[:0], 4, 4).shape == (0, 4)
