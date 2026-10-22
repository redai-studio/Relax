# Copyright (c) 2026 Relax Authors. All Rights Reserved.
# Copyright 2025-2026 RadixArk, Inc. (miles).
# Copyright 2025 Zhipu AI.
# SPDX-License-Identifier: Apache-2.0
# Adapted for Relax.

"""Sparse Flex block masks for selected-token causal attention."""

import torch
import triton
import triton.language as tl
from torch.nn.attention.flex_attention import BlockMask


@triton.jit
def _tile_flags(Bits, Starts, Partial, Full, tokens, blocks, WORDS: tl.constexpr, KEY_TILES: tl.constexpr):
    query = tl.program_id(0) * 128 + tl.arange(0, 128)[:, None]
    key_column = tl.program_id(1) * KEY_TILES + tl.arange(0, KEY_TILES)
    key_tile = key_column[None, :]
    key_begin = key_tile * 128
    key_end = key_begin + 127
    start = tl.load(Starts + query, query < tokens, other=0)
    position = query - start
    complete = (position + 1) // 4

    lower = tl.maximum((key_begin - start) // 4, 0)
    upper = tl.minimum((tl.minimum(key_end, query) - start) // 4, complete - 1)

    valid = (query < tokens) & (key_tile < blocks) & (lower <= upper) & (tl.minimum(key_end, query) >= start)
    lower_word, upper_word = lower // 32, upper // 32
    lower_bits = tl.load(Bits + query * WORDS + lower_word, valid & (lower_word < WORDS), other=0).to(tl.uint32)
    upper_bits = tl.load(Bits + query * WORDS + upper_word, valid & (upper_word < WORDS), other=0).to(tl.uint32)
    all_bits = tl.full((), 0xFFFFFFFF, tl.uint32)
    lower_mask = all_bits << (lower % 32)
    upper_mask = all_bits >> (31 - upper % 32)
    same_word = lower_word == upper_word
    lower_mask = tl.where(same_word, lower_mask & upper_mask, lower_mask)
    any_previous = valid & (((lower_bits & lower_mask) != 0) | (~same_word & ((upper_bits & upper_mask) != 0)))
    all_previous = (lower > upper) | (
        valid & ((lower_bits & lower_mask) == lower_mask) & (same_word | ((upper_bits & upper_mask) == upper_mask))
    )
    tail_begin = start + complete * 4
    any_tail = (
        (query < tokens) & (key_tile < blocks) & (tl.minimum(key_end, query) >= tl.maximum(key_begin, tail_begin))
    )
    any_row = any_previous | any_tail
    full_row = (
        (query < tokens)
        & (key_tile < blocks)
        & (key_end < tokens)
        & (key_begin >= start)
        & (key_end <= query)
        & all_previous
    )
    present = tl.max(any_row.to(tl.int32), axis=0) != 0
    full = tl.min(full_row.to(tl.int32), axis=0) != 0
    offsets = tl.program_id(0) * blocks + key_column
    tl.store(Partial + offsets, present & ~full, key_column < blocks)
    tl.store(Full + offsets, full, key_column < blocks)


@triton.jit
def _pack_tiles(Partial, Full, PartialCount, PartialIndices, FullCount, FullIndices, blocks, WIDTH: tl.constexpr):
    row = tl.program_id(0)
    columns = tl.arange(0, WIDTH)
    partial = tl.load(Partial + row * blocks + columns, columns < blocks, other=0).to(tl.int32)
    full = tl.load(Full + row * blocks + columns, columns < blocks, other=0).to(tl.int32)
    partial_position = tl.cumsum(partial) - 1
    full_position = tl.cumsum(full) - 1
    tl.store(PartialIndices + row * blocks + partial_position, columns, partial != 0)
    tl.store(FullIndices + row * blocks + full_position, columns, full != 0)
    tl.store(PartialCount + row, tl.sum(partial))
    tl.store(FullCount + row, tl.sum(full))


def create(bits: torch.Tensor, starts: torch.Tensor, mask_mod) -> BlockMask:
    pass
    tokens, words = bits.shape
    blocks = triton.cdiv(tokens, 128)
    partial = torch.empty((blocks, blocks), device=bits.device, dtype=torch.bool)
    full = torch.empty_like(partial)
    _tile_flags[(blocks, triton.cdiv(blocks, 16))](bits, starts, partial, full, tokens, blocks, words, 16, num_warps=4)
    counts = torch.empty((1, 1, blocks), device=bits.device, dtype=torch.int32)
    full_counts = torch.empty_like(counts)
    indices = torch.zeros((1, 1, blocks, blocks), device=bits.device, dtype=torch.int32)
    full_indices = torch.zeros_like(indices)
    _pack_tiles[(blocks,)](
        partial, full, counts, indices, full_counts, full_indices, blocks, triton.next_power_of_2(blocks), num_warps=4
    )
    return BlockMask.from_kv_blocks(
        counts, indices, full_counts, full_indices, BLOCK_SIZE=128, mask_mod=mask_mod, seq_lengths=(tokens, tokens)
    )
