# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Validate or install tiles of already partitioned raw BF16 coordinates."""

import triton
import triton.language as tl


@triton.jit
def _add(left, right):
    return left + right


@triton.jit
def shard_tiles(
    raw,
    rows,
    target,
    invalid,
    WRITE: tl.constexpr,
    BLOCK: tl.constexpr,
    REGION: tl.constexpr = False,
    OFFSET: tl.constexpr = 0,
    ROWS: tl.constexpr = 1,
    STRIDE: tl.constexpr = 1,
    WIDTH: tl.constexpr = 1,
    WORD_BYTES: tl.constexpr = 2,
    written=None,
):
    row = rows + tl.program_id(0) * 6
    po, vo = tl.load(row), tl.load(row + 1)
    count, has_prev = tl.load(row + 2), tl.load(row + 3)
    base, limit = tl.load(row + 4), tl.load(row + 5)
    lane = tl.arange(0, BLOCK)
    present = lane < count
    ptr = (raw + po).to(tl.pointer_type(tl.int32)) + lane
    pos = tl.load(ptr, mask=present, other=0)
    if WRITE:
        if REGION:
            rel = pos.to(tl.int64) - OFFSET
            present = present & (rel >= 0) & (rel < ROWS * STRIDE) & (rel % STRIDE < WIDTH)
            pos = (rel // STRIDE) * WIDTH + rel % STRIDE
            tl.atomic_add(written, tl.reduce(present.to(tl.int32), 0, _add).to(tl.int64), sem="relaxed")
        word_type: tl.constexpr = tl.int16 if WORD_BYTES == 2 else tl.int32
        word = tl.load((raw + vo).to(tl.pointer_type(word_type)) + lane, mask=present, other=0)
        tl.store(target + base + pos.to(tl.int64), word, mask=present)
    else:
        compare = present & ((lane > 0) | (has_prev != 0))
        prev = tl.load(ptr - 1, mask=compare, other=0)
        bad = (present & ((pos < 0) | (pos >= limit))) | (compare & (pos <= prev))
        tl.atomic_or(invalid + tl.full((BLOCK,), 0, tl.int32), 1, mask=bad)
