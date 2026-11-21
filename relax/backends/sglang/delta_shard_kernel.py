# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Validate or install tiles of already partitioned raw BF16 coordinates."""

import triton
import triton.language as tl


@triton.jit
def shard_tiles(
    raw,
    rows,
    target,
    invalid,
    WRITE: tl.constexpr,
    BLOCK: tl.constexpr,
    BROADCAST: tl.constexpr = False,
    DOWN: tl.constexpr = False,
    HIDDEN: tl.constexpr = 1,
    LOCAL: tl.constexpr = 1,
    TP: tl.constexpr = 1,
    RANK: tl.constexpr = 0,
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
        if BROADCAST:
            if DOWN:
                column = pos % (TP * LOCAL)
                present = present & (column >= RANK * LOCAL) & (column < (RANK + 1) * LOCAL)
                pos = (pos // (TP * LOCAL)) * LOCAL + column - RANK * LOCAL
            else:
                present = present & (pos >= RANK * LOCAL * HIDDEN) & (pos < (RANK + 1) * LOCAL * HIDDEN)
                pos = pos - RANK * LOCAL * HIDDEN
        word = tl.load((raw + vo).to(tl.pointer_type(tl.int16)) + lane, mask=present, other=0)
        tl.store(target + base + pos.to(tl.int64), word, mask=present)
    else:
        compare = present & ((lane > 0) | (has_prev != 0))
        prev = tl.load(ptr - 1, mask=compare, other=0)
        bad = (present & ((pos < 0) | (pos >= limit))) | (compare & (pos <= prev))
        tl.atomic_or(invalid + tl.full((BLOCK,), 0, tl.int32), 1, mask=bad)
