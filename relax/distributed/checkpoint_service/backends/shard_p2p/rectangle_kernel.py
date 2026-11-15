# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Optional Triton batched copies into current native expert allocations."""

import triton
import triton.language as tl


BLOCK_SIZE = 8192


@triton.jit
def _copy(metadata, checks, N: tl.constexpr, COLUMNS: tl.constexpr, BLOCK: tl.constexpr, VERIFY: tl.constexpr):
    descriptor = metadata + tl.program_id(1) * 7
    source = tl.load(descriptor).to(tl.pointer_type(tl.uint16))
    target = tl.load(descriptor + 1).to(tl.pointer_type(tl.uint16))
    source_offset, target_offset = tl.load(descriptor + 2), tl.load(descriptor + 3)
    source_stride, target_stride = tl.load(descriptor + 5), tl.load(descriptor + 6)
    position = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    row, column = position // COLUMNS, position % COLUMNS
    value = tl.load(source + source_offset + row * source_stride + column, mask=position < N, other=0)
    destination = target + target_offset + row * target_stride + column
    if VERIFY:
        native = tl.load(destination, mask=position < N, other=0)
        mismatch = tl.sum(((value != native) & (position < N)).to(tl.int32), 0) > 0
    tl.store(destination, value, mask=position < N)
    if VERIFY:
        # Every tile must replace its failure sentinel, including matching
        # tiles. A skipped launch must never appear to have passed validation.
        tl.store(checks + tl.program_id(1) * tl.cdiv(N, BLOCK) + tl.program_id(0), mismatch.to(tl.int32))


def launch(metadata, elements: int, columns: int, checks=None) -> None:
    _copy[(triton.cdiv(elements, BLOCK_SIZE), metadata.shape[0])](
        metadata, checks if checks is not None else 0, elements, columns, BLOCK_SIZE, checks is not None, num_warps=4
    )
