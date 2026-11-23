# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Lossless document-local QSA bitsets to ascending selected block IDs."""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from torch import Tensor
from triton.language.extra.cuda import libdevice


@triton.jit(do_not_specialize=["nbytes"])
def _unpack_bits(
    bits,
    positions,
    output,
    errors,
    nbytes,
    topk: tl.constexpr,
    ratio: tl.constexpr,
    WORDS: tl.constexpr,
    SLOTS: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    word_id = tl.arange(0, WORDS)
    byte_id = word_id * 4
    # Rows have the original byte stride, which need not be divisible by four.
    a = tl.load(bits + row * nbytes + byte_id, byte_id < nbytes, other=0).to(tl.uint32)
    b = tl.load(bits + row * nbytes + byte_id + 1, byte_id + 1 < nbytes, other=0).to(tl.uint32)
    c = tl.load(bits + row * nbytes + byte_id + 2, byte_id + 2 < nbytes, other=0).to(tl.uint32)
    d = tl.load(bits + row * nbytes + byte_id + 3, byte_id + 3 < nbytes, other=0).to(tl.uint32)
    word = a | (b << 8) | (c << 16) | (d << 24)
    count = libdevice.popc(word.to(tl.int32)).to(tl.int32)
    prefix = tl.cumsum(count, 0) - count
    total = tl.sum(count, 0)
    position = tl.load(positions + row).to(tl.int64)
    visible = (position + 1) // ratio
    # Reject future/reserved/padding bits without silently dropping them.
    allowed_count = tl.minimum(tl.maximum(visible - word_id * 32, 0), 32)
    allowed_mask = tl.where(allowed_count >= 32, 0xFFFFFFFF, (1 << allowed_count) - 1).to(tl.uint32)
    invalid = (total > topk) | (position < 0) | (tl.sum(((word & ~allowed_mask) != 0).to(tl.int32), 0) > 0)
    tl.store(errors + row, invalid)
    slot = tl.arange(0, SLOTS)
    # Initialization and scatter write disjoint regions, including across warps.
    tl.store(output + row * topk + slot, -1, (slot >= total) & (slot < topk))
    offset = prefix
    while tl.sum((word != 0).to(tl.int32), 0) > 0:
        active = word != 0
        bit = libdevice.ffs(word.to(tl.int32)) - 1
        tl.store(output + row * topk + offset, word_id * 32 + bit, active & (offset < topk))
        offset += active.to(tl.int32)
        word = word & (word - 1)


def bits_to_ids(bits: Tensor, positions: Tensor, topk: int, ratio: int) -> Tensor:
    """Convert complete blocks only; attention appends the causal tail
    separately.

    Invalid input raises a device-side assertion asynchronously. The production
    path performs no host synchronization and never allocates a dense boolean
    mask.
    """
    if bits.ndim != 2 or bits.dtype != torch.uint8 or not bits.is_cuda or not bits.is_contiguous():
        raise ValueError("QSA bitsets must be contiguous CUDA uint8 rows")
    if positions.numel() != bits.shape[0] or not positions.is_contiguous() or positions.device != bits.device:
        raise ValueError("QSA positions must contain one contiguous value per bitset row")
    if positions.dtype not in (torch.int32, torch.int64) or topk <= 0 or ratio <= 0:
        raise ValueError("QSA positions must be integers and the selection budget/ratio positive")
    rows, nbytes = bits.shape
    if nbytes <= 0:
        raise ValueError("QSA bitsets require at least the reserved tail byte")
    output = torch.empty((rows, topk), dtype=torch.int32, device=bits.device)
    if rows == 0:
        return output
    errors = torch.empty((rows,), dtype=torch.bool, device=bits.device)
    _unpack_bits[(rows,)](
        bits,
        positions,
        output,
        errors,
        nbytes,
        topk,
        ratio,
        triton.next_power_of_2(triton.cdiv(nbytes, 4)),
        triton.next_power_of_2(topk),
        num_warps=4,
    )
    torch._assert_async(~errors.any(), "QSA bitset exceeds its budget or contains a future/reserved block")
    return output
