# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""QSA computation helpers; no model-method installation."""

from __future__ import annotations

import math

import torch
import triton
import triton.language as tl
from torch import Tensor


@triton.jit(do_not_specialize=["columns", "rows"])
def _score_kernel(
    scores,
    positions,
    output,
    columns,
    rows,
    ratio: tl.constexpr,
    scale: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    row = offsets // columns
    col = offsets % columns
    valid = row < rows
    base = row * 4 * columns + col
    a = tl.load(scores + base, valid, other=0)
    b = tl.load(scores + base + columns, valid, other=0)
    c = tl.load(scores + base + 2 * columns, valid, other=0)
    d = tl.load(scores + base + 3 * columns, valid, other=0)
    # ATen adds the reduction identity +0 to each head, normalizing -0.
    a = tl.where(a <= 0, 0.0, a)
    b = tl.where(b <= 0, 0.0, b)
    c = tl.where(c <= 0, 0.0, c)
    d = tl.where(d <= 0, 0.0, d)
    # Match ATen's sequential H=4 reduction and scalar-divide reciprocal multiply.
    value = ((a + b) + c) + d
    value = value * scale
    visible = (tl.load(positions + row, valid, other=0).to(tl.int64) + 1) // ratio
    value = tl.where(col < visible, value, float("-inf"))
    tl.store(output + offsets, value, valid)


def score_postprocess(scores: Tensor, positions: Tensor, ratio: int, dim: int) -> Tensor:
    rows, heads, columns = scores.shape
    assert heads == 4 and scores.dtype == torch.float32 and scores.is_contiguous()
    output = torch.empty((rows, columns), device=scores.device, dtype=torch.float32)
    _score_kernel[(triton.cdiv(rows * columns, 256),)](
        scores,
        positions,
        output,
        columns,
        rows,
        ratio,
        1.0 / math.sqrt(dim),
        256,
        enable_fp_fusion=False,
    )
    return output


@triton.jit(do_not_specialize=["columns", "nbytes"])
def _pack_all_candidates_kernel(
    scores, positions, output, columns, nbytes, ratio: tl.constexpr, scale: tl.constexpr, BLOCK: tl.constexpr
):
    row = tl.program_id(0).to(tl.int64)
    byte_start = tl.program_id(1) * BLOCK
    byte = byte_start + tl.arange(0, BLOCK)
    result = tl.full((BLOCK,), 0, tl.int32)
    # The output keeps the full document stride, including zero padding beyond
    # this tile's candidate range. Those CTAs need no score or position loads.
    if byte_start * 8 < columns:
        bit = tl.arange(0, 8)
        col = byte[:, None] * 8 + bit[None, :]
        valid = col < columns
        base = row * 4 * columns + col
        a = tl.load(scores + base, valid, other=0)
        b = tl.load(scores + base + columns, valid, other=0)
        c = tl.load(scores + base + 2 * columns, valid, other=0)
        d = tl.load(scores + base + 3 * columns, valid, other=0)
        # Keep exactly the same arithmetic as _score_kernel, including overflow
        # and NaN propagation. TopK(k=columns) followed by packing retains every
        # finite causal score, regardless of its rank.
        a = tl.where(a <= 0, 0.0, a)
        b = tl.where(b <= 0, 0.0, b)
        c = tl.where(c <= 0, 0.0, c)
        d = tl.where(d <= 0, 0.0, d)
        value = (((a + b) + c) + d) * scale
        visible = (tl.load(positions + row).to(tl.int64) + 1) // ratio
        finite = (tl.abs(value) != float("inf")) & (value == value)
        selected = valid & (col < visible) & finite
        result = tl.sum(selected.to(tl.int32) * (1 << bit[None, :]), 1)
    tl.store(output + row * nbytes + byte, result.to(tl.uint8), byte < nbytes)


def pack_all_candidates(scores: Tensor, positions: Tensor, ratio: int, dim: int, output: Tensor) -> None:
    """Fuse score processing and packing when TopK retains every candidate."""
    rows, heads, columns = scores.shape
    assert heads == 4 and scores.dtype == torch.float32 and scores.is_contiguous()
    nbytes = output.shape[1]
    if rows == 0 or nbytes == 0:
        return
    _pack_all_candidates_kernel[(rows, triton.cdiv(nbytes, 64))](
        scores, positions, output, columns, nbytes, ratio, 1.0 / math.sqrt(dim), 64, enable_fp_fusion=False
    )


@triton.jit
def _topk_mask(keys, valid, k: tl.constexpr):
    # Search ordered uint32 keys from the most significant bit. An exact count
    # can stop between adjacent scores: the chosen set already contains k IDs.
    threshold = tl.full((), 0, tl.uint32)
    shift = 31
    count = -1
    while (shift >= 0) & (count != k):
        proposed = threshold | (tl.full((), 1, tl.uint32) << shift)
        count = tl.sum((valid & (keys >= proposed)).to(tl.int32), 0)
        threshold = tl.where(count >= k, proposed, threshold)
        shift -= 1
    if count == k:
        chosen = valid & (keys >= threshold)
    else:
        # Match ATen CUDA radix TopK: fill cutoff ties in original column order.
        greater = valid & (keys > threshold)
        equal = valid & (keys == threshold)
        remaining = k - tl.sum(greater.to(tl.int32), 0)
        chosen = greater | (equal & (tl.cumsum(equal.to(tl.int32), 0) <= remaining))
    return chosen


@triton.jit(do_not_specialize=["columns", "nbytes"])
def _score_topk_pack_kernel(
    scores,
    positions,
    output,
    columns,
    nbytes,
    ratio: tl.constexpr,
    scale: tl.constexpr,
    k: tl.constexpr,
    BLOCK: tl.constexpr,
    BYTES: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    col = tl.arange(0, BLOCK)
    valid = col < columns
    base = row * 4 * columns + col
    a = tl.load(scores + base, valid, other=0)
    b = tl.load(scores + base + columns, valid, other=0)
    c = tl.load(scores + base + 2 * columns, valid, other=0)
    d = tl.load(scores + base + 3 * columns, valid, other=0)
    a = tl.where(a <= 0, 0.0, a)
    b = tl.where(b <= 0, 0.0, b)
    c = tl.where(c <= 0, 0.0, c)
    d = tl.where(d <= 0, 0.0, d)
    value = (((a + b) + c) + d) * scale
    visible = (tl.load(positions + row).to(tl.int64) + 1) // ratio
    value = tl.where(col < visible, value, float("-inf"))
    if visible > k:
        # ATen TopKTypeConfig<float>: NaNs rank above +Inf; signed zeros have
        # distinct keys. Nonfinite scores consume slots BEFORE finite filtering.
        raw = value.to(tl.uint32, bitcast=True)
        keys = raw ^ tl.where((raw & 0x80000000) != 0, 0xFFFFFFFF, 0x80000000).to(tl.uint32)
        keys = tl.where(value == value, keys, 0xFFFFFFFF).to(tl.uint32)
        keys = tl.where(valid, keys, 0)
        chosen = _topk_mask(keys, valid, k)
        selected = chosen & (keys > 0x007FFFFF) & (keys < 0xFF800000)
    else:
        # A row can see <=k blocks even in a wider tile. Keep its original GEMM
        # geometry, but avoid selection when all visible scores are retained.
        selected = valid & (col < visible) & (value == value) & (tl.abs(value) != float("inf"))
    grouped = tl.reshape(selected, (BLOCK // 8, 8))
    packed = tl.sum(grouped.to(tl.int32) * (1 << tl.arange(0, 8)[None, :]), 1)
    byte = tl.arange(0, BLOCK // 8)
    used = (columns + 7) // 8
    tl.store(output + row * nbytes + byte, packed.to(tl.uint8), byte < used)
    padding = tl.arange(0, BYTES)
    tl.store(output + row * nbytes + padding, 0, (padding >= used) & (padding < nbytes))


def score_topk_pack(scores: Tensor, positions: Tensor, ratio: int, dim: int, k: int, output: Tensor) -> None:
    """Fuse H4 score processing, exact CUDA radix selection and bitset
    packing."""
    rows, heads, columns = scores.shape
    assert heads == 4 and scores.dtype == torch.float32 and scores.is_contiguous()
    assert 0 <= k <= columns <= 32768
    if rows == 0:
        return
    if k == 0:
        output.zero_()
        return
    _score_topk_pack_kernel[(rows,)](
        scores,
        positions,
        output,
        columns,
        output.shape[1],
        ratio,
        1.0 / math.sqrt(dim),
        k,
        max(8, triton.next_power_of_2(columns)),
        triton.next_power_of_2(output.shape[1]),
        num_warps=4 if columns <= 8192 else (8 if columns <= 16384 else 16),
        enable_fp_fusion=False,
    )


@triton.jit(do_not_specialize=["k", "nbytes", "word_count"])
def _pack_kernel(
    ids, values, words, output, k, nbytes, word_count, BLOCK: tl.constexpr, WORDS: tl.constexpr, BYTES: tl.constexpr
):
    # Each row owns one CTA and disjoint scratch. Both barriers are needed:
    # initialize before any warp's atomic OR, then finish all ORs before reading.
    row = tl.program_id(0).to(tl.int64)
    word_offsets = tl.arange(0, WORDS)
    tl.store(words + row * word_count + word_offsets, 0, word_offsets < word_count)
    tl.debug_barrier()
    col = tl.arange(0, BLOCK)
    value = tl.load(values + row * k + col, col < k, other=float("-inf"))
    block = tl.load(ids + row * k + col, col < k, other=0)
    valid = (col < k) & (tl.abs(value) != float("inf")) & (value == value)
    bits = (1 << (block % 32)).to(tl.uint32)
    tl.atomic_or(words + row * word_count + block // 32, bits, mask=valid, sem="relaxed")
    tl.debug_barrier()
    byte = tl.arange(0, BYTES)
    word = tl.load(words + row * word_count + byte // 4, byte < nbytes, other=0).to(tl.uint32)
    value = (word >> (8 * (byte % 4))) & 255
    tl.store(output + row * nbytes + byte, value.to(tl.uint8), byte < nbytes)


def pack_topk(ids: Tensor, values: Tensor, output: Tensor) -> None:
    """Pack contiguous TopK results into the original, unpadded byte stride.

    Finite entries must have nonnegative block IDs within the output bitset.
    Scratch is private to each call, including calls on different CUDA streams.
    """
    rows, k = ids.shape
    nbytes = output.shape[1]
    if rows == 0 or nbytes == 0:
        return
    if k == 0:
        output.zero_()
        return
    word_count = triton.cdiv(nbytes, 4)
    words = torch.empty((rows, word_count), device=ids.device, dtype=torch.int32)
    _pack_kernel[(rows,)](
        ids,
        values,
        words,
        output,
        k,
        nbytes,
        word_count,
        triton.next_power_of_2(k),
        triton.next_power_of_2(word_count),
        triton.next_power_of_2(nbytes),
        num_ctas=1,
    )
