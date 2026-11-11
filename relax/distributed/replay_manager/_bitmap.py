# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Lossless GPU set codec. Header < 0 denotes the complete prefix (-header-1).

Nonnegative headers contain the selected count, followed by uint32 bitmap
words. Invalid top-k slots decode as -1. Inputs must contain unique valid token
IDs.
"""

import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice

from .codecs import KEYE_BITMAP


@triton.jit
def _group_or(ka, va, kb, vb):
    return kb, tl.where(ka == kb, va | vb, vb)


@triton.jit
def _encode(X, C, Y, XS: tl.constexpr, YS: tl.constexpr, K: tl.constexpr, WORDS: tl.constexpr, ORDERED: tl.constexpr):
    row = tl.program_id(0)
    j = tl.arange(0, K)
    ids = tl.load(X + row * XS + j)
    if ORDERED:
        count = tl.load(C + row)
        ids = tl.where(j < count, ids, -1)
    else:
        count = tl.sum((ids >= 0).to(tl.int32), 0)
    maximum = tl.max(ids, 0)
    full = maximum + 1 == count
    tl.store(Y + row * YS, tl.where(full, -count - 1, count))
    words = tl.arange(0, WORDS)
    tl.store(Y + row * YS + 1 + words, 0)
    tl.debug_barrier()
    if not full:
        ids = tl.where(ids >= 0, ids, 0x7FFFFFFF)
        if not ORDERED:
            ids = tl.sort(ids, descending=False)
        keys = ids // 32
        bits = tl.full((K,), 1, tl.uint32) << (ids % 32)
        _, merged = tl.associative_scan((keys, bits), 0, _group_or)
        next_key = tl.gather(keys, tl.minimum(j + 1, K - 1), 0)
        last = (j == K - 1) | (keys != next_key)
        tl.store(Y + row * YS + 1 + keys, merged, last & (ids < WORDS * 32))


@triton.jit
def _decode(DATA, ROWS, DESTS, OUT, STRIDE: tl.constexpr, K: tl.constexpr, WORDS: tl.constexpr):
    row = tl.program_id(0)
    source = tl.load(ROWS + row)
    target = tl.load(DESTS + row)
    header = tl.load(DATA + source * STRIDE)
    j = tl.arange(0, K)
    tl.store(OUT + target * K + j, -1)
    tl.debug_barrier()
    if header < 0:
        tl.store(OUT + target * K + j, tl.where(j < -header - 1, j, -1))
    else:
        word = tl.arange(0, WORDS)
        value = tl.load(DATA + source * STRIDE + 1 + word, word < STRIDE - 1, other=0).to(tl.uint32)
        count = libdevice.popc(value.to(tl.int32))
        prefix = tl.cumsum(count, 0) - count
        bit = tl.arange(0, 32)
        lower = (tl.full((32,), 1, tl.uint32) << bit) - 1
        rank = libdevice.popc((value[:, None] & lower[None, :]).to(tl.int32))
        selected = ((value[:, None] >> bit[None, :]) & 1) != 0
        dst = prefix[:, None] + rank
        tl.store(OUT + target * K + dst, word[:, None] * 32 + bit[None, :], selected & (dst < K))


def encode_into(indices: torch.Tensor, output: torch.Tensor) -> None:
    """Encode directly into stable per-layer capture scratch, without a
    copy."""
    words = output.shape[1] - 1
    if words <= 0 or words & (words - 1) or indices.shape[1] != KEYE_BITMAP.topk:
        raise ValueError("Keye bitmap scratch requires a power-of-two word capacity and top-k 2048")
    _encode[(len(indices),)](
        indices, indices, output, indices.stride(0), output.stride(0), KEYE_BITMAP.topk, words, False, num_warps=8
    )


def encode_compacted_into(indices: torch.Tensor, counts: torch.Tensor, output: torch.Tensor) -> None:
    """Reuse attention's sorted IDs/counts; do not normalize, pad or sort
    again."""
    words = output.shape[1] - 1
    if words <= 0 or words & (words - 1) or indices.shape[1] != KEYE_BITMAP.topk or counts.shape != indices.shape[:1]:
        raise ValueError("Compacted Keye capture requires sorted top-k 2048 and one count per row")
    _encode[(len(indices),)](
        indices,
        counts,
        output,
        indices.stride(0),
        output.stride(0),
        KEYE_BITMAP.topk,
        words,
        True,
        num_warps=8,
    )


def decode_into(data: torch.Tensor, rows: torch.Tensor, destinations: torch.Tensor, output: torch.Tensor) -> None:
    """Decode selected source rows straight into the final microbatch
    buffer."""
    _decode[(len(rows),)](
        data,
        rows,
        destinations,
        output,
        data.shape[1],
        KEYE_BITMAP.topk,
        triton.next_power_of_2(max(1, data.shape[1] - 1)),
        num_warps=8,
    )
