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
def _encode(
    X,
    C,
    Y,
    XS: tl.constexpr,
    YS: tl.constexpr,
    K: tl.constexpr,
    WORDS: tl.constexpr,
    ORDERED: tl.constexpr,
    WIDTH: tl.constexpr,
    QSA_RATIO: tl.constexpr = 1,
):
    row = tl.program_id(0)
    j = tl.arange(0, K)
    ids = tl.load(X + row * XS + j, j < WIDTH, other=-1)
    if QSA_RATIO != 1:
        position = tl.load(C + row)
        valid = (ids >= 0) & (ids % QSA_RATIO == 0) & (ids + QSA_RATIO - 1 <= position)
        ids = tl.where(valid, ids // QSA_RATIO, -1)
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
        if QSA_RATIO != 1:
            # QSA already produced unique expanded token IDs. Set the complete
            # block bits directly instead of sorting/padding another ID tensor.
            bits = tl.full((K,), 1, tl.uint32) << (ids % 32)
            tl.atomic_or(Y + row * YS + 1 + ids // 32, bits, (ids >= 0) & (ids < WORDS * 32), sem="relaxed")
        else:
            ids = tl.where(ids >= 0, ids, 0x7FFFFFFF)
            if not ORDERED:
                ids = tl.sort(ids, descending=False)
            keys = ids // 32
            bits = tl.full((K,), 1, tl.uint32) << (ids % 32)
            _, merged = tl.associative_scan((keys, bits), 0, _group_or)
            next_key = tl.gather(keys, tl.minimum(j + 1, K - 1), 0)
            last = (j == K - 1) | (keys != next_key)
            tl.store(Y + row * YS + 1 + keys, merged, last & (ids < WORDS * 32))


@triton.jit(do_not_specialize=["SOURCE_START", "DEST_START"])
def _decode(
    DATA,
    ROWS,
    DESTS,
    OUT,
    STRIDE: tl.constexpr,
    K: tl.constexpr,
    WORDS: tl.constexpr,
    SOURCE_START,
    DEST_START,
    CONTIGUOUS: tl.constexpr,
):
    row = tl.program_id(0)
    if CONTIGUOUS:
        source = SOURCE_START.to(tl.int64) + row
        target = DEST_START.to(tl.int64) + row
    else:
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
        indices,
        indices,
        output,
        indices.stride(0),
        output.stride(0),
        KEYE_BITMAP.topk,
        words,
        False,
        KEYE_BITMAP.topk,
        num_warps=8,
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
        KEYE_BITMAP.topk,
        num_warps=8,
    )


def encode_qsa_tokens_into(tokens: torch.Tensor, positions: torch.Tensor, output: torch.Tensor) -> None:
    """Filter R4 starts and partial tails directly into encoded capture
    scratch."""
    words = output.shape[1] - 1
    if words <= 0 or words & (words - 1) or tokens.shape[1] != 2051 or positions.shape != tokens.shape[:1]:
        raise ValueError("QSA bitmap capture requires 2051 expanded tokens, positions, and power-of-two scratch")
    _encode[(len(tokens),)](
        tokens,
        positions,
        output,
        tokens.stride(0),
        output.stride(0),
        triton.next_power_of_2(tokens.shape[1]),
        words,
        False,
        tokens.shape[1],
        4,
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
        output.shape[1],
        triton.next_power_of_2(max(1, data.shape[1] - 1)),
        0,
        0,
        False,
        num_warps=8,
    )


def decode_contiguous_into(
    data: torch.Tensor, source_start: int, destination_start: int, count: int, output: torch.Tensor
) -> None:
    """Decode CPU-known contiguous rows without allocating GPU coordinates."""
    if min(source_start, destination_start, count) < 0 or (
        source_start + count > data.shape[0] or destination_start + count > output.shape[0]
    ):
        raise ValueError("Contiguous replay decode rows exceed the source or output")
    if not count:
        return
    _decode[(count,)](
        data,
        data,
        data,
        output,
        data.shape[1],
        output.shape[1],
        triton.next_power_of_2(max(1, data.shape[1] - 1)),
        source_start,
        destination_start,
        True,
        num_warps=8,
    )


@triton.jit(do_not_specialize=["SOURCE_START", "DEST_START"])
def _copy_dense_bits(
    DATA,
    ROWS,
    DESTS,
    OUT,
    STRIDE: tl.constexpr,
    WORDS: tl.constexpr,
    BLOCK: tl.constexpr,
    SOURCE_START,
    DEST_START,
    CONTIGUOUS: tl.constexpr,
):
    row = tl.program_id(0)
    source = SOURCE_START + row if CONTIGUOUS else tl.load(ROWS + row)
    target = DEST_START + row if CONTIGUOUS else tl.load(DESTS + row)
    header = tl.load(DATA + source * STRIDE)
    word = tl.arange(0, BLOCK)
    value = tl.load(DATA + source * STRIDE + 1 + word, (word < STRIDE - 1) & (word < WORDS), other=0).to(tl.uint32)
    count = -header - 1
    remaining = tl.minimum(tl.maximum(count - word * 32, 0), 32)
    prefix = tl.where(remaining == 32, 0xFFFFFFFF, (tl.full((BLOCK,), 1, tl.uint32) << remaining) - 1)
    tl.store(OUT + target * WORDS + word, tl.where(header < 0, prefix, value), word < WORDS)


def copy_dense_bits_into(
    data: torch.Tensor,
    rows: torch.Tensor | None,
    destinations: torch.Tensor | None,
    output: torch.Tensor,
    *,
    source_start: int = 0,
    destination_start: int = 0,
    count: int | None = None,
) -> None:
    contiguous = count is not None
    count = int(count) if contiguous else len(rows)
    if contiguous and (
        min(source_start, destination_start, count) < 0
        or source_start + count > len(data)
        or destination_start + count > len(output)
    ):
        raise ValueError("Contiguous replay bits exceed the source or output")
    if not count:
        return
    _copy_dense_bits[(count,)](
        data,
        data if contiguous else rows,
        data if contiguous else destinations,
        output,
        data.shape[1],
        output.shape[1],
        triton.next_power_of_2(output.shape[1]),
        source_start,
        destination_start,
        contiguous,
        num_warps=4,
    )


@triton.jit
def _scatter_missing_bits(IDS, ROWS, OUT, WORDS: tl.constexpr, K: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    target = tl.load(ROWS + row)
    word = tl.arange(0, BLOCK)
    tl.store(OUT + target * WORDS + word, 0, word < WORDS)
    tl.debug_barrier()
    column = tl.arange(0, K)
    ids = tl.load(IDS + row * K + column)
    tl.atomic_or(
        OUT + target * WORDS + ids // 32,
        tl.full((K,), 1, tl.uint32) << (ids % 32),
        (ids >= 0) & (ids < WORDS * 32),
        sem="relaxed",
    )


def scatter_missing_bits_into(ids: torch.Tensor, rows: torch.Tensor, output: torch.Tensor) -> None:
    if len(rows):
        _scatter_missing_bits[(len(rows),)](
            ids, rows, output, output.shape[1], ids.shape[1], triton.next_power_of_2(output.shape[1]), num_warps=4
        )
