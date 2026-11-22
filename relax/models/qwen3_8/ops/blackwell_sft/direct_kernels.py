# Copyright (c) 2026 Relax Authors. All Rights Reserved.
# Copyright 2025-2026 RadixArk, Inc. (miles).
# Copyright 2025 Zhipu AI.
# SPDX-License-Identifier: Apache-2.0
# Adapted for Relax.

"""Grouped sparse GQA with BF16 products and FP32 KV atomics."""

import triton
import triton.language as tl


@triton.jit
def _forward(
    Q,
    K,
    V,
    I,
    Positions,
    Rows,
    O,
    L,
    HQ: tl.constexpr,
    HK: tl.constexpr,
    D: tl.constexpr,
    S: tl.constexpr,
    G: tl.constexpr,
    BH: tl.constexpr,
    BK: tl.constexpr,
    SCALE: tl.constexpr,
    BF16_PRODUCTS: tl.constexpr,
):
    t, kh = tl.program_id(0), tl.program_id(1)
    physical = tl.load(Rows + t)
    position = tl.load(Positions + physical)
    doc_start = physical - position
    block_count = (S - 4) // 4
    h = tl.arange(0, BH)
    d = tl.arange(0, D)
    b = tl.arange(0, BK)
    q = tl.load(Q + (t * HQ + kh * G + h[:, None]) * D + d[None, :], h[:, None] < G, other=0)
    maximum = tl.full((BH,), -float("inf"), tl.float32)
    normalizer = tl.zeros((BH,), tl.float32)
    acc = tl.zeros((BH, D), tl.float32)
    for start in range(tl.cdiv(S, BK)):
        offset = start * BK + b
        block = tl.load(I + physical * block_count + offset // 4, offset < 4 * block_count, other=-1)
        selected = tl.where(block >= 0, doc_start + 4 * block + offset % 4, -1)
        tail = offset - 4 * block_count
        tail_id = tl.where(tail < (position + 1) % 4, doc_start + position // 4 * 4 + tail, -1)
        ids = tl.where(offset < 4 * block_count, selected, tail_id)
        valid = ids >= 0
        k = tl.load(K + (tl.maximum(ids, 0)[:, None] * HK + kh) * D + d[None, :], valid[:, None], other=0)
        v = tl.load(V + (tl.maximum(ids, 0)[:, None] * HK + kh) * D + d[None, :], valid[:, None], other=0)
        scores = tl.dot(q, tl.trans(k)) * SCALE
        scores = tl.where(valid[None, :] & (h[:, None] < G), scores, -float("inf"))
        next_maximum = tl.maximum(maximum, tl.max(scores, 1))
        safe_maximum = tl.where(next_maximum == -float("inf"), 0.0, next_maximum)
        p = tl.exp(scores - safe_maximum[:, None])
        alpha = tl.where(maximum == -float("inf"), 0.0, tl.exp(maximum - safe_maximum))
        normalizer = normalizer * alpha + tl.sum(p, 1)
        if BF16_PRODUCTS:
            product = tl.dot(p.to(v.dtype), v)
        else:
            product = tl.dot(p, v.to(tl.float32), input_precision="tf32x3")
        acc = acc * alpha[:, None] + product
        maximum = next_maximum
    denom = tl.where(normalizer == 0.0, 1.0, normalizer)
    tl.store(O + (t * HQ + kh * G + h[:, None]) * D + d[None, :], acc / denom[:, None], h[:, None] < G)
    tl.store(L + t * HQ + kh * G + h, tl.where(normalizer == 0.0, -float("inf"), maximum + tl.log(denom)), h < G)


@triton.jit
def _backward(
    Q,
    K,
    V,
    I,
    Positions,
    Rows,
    O,
    L,
    DO,
    DQ,
    DK,
    DV,
    HQ: tl.constexpr,
    HK: tl.constexpr,
    D: tl.constexpr,
    S: tl.constexpr,
    G: tl.constexpr,
    BH: tl.constexpr,
    BK: tl.constexpr,
    SCALE: tl.constexpr,
    BF16_PRODUCTS: tl.constexpr,
):
    t, kh = tl.program_id(0), tl.program_id(1)
    physical = tl.load(Rows + t)
    position = tl.load(Positions + physical)
    doc_start = physical - position
    block_count = (S - 4) // 4
    h, d, b = tl.arange(0, BH), tl.arange(0, D), tl.arange(0, BK)
    qptr = (t * HQ + kh * G + h[:, None]) * D + d[None, :]
    q = tl.load(Q + qptr, h[:, None] < G, other=0)
    do = tl.load(DO + qptr, h[:, None] < G, other=0)
    out = tl.load(O + qptr, h[:, None] < G, other=0)
    lse = tl.load(L + t * HQ + kh * G + h, h < G, other=-float("inf"))
    delta = tl.sum(do.to(tl.float32) * out, 1)
    dq = tl.zeros((BH, D), tl.float32)
    for start in range(tl.cdiv(S, BK)):
        offset = start * BK + b
        block = tl.load(I + physical * block_count + offset // 4, offset < 4 * block_count, other=-1)
        selected = tl.where(block >= 0, doc_start + 4 * block + offset % 4, -1)
        tail = offset - 4 * block_count
        tail_id = tl.where(tail < (position + 1) % 4, doc_start + position // 4 * 4 + tail, -1)
        ids = tl.where(offset < 4 * block_count, selected, tail_id)
        valid = ids >= 0
        ptr = (tl.maximum(ids, 0)[:, None] * HK + kh) * D + d[None, :]
        k = tl.load(K + ptr, valid[:, None], other=0)
        v = tl.load(V + ptr, valid[:, None], other=0)
        scores = tl.dot(q, tl.trans(k)) * SCALE
        p = tl.exp(scores - tl.where(lse == -float("inf"), 0.0, lse)[:, None])
        p = tl.where(valid[None, :] & (h[:, None] < G) & (lse[:, None] != -float("inf")), p, 0.0)
        dp = tl.dot(do, tl.trans(v))
        ds = p * (dp - delta[:, None]) * SCALE
        if BF16_PRODUCTS:
            dq += tl.dot(ds.to(k.dtype), k)
            dk = tl.dot(tl.trans(ds.to(q.dtype)), q)
            dv = tl.dot(tl.trans(p.to(do.dtype)), do)
        else:
            dq += tl.dot(ds, k.to(tl.float32), input_precision="tf32x3")
            dk = tl.dot(tl.trans(ds), q.to(tl.float32), input_precision="tf32x3")
            dv = tl.dot(tl.trans(p), do.to(tl.float32), input_precision="tf32x3")
        tl.atomic_add(DK + ptr, dk, valid[:, None], sem="relaxed")
        tl.atomic_add(DV + ptr, dv, valid[:, None], sem="relaxed")
    tl.store(DQ + qptr, dq, h[:, None] < G)
