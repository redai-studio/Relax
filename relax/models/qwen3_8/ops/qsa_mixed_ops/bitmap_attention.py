# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Bitmap-native QSA with shared GQA forward and direct KV gradient
reduction."""

from collections.abc import Sequence

import torch
import triton
import triton.language as tl


FORWARD_BM = 128
FORWARD_BN = 64
FORWARD_WARPS = 8
FORWARD_STAGES = 2


@triton.jit
def _mask(Bits, start, q, k, N, W: tl.constexpr):
    block = k[None, :] // 4
    complete = (q[:, None] + 1) // 4
    # Every <=128-token doc-local aligned tile falls within one bitmap word.
    first_key = tl.min(k, 0)
    word = tl.load(Bits + (start + q) * W + first_key // 128, (q < N) & (first_key < N), other=0)
    selected = ((word[:, None] >> (block % 32)) & 1) != 0
    return (
        (q[:, None] < N)
        & (k[None, :] < N)
        & (k[None, :] <= q[:, None])
        & (((block < complete) & selected) | (block == complete))
    )


@triton.jit
def _fwd(
    Q,
    K,
    V,
    Bits,
    Bounds,
    O,
    L,
    HQ: tl.constexpr,
    HK: tl.constexpr,
    D: tl.constexpr,
    W: tl.constexpr,
    SCALE: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    QPACK: tl.constexpr,
):
    tile, kh, doc = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    start = tl.load(Bounds + doc)
    n = tl.load(Bounds + doc + 1) - start
    if tile * QPACK >= n:
        return
    row = tl.arange(0, BM)
    group = HQ // HK
    qi = tl.where(row < QPACK * group, tile * QPACK + row // group, n)
    h = kh * group + row % group
    ki = tl.arange(0, BN)
    d = tl.arange(0, D)
    q = tl.load(Q + ((start + qi[:, None]) * HQ + h[:, None]) * D + d[None, :], qi[:, None] < n, other=0)
    maximum = tl.full((BM,), -float("inf"), tl.float32)
    z = tl.zeros((BM,), tl.float32)
    acc = tl.zeros((BM, D), tl.float32)
    for base in range(tl.cdiv(tl.minimum((tile + 1) * QPACK, n), BN)):
        kk = base * BN + ki
        k = tl.load(K + ((start + kk[:, None]) * HK + kh) * D + d[None, :], kk[:, None] < n, other=0)
        v = tl.load(V + ((start + kk[:, None]) * HK + kh) * D + d[None, :], kk[:, None] < n, other=0)
        score = tl.dot(q, tl.trans(k)) * (SCALE * 1.4426950408889634)
        valid = _mask(Bits, start, qi, kk, n, W)
        score = tl.where(valid, score, -float("inf"))
        newmax = tl.maximum(maximum, tl.max(score, 1))
        safe = tl.where(newmax == -float("inf"), 0.0, newmax)
        p = tl.exp2(score - safe[:, None])
        alpha = tl.exp2(maximum - safe)
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
        z = z * alpha + tl.sum(p, 1)
        maximum = newmax
    tl.store(
        O + ((start + qi[:, None]) * HQ + h[:, None]) * D + d[None, :],
        acc / tl.maximum(z[:, None], 1.0e-30),
        qi[:, None] < n,
    )
    tl.store(L + (start + qi) * HQ + h, maximum + tl.log2(z), qi < n)


@triton.jit
def _delta(O, DO, Delta, S: tl.constexpr, HQ: tl.constexpr, D: tl.constexpr, B: tl.constexpr):
    r = tl.program_id(0) * B + tl.arange(0, B)
    d = tl.arange(0, D)
    a = tl.load(O + r[:, None] * D + d[None, :], r[:, None] < S * HQ, other=0).to(tl.float32)
    b = tl.load(DO + r[:, None] * D + d[None, :], r[:, None] < S * HQ, other=0).to(tl.float32)
    tl.store(Delta + r, tl.sum(a * b, 1), r < S * HQ)


@triton.jit
def _dq(
    Q,
    K,
    V,
    Bits,
    Bounds,
    L,
    Delta,
    DO,
    DQ,
    HQ: tl.constexpr,
    HK: tl.constexpr,
    D: tl.constexpr,
    W: tl.constexpr,
    SCALE: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
):
    tile, h, doc = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    start = tl.load(Bounds + doc)
    n = tl.load(Bounds + doc + 1) - start
    if tile * BM >= n:
        return
    qi = tile * BM + tl.arange(0, BM)
    ki = tl.arange(0, BN)
    d = tl.arange(0, D)
    kh = h // (HQ // HK)
    qp = ((start + qi[:, None]) * HQ + h) * D + d[None, :]
    q = tl.load(Q + qp, qi[:, None] < n, other=0)
    do = tl.load(DO + qp, qi[:, None] < n, other=0)
    lse = tl.load(L + (start + qi) * HQ + h, qi < n, other=0)
    delta = tl.load(Delta + (start + qi) * HQ + h, qi < n, other=0)
    acc = tl.zeros((BM, D), tl.float32)
    for base in range(tl.cdiv(tl.minimum((tile + 1) * BM, n), BN)):
        kk = base * BN + ki
        ptr = ((start + kk[:, None]) * HK + kh) * D + d[None, :]
        k = tl.load(K + ptr, kk[:, None] < n, other=0)
        v = tl.load(V + ptr, kk[:, None] < n, other=0)
        score = tl.dot(q, tl.trans(k)) * (SCALE * 1.4426950408889634)
        p = tl.where(_mask(Bits, start, qi, kk, n, W), tl.exp2(score - lse[:, None]), 0.0)
        dp = tl.dot(do, tl.trans(v))
        ds = (p * (dp - delta[:, None]) * SCALE).to(q.dtype)
        acc += tl.dot(ds, k)
    tl.store(DQ + qp, acc, qi[:, None] < n)


@triton.jit
def _dkdv(
    Q,
    K,
    V,
    Bits,
    Bounds,
    L,
    Delta,
    DO,
    DK,
    DV,
    HQ: tl.constexpr,
    HK: tl.constexpr,
    D: tl.constexpr,
    W: tl.constexpr,
    SCALE: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    QPACK: tl.constexpr,
):
    tile, kh, doc = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    start = tl.load(Bounds + doc)
    n = tl.load(Bounds + doc + 1) - start
    if tile * BN >= n:
        return
    ki = tile * BN + tl.arange(0, BN)
    qr = tl.arange(0, BM)
    d = tl.arange(0, D)
    ptr = ((start + ki[:, None]) * HK + kh) * D + d[None, :]
    k = tl.load(K + ptr, ki[:, None] < n, other=0)
    v = tl.load(V + ptr, ki[:, None] < n, other=0)
    dk = tl.zeros((BN, D), tl.float32)
    dv = tl.zeros((BN, D), tl.float32)
    for base in range(tile * BN // QPACK, tl.cdiv(n, QPACK)):
        qi = base * QPACK + qr // (HQ // HK)
        qi = tl.where(qr < QPACK * (HQ // HK), qi, n)
        h = kh * (HQ // HK) + qr % (HQ // HK)
        qp = ((start + qi[:, None]) * HQ + h[:, None]) * D + d[None, :]
        q = tl.load(Q + qp, qi[:, None] < n, other=0)
        do = tl.load(DO + qp, qi[:, None] < n, other=0)
        lse = tl.load(L + (start + qi) * HQ + h, qi < n, other=0)
        delta = tl.load(Delta + (start + qi) * HQ + h, qi < n, other=0)
        score = tl.dot(q, tl.trans(k)) * (SCALE * 1.4426950408889634)
        p = tl.where(_mask(Bits, start, qi, ki, n, W), tl.exp2(score - lse[:, None]), 0.0)
        dp = tl.dot(do, tl.trans(v))
        ds = (p * (dp - delta[:, None]) * SCALE).to(q.dtype)
        dk += tl.dot(tl.trans(ds), q)
        dv += tl.dot(tl.trans(p.to(q.dtype)), do)
    out = ((start + ki[:, None]) * HK + kh) * D + d[None, :]
    tl.store(DK + out, dk, ki[:, None] < n)
    tl.store(DV + out, dv, ki[:, None] < n)


_KV_TILE = (64, 64, 8, 1, 10)


class _Attention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, bits, bounds, scale, bm, bn, warps, stages):
        if not all(x.is_contiguous() for x in (q, k, v, bits)):
            raise ValueError("Experimental bitmap attention requires contiguous token-major inputs")
        s, hq, d = q.shape
        hk = k.shape[1]
        if d != 256 or hq % hk or q.dtype != torch.bfloat16:
            raise ValueError("Experimental kernel supports BF16 D256 GQA")
        qpack = FORWARD_BM // (hq // hk)
        if qpack < 1:
            raise ValueError("GQA ratio exceeds the forward tile capacity")
        maximum = max(b - a for a, b in zip(bounds, bounds[1:]))
        host = torch.tensor(bounds, dtype=torch.int32, pin_memory=True)
        cu = host.to(q.device, non_blocking=True)
        out = torch.empty_like(q)
        lse = torch.empty((s, hq), device=q.device, dtype=torch.float32)
        args = (hq, hk, d, bits.shape[1], scale, bm, bn)
        _fwd[(triton.cdiv(maximum, qpack), hk, len(bounds) - 1)](
            q,
            k,
            v,
            bits,
            cu,
            out,
            lse,
            hq,
            hk,
            d,
            bits.shape[1],
            scale,
            FORWARD_BM,
            FORWARD_BN,
            qpack,
            num_warps=FORWARD_WARPS,
            num_stages=FORWARD_STAGES,
        )
        ctx.save_for_backward(q, k, v, bits, cu, out, lse)
        ctx.meta = (args, maximum, len(bounds) - 1, warps, stages)
        return out

    @staticmethod
    def backward(ctx, grad):
        q, k, v, bits, cu, out, lse = ctx.saved_tensors
        args, maximum, docs, warps, stages = ctx.meta
        hq, hk, d, w, scale, bm, bn = args
        grad = grad.contiguous()
        delta = torch.empty_like(lse)
        dq = torch.empty_like(q)
        dk = torch.empty_like(k)
        dv = torch.empty_like(v)
        _delta[(triton.cdiv(q.shape[0] * hq, 32),)](out, grad, delta, q.shape[0], hq, d, 32)
        _dq[(triton.cdiv(maximum, bm), hq, docs)](
            q, k, v, bits, cu, lse, delta, grad, dq, *args, num_warps=warps, num_stages=stages
        )
        kv_bm, kv_bn, kv_warps, kv_stages, qpack = _KV_TILE
        qpack = min(qpack, kv_bm // (hq // hk))
        if qpack < 1:
            raise ValueError("GQA group is larger than the backward query tile")
        kv_args = (hq, hk, d, w, scale, kv_bm, kv_bn, qpack)
        _dkdv[(triton.cdiv(maximum, kv_bn), hk, docs)](
            q, k, v, bits, cu, lse, delta, grad, dk, dv, *kv_args, num_warps=kv_warps, num_stages=kv_stages
        )
        return dq, dk, dv, None, None, None, None, None, None, None


def attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    bits: torch.Tensor,
    scale: float,
    bounds: Sequence[int],
    bm: int = 64,
    bn: int = 64,
    warps: int = 8,
    stages: int = 1,
) -> torch.Tensor:
    if bn not in (16, 32, 64, 128):
        raise ValueError("Bitmap query-gradient KV tiles must fit within one 128-token word")
    bounds = tuple(bounds)
    if q.ndim != 3 or k.ndim != 3 or v.shape != k.shape:
        raise ValueError("Expected Q[S,HQ,D] and K/V[S,HK,D]")
    if not all(x.is_cuda and x.device == q.device and x.dtype == torch.bfloat16 for x in (q, k, v)):
        raise ValueError("Q/K/V must be BF16 tensors on the same CUDA device")
    if q.shape[0] != k.shape[0] or q.shape[2] != k.shape[2] or k.shape[1] < 1 or q.shape[1] < 1:
        raise ValueError("Incompatible Q/K/V shapes")
    if len(bounds) < 2 or bounds[0] != 0 or bounds[-1] != q.shape[0] or any(b < a for a, b in zip(bounds, bounds[1:])):
        raise ValueError("Bounds must partition the token dimension")
    if q.shape[2] != 256 or q.shape[1] % k.shape[1] or q.shape[1] // k.shape[1] > 64:
        raise ValueError("Bitmap QSA requires D256 and a GQA ratio at most 64")
    maximum = max(b - a for a, b in zip(bounds, bounds[1:]))
    if (
        bits.device != q.device
        or bits.dtype != torch.int32
        or bits.ndim != 2
        or bits.shape[0] != q.shape[0]
        or bits.shape[1] < triton.cdiv(maximum, 128)
    ):
        raise ValueError("Bitmap must contain enough int32 words per query")
    if not q.shape[0]:
        return q + (k.sum() + v.sum()) * 0
    return _Attention.apply(q, k, v, bits, bounds, scale, bm, bn, warps, stages)
