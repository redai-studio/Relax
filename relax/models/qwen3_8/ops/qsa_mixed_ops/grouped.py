# Copyright (c) 2026 Relax Authors. All Rights Reserved.
# Copyright 2025-2026 RadixArk, Inc. (miles).
# Copyright 2025 Zhipu AI.
# SPDX-License-Identifier: Apache-2.0
# Adapted for Relax.

"""Grouped sparse GQA with BF16 products and FP32 KV atomics."""

import os

import torch
import triton
import triton.language as tl


@triton.jit
def _forward(
    Q,
    K,
    V,
    I,
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
    h = tl.arange(0, BH)
    d = tl.arange(0, D)
    b = tl.arange(0, BK)
    q = tl.load(Q + (t * HQ + kh * G + h[:, None]) * D + d[None, :], h[:, None] < G, other=0)
    maximum = tl.full((BH,), -float("inf"), tl.float32)
    normalizer = tl.zeros((BH,), tl.float32)
    acc = tl.zeros((BH, D), tl.float32)
    for start in range(tl.cdiv(S, BK)):
        offset = start * BK + b
        ids = tl.load(I + t * S + offset, offset < S, other=-1)
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
        ids = tl.load(I + t * S + offset, offset < S, other=-1)
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


class _Grouped(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, indices, scale, bf16_products=False, block_k=32, gradient_group=None):
        assert q.dtype == k.dtype == v.dtype == torch.bfloat16
        q, k, v, indices = [x.contiguous() for x in (q, k, v, indices)]
        t, hq, d = q.shape
        assert k.shape == v.shape and k.shape[0] >= t and k.shape[2] == d
        hk = k.shape[1]
        assert hq % hk == 0 and d == 256
        params = dict(
            HQ=hq,
            HK=hk,
            D=d,
            S=indices.shape[1],
            G=hq // hk,
            BH=max(16, triton.next_power_of_2(hq // hk)),
            BK=block_k,
            SCALE=scale,
            BF16_PRODUCTS=bf16_products,
        )
        out = torch.empty(q.shape, device=q.device, dtype=torch.float32)
        lse = torch.empty((t, hq), device=q.device, dtype=torch.float32)
        launch = dict(num_warps=4, num_stages=1 if bf16_products else 3)
        _forward[(t, hk)](q, k, v, indices, out, lse, **params, **launch)
        ctx.save_for_backward(q, k, v, indices, out, lse)
        ctx.gradient_group = gradient_group
        backward_block_k = int(os.environ.get("RELAX_QWEN38_QSA_GROUPED_BACKWARD_BLOCK_K", str(block_k)))
        if backward_block_k not in (16, 32, 64, 128):
            raise ValueError("Unsupported Grouped QSA backward block size")
        ctx.params = dict(params, BK=backward_block_k) if bf16_products else params
        ctx.launch = launch
        backward_warps = int(os.environ.get("RELAX_QWEN38_QSA_GROUPED_BACKWARD_WARPS", "4"))
        assert backward_warps in (4, 8, 16)
        backward_stages = int(os.environ.get("RELAX_QWEN38_QSA_GROUPED_BACKWARD_STAGES", "1"))
        if backward_stages not in (1, 2, 3):
            raise ValueError("Grouped QSA backward stages must be 1, 2 or 3")
        ctx.backward_launch = dict(launch, num_warps=backward_warps)
        if bf16_products:
            ctx.backward_launch["num_stages"] = backward_stages
        return out.to(q.dtype)

    @staticmethod
    def backward(ctx, grad):
        q, k, v, indices, out, lse = ctx.saved_tensors
        dq = torch.empty_like(q, dtype=torch.float32)
        dk = torch.zeros_like(k, dtype=torch.float32)
        dv = torch.zeros_like(v, dtype=torch.float32)
        _backward[(q.shape[0], k.shape[1])](
            q, k, v, indices, out, lse, grad.contiguous(), dq, dk, dv, **ctx.params, **ctx.backward_launch
        )
        if ctx.gradient_group is not None:
            import torch.distributed as dist

            combined = torch.cat((dk, dv), dim=0)
            dist.all_reduce(combined, group=ctx.gradient_group)
            combined.div_(ctx.gradient_group.size())
            dk, dv = combined.chunk(2, dim=0)
        return (dq.to(q.dtype), dk.to(k.dtype), dv.to(v.dtype), None, None, None, None, None)[
            : len(ctx.needs_input_grad)
        ]


def attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, indices: torch.Tensor, scale: float) -> torch.Tensor:
    return _Grouped.apply(q, k, v, indices, scale)


def attention_bf16(q, k, v, indices, scale, block_k=64, gradient_group=None):
    return _Grouped.apply(q, k, v, indices, scale, True, block_k, gradient_group)
