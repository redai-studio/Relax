# Copyright (c) 2026 Relax Authors. All Rights Reserved.
# Copyright 2025-2026 RadixArk, Inc. (miles).
# SPDX-License-Identifier: Apache-2.0
# Adapted for Relax tensor layouts and explicit FP32 rounding.
"""Gated residual write and its first-order gradients, without intermediate
streams."""

import torch
import triton
import triton.language as tl
from torch.autograd.function import once_differentiable
from triton.language.extra.cuda import libdevice


@triton.jit
def _combine_fwd(R, Y, A, O, C: tl.constexpr, N: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    stream = tl.program_id(1)
    c = tl.arange(0, BLOCK)
    offset = (row * N + stream) * C + c
    r = tl.load(R + offset, c < C, other=0).to(tl.float32)
    y = tl.load(Y + row * C + c, c < C, other=0).to(tl.float32)
    a = tl.load(A + row * N + stream).to(tl.float32)
    tl.store(O + offset, libdevice.add_rn(r, libdevice.mul_rn(a, y)), c < C)


@triton.jit
def _combine_bwd(DO, Y, A, DY, DA, C: tl.constexpr, N: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    c = tl.arange(0, BLOCK)
    y = tl.load(Y + row * C + c, c < C, other=0).to(tl.float32)
    dy = tl.full((BLOCK,), 0, tl.float32)
    for stream in tl.static_range(N):
        grad = tl.load(DO + (row * N + stream) * C + c, c < C, other=0).to(tl.float32)
        a = tl.load(A + row * N + stream).to(tl.float32)
        dy += a * grad
        tl.store(DA + row * N + stream, tl.sum(grad * y, 0))
    tl.store(DY + row * C + c, dy, c < C)


class _Combine(torch.autograd.Function):
    @staticmethod
    def forward(ctx, residual: torch.Tensor, output: torch.Tensor, inject: torch.Tensor, n: int) -> torch.Tensor:
        r = residual.reshape(-1, residual.shape[-1]).contiguous()
        y = output.reshape(-1, output.shape[-1]).contiguous()
        a = inject.reshape(-1, n).contiguous()
        result = torch.empty_like(r)
        if r.shape[0]:
            _combine_fwd[(r.shape[0], n)](
                r, y, a, result, y.shape[-1], n, triton.next_power_of_2(y.shape[-1]), enable_fp_fusion=False
            )
        ctx.save_for_backward(y, a)
        ctx.n = n
        ctx.shapes = residual.shape, output.shape, inject.shape
        return result.reshape(residual.shape)

    @staticmethod
    @once_differentiable
    def backward(ctx, grad: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, None]:
        y, a = ctx.saved_tensors
        grad = grad.reshape(-1, ctx.n * y.shape[-1]).contiguous()
        dy, da = torch.empty_like(y), torch.empty_like(a)
        if grad.shape[0]:
            _combine_bwd[(grad.shape[0],)](
                grad, y, a, dy, da, y.shape[-1], ctx.n, triton.next_power_of_2(y.shape[-1]), enable_fp_fusion=False
            )
        sr, sy, sa = ctx.shapes
        return grad.reshape(sr), dy.reshape(sy), da.reshape(sa), None


def combine(residual: torch.Tensor, output: torch.Tensor, inject: torch.Tensor, n: int) -> torch.Tensor:
    return _Combine.apply(residual, output, inject, n)
