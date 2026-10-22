# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""BF16 shared-expert scalar gate with native multiply rounding in backward."""

import torch
import triton
import triton.language as tl


@triton.jit
def forward_kernel(X, G, Y, H: tl.constexpr, B: tl.constexpr):
    row = tl.program_id(0)
    col = tl.arange(0, B)
    mask = col < H
    x = tl.load(X + row * H + col, mask, 0).to(tl.float32)
    g = tl.load(G + row).to(tl.float32)
    tl.store(Y + row * H + col, x * g, mask)


@triton.jit
def backward_kernel(X, G, D, DX, DG, H: tl.constexpr, B: tl.constexpr):
    row = tl.program_id(0)
    col = tl.arange(0, B)
    mask = col < H
    x = tl.load(X + row * H + col, mask, 0).to(tl.float32)
    g = tl.load(G + row).to(tl.float32)
    d = tl.load(D + row * H + col, mask, 0).to(tl.float32)
    tl.store(DX + row * H + col, d * g, mask)
    # Native MulBackward creates a BF16 product before sum_to_size reduction.
    product = (d * x).to(tl.bfloat16).to(tl.float32)
    tl.store(DG + row, tl.sum(product, axis=0))


class Gate(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, g, warps):
        assert x.dtype == g.dtype == torch.bfloat16 and x.is_cuda and g.is_cuda
        assert x.is_contiguous() and g.is_contiguous() and x.shape[:-1] == g.shape[:-1] and g.shape[-1] == 1
        assert x.device == g.device and x.shape[-1] == 2048 and warps in (4, 8)
        h = x.shape[-1]
        n = x.numel() // h
        assert n > 0
        y = torch.empty_like(x)
        forward_kernel[(n,)](x, g, y, h, triton.next_power_of_2(h), num_warps=warps)
        ctx.save_for_backward(x, g)
        ctx.warps = warps
        return y

    @staticmethod
    def backward(ctx, dy):
        x, g = ctx.saved_tensors
        dy = dy.contiguous()
        dx = torch.empty_like(x)
        dg = torch.empty_like(g)
        h = x.shape[-1]
        n = x.numel() // h
        backward_kernel[(n,)](x, g, dy, dx, dg, h, triton.next_power_of_2(h), num_warps=ctx.warps)
        return dx, dg, None


def fused_gate(x, g, warps=4):
    return Gate.apply(x, g, warps)
