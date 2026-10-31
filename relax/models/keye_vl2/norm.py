# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""RMSNorm with the same row reduction for prefill, decode and training."""

import torch
import triton
import triton.language as tl
from transformer_engine.pytorch import RMSNorm as TERMSNorm


@triton.jit
def _forward(X, W, Y, R, N: tl.constexpr, EPS: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    x = tl.load(X + row * N + col, col < N, other=0).to(tl.float32)
    w = tl.load(W + col, col < N, other=0).to(tl.float32)
    inverse = tl.rsqrt(tl.sum(x * x, 0) / N + EPS)
    tl.store(Y + row * N + col, (x * inverse) * w, col < N)
    tl.store(R + row, inverse)


class _RMSNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
        x = x.contiguous()
        width = x.shape[-1]
        output = torch.empty_like(x)
        inverse = torch.empty(x.numel() // width, device=x.device, dtype=torch.float32)
        _forward[(inverse.numel(),)](
            x,
            weight,
            output,
            inverse,
            width,
            eps,
            triton.next_power_of_2(width),
            num_warps=4,
            enable_fp_fusion=False,
        )
        ctx.save_for_backward(x, weight, inverse)
        return output

    @staticmethod
    def backward(ctx, grad: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, None]:
        x, weight, inverse = ctx.saved_tensors
        shape = x.shape
        x = x.reshape(-1, shape[-1]).float()
        dy = grad.reshape_as(x).float()
        inverse = inverse[:, None]
        weighted = dy * weight.float()
        dx = (weighted - x * (weighted * x).mean(-1, keepdim=True) * inverse.square()) * inverse
        dw = (dy * x * inverse).sum(0)
        return dx.reshape(shape).to(grad.dtype), dw.to(weight.dtype), None


class RMSNorm(TERMSNorm):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _RMSNorm.apply(x, self.weight, self.eps)
