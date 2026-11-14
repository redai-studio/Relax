# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""CP1 partial RoPE, retaining native BF16 multiplication rounding."""

import torch
import triton
import triton.language as tl


@triton.jit
def _apply(
    X,
    C,
    S,
    Y,
    N,
    H: tl.constexpr,
    D: tl.constexpr,
    R: tl.constexpr,
    XS: tl.constexpr,
    XH: tl.constexpr,
    XD: tl.constexpr,
    BACK: tl.constexpr,
    BLOCK: tl.constexpr,
):
    o = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = o < N * H * D
    d = o % D
    h = o // D % H
    n = o // (D * H)
    rot = d < R
    partner = tl.where(d < R // 2, d + R // 2, d - R // 2)
    x = tl.load(X + n * XS + h * XH + d * XD, valid, 0).to(tl.float32)
    p = tl.load(X + n * XS + h * XH + partner * XD, valid & rot, 0).to(tl.float32)
    c = tl.load(C + n * R + d, valid & rot, 0).to(tl.float32)
    if BACK:
        s = tl.load(S + n * R + partner, valid & rot, 0).to(tl.float32)
        sign = tl.where(d < R // 2, 1.0, -1.0)
    else:
        s = tl.load(S + n * R + d, valid & rot, 0).to(tl.float32)
        sign = tl.where(d < R // 2, -1.0, 1.0)
    a = (x * c).to(tl.bfloat16).to(tl.float32)
    b = (p * sign * s).to(tl.bfloat16).to(tl.float32)
    y = tl.where(rot, a + b, x)
    if BACK:
        # Native partial-slice backward accumulates a zero-padded tail/rotary
        # contribution, canonicalizing signed zero in the final input gradient.
        y = tl.where(y == 0.0, 0.0, y)
    tl.store(Y + o, y, valid)


class Fused(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, c, s):
        assert x.dtype == torch.bfloat16 and x.is_cuda
        assert x.ndim in (3, 4) and (x.ndim == 3 or x.shape[1] == 1)
        assert c.shape == s.shape and c.shape[0] == x.shape[0]
        assert c.dtype == s.dtype == x.dtype and c.is_contiguous() and s.is_contiguous()
        assert not c.requires_grad and not s.requires_grad
        n, h, d = x.shape[0], x.shape[-2], x.shape[-1]
        r = c.shape[-1]
        assert r % 2 == 0 and 0 < r <= d and all(v == 1 for v in c.shape[1:-1])
        y = torch.empty(x.shape, device=x.device, dtype=x.dtype)
        _apply[(triton.cdiv(n * h * d, 1024),)](
            x, c, s, y, n, h, d, r, x.stride(0), x.stride(-2), x.stride(-1), False, 1024, enable_fp_fusion=False
        )
        ctx.save_for_backward(c, s)
        ctx.geometry = (n, h, d, r)
        return y

    @staticmethod
    def backward(ctx, dy):
        c, s = ctx.saved_tensors
        n, h, d, r = ctx.geometry
        dx = torch.empty(dy.shape, device=dy.device, dtype=dy.dtype)
        _apply[(triton.cdiv(n * h * d, 1024),)](
            dy, c, s, dx, n, h, d, r, dy.stride(0), dy.stride(-2), dy.stride(-1), True, 1024, enable_fp_fusion=False
        )
        return dx, None, None


def apply(x, freqs):
    assert freqs.shape[0] == x.shape[0] and not freqs.requires_grad
    c = torch.cos(freqs).to(x.dtype).contiguous()
    s = torch.sin(freqs).to(x.dtype).contiguous()
    return Fused.apply(x, c, s)
