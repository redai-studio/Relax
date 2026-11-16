# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Isolated BF16 TE ScaledSwiGLU forward and backward; no global mutation."""

import torch
import triton
import triton.language as tl
from transformer_engine.pytorch.ops.basic.swiglu import ScaledSwiGLU, clear_tensor_data


@triton.jit
def _backward(X, S, DY, DX, DS, Y0: tl.constexpr, Y1: tl.constexpr, N: tl.constexpr):
    row = tl.program_id(0)
    col = tl.arange(0, N)
    a = tl.load(X + row * (2 * N) + col).to(tl.float32)
    b = tl.load(X + row * (2 * N) + N + col).to(tl.float32)
    scale = tl.load(S + row).to(tl.float32)
    dy = tl.load(DY + row * Y0 + col * Y1).to(tl.float32)
    scaled_dy = (dy * scale).to(tl.bfloat16).to(tl.float32)
    sigmoid = 1.0 / (1.0 + tl.exp(-a))
    silu = a * sigmoid
    da = scaled_dy * b * (sigmoid + a * sigmoid * (1.0 - sigmoid))
    db = scaled_dy * silu
    tl.store(DX + row * (2 * N) + col, da)
    tl.store(DX + row * (2 * N) + N + col, db)
    # torch.linalg.vecdot on this real BF16 path materializes BF16 mul before sum.
    activation = (silu * b).to(tl.bfloat16).to(tl.float32)
    product = (activation * dy).to(tl.bfloat16).to(tl.float32)
    tl.store(DS + row, tl.sum(product, 0))


def backward(input_, scales, grad_output):
    assert input_.dtype == scales.dtype == grad_output.dtype == torch.bfloat16
    assert input_.ndim == grad_output.ndim == 2 and scales.ndim == 1
    assert input_.shape[1] == 1024 and grad_output.shape == (input_.shape[0], 512)
    assert input_.is_contiguous() and scales.is_contiguous()
    assert scales.numel() == input_.shape[0]
    dx = torch.empty_like(input_)
    ds = torch.empty_like(scales)
    if input_.shape[0]:
        _backward[(input_.shape[0],)](
            input_,
            scales,
            grad_output,
            dx,
            ds,
            grad_output.stride(0),
            grad_output.stride(1),
            512,
            num_warps=4,
            enable_fp_fusion=False,
        )
    return dx, ds


@triton.jit(do_not_specialize=["ROWS"])
def _forward(X, S, Y, ROWS, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    row = offsets // 512
    col = offsets % 512
    a = tl.load(X + row * 1024 + col, row < ROWS, other=0).to(tl.float32)
    b = tl.load(X + row * 1024 + 512 + col, row < ROWS, other=0).to(tl.float32)
    scale = tl.load(S + row, row < ROWS, other=0).to(tl.float32)
    sigmoid = 1.0 / (1.0 + tl.exp(-a))
    silu = a * sigmoid
    activation = (silu * b).to(tl.bfloat16).to(tl.float32)
    tl.store(Y + offsets, activation * scale, row < ROWS)


def forward(input_, scales):
    assert type(input_) is type(scales) is torch.Tensor
    assert input_.dtype == scales.dtype == torch.bfloat16
    assert input_.ndim == 2 and input_.shape[1] == 1024 and scales.ndim == 1
    assert input_.is_contiguous() and scales.is_contiguous()
    assert scales.numel() == input_.shape[0]
    output = torch.empty((input_.shape[0], 512), device=input_.device, dtype=input_.dtype)
    if input_.shape[0]:
        _forward[(triton.cdiv(output.numel(), 1024),)](
            input_, scales, output, input_.shape[0], 1024, num_warps=4, enable_fp_fusion=False
        )
    return output


class QwenScaledSwiGLU(ScaledSwiGLU):
    def fuser_forward(
        self,
        basic_op_ctxs,
        input_,
        *,
        basic_op_extra_inputs,
        prev_op_grad_output_quantizer,
        next_op_input_quantizer,
        basic_op_kwargs,
    ):
        from transformer_engine.pytorch.ops.basic.swiglu import is_cpu_offload_enabled, maybe_dequantize

        assert len(basic_op_ctxs) == len(basic_op_extra_inputs) == 1
        assert not is_cpu_offload_enabled()
        assert prev_op_grad_output_quantizer is None and next_op_input_quantizer is None
        extra = basic_op_extra_inputs[0][0]
        dtype = torch.get_autocast_dtype("cuda") if torch.is_autocast_enabled() else input_.dtype
        assert dtype == torch.bfloat16
        x = maybe_dequantize(input_, dtype)
        scales = maybe_dequantize(extra, dtype)
        out = forward(x, scales)
        ctx = basic_op_ctxs[0]
        if ctx.requires_grad:
            ctx.input_requires_grad = True
            ctx.extra_input_requires_grad = extra.requires_grad
            ctx.dtype = dtype
            ctx.save_for_backward(x, scales)
        return out, [()]

    def fuser_backward(self, basic_op_ctxs, grad_output, *, basic_op_grad_extra_outputs):
        assert len(basic_op_ctxs) == 1
        ctx = basic_op_ctxs[0]
        assert ctx.input_requires_grad and ctx.extra_input_requires_grad
        input_, scales = ctx.saved_tensors
        assert type(input_) is type(scales) is type(grad_output) is torch.Tensor
        dx, ds = backward(input_, scales, grad_output)
        clear_tensor_data(input_)
        return dx, [()], [(ds,)]
