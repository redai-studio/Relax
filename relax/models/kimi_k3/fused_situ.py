# Copyright (c) 2026 Relax Authors. All Rights Reserved.
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Adapted from NVIDIA-NeMo/Automodel situ_triton.py at
# 97e7a4844ddb437f1b02a266980ab3859356001b (Apache-2.0).
# https://github.com/NVIDIA-NeMo/Automodel/blob/97e7a4844ddb437f1b02a266980ab3859356001b/nemo_automodel/components/models/kimi_k3/situ_triton.py
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed
# under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
# CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.

"""Fused unweighted SiTU for the Bridge K3 activation contract.

Unlike Automodel's weighted expert activation, routing multiplication remains
owned by Megatron. Both kernels tile rows and columns; the backward does not
need Automodel's full-row loop/reduction because there is no routing gradient.
FP32 intermediates are retained; approximate transcendental math is opt-in.
Import this module lazily: Triton is only needed for the CUDA fast path.
"""

from typing import Any

import torch
import triton
import triton.language as tl
from torch import Tensor
from torch.autograd.function import once_differentiable
from triton.language.extra.cuda import libdevice


@triton.jit
def _tanh(x, FAST: tl.constexpr):
    if FAST:
        return tl.inline_asm_elementwise(
            "tanh.approx.f32 $0, $1;", "=r,r", [x], dtype=tl.float32, is_pure=True, pack=1
        )
    return libdevice.tanh(x)


@triton.jit
def _sigmoid(x, FAST: tl.constexpr):
    if FAST:
        denom = 1.0 + tl.exp2(x * -1.4426950408889634)
        return tl.inline_asm_elementwise(
            "rcp.approx.ftz.f32 $0, $1;", "=r,r", [denom], dtype=tl.float32, is_pure=True, pack=1
        )
    return tl.sigmoid(x)


@triton.jit
def _forward(
    X,
    Y,
    ROWS,
    HALF: tl.constexpr,
    BETA: tl.constexpr,
    LINEAR_BETA: tl.constexpr,
    FAST: tl.constexpr,
    BR: tl.constexpr,
    BC: tl.constexpr,
):
    rows = tl.program_id(0) * BR + tl.arange(0, BR)
    cols = tl.program_id(1) * BC + tl.arange(0, BC)
    mask = (rows[:, None] < ROWS) & (cols[None, :] < HALF)
    offsets = rows.to(tl.int64)[:, None] * (2 * HALF) + cols[None, :]
    gate = tl.load(X + offsets, mask, 0).to(tl.float32)
    up = tl.load(X + offsets + HALF, mask, 0).to(tl.float32)
    bounded_gate = BETA * _tanh(gate / BETA, FAST) * _sigmoid(gate, FAST)
    bounded_up = LINEAR_BETA * _tanh(up / LINEAR_BETA, FAST)
    tl.store(Y + rows.to(tl.int64)[:, None] * HALF + cols[None, :], bounded_gate * bounded_up, mask)


@triton.jit
def _backward(
    X,
    DY,
    DX,
    ROWS,
    HALF: tl.constexpr,
    BETA: tl.constexpr,
    LINEAR_BETA: tl.constexpr,
    FAST: tl.constexpr,
    BR: tl.constexpr,
    BC: tl.constexpr,
):
    rows = tl.program_id(0) * BR + tl.arange(0, BR)
    cols = tl.program_id(1) * BC + tl.arange(0, BC)
    mask = (rows[:, None] < ROWS) & (cols[None, :] < HALF)
    offsets = rows.to(tl.int64)[:, None] * (2 * HALF) + cols[None, :]
    gate = tl.load(X + offsets, mask, 0).to(tl.float32)
    up = tl.load(X + offsets + HALF, mask, 0).to(tl.float32)
    dy = tl.load(DY + rows.to(tl.int64)[:, None] * HALF + cols[None, :], mask, 0).to(tl.float32)
    tg = _tanh(gate / BETA, FAST)
    sg = _sigmoid(gate, FAST)
    tu = _tanh(up / LINEAR_BETA, FAST)
    a = BETA * tg * sg
    da = (1.0 - tg * tg) * sg + BETA * tg * sg * (1.0 - sg)
    dg = dy * (LINEAR_BETA * tu) * da
    du = dy * a * (1.0 - tu * tu)
    tl.store(DX + offsets, dg, mask)
    tl.store(DX + offsets + HALF, du, mask)


class _SiTU(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, inputs: Tensor, beta: float, linear_beta: float, fast_math: bool) -> Tensor:
        flat = inputs.reshape(-1, inputs.shape[-1]).contiguous()
        half = flat.shape[-1] // 2
        output = inputs.new_empty((*inputs.shape[:-1], half))
        if flat.shape[0] and half:
            grid = (triton.cdiv(flat.shape[0], 8), triton.cdiv(half, 256))
            _forward[grid](
                flat,
                output,
                flat.shape[0],
                half,
                beta,
                linear_beta,
                fast_math,
                8,
                256,
                num_warps=4,
                enable_fp_fusion=False,
            )
        ctx.save_for_backward(flat)
        ctx.input_shape = inputs.shape
        ctx.constants = (half, beta, linear_beta, fast_math)
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx: Any, grad_output: Tensor) -> tuple[Tensor, None, None, None]:
        (flat,) = ctx.saved_tensors
        half, beta, linear_beta, fast_math = ctx.constants
        grad = torch.empty_like(flat)
        if flat.shape[0] and half:
            grid = (triton.cdiv(flat.shape[0], 8), triton.cdiv(half, 256))
            _backward[grid](
                flat,
                grad_output.contiguous(),
                grad,
                flat.shape[0],
                half,
                beta,
                linear_beta,
                fast_math,
                8,
                256,
                num_warps=4,
                enable_fp_fusion=False,
            )
        return grad.reshape(ctx.input_shape), None, None, None


def fused_situ_and_mul(
    inputs: Tensor, beta: float = 4.0, linear_beta: float = 25.0, *, fast_math: bool = False
) -> Tensor:
    """SiTU with FP32 intermediates and the same output dtype as the input.

    Supports CUDA fp16/bf16/fp32 and arbitrary leading dimensions. First-order
    gradients only. Strided input and output gradients are copied as needed.
    Approximate math is experimental and disabled by default.
    """
    if not inputs.is_cuda or inputs.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError("Fused SiTU requires a CUDA fp16, bf16 or fp32 tensor.")
    if inputs.ndim == 0 or inputs.shape[-1] == 0 or inputs.shape[-1] % 2:
        raise ValueError("SiTU requires a positive even last dimension.")
    if beta <= 0 or linear_beta <= 0:
        raise ValueError("SiTU beta and linear_beta must be positive.")
    return _SiTU.apply(inputs, float(beta), float(linear_beta), fast_math)
