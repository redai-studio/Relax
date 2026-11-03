# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Weight-only MXFP4 QDQ matching the current DSv4 Bridge exporter.

The functions return detached, independent tensors. The caller owns the STE or
custom backward; master weights are never modified here. Blocks containing
NaN/Inf pass through unchanged so training failures remain observable.
"""

from functools import lru_cache
from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    import torch


def _validate_weight(weight: "torch.Tensor") -> None:
    import torch

    if weight.ndim != 2 or weight.shape[1] % 32:
        raise ValueError("MXFP4 QDQ requires a 2-D weight with K divisible by 32")
    if weight.dtype not in (torch.bfloat16, torch.float32):
        raise TypeError("MXFP4 QDQ supports bfloat16 and float32 weights")
    if not weight.is_contiguous():
        raise ValueError("MXFP4 QDQ requires a contiguous weight")


def mxfp4_qdq_reference(weight: "torch.Tensor") -> "torch.Tensor":
    """Small-tensor eager reference, including Bridge's midpoint and zero
    rules."""
    import torch

    _validate_weight(weight)
    with torch.no_grad():
        grouped = weight.detach().float().reshape(-1, 32)
        finite_group = torch.isfinite(grouped).all(dim=1, keepdim=True)
        quantized_input = torch.where(finite_group, grouped, 0.0)
        amax = quantized_input.abs().amax(dim=1, keepdim=True)
        scale = torch.where(amax > 0, amax / 6.0, torch.ones_like(amax))
        exponent = torch.ceil(torch.log2(scale.clamp(min=2.0**-127, max=2.0**127))).to(torch.int32)
        # Construct exact powers of two: CUDA exp2(-127) need not return the
        # exact subnormal value produced by the CPU checkpoint exporter.
        scale_bits = torch.where(exponent == -127, 0x00400000, (exponent + 127) << 23)
        scale = scale_bits.view(torch.float32)
        normalized = quantized_input / scale
        boundaries = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0], device=weight.device)
        codes = torch.bucketize(normalized.abs(), boundaries)
        codes = codes | ((normalized < 0).to(torch.int64) * 8)
        table = torch.tensor(
            [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
            dtype=torch.float32,
            device=weight.device,
        )
        result = (table[codes] * scale).reshape(weight.shape).to(weight.dtype)
        return torch.where(finite_group.expand_as(grouped).reshape(weight.shape), result, weight.detach())


@lru_cache(maxsize=1)
def _cuda_kernel() -> Any:
    import triton
    import triton.language as tl
    from triton.language.extra.cuda import libdevice

    @triton.jit
    def kernel(src, dst, groups: tl.constexpr, GROUPS_PER_PROGRAM: tl.constexpr):
        group = tl.program_id(0) * GROUPS_PER_PROGRAM + tl.arange(0, GROUPS_PER_PROGRAM)
        offset = group[:, None] * 32 + tl.arange(0, 32)[None, :]
        original = tl.load(src + offset, mask=group[:, None] < groups, other=0)
        value = original.to(tl.float32)
        finite_group = tl.sum((~(tl.abs(value) < float("inf"))).to(tl.int32), axis=1) == 0
        quantized_input = tl.where(finite_group[:, None], value, 0.0)
        amax = tl.max(tl.abs(quantized_input), axis=1)
        scale = tl.where(amax > 0, tl.div_rn(amax, 6.0), 1.0)
        minimum_scale = tl.full((), 2.0**-127, tl.float32)
        maximum_scale = tl.full((), 2.0**127, tl.float32)
        scale = tl.minimum(tl.maximum(scale, minimum_scale), maximum_scale)
        exponent = libdevice.ceil(libdevice.log2(scale)).to(tl.int32)
        scale_bits = tl.where(exponent == -127, 0x00400000, (exponent + 127) << 23)
        scale = scale_bits.to(tl.float32, bitcast=True)
        normalized = tl.div_rn(quantized_input, scale[:, None])
        magnitude = tl.abs(normalized)
        code = (
            (magnitude > 0.25).to(tl.int32)
            + (magnitude > 0.75).to(tl.int32)
            + (magnitude > 1.25).to(tl.int32)
            + (magnitude > 1.75).to(tl.int32)
            + (magnitude > 2.5).to(tl.int32)
            + (magnitude > 3.5).to(tl.int32)
            + (magnitude > 5.0).to(tl.int32)
        )
        decoded = tl.where(code <= 4, code * 0.5, tl.where(code == 5, 3.0, tl.where(code == 6, 4.0, 6.0)))
        result = decoded * scale[:, None]
        sign = tl.where(normalized < 0, 0x80000000, 0).to(tl.uint32)
        result = (result.to(tl.uint32, bitcast=True) | sign).to(tl.float32, bitcast=True)
        result = tl.where(finite_group[:, None], result.to(original.dtype), original)
        tl.store(dst + offset, result, mask=group[:, None] < groups)

    return kernel


def mxfp4_qdq(weight: "torch.Tensor") -> "torch.Tensor":
    """QDQ contiguous BF16/FP32 weights along K32 without modifying the input.

    CUDA uses one fused kernel, one output allocation and no GPU-to-CPU sync.
    Blocks containing NaN/Inf pass through unchanged, intentionally differing
    from the exporter's handling of invalid weights. There is no autograd
    backward; use an STE at the caller if differentiating through this
    operation.
    """
    import torch

    _validate_weight(weight)
    if not weight.is_cuda:
        return mxfp4_qdq_reference(weight)
    output = torch.empty_like(weight)
    groups = weight.numel() // 32
    if groups:
        _cuda_kernel()[((groups + 7) // 8,)](weight, output, groups, GROUPS_PER_PROGRAM=8, num_warps=4)
    return output
