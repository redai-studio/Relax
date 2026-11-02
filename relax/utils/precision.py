# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Linear projections with low-precision operands and direct FP32 output."""

import torch


def linear_fp32_output(input: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor:
    """Produce FP32 GEMM output without rounding logits through BF16.

    Call under no_grad or inside an autograd Function. Return the base
    allocation because vocab-parallel cross entropy may modify its input in
    place.
    """
    if not input.is_cuda or input.dtype not in (torch.bfloat16, torch.float16) or weight.dtype != input.dtype:
        raise ValueError("direct FP32 output requires CUDA BF16/FP16 input and matching weight dtype")
    output = torch.empty((*input.shape[:-1], weight.shape[0]), dtype=torch.float32, device=input.device)
    torch.mm(
        input.reshape(-1, input.shape[-1]),
        weight.t(),
        out=output.view(-1, weight.shape[0]),
        out_dtype=torch.float32,
    )
    if bias is not None:
        output.add_(bias)
    return output
