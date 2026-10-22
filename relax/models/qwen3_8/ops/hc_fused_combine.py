# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Eligibility for the fused HC residual write."""

import torch


def supported(
    residual: torch.Tensor, gates: torch.Tensor, output: torch.Tensor, bias: torch.Tensor | None, n: int
) -> bool:
    return (
        bias is None
        and residual.is_cuda
        and residual.dtype in (torch.bfloat16, torch.float16, torch.float32)
        and output.dtype == residual.dtype
        and gates.dtype == torch.float32
        and output.device == residual.device == gates.device
        and n in (1, 4)
        and residual.ndim == output.ndim == gates.ndim == 3
        and residual.shape[:2] == output.shape[:2] == gates.shape[:2]
        and residual.shape[-1] == n * output.shape[-1]
        and 0 < output.shape[-1] <= 4096
        and gates.shape[-1] == n
    )
