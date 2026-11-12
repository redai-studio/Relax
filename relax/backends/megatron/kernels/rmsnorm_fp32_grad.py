# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from types import MethodType

import torch
import triton
import triton.language as tl
from torch import nn
from transformer_engine.pytorch import RMSNorm
from transformer_engine.pytorch.constants import TE_DType
from transformer_engine.pytorch.cpp_extensions import rmsnorm_bwd, rmsnorm_fwd


@triton.jit
def _weight_grad_partials(x, dy, rsigma, partials, rows, width, block_rows: tl.constexpr, block_cols: tl.constexpr):
    row = tl.program_id(0) * block_rows + tl.arange(0, block_rows)
    col = tl.program_id(1) * block_cols + tl.arange(0, block_cols)
    offsets = row[:, None].to(tl.int64) * width + col[None, :]
    mask = (row[:, None] < rows) & (col[None, :] < width)
    x_value = tl.load(x + offsets, mask, 0).to(tl.float32)
    dy_value = tl.load(dy + offsets, mask, 0).to(tl.float32)
    inverse_rms = tl.load(rsigma + row, row < rows, 0)
    dw = tl.sum((x_value * inverse_rms[:, None]) * dy_value, axis=0)
    tl.store(partials + tl.program_id(0) * width + col, dw, col < width)


class _RMSNormFP32WeightGrad(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        weight: torch.Tensor,
        eps: float,
        zero_centered_gamma: bool,
        forward_sm_margin: int,
        backward_sm_margin: int,
    ) -> torch.Tensor:
        flat_x = x.contiguous().view(-1, weight.numel())
        compute_weight = weight.to(x.dtype)
        y, _, rsigma = rmsnorm_fwd(
            flat_x, compute_weight, eps, None, None, TE_DType[x.dtype], forward_sm_margin, zero_centered_gamma
        )
        ctx.save_for_backward(flat_x, compute_weight, rsigma)
        ctx.zero_centered_gamma = zero_centered_gamma
        ctx.backward_sm_margin = backward_sm_margin
        return y.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor | None, ...]:
        x, compute_weight, rsigma = ctx.saved_tensors
        dy = grad_output.contiguous().view_as(x)
        dx = None
        if ctx.needs_input_grad[0]:
            dx, _ = rmsnorm_bwd(dy, x, rsigma, compute_weight, ctx.backward_sm_margin, ctx.zero_centered_gamma)
            dx = dx.view_as(grad_output)
        dw = None
        if ctx.needs_input_grad[1]:
            rows, width = x.shape
            block_rows, block_cols = 128, 32
            row_blocks = triton.cdiv(rows, block_rows)
            if row_blocks:
                partials = torch.empty((row_blocks, width), dtype=torch.float32, device=x.device)
                _weight_grad_partials[(row_blocks, triton.cdiv(width, block_cols))](
                    x, dy, rsigma, partials, rows, width, block_rows, block_cols
                )
                dw = partials.sum(dim=0)
            else:
                dw = torch.zeros_like(compute_weight, dtype=torch.float32)
        return dx, dw, None, None, None, None


def _rmsnorm_forward(module: RMSNorm, x: torch.Tensor) -> torch.Tensor:
    if module.weight.dtype != torch.float32:
        raise RuntimeError("Megatron must preserve the FP32 RMSNorm parameter through mixed-precision wrapping")
    if not x.is_cuda or x.shape[-1:] != module.weight.shape:
        raise ValueError("FP32 RMSNorm weight gradients require a CUDA tensor with the original hidden dimension")
    margins = module._sm_margins
    return _RMSNormFP32WeightGrad.apply(
        x.to(torch.bfloat16),
        module.weight,
        module.eps,
        module.zero_centered_gamma,
        margins["forward" if torch.is_grad_enabled() else "inference"],
        margins["backward"],
    )


def enable_rmsnorm_fp32_weight_grad(model: nn.Module) -> list[str]:
    """Use the TE forward/input backward and accumulate standalone norm weight
    gradients in FP32."""
    from megatron.core.transformer.module import mark_keep_in_fp32

    updated = []
    for name, module in model.named_modules():
        if type(module) is not RMSNorm or not module.weight.requires_grad:
            continue
        # These are the standalone attention norms. TE fused sequences call
        # child operations directly and do not use their individual forward methods.
        if name.rsplit(".", 1)[-1] not in {"out_norm", "q_layernorm", "k_layernorm"}:
            continue
        if getattr(module.weight, "keep_in_fp32", False):
            continue
        if module.weight.ndim != 1:
            raise ValueError(f"Expected a one-dimensional standalone RMSNorm weight for {name}")
        module.weight.data = module.weight.detach().float()
        mark_keep_in_fp32(module.weight)
        module.forward = MethodType(_rmsnorm_forward, module)
        updated.append(name)
    return updated
