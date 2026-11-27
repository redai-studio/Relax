# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Standalone grouped RMSNorm using the QHC kernels and checkpoint weight
layout."""

import torch

from .qhc_triton import _grouped_rmsnorm_backward_kernel, _grouped_rmsnorm_forward_kernel, _rms_launch_plan


class _GroupedRMSNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, group_size, eps):
        if x.device.type not in ("npu", "privateuseone") or x.shape[-1] % group_size:
            raise ValueError("Triton grouped RMSNorm requires NPU tensors and complete groups")
        x = x.contiguous()
        width, rows = x.shape[-1], x.numel() // x.shape[-1]
        groups = width // group_size
        output = torch.empty_like(x)
        rstd = torch.empty((groups, rows), dtype=torch.float32, device=x.device)
        if rows:
            br, bc, programs, even = _rms_launch_plan(rows, group_size, x.element_size(), x.device, backward=False)
            _grouped_rmsnorm_forward_kernel[(programs,)](
                output,
                x,
                weight,
                rstd,
                rows,
                width,
                eps,
                n_groups=groups,
                group_size=group_size,
                BLOCK_ROWS=br,
                BLOCK_COLS=bc,
                EVEN_COLS=even,
                STORE_RSTD=True,
            )
        ctx.save_for_backward(x, weight, rstd)
        ctx.group_size = group_size
        return output

    @staticmethod
    def backward(ctx, grad_output):
        x, weight, rstd = ctx.saved_tensors
        group_size = ctx.group_size
        width, rows = x.shape[-1], x.numel() // x.shape[-1]
        grad_x = torch.empty_like(x)
        if not rows:
            return grad_x, torch.zeros_like(weight), None, None
        br, bc, programs, even = _rms_launch_plan(rows, group_size, x.element_size(), x.device, backward=True)
        partial = torch.empty((programs, width), dtype=torch.float32, device=x.device)
        _grouped_rmsnorm_backward_kernel[(programs,)](
            grad_x,
            grad_output.contiguous(),
            x,
            weight,
            rstd,
            partial,
            rows,
            width,
            n_groups=width // group_size,
            group_size=group_size,
            total_size=width,
            BLOCK_ROWS=br,
            BLOCK_COLS=bc,
            EVEN_COLS=even,
        )
        return grad_x, partial.sum(0).to(weight.dtype), None, None


grouped_rmsnorm = _GroupedRMSNorm.apply
