# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Preserve full-contraction BF16 rounding for TP attention projections."""

from typing import Any

import torch
import torch.distributed as dist
import torch.nn.functional as F
from megatron.core.extensions.transformer_engine import TERowParallelLinear
from megatron.core.tensor_parallel.mappings import reduce_scatter_to_sequence_parallel_region
from torch.autograd.function import FunctionCtx


def _gather_columns(tensor: torch.Tensor, group: dist.ProcessGroup) -> torch.Tensor:
    parts = [torch.empty_like(tensor) for _ in range(group.size())]
    dist.all_gather(parts, tensor.contiguous(), group=group)
    return torch.cat(parts, dim=-1)


class _FullRowProjection(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: FunctionCtx,
        inputs: torch.Tensor,
        weight: torch.Tensor,
        group: dist.ProcessGroup,
        fuse_wgrad: bool,
    ) -> torch.Tensor:
        ctx.weight_ref = weight
        ctx.fuse_wgrad = fuse_wgrad and getattr(weight, "main_grad", None) is not None
        ctx.save_for_backward(inputs, weight)
        full_inputs, full_weight = _gather_columns(inputs, group), _gather_columns(weight, group)
        if group.rank() == 0:
            return F.linear(full_inputs, full_weight)
        return inputs.new_zeros((*inputs.shape[:-1], weight.shape[0]))

    @staticmethod
    def backward(ctx: FunctionCtx, grad: torch.Tensor) -> tuple[Any, ...]:
        # SP reduce-scatter's backward has already gathered full-sequence dY.
        # Local columns then have the ordinary row-parallel gradient equations.
        inputs, weight = ctx.saved_tensors
        input_grad = weight_grad = None
        if ctx.needs_input_grad[0]:
            input_grad = grad.matmul(weight)
        if ctx.needs_input_grad[1]:
            left, right = grad.reshape(-1, grad.shape[-1]).t(), inputs.reshape(-1, inputs.shape[-1])
            weight_grad = torch.mm(left, right, out_dtype=torch.float32) if ctx.fuse_wgrad else torch.mm(left, right)
            if ctx.fuse_wgrad:
                ctx.weight_ref.main_grad.add_(weight_grad)
                ctx.weight_ref.grad_added_to_main_grad = True
                weight_grad = torch.zeros_like(ctx.weight_ref)
        return input_grad, weight_grad, None, None


class KeyeRowParallelLinear(TERowParallelLinear):
    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, None]:
        if self._tp_group.size() == 1:
            return super().forward(inputs)
        if self.use_bias or not self.config.sequence_parallel:
            raise ValueError("Keye TP attention projection requires a bias-free sequence-parallel layer")
        if inputs.dtype != torch.bfloat16 or any(
            getattr(self.config, name, False) for name in ("fp8", "fp4", "delay_wgrad_compute", "tp_comm_overlap")
        ):
            raise ValueError("Keye TP attention projection supports synchronous BF16 without delayed wgrad")
        output = _FullRowProjection.apply(
            inputs, self.weight, self._tp_group, self.config.gradient_accumulation_fusion
        )
        self.is_first_microbatch = False
        return reduce_scatter_to_sequence_parallel_region(output, group=self._tp_group), None
