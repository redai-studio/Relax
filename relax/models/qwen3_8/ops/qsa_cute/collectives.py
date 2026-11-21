# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Explicit-group CP all-gather with summed remote-key gradients."""

from typing import Any

import torch
import torch.distributed as dist
from torch import Tensor

from relax.models.qwen3_8.ops.qsa_cute.profiling import profile_range


class _GatherSequence(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, value: Tensor, group: dist.ProcessGroup) -> Tensor:
        if group is None:
            raise ValueError("QSA CP requires an explicit process group")
        size = dist.get_world_size(group)
        ctx.group = group
        ctx.size = size
        if size == 1:
            return value
        output = value.new_empty((size * value.shape[0], *value.shape[1:]))
        with profile_range("CP/all_gather"):
            dist.all_gather_into_tensor(output, value.contiguous(), group=group)
        return output

    @staticmethod
    def backward(ctx: Any, grad: Tensor) -> tuple[Tensor, None]:
        if ctx.size == 1:
            return grad, None
        local = grad.new_empty((grad.shape[0] // ctx.size, *grad.shape[1:]))
        with profile_range("CP/reduce_scatter"):
            dist.reduce_scatter_tensor(local, grad.contiguous(), group=ctx.group)
        return local, None


def gather_sequence(value: Tensor, group: dist.ProcessGroup, reorder: Tensor | None = None) -> Tensor:
    gathered = _GatherSequence.apply(value, group)
    with profile_range("CP/reorder"):
        return gathered if reorder is None else gathered.index_select(0, reorder)
