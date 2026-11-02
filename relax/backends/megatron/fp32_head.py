# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Opt-in FP32 LM logits, retaining MCore's gradient and parallel semantics.

Reference: agi-megatron-lm tensor_parallel/layers.py output_dtype path.
"""

from functools import lru_cache
from types import MethodType
from typing import Any

import torch

from relax.utils.precision import linear_fp32_output


@lru_cache(maxsize=1)
def _autograd_functions() -> tuple[type, type]:
    from megatron.core.tensor_parallel import layers

    class Trainable(torch.autograd.Function):
        @staticmethod
        @layers.custom_fwd
        def forward(
            ctx: Any,
            input: torch.Tensor,
            weight: torch.Tensor,
            bias: torch.Tensor | None,
            fusion: bool,
            allreduce: bool,
            sequence_parallel: bool,
            buffer: list[torch.Tensor] | None,
            limit: int | None,
            group: Any,
        ) -> torch.Tensor:
            ctx.save_for_backward(input, weight)
            ctx.main_grad = getattr(weight, "main_grad", None) if fusion else None
            ctx.use_bias = bias is not None
            ctx.gradient_accumulation_fusion = fusion
            ctx.allreduce_dgrad = allreduce
            ctx.sequence_parallel = sequence_parallel
            ctx.grad_output_buffer = buffer
            ctx.wgrad_deferral_limit = limit
            ctx.tp_group = group
            if sequence_parallel:
                shape = list(input.shape)
                shape[0] *= group.size()
                total_input = layers.get_global_memory_buffer().get_tensor(shape, input.dtype, "mpu")
                layers.dist_all_gather_func(total_input, input, group=group)
            else:
                total_input = input
            return linear_fp32_output(total_input, weight, bias)

        @staticmethod
        @layers.custom_bwd
        def backward(ctx: Any, grad_output: torch.Tensor) -> tuple[Any, ...]:
            # Use native communication, deferred wgrad, main_grad accumulation
            # and DDP dummy-gradient hooks, with the original operand precision.
            return layers.LinearWithGradAccumulationAndAsyncCommunication.backward(
                ctx, grad_output.to(ctx.saved_tensors[0].dtype)
            )

    class Frozen(torch.autograd.Function):
        @staticmethod
        @layers.custom_fwd
        def forward(
            ctx: Any,
            input: torch.Tensor,
            weight: torch.Tensor,
            bias: torch.Tensor | None,
            allreduce: bool,
            group: Any,
        ) -> torch.Tensor:
            ctx.save_for_backward(weight)
            ctx.allreduce_dgrad = allreduce
            ctx.tp_group = group
            return linear_fp32_output(input, weight, bias)

        @staticmethod
        @layers.custom_bwd
        def backward(ctx: Any, grad_output: torch.Tensor) -> tuple[Any, ...]:
            return layers.LinearWithFrozenWeight.backward(ctx, grad_output.to(ctx.saved_tensors[0].dtype))

    return Trainable, Frozen


def _forward_impl(
    self: torch.nn.Module,
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    gradient_accumulation_fusion: bool,
    allreduce_dgrad: bool,
    sequence_parallel: bool,
    grad_output_buffer: list | None = None,
    wgrad_deferral_limit: int | None = None,
    tp_group: Any = None,
) -> torch.Tensor:
    trainable, frozen = _autograd_functions()
    if weight.requires_grad:
        return trainable.apply(
            input,
            weight,
            bias,
            gradient_accumulation_fusion,
            allreduce_dgrad,
            sequence_parallel,
            grad_output_buffer,
            wgrad_deferral_limit,
            tp_group,
        )
    if grad_output_buffer is not None or wgrad_deferral_limit is not None:
        raise ValueError("FP32 frozen head does not support deferred embedding wgrad")
    if sequence_parallel:
        from megatron.core.tensor_parallel.mappings import gather_from_sequence_parallel_region

        input = gather_from_sequence_parallel_region(input, tensor_parallel_output_grad=True, group=tp_group)
    return frozen.apply(input, weight, bias, allreduce_dgrad, tp_group)


def configure_fp32_lm_head(model: torch.nn.Module) -> int:
    """Install before DDP/optimizer setup; keep parameters and class forward.

    Changing _forward_impl also covers chunked MTP's explicit class.forward.
    """
    import inspect

    from megatron.core.tensor_parallel import layers

    expected = (
        "ctx",
        "input",
        "weight",
        "bias",
        "gradient_accumulation_fusion",
        "allreduce_dgrad",
        "sequence_parallel",
        "grad_output_buffer",
        "wgrad_deferral_limit",
        "tp_group",
    )
    if tuple(inspect.signature(layers.LinearWithGradAccumulationAndAsyncCommunication.forward).parameters) != expected:
        raise RuntimeError("FP32 LM head requires the validated MCore 0.19 linear autograd interface")
    count = 0
    for name, module in model.named_modules():
        if name.rsplit(".", 1)[-1] != "output_layer" or isinstance(module, torch.nn.Identity):
            continue
        if not isinstance(module, layers.ColumnParallelLinear):
            raise TypeError(f"FP32 LM head requires ColumnParallelLinear, got {type(module).__name__}")
        if not getattr(module, "_relax_fp32_output", False):
            module._forward_impl = MethodType(_forward_impl, module)
            module._relax_fp32_output = True
        count += 1
    return count
