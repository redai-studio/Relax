# Copyright (c) 2026 Relax Authors. All Rights Reserved.
# Copyright 2025-2026 RadixArk, Inc. (miles).
# SPDX-License-Identifier: Apache-2.0
# Adapted for Relax chunking, TP gradients and MTP metrics.
"""Opt-in per-token linear CE; preserve native TP dgrad and FP32 main_grad
accumulation.

Forward fuses GEMM, BF16 rounding, LSE and target selection. Backward
recomputes GEMM with a probability-gradient epilogue, retaining no vocabulary
activations. MTP can request temporary logits for its existing acceptance
metrics.
"""

from functools import lru_cache
from typing import Any

import torch
import torch.distributed as dist
from torch.autograd.function import once_differentiable

from relax.utils.env import Envs
from relax.utils.logging_utils import get_logger


@lru_cache(maxsize=2)
def _log_enabled(metric_logits: bool) -> None:
    get_logger(__name__).info(
        "Fused linear CE active: Blackwell BF16, per-token loss, temporary metric logits=%s", metric_logits
    )


def supported(
    layer: Any, hidden: torch.Tensor, weight: torch.Tensor | None, gather_output: bool | None = None
) -> bool:
    if not Envs.RELAX_SFT_FUSED_LINEAR_CE or not hidden.is_cuda or weight is None:
        return False
    from megatron.core.tensor_parallel.layers import ColumnParallelLinear

    # Functional GEMM must not bypass adapters, biases, quantization or arbitrary hooks.
    if type(layer) is not ColumnParallelLinear:
        try:
            from megatron.core.transformer.linear_cross_entropy import LinearCrossEntropyModule
        except ImportError:
            return False
        # New MCore uses this exact native head even when CE fusion is off.
        # Its default forward delegates to ColumnParallelLinear; arbitrary
        # subclasses can override that contract and must remain excluded.
        if type(layer) is not LinearCrossEntropyModule:
            return False
    if layer.bias is not None or (gather_output if gather_output is not None else layer.gather_output):
        return False
    if layer._forward_hooks or layer._backward_hooks or layer._backward_pre_hooks:
        return False
    if any(getattr(layer, name, False) for name in ("disable_grad_reduce", "explicit_expert_comm", "is_expert")):
        return False
    from torch.nn.modules import module as module_hooks

    if any(
        getattr(module_hooks, name)
        for name in (
            "_global_forward_pre_hooks",
            "_global_forward_hooks",
            "_global_backward_pre_hooks",
            "_global_backward_hooks",
        )
    ):
        return False
    for hook in layer._forward_pre_hooks.values():
        if (
            getattr(hook, "__module__", "") != "megatron.core.distributed.distributed_data_parallel"
            or getattr(hook, "__qualname__", "") != "DistributedDataParallel._make_forward_pre_hook.<locals>.hook"
        ):
            return False
    if getattr(layer.config, "defer_embedding_wgrad_compute", False):
        return False
    return (
        torch.cuda.get_device_capability(hidden.device)[0] == 10
        and hidden.device == weight.device
        and weight.dtype == torch.bfloat16
        and weight.ndim == 2
        and weight.is_contiguous()
        and hidden.shape[-1] == weight.shape[-1]
        and weight.shape[0] % 8 == weight.shape[1] % 8 == 0
    )


class _LinearCE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hidden, weight, labels, tp_group, inverse_temperature, accumulate, return_logits):
        from megatron.core.utils import ensure_params_ready

        from .fused_linear_ce_ops import linear_ce_forward

        ctx.set_materialize_grads(False)
        ensure_params_ready([weight])
        x = hidden.reshape(-1, hidden.shape[-1]).to(weight.dtype).contiguous()
        labels = labels.reshape(-1).contiguous()
        size = dist.get_world_size(tp_group) if tp_group is not None else 1
        rank = dist.get_rank(tp_group) if tp_group is not None else 0
        start = rank * weight.shape[0]
        local_target = labels - start
        local_target = torch.where((local_target >= 0) & (local_target < weight.shape[0]), local_target, -100)
        target_logit = x.new_zeros(x.shape[0], dtype=torch.float32)
        result = linear_ce_forward(
            x,
            weight.T,
            target=local_target,
            target_logit=target_logit,
            inverse_temperature=inverse_temperature,
            store_d=return_logits,
            tuned=False,
        )
        local_lse = result["lse"]
        if size > 1:
            maximum = local_lse.clone()
            dist.all_reduce(maximum, op=dist.ReduceOp.MAX, group=tp_group)
            stats = torch.stack(((local_lse - maximum).exp(), target_logit))
            dist.all_reduce(stats, op=dist.ReduceOp.SUM, group=tp_group)
            lse, target_logit = maximum + stats[0].log(), stats[1]
        else:
            lse = local_lse
        losses = torch.where(labels == -100, 0.0, lse - target_logit)
        ctx.save_for_backward(x, weight, labels, local_target, lse)
        ctx.weight_ref = weight  # Preserve main_grad attributes through checkpoint saved-tensor hooks.
        ctx.tp_group, ctx.tp_size = tp_group, size
        ctx.inverse_temperature, ctx.accumulate = inverse_temperature, accumulate
        ctx.hidden_shape, ctx.hidden_dtype = hidden.shape, hidden.dtype
        logits = result.get("D") if return_logits else None
        if logits is not None:
            ctx.mark_non_differentiable(logits)
        return losses.reshape(hidden.shape[:-1]), logits

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_loss, _grad_logits):
        if grad_loss is None:
            return (None,) * 7
        from .fused_linear_ce_ops import linear_ce_backward

        x, weight, labels, local_target, lse = ctx.saved_tensors
        grad = grad_loss.reshape(-1).float().contiguous()
        grad = torch.where(labels == -100, 0.0, grad)
        result = linear_ce_backward(
            x,
            weight.T,
            lse=lse,
            grad=grad,
            target=local_target,
            vocab=torch.arange(weight.shape[0], device=x.device, dtype=torch.int32),
            inverse_temperature=ctx.inverse_temperature,
            tuned=False,
        )
        dlogits = result["D"]
        dx = None
        if ctx.needs_input_grad[0]:
            dx = dlogits @ weight
            if ctx.tp_size > 1:
                # SP was gathered with tensor_parallel_output_grad=False. Sum
                # vocabulary contributions here; the gather backward only splits.
                dist.all_reduce(dx, op=dist.ReduceOp.SUM, group=ctx.tp_group)
            dx = dx.reshape(ctx.hidden_shape).to(ctx.hidden_dtype)
        dw = None
        if ctx.needs_input_grad[1]:
            parameter = ctx.weight_ref
            if ctx.accumulate:
                from megatron.core.tensor_parallel.layers import fused_weight_gradient_mlp_cuda

                if parameter.main_grad.dtype != torch.float32:
                    raise ValueError("Fused linear CE requires an FP32 main_grad buffer")
                fused_weight_gradient_mlp_cuda.wgrad_gemm_accum_fp32(x, dlogits, parameter.main_grad)
                if hasattr(parameter, "grad_added_to_main_grad"):
                    dw = (
                        torch.zeros_like(weight)
                        if getattr(parameter, "zero_out_wgrad", False)
                        else torch.empty_like(weight)
                    )
                    parameter.grad_added_to_main_grad = True
            else:
                dw = (dlogits.T @ x).to(weight.dtype)
        return dx, dw, None, None, None, None, None


def linear_cross_entropy(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    labels: torch.Tensor,
    tp_group: Any,
    *,
    temperature: float = 1.0,
    accumulate: bool = False,
    return_logits: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Return unreduced losses in hidden row order; SP gathering is the
    caller's job."""
    if Envs.RELAX_SFT_FUSED_LINEAR_CE and temperature == 1.0:
        from .kernels.linear_cross_entropy import linear_cross_entropy as backport_linear_cross_entropy

        return backport_linear_cross_entropy(
            hidden, weight, labels, tp_group, accumulate=accumulate, return_logits=return_logits
        )
    if temperature <= 0:
        raise ValueError("Linear CE temperature must be positive")
    if hidden.numel() == 0:
        raise ValueError("Linear CE expects at least one row")
    if accumulate and weight.requires_grad and not hasattr(weight, "main_grad"):
        raise ValueError("Fused gradient accumulation requires the original head Parameter.main_grad")
    result = _LinearCE.apply(hidden, weight, labels, tp_group, 1.0 / temperature, accumulate, return_logits)
    _log_enabled(return_logits)
    return result
