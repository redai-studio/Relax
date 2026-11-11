# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Opt-in SFT roundoff control for changing head row counts.

The BLAS policy is process-wide, including body matmuls. The native head keeps
its BF16 GEMMs and weight-gradient implementation, with an FP32 TP SUM before
rounding dgrad back to BF16. This does not promise bitwise equality or training
convergence.
"""

from argparse import Namespace
from collections.abc import Iterator
from contextlib import contextmanager
from types import ModuleType
from typing import Any

import torch

from relax.utils.env import Envs
from relax.utils.logging_utils import get_logger


logger = get_logger(__name__)


def configure_sft_stable_gradients(args: Namespace) -> bool:
    """Fix the opt-in BLAS policy before model construction in this worker."""
    enabled = Envs.RELAX_SFT_STABLE_GRADIENTS
    args._relax_sft_stable_gradients = enabled
    args._relax_sft_stable_gradients_config = {}
    if not enabled:
        return False  # Older Torch versions must not touch the experimental API.
    if not (getattr(args, "loss_type", None) == "sft" and getattr(args, "sft_chunked_logits", False)):
        raise ValueError("RELAX_SFT_STABLE_GRADIENTS requires chunked SFT logits")
    if not getattr(args, "bf16", False):
        raise ValueError("RELAX_SFT_STABLE_GRADIENTS requires BF16 training")
    try:
        torch.backends.cuda.preferred_blas_library("cublaslt")
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = (False, False)
        effective = {
            "preferred_blas_library": str(torch.backends.cuda.preferred_blas_library()),
            "allow_bf16_reduced_precision_reduction": torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
            "allow_bf16_reduced_precision_reduction_split_k": (
                torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction_split_k
            ),
        }
    except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
        raise RuntimeError("SFT stable gradients requires Torch cuBLASLt and BF16 reduction/split-K controls") from exc
    if (
        "cublaslt" not in effective["preferred_blas_library"].lower()
        or effective["allow_bf16_reduced_precision_reduction"] is not False
        or effective["allow_bf16_reduced_precision_reduction_split_k"] is not False
    ):
        raise RuntimeError(f"SFT stable-gradient BLAS policy was not applied: {effective}")
    args._relax_sft_stable_gradients_config = effective
    logger.info("SFT stable gradients: process-wide body/head Torch BLAS policy=%s; head TP SUM=FP32", effective)
    return True


def _validate_head(layer: torch.nn.Module, weight: torch.Tensor | None, gather_output: bool | None) -> None:
    from megatron.core.tensor_parallel.layers import ColumnParallelLinear

    native_types = (ColumnParallelLinear,)
    try:
        from megatron.core.transformer.linear_cross_entropy import LinearCrossEntropyModule

        native_types += (LinearCrossEntropyModule,)
    except ImportError:
        pass
    reason = None
    if type(layer) not in native_types:
        reason = "an exact native ColumnParallelLinear/LinearCrossEntropyModule head is required"
    elif weight is None or weight is not getattr(layer, "weight", None):
        reason = "external or tied head weights are unsupported"
    elif layer.bias is not None or (layer.gather_output if gather_output is None else gather_output):
        reason = "bias and gathered logits are unsupported"
    elif any(getattr(layer, name, False) for name in ("disable_grad_reduce", "explicit_expert_comm", "is_expert")):
        reason = "custom/expert gradient communication is unsupported"
    elif (
        getattr(layer.config, "cross_entropy_loss_fusion", False)
        and getattr(layer.config, "cross_entropy_fusion_impl", None) == "linear"
    ):
        reason = "the MCore labels-based fused head is unsupported"
    elif getattr(layer.config, "defer_embedding_wgrad_compute", False) or any(
        getattr(layer, name, None) is not None for name in ("grad_output_buffer", "embedding_activation_buffer")
    ):
        reason = "deferred head weight gradients are unsupported"
    elif getattr(layer.config, "cpu_offloading", False) or getattr(
        getattr(layer.config, "_cpu_offloading_context", None), "inside_context", False
    ):
        reason = "head CPU offloading is unsupported"
    elif layer._forward_hooks or layer._backward_hooks or layer._backward_pre_hooks:
        reason = "custom head hooks are unsupported"
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
        reason = "global module hooks are unsupported"
    for hook in layer._forward_pre_hooks.values():
        if (
            getattr(hook, "__module__", "") != "megatron.core.distributed.distributed_data_parallel"
            or getattr(hook, "__qualname__", "") != "DistributedDataParallel._make_forward_pre_hook.<locals>.hook"
        ):
            reason = "only native DDP parameter-readiness pre-hooks are supported"
    if reason is not None:
        raise ValueError(f"SFT stable gradients: {reason}")
    if weight.dtype != torch.bfloat16 or weight.ndim != 2 or not weight.is_contiguous():
        raise ValueError("SFT stable gradients requires a contiguous BF16 head weight")
    if getattr(layer, "tp_group", None) is None:
        raise ValueError("SFT stable gradients requires an explicit head TP process group")
    if layer.gradient_accumulation_fusion and weight.requires_grad:
        if getattr(weight, "main_grad", None) is None or weight.main_grad.dtype != torch.float32:
            raise ValueError("SFT stable gradients requires the original head Parameter's FP32 main_grad")


def set_sft_stable_head(layer: torch.nn.Module | None, enabled: bool) -> None:
    """Install the static policy after DDP allocated the head's main_grad."""
    if layer is None:  # A non-last pipeline stage has no output head.
        return
    if enabled:
        _validate_head(layer, getattr(layer, "weight", None), None)
    layer._relax_sft_stable_gradients = enabled


def fused_head_precision_kwargs(
    layer: torch.nn.Module, module: ModuleType, weight: torch.Tensor | None, gather_output: bool | None = None
) -> dict[str, Any] | None:
    """None selects the native fallback when an older fused kernel lacks
    support."""
    if not getattr(layer, "_relax_sft_stable_gradients", False):
        return {}  # Preserve old fused call signatures when the policy is off.
    _validate_head(layer, weight, gather_output)
    return {"tp_dgrad_fp32": True} if getattr(module, "SUPPORTS_FP32_TP_DGRAD", False) else None


@contextmanager
def native_head_precision(
    layer: torch.nn.Module, input_: torch.Tensor, weight: torch.Tensor | None = None, gather_output: bool | None = None
) -> Iterator[torch.Tensor]:
    """Keep native GEMM/GAF; perform exactly one FP32 TP dgrad SUM.

    Callers already gathered SP with tensor_parallel_output_grad=False. Native
    backward captures the disabled communication flags before they are
    restored; the outer native TP mapping owns the only SUM.
    """
    if not getattr(layer, "_relax_sft_stable_gradients", False):
        yield input_
        return
    from megatron.core.tensor_parallel.mappings import copy_to_tensor_model_parallel_region
    from megatron.core.utils import ensure_params_ready

    resolved_weight = layer.weight if weight is None else weight
    _validate_head(layer, resolved_weight, gather_output)
    if layer.sequence_parallel:
        raise ValueError("SFT stable-gradient head requires the caller's existing full-sequence SP gather")
    if not input_.is_cuda or input_.device != resolved_weight.device or input_.dtype != resolved_weight.dtype:
        raise ValueError("SFT stable-gradient head requires matching CUDA BF16 input and weight")
    if input_.shape[-1] != resolved_weight.shape[-1] or torch.cuda.is_current_stream_capturing():
        raise ValueError("SFT stable-gradient head requires matching hidden size and CUDA graphs disabled")
    ensure_params_ready([resolved_weight])
    mapped = copy_to_tensor_model_parallel_region(input_.float(), group=layer.tp_group).to(input_.dtype)
    old_disable, old_allreduce = layer.disable_grad_reduce, layer.allreduce_dgrad
    layer.disable_grad_reduce, layer.allreduce_dgrad = True, False
    try:
        yield mapped
    finally:
        layer.disable_grad_reduce, layer.allreduce_dgrad = old_disable, old_allreduce
