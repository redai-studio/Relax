# Copyright (c) 2026 Relax Authors. All Rights Reserved.
# Copyright 2025-2026 RadixArk, Inc. (miles).
# SPDX-License-Identifier: Apache-2.0
# Adapted for Relax chunking, TP gradients and MTP metrics.
"""Fused linear CE and log-prob/entropy with native-head fallback.

SFT/MTP retain their existing CE kernels and precision semantics. Policy
consumers use FP32-accumulator log-prob/entropy tiles with recomputed backward;
response slicing, objectives and reducers remain in loss.py.

The tiled statistics/recomputation design follows verl at
0938fcb3190bc8eca3091f77aead3cd44c4d8869; see kernels/rl_logprobs.py.
"""

import inspect
import math
from dataclasses import dataclass
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
    layer: Any,
    hidden: torch.Tensor,
    weight: torch.Tensor | None,
    gather_output: bool | None = None,
    *,
    enabled: bool = False,
) -> bool:
    if not (enabled or Envs.RELAX_SFT_FUSED_LINEAR_CE) or not hidden.is_cuda or weight is None:
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
    enabled: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Return unreduced losses in hidden row order; SP gathering is the
    caller's job."""
    if (enabled or Envs.RELAX_SFT_FUSED_LINEAR_CE) and temperature == 1.0:
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


# Bounded forward statistics and backward BF16 dlogits workspace, independent
# of the microbatch's token count. These are implementation tiles, not knobs.
# TODO(agent): Resolve the strict pretrained multi-update BF16 parameter parity gate.
_ROWS = 128


class _LinearLogProbs(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hidden, weight, labels, group, temperature, entropy_grad, accumulate):
        from .kernels.rl_logprobs import forward_stats

        ctx.set_materialize_grads(False)
        x = hidden.reshape(-1, hidden.shape[-1]).contiguous()
        labels = labels.reshape(-1).contiguous()
        size = dist.get_world_size(group) if group is not None else 1
        offset = (dist.get_rank(group) if group is not None else 0) * weight.shape[0]
        maximum = torch.empty(x.shape[0], dtype=torch.float32, device=x.device)
        stats = torch.empty((3, x.shape[0]), dtype=torch.float32, device=x.device)
        for start in range(0, len(x), _ROWS):
            end = min(start + _ROWS, len(x))
            tiles = forward_stats(x[start:end], weight, labels[start:end], offset, temperature)
            m = tiles[:, :, 0].amax(0)
            factor = (tiles[:, :, 0] - m).exp()
            maximum[start:end] = m
            stats[0, start:end] = (tiles[:, :, 1] * factor).sum(0)
            stats[1, start:end] = ((tiles[:, :, 2] + (tiles[:, :, 0] - m) * tiles[:, :, 1]) * factor).sum(0)
            stats[2, start:end] = tiles[:, :, 3].sum(0)
        if size > 1 and len(x):
            global_max = maximum.clone()
            dist.all_reduce(global_max, op=dist.ReduceOp.MAX, group=group)
            factor = (maximum - global_max).exp()
            stats[1] = (stats[1] + (maximum - global_max) * stats[0]) * factor
            stats[0] *= factor
            dist.all_reduce(stats, op=dist.ReduceOp.SUM, group=group)
            maximum = global_max
        denom, weighted, target = stats.unbind(0)
        mean = weighted / denom
        log_probs = target - maximum - denom.log()
        entropy = (denom.log() - mean).reshape(hidden.shape[:-1])
        if not entropy_grad:
            ctx.mark_non_differentiable(entropy)
        ctx.save_for_backward(x, weight, labels, maximum, denom, mean if entropy_grad else mean.new_empty(0))
        # Saved-tensor hooks may return a Tensor without Parameter attributes.
        ctx.weight_ref = weight
        ctx.group, ctx.size, ctx.offset = group, size, offset
        ctx.temperature, ctx.accumulate = temperature, accumulate
        ctx.hidden_shape = hidden.shape
        return log_probs.reshape(hidden.shape[:-1]), entropy

    @staticmethod
    @once_differentiable
    def backward(ctx, dlog_probs, dentropy):
        if dlog_probs is None and dentropy is None:
            return (None,) * 7
        from .kernels.rl_logprobs import backward_dlogits

        x, weight, labels, maximum, denom, mean = ctx.saved_tensors
        a = None if dlog_probs is None else dlog_probs.reshape(-1).float().contiguous()
        b = None if dentropy is None else dentropy.reshape(-1).float().contiguous()
        dx = torch.empty_like(x) if ctx.needs_input_grad[0] else None
        parameter = ctx.weight_ref
        dw = None
        if ctx.needs_input_grad[1]:
            dw = parameter.main_grad if ctx.accumulate else torch.zeros_like(weight, dtype=torch.float32)
        for start in range(0, len(x), _ROWS):
            end = min(start + _ROWS, len(x))
            dlogits = backward_dlogits(
                x[start:end],
                weight,
                labels[start:end],
                maximum[start:end],
                denom[start:end],
                mean[start:end],
                None if a is None else a[start:end],
                None if b is None else b[start:end],
                ctx.offset,
                ctx.temperature,
            )
            if dx is not None:
                torch.mm(dlogits, weight, out=dx[start:end])
            if dw is not None:
                # BF16 operands, FP32 accumulation; sum token chunks in FP32.
                dw.add_(torch.mm(dlogits.T, x[start:end], out_dtype=torch.float32))
        if dx is not None and ctx.size > 1 and len(x):
            dist.all_reduce(dx, op=dist.ReduceOp.SUM, group=ctx.group)
        if dw is not None:
            if ctx.accumulate:
                dw = None
                if hasattr(parameter, "grad_added_to_main_grad"):
                    dw = (
                        torch.zeros_like(weight)
                        if getattr(parameter, "zero_out_wgrad", False)
                        else torch.empty_like(weight)
                    )
                    parameter.grad_added_to_main_grad = True
            else:
                dw = dw.to(weight.dtype)
        return None if dx is None else dx.reshape(ctx.hidden_shape), dw, None, None, None, None, None


def linear_log_probs(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    labels: torch.Tensor,
    group: Any,
    *,
    temperature: float = 1.0,
    entropy_grad: bool = True,
    accumulate: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Unreduced FP32 statistics in hidden row order; caller owns SP gather."""
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("RL fused log-probs require a finite positive temperature")
    if not hidden.is_cuda or hidden.dtype != torch.bfloat16 or weight.dtype != hidden.dtype:
        raise ValueError("RL fused log-probs require CUDA BF16 operands")
    if labels.shape != hidden.shape[:-1] or weight.ndim != 2 or hidden.shape[-1] != weight.shape[-1]:
        raise ValueError("RL fused log-probs hidden/weight/label shapes do not match")
    if accumulate and weight.requires_grad and getattr(weight, "main_grad", None) is None:
        raise ValueError("RL fused gradient accumulation requires Parameter.main_grad")
    if accumulate and weight.requires_grad and weight.main_grad.dtype != torch.float32:
        raise ValueError("RL fused gradient accumulation requires FP32 main_grad")
    return _LinearLogProbs.apply(hidden, weight, labels, group, temperature, entropy_grad, accumulate)


@dataclass
class OutputContext:
    temperature: float
    entropy_grad: bool
    fused: bool = False
    labels: torch.Tensor | None = None


@lru_cache(maxsize=64)
def _diagnostic(reason: str) -> None:
    get_logger(__name__).info("RL fused log-probs: %s", reason)


@lru_cache(maxsize=None)
def _tp_control_group(group: Any) -> Any:
    # Only output PP stages participate; other TP groups need not enter creation.
    return dist.new_group(ranks=dist.get_process_group_ranks(group), backend="gloo", use_local_synchronization=True)


def _tp_fallback_reason(reason: str | None, group: Any) -> str | None:
    if dist.get_world_size(group) == 1:
        return reason
    # Eligibility is CPU metadata. Keep its reduction on CPU to avoid object
    # serialization and CUDA-to-host synchronization on every microbatch.
    unsupported = torch.tensor(reason is not None, dtype=torch.int32, device="cpu")
    dist.all_reduce(unsupported, op=dist.ReduceOp.MAX, group=_tp_control_group(group))
    if unsupported.item():
        return reason or "unsupported peer TP shard"
    return None


def _head_reason(layer: Any) -> str | None:
    from megatron.core.tensor_parallel.layers import ColumnParallelLinear
    from torch.nn.modules import module as hooks

    if type(layer) is not ColumnParallelLinear:
        try:
            from megatron.core.transformer.linear_cross_entropy import LinearCrossEntropyModule
        except ImportError:
            return "non-native head"
        # Its plain-logits forward delegates to ColumnParallelLinear. Native
        # fused CE remains excluded below; arbitrary subclasses stay excluded.
        if type(layer) is not LinearCrossEntropyModule:
            return "non-native head"
    if "forward" in layer.__dict__:
        return "custom head forward"
    if getattr(layer._forward_impl, "__func__", None) is not ColumnParallelLinear._forward_impl:
        return "custom head implementation"
    if layer.bias is not None or layer.gather_output:
        return "bias or gathered vocabulary"
    if layer._forward_hooks or layer._backward_hooks or layer._backward_pre_hooks:
        return "head hooks"
    if any(
        getattr(hooks, name)
        for name in (
            "_global_forward_pre_hooks",
            "_global_forward_hooks",
            "_global_backward_pre_hooks",
            "_global_backward_hooks",
        )
    ):
        return "global module hooks"
    for hook in layer._forward_pre_hooks.values():
        if (
            getattr(hook, "__module__", "") != "megatron.core.distributed.distributed_data_parallel"
            or getattr(hook, "__qualname__", "") != "DistributedDataParallel._make_forward_pre_hook.<locals>.hook"
        ):
            return "custom head pre-hook"
    for name in ("disable_grad_reduce", "explicit_expert_comm", "is_expert"):
        if getattr(layer, name, False):
            return name
    if getattr(layer, "gtp_remat_size", 1) != 1:
        return "GTP rematerialization"
    if getattr(layer.config, "defer_embedding_wgrad_compute", False):
        return "deferred embedding wgrad"
    if getattr(layer.config, "use_mup", False):
        return "custom logits scaling"
    if getattr(layer.config, "cross_entropy_loss_fusion", False):
        return "native fused CE"
    return None


def _cp_labels(args: Any, batch: dict) -> torch.Tensor:
    """Shift full samples before reproducing data.get_batch's CP partition.

    Input/labels here are CP-local but not sequence-parallel scattered.
    End/padding labels are unused by the RL response slicer.
    """
    from .cp_utils import slice_with_cp

    shifted = [torch.cat((tokens[1:], tokens.new_zeros(1))) for tokens in batch["unconcat_tokens"]]
    if args.qkv_format == "bshd":
        return torch.stack([slice_with_cp(labels, 0, "bshd", batch["max_seq_lens"][0]) for labels in shifted])
    labels = torch.cat([slice_with_cp(labels, 0, "thd") for labels in shifted])
    return torch.nn.functional.pad(labels, (0, batch["tokens"].numel() - labels.numel())).unsqueeze(0)


def prepare_output_processor(model: torch.nn.Module, args: Any, batch: dict, *, consumer: Any) -> dict:
    """Decide before model forward; return no hook for unsupported consumers.

    Real head/operand validation also happens inside the hook BEFORE bypassing
    the projection. The context records whether this microbatch returned stats.
    """
    if not getattr(args, "fused_linear_ce", False):
        return {}
    from megatron.core import parallel_state as mpu
    from megatron.core.models.gpt.gpt_model import GPTModel

    from relax.algorithms import get_algorithm

    role = getattr(model, "role", None)
    inner = model
    while hasattr(inner, "module"):
        inner = inner.module
    # Only the output stage owns a head. Earlier PP stages keep hidden P2P.
    if type(inner) is GPTModel and not inner.post_process:
        return {}
    reason = None
    if role != "actor":
        reason = "model role is not actor"
    elif consumer not in ("policy_loss", "log_probs") or getattr(args, "loss_type", None) != "policy_loss":
        reason = "consumer is not RL policy log-probs"
    elif type(inner) is not GPTModel or "output_processor" not in inspect.signature(inner.forward).parameters:
        reason = "model has no validated native GPT output hook"
    elif getattr(inner._scale_logits, "__func__", None) is not GPTModel._scale_logits:
        reason = "custom logits scaling"
    elif mpu.get_virtual_pipeline_model_parallel_world_size() is not None:
        reason = "VPP"
    elif mpu.get_context_parallel_world_size() > 1 and args.allgather_cp:
        reason = "contiguous allgather CP layout"
    elif batch.get("dynamic_cp_size") is not None or getattr(args, "dynamic_context_parallel", False):
        reason = "dynamic CP"
    elif any(
        getattr(args, name, False)
        for name in (
            "enable_mtp_training",
            "mtp_only_training",
            "is_vl_model",
            "uses_unsplit_forward",
            "use_opd",
            "use_lora",
            "cross_entropy_loss_fusion",
            "use_combined_1f1b",
        )
    ) or getattr(inner.config, "mtp_num_layers", None):
        reason = "MTP/VL/LoRA/OPD/native CE/combined 1F1B"
    elif getattr(args, "lora_rank", 0) > 0:
        reason = "LoRA"
    elif any(batch.get(name) is not None for name in ("log_prob_keep_mask", "sampling_support_mask")):
        reason = "sampling support mask"
    elif batch.get("multimodal_train_inputs") is not None or batch.get("rollout_topk_token_ids") is not None:
        reason = "multimodal or top-k consumer"
    elif get_algorithm(args.advantage_estimator).policy_loss_fn == "score_centering":
        reason = "score centering"
    elif any(r >= t for t, r in zip(batch["total_lengths"], batch["response_lengths"], strict=True)):
        reason = "response without prompt"
    else:
        reason = _head_reason(inner.output_layer)
    group = mpu.get_tensor_model_parallel_group()
    reason = _tp_fallback_reason(reason, group)
    if reason is not None:
        _diagnostic("fallback to native head precision: " + reason)
        return {}
    labels = _cp_labels(args, batch) if mpu.get_context_parallel_world_size() > 1 else None
    context = OutputContext(args.rollout_temperature, getattr(args, "entropy_coef", 0.0) != 0, labels=labels)
    return {"output_processor": output_processor, "output_processor_context": context}


def output_processor(*, hidden_states, output_layer, output_weight, input_ids, context, **kwargs):
    from megatron.core.tensor_parallel.mappings import gather_from_sequence_parallel_region

    weight = output_weight if output_weight is not None else output_layer.weight
    reason = _head_reason(output_layer)
    labels = context.labels
    expected_rows = hidden_states.shape[0] * (output_layer.tp_group.size() if output_layer.sequence_parallel else 1)
    if labels is not None and labels.shape != (hidden_states.shape[1], expected_rows):
        reason = "CP label/hidden layout mismatch"
    if reason is None and (
        weight is None
        or not hidden_states.is_cuda
        or hidden_states.dtype != torch.bfloat16
        or weight.dtype != torch.bfloat16
        or not weight.is_contiguous()
        or weight.device != hidden_states.device
        or hidden_states.shape[-1] != weight.shape[-1]
        or weight.shape[0] == 0
        or getattr(weight, "__fsdp_param__", False)
    ):
        reason = "unsupported operands"
    if reason is None and output_layer.gradient_accumulation_fusion and weight.requires_grad:
        main_grad = getattr(weight, "main_grad", None)
        if main_grad is None or main_grad.dtype != torch.float32:
            reason = "head requires an FP32 main_grad buffer"
    # All TP ranks must choose the same branch, including frozen/empty shards.
    group = output_layer.tp_group
    reason = _tp_fallback_reason(reason, group)
    if reason is not None:
        _diagnostic("fallback to native head precision: " + reason)
        logits, _ = output_layer(
            hidden_states, weight=output_weight, runtime_gather_output=kwargs["runtime_gather_output"]
        )
        return kwargs["scale_logits"](logits).transpose(0, 1).contiguous()
    # Reuse MCore DDP's actual parameter-gather hook before consuming weight.
    for hook in output_layer._forward_pre_hooks.values():
        hook(output_layer, (hidden_states,))
    if output_layer.sequence_parallel:
        hidden_states = gather_from_sequence_parallel_region(
            hidden_states,
            tensor_parallel_output_grad=False,
            group=group,
        )
    # Same next-token coordinates as get_responses. End/padding rows are never
    # selected by the response consumer. Include padded vocabulary in softmax.
    labels = input_ids.roll(-1, dims=-1) if labels is None else labels.to(hidden_states.device)
    labels = labels.transpose(0, 1).contiguous()
    log_probs, entropy = linear_log_probs(
        hidden_states,
        weight,
        labels,
        group,
        temperature=context.temperature,
        entropy_grad=context.entropy_grad,
        accumulate=output_layer.gradient_accumulation_fusion,
    )
    context.fused = True
    _diagnostic("active: BF16 operands, FP32 accumulator, tiled forward/recomputed backward")
    return torch.stack((log_probs, entropy), dim=-1).transpose(0, 1).contiguous()
