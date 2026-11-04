# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from __future__ import annotations

import copy
import inspect
from contextvars import ContextVar
from dataclasses import dataclass, field
from functools import wraps
from typing import Any, Callable

import torch
from torch.nn import functional as F

from relax.backends.megatron.chunkflow.metadata import execution_optimizations_enabled, int32_metadata
from relax.backends.megatron.chunkflow.runtime import RECOMPUTE_ATTR, UNIT_ATTR, ChunkflowRuntime
from relax.backends.megatron.chunkflow.state_store import ChunkState


_ORIGINAL = "_chunkflow_original_gdn"


@dataclass
class _Context:
    module: Any
    packed: Any
    previous: dict[str, torch.Tensor] | None = None
    current: dict[str, torch.Tensor] | None = None
    seen: set[str] = field(default_factory=set)


_ACTIVE: ContextVar[_Context | None] = ContextVar("chunkflow_gdn", default=None)


class _StateGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, output: torch.Tensor, final_state: torch.Tensor, leaf: torch.Tensor) -> torch.Tensor:
        ctx.leaf = leaf
        return output

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None, None]:
        # The final state is a separate kernel output, so inject its gradient explicitly.
        return grad_output, ctx.leaf.grad, None


def _wrap_kernel(kernel: Callable, name: str) -> Callable:
    signature = inspect.signature(kernel)
    fields = ("x",) if name == "conv" else ("q", "k", "v", "g", "beta")
    optimize_execution = execution_optimizations_enabled()

    @wraps(kernel)
    def call(*args, **kwargs):
        context = _ACTIVE.get()
        if context is None:
            return kernel(*args, **kwargs)
        if context.current is None:
            if "cu_seqlens_cpu" in signature.parameters:
                bound = signature.bind(*args, **kwargs)
                lengths = context.packed._chunkflow_gdn_lengths
                if bound.arguments.get("cu_seqlens") is not None and (
                    name != "conv" or bound.arguments["x"].shape[1] == lengths[0]
                ):
                    bound.arguments["cu_seqlens_cpu"] = lengths[2]
                    return kernel(*bound.args, **bound.kwargs)
            return kernel(*args, **kwargs)
        bound = signature.bind(*args, **kwargs)
        if bound.arguments.get("cp_context") is not None:
            raise NotImplementedError("ChunkFlow GDN state transfer requires CP=1")
        length = getattr(context.packed, UNIT_ATTR).num_tokens
        padded_length = bound.arguments[fields[0]].shape[1]
        if padded_length < length:
            raise ValueError("GDN kernel input is shorter than its ChunkFlow unit")
        if padded_length > length or not optimize_execution:
            for key in fields:
                bound.arguments[key] = bound.arguments[key][:, :length]
        bound.arguments.update(initial_state=context.previous.get(name), output_final_state=True, cu_seqlens=None)
        for key in ("cu_seqlens_cpu", "chunk_indices"):
            if key in signature.parameters:
                bound.arguments[key] = None
        output, final_state = kernel(*bound.args, **bound.kwargs)
        if not isinstance(final_state, torch.Tensor):
            raise RuntimeError(f"GDN {name} kernel did not return its final state")
        if torch.is_grad_enabled() and output.requires_grad and not final_state.requires_grad:
            raise RuntimeError(f"GDN {name} kernel must support final-state gradients")
        context.seen.add(name)
        if name not in context.current:
            context.current[name] = final_state.detach().requires_grad_(True)
        output = _StateGradient.apply(output, final_state, context.current[name])
        if padded_length > length:
            output = F.pad(output, (0, 0) * (output.ndim - 2) + (0, padded_length - length))
        return output, final_state

    return call


def _wrap_forward(module: Any, original: Callable, runtime: ChunkflowRuntime, *, prepare: bool) -> Callable:
    signature = inspect.signature(original)

    @wraps(original)
    def forward(*args, **kwargs):
        bound = signature.bind(*args, **kwargs)
        packed = bound.arguments.get("packed_seq_params")
        unit = getattr(packed, UNIT_ATTR, None)
        if unit is None:
            return original(*args, **kwargs)
        if prepare:
            if getattr(packed, "qkv_format", None) != "thd":
                raise NotImplementedError("ChunkFlow GDN requires packed THD inputs")
            total = bound.arguments["hidden_states"].shape[0] * module.sp_size
            lengths = getattr(packed, "_chunkflow_gdn_lengths", None)
            if lengths is None:
                cu = list(packed.cu_seqlens_q_cpu)
                if cu[0] != 0 or cu[-1] != unit.num_tokens or cu[-1] > total:
                    raise ValueError("GDN unit lengths do not match its packed input")
                if cu[-1] < total:
                    cu.append(total)
                lengths = (
                    total,
                    packed.cu_seqlens_q
                    if execution_optimizations_enabled() and unit.num_tokens == total
                    else int32_metadata(cu, packed.cu_seqlens_q.device),
                    torch.tensor(cu, dtype=torch.int64),
                )
                packed._chunkflow_gdn_lengths = lengths
            if lengths[0] != total:
                raise ValueError("GDN physical unit length changed between layers")
            packed = copy.copy(packed)
            packed.cu_seqlens_q = lengths[1]
            packed.cu_seqlens_kv = packed.cu_seqlens_q
            packed.cu_seqlens_q_padded = packed.cu_seqlens_kv_padded = None
            packed._chunkflow_gdn_total = total
            bound.arguments["packed_seq_params"] = packed
        context = _Context(module, packed)
        if not prepare and unit.is_dependent:
            if len(unit.chunks) != 1:
                raise NotImplementedError("ChunkFlow GDN requires one dependent chunk per unit")
            chunk = unit.chunks[0]
            if len(chunk.segments) != 1:
                raise NotImplementedError("ChunkFlow GDN requires one dependent segment per chunk")
            store = runtime.store(module.layer_number, chunk.group)
            if chunk.group_idx == len(store) and not getattr(packed, RECOMPUTE_ATTR, False):
                store.put(chunk.group_idx, ChunkState({}))
            states = store.lookup(chunk.group_idx)
            context.previous = states[-2].tensors if chunk.group_idx else {}
            context.current = states[-1].tensors
        token = _ACTIVE.set(context)
        try:
            result = original(*bound.args, **bound.kwargs)
            if context.current is not None and context.seen != {"conv", "recurrent"}:
                raise RuntimeError("GDN execution bypassed the state-aware convolution or recurrence")
            return result
        finally:
            _ACTIVE.reset(token)

    return forward


def _wrap_resolve(module: Any, original: Callable) -> Callable:
    @wraps(original)
    def resolve(padded, actual, total, name, cp_size=1):
        context = _ACTIVE.get()
        if context is not None and context.module is module:
            if cp_size != 1 or total != context.packed._chunkflow_gdn_total:
                raise ValueError("GDN packed lengths or CP layout changed inside a ChunkFlow unit")
            return actual
        return original(padded, actual, total, name, cp_size=cp_size)

    return resolve


def install(model: Any, runtime: ChunkflowRuntime) -> int:
    modules = [m for m in model.modules() if any(c.__name__ == "GatedDeltaNet" for c in type(m).__mro__)]
    if not modules:
        return 0
    from megatron.core.ssm import gated_delta_net

    for module in modules:
        if (
            getattr(module.config, "context_parallel_size", 1) != 1
            or getattr(module.config, "deterministic_mode", False)
            or getattr(module, "gdn_pre_gated_delta_rule_fusion", False)
        ):
            raise NotImplementedError("ChunkFlow GDN requires CP=1 and unfused, nondeterministic GDN preprocessing")
        if getattr(module.config, "recompute_granularity", None) == "full":
            raise NotImplementedError(
                "ChunkFlow hybrid GDN models require selective GDN, not full-layer recomputation"
            )
        if not all(hasattr(module, name) for name in ("_forward_compute", "_resolve_cu_seqlens", "gated_delta_rule")):
            raise NotImplementedError("ChunkFlow GDN requires Megatron's _forward_compute interface")
    count = 0
    for module in modules:
        if hasattr(module, _ORIGINAL):
            continue
        preprocess = getattr(module, "pre_gated_delta_rule", None)
        owner = inspect.getmodule(inspect.unwrap(preprocess)) if preprocess is not None else gated_delta_net
        if owner is None or not hasattr(owner, "causal_conv1d"):
            raise NotImplementedError("ChunkFlow GDN requires an unfused causal_conv1d preprocessing path")
        if not hasattr(owner.causal_conv1d, "_chunkflow_original"):
            original_conv = owner.causal_conv1d
            owner.causal_conv1d = _wrap_kernel(original_conv, "conv")
            owner.causal_conv1d._chunkflow_original = original_conv
        original = (module.forward, module._forward_compute, module._resolve_cu_seqlens, module.gated_delta_rule)
        setattr(module, _ORIGINAL, original)
        module.forward = _wrap_forward(module, original[0], runtime, prepare=True)
        module._forward_compute = _wrap_forward(module, original[1], runtime, prepare=False)
        module._resolve_cu_seqlens = _wrap_resolve(module, original[2])
        module.gated_delta_rule = _wrap_kernel(original[3], "recurrent")
        count += 1
    return count


def uninstall(model: Any) -> int:
    count = 0
    for module in model.modules():
        original = getattr(module, _ORIGINAL, None)
        if original is not None:
            module.forward, module._forward_compute, module._resolve_cu_seqlens, module.gated_delta_rule = original
            delattr(module, _ORIGINAL)
            count += 1
    return count
