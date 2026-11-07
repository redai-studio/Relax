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

from examples.chunkflow.metadata import execution_optimizations_enabled, int32_metadata
from examples.chunkflow.runtime import RECOMPUTE_ATTR, UNIT_ATTR, ChunkflowRuntime
from examples.chunkflow.state_store import ChunkState


_CONVOLUTIONS: dict[Any, tuple[Callable, int]] = {}


@dataclass
class _Context:
    module: Any
    packed: Any
    previous: dict[str, torch.Tensor] | None = None
    current: ChunkState | None = None
    seen: set[str] = field(default_factory=set)


_ACTIVE: ContextVar[_Context | None] = ContextVar("chunkflow_gdn", default=None)


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
        output = context.current.link_output(name, final_state, output)
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
            states = runtime.history(
                module.layer_number,
                chunk.group,
                chunk.group_idx,
                recompute=getattr(packed, RECOMPUTE_ATTR, False),
                allow_replay=True,  # Native selective checkpoint re-enters _forward_compute.
            )
            context.previous = states[-2].tensors if chunk.group_idx else {}
            context.current = states[-1]
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


def _conv_owner(module: Any) -> Any:
    preprocess = getattr(module, "pre_gated_delta_rule", None)
    if preprocess is None:
        from megatron.core.ssm import gated_delta_net

        owner = gated_delta_net
    else:
        owner = inspect.getmodule(inspect.unwrap(preprocess))
    if owner is None or not callable(getattr(owner, "causal_conv1d", None)):
        raise NotImplementedError("ChunkFlow GDN requires an unfused causal_conv1d preprocessing path")
    return owner


def validate(module: Any) -> None:
    if (
        getattr(module.config, "context_parallel_size", 1) != 1
        or getattr(module.config, "deterministic_mode", False)
        or getattr(module, "gdn_pre_gated_delta_rule_fusion", False)
    ):
        raise NotImplementedError("ChunkFlow GDN requires CP=1 and unfused, nondeterministic GDN preprocessing")
    if getattr(module.config, "recompute_granularity", None) == "full":
        raise NotImplementedError("ChunkFlow hybrid GDN models require selective GDN, not full-layer recomputation")
    if not all(
        callable(getattr(module, name, None))
        for name in ("_forward_compute", "_resolve_cu_seqlens", "gated_delta_rule")
    ):
        raise NotImplementedError("ChunkFlow GDN requires Megatron's _forward_compute interface")
    for kernel in (_conv_owner(module).causal_conv1d, module.gated_delta_rule):
        if not {"initial_state", "output_final_state"} <= inspect.signature(kernel).parameters.keys():
            raise NotImplementedError("ChunkFlow GDN requires native initial/final-state kernel interfaces")


def install(module: Any, chunk_size: int, runtime: ChunkflowRuntime) -> Callable[[], None]:
    """Install one layer through the common adapter transaction; preserve
    native math."""
    owner = _conv_owner(module)
    methods = {
        "forward": _wrap_forward(module, module.forward, runtime, prepare=True),
        "_forward_compute": _wrap_forward(module, module._forward_compute, runtime, prepare=False),
        "_resolve_cu_seqlens": _wrap_resolve(module, module._resolve_cu_seqlens),
        "gated_delta_rule": _wrap_kernel(module.gated_delta_rule, "recurrent"),
    }
    previous = {name: module.__dict__[name] for name in methods if name in module.__dict__}
    original_conv, users = _CONVOLUTIONS.get(owner, (owner.causal_conv1d, 0))
    wrapped_conv = owner.causal_conv1d if users else _wrap_kernel(original_conv, "conv")

    def undo() -> None:
        for name in methods:
            if name in previous:
                setattr(module, name, previous[name])
            else:
                module.__dict__.pop(name, None)
        original, users = _CONVOLUTIONS[owner]
        if users == 1:
            owner.causal_conv1d = original
            del _CONVOLUTIONS[owner]
        else:
            _CONVOLUTIONS[owner] = (original, users - 1)

    owner.causal_conv1d = wrapped_conv
    _CONVOLUTIONS[owner] = (original_conv, users + 1)
    try:
        for name, method in methods.items():
            setattr(module, name, method)
    except Exception:
        undo()
        raise
    return undo
