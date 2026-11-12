# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Delegate PLE n-grams, gating and causal convolution to the native layer."""

from __future__ import annotations

import inspect
import os
from collections.abc import Callable
from contextvars import ContextVar
from functools import wraps
from typing import Any

from examples.chunkflow.adapters import prefix
from examples.chunkflow.runtime import UNIT_ATTR, ChunkflowRuntime


_ACTIVE: ContextVar[Any] = ContextVar("chunkflow_ple_batch", default=None)
CONTEXT_ATTR = "_chunkflow_ple_context"


def validate(module: Any) -> None:
    prefix.validate(module)
    if f"{type(module).__module__}.{type(module).__qualname__}" != (
        "megatron.core.transformer.per_layer_embedding.PerLayerEmbedding"
    ):
        raise NotImplementedError("ChunkFlow PLE requires the exact native PerLayerEmbedding class")
    if os.environ.get("RELAX_QWEN38_PLE_STREAMED", "0") not in ("", "0"):
        raise NotImplementedError("ChunkFlow PLE requires RELAX_QWEN38_PLE_STREAMED=0 with CP=1")
    if "forward" in module.__dict__ or "prepare" in module.__dict__:
        raise NotImplementedError("Patched PLE methods require a separate adapter")


def install_context(model: Any) -> Callable[[], None]:
    original = model.forward
    signature = inspect.signature(original)
    had_forward = "forward" in model.__dict__

    @wraps(original)
    def forward(*args: Any, **kwargs: Any) -> Any:
        packed = signature.bind(*args, **kwargs).arguments.get("packed_seq_params")
        token = _ACTIVE.set(packed)
        try:
            return original(*args, **kwargs)
        finally:
            _ACTIVE.reset(token)

    model.forward = forward

    def undo() -> None:
        if had_forward:
            model.forward = original
        else:
            del model.forward

    return undo


def install(module: Any, chunk_size: int, runtime: ChunkflowRuntime) -> Callable[[], None]:
    del chunk_size
    original_forward, original_prepare = module.forward, module.prepare

    @wraps(original_prepare)
    def prepare(*args: Any, **kwargs: Any) -> Any:
        packed = _ACTIVE.get()
        if getattr(packed, UNIT_ATTR, None) is None:
            return original_prepare(*args, **kwargs)
        tokens = getattr(packed, prefix.TOKENS_ATTR, None)
        if tokens is None:
            raise ValueError("PLE prefix replay requires the producer's original prefix token IDs")
        size, _, _ = prefix.parallel(module)
        native_packed, logical, total = prefix.layout(packed, size, minimum=size * module.halo)
        if tokens.shape != (1, logical):
            raise ValueError("PLE prefix token IDs do not match the unit's sequence position")
        tokens = prefix.pad(tokens.T, total).T.contiguous()
        return original_prepare(tokens, cu_seqlens=native_packed.cu_seqlens_q)

    @wraps(original_forward)
    def forward(hidden_states: Any) -> Any:
        packed = _ACTIVE.get()
        if getattr(packed, UNIT_ATTR, None) is None:
            return original_forward(hidden_states)
        size, _, _ = prefix.parallel(module)
        call = prefix.build(module, hidden_states, packed, runtime, "ple", minimum=size * module.halo)
        # prepare is called for every model forward, including ChunkFlow Fr.
        return call.restore(original_forward(call.hidden))

    module.prepare, module.forward = prepare, forward

    def undo() -> None:
        del module.prepare
        del module.forward

    return undo
