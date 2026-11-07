# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Keep native QSA selection and attention by replaying complete prefixes."""

from __future__ import annotations

import inspect
import os
from collections.abc import Callable
from contextvars import ContextVar
from functools import wraps
from typing import Any

import torch

from examples.chunkflow.adapters import prefix
from examples.chunkflow.runtime import UNIT_ATTR, ChunkflowRuntime


def validate(module: Any) -> None:
    prefix.validate(module)
    core = module.core_attention
    if f"{type(core).__module__}.{type(core).__qualname__}" != (
        "megatron.core.transformer.experimental_attention_variant.qsa.QSACoreAttention"
    ):
        raise NotImplementedError("Native QSA prefix replay requires the exact Megatron QSACoreAttention")
    if getattr(module.config, "qsa_indexer_loss_coeff", 0):
        raise NotImplementedError("Native QSA prefix replay does not support Stage-2 indexer auxiliary loss")
    if core.sparse_backend not in ("flex", "dense_masked"):
        raise NotImplementedError("Native QSA prefix replay requires the flex or dense_masked backend")
    if f"{type(module.indexer).__module__}.{type(module.indexer).__qualname__}" != (
        "megatron.core.transformer.experimental_attention_variant.qsa.QSAIndexer"
    ):
        raise NotImplementedError("Native QSA prefix replay requires the exact Megatron QSAIndexer")
    for name in (
        "RELAX_QWEN38_QSA_FUSED_SELECTOR",
        "RELAX_QWEN38_QSA_FUSED_ATTENTION",
        "RELAX_QWEN38_QSA_FIXED_POOL",
        "RELAX_QWEN38_QSA_TP_SELECTOR",
        "RELAX_QWEN38_RETAIN_QSA_ATTENTION",
        "RELAX_QWEN38_QSA_CPU_LAYOUT",
        "RELAX_QWEN38_QSA_LOCAL_QUERY",
        "RELAX_QWEN38_QSA_RECOMPUTE_REPLAY",
        "RELAX_QWEN38_QSA_SPARSE_BLOCKMASK",
    ):
        if os.environ.get(name, "0") not in ("", "0"):
            raise NotImplementedError(f"Native QSA prefix replay requires {name}=0")
    if (
        "forward" in module.__dict__
        or "forward" in core.__dict__
        or any(name in module.indexer.__dict__ for name in ("forward", "_pool_keys", "_select_blocks"))
    ):
        raise NotImplementedError("Patched QSA forward requires a separate adapter")


def install(module: Any, chunk_size: int, runtime: ChunkflowRuntime) -> Callable[[], None]:
    del chunk_size
    original = module.forward
    original_pool = module.indexer._pool_keys
    original_select = module.indexer._select_blocks
    had_pool = "_pool_keys" in module.indexer.__dict__
    had_select = "_select_blocks" in module.indexer.__dict__
    signature = inspect.signature(original)
    parent_length: ContextVar[int | None] = ContextVar("chunkflow_qsa_parent_length", default=None)
    selection_layout: ContextVar[Any] = ContextVar("chunkflow_qsa_selection_layout", default=None)

    @wraps(original_pool)
    def pool_keys(raw_keys: Any, doc_ids: Any, positions: Any, max_doc_len: int) -> Any:
        # Native top-k tie ordering can change with the score width. Preserve
        # the parent pack's block axis, including its masked future blocks.
        return original_pool(raw_keys, doc_ids, positions, parent_length.get() or max_doc_len)

    @wraps(original_select)
    def select_blocks(
        q: torch.Tensor,
        pooled_keys: torch.Tensor,
        block_valid: torch.Tensor,
        doc_ids: torch.Tensor,
        positions: torch.Tensor,
        uniform_doc_len: int | None = None,
        output_format: str = "bits",
    ) -> tuple[torch.Tensor, bool]:
        layout = selection_layout.get()
        if layout is None:
            return original_select(q, pooled_keys, block_valid, doc_ids, positions, uniform_doc_len, output_format)
        if output_format != "bits":
            raise NotImplementedError("Native QSA prefix replay requires bitset selection")
        cu, segments = layout
        # Restore the selector's original rows AND columns. TopK ties and the
        # native GEMM/tile choice depend on both dimensions and document count.
        rows = torch.cat(
            [torch.arange(cu[s.seq_id], cu[s.seq_id] + s.prefix + s.length, device=q.device) for s in segments]
        )
        docs = torch.tensor([s.seq_id for s in segments], device=q.device)
        full_q = q.new_zeros((cu[-1], *q.shape[1:])).index_copy_(0, rows, q[: rows.numel()])
        full_keys = pooled_keys.new_zeros((len(cu) - 1, *pooled_keys.shape[1:]))
        full_keys.index_copy_(0, docs, pooled_keys[: len(segments)])
        lengths = torch.tensor([b - a for a, b in zip(cu[:-1], cu[1:])], device=q.device)
        full_docs = torch.repeat_interleave(torch.arange(len(cu) - 1, device=q.device), lengths, output_size=cu[-1])
        starts = torch.tensor(cu[:-1], device=q.device)
        full_positions = torch.arange(cu[-1], device=q.device) - starts[full_docs]
        full_valid = torch.arange(pooled_keys.shape[1], device=q.device)[None, :] < (
            lengths[:, None] // module.indexer.compress_ratio
        )
        selected, all_selected = original_select(
            full_q,
            full_keys,
            full_valid,
            full_docs.to(doc_ids.dtype),
            full_positions.to(positions.dtype),
            None,
            output_format="bits",
        )
        # Padding documents never contribute to loss or real-token attention.
        return prefix.pad(selected.index_select(0, rows), q.shape[0]), all_selected

    @wraps(original)
    def forward(*args: Any, **kwargs: Any) -> Any:
        bound = signature.bind(*args, **kwargs)
        packed = bound.arguments.get("packed_seq_params")
        if getattr(packed, UNIT_ATTR, None) is None:
            return original(*args, **kwargs)
        maximum = getattr(packed, prefix.MAX_LENGTH_ATTR, None)
        if not isinstance(maximum, int) or maximum <= 0:
            raise ValueError("QSA prefix replay requires the parent pack's maximum physical document length")
        cu = getattr(packed, prefix.PARENT_CU_ATTR, None)
        segments = [s for c in getattr(packed, UNIT_ATTR).chunks for s in c.segments]
        if (
            not isinstance(cu, tuple)
            or len(cu) < 2
            or cu[0] != 0
            or any(not isinstance(value, int) for value in cu)
            or any(a >= b for a, b in zip(cu[:-1], cu[1:]))
            or max(b - a for a, b in zip(cu[:-1], cu[1:])) != maximum
            or any(s.seq_id >= len(cu) - 1 or s.prefix + s.length > cu[s.seq_id + 1] - cu[s.seq_id] for s in segments)
        ):
            raise ValueError("QSA prefix replay requires producer-owned parent physical boundaries")
        for name in (
            "attention_mask",
            "attention_bias",
            "key_value_states",
            "inference_context",
            "inference_params",
            "rotary_pos_cos",
            "rotary_pos_sin",
            "rotary_pos_cos_sin",
            "sequence_len_offset",
        ):
            if bound.arguments.get(name) is not None:
                raise NotImplementedError(f"Native QSA prefix replay does not accept {name}")
        call = prefix.build(module, bound.arguments["hidden_states"], packed, runtime, "qsa")
        bound.arguments.update(
            hidden_states=call.hidden,
            packed_seq_params=call.packed,
            rotary_pos_emb=call.rotary(
                bound.arguments.get("rotary_pos_emb"),
                absolute=getattr(module.config, "mrope_section", None) is not None,
            ),
        )
        token = parent_length.set(maximum)
        layout_token = selection_layout.set((cu, segments))
        try:
            output, bias = original(*bound.args, **bound.kwargs)
        finally:
            parent_length.reset(token)
            selection_layout.reset(layout_token)
        return call.restore(output), bias

    module.indexer._pool_keys = pool_keys
    module.indexer._select_blocks = select_blocks
    module.forward = forward

    def undo() -> None:
        del module.forward
        if had_pool:
            module.indexer._pool_keys = original_pool
        else:
            del module.indexer._pool_keys
        if had_select:
            module.indexer._select_blocks = original_select
        else:
            del module.indexer._select_blocks

    return undo
