# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Opt-in community Sparse-GQA backends for Megatron's selected-ID QSA path.

The native QSA indexer, Stage-2 loss, checkpointing, and CP reconstruction stay
in Megatron. Only the core attention forward/backward is replaced. Community
backends receive the same document-local selected blocks as ``id_sparse``.
"""

from __future__ import annotations

import os

import torch
from megatron.core.transformer.experimental_attention_variant import qsa
from torch import Tensor

from relax.utils.env import Envs
from relax.utils.logging_utils import get_logger


logger = get_logger(__name__)

_BACKEND = Envs.RELAX_QWEN38_QSA_COMMUNITY_BACKEND.strip()
_SUPPORTED = ("pr7553", "tilelang7060", "mixed")


def validate_backend() -> None:
    if _BACKEND not in _SUPPORTED:
        raise ValueError(f"Unsupported QSA community backend {_BACKEND!r}; choose one of {_SUPPORTED}")
    if os.environ.get("MCORE_QSA_SPARSE_BACKEND", "id_sparse") != "id_sparse":
        raise ValueError("Community QSA attention requires MCORE_QSA_SPARSE_BACKEND=id_sparse")


def _packed_blocks(selection: qsa.QSASelection, sequence_length: int) -> tuple[Tensor, Tensor, Tensor]:
    """Map packed document-local IDs to compact global blocks, including ragged
    tails."""
    prefix = selection.compact_block_prefix
    starts = selection.compact_block_starts
    if prefix is None or starts is None:
        raise ValueError("Packed selected-ID QSA requires compact block metadata")
    docs = selection.doc_ids.reshape(-1).long()
    positions = selection.positions.reshape(-1)
    num_docs = prefix.numel() - 1
    upper_bound = starts.numel()
    num_blocks = upper_bound + num_docs
    block_bases = torch.zeros(num_blocks, dtype=torch.int32, device=starts.device)
    block_ends = torch.zeros_like(block_bases)

    pool_rows = torch.arange(upper_bound, device=starts.device, dtype=torch.int32)
    block_docs = torch.searchsorted(prefix[1:].contiguous(), pool_rows, right=True).clamp(max=num_docs - 1)
    valid = pool_rows < prefix[-1]
    complete_indices = (pool_rows + block_docs).long()
    block_bases.scatter_(0, complete_indices, torch.where(valid, starts.int(), 0))
    block_ends.scatter_(0, complete_indices, torch.where(valid, starts.int() + 4, 0))

    physical = torch.arange(sequence_length, device=starts.device, dtype=torch.int32)
    doc_starts = torch.zeros(num_docs, device=starts.device, dtype=torch.int32)
    doc_starts.scatter_(0, docs, physical - positions)
    lengths = torch.zeros_like(doc_starts)
    lengths.scatter_add_(0, docs, torch.ones_like(positions))
    tail_indices = (prefix[1:] + torch.arange(num_docs, device=starts.device, dtype=torch.int32)).long()
    block_bases.scatter_(0, tail_indices, doc_starts + (lengths // 4) * 4)
    block_ends.scatter_(0, tail_indices, doc_starts + lengths)
    row_offsets = prefix[docs] + docs.int()
    return block_bases, block_ends, row_offsets.view(1, sequence_length)


def _pr_routes(
    selection: qsa.QSASelection, batch: int, sequence_length: int, is_thd: bool
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Convert native document-local complete block IDs to PR #7553's global
    layout."""
    ids = selection.selected_ids.view(batch, sequence_length, -1)
    positions = selection.positions
    if is_thd:
        if selection.compact_block_prefix is not None:
            block_bases, block_ends, offsets = _packed_blocks(selection, sequence_length)
        else:
            # Stage-1 may omit compact metadata. Assign each document-local
            # block exactly one dK/dV writer, including short document tails.
            # The static token-sized capacity avoids a device-to-host count;
            # unused slots stay empty and therefore perform no gradient stores.
            physical = torch.arange(sequence_length, device=ids.device, dtype=torch.int32)
            local_positions = positions.reshape(-1)
            starts_block = local_positions.remainder(selection.compress_ratio) == 0
            block_ids = starts_block.to(torch.int32).cumsum(0, dtype=torch.int32) - 1
            block_bases = torch.zeros_like(physical)
            block_bases.scatter_add_(0, block_ids.long(), physical * starts_block)
            block_ends = torch.zeros_like(physical)
            block_ends.scatter_reduce_(0, block_ids.long(), physical + 1, reduce="amax", include_self=True)
            offsets = (block_ids - local_positions // selection.compress_ratio).view(1, -1)
    else:
        bases = torch.arange((sequence_length + 3) // 4, device=ids.device, dtype=torch.int32) * 4
        block_bases = bases
        block_ends = (bases + 4).clamp_max(sequence_length)
        offsets = torch.zeros_like(positions)
        visible = (positions + 1) // selection.compress_ratio

    count = (ids >= 0).sum(dim=-1).to(torch.int32)
    if is_thd:
        visible = (positions + 1) // selection.compress_ratio
    global_ids = (ids + offsets.unsqueeze(-1)).masked_fill(ids < 0, 0)
    routes = torch.zeros((batch, sequence_length, ids.shape[-1] + 1), device=ids.device, dtype=torch.int32)
    routes[..., : ids.shape[-1]] = global_ids
    routes.scatter_(2, count.unsqueeze(-1).long(), (offsets + visible).unsqueeze(-1))
    has_tail = ((positions + 1) % selection.compress_ratio != 0).to(torch.int32)
    return routes, count + has_tail, block_bases, block_ends


def _tilelang_routes(selection: qsa.QSASelection, batch: int, sequence_length: int) -> Tensor:
    """Expand each selected four-token block and the current incomplete
    tail."""
    ids = selection.selected_ids.view(batch, sequence_length, -1)
    positions = selection.positions
    physical = torch.arange(sequence_length, device=ids.device, dtype=torch.int32).view(1, -1)
    document_start = physical - positions
    offsets = torch.arange(selection.compress_ratio, device=ids.device, dtype=torch.int32)
    selected = document_start[..., None, None] + ids[..., None] * selection.compress_ratio + offsets
    selected = selected.masked_fill(ids[..., None] < 0, -1).flatten(-2)
    tail_offsets = torch.arange(selection.compress_ratio - 1, device=ids.device, dtype=torch.int32)
    tail_count = (positions + 1) % selection.compress_ratio
    tail_start = document_start + (positions // selection.compress_ratio) * selection.compress_ratio
    tail = (tail_start[..., None] + tail_offsets).masked_fill(tail_offsets >= tail_count[..., None], -1)
    return torch.cat((selected, tail), dim=-1).contiguous()


def _community_id_sparse_forward(
    self: qsa.QSACoreAttention,
    query: Tensor,
    key: Tensor,
    value: Tensor,
    selection: qsa.QSASelection,
    is_thd: bool,
) -> Tensor:
    if selection.selected_ids is None or not query.is_cuda:
        return qsa.QSACoreAttention._relax_native_id_sparse_forward(self, query, key, value, selection, is_thd)
    if (
        selection.compress_ratio != 4
        or query.dtype != torch.bfloat16
        or key.dtype != query.dtype
        or value.dtype != query.dtype
    ):
        raise NotImplementedError("Community QSA attention requires BF16 Q/K/V and 4-token blocks")
    if is_thd:
        q, k, v = (tensor.unsqueeze(0).contiguous() for tensor in (query, key, value))
    else:
        q, k, v = (tensor.permute(1, 0, 2, 3).contiguous() for tensor in (query, key, value))
    batch, sequence_length, heads, width = q.shape
    if selection.positions.shape != (batch, sequence_length) or k.shape[1] != sequence_length:
        raise ValueError("Community QSA attention requires full-sequence selected IDs and K/V")

    if _BACKEND == "pr7553":
        from relax.models.qwen3_8.ops.qsa_community_ops import pr_triton_qsa

        routes, counts, bases, ends = _pr_routes(selection, batch, sequence_length, is_thd)
        output = pr_triton_qsa.qsa_sparse_attention(
            q,
            k,
            v,
            routes,
            counts,
            bases,
            ends,
            scale=self.softmax_scale,
            q_positions=torch.arange(sequence_length, device=q.device, dtype=torch.int64),
        )
    elif _BACKEND == "mixed":
        from relax.models.qwen3_8.ops.qsa_mixed_ops import attention_for_selection

        output = attention_for_selection(q, k, v, selection, self.softmax_scale, is_thd)
    else:
        from relax.models.qwen3_8.ops.qsa_community_ops.qsa_tilelang import tilelang_sparse_gqa_attention

        if width != 256:
            raise NotImplementedError("TileLang QSA attention requires head_dim=256")
        routes = _tilelang_routes(selection, batch, sequence_length)
        output = tilelang_sparse_gqa_attention(q, k, v, routes, softmax_scale=self.softmax_scale)

    if is_thd:
        return output.squeeze(0).reshape(sequence_length, heads * width)
    return output.permute(1, 0, 2, 3).reshape(sequence_length, batch, heads * width)
