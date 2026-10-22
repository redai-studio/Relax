# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Use validated producer CPU boundaries without changing QSA routing
arithmetic."""

from __future__ import annotations

import math
from contextvars import ContextVar
from typing import Any

import torch
from megatron.core.transformer.experimental_attention_variant import qsa
from torch import Tensor

from relax.backends.megatron.qsa_layout_metadata import get_cpu_layout


_layout: ContextVar[tuple[int, int, tuple[int, ...]] | None] = ContextVar("qsa_cpu_layout", default=None)
_enabled: ContextVar[bool] = ContextVar("qsa_cpu_layout_enabled", default=False)


def _sequence_layout(
    batch_size: int, seq_len: int, packed_seq_params: Any, device: Any, stage2_max_doc_len: int | None = None
) -> tuple[Tensor, Tensor, int]:
    layout = get_cpu_layout(packed_seq_params, batch_size, seq_len) if _enabled.get() else None
    _layout.set(layout)
    if layout is None or packed_seq_params is None or packed_seq_params.qkv_format != "thd":
        return qsa._relax_native_sequence_layout(batch_size, seq_len, packed_seq_params, device, stage2_max_doc_len)
    cu = packed_seq_params.cu_seqlens_q_padded
    if cu is None:
        cu = packed_seq_params.cu_seqlens_q
    starts = cu.to(device=device, dtype=torch.long)[:-1]
    rows = torch.arange(seq_len, device=device, dtype=torch.long)
    docs = (torch.searchsorted(starts, rows, right=True) - 1).clamp_min(0)
    positions = rows - starts[docs]
    return docs.unsqueeze(0).to(torch.int32), positions.unsqueeze(0).to(torch.int32), layout[1]


def _indexer_forward(
    self: Any, hidden_states: Tensor, rotary_pos_emb: Any, packed_seq_params: Any = None, output_format: str = "bits"
) -> Any:
    # Stage-2 and ID routing retain their existing paths, including autograd.
    enabled_token = _enabled.set(self.config.qsa_indexer_loss_coeff == 0 and output_format == "bits")
    layout_token = _layout.set(None)
    try:
        return qsa.QSAIndexer._relax_native_forward(
            self, hidden_states, rotary_pos_emb, packed_seq_params, output_format
        )
    finally:
        _layout.reset(layout_token)
        _enabled.reset(enabled_token)


def _pool_keys(
    self, raw_keys: Tensor, doc_ids: Tensor, positions: Tensor, max_doc_len: int
) -> tuple[Tensor, Tensor, Tensor]:
    """Mean-pool complete blocks of ``compress_ratio`` consecutive tokens per
    document.

    Args:
        raw_keys: [T, D] key of every token (flattened batch).
        doc_ids / positions: [T] int32.

    Returns:
        ``(pooled [n_docs, n_blocks_max, D], block_positions [n_docs, n_blocks_max],
        block_valid [n_docs, n_blocks_max])``. Only complete blocks are valid.
    """
    layout = _layout.get()
    if layout is None:
        return qsa.QSAIndexer._relax_native_pool_keys(self, raw_keys, doc_ids, positions, max_doc_len)
    R = self.compress_ratio
    n_docs = layout[0]
    n_blocks_max = max_doc_len // R
    D = raw_keys.shape[-1]
    block_of_token = positions.long() // R
    in_block = block_of_token < n_blocks_max  # tokens of the incomplete tail are dropped
    # A block is complete when its document has >= (block+1)*R tokens.
    doc_lens = torch.zeros(n_docs, dtype=torch.long, device=raw_keys.device)
    doc_lens.scatter_add_(0, doc_ids.long(), torch.ones_like(doc_ids, dtype=torch.long))
    blocks_per_doc = doc_lens // R
    block_valid = torch.arange(n_blocks_max, device=raw_keys.device).unsqueeze(0) < blocks_per_doc.unsqueeze(1)
    flat_index = (doc_ids.long() * n_blocks_max + block_of_token)[in_block]
    pooled = torch.zeros(n_docs * n_blocks_max, D, dtype=torch.float32, device=raw_keys.device)
    pooled.index_add_(0, flat_index, raw_keys[in_block].float())
    pooled = (pooled / R).to(raw_keys.dtype).view(n_docs, n_blocks_max, D)
    pooled = pooled * block_valid.unsqueeze(-1).to(pooled.dtype)
    block_positions = (
        (torch.arange(n_blocks_max, device=raw_keys.device, dtype=torch.long) * R).unsqueeze(0).expand(n_docs, -1)
    )
    return pooled, block_positions, block_valid


@torch.no_grad()
def _select_blocks(
    self,
    q: Tensor,
    pooled_keys: Tensor,
    block_valid: Tensor,
    doc_ids: Tensor,
    positions: Tensor,
    uniform_doc_len: int | None = None,
    output_format: str = "bits",
) -> tuple[Tensor, bool]:
    """Top-k complete blocks per query.

    Args:
        q: [T, H, D] normalized + rotated indexer queries.
        pooled_keys: [n_docs, n_blocks_max, D] normalized + rotated block keys.
        block_valid: [n_docs, n_blocks_max].
        doc_ids / positions: [T] int32.
        uniform_doc_len: Length of each document in the non-THD batch layout.
        output_format: ``bits`` retains the existing Flex mask contract;
            ``ids`` retains ``top.indices`` without building a bitset.

    Returns:
        ``(selection, all_selected)``; selection is either
        ``selected_bits [T, nbytes] uint8`` or ``selected_ids [T, K] int32``.
    """
    layout = _layout.get()
    if layout is None:
        return qsa.QSAIndexer._relax_native_select_blocks(
            self, q, pooled_keys, block_valid, doc_ids, positions, uniform_doc_len, output_format
        )
    T, H, D = q.shape
    R = self.compress_ratio
    n_blocks_max = pooled_keys.shape[1]
    if output_format not in ("bits", "ids"):
        raise ValueError(f"unsupported QSA selection format: {output_format}")
    # One extra (always clear) bit for the incomplete tail block: the mask kernels look up
    # ``position // R`` for every key, including keys of the block that is still open.
    nbytes = (n_blocks_max + 1 + 7) // 8
    # Retain compact TopK IDs directly for the selected-ID kernel. This
    # avoids the O(T * n_blocks_max) bitset, including at long lengths.
    if output_format == "ids":
        selected = torch.full((T, self.block_topk), -1, dtype=torch.int32, device=q.device)
    else:
        selected = torch.empty(T, nbytes, dtype=torch.uint8, device=q.device)
    visible = ((positions.long() + 1) // R).clamp_max(n_blocks_max)  # complete blocks per query
    all_selected = layout[1] // R <= self.block_topk
    block_ids = torch.arange(n_blocks_max, device=q.device)
    # Bound scores, converted queries and the temporary scatter buffer together. The
    # selected-ID output remains O(T * block_topk) when requested.
    output_bytes = 4 * self.block_topk if output_format == "ids" else nbytes
    bytes_per_query = 4 * (n_blocks_max * H + H * D + output_bytes)
    tile = max(1, min(T, qsa._QSA_SELECT_TILE_BYTES // max(1, bytes_per_query)))
    if all_selected or n_blocks_max == 0:
        # Every visible block is selected: the pattern is plain causal attention. Still
        # materialize the requested output so forced sparse runs see the selection.
        for start in range(0, T, tile):
            end = min(T, start + tile)
            if output_format == "ids":
                ids = torch.arange(self.block_topk, device=q.device, dtype=torch.int32)
                selected[start:end] = torch.where(ids[None, :] < visible[start:end, None], ids[None, :], -1)
                continue
            bits = torch.zeros(end - start, nbytes, dtype=torch.int32, device=q.device)
            if n_blocks_max > 0:
                vis = (block_ids.unsqueeze(0) < visible[start:end].unsqueeze(1)).to(torch.int32)
                weights = (1 << (block_ids & 7)).to(torch.int32).unsqueeze(0)
                byte_idx = (block_ids >> 3).unsqueeze(0).expand(end - start, -1)
                bits.scatter_add_(1, byte_idx, vis * weights)
            selected[start:end] = bits.to(torch.uint8)
        return selected, True

    n_docs = pooled_keys.shape[0]
    k = min(self.block_topk, n_blocks_max)
    # Multiple short packed documents make a GEMM and scatter per document slower than
    # one batched GEMM. Permit a gathered-key path only when its query tile, including
    # both the gathered source and fp32 copy, fits the same workspace budget. Long
    # documents keep the shared-key path below and never expand keys per query.
    # Require at least one average document per gathered tile so a few long
    # documents do not trade their shared GEMMs for many gathered-key GEMMs.
    gathered_bytes_per_query = (
        n_blocks_max * D * (pooled_keys.element_size() + 4)
        + n_blocks_max * (H * 4 + 8)
        + H * D * 4
        + output_bytes * 4
        + k * 32
    )
    gathered_tile = max(1, min(T, qsa._QSA_SELECT_TILE_BYTES // gathered_bytes_per_query))
    if n_docs > 1 and gathered_tile >= min(T, 32) and T <= n_docs * gathered_tile:
        for start in range(0, T, gathered_tile):
            end = min(T, start + gathered_tile)
            q_c = q[start:end].float()
            k_c = pooled_keys[doc_ids[start:end].long()].float()
            scores = torch.einsum("thd,tnd->thn", q_c, k_c)
            scores = scores.relu_().sum(dim=1) / math.sqrt(D)
            vis = block_ids.unsqueeze(0) < visible[start:end].unsqueeze(1)
            scores.masked_fill_(~vis, float("-inf"))
            top = torch.topk(scores, k=k, dim=-1)
            valid = torch.isfinite(top.values)
            if output_format == "ids":
                selected[start:end, :k] = top.indices.masked_fill(~valid, -1).to(torch.int32)
                continue
            blocks = top.indices.masked_fill(~valid, 0)
            bit_val = ((1 << (blocks & 7)) * valid.to(blocks.dtype)).to(torch.int32)
            bits = torch.zeros(end - start, nbytes, dtype=torch.int32, device=q.device)
            bits.scatter_add_(1, blocks >> 3, bit_val)
            selected[start:end] = bits.to(torch.uint8)
        return selected, False

    # A single document or a non-THD batch has offsets known from tensor shapes.
    # For packed long documents, doc_ids are monotone after CP reconstruction;
    # empty documents give equal adjacent offsets and need no work.
    offsets = layout[2]
    for doc in range(n_docs):
        doc_start, doc_end = offsets[doc], offsets[doc + 1]
        if doc_start == doc_end:
            continue
        key_t = pooled_keys[doc].float().T  # shared [D, n_blocks_max] GEMM operand
        for start in range(doc_start, doc_end, tile):
            end = min(doc_end, start + tile)
            q_c = q[start:end].float().reshape(-1, D)  # [t * H, D]
            scores = (q_c @ key_t).view(end - start, H, n_blocks_max)
            scores = scores.relu_().sum(dim=1) / math.sqrt(D)
            vis = block_ids.unsqueeze(0) < visible[start:end].unsqueeze(1)
            scores.masked_fill_(~vis, float("-inf"))
            top = torch.topk(scores, k=k, dim=-1)
            valid = torch.isfinite(top.values)
            if output_format == "ids":
                selected[start:end, :k] = top.indices.masked_fill(~valid, -1).to(torch.int32)
                continue
            blocks = top.indices.masked_fill(~valid, 0)  # compact [t, k] block IDs
            byte_idx = blocks >> 3
            # Top-k blocks are distinct; summing their bit values is equivalent to OR.
            bit_val = ((1 << (blocks & 7)) * valid.to(blocks.dtype)).to(torch.int32)
            bits = torch.zeros(end - start, nbytes, dtype=torch.int32, device=q.device)
            bits.scatter_add_(1, byte_idx, bit_val)
            selected[start:end] = bits.to(torch.uint8)
    return selected, False
