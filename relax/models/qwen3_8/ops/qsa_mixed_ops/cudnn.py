# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Mixed Flash/PR1765 attention using the frozen selector's native block IDs.

The PR1765 package is a separately pinned optional dependency. Every autograd
invocation owns its metadata; no selection/activation cache spans microbatches.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor
from torch.autograd.function import once_differentiable

from relax.utils.env import Envs


def document_groups(bounds: tuple[int, ...], total: int) -> tuple[tuple, tuple]:
    """Use the same dense causal boundary as the original mixed0 backend."""
    if (
        not isinstance(bounds, tuple)
        or len(bounds) < 2
        or bounds[0] != 0
        or bounds[-1] != total
        or any(type(value) is not int for value in bounds)
        or any(left > right for left, right in zip(bounds, bounds[1:]))
    ):
        raise ValueError("Flash/cuDNN mixed attention requires exact producer CPU boundaries")
    flash_max_len = Envs.RELAX_QWEN38_QSA_MIXED_FLASH_MAX_LEN
    if flash_max_len == -1:
        flash_max_len = 2051
    if not 0 <= flash_max_len <= 2051:
        raise ValueError("Mixed QSA Flash maximum length exceeds the dense selector budget")
    short, long = [], []
    for start, end in zip(bounds, bounds[1:]):
        if start != end:
            (short if end - start <= flash_max_len else long).append((start, end))
    return tuple(short), tuple(long)


def block_metadata(ids: Tensor) -> tuple[Tensor, ...]:
    """Build document-local PR metadata, including the incomplete causal tail.

    Native IDs select complete four-token blocks. PR1765 consumes a byte mask,
    and will not add the current incomplete block unless we mark it explicitly.
    All operations stay on the input device, without reading CUDA scalar
    values.
    """
    if ids.ndim != 2 or ids.shape[0] < 1 or ids.shape[1] != 512 or ids.dtype != torch.int32:
        raise ValueError("PR1765 mixed attention requires int32 [document_tokens, 512] block IDs")
    tokens = ids.shape[0]
    blocks = (tokens + 3) // 4
    positions = torch.arange(tokens, dtype=torch.int32, device=ids.device)
    visible = (positions[:, None] + 1) // 4
    valid = (ids >= 0) & (ids < visible)
    selected = torch.zeros((tokens, blocks), dtype=torch.uint8, device=ids.device)
    selected.scatter_reduce_(1, ids.clamp(0, blocks - 1).long(), valid.byte(), reduce="amax", include_self=True)
    tail = ((positions[:, None] + 1) % 4 != 0).byte()
    selected.scatter_reduce_(1, (positions[:, None] // 4).long(), tail, reduce="amax", include_self=True)
    zero = torch.zeros_like(positions)
    first = torch.arange(blocks, dtype=torch.int32, device=ids.device) * 4
    last = (first + 3).clamp_max(tokens - 1)
    return selected, zero, positions, zero, zero, first, last


def packed_block_metadata(ids: Tensor, bounds: tuple[int, ...]) -> tuple[Tensor, ...]:
    """Keep each document's block numbering independent inside a small NV
    pack."""
    if ids.ndim != 2 or ids.shape[1] != 512 or ids.dtype != torch.int32:
        raise ValueError("Packed PR1765 requires int32 [packed_tokens, 512] block IDs")
    total = ids.shape[0]
    document_groups(bounds, total)
    lengths = tuple(end - start for start, end in zip(bounds, bounds[1:]))
    counts = tuple((length + 3) // 4 for length in lengths)
    block_offsets = [0]
    for count in counts:
        block_offsets.append(block_offsets[-1] + count)
    blocks = block_offsets[-1]
    if not blocks:
        raise ValueError("Packed PR1765 metadata requires at least one token")
    repeats = torch.tensor(lengths, dtype=torch.int32, device=ids.device)
    # Supplying output_size avoids a CUDA scalar read for repeat_interleave.
    tok_base = torch.repeat_interleave(
        torch.tensor(bounds[:-1], dtype=torch.int32, device=ids.device), repeats, output_size=total
    )
    blk_base = torch.repeat_interleave(
        torch.tensor(block_offsets[:-1], dtype=torch.int32, device=ids.device), repeats, output_size=total
    )
    hi = torch.arange(total, dtype=torch.int32, device=ids.device)
    local = hi - tok_base
    valid = (ids >= 0) & (ids < (local[:, None] + 1) // 4)
    selected = torch.zeros((total, blocks), dtype=torch.uint8, device=ids.device)
    indices = (ids.long() + blk_base[:, None]).clamp(0, blocks - 1)
    selected.scatter_reduce_(1, indices, valid.byte(), reduce="amax", include_self=True)
    tail = ((local[:, None] + 1) % 4 != 0).byte()
    selected.scatter_reduce_(
        1, (blk_base[:, None] + local[:, None] // 4).long(), tail, reduce="amax", include_self=True
    )
    first_parts = [
        torch.arange(count, dtype=torch.int32, device=ids.device) * 4 + start
        for start, count in zip(bounds[:-1], counts)
    ]
    first = torch.cat(first_parts)
    last = torch.cat([(part + 3).clamp_max(end - 1) for part, end in zip(first_parts, bounds[1:])])
    return selected, tok_base, hi, blk_base, tok_base, first, last


def long_document_packs(spans: tuple[tuple[int, int], ...], maximum: int) -> tuple[tuple[tuple[int, int], ...], ...]:
    """Bound optional batching without crossing the PR's 16K forward
    dispatch."""
    if not 0 <= maximum <= 16384:
        raise ValueError("NV mini-pack maximum must be between 0 and 16384 tokens")
    packs, current, size = [], [], 0
    for span in spans:
        length = span[1] - span[0]
        if current and (not maximum or size + length > maximum):
            packs.append(tuple(current))
            current, size = [], 0
        current.append(span)
        size += length
    if current:
        packs.append(tuple(current))
    return tuple(packs)


class _CudnnAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, q: Tensor, k: Tensor, v: Tensor, ids: Tensor, scale: float) -> Tensor:
        from cudnn.QSA.api import qsa_forward

        q, k, v = (tensor.contiguous() for tensor in (q, k, v))
        metadata = block_metadata(ids)
        out, out_fp32, lse = qsa_forward(q, k, v, *metadata, scale, 4)
        ctx.save_for_backward(q, k, v, out_fp32, lse, *metadata)
        ctx.scale = scale
        return out

    @staticmethod
    @once_differentiable
    def backward(ctx: Any, grad: Tensor) -> tuple[Tensor, Tensor, Tensor, None, None]:
        from cudnn.QSA.api import qsa_backward

        q, k, v, out_fp32, lse, *metadata = ctx.saved_tensors
        dq, dk, dv = qsa_backward(grad.contiguous(), q, k, v, out_fp32, lse, *metadata, ctx.scale, 4)
        return dq, dk, dv, None, None


class _PackedCudnnAttention(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any, q: Tensor, k: Tensor, v: Tensor, ids: Tensor, scale: float, bounds: tuple[int, ...]
    ) -> Tensor:
        from cudnn.QSA.api import qsa_forward

        q, k, v = (tensor.contiguous() for tensor in (q, k, v))
        metadata = packed_block_metadata(ids, bounds)
        out, out_fp32, lse = qsa_forward(q, k, v, *metadata, scale, 4)
        ctx.save_for_backward(q, k, v, out_fp32, lse, *metadata)
        ctx.scale = scale
        return out

    @staticmethod
    @once_differentiable
    def backward(ctx: Any, grad: Tensor) -> tuple[Tensor, Tensor, Tensor, None, None, None]:
        from cudnn.QSA.api import qsa_backward

        q, k, v, out_fp32, lse, *metadata = ctx.saved_tensors
        dq, dk, dv = qsa_backward(grad.contiguous(), q, k, v, out_fp32, lse, *metadata, ctx.scale, 4)
        return dq, dk, dv, None, None, None


def _flash(q: Tensor, k: Tensor, v: Tensor, lengths: tuple[int, ...], scale: float) -> Tensor:
    from flash_attn import flash_attn_varlen_func

    bounds = [0]
    for length in lengths:
        bounds.append(bounds[-1] + length)
    cu = torch.tensor(bounds, dtype=torch.int32, device=q.device)
    maximum = max(lengths)
    return flash_attn_varlen_func(
        q.contiguous(), k.contiguous(), v.contiguous(), cu, cu, maximum, maximum, softmax_scale=scale, causal=True
    )


def attention_for_selection(
    query: Tensor, key: Tensor, value: Tensor, selection: Any, scale: float, bounds: tuple[int, ...]
) -> Tensor:
    """Keep short documents batched for Flash; run NV on each long document.

    Document-local NV calls avoid a quadratic byte mask across unrelated packed
    documents. Routing, copies and metadata remain part of end-to-end timing.
    The caller retains the existing TP2/CP1/frozen-indexer mixed contract.
    """
    if query.ndim != 4 or key.ndim != 4 or value.shape != key.shape:
        raise ValueError("Flash/cuDNN mixed attention requires batch-major BTHD Q/K/V")
    batch, total, heads, dim = query.shape
    if heads != 12 or dim != 256 or key.shape != (batch, total, 1, 256):
        raise NotImplementedError("PR1765 mixed integration requires TP2 local 12Q/1KV/D256")
    if query.dtype != torch.bfloat16 or key.dtype != query.dtype or value.dtype != query.dtype:
        raise NotImplementedError("PR1765 mixed integration requires BF16 Q/K/V")
    if selection.compress_ratio != 4 or selection.selected_ids is None:
        raise ValueError("PR1765 mixed integration requires native four-token block IDs")
    ids = selection.selected_ids
    if ids.dtype != torch.int32 or ids.numel() != batch * total * 512:
        raise ValueError("Flash/cuDNN mixed integration requires the production 2048-token selector budget")
    short, long = document_groups(bounds, total)
    if total == 0:
        return query.clone()
    if not long:
        lengths = tuple(end - start for start, end in short)
        return torch.stack([_flash(q, k, v, lengths, scale) for q, k, v in zip(query, key, value)])
    ids = ids.reshape(batch, total, 512)
    maximum = Envs.RELAX_QWEN38_QSA_NV_MAX_PACK_TOKENS
    packs = long_document_packs(long, maximum)
    outputs = []
    lengths = tuple(end - start for start, end in zip(bounds, bounds[1:]))
    document_indices = {span: index for index, span in enumerate(zip(bounds, bounds[1:]))}
    for row in range(batch):
        # A shared split collects gradients once instead of expanding every
        # document view back into a full packed Q/K/V gradient tensor.
        row_qkv = tuple(tensor[row] for tensor in (query, key, value))
        document_qkv = tuple(tensor.split(lengths, dim=0) for tensor in row_qkv)
        parts, order = [], []
        if short:
            rows = torch.cat([torch.arange(start, end, device=query.device) for start, end in short])
            q, k, v = (tensor.index_select(0, rows) for tensor in row_qkv)
            parts.append(_flash(q, k, v, tuple(end - start for start, end in short), scale))
            order.append(rows)
        for pack in packs:
            if len(pack) == 1:
                start, end = pack[0]
                index = document_indices[(start, end)]
                q, k, v = (chunks[index] for chunks in document_qkv)
                parts.append(_CudnnAttention.apply(q, k, v, ids[row, start:end], scale))
                order.append(torch.arange(start, end, device=query.device))
                continue
            indices = [document_indices[span] for span in pack]
            q, k, v = (torch.cat([chunks[index] for index in indices]) for chunks in document_qkv)
            packed_ids = torch.cat([ids[row, start:end] for start, end in pack])
            packed_bounds = [0]
            for start, end in pack:
                packed_bounds.append(packed_bounds[-1] + end - start)
            parts.append(_PackedCudnnAttention.apply(q, k, v, packed_ids, scale, tuple(packed_bounds)))
            order.append(torch.cat([torch.arange(start, end, device=query.device) for start, end in pack]))
        if not short:
            outputs.append(torch.cat(parts))
        else:
            permutation = torch.cat(order)
            inverse = torch.empty_like(permutation)
            inverse.scatter_(0, permutation, torch.arange(total, device=query.device))
            outputs.append(torch.cat(parts).index_select(0, inverse))
    return torch.stack(outputs)
