# Copyright (c) 2026 Relax Authors. All Rights Reserved.
# Copyright 2025-2026 RadixArk, Inc. (miles).
# Copyright 2025 Zhipu AI.
# SPDX-License-Identifier: Apache-2.0
# Adapted for Relax.

"""Frozen CP1 selector with TP query sharding and native block-ID output."""

from __future__ import annotations

import os
from typing import Any

import torch
import torch.distributed as dist
from torch import Tensor

from relax.distributed.replay_manager.codecs import _QSABitmapReplay


def gemma_rmsnorm(x: Tensor, weight: Tensor, eps: float) -> Tensor:
    """Keep the image's FP32 reduction and zero-centered weight arithmetic."""
    acc = x.dtype if x.dtype in (torch.float32, torch.float64) else torch.float32
    xa = x.to(acc)
    var = xa.pow(2).mean(dim=-1, keepdim=True)
    return ((xa * torch.rsqrt(var + eps)) * (1.0 + weight.to(acc))).to(x.dtype)


def index_scores(query: Tensor, keys: Tensor) -> Tensor:
    """BF16 GEMM with FP32 output and full-precision reduction."""
    if query.dtype != torch.bfloat16 or keys.dtype != torch.bfloat16:
        raise NotImplementedError("TP selector requires BF16 queries and pooled keys")
    rows, heads, dim = query.shape
    previous = torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
    try:
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
        scores = torch.mm(query.reshape(-1, dim), keys.T, out_dtype=torch.float32)
    finally:
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = previous
    return scores.reshape(rows, heads, keys.shape[0]).transpose(1, 2)


@torch.compile(fullgraph=True, dynamic=True)
def compiled_score_topk(query: Tensor, keys: Tensor, positions: Tensor, topk: int) -> Tensor:
    """Fuse score postprocessing; keep the same FP32 GEMM output and sorted
    top-k."""
    rows, heads, dim = query.shape
    scores = torch.mm(query.reshape(-1, dim), keys.T, out_dtype=torch.float32)
    scores = scores.reshape(rows, heads, keys.shape[0]).transpose(1, 2)
    logits = scores.relu().sum(-1) / 128**0.5
    visible = torch.arange(keys.shape[0], device=query.device)[None, :] < (positions[:, None] + 1) // 4
    logits = logits.masked_fill(~visible, -torch.inf)
    values, ids = logits.topk(min(topk, keys.shape[0]), dim=-1)
    return ids.masked_fill(values == -torch.inf, -1).to(torch.int32)


def document_bounds(packed: Any, total: int) -> tuple[int, ...]:
    """Read the producer's immutable CPU layout without CUDA scalar reads."""
    from relax.backends.megatron.qsa_layout_metadata import valid_physical_layout

    if packed is None:
        return (0, total)
    if packed.qkv_format != "thd":
        raise NotImplementedError("TP selector supports packed THD or one un-packed document")
    record = valid_physical_layout(packed)
    if getattr(packed, "qsa_physical_layout_cpu", None) is not None and record is None:
        raise ValueError("TP selector received a stale producer physical-layout binding")
    bounds = record[2] if record is not None else getattr(packed, "qsa_select_offsets_cpu", None)
    cu = packed.cu_seqlens_q_padded
    if cu is None:
        cu = packed.cu_seqlens_q
    if (
        not isinstance(bounds, tuple)
        or len(bounds) < 2
        or bounds[0] != 0
        or bounds[-1] != total
        or any(type(value) is not int or value < 0 for value in bounds)
        or any(left > right for left, right in zip(bounds, bounds[1:]))
        or tuple(cu.shape) != (len(bounds),)
    ):
        raise ValueError("TP selector requires exact producer CPU document boundaries")
    return bounds


def query_intervals(first: int, last: int, missing: tuple[int, ...] | None) -> tuple[tuple[int, int], ...]:
    """Coalesce authoritative CPU missing rows without device
    synchronization."""
    if missing is None:
        return ((first, last),) if first < last else ()
    result = []
    for row in missing:
        if first <= row < last:
            if result and result[-1][1] == row:
                result[-1] = (result[-1][0], row + 1)
            else:
                result.append((row, row + 1))
    return tuple(result)


@torch.no_grad()
def forward(
    self: Any,
    hidden_states: Tensor,
    rotary_pos_emb: Any,
    packed_seq_params: Any = None,
    output_format: str = "bits",
    *,
    replay_data: tuple[Tensor, Tensor] | _QSABitmapReplay | None = None,
) -> Any:
    """Run image norm/pool/RoPE/TopK, sharding query work over the explicit TP
    group."""
    from megatron.core.models.common.embeddings.rope_utils import _apply_rotary_pos_emb_bshd, apply_rotary_pos_emb
    from megatron.core.tensor_parallel.mappings import gather_from_sequence_parallel_region
    from megatron.core.transformer.experimental_attention_variant.qsa import QSASelection

    if (
        self.config.context_parallel_size != 1
        or self.config.qsa_indexer_loss_coeff != 0
        or output_format != "ids"
        or any(parameter.requires_grad for parameter in self.parameters())
    ):
        raise NotImplementedError("TP selector supports CP1 frozen Stage-1 selected IDs")
    if not self.config.layernorm_zero_centered_gamma:
        raise NotImplementedError("Gemma RMSNorm requires zero-centered gamma weights")
    if (
        self.n_heads != 4
        or self.kv_heads != 1
        or self.head_dim != 128
        or self.compress_ratio != 4
        or not 0 < self.token_budget <= 2048
        or self.token_budget % self.compress_ratio
    ):
        raise NotImplementedError("TP selector requires 4Q/1KV/D128, ratio 4 and budget <= 2048")
    tp, cp = self.pg_collection.tp, self.pg_collection.cp
    if tp is None or cp is None or dist.get_world_size(group=cp) != 1:
        raise ValueError("TP selector requires explicit TP and singleton CP process groups")
    if hidden_states.ndim != 3 or hidden_states.shape[1] != 1 or not hidden_states.is_cuda:
        raise NotImplementedError("TP selector requires CUDA [sequence, 1, hidden] input")
    if hidden_states.dtype != torch.bfloat16 or rotary_pos_emb is None:
        raise ValueError("TP selector requires BF16 hidden states and attention RoPE")
    freqs = rotary_pos_emb[0] if isinstance(rotary_pos_emb, tuple) else rotary_pos_emb
    if freqs.ndim != 4 or tuple(freqs.shape[1:3]) != (1, 1):
        raise NotImplementedError("TP selector currently supports standard 4D rotary frequencies")
    size, rank = dist.get_world_size(group=tp), dist.get_rank(group=tp)
    # Reference indexer rows are replicated (not SP sliced) on every TP rank.
    # Thus every rank takes the same gather branch, even if its own query shard
    # has no holes. Legacy raw records without CPU metadata keep full fallback.
    bitmap = isinstance(replay_data, _QSABitmapReplay)
    if bitmap and replay_data.bits.device != hidden_states.device:
        raise ValueError("Direct QSA bitmap device differs from the model input")
    missing = (
        replay_data.missing_rows
        if bitmap
        else getattr(replay_data[0], "_relax_indexer_missing_rows", None)
        if replay_data is not None
        else None
    )
    reference_rows = replay_data is not None and missing is not None
    total = (
        (replay_data.bits.shape[0] if bitmap else replay_data[0].shape[0])
        if reference_rows
        else hidden_states.shape[0] * (size if self.config.sequence_parallel else 1)
    )
    if total == 0 or total % size:
        raise ValueError("TP selector requires a nonempty sequence divisible by TP size")
    bounds = document_bounds(packed_seq_params, total)
    maximum = max(b - a for a, b in zip(bounds, bounds[1:]))
    ratio, budget, topk = self.compress_ratio, self.token_budget, self.block_topk
    local_projection = (
        reference_rows
        and missing != ()
        and size > 1
        and self.config.sequence_parallel
        and not self.config.apply_rope_fusion
        and any(
            end - start > budget + ratio - 1 and query_intervals(start, end, missing)
            for start, end in zip(bounds, bounds[1:])
        )
    )
    if missing != () and self.config.sequence_parallel and size > 1 and not local_projection:
        hidden_states = gather_from_sequence_parallel_region(
            hidden_states, tensor_parallel_output_grad=False, group=tp
        )
    hidden = hidden_states[:, 0]
    cu = None
    if packed_seq_params is not None:
        cu = packed_seq_params.cu_seqlens_q_padded
        if cu is None:
            cu = packed_seq_params.cu_seqlens_q
    layout_cu = cu if cu is not None else torch.tensor(bounds, dtype=torch.int32, device=hidden.device)
    rows = torch.arange(total, dtype=torch.int32, device=hidden.device)
    docs = torch.searchsorted(layout_cu[1:], rows, right=True)
    positions = rows - layout_cu[docs.long()]
    all_short = maximum <= budget + ratio - 1
    # The image's all-short zero-row sentinel becomes legal native block IDs;
    # projection, norm and scoring are still skipped entirely for these packs.
    shard = total // size
    first_query, last_query = rank * shard, (rank + 1) * shard
    local_missing = tuple(row for row in missing if first_query <= row < last_query) if bitmap else ()
    compact_rows = {row: i for i, row in enumerate(local_missing)}
    if bitmap:
        selected = torch.full((len(local_missing), topk), -1, dtype=torch.int32, device=hidden.device)
    elif replay_data is not None and missing == ():
        selected = replay_data[0]
    elif replay_data is not None and missing is not None:
        selected = replay_data[0][first_query:last_query].clone()
    else:
        selected = torch.full((shard, topk), -1, dtype=torch.int32, device=hidden.device)
    needs_projection = not all_short and (
        replay_data is None
        or any(
            end - start > budget + ratio - 1
            and query_intervals(max(start, first_query), min(end, last_query), missing)
            for start, end in zip(bounds, bounds[1:])
        )
    )
    if needs_projection or local_projection:
        qk, _ = self.index_qk_proj(hidden)
        query_rows = shard if local_projection else total
        q = gemma_rmsnorm(
            qk[:, : self.n_heads * self.head_dim].reshape(query_rows, self.n_heads, self.head_dim),
            self.q_layernorm.weight,
            self.config.layernorm_epsilon,
        )
        raw_keys = qk[:, self.n_heads * self.head_dim :].reshape(query_rows, self.head_dim)
        if local_projection:
            combined_keys = raw_keys.new_empty((total, self.head_dim))
            dist.all_gather_into_tensor(combined_keys, raw_keys.contiguous(), group=tp)
            raw_keys = combined_keys
            # Match CP1 THD RoPE's packed-absolute vs per-document frequency
            # lookup, including its padding clamp, on only this query shard.
            freq_rows = (
                rows[first_query:last_query]
                if cu is None or freqs.shape[0] > maximum
                else positions[first_query:last_query]
            )
            local_freqs = freqs[freq_rows.long().clamp(min=0, max=freqs.shape[0] - 1)]
            q = _apply_rotary_pos_emb_bshd(
                q.unsqueeze(1), local_freqs, rotary_interleaved=self.config.rotary_interleaved
            ).squeeze(1)
        elif cu is None:
            q = _apply_rotary_pos_emb_bshd(
                q.unsqueeze(1), freqs, rotary_interleaved=self.config.rotary_interleaved
            ).squeeze(1)
        else:
            q = apply_rotary_pos_emb(q, freqs, config=self.config, cu_seqlens=cu, cp_group=cp, max_seqlen=maximum)
    absolute = getattr(self.config, "mrope_section", None) is not None and freqs.shape[0] == total
    for start, end in zip(bounds, bounds[1:]):
        intervals = query_intervals(max(start, first_query), min(end, last_query), missing)
        if not intervals:
            continue
        blocks = (end - start) // ratio
        if end - start <= budget + ratio - 1:
            ids = torch.arange(min(topk, blocks), device=hidden.device, dtype=torch.int32)
            for first, last in intervals:
                local_positions = torch.arange(first - start, last - start, device=hidden.device)
                destination = compact_rows[first] if bitmap else first - first_query
                selected[destination : destination + last - first, : ids.numel()] = torch.where(
                    ids[None, :] < (local_positions[:, None] + 1) // ratio, ids[None, :], -1
                )
            continue
        pooled = raw_keys[start : start + blocks * ratio].float().reshape(blocks, ratio, self.head_dim).mean(1)
        pooled = gemma_rmsnorm(pooled.to(raw_keys.dtype), self.k_layernorm.weight, self.config.layernorm_epsilon)
        block_positions = torch.arange(blocks, device=hidden.device) * ratio
        key_freqs = freqs[block_positions + (start if absolute else 0)]
        pooled = _apply_rotary_pos_emb_bshd(
            pooled[:, None, None, :], key_freqs, rotary_interleaved=self.config.rotary_interleaved
        )[:, 0, 0]
        chunks = (
            (a, min(a + 512, last - start))
            for first, last in intervals
            for a in range(first - start, last - start, 512)
        )
        for a, b in chunks:
            local_positions = torch.arange(a, b, device=hidden.device)
            q_start = start + a - (first_query if local_projection else 0)
            q_stop = start + b - (first_query if local_projection else 0)
            if os.environ.get("RELAX_QWEN38_QSA_SELECTOR_COMPILE", "0") == "1":
                previous = torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
                try:
                    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
                    ids = compiled_score_topk(q[q_start:q_stop], pooled, local_positions, topk)
                finally:
                    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = previous
            else:
                scores = index_scores(q[q_start:q_stop], pooled)
                logits = scores.relu().sum(-1) / self.head_dim**0.5
                visible = torch.arange(blocks, device=hidden.device)[None, :] < (local_positions[:, None] + 1) // ratio
                logits.masked_fill_(~visible, -torch.inf)
                values, ids = logits.topk(min(topk, blocks), dim=-1)
                ids.masked_fill_(values == -torch.inf, -1)
            destination = compact_rows[start + a] if bitmap else start + a - first_query
            selected[destination : destination + b - a, : ids.shape[1]] = ids.to(torch.int32)
    if size > 1 and missing != ():
        if replay_data is not None and missing is not None:
            # Every TP rank already has the captured rows. Only one query
            # shard owns each hole, so SUM of (id + 1) preserves exact int32
            # IDs, including -1 padding, without gathering the full selection.
            updates = selected.new_zeros((len(missing), topk))
            owned = [
                (i, compact_rows[row] if bitmap else row - first_query)
                for i, row in enumerate(missing)
                if first_query <= row < last_query
            ]
            host_rows = torch.empty(len(missing), dtype=torch.long, pin_memory=True)
            host_rows.numpy()[:] = missing
            missing_rows = host_rows.to(hidden.device, non_blocking=True)
            if owned:
                host_owned = torch.empty((2, len(owned)), dtype=torch.long, pin_memory=True)
                host_owned.numpy()[:] = tuple(zip(*owned))
                destinations, local_rows = host_owned.to(hidden.device, non_blocking=True).unbind()
                updates.index_copy_(0, destinations, selected.index_select(0, local_rows) + 1)
            dist.all_reduce(updates, op=dist.ReduceOp.SUM, group=tp)
            if bitmap:
                selected = updates - 1
            else:
                selected = replay_data[0].clone()
                selected.index_copy_(0, missing_rows, updates - 1)
        else:
            combined = selected.new_empty((total, topk))
            dist.all_gather_into_tensor(combined, selected.contiguous(), group=tp)
            selected = combined
    if replay_data is not None and missing is None:
        selected = torch.where(replay_data[1][:, None], replay_data[0], selected)
    bits = None
    if bitmap:
        words = ((maximum // ratio + 1) + 31) // 32
        if replay_data.bits.shape[1] < words:
            dense = torch.nn.functional.pad(replay_data.bits, (0, words - replay_data.bits.shape[1]))
        else:
            dense = replay_data.bits.clone()
        words = dense.shape[1]
        if missing:
            from relax.distributed.replay_manager._bitmap import scatter_missing_bits_into

            if size == 1:
                host_rows = torch.empty(len(missing), dtype=torch.long, pin_memory=True)
                host_rows.numpy()[:] = missing
                missing_rows = host_rows.to(hidden.device, non_blocking=True)
            scatter_missing_bits_into(selected, missing_rows, dense)
        bits = dense.view(torch.uint8).reshape(-1)
    selection = QSASelection(
        doc_ids=docs.to(torch.int32).unsqueeze(0),
        positions=positions.to(torch.int32).unsqueeze(0),
        selected_bits=bits,
        bits_per_row=words * 4 if bitmap else 0,
        bits_per_row_t=(
            torch.full((), words * 4, dtype=torch.int64, device=hidden.device)
            if bitmap
            else torch.zeros((), dtype=torch.int64, device=hidden.device)
        ),
        compress_ratio=ratio,
        all_selected=False,
        selected_ids=None if bitmap else selected,
        max_complete_blocks=maximum // ratio,
    )
    from relax.models.qwen3_8.ops.qsa_mixed_ops.selection import bind_document_bounds

    return bind_document_bounds(selection, bounds)
