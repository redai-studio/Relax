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


@torch.no_grad()
def forward(
    self: Any,
    hidden_states: Tensor,
    rotary_pos_emb: Any,
    packed_seq_params: Any = None,
    output_format: str = "bits",
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
    if self.config.sequence_parallel and size > 1:
        hidden_states = gather_from_sequence_parallel_region(
            hidden_states, tensor_parallel_output_grad=False, group=tp
        )
    hidden = hidden_states[:, 0]
    total = hidden.shape[0]
    if total == 0 or total % size:
        raise ValueError("TP selector requires a nonempty sequence divisible by TP size")
    bounds = document_bounds(packed_seq_params, total)
    maximum = max(b - a for a, b in zip(bounds, bounds[1:]))
    ratio, budget, topk = self.compress_ratio, self.token_budget, self.block_topk
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
    selected = torch.full((shard, topk), -1, dtype=torch.int32, device=hidden.device)
    if not all_short:
        qk, _ = self.index_qk_proj(hidden)
        q = gemma_rmsnorm(
            qk[:, : self.n_heads * self.head_dim].reshape(total, self.n_heads, self.head_dim),
            self.q_layernorm.weight,
            self.config.layernorm_epsilon,
        )
        raw_keys = qk[:, self.n_heads * self.head_dim :].reshape(total, self.head_dim)
        if cu is None:
            q = _apply_rotary_pos_emb_bshd(
                q.unsqueeze(1), freqs, rotary_interleaved=self.config.rotary_interleaved
            ).squeeze(1)
        else:
            q = apply_rotary_pos_emb(q, freqs, config=self.config, cu_seqlens=cu, cp_group=cp, max_seqlen=maximum)
    absolute = getattr(self.config, "mrope_section", None) is not None and freqs.shape[0] == total
    for start, end in zip(bounds, bounds[1:]):
        first, last = max(start, first_query), min(end, last_query)
        if first >= last:
            continue
        blocks = (end - start) // ratio
        if end - start <= budget + ratio - 1:
            local_positions = torch.arange(first - start, last - start, device=hidden.device)
            ids = torch.arange(min(topk, blocks), device=hidden.device, dtype=torch.int32)
            selected[first - first_query : last - first_query, : ids.numel()] = torch.where(
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
        for a in range(first - start, last - start, 512):
            b = min(a + 512, last - start)
            local_positions = torch.arange(a, b, device=hidden.device)
            if os.environ.get("RELAX_QWEN38_QSA_SELECTOR_COMPILE", "0") == "1":
                previous = torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
                try:
                    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
                    ids = compiled_score_topk(q[start + a : start + b], pooled, local_positions, topk)
                finally:
                    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = previous
            else:
                scores = index_scores(q[start + a : start + b], pooled)
                logits = scores.relu().sum(-1) / self.head_dim**0.5
                visible = torch.arange(blocks, device=hidden.device)[None, :] < (local_positions[:, None] + 1) // ratio
                logits.masked_fill_(~visible, -torch.inf)
                values, ids = logits.topk(min(topk, blocks), dim=-1)
                ids.masked_fill_(values == -torch.inf, -1)
            selected[start + a - first_query : start + b - first_query, : ids.shape[1]] = ids.to(torch.int32)
    if size > 1:
        combined = selected.new_empty((total, topk))
        dist.all_gather_into_tensor(combined, selected.contiguous(), group=tp)
        selected = combined
    selection = QSASelection(
        doc_ids=docs.to(torch.int32).unsqueeze(0),
        positions=positions.to(torch.int32).unsqueeze(0),
        selected_bits=None,
        bits_per_row=0,
        bits_per_row_t=torch.zeros((), dtype=torch.int64, device=hidden.device),
        compress_ratio=ratio,
        all_selected=False,
        selected_ids=selected,
        max_complete_blocks=maximum // ratio,
    )
    from relax.models.qwen3_8.ops.qsa_mixed_ops.selection import bind_document_bounds

    return bind_document_bounds(selection, bounds)
