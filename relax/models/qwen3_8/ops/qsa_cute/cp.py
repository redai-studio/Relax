# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Local-query QSA CP on the pinned Bridge's per-document zigzag layout."""

from __future__ import annotations

from typing import Any

import torch
import torch.distributed as dist
from torch import Tensor

from relax.models.qwen3_8.ops.qsa_cute.collectives import gather_sequence
from relax.models.qwen3_8.ops.qsa_cute.layout import validate_bounds
from relax.models.qwen3_8.ops.qsa_cute.profiling import profile_range
from relax.models.qwen3_8.ops.qsa_cute.selector import select_ids


def selector_query_plan(
    bounds: tuple[int, ...], cp_size: int, tp_size: int, token_budget: int
) -> tuple[list[list[tuple[int, int, int, int]]], list[int]]:
    """Assign complete original 512-row score tiles across TP replicas.

    Entries hold CP-local start/end, TP owner and its packed output offset.
    Cycling owners across documents also balances documents with one tile.
    """
    if cp_size < 1 or tp_size < 1:
        raise ValueError("Selector partition sizes must be positive")
    validate_bounds(bounds, bounds[-1], cp_size)
    plans, counts = [], [0] * tp_size
    offset, tile = 0, 0
    for start, end in zip(bounds, bounds[1:]):
        rows = (end - start) // cp_size
        tasks = []
        if end - start > token_budget + 3:
            for first in range(0, rows, 512):
                last = min(first + 512, rows)
                owner = tile % tp_size
                tasks.append((offset + first, offset + last, owner, counts[owner]))
                counts[owner] += last - first
                tile += 1
        plans.append(tasks)
        offset += rows
    return plans, counts


def restore_selected_shards(selected: Tensor, gathered: Tensor, plans: list[list[tuple[int, int, int, int]]]) -> None:
    """Restore disjoint long-document score tiles, retaining short-path IDs."""
    for tasks in plans:
        for first, last, owner, source in tasks:
            selected[first:last] = gathered[owner, source : source + last - first]


def rotary_rows(freqs: Tensor, query_positions: Tensor, document_positions: Tensor, *, absolute: bool) -> Tensor:
    """Select from a global physical mRoPE table or a document-relative text
    table."""
    return freqs.index_select(0, query_positions if absolute else document_positions)


def producer_bounds(packed: Any, total: int, cp_size: int) -> tuple[int, ...]:
    from relax.backends.megatron.qsa_layout_metadata import valid_physical_layout

    if packed is None:
        bounds = (0, total)
    else:
        if getattr(packed, "local_cp_size", None) is not None:
            raise NotImplementedError("CuTe QSA currently supports static CP groups only")
        if packed.qkv_format != "thd" or getattr(packed, "cp_partition_mode", "zigzag") != "zigzag":
            raise ValueError("CuTe QSA CP requires the producer's per-document zigzag THD layout")
        record = valid_physical_layout(packed)
        if getattr(packed, "qsa_physical_layout_cpu", None) is not None and record is None:
            raise ValueError("CuTe QSA CP received a stale producer layout binding")
        bounds = record[2] if record is not None else getattr(packed, "qsa_select_offsets_cpu", None)
    validate_bounds(bounds, total, cp_size)
    return bounds


@torch.no_grad()
def forward(
    self: Any,
    hidden_states: Tensor,
    rotary_pos_emb: Any,
    packed_seq_params: Any = None,
    output_format: str = "ids",
) -> Any:
    from megatron.core.models.common.embeddings.rope_utils import _apply_rotary_pos_emb_bshd
    from megatron.core.tensor_parallel.mappings import gather_from_sequence_parallel_region

    from relax.models.qwen3_8.ops.qsa_cute.triton_qsa_select import qsa_cp_layout_static
    from relax.models.qwen3_8.ops.qsa_mixed_ops.selection import MixedQSASelection
    from relax.utils.env import Envs

    selector_backend = Envs.RELAX_QWEN38_QSA_SELECTOR_BACKEND
    if selector_backend not in ("torch", "radix"):
        raise ValueError("QSA selector backend must be torch or radix")

    if (
        output_format != "ids"
        or self.config.qsa_indexer_loss_coeff != 0
        or any(parameter.requires_grad for parameter in self.parameters())
    ):
        raise NotImplementedError("CuTe QSA CP currently requires a frozen Stage-1 selected-ID indexer")
    if self.compress_ratio != 4 or self.kv_heads != 1 or self.token_budget <= 0 or self.token_budget % 4:
        raise ValueError("CuTe QSA CP requires ratio 4, one index-key head, and a positive four-token budget")
    if getattr(self.config, "attention_dropout", 0) != 0:
        raise ValueError("CuTe QSA CP does not implement attention dropout")
    cp, tp = self.pg_collection.cp, self.pg_collection.tp
    if cp is None or tp is None:
        raise ValueError("CuTe QSA CP requires explicit CP and TP groups")
    size, rank = dist.get_world_size(cp), dist.get_rank(cp)
    if size <= 1:
        raise ValueError("CuTe CP selector requires CP > 1")
    if self.config.sequence_parallel and dist.get_world_size(tp) > 1:
        hidden_states = gather_from_sequence_parallel_region(
            hidden_states, tensor_parallel_output_grad=False, group=tp
        )
    if hidden_states.ndim != 3 or hidden_states.shape[1] != 1:
        raise ValueError("CuTe CP selector supports packed THD or a single unpacked sequence")
    local_length = hidden_states.shape[0]
    total = local_length * size
    bounds = producer_bounds(packed_seq_params, total, size)
    cu = torch.tensor(bounds, dtype=torch.long, device=hidden_states.device)
    query_positions, reorder = qsa_cp_layout_static(cu, local_length, size, rank, "per_document")
    docs = torch.searchsorted(cu[1:], query_positions, right=True)
    positions = query_positions - cu[docs]
    freqs = rotary_pos_emb[0] if isinstance(rotary_pos_emb, tuple) else rotary_pos_emb
    if freqs is None or freqs.ndim != 4:
        raise ValueError("CuTe QSA CP requires the model's rotary frequencies")
    if getattr(self.config, "mrope_section", None) is not None and getattr(self.config, "apply_rope_fusion", False):
        raise ValueError("CuTe QSA CP requires --no-rope-fusion for sequence-major mRoPE frequencies")
    absolute = getattr(self.config, "mrope_section", None) is not None or packed_seq_params is None
    if absolute and freqs.shape[0] == local_length:
        freqs = gather_sequence(freqs, cp, reorder)
    if absolute and freqs.shape[0] != total:
        raise ValueError("CP-local or global mRoPE frequencies must describe the physical sequence")
    qk, _ = self.index_qk_proj(hidden_states.detach())
    q, raw_keys = qk.split((self.n_heads * self.head_dim, self.head_dim), dim=-1)
    q = self.q_layernorm(q.reshape(-1, self.head_dim)).view(local_length, 1, self.n_heads, self.head_dim)
    qfreq = rotary_rows(freqs, query_positions, positions, absolute=absolute)
    q = _apply_rotary_pos_emb_bshd(q, qfreq, rotary_interleaved=self.rotary_interleaved)[:, 0]
    with profile_range("selector/raw_key_gather"):
        global_keys = gather_sequence(raw_keys[:, 0], cp, reorder)
    selected = torch.full((local_length, self.block_topk), -1, dtype=torch.int32, device=q.device)
    shard_queries = Envs.RELAX_QWEN38_QSA_CP_TP_SELECTOR and dist.get_world_size(tp) > 1
    if shard_queries and selector_backend != "torch":
        raise ValueError("QSA CP TP query partition currently requires the torch selector")
    if shard_queries:
        plans, counts = selector_query_plan(bounds, size, dist.get_world_size(tp), self.token_budget)
        tp_rank, padded_rows = dist.get_rank(tp), max(counts)
        shard_ids = selected.new_full((padded_rows, self.block_topk), -1)
    # CPU segment lengths make the per-document slices static without reading GPU
    # offsets or synchronizing to discover this rank's document membership.
    offset = 0
    for document, (start, end) in enumerate(zip(bounds, bounds[1:])):
        rows = (end - start) // size
        if rows == 0:
            continue
        block_count = (end - start) // 4
        local_positions = positions[offset : offset + rows]
        if end - start <= self.token_budget + 3:
            block_ids = torch.arange(min(block_count, self.block_topk), device=q.device)
            selected[offset : offset + rows, : block_ids.numel()] = torch.where(
                block_ids[None, :] < (local_positions[:, None] + 1) // 4, block_ids, -1
            ).int()
        elif block_count:
            pooled = global_keys[start : start + 4 * block_count].float().view(block_count, 4, self.head_dim).mean(1)
            pooled = self.k_layernorm(pooled.to(global_keys.dtype))
            block_pos = 4 * torch.arange(block_count, device=q.device)
            kfreq = rotary_rows(freqs, block_pos + start, block_pos, absolute=absolute)
            pooled = _apply_rotary_pos_emb_bshd(
                pooled[:, None, None, :], kfreq, rotary_interleaved=self.rotary_interleaved
            )[:, 0, 0]
            with profile_range("selector/" + selector_backend):
                if shard_queries:
                    for first, last, owner, target in plans[document]:
                        if owner == tp_rank:
                            shard_ids[target : target + last - first] = select_ids(
                                q[first:last], pooled, positions[first:last], self.block_topk, impl="torch"
                            )
                else:
                    selected[offset : offset + rows] = select_ids(
                        q[offset : offset + rows],
                        pooled,
                        local_positions,
                        self.block_topk,
                        impl="torch" if selector_backend == "torch" else "auto",
                    )
        offset += rows
    if shard_queries and padded_rows:
        with profile_range("selector/tp_ids_gather"):
            gathered = selected.new_empty((dist.get_world_size(tp) * padded_rows, self.block_topk))
            dist.all_gather_into_tensor(gathered, shard_ids, group=tp)
            restore_selected_shards(selected, gathered.view(dist.get_world_size(tp), padded_rows, -1), plans)
    return MixedQSASelection(
        doc_ids=docs.int()[None],
        positions=positions.int()[None],
        selected_bits=None,
        bits_per_row=0,
        bits_per_row_t=torch.zeros((), dtype=torch.long, device=q.device),
        compress_ratio=4,
        all_selected=False,
        selected_ids=selected,
        max_complete_blocks=max(b - a for a, b in zip(bounds, bounds[1:])) // 4,
        _relax_qsa_mixed_bounds=bounds,
        _relax_qsa_query_positions=query_positions,
        _relax_qsa_key_reorder=reorder,
    )


def core_forward(self: Any, query: Tensor, key: Tensor, value: Tensor, selection: Any, packed: Any) -> Tensor:
    from relax.models.qwen3_8.ops.qsa_cute.adapter import sparse_attention

    group = self.pg_collection.cp
    if group is None or dist.get_world_size(group) <= 1:
        raise ValueError("CuTe QSA local selection requires its original CP group")
    is_thd = packed is not None and packed.qkv_format == "thd"
    q, k, v = [tensor.unsqueeze(0) if is_thd else tensor.permute(1, 0, 2, 3) for tensor in (query, key, value)]
    if q.shape[0] != 1 or selection._relax_qsa_query_positions.shape != (q.shape[1],):
        raise ValueError("CuTe QSA CP query tensor does not match its selection")
    with profile_range("attention/kv_gather"):
        kv = gather_sequence(torch.cat((k[0], v[0]), dim=-2), group, selection._relax_qsa_key_reorder)
    k, v = kv.split(k.shape[-2], dim=-2)
    output = sparse_attention(
        q,
        k[None],
        v[None],
        selection.selected_ids[None],
        selection._relax_qsa_mixed_bounds,
        self.softmax_scale,
        selection._relax_qsa_query_positions,
    )[0]
    output = output.reshape(query.shape[0], -1)
    return output if is_thd else output[:, None]
