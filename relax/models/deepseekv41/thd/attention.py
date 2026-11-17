# Copyright (c) 2026 Relax Authors. All Rights Reserved.
# Modified for Relax integration; see NOTICE.

"""Packed V4.1 execution with sequence-local retrieval and CP group ownership.

The layout/ownership rules follow Miles #4488 (faecb81a) and Core #8018
(28aa0d87). Projections, quantization and sparse attention retain Relax's
BSHD implementation. Only retrieval is split by segment; attention executes
once over the packed queries and the shared, sequence-major KV stream.
"""

from typing import Any

import torch
import torch.nn.functional as F
from megatron.core.tensor_parallel.mappings import (
    copy_to_tensor_model_parallel_region,
    gather_from_sequence_parallel_region,
    scatter_to_sequence_parallel_region,
)

from ..ops.position import all_gather_cp, apply_rotary_emb, attention_freqs
from ..ops.quant import compressed_kv_stored, fake_quant_compressed_kv, kv_norm_rope_fp8
from ..ops.sparse_attention import sparse_attention
from .layout import ThdLayout


def _owned_groups(layout: ThdLayout, ratio: int, rank: int) -> tuple[tuple[int, int, int], ...]:
    """(segment, first group, group count), owned by the group's last token."""
    begin, end = rank * layout.local_length, (rank + 1) * layout.local_length
    groups = []
    for seq, (start, length) in enumerate(zip(layout.seq_starts, layout.seq_lens)):
        first = max(0, (begin - start) // ratio)
        stop = min(length // ratio, max(0, (end - start) // ratio))
        if stop > first:
            groups.append((seq, first, stop - first))
    return tuple(groups)


def _compressed_inputs(
    hidden: torch.Tensor, layout: ThdLayout, ratio: int, cp_group: Any
) -> tuple[torch.Tensor, torch.Tensor, tuple[int, ...], int]:
    """Batch complete groups, including the single left halo needed for ratio 2.

    Every rank executes the same differentiable collective, even when it owns
    no groups. Dummy groups keep kernels nonempty; slicing them out after the
    gather retains the zero-gradient path to both collectives on padding ranks.
    """
    cp_size = cp_group.size() if cp_group is not None else 1
    cp_rank = torch.distributed.get_rank(cp_group) if cp_group is not None else 0
    owners = tuple(_owned_groups(layout, ratio, rank) for rank in range(cp_size))
    counts = tuple(sum(count for _, _, count in groups) for groups in owners)
    capacity = max(1, max(counts))
    halo = ratio - 1 if cp_size > 1 else 0
    if halo:
        boundaries = all_gather_cp(hidden[-halo:].contiguous(), dim=0, cp_group=cp_group)
        # Rank zero must retain its all-gather backward even though it has no left neighbour.
        left = boundaries[(cp_rank - 1) * halo : cp_rank * halo] if cp_rank else boundaries[:halo] * 0
        source = torch.cat((left, hidden), dim=0)
    else:
        source = hidden
    local_groups = owners[cp_rank]
    positions = [
        torch.arange(first, first + count, device=hidden.device, dtype=torch.long) * ratio
        for _, first, count in local_groups
    ]
    starts = [position + layout.seq_starts[seq] for position, (seq, _, _) in zip(positions, local_groups)]
    if starts:
        rows = torch.cat(starts) - layout.global_start + halo
        rotary_positions = torch.cat(positions)
    else:
        rows = torch.empty(0, device=hidden.device, dtype=torch.long)
        rotary_positions = rows
    count = counts[cp_rank]
    rows = F.pad(rows, (0, capacity - count))
    rotary_positions = F.pad(rotary_positions, (0, capacity - count))
    indices = (rows[:, None] + torch.arange(ratio, device=hidden.device)).flatten()
    # A dummy ratio-2 group may be longer than a one-token local shard.
    indices = indices.clamp_max(source.shape[0] - 1)
    valid = (torch.arange(capacity, device=hidden.device) < count).repeat_interleave(ratio)
    compact = source.index_select(0, indices) * valid[:, None, None]
    return compact, rotary_positions, counts, capacity


def _gather_compressed(value: torch.Tensor, counts: tuple[int, ...], capacity: int, cp_group: Any) -> torch.Tensor:
    if len(counts) > 1:
        value = all_gather_cp(value, dim=1, cp_group=cp_group)
    # Contiguous CP ownership is monotonic in physical sequence positions, so
    # removing each rank's dummy suffix directly recovers sequence-major order.
    return torch.cat([value[:, rank * capacity : rank * capacity + count] for rank, count in enumerate(counts)], dim=1)


def _window_indices(layout: ThdLayout, positions: torch.Tensor, valid: torch.Tensor, window: int) -> torch.Tensor:
    physical = torch.arange(layout.local_length, device=positions.device) + layout.global_start
    start = torch.maximum(physical - window + 1, physical - positions)
    keys = start[:, None] + torch.arange(min(window, layout.total_tokens), device=positions.device)
    return torch.where(valid[:, None] & (keys <= physical[:, None]), keys, -1).int().unsqueeze(0)


@torch.no_grad()
def _retrieve(
    indexer: Any,
    x: torch.Tensor,
    qr: torch.Tensor,
    keys: torch.Tensor,
    freqs: torch.Tensor,
    layout: ThdLayout,
    ratio: int,
    candidates: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Select within each segment, then convert top-k to canonical packed rows."""
    topk = torch.full((1, layout.local_length, indexer.topk), -1, device=x.device, dtype=torch.int64)
    width = layout.candidate_width(ratio)
    if indexer.candidate_source:
        candidates = torch.zeros((1, layout.local_length, width), device=x.device, dtype=torch.bool)
    elif indexer.uses_candidates and candidates is None:
        raise RuntimeError("DeepSeek V4.1 candidate source must run before this indexer.")
    q, weights = indexer.query_features(x, qr, freqs)
    offsets = layout.compressed_offsets(ratio)
    for span in layout.spans():
        first, last = offsets[span.seq_index : span.seq_index + 2]
        if first == last:
            continue
        begin, end = span.local_start, span.local_end
        lengths = (torch.arange(end - begin, device=x.device) + span.position_start + 1) // ratio
        segment_candidates = None
        if indexer.uses_candidates:
            # Candidate columns are relative to each sample, independent of its
            # offset in the shared packed compressed-key stream.
            segment_candidates = candidates[:, begin:end, : last - first]
        indices, selected = indexer.select(
            q[:, begin:end], keys[:, first:last], weights[:, begin:end], lengths, segment_candidates
        )
        topk[:, begin:end] = torch.where(indices >= 0, indices + first, -1)
        if indexer.candidate_source:
            candidates[:, begin:end, : last - first] = selected
    return topk, candidates


def forward_attention(self: Any, hidden_states: torch.Tensor, packed_seq_params: Any) -> tuple[torch.Tensor, None]:
    if self.sequence_parallel:
        hidden_states = gather_from_sequence_parallel_region(
            hidden_states, tensor_parallel_output_grad=False, group=self.tp_group
        )
    if hidden_states.shape[1] != 1:
        raise ValueError("DeepSeek V4.1 THD attention requires a flattened batch dimension of one.")
    layout = self.runtime.thd_layout
    if layout is None:
        cp_rank = torch.distributed.get_rank(self.cp_group) if self.cp_group is not None else 0
        layout = ThdLayout.from_packed_seq_params(
            packed_seq_params, cp_rank=cp_rank, cp_size=self.cp_size, seqlen_local=hidden_states.shape[0]
        )
    if layout.local_length != hidden_states.shape[0]:
        raise ValueError("DeepSeek V4.1 THD layout does not match the gathered local sequence length.")
    x = hidden_states.transpose(0, 1)
    positions, valid = layout.positions(x.device), layout.valid_mask(x.device)
    ratio, rd, rt = self.compress_ratio, self.rope_head_dim, self.runtime
    # Nearby packed lengths share a table; configured YaRN frequencies do not
    # depend on this allocation length, so valid positions remain unchanged.
    frequency_capacity = 1 << (max(layout.seq_lens, default=1) - 1).bit_length()
    freqs_full = attention_freqs(self.config, frequency_capacity, bool(ratio), x.device)
    freqs = freqs_full.index_select(0, positions)
    qr = self.q_layernorm(self.linear_q_down_proj(x)[0])
    q = self.linear_q_up_proj(qr)[0].unflatten(-1, (self.n_local_heads, self.head_dim)).clone()
    apply_rotary_emb(q[..., -rd:], freqs)
    kv = kv_norm_rope_fp8(self.linear_kv_proj(x)[0], self.kv_layernorm.weight, self.eps, freqs, rd)
    topk = _window_indices(layout, positions, valid, self.window_size)
    if self.cp_size > 1:
        kv = all_gather_cp(kv, dim=1, cp_group=self.cp_group)
    if ratio:
        if self.is_kv_source:
            compact, compressed_positions, counts, capacity = _compressed_inputs(
                hidden_states, layout, ratio, self.cp_group
            )
            latent = self.core_attention.compressor(compact).transpose(0, 1)
            compressed_freqs = freqs_full.index_select(0, compressed_positions)
            if self.is_index_source:
                index_k = self.core_attention.indexer.index_keys(latent, compressed_freqs)
                rt.index_k[self.layer_id] = _gather_compressed(index_k, counts, capacity, self.cp_group)
            latent = latent.clone(memory_format=torch.contiguous_format)
            apply_rotary_emb(latent[..., -rd:], compressed_freqs)
            latent = compressed_kv_stored(fake_quant_compressed_kv(latent), rd)
            rt.latent[self.layer_id] = _gather_compressed(latent, counts, capacity, self.cp_group)
        if self.is_index_source:
            rt.topk, rt.candidates = _retrieve(
                self.core_attention.indexer,
                x,
                qr,
                rt.index_k[self.kv_source_layer],
                freqs,
                layout,
                ratio,
                rt.candidates,
            )
        if rt.topk is None:
            raise RuntimeError(f"DeepSeek V4.1 layer {self.layer_id} has no preceding index source.")
        compressed_indices = torch.where(rt.topk >= 0, rt.topk + layout.total_tokens, -1).int()
        topk = F.pad(topk, (0, compressed_indices.shape[-1]), value=-1)
        for span in layout.spans():
            # Match each sample's BSHD sparse reduction blocks. Putting every
            # compressed list after 128 window slots moves short samples' keys
            # between online-softmax blocks; BF16 probability rounding then
            # changes QAT values and later discrete retrieval decisions.
            column = min(self.window_size, span.seq_length)
            topk[:, span.local_start : span.local_end, column : column + compressed_indices.shape[-1]] = (
                compressed_indices[:, span.local_start : span.local_end]
            )
        kv = torch.cat((kv, rt.latent[self.kv_source_layer]), dim=1)
    kv = copy_to_tensor_model_parallel_region(kv, group=self.tp_group, all_reduce_grad_fp32=True)
    output = sparse_attention(
        q, kv.unsqueeze(2), topk.unsqueeze(2), self.softmax_scale, attn_sink=self.core_attention.attn_sink
    )
    output = torch.cat((output[..., :-rd], apply_rotary_emb(output[..., -rd:].clone(), freqs, inverse=True)), dim=-1)
    output = output.view(1, layout.local_length, self.n_local_groups, -1)
    weights = self.linear_o_group_proj.view(self.n_local_groups, self.o_lora_rank, -1)
    output = torch.einsum("bsgd,grd->bsgr", output, weights)
    output, _ = self.linear_proj(output.flatten(2))
    output = output.transpose(0, 1) * valid[:, None, None]
    if self.sequence_parallel:
        output = scatter_to_sequence_parallel_region(output, group=self.tp_group)
    return output, None
