# Copyright (c) 2026 Relax Authors. All Rights Reserved.
# Modified for Relax integration; see NOTICE.

"""Unpacked V4.1 attention; retained as the numerical baseline for THD."""

from typing import Any

import einops
import torch
from megatron.core.tensor_parallel.mappings import (
    copy_to_tensor_model_parallel_region,
    gather_from_sequence_parallel_region,
    scatter_to_sequence_parallel_region,
)

from ..ops.position import all_gather_cp, apply_rotary_emb, attention_freqs, cp_freqs, cp_positions, window_indices
from ..ops.quant import compressed_kv_stored, fake_quant_compressed_kv, kv_norm_rope_fp8
from ..ops.sparse_attention import sparse_attention


def forward_attention(self: Any, hidden_states: torch.Tensor) -> tuple[torch.Tensor, None]:
    if self.sequence_parallel:
        hidden_states = gather_from_sequence_parallel_region(
            hidden_states, tensor_parallel_output_grad=False, group=self.tp_group
        )
    x = einops.rearrange(hidden_states, "s b d -> b s d")
    bsz, seqlen, _ = x.shape
    rd = self.rope_head_dim
    ratio = self.compress_ratio
    rt = self.runtime
    cp = self.cp_size
    seqlen_global = seqlen * cp
    if cp > 1 and ratio:
        assert seqlen % ratio == 0, f"context-parallel chunk {seqlen} must be a multiple of the compress ratio {ratio}"

    freqs_full = attention_freqs(self.config, seqlen_global, bool(ratio), x.device)
    freqs_cis = cp_freqs(freqs_full, seqlen, self.cp_group)

    qr = self.q_layernorm(self.linear_q_down_proj(x)[0])
    q = self.linear_q_up_proj(qr)[0].unflatten(-1, (self.n_local_heads, self.head_dim)).clone()
    apply_rotary_emb(q[..., -rd:], freqs_cis)

    kv = kv_norm_rope_fp8(self.linear_kv_proj(x)[0], self.kv_layernorm.weight, self.eps, freqs_cis, rd)

    q_positions = cp_positions(seqlen, self.cp_group, x.device)
    topk_idxs = window_indices(q_positions, self.window_size, seqlen_global, bsz)
    if cp > 1:
        kv = all_gather_cp(kv, dim=1, cp_group=self.cp_group)

    if ratio:
        offset = seqlen_global
        compress_lens = (q_positions + 1) // ratio
        if self.is_kv_source:
            latent = self.core_attention.compressor(hidden_states)
            latent = einops.rearrange(latent, "n b d -> b n d")
            n_groups = latent.size(1)
            freqs_compress = cp_freqs(freqs_full, seqlen, self.cp_group, stride=ratio)[:n_groups]
            if self.is_index_source:
                index_k = self.core_attention.indexer.index_keys(latent, freqs_compress)
                if cp > 1:
                    index_k = all_gather_cp(index_k, dim=1, cp_group=self.cp_group)
                rt.index_k[self.layer_id] = index_k
            latent = latent.clone(memory_format=torch.contiguous_format)
            apply_rotary_emb(latent[..., -rd:], freqs_compress)
            latent = fake_quant_compressed_kv(latent)
            latent = compressed_kv_stored(latent, rd)
            if cp > 1:
                latent = all_gather_cp(latent, dim=1, cp_group=self.cp_group)
            rt.latent[self.layer_id] = latent
        latent = rt.latent[self.kv_source_layer]
        if self.is_index_source:
            topk_rel, candidates = self.core_attention.indexer(
                x, qr, rt.index_k[self.kv_source_layer], freqs_cis, compress_lens, rt.candidates
            )
            rt.topk = topk_rel
            if candidates is not None:
                rt.candidates = candidates
        topk_rel = rt.topk
        assert topk_rel is not None, f"layer {self.layer_id}: no index_source layer ran before it"
        compress_idxs = torch.where(topk_rel < 0, -1, topk_rel + offset).int()
        topk_idxs = torch.cat([topk_idxs, compress_idxs], dim=-1)
        kv = torch.cat([kv, latent], dim=1)

    topk_idxs = topk_idxs.int()
    kv = copy_to_tensor_model_parallel_region(kv, group=self.tp_group, all_reduce_grad_fp32=True)
    o = sparse_attention(
        q,
        kv.unsqueeze(2),
        topk_idxs.unsqueeze(2),
        self.softmax_scale,
        attn_sink=self.core_attention.attn_sink,
    )
    o = torch.cat((o[..., :-rd], apply_rotary_emb(o[..., -rd:].clone(), freqs_cis, inverse=True)), dim=-1)

    o = o.view(bsz, seqlen, self.n_local_groups, -1)
    wo_a = self.linear_o_group_proj.view(self.n_local_groups, self.o_lora_rank, -1)
    o = torch.einsum("bsgd,grd->bsgr", o, wo_a)
    out, _ = self.linear_proj(o.flatten(2))
    output = einops.rearrange(out, "b s d -> s b d")
    if self.sequence_parallel:
        output = scatter_to_sequence_parallel_region(output, group=self.tp_group)
    return output, None
