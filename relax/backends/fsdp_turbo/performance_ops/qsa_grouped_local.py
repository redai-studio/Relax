# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Grouped QSA for the unpadded MBS1, CP1 SFT pipeline."""

import torch
import torch.nn.functional as F


def qsa_grouped_local_forward(
    self, hidden_states, position_embeddings, attention_mask, past_key_values=None, **kwargs
):
    from transformers.models.qwen4_exp.modeling_qwen4_exp import apply_rotary_pos_emb

    from .qsa_grouped_ascendc import grouped_attention

    batch, local_length, _ = hidden_states.shape
    if batch != 1:
        raise ValueError("Grouped QSA currently requires the validated MBS=1 pipeline")
    if past_key_values is not None:
        raise ValueError("Grouped QSA CP1 supports full-sequence training without KV cache")
    size, rank = 1, 0
    padded_length = length = local_length
    if attention_mask is None or attention_mask.shape[-2:] != (length, length):
        raise ValueError("Grouped QSA CP1 requires the full causal mask")
    heads, kv_heads = self.config.num_attention_heads, self.config.num_key_value_heads
    padded_heads = ((heads + size - 1) // size) * size
    local_heads = padded_heads // size
    query, gate = self.q_proj(hidden_states).view(batch, local_length, heads, self.head_dim * 2).chunk(2, -1)
    gate = gate.reshape(batch, local_length, -1)
    query = self.q_norm(query).transpose(1, 2)
    key = self.k_norm(self.k_proj(hidden_states).view(batch, local_length, kv_heads, self.head_dim)).transpose(1, 2)
    value = self.v_proj(hidden_states).view(batch, local_length, kv_heads, self.head_dim).transpose(1, 2)
    query, key = apply_rotary_pos_emb(query, key, *position_embeddings)
    if padded_heads != heads:
        query = F.pad(query, (0, 0, 0, 0, 0, padded_heads - heads))
    with torch.no_grad():
        full_hidden = hidden_states.detach()
        full_positions = tuple(x.detach() for x in position_embeddings)
        mask = attention_mask if attention_mask.dtype == torch.bool else attention_mask >= 0
        selected = self.indexer(full_hidden, full_positions, mask, return_block_ids=True)
        budget, block = self.indexer.token_budget, self.indexer.compress_ratio
        if block != 4 or self.head_dim != 256:
            raise ValueError("Grouped QSA requires block_size=4 and head_dim=256")
        prefix = min(length, budget + block - 1)
        slots = torch.arange(budget // block + 1, dtype=torch.int32, device=hidden_states.device)
        rows = torch.arange(prefix, dtype=torch.int32, device=hidden_states.device)
        dense = torch.where(slots[None, :] <= rows[:, None] // block, slots[None, :], -1)
        blocks = torch.cat((dense, selected[0]), dim=0)
        carrier = torch.where(blocks >= 0, blocks * block, -1).contiguous()
        cu = torch.tensor([0, length], dtype=torch.int64, device=hidden_states.device)
        row_ids = torch.arange(length, dtype=torch.int32, device=hidden_states.device).view(1, -1)
    if self.training and self.attention_dropout:
        raise ValueError("Grouped QSA requires dropout=0")
    if padded_heads != heads or heads % kv_heads:
        raise ValueError("Grouped QSA requires divisible Q/KV head geometry")
    # The MBS1 input pipeline has no sequence padding; preserve grouped KV heads.
    chunks = []
    q_per_kv = heads // kv_heads
    start_head, end_head = rank * local_heads, (rank + 1) * local_heads
    for kv_head in range(kv_heads):
        lo, hi = max(start_head, kv_head * q_per_kv), min(end_head, (kv_head + 1) * q_per_kv)
        if lo >= hi:
            continue
        q = query[0, lo - start_head : hi - start_head].transpose(0, 1).contiguous()
        k = key[0, kv_head : kv_head + 1].transpose(0, 1).contiguous()
        v = value[0, kv_head : kv_head + 1].transpose(0, 1).contiguous()
        chunks.append(grouped_attention(q, k, v, carrier, row_ids, cu, cu, (0, length), (0, length), self.scaling))
    output = torch.cat(chunks, dim=1).unsqueeze(0)
    output = F.pad(output, (0, 0, 0, 0, 0, padded_length - length))
    output = output[:, :, :heads].reshape(batch, local_length, -1) * torch.sigmoid(gate)
    return self.o_proj(output), None
