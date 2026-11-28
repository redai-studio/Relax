# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Grouped QSA for the unpadded MBS1, CP1 SFT pipeline."""

import torch


def validate_grouped_mask(mask, length):
    key = None if torch.is_inference(mask) else (mask._version, length)
    if key is not None and getattr(mask, "_relax_grouped_validation", None) == key:
        return
    allowed = mask if mask.dtype == torch.bool else mask >= 0
    keys = torch.arange(length, device=mask.device)
    valid = torch.ones((), dtype=torch.bool, device=mask.device)
    for start in range(0, length, 128):
        rows = torch.arange(start, min(start + 128, length), device=mask.device)
        expected = keys[None, :] <= rows[:, None]
        valid = valid & (allowed[..., start : start + 128, :] == expected).all()
    # NPU _assert_async is a CPU fallback. Check once per distinct input,
    # never once per query chunk. Mutation invalidates the cached result.
    if not bool(valid):
        raise ValueError("Grouped QSA requires an unpadded causal mask")
    mask._relax_grouped_validation = key


def qsa_grouped_local_forward(
    self, hidden_states, position_embeddings, attention_mask, past_key_values=None, **kwargs
):
    from transformers.models.qwen4_exp.modeling_qwen4_exp import apply_rotary_pos_emb

    from .qsa_context_parallel import _lightning_routes
    from .qsa_grouped_ascendc import grouped_attention

    batch, local_length, _ = hidden_states.shape
    if batch != 1:
        raise ValueError("Grouped QSA currently requires the validated MBS=1 pipeline")
    if past_key_values is not None:
        raise ValueError("Grouped QSA CP1 supports full-sequence training without KV cache")
    length = local_length
    if attention_mask is None or attention_mask.shape[-2:] != (length, length):
        raise ValueError("Grouped QSA CP1 requires the full causal mask")
    validate_grouped_mask(attention_mask, length)
    heads, kv_heads = self.config.num_attention_heads, self.config.num_key_value_heads
    query, gate = self.q_proj(hidden_states).view(batch, local_length, heads, self.head_dim * 2).chunk(2, -1)
    gate = gate.reshape(batch, local_length, -1)
    query = self.q_norm(query).transpose(1, 2)
    key = self.k_norm(self.k_proj(hidden_states).view(batch, local_length, kv_heads, self.head_dim)).transpose(1, 2)
    value = self.v_proj(hidden_states).view(batch, local_length, kv_heads, self.head_dim).transpose(1, 2)
    query, key = apply_rotary_pos_emb(query, key, *position_embeddings)
    with torch.no_grad():
        full_hidden = hidden_states.detach()
        full_positions = tuple(x.detach() for x in position_embeddings)
        selected = _lightning_routes(self.indexer, full_hidden, full_positions, block_ids=True)
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
    if heads % kv_heads:
        raise ValueError("Grouped QSA requires divisible Q/KV head geometry")
    output = grouped_attention(
        query[0].transpose(0, 1).contiguous(),
        key[0].transpose(0, 1).contiguous(),
        value[0].transpose(0, 1).contiguous(),
        carrier,
        row_ids,
        cu,
        cu,
        (0, length),
        (0, length),
        self.scaling,
    ).unsqueeze(0)
    output = output.reshape(batch, local_length, -1) * torch.sigmoid(gate)
    return self.o_proj(output), None
