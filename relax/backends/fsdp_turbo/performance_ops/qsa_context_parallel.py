# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""QSA Ulysses adapters using existing Turbo groups."""

import torch
import torch.distributed as dist
import torch.nn.functional as F


def _index_inputs(indexer, hidden, positions):
    from fsdp_turbo.models.qwen.modeling_qwen4_exp import _apply_indexer_query_rope

    batch, length, _ = hidden.shape
    width = indexer.index_n_heads * indexer.index_head_dim
    projected = indexer.index_qk_proj(hidden)
    query, raw_keys = projected.split([width, indexer.index_kv_heads * indexer.index_head_dim], -1)
    query = query.view(batch, length, indexer.index_n_heads, indexer.index_head_dim)
    query = _apply_indexer_query_rope(indexer.q_layernorm(query), *positions)
    return query, raw_keys


def _lightning_routes(indexer, hidden, positions, *, block_ids):
    from .qsa_indexer import qsa_lightning_selected_token_mask

    query, raw_keys = _index_inputs(indexer, hidden, positions)
    # ID outputs need only shape/dtype, not a dense attention mask. Validity
    # is applied by the dense consumer or rejected by the grouped consumer.
    return qsa_lightning_selected_token_mask(
        query,
        raw_keys,
        *positions,
        torch.empty(0, dtype=torch.bool, device=hidden.device),
        indexer.k_layernorm,
        token_budget=indexer.token_budget,
        compress_ratio=indexer.compress_ratio,
        return_token_ids=not block_ids,
        return_block_ids=block_ids,
    )


def _ids_chunk_mask(ids, validity, start: int, end: int):
    length = validity.shape[-1]
    selected = ids[:, start:end].long()
    selected = torch.where((selected >= 0) & (selected < length), selected, length)
    mask = torch.zeros((ids.shape[0], 1, end - start, length + 1), dtype=torch.bool, device=ids.device)
    mask.scatter_(-1, selected.unsqueeze(1), True)
    keys = torch.arange(length, device=ids.device)
    rows = torch.arange(start, end, device=ids.device)
    return mask[..., :length] & validity[:, None, None, :] & (keys[None, :] <= rows[:, None])


def qsa_forward(self, hidden_states, position_embeddings, attention_mask, *, ulysses_group, **kwargs):
    from fsdp_turbo.distributed.context_parallel.kv_allgather.utils import kv_allgather
    from fsdp_turbo.distributed.context_parallel.ulysses.ulysses_attention import _gather_sequence
    from fsdp_turbo.distributed.context_parallel.ulysses.utils import ulysses_all_to_all
    from transformers.models.qwen4_exp.modeling_qwen4_exp import apply_rotary_pos_emb

    from ..qwen4_attention import npu_attention

    batch, local_length, _ = hidden_states.shape
    size, rank = dist.get_world_size(ulysses_group), dist.get_rank(ulysses_group)
    padded_length = local_length * size
    length = kwargs["cp_sequence_length"]
    if not 0 < length <= padded_length or attention_mask.shape != (batch, padded_length):
        raise ValueError("Ulysses QSA requires original sequence length and full padded validity")
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
    query = ulysses_all_to_all(query, ulysses_group, scatter_dim=1, gather_dim=2)[:, :, :length].contiguous()
    head_map = (
        torch.arange(rank * local_heads, (rank + 1) * local_heads, device=hidden_states.device) // (heads // kv_heads)
    ).clamp(max=kv_heads - 1)
    key = kv_allgather(key, ulysses_group, gather_dim=2)[:, :, :length].index_select(1, head_map)
    value = kv_allgather(value, ulysses_group, gather_dim=2)[:, :, :length].index_select(1, head_map)
    with torch.no_grad():
        full_hidden = _gather_sequence(hidden_states.detach(), ulysses_group)[:, :length]
        full_positions = tuple(_gather_sequence(x.detach(), ulysses_group)[:, :length] for x in position_embeddings)
        validity = attention_mask[:, :length].bool()
        indexer = self.indexer
        if getattr(indexer, "qsa_triton_validate_prefix_mask", True):
            torch._assert_async(
                (~(validity[:, 1:] & ~validity[:, :-1])).all(),
                "QSA requires right-padded prefix validity",
            )
        if self._relax_lightning_indexer:
            ids = _lightning_routes(indexer, full_hidden, full_positions, block_ids=False)
        else:
            from fsdp_turbo.ops.qsa import qsa_compress_block_keys

            padding = (-length) % indexer.compress_ratio
            full_hidden = F.pad(full_hidden, (0, 0, 0, padding))
            full_positions = (
                F.pad(full_positions[0], (0, 0, 0, padding), value=1),
                F.pad(full_positions[1], (0, 0, 0, padding), value=0),
            )
            index_query, raw_keys = _index_inputs(indexer, full_hidden, full_positions)
            block_keys = qsa_compress_block_keys(
                raw_keys,
                *full_positions,
                indexer.k_layernorm.weight,
                rms_norm_eps=indexer.k_layernorm.eps,
                compress_ratio=indexer.compress_ratio,
            )
    rate = indexer.compress_ratio
    chunk_size = getattr(self.config, "qsa_cp_query_chunk_size", 128) or 128
    if not isinstance(chunk_size, int) or chunk_size <= 0:
        raise ValueError("qsa_cp_query_chunk_size must be positive")
    chunk_size = ((chunk_size + rate - 1) // rate) * rate
    dropout = self.attention_dropout if self.training else 0.0
    outputs = []
    for start in range(0, length, chunk_size):
        end = min(length, start + chunk_size)

        def attend(q, k, v, start=start, end=end):
            with torch.no_grad():
                if self._relax_lightning_indexer:
                    mask = _ids_chunk_mask(ids, validity, start, end)
                else:
                    from fsdp_turbo.ops.qsa import qsa_selected_token_mask

                    route_end = ((end + rate - 1) // rate) * rate
                    keys = torch.arange(route_end, device=q.device)
                    rows = torch.arange(start, route_end, device=q.device)
                    route_validity = F.pad(validity, (0, (-length) % rate), value=False)
                    base = (keys[None, :] <= rows[:, None])[None, None] & route_validity[:, None, None, :route_end]
                    mask = (
                        qsa_selected_token_mask(
                            index_query[:, start:route_end],
                            None,
                            None,
                            None,
                            base,
                            None,
                            rms_norm_eps=indexer.k_layernorm.eps,
                            token_budget=indexer.token_budget,
                            compress_ratio=rate,
                            block_keys=block_keys[:, : route_end // rate],
                            validate_prefix_mask=False,
                            query_chunk_size=getattr(indexer, "qsa_triton_query_chunk_size", 0) or None,
                        )
                        & base
                    )
                    mask = mask[..., : end - start, : min(route_end, length)]
                    mask = F.pad(mask, (0, length - mask.shape[-1]), value=False)
            if self._relax_cp_npu_attention:
                return npu_attention(self, q, k, v, mask, dropout=dropout, scaling=self.scaling)[0]
            return F.scaled_dot_product_attention(
                q, k, v, attn_mask=mask, dropout_p=dropout, scale=self.scaling
            ).transpose(1, 2)

        if torch.is_grad_enabled() and any(t.requires_grad for t in (query, key, value)):
            from torch.utils.checkpoint import checkpoint

            # Rebuild the bounded mask in backward instead of saving all chunks (S²).
            outputs.append(checkpoint(attend, query[:, :, start:end], key, value, use_reentrant=False))
        else:
            outputs.append(attend(query[:, :, start:end], key, value))
    output = torch.cat(outputs, dim=1)
    output = F.pad(output, (0, 0, 0, 0, 0, padded_length - length))
    output = ulysses_all_to_all(output.contiguous(), ulysses_group, scatter_dim=1, gather_dim=2)
    output = output[:, :, :heads].reshape(batch, local_length, -1) * torch.sigmoid(gate)
    return self.o_proj(output), None


def qsa_grouped_forward(self, hidden_states, position_embeddings, attention_mask, *, ulysses_group, **kwargs):
    from fsdp_turbo.distributed.context_parallel.kv_allgather.utils import kv_allgather
    from fsdp_turbo.distributed.context_parallel.ulysses.ulysses_attention import _gather_sequence
    from fsdp_turbo.distributed.context_parallel.ulysses.utils import ulysses_all_to_all
    from transformers.models.qwen4_exp.modeling_qwen4_exp import apply_rotary_pos_emb

    from .qsa_grouped_ascendc import grouped_attention

    batch, local_length, _ = hidden_states.shape
    if batch != 1:
        raise ValueError("Grouped QSA currently requires the validated MBS=1 pipeline")
    size, rank = dist.get_world_size(ulysses_group), dist.get_rank(ulysses_group)
    padded_length = local_length * size
    length = kwargs["cp_sequence_length"]
    if not 0 < length <= padded_length or attention_mask.shape != (batch, padded_length):
        raise ValueError("Ulysses QSA requires original sequence length and full padded validity")
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
    query = ulysses_all_to_all(query, ulysses_group, scatter_dim=1, gather_dim=2)[:, :, :length].contiguous()
    key = kv_allgather(key, ulysses_group, gather_dim=2)[:, :, :length]
    value = kv_allgather(value, ulysses_group, gather_dim=2)[:, :, :length]
    with torch.no_grad():
        full_hidden = _gather_sequence(hidden_states.detach(), ulysses_group)[:, :length]
        full_positions = tuple(_gather_sequence(x.detach(), ulysses_group)[:, :length] for x in position_embeddings)
        validity = attention_mask[:, :length].bool()
        torch._assert_async(validity.all(), "Grouped QSA does not support sample padding")
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
    if padded_heads != heads or heads % kv_heads:
        raise ValueError("Grouped QSA requires divisible Q/KV head geometry")
    # Keep the baseline Ulysses/all-gather autograd, without materializing repeated KV heads.
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
    output = ulysses_all_to_all(output.contiguous(), ulysses_group, scatter_dim=1, gather_dim=2)
    output = output[:, :, :heads].reshape(batch, local_length, -1) * torch.sigmoid(gate)
    return self.o_proj(output), None
