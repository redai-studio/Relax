# Copyright (c) 2026 Relax Authors. All Rights Reserved.
# Adapted from MindSpeed-MM; full third-party license notices are in __init__.py.
"""CANN LightningIndexerV2 backend for the frozen Qwen4-Exp QSA router."""

from __future__ import annotations

from collections.abc import Callable

import torch
from cann_ops_transformer.ops import lightning_indexer, lightning_indexer_metadata


IS_NPU_AVAILABLE = hasattr(torch, "npu")

_LIGHTNING_OPS = None
_METADATA_CACHE: dict[tuple, tuple[torch.Tensor, torch.Tensor]] = {}
_WEIGHT_CACHE: dict[tuple, torch.Tensor] = {}


def rotate_half(x):
    first, second = x.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


# Adapted from transformers.models.glm.modular_glm.apply_rotary_pos_emb
def apply_rotary_pos_emb(q, k=None, cos=None, sin=None, unsqueeze_dim=2):
    """Applies Rotary Position Embedding to the query and key tensors.

    Removes the interleaving of cos and sin from GLM

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)

    # Keep half or full tensor for later concatenation
    rotary_dim = cos.shape[-1]
    q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
    if k is not None:
        k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]

    # Apply rotary embeddings on the first half or full tensor
    if IS_NPU_AVAILABLE:
        import torch_npu

        q_embed = torch_npu.npu_rotary_mul(q_rot, cos, sin)
        if k is not None:
            k_embed = torch_npu.npu_rotary_mul(k_rot, cos, sin)
    else:
        q_embed = (q_rot * cos) + (rotate_half(q_rot) * sin)
        if k is not None:
            k_embed = (k_rot * cos) + (rotate_half(k_rot) * sin)

    # Concatenate back to full shape
    q_embed = torch.cat([q_embed, q_pass], dim=-1)
    if k is None:
        return q_embed
    k_embed = torch.cat([k_embed, k_pass], dim=-1)
    return q_embed, k_embed


def _constant_head_weights(
    batch: int,
    rows: int,
    heads: int,
    head_dim: int,
    device: torch.device,
) -> torch.Tensor:
    key = (str(device), batch, rows, heads, head_dim)
    weights = _WEIGHT_CACHE.get(key)
    if weights is None:
        weights = torch.full(
            (batch, rows, heads),
            head_dim**-0.5,
            dtype=torch.float32,
            device=device,
        )
        _WEIGHT_CACHE[key] = weights
    return weights


def _lightning_metadata(
    *,
    batch: int,
    query_rows: int,
    key_blocks: int,
    query_heads: int,
    head_dim: int,
    block_topk: int,
    compress_ratio: int,
    key_residual: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    cache_key = (
        str(device),
        batch,
        query_rows,
        key_blocks,
        query_heads,
        head_dim,
        block_topk,
        compress_ratio,
        key_residual,
    )
    cached = _METADATA_CACHE.get(cache_key)
    if cached is not None:
        return cached

    residual = torch.full((batch,), key_residual, dtype=torch.int32, device=device)
    metadata = lightning_indexer_metadata(
        num_heads_q=query_heads,
        num_heads_k=1,
        head_dim=head_dim,
        topk=block_topk,
        cmp_residual_k=residual,
        batch_size=batch,
        max_seqlen_q=query_rows,
        max_seqlen_k=key_blocks,
        layout_q="BSND",
        layout_k="BSND",
        mask_mode=3,
        cmp_ratio=compress_ratio,
    )
    cached = (metadata, residual)
    _METADATA_CACHE[cache_key] = cached
    return cached


def _compressed_index_keys(
    raw_keys: torch.Tensor,
    full_cos: torch.Tensor,
    full_sin: torch.Tensor,
    k_norm: Callable[[torch.Tensor], torch.Tensor],
    compress_ratio: int,
) -> torch.Tensor:
    batch, kv_length, head_dim = raw_keys.shape
    blocks = kv_length // compress_ratio
    grouped = raw_keys[:, : blocks * compress_ratio].reshape(batch, blocks, compress_ratio, head_dim)
    # Preserve the reference's FP32 reduction followed by a cast before RMSNorm.
    pooled = grouped.float().mean(dim=2).to(raw_keys.dtype)
    normalized = k_norm(pooled)
    block_cos = full_cos[:, : blocks * compress_ratio : compress_ratio]
    block_sin = full_sin[:, : blocks * compress_ratio : compress_ratio]
    return apply_rotary_pos_emb(normalized.unsqueeze(2), cos=block_cos, sin=block_sin).squeeze(2)


@torch.no_grad()
def qsa_lightning_selected_token_mask(
    index_queries: torch.Tensor,
    raw_keys: torch.Tensor,
    full_cos: torch.Tensor,
    full_sin: torch.Tensor,
    attention_mask: torch.Tensor,
    k_norm: Callable[[torch.Tensor], torch.Tensor],
    *,
    rms_norm_eps: float,
    token_budget: int,
    compress_ratio: int,
    return_token_ids: bool = False,
    return_block_ids: bool = False,
    fallback_to_triton: bool = True,
    id2mask_implementation: str = "eager",
) -> torch.Tensor:
    """Run QSA routing with LightningIndexerV2 and preserve Qwen tail
    semantics.

    The CANN operator selects complete compressed blocks. Qwen additionally
    keeps the current incomplete causal block, which is appended explicitly.
    When ``return_block_ids`` is true, dense-prefix rows are omitted; the
    attention adapter restores them with ``complete_block_ids``.
    """
    if return_token_ids and return_block_ids:
        raise ValueError("return_token_ids and return_block_ids are mutually exclusive")
    if id2mask_implementation not in ("eager", "triton"):
        raise ValueError("qsa_id2mask_implementation must be eager or triton")
    batch, query_length, query_heads, head_dim = index_queries.shape
    kv_length = raw_keys.shape[1]
    if (
        index_queries.device.type != "npu"
        or index_queries.dtype not in (torch.float16, torch.bfloat16)
        or raw_keys.dtype != index_queries.dtype
        or raw_keys.device != index_queries.device
        or raw_keys.shape != (batch, kv_length, head_dim)
        or head_dim != 128
        or compress_ratio not in (1, 4, 128)
        or token_budget <= 0
        or token_budget % compress_ratio
        or query_length > kv_length
        or full_cos.shape[0] not in (1, batch)
        or full_cos.shape[1] != kv_length
        or full_sin.shape != full_cos.shape
    ):
        raise ValueError("QSA shape is unsupported by CANN LightningIndexerV2")
    block_topk = token_budget // compress_ratio
    query_position_offset = kv_length - query_length
    dense_prefix_rows = min(
        query_length,
        max(0, token_budget + compress_ratio - 1 - query_position_offset),
    )

    if full_cos.shape[0] == 1 and batch != 1:
        full_cos = full_cos.expand(batch, -1, -1)
        full_sin = full_sin.expand(batch, -1, -1)

    sparse_rows = query_length - dense_prefix_rows
    if sparse_rows:
        compressed = _compressed_index_keys(raw_keys, full_cos, full_sin, k_norm, compress_ratio).contiguous()
        metadata, residual = _lightning_metadata(
            batch=batch,
            query_rows=sparse_rows,
            key_blocks=compressed.shape[1],
            query_heads=query_heads,
            head_dim=head_dim,
            block_topk=block_topk,
            compress_ratio=compress_ratio,
            key_residual=kv_length % compress_ratio,
            device=index_queries.device,
        )
        weights = _constant_head_weights(batch, sparse_rows, query_heads, head_dim, index_queries.device)
        selected_blocks, _ = lightning_indexer(
            index_queries[:, dense_prefix_rows:].contiguous(),
            compressed.unsqueeze(2),
            weights,
            block_topk,
            cmp_residual_k=residual,
            metadata=metadata,
            max_seqlen_q=-1,
            layout_q="BSND",
            layout_k="BSND",
            mask_mode=3,
            cmp_ratio=compress_ratio,
            return_value=0,
        )
        selected_blocks = selected_blocks.squeeze(2)
        if id2mask_implementation == "triton" and not (return_token_ids or return_block_ids):
            from .id2mask import lightning_ids_to_mask

            return lightning_ids_to_mask(
                selected_blocks,
                query_length=query_length,
                kv_length=kv_length,
                token_budget=token_budget,
                compress_ratio=compress_ratio,
                dtype=attention_mask.dtype,
            )
        positions = torch.arange(
            dense_prefix_rows + query_position_offset,
            query_length + query_position_offset,
            dtype=torch.int32,
            device=index_queries.device,
        )
        visible_counts = positions + 1
        tail_blocks = torch.where(
            visible_counts % compress_ratio != 0,
            visible_counts // compress_ratio,
            -1,
        )
        selected_blocks = torch.cat(
            (
                selected_blocks,
                tail_blocks.view(1, sparse_rows, 1).expand(batch, -1, -1),
            ),
            dim=-1,
        )
    else:
        if id2mask_implementation == "triton" and not (return_token_ids or return_block_ids):
            from .id2mask import lightning_ids_to_mask

            return lightning_ids_to_mask(
                torch.empty((batch, 0, block_topk), dtype=torch.int32, device=index_queries.device),
                query_length=query_length,
                kv_length=kv_length,
                token_budget=token_budget,
                compress_ratio=compress_ratio,
                dtype=attention_mask.dtype,
            )
        selected_blocks = torch.empty(
            (batch, 0, block_topk + 1),
            dtype=torch.int32,
            device=index_queries.device,
        )

    if return_block_ids:
        return selected_blocks

    output_width = token_budget + compress_ratio - 1
    # Padding slots are absorbed by the extra (kv_length) column of the mask
    # that is dropped right after the scatter. Use -1 for the token-id ABI and
    # kv_length for the mask path, so the scatter index stays in range and int32
    # (torch's scatter accepts int32 indices on NPU; the aclnn op registers both
    # INT32 and INT64).
    pad = -1 if return_token_ids else kv_length
    selected_tokens = torch.full(
        (batch, query_length, output_width),
        pad,
        dtype=torch.int32,
        device=index_queries.device,
    )
    if dense_prefix_rows:
        dense_positions = torch.arange(
            query_position_offset,
            dense_prefix_rows + query_position_offset,
            dtype=torch.int32,
            device=index_queries.device,
        )
        dense_ids = torch.arange(output_width, dtype=torch.int32, device=index_queries.device)
        dense_valid = dense_ids.view(1, 1, -1) <= dense_positions.view(1, -1, 1)
        selected_tokens[:, :dense_prefix_rows] = torch.where(dense_valid, dense_ids.view(1, 1, -1), pad)
    if sparse_rows:
        lanes = torch.arange(compress_ratio, dtype=torch.int32, device=index_queries.device)
        expanded = (selected_blocks[..., None] * compress_ratio + lanes).flatten(-2)
        sparse_positions = torch.arange(
            dense_prefix_rows + query_position_offset,
            query_length + query_position_offset,
            dtype=torch.int32,
            device=index_queries.device,
        )
        # Compare against 0-d int32 scalars: with a Python int operand the NPU
        # promotes the comparison to int64 (an extra cast over the full tensor).
        # Hoisting the ">= 0" test to the compact block tensor was measured to be
        # slower (the expand/flatten materialises a full-width bool copy).
        zero_i32 = torch.zeros((), dtype=torch.int32, device=index_queries.device)
        upper_i32 = torch.full((), kv_length, dtype=torch.int32, device=index_queries.device)
        valid = (expanded >= zero_i32) & (expanded < upper_i32) & (expanded <= sparse_positions.view(1, -1, 1))
        expanded = torch.where(valid, expanded, pad)
        selected_tokens[:, dense_prefix_rows:] = expanded[..., :output_width]

    if return_token_ids:
        return selected_tokens
    allowed = torch.zeros(
        (batch, 1, query_length, kv_length + 1),
        dtype=torch.bool,
        device=index_queries.device,
    )
    # selected_tokens already carries kv_length as the padding value on this
    # path, so it doubles as the scatter index: no torch.where, no int64 cast.
    allowed.scatter_(-1, selected_tokens.unsqueeze(1), True)
    allowed = allowed[..., :kv_length]
    if attention_mask.dtype == torch.bool:
        return allowed
    return torch.where(
        allowed,
        attention_mask.new_zeros(()),
        torch.finfo(attention_mask.dtype).min,
    )


@torch.no_grad()
def qwen4_exp_qsa_indexer_forward_lightning(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    attention_mask: torch.Tensor,
    past_key_values=None,
    return_block_ids: bool = False,
) -> torch.Tensor:
    """Drop-in frozen Qwen4ExpTextQSAIndexer forward using CANN V2."""
    batch, sequence, _ = hidden_states.shape
    full_cos, full_sin = position_embeddings
    if getattr(self, "qsa_triton_validate_prefix_mask", True):
        from ..qwen4_indexer import validate_lightning_mask

        validate_lightning_mask(attention_mask, sequence, full_cos.shape[1])
    current_cos = full_cos[:, -sequence:]
    current_sin = full_sin[:, -sequence:]
    projected = self.index_qk_proj(hidden_states)
    query_width = self.index_n_heads * self.index_head_dim
    query, raw_keys = projected.split([query_width, self.index_kv_heads * self.index_head_dim], dim=-1)
    if self.index_kv_heads != 1:
        raise ValueError("LightningIndexerV2 requires indexer_kv_heads == 1")
    query = query.view(batch, sequence, self.index_n_heads, self.index_head_dim)
    raw_keys = raw_keys.view(batch, sequence, self.index_head_dim)
    query = apply_rotary_pos_emb(self.q_layernorm(query), cos=current_cos, sin=current_sin)
    if past_key_values is not None:
        raw_keys = past_key_values.update_indexer(raw_keys, self.layer_idx)
    result = qsa_lightning_selected_token_mask(
        query,
        raw_keys,
        full_cos,
        full_sin,
        attention_mask,
        self.k_layernorm,
        rms_norm_eps=self.k_layernorm.eps,
        token_budget=self.token_budget,
        compress_ratio=self.compress_ratio,
        return_block_ids=return_block_ids,
        id2mask_implementation=getattr(self, "qsa_id2mask_implementation", "eager"),
    )
    return result
