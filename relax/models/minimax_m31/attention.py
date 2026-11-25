# Copyright (c) 2026 Relax Authors. All Rights Reserved.
# Copyright 2025-2026 RadixArk, Inc. (Miles).
# Copyright 2025 Zhipu AI (Slime).
# SPDX-License-Identifier: Apache-2.0
# Adapted for Relax from Miles (https://github.com/radixark/miles), commit 3a7659a4.
# Miles includes software developed from Slime (https://github.com/THUDM/slime).

"""MiniMax-M3 self-attention: GQA restricted to the MSA indexer's top-k key
blocks.

Index head ``h`` scores each causal key block by its max logit and picks, with
the query's own block forced in, the blocks kv head ``h`` attends to.
"""

import torch
import torch.distributed as dist
from megatron.core.extensions.transformer_engine import TELinear
from megatron.core.ssm.mamba_context_parallel import _thd_partitioned_indices, _undo_attention_load_balancing
from megatron.core.tensor_parallel.mappings import gather_from_sequence_parallel_region
from megatron.core.transformer.attention import SelfAttention
from megatron.core.transformer.module import MegatronModule
from torch import Tensor

from .msa_scores import msa_block_max_scores
from .packed_layout import PackedBlockLayout
from .sparse_attention import qsa_block_sparse_attention


# topk priority: init blocks, then local blocks, then the best scores; future blocks never
_INIT_BLOCK_SCORE = 1e30
_LOCAL_BLOCK_SCORE = 1e29
_INVALID_BLOCK_SCORE = -1e30


def gemma_rmsnorm_fp32(x: Tensor, weight: Tensor, eps: float) -> Tensor:
    xf = x.float()
    xf = xf * torch.rsqrt(xf.pow(2).mean(dim=-1, keepdim=True) + eps)
    return xf * (1.0 + weight.float())


def apply_neox_rope_fp32(x: Tensor, rotary_pos_emb: Tensor, positions: Tensor) -> Tensor:
    """Neox rope on the leading ``rotary_dim`` dims of fp32 ``x`` ``[T, heads,
    head_dim]``."""
    freqs = rotary_pos_emb.flatten(1)
    half = freqs.shape[-1] // 2
    freqs = freqs[:, :half].float().contiguous()
    cos = freqs.cos()[positions.long()].unsqueeze(-2)
    sin = freqs.sin()[positions.long()].unsqueeze(-2)
    x1 = x[..., :half]
    x2 = x[..., half : 2 * half]
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin, x[..., 2 * half :]], dim=-1)


class MiniMaxRMSNorm(torch.nn.Module):
    """Per-head gemma RMSNorm that stays in fp32, so rope runs in fp32 and q/k
    round to bf16 once as in sglang."""

    def __init__(self, config, hidden_size: int, eps: float):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(hidden_size, dtype=config.params_dtype))
        self.weight.sequence_parallel = config.sequence_parallel
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        return gemma_rmsnorm_fp32(x, self.weight, self.eps)


class MiniMaxM3Indexer(MegatronModule):
    """Scores key blocks per index head and returns each query's selected block
    set."""

    def __init__(self, config):
        super().__init__(config)
        sparse = config.msa_config
        self.n_heads = sparse["sparse_num_index_heads"]
        self.head_dim = sparse["sparse_index_dim"]
        self.topk_blocks = sparse["sparse_topk_blocks"]
        self.block_size = sparse["sparse_block_size"]
        self.local_blocks = sparse["sparse_local_block"]
        self.init_blocks = sparse["sparse_init_block"]
        self.eps = config.layernorm_epsilon
        self.index_q_proj = TELinear(
            config.hidden_size,
            self.n_heads * self.head_dim,
            config=config,
            init_method=config.init_method,
            bias=False,
            skip_bias_add=False,
            skip_weight_param_allocation=False,
            parallel_mode="duplicated",
        )
        self.index_k_proj = TELinear(
            config.hidden_size,
            self.head_dim,
            config=config,
            init_method=config.init_method,
            bias=False,
            skip_bias_add=False,
            skip_weight_param_allocation=False,
            parallel_mode="duplicated",
        )
        self.index_q_norm = torch.nn.Parameter(torch.zeros(self.head_dim, dtype=config.params_dtype))
        self.index_k_norm = torch.nn.Parameter(torch.zeros(self.head_dim, dtype=config.params_dtype))
        for p in self.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def project(self, hidden_states: Tensor, positions: Tensor, rotary_pos_emb: Tensor, heads: list[int]):
        """``[T, hidden] -> (q [T, len(heads), D], k [T, D])`` in bf16, normed
        then roped in fp32."""
        q, _ = self.index_q_proj(hidden_states)
        k, _ = self.index_k_proj(hidden_states)
        q = q.view(-1, self.n_heads, self.head_dim)[:, heads]
        q = gemma_rmsnorm_fp32(q, self.index_q_norm, self.eps)
        k = gemma_rmsnorm_fp32(k.view(-1, 1, self.head_dim), self.index_k_norm, self.eps)
        q = apply_neox_rope_fp32(q, rotary_pos_emb, positions).to(hidden_states.dtype)
        k = apply_neox_rope_fp32(k, rotary_pos_emb, positions).to(hidden_states.dtype)
        return q, k.squeeze(1)

    @torch.no_grad()
    def select_flags(
        self, q: Tensor, k: Tensor, layout: PackedBlockLayout, bounds: list[int], *, cp_size: int = 1, cp_rank: int = 0
    ) -> Tensor:
        """Return selected blocks as packed flags of shape ``[H, T, N]``."""
        positions = layout.positions
        bs = self.block_size
        scores = msa_block_max_scores(q, k, bounds, bs, q.shape[-1] ** -0.5, cp_size=cp_size, cp_rank=cp_rank)
        block_ids = torch.arange(scores.shape[-1], device=q.device)
        valid_blocks = positions.long() // bs + 1
        valid = block_ids[None, :] < valid_blocks[:, None]
        local = valid & (block_ids[None, :] >= (valid_blocks - self.local_blocks).clamp(min=0)[:, None])
        init = valid & (block_ids[None, :] < self.init_blocks)
        scores = torch.where(valid, scores, _INVALID_BLOCK_SCORE)
        scores = torch.where(local, _LOCAL_BLOCK_SCORE, scores)
        scores = torch.where(init, _INIT_BLOCK_SCORE, scores)
        kk = min(self.topk_blocks, scores.shape[-1])
        top_score, top_idx = torch.topk(scores, kk, dim=-1)
        selected = top_score > _INVALID_BLOCK_SCORE
        # Invalid top-k entries can exceed the packed block range. Scatter
        # zero into a safe column without overwriting a valid block's flag.
        packed_blocks = torch.where(selected, layout.token_block_start[None, :, None] + top_idx, 0)
        flags = torch.zeros(*top_idx.shape[:2], layout.num_blocks, dtype=torch.uint8, device=q.device)
        return flags.scatter_reduce_(2, packed_blocks, selected.to(torch.uint8), "amax")


class MiniMaxM3SparseCoreAttention(MegatronModule):
    """Core attention restricted, per kv head, to the blocks its index head
    selected."""

    def __init__(self, config, owner):
        super().__init__(config)
        object.__setattr__(self, "_owner", owner)
        self.softmax_scale = config.kv_channels**-0.5

    def forward(
        self, query: Tensor, key: Tensor, value: Tensor, attention_mask=None, *, packed_seq_params=None, **kwargs
    ):
        block_form = self._owner._msa_block_form
        assert block_form is not None, "MiniMax-M3 core attention ran without a published block selection"
        assert query.dim() == 3, f"MiniMax-M3 sparse attention expects thd inputs, got {tuple(query.shape)}"
        flags, lo, hi, blk_base, tok_base, block_first, block_last, block_size = block_form
        query, key = query.to(value.dtype), key.to(value.dtype)
        cp_group = None
        if packed_seq_params is not None and self.config.context_parallel_size > 1:
            cp_group = self._owner.pg_collection.cp

        T, hq, d = query.shape
        hkv = key.shape[1]
        group = hq // hkv
        outs = []
        for h in range(hkv):
            outs.append(
                qsa_block_sparse_attention(
                    query[:, h * group : (h + 1) * group],
                    key[:, h : h + 1],
                    value[:, h : h + 1],
                    flags[h],
                    lo,
                    hi,
                    blk_base,
                    tok_base,
                    block_first,
                    block_last,
                    self.softmax_scale,
                    block_size,
                    cp_group=cp_group,
                    packed_seq_params=packed_seq_params,
                )
            )
        return torch.cat(outs, dim=1).reshape(T, hq * d)


class MiniMaxM31Attention(SelfAttention):
    """Megatron GQA self-attention; on sparse layers restricted to the MiniMax
    (MSA) indexer's blocks."""

    def __init__(self, config, submodules, layer_number=1, *args, **kwargs):
        super().__init__(config, submodules, layer_number, *args, **kwargs)
        if config.rotary_interleaved:
            raise NotImplementedError("MiniMax-M3 uses neox rope; rotary_interleaved is unsupported")
        self._msa_block_form = None
        if self.checkpoint_core_attention:
            raise NotImplementedError("MiniMax-M3 sparse attention does not support selective core_attn recompute")
        self.indexer = MiniMaxM3Indexer(config)
        self.core_attention = MiniMaxM3SparseCoreAttention(config, owner=self)

    def forward(self, hidden_states: Tensor, *args, **kwargs):
        packed = kwargs.get("packed_seq_params")
        assert packed is not None and packed.cu_seqlens_q is not None, "MiniMax-M3 sparse attention needs thd inputs"
        rotary_pos_emb = kwargs.get("rotary_pos_emb")
        assert rotary_pos_emb is not None, "the MSA indexer reuses the attention rope table"
        if isinstance(rotary_pos_emb, tuple):
            rotary_pos_emb = rotary_pos_emb[0]

        with torch.no_grad():
            tp_group = self.tp_group
            tp_size = dist.get_world_size(tp_group)
            indexer_states = hidden_states
            if self.config.sequence_parallel and tp_size > 1:
                indexer_states = gather_from_sequence_parallel_region(
                    hidden_states, tensor_parallel_output_grad=False, group=tp_group
                )
            seq = indexer_states.shape[0]
            bounds = packed.cu_seqlens_q_cpu
            cp_group = self.pg_collection.cp
            cp_size = self.config.context_parallel_size
            cp_rank = dist.get_rank(cp_group) if cp_size > 1 else 0
            if bounds[-1] != seq * cp_size:
                raise ValueError("MiniMax-M3.1 requires THD metadata for the complete packed sequence.")
            query_ids = None
            if cp_size > 1:
                cu_seqlens = packed.cu_seqlens_q_padded
                if cu_seqlens is None:
                    cu_seqlens = packed.cu_seqlens_q
                query_ids = _thd_partitioned_indices(cu_seqlens, bounds[-1], cp_size, cp_rank).long()
            layout = PackedBlockLayout(bounds, self.indexer.block_size, hidden_states.device, query_ids)
            tp_rank = dist.get_rank(tp_group)
            kv_heads_per_rank = self.config.num_query_groups // tp_size
            local_heads = list(range(tp_rank * kv_heads_per_rank, (tp_rank + 1) * kv_heads_per_rank))
            q, k = self.indexer.project(indexer_states[:, 0], layout.positions, rotary_pos_emb, local_heads)
            if cp_size > 1:
                k = gather_from_sequence_parallel_region(k, tensor_parallel_output_grad=False, group=cp_group)
                k = _undo_attention_load_balancing(k, cp_size, packed)
            token_start = layout.token_start.to(torch.int32)
            self._msa_block_form = (
                self.indexer.select_flags(q, k, layout, bounds, cp_size=cp_size, cp_rank=cp_rank),
                token_start,
                layout.query_ids.to(torch.int32),
                layout.token_block_start.to(torch.int32),
                token_start,
                layout.block_first_token.to(torch.int32),
                layout.block_last_token.to(torch.int32),
                self.indexer.block_size,
            )
        try:
            return super().forward(hidden_states, *args, **kwargs)
        finally:
            self._msa_block_form = None
