# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Autograd wrapper for the fused tilelang indexer (DSA top-k scoring)."""

import torch


def pytorch_extract_topk_scores(logits, topk_indices, dim=-1):
    """Gather top-k scores from logits using topk_indices, masking invalid (-1)
    entries."""
    valid_mask = topk_indices != -1
    safe_indices = topk_indices.clamp(min=0).to(torch.int64)
    scores = torch.gather(logits, dim=dim, index=safe_indices)
    scores = torch.where(valid_mask, scores, float("-inf"))
    return scores


def pytorch_topk_with_invalid_padding(logits: torch.Tensor, topk: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Keep the fixed DSA width even when a packed microbatch has fewer keys.

    SGLang and SparseMLA represent invalid selections with -1. Match slime's
    short-context path instead of calling torch.topk with an out-of-range k.
    """
    if topk <= 0:
        raise ValueError("DSA topk must be positive")
    selected = min(topk, logits.shape[-1])
    scores, indices = torch.topk(logits, selected, dim=-1)
    indices = indices.to(torch.int32).masked_fill(scores == -torch.inf, -1)
    if selected < topk:
        scores = torch.nn.functional.pad(scores, (0, topk - selected), value=-torch.inf)
        indices = torch.nn.functional.pad(indices, (0, topk - selected), value=-1)
    return scores, indices


class IndexerFunction(torch.autograd.Function):
    """Fused indexer autograd function backed by tilelang fwd/bwd kernels."""

    @staticmethod
    def forward(
        ctx,
        index_q: torch.Tensor,
        index_k: torch.Tensor,
        weights: torch.Tensor,
        cu_seqlen_ks: torch.Tensor,
        cu_seqlen_ke: torch.Tensor,
        topk: int,
        topk_indices: torch.Tensor | None = None,
    ):
        from .tilelang_indexer_fwd import indexer_fwd_interface

        _, head_num, _ = index_q.shape

        logits = indexer_fwd_interface(index_q, index_k, weights, cu_seqlen_ks, cu_seqlen_ke, clean_logits=True)

        if topk_indices is None:
            index_score, topk_indices = pytorch_topk_with_invalid_padding(logits, topk)

        index_score = pytorch_extract_topk_scores(logits, topk_indices)

        ctx.save_for_backward(index_q, index_k, weights, cu_seqlen_ks, cu_seqlen_ke, topk_indices)
        ctx.topk = topk
        ctx.head_num = head_num

        return index_score, topk_indices

    @staticmethod
    def backward(ctx, grad_scores, grad_indices):
        from .tilelang_indexer_bwd import indexer_bwd_interface

        index_q, index_k, weights, cu_seqlen_ks, cu_seqlen_ke, topk_indices = ctx.saved_tensors

        grad_q, grad_w, grad_k = indexer_bwd_interface(index_q, weights, index_k, topk_indices, grad_scores)

        return grad_q, grad_k, grad_w, None, None, None, None


def lighting_indexer(
    index_q: torch.Tensor,
    index_k: torch.Tensor,
    weights: torch.Tensor,
    cu_seqlen_ks: torch.Tensor,
    cu_seqlen_ke: torch.Tensor,
    topk: int,
    topk_indices: torch.Tensor | None = None,
):
    """Run the fused indexer to obtain top-k scores and indices for sparse
    MLA."""
    return IndexerFunction.apply(index_q, index_k, weights.squeeze(-1), cu_seqlen_ks, cu_seqlen_ke, topk, topk_indices)


def generate_varlen_mask_params(cu_seqlens: torch.Tensor, *, seq_len: int):
    """Compute per-token (start, end) bounds for variable-length sequences from
    cu_seqlens."""
    q_indices = torch.arange(0, seq_len, device=cu_seqlens.device)
    seq_indices = torch.searchsorted(cu_seqlens, q_indices, right=True) - 1
    starts = cu_seqlens[seq_indices]
    ends = q_indices + 1
    return starts, ends
