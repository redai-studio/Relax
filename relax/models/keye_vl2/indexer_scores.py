# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Fixed-tile BF16 indexer arithmetic shared by dense and paged attention."""

import torch
import triton
import triton.language as tl
from torch.autograd.function import FunctionCtx


@triton.jit
def _index_scores(
    Q,
    K,
    W,
    TABLE,
    REQUESTS,
    LENGTHS,
    OUT,
    SCALED,
    QS: tl.constexpr,
    WS: tl.constexpr,
    TS: tl.constexpr,
    WIDTH: tl.constexpr,
    ROWS: tl.constexpr,
    PAGED: tl.constexpr,
    SAVE_SCALED: tl.constexpr,
):
    row, tile = tl.program_id(0), tl.program_id(1)
    n = tile * 128 + tl.arange(0, 128)
    length = tl.load(LENGTHS + row) if PAGED else WIDTH
    result = tl.full((128,), float("-inf"), tl.float32)
    if tile * 128 < length:
        if PAGED:
            request = tl.load(REQUESTS + row)
            slots = tl.load(TABLE + request * TS + n, mask=(n < length) & (n < WIDTH), other=0)
        else:
            slots = n
        h = tl.arange(0, 16)
        d = tl.arange(0, 64)
        q = tl.load(Q + row * QS + h[:, None] * 64 + d[None, :])
        k = tl.load(K + slots[None, :] * 64 + d[:, None], mask=(n[None, :] < length) & (n[None, :] < WIDTH), other=0)
        # The tile and reduction order do not depend on the sequence length.
        # Preserve each BF16 boundary of matmul -> scale -> weighted ReLU -> sum.
        dots = tl.dot(q, k, input_precision="ieee").to(q.dtype).to(tl.float32)
        scaled = (dots * 0.125).to(q.dtype)
        weights = tl.load(W + row * WS + h)
        weighted = (tl.maximum(scaled.to(tl.float32), 0) * weights[:, None].to(tl.float32)).to(q.dtype)
        result = tl.sum(weighted.to(tl.float32), axis=0)
        result = tl.where(n < length, result, float("-inf"))
        if SAVE_SCALED:
            tl.store(SCALED + (h[:, None] * ROWS + row) * WIDTH + n[None, :], scaled, mask=n[None, :] < WIDTH)
    tl.store(OUT + row * WIDTH + n, result, mask=n < WIDTH)


def _dense_scores(
    query: torch.Tensor, key: torch.Tensor, weights: torch.Tensor, *, save_scaled: bool = False
) -> tuple[torch.Tensor, torch.Tensor | None]:
    query, key, weights = query.contiguous(), key.contiguous(), weights.contiguous()
    rows, width = query.shape[0], key.shape[0]
    output = query.new_empty((rows, width))
    scaled = query.new_empty((16, rows, width)) if save_scaled else None
    if rows == 0 or width == 0:
        return output, scaled
    _index_scores[(rows, triton.cdiv(width, 128))](
        query,
        key,
        weights,
        None,
        None,
        None,
        output,
        scaled,
        query.stride(0),
        weights.stride(0),
        0,
        width,
        rows,
        False,
        save_scaled,
        num_warps=4,
        enable_fp_fusion=False,
    )
    return output, scaled


class _IndexerScores(torch.autograd.Function):
    @staticmethod
    def forward(ctx: FunctionCtx, query: torch.Tensor, key: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        if any(ctx.needs_input_grad):
            ctx.save_for_backward(query, key, weights)
        return _dense_scores(query, key, weights)[0]

    @staticmethod
    def backward(ctx: FunctionCtx, grad: torch.Tensor) -> tuple[torch.Tensor | None, ...]:
        query, key, weights = ctx.saved_tensors
        # Recompute the same fixed-tile dot: a shape-dependent GEMM here could
        # change the ReLU gate and dWeights relative to the actual forward.
        _, scaled = _dense_scores(query, key, weights, save_scaled=True)
        assert scaled is not None
        upstream = grad.unsqueeze(0)
        dweights = (upstream * scaled.relu()).sum(-1).transpose(0, 1) if ctx.needs_input_grad[2] else None
        dquery = dkey = None
        if ctx.needs_input_grad[0] or ctx.needs_input_grad[1]:
            ddots = upstream * weights.transpose(0, 1).unsqueeze(-1)
            ddots = ddots.masked_fill(scaled <= 0, 0) * 0.125
            if ctx.needs_input_grad[0]:
                dquery = torch.einsum("hqk,kd->qhd", ddots, key.squeeze(-2))
            if ctx.needs_input_grad[1]:
                dkey = torch.einsum("hqk,qhd->kd", ddots, query).unsqueeze(-2)
        return dquery, dkey, dweights


def dense_indexer_scores(query: torch.Tensor, key: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    if query.shape[1:] != (16, 64) or key.shape[1:] != (1, 64) or weights.shape != query.shape[:2]:
        raise ValueError("Keye indexer requires query [Q,16,64], key [K,1,64], and weights [Q,16]")
    return _IndexerScores.apply(query, key, weights)


def paged_indexer_scores(
    query: torch.Tensor,
    cache: torch.Tensor,
    weights: torch.Tensor,
    table: torch.Tensor,
    requests: torch.Tensor,
    lengths: torch.Tensor,
    width: int,
) -> torch.Tensor:
    scores = query.new_empty((query.shape[0], width))
    _index_scores[(query.shape[0], triton.cdiv(width, 128))](
        query,
        cache,
        weights,
        table,
        requests,
        lengths,
        scores,
        None,
        query.stride(0),
        weights.stride(0),
        table.stride(0),
        width,
        query.shape[0],
        True,
        False,
        num_warps=4,
        enable_fp_fusion=False,
    )
    return scores
