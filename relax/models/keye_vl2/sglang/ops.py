# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Graph-safe Keye operators with matching training and rollout arithmetic."""

from functools import lru_cache
from typing import Any

import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice


@lru_cache(maxsize=16)
def _frequencies(dim: int, theta: float, device: torch.device) -> torch.Tensor:
    return 1.0 / (theta ** (torch.arange(0, dim, 2, device=device, dtype=torch.float32) / dim))


@triton.jit
def _rope(X, P, F, Y, H: tl.constexpr, D: tl.constexpr, PS: tl.constexpr, AXES: tl.constexpr):
    row = tl.program_id(0)
    d = tl.arange(0, D // 2)
    token = row // H
    axis = tl.where(d < D // 8, 0, tl.where(d < D // 8 + D * 3 // 16, 1, 2))
    position = tl.load(P + token * PS + (axis if AXES else 0)).to(tl.float32)
    angle = position * tl.load(F + d)
    c = libdevice.cos(angle).to(X.dtype.element_ty)
    s = libdevice.sin(angle).to(X.dtype.element_ty)
    a = tl.load(X + row * D + d)
    b = tl.load(X + row * D + d + D // 2)
    # Match the BF16 reference's separately rounded products (no fused FMA).
    ac = (a.to(tl.float32) * c.to(tl.float32)).to(a.dtype).to(tl.float32)
    bs = ((-b).to(tl.float32) * s.to(tl.float32)).to(a.dtype).to(tl.float32)
    bc = (b.to(tl.float32) * c.to(tl.float32)).to(a.dtype).to(tl.float32)
    ass = (a.to(tl.float32) * s.to(tl.float32)).to(a.dtype).to(tl.float32)
    tl.store(Y + row * D + d, ac + bs)
    tl.store(Y + row * D + d + D // 2, bc + ass)


def rope(x: torch.Tensor, positions: torch.Tensor, theta: float) -> torch.Tensor:
    x = x.contiguous()
    positions = positions.contiguous()
    output = torch.empty_like(x)
    _rope[(x.shape[0] * x.shape[1],)](
        x,
        positions,
        _frequencies(x.shape[-1], theta, x.device),
        output,
        x.shape[1],
        x.shape[2],
        positions.stride(0),
        positions.ndim == 2,
        enable_fp_fusion=False,
    )
    return output


@triton.jit
def _index_scores(
    Q,
    K,
    W,
    TABLE,
    REQUESTS,
    LENGTHS,
    OUT,
    QS: tl.constexpr,
    WS: tl.constexpr,
    TS: tl.constexpr,
    WIDTH: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row, tile = tl.program_id(0), tl.program_id(1)
    n = tile * BLOCK + tl.arange(0, BLOCK)
    length = tl.load(LENGTHS + row)
    result = tl.full((BLOCK,), float("-inf"), tl.float32)
    if tile * BLOCK < length:
        request = tl.load(REQUESTS + row)
        slots = tl.load(TABLE + request * TS + n, mask=(n < length) & (n < WIDTH), other=0)
        h = tl.arange(0, 16)
        d = tl.arange(0, 64)
        q = tl.load(Q + row * QS + h[:, None] * 64 + d[None, :])
        k = tl.load(K + slots[None, :] * 64 + d[:, None], mask=n[None, :] < length, other=0)
        # Keep every BF16 rounding boundary of indexer_scores, including
        # matmul -> scale -> head weight -> reduction, to preserve top-k.
        dots = tl.dot(q, k).to(q.dtype).to(tl.float32)
        scores = (dots * 0.125).to(q.dtype).to(tl.float32)
        weights = tl.load(W + row * WS + h)
        weighted = (tl.maximum(scores, 0) * weights[:, None].to(tl.float32)).to(q.dtype)
        result = tl.sum(weighted.to(tl.float32), axis=0)
        result = tl.where(n < length, result, float("-inf"))
    tl.store(OUT + row * WIDTH + n, result, mask=n < WIDTH)


def paged_scores(
    query: torch.Tensor,
    cache: torch.Tensor,
    weights: torch.Tensor,
    table: torch.Tensor,
    requests: torch.Tensor,
    lengths: torch.Tensor,
    width: int,
) -> torch.Tensor:
    scores = torch.empty((query.shape[0], width), device=query.device, dtype=query.dtype)
    _index_scores[(query.shape[0], triton.cdiv(width, 128))](
        query,
        cache,
        weights,
        table,
        requests,
        lengths,
        scores,
        query.stride(0),
        weights.stride(0),
        table.stride(0),
        width,
        128,
        enable_fp_fusion=False,
    )
    return scores


def paged_decode(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    index_query: torch.Tensor,
    index_cache: torch.Tensor,
    weights: torch.Tensor,
    table: torch.Tensor,
    requests: torch.Tensor,
    lengths: torch.Tensor,
    width: int,
) -> torch.Tensor:
    """Keep KV paged while using the same sparse forward as training."""
    from relax.models.keye_vl2.sparse_attention import compact_indices, sparse_attention

    scores = paged_scores(index_query, index_cache, weights, table, requests, lengths, width)
    selected = scores.topk(min(2048, width), dim=-1).indices
    ordered, counts = compact_indices(selected, selected < lengths[:, None])
    slots = table[requests.long()[:, None], ordered.long()].int().contiguous()
    return sparse_attention(query, key, value, slots, counts).flatten(1)


@triton.jit
def _merge_experts(X, ORDER, Y, H: tl.constexpr, K: tl.constexpr, BLOCK: tl.constexpr):
    token = tl.program_id(0)
    h = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    total = tl.full((BLOCK,), 0, tl.float32)
    for slot in range(K):
        expert = tl.load(ORDER + token * K + slot)
        value = tl.load(X + (token * K + expert) * H + h, mask=h < H, other=0).to(tl.float32)
        total = total + value
    tl.store(Y + token * H + h, total, mask=h < H)


def merge_experts(routed: torch.Tensor, expert_ids: torch.Tensor) -> torch.Tensor:
    """Match TE unpermute: descending expert order, sequential FP32 adds."""
    tokens, topk, hidden = routed.shape
    order = expert_ids.argsort(1, descending=True)
    output = torch.empty((tokens, hidden), device=routed.device, dtype=routed.dtype)
    _merge_experts[(tokens, triton.cdiv(hidden, 256))](routed, order, output, hidden, topk, 256)
    return output


def expert_forward(
    hidden: torch.Tensor,
    gate_up: torch.Tensor,
    down: torch.Tensor,
    probabilities: torch.Tensor,
    expert_ids: torch.Tensor,
    tp_group: Any = None,
) -> torch.Tensor:
    """Match Megatron's unfused BF16 SwiGLU and pre-down routing weights."""
    from sglang.kernels.ops.moe.fused_moe_triton_kernels import invoke_fused_moe_kernel
    from sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe import _prepare_fused_moe_run

    if hidden.shape[0] == 0:
        return hidden
    sharded = tp_group is not None and tp_group.world_size > 1
    # Config selection inspects only shapes. Use the complete contraction's
    # geometry so column sharding does not change the GEMM reduction tiles.
    config_weight = (
        torch.empty((gate_up.shape[0], down.shape[2] * 2, gate_up.shape[2]), device="meta") if sharded else gate_up
    )
    config, down_config, *_, sorted_ids, block_experts, padded_count = _prepare_fused_moe_run(
        hidden,
        config_weight,
        down,
        expert_ids,
        use_fp8_w8a8=False,
        use_int8_w8a8=False,
        use_int8_w8a16=False,
        use_int4_w4a16=False,
        per_channel_quant=False,
        block_shape=None,
    )
    tokens, topk = expert_ids.shape
    projected = torch.empty((tokens * topk, gate_up.shape[1]), device=hidden.device, dtype=hidden.dtype)
    options = dict(
        compute_type=tl.bfloat16,
        use_fp8_w8a8=False,
        use_int8_w8a8=False,
        use_int8_w8a16=False,
        use_int4_w4a16=False,
        per_channel_quant=False,
        filter_expert=True,
    )
    invoke_fused_moe_kernel(
        hidden,
        gate_up,
        None,
        projected,
        None,
        None,
        None,
        probabilities,
        expert_ids,
        sorted_ids,
        block_experts,
        padded_count,
        False,
        topk,
        config,
        **options,
    )
    gate, up = projected.chunk(2, -1)
    # Each BF16 boundary matters: SiLU -> multiply up -> multiply FP32 route
    # probability -> BF16, all before the down GEMM, as in TEGroupedMLP.
    activated = (torch.nn.functional.silu(gate) * up * probabilities.reshape(-1, 1)).to(hidden.dtype)
    if sharded:
        activated = tp_group.all_gather(activated, dim=1)
        # The existing MoE TP reduction broadcasts this complete result. Do
        # not round partial down projections or merge partial expert outputs.
        if tp_group.rank_in_group != 0:
            return torch.zeros_like(hidden)
    routed = torch.empty((tokens, topk, down.shape[1]), device=hidden.device, dtype=hidden.dtype)
    invoke_fused_moe_kernel(
        activated,
        down,
        None,
        routed,
        None,
        None,
        None,
        probabilities,
        expert_ids,
        sorted_ids,
        block_experts,
        padded_count,
        False,
        1,
        down_config or config,
        **options,
    )
    return merge_experts(routed, expert_ids)
