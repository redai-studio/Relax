# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Training-oriented fused QSA attention with an explicit backward.

FlexAttention evaluates the QSA ``mask_mod`` once per token pair, and each
evaluation gathers two document ids, two positions and one selection byte. At
packed 128K the routed pattern is only ~20% dense but scattered at the R=4
group granularity, so the 128x128 coarse classifier keeps essentially every
causal tile: the pair-level mask work, not the matmul, dominates.

These kernels keep the dense tile shape and tensor-core ``tl.dot`` but reduce
the mask to one 32-bit selection word per (query, 128-token region), turning
``O(BM * BN)`` gathers per tile into ``O(BM + BN)`` metadata loads. The backward
reduces dK/dV over KV blocks instead of atomically scattering them, so
gradients are deterministic and bitwise reproducible across runs.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from torch import Tensor


_BLOCK = 128
# A 256-wide head already fills shared memory with one Q/K/V stage.
_NUM_STAGES = 1


@triton.jit
def _admitted(
    Positions,
    DocIds,
    Bits32,
    query_global,
    query_valid,
    query_position,
    query_doc,
    tail_start,
    start_n,
    words_per_row,
    kv_len,
    RATIO: tl.constexpr,
    BN: tl.constexpr,
):
    """QSA ``mask_mod`` for one tile, evaluated from per-row selection
    words."""
    offs_n = start_n + tl.arange(0, BN)
    key_valid = offs_n < kv_len
    key_position = tl.load(Positions + offs_n, key_valid, other=0).to(tl.int32)
    key_doc = tl.load(DocIds + offs_n, key_valid, other=-2).to(tl.int32)
    block_id = key_position // RATIO
    # One word covers 32 R=4 groups, i.e. the 128 tokens of a classifier tile.
    word = tl.load(
        Bits32 + query_global[:, None].to(tl.int64) * words_per_row + (block_id >> 5)[None, :],
        query_valid[:, None] & key_valid[None, :],
        other=0,
    )
    selected = ((word >> (block_id & 31)[None, :]) & 1) != 0
    admitted = (query_doc[:, None] == key_doc[None, :]) & (key_position[None, :] <= query_position[:, None])
    admitted &= selected | (key_position[None, :] >= tail_start[:, None])
    return admitted & query_valid[:, None] & key_valid[None, :], offs_n, key_valid


@triton.jit
def _query_metadata(
    Positions,
    DocIds,
    QueryIds,
    offs_m,
    q_len,
    RATIO: tl.constexpr,
    HAS_QUERY_IDS: tl.constexpr,
):
    query_valid = offs_m < q_len
    if HAS_QUERY_IDS:
        query_global = tl.load(QueryIds + offs_m, query_valid, other=0).to(tl.int32)
    else:
        query_global = offs_m.to(tl.int32)
    query_position = tl.load(Positions + query_global, query_valid, other=0).to(tl.int32)
    query_doc = tl.load(DocIds + query_global, query_valid, other=-1).to(tl.int32)
    tail_start = ((query_position + 1) // RATIO) * RATIO
    return query_valid, query_global, query_position, query_doc, tail_start


@triton.jit(do_not_specialize=["q_len", "kv_len", "words_per_row", "blocks_per_row"])
def _qsa_fused_fwd(
    Q,
    K,
    V,
    O,
    Lse,
    Positions,
    DocIds,
    Bits32,
    QueryIds,
    BlockCounts,
    BlockIndices,
    q_len,
    kv_len,
    words_per_row,
    blocks_per_row,
    stride_qm,
    stride_qh,
    stride_kn,
    stride_kh,
    scale,
    HQ: tl.constexpr,
    HK: tl.constexpr,
    DIM: tl.constexpr,
    RATIO: tl.constexpr,
    BLOCK: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    HAS_QUERY_IDS: tl.constexpr,
    OutResidual=None,
):
    start_m = tl.program_id(0) * BM
    head = tl.program_id(1)
    kv_head = head // (HQ // HK)
    offs_m = start_m + tl.arange(0, BM)
    offs_d = tl.arange(0, DIM)
    query_valid, query_global, query_position, query_doc, tail_start = _query_metadata(
        Positions, DocIds, QueryIds, offs_m, q_len, RATIO, HAS_QUERY_IDS
    )
    query = tl.load(
        Q + offs_m[:, None] * stride_qm + head * stride_qh + offs_d[None, :],
        query_valid[:, None],
        other=0.0,
    )
    running_max = tl.full((BM,), -float("inf"), tl.float32)
    running_sum = tl.zeros((BM,), tl.float32)
    acc = tl.zeros((BM, DIM), tl.float32)

    classifier_block = start_m // BLOCK
    count = tl.load(BlockCounts + classifier_block)
    for index in range(count):
        block = tl.load(BlockIndices + classifier_block * blocks_per_row + index)
        for sub in tl.static_range(BLOCK // BN):
            start_n = block * BLOCK + sub * BN
            admitted, offs_n, key_valid = _admitted(
                Positions,
                DocIds,
                Bits32,
                query_global,
                query_valid,
                query_position,
                query_doc,
                tail_start,
                start_n,
                words_per_row,
                kv_len,
                RATIO,
                BN,
            )
            key = tl.load(
                K + offs_n[:, None] * stride_kn + kv_head * stride_kh + offs_d[None, :],
                key_valid[:, None],
                other=0.0,
            )
            scores = tl.dot(query, tl.trans(key)) * scale
            scores = tl.where(admitted, scores, -float("inf"))
            tile_max = tl.max(scores, 1)
            next_max = tl.maximum(running_max, tile_max)
            # A tile may admit nothing for this query; keep -inf - -inf out of exp.
            safe_max = tl.where(next_max == -float("inf"), 0.0, next_max)
            alpha = tl.where(running_max == -float("inf"), 0.0, tl.exp(running_max - safe_max))
            probabilities = tl.where(admitted, tl.exp(scores - safe_max[:, None]), 0.0)
            value = tl.load(
                V + offs_n[:, None] * stride_kn + kv_head * stride_kh + offs_d[None, :],
                key_valid[:, None],
                other=0.0,
            )
            # Preserve the low part for cancellation-sensitive backward delta.
            probabilities_hi = probabilities.to(value.dtype)
            probabilities_lo = (probabilities - probabilities_hi.to(tl.float32)).to(value.dtype)
            acc = acc * alpha[:, None] + tl.dot(probabilities_hi, value)
            acc += tl.dot(probabilities_lo, value)
            running_sum = running_sum * alpha + tl.sum(probabilities, 1)
            running_max = next_max

    empty = running_sum == 0.0
    denominator = tl.where(empty, 1.0, running_sum)
    output = acc / denominator[:, None]
    rounded_output = output.to(O.dtype.element_ty)
    output_offset = offs_m[:, None] * stride_qm + head * stride_qh + offs_d[None, :]
    tl.store(O + output_offset, rounded_output, query_valid[:, None])
    if OutResidual is not None:
        residual = output - rounded_output.to(tl.float32)
        tl.store(OutResidual + output_offset, residual, query_valid[:, None])
    lse = tl.where(empty, -float("inf"), running_max + tl.log(denominator))
    tl.store(Lse + head * q_len + offs_m, lse, query_valid)


@triton.jit(do_not_specialize=["q_len"])
def _qsa_fused_delta(
    O,
    DO,
    Delta,
    q_len,
    stride_qm,
    stride_qh,
    HQ: tl.constexpr,
    DIM: tl.constexpr,
    BM: tl.constexpr,
    OutResidual=None,
):
    offs_m = tl.program_id(0) * BM + tl.arange(0, BM)
    head = tl.program_id(1)
    offs_d = tl.arange(0, DIM)
    valid = offs_m < q_len
    base = offs_m[:, None] * stride_qm + head * stride_qh + offs_d[None, :]
    out = tl.load(O + base, valid[:, None], other=0.0).to(tl.float32)
    if OutResidual is not None:
        out += tl.load(OutResidual + base, valid[:, None], other=0.0).to(tl.float32)
    grad = tl.load(DO + base, valid[:, None], other=0.0).to(tl.float32)
    tl.store(Delta + head * q_len + offs_m, tl.sum(out * grad, 1), valid)


@triton.jit(do_not_specialize=["q_len", "kv_len", "words_per_row", "blocks_per_row"])
def _qsa_fused_bwd_dq(
    Q,
    K,
    V,
    DO,
    DQ,
    Lse,
    Delta,
    Positions,
    DocIds,
    Bits32,
    QueryIds,
    BlockCounts,
    BlockIndices,
    q_len,
    kv_len,
    words_per_row,
    blocks_per_row,
    stride_qm,
    stride_qh,
    stride_kn,
    stride_kh,
    scale,
    HQ: tl.constexpr,
    HK: tl.constexpr,
    DIM: tl.constexpr,
    RATIO: tl.constexpr,
    BLOCK: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    HAS_QUERY_IDS: tl.constexpr,
):
    start_m = tl.program_id(0) * BM
    head = tl.program_id(1)
    kv_head = head // (HQ // HK)
    offs_m = start_m + tl.arange(0, BM)
    offs_d = tl.arange(0, DIM)
    query_valid, query_global, query_position, query_doc, tail_start = _query_metadata(
        Positions, DocIds, QueryIds, offs_m, q_len, RATIO, HAS_QUERY_IDS
    )
    base = offs_m[:, None] * stride_qm + head * stride_qh + offs_d[None, :]
    query = tl.load(Q + base, query_valid[:, None], other=0.0)
    grad_out = tl.load(DO + base, query_valid[:, None], other=0.0)
    lse = tl.load(Lse + head * q_len + offs_m, query_valid, other=0.0)
    delta = tl.load(Delta + head * q_len + offs_m, query_valid, other=0.0)
    finite = lse != -float("inf")
    lse = tl.where(finite, lse, 0.0)
    dq = tl.zeros((BM, DIM), tl.float32)

    classifier_block = start_m // BLOCK
    count = tl.load(BlockCounts + classifier_block)
    for index in range(count):
        block = tl.load(BlockIndices + classifier_block * blocks_per_row + index)
        for sub in tl.static_range(BLOCK // BN):
            start_n = block * BLOCK + sub * BN
            admitted, offs_n, key_valid = _admitted(
                Positions,
                DocIds,
                Bits32,
                query_global,
                query_valid,
                query_position,
                query_doc,
                tail_start,
                start_n,
                words_per_row,
                kv_len,
                RATIO,
                BN,
            )
            admitted &= finite[:, None]
            kv_base = offs_n[:, None] * stride_kn + kv_head * stride_kh + offs_d[None, :]
            key = tl.load(K + kv_base, key_valid[:, None], other=0.0)
            value = tl.load(V + kv_base, key_valid[:, None], other=0.0)
            scores = tl.dot(query, tl.trans(key)) * scale
            probabilities = tl.where(admitted, tl.exp(scores - lse[:, None]), 0.0)
            grad_probabilities = tl.dot(grad_out, tl.trans(value))
            grad_scores = probabilities * (grad_probabilities - delta[:, None]) * scale
            # Retain cancellation information through the dS @ K reduction.
            scores_hi = grad_scores.to(key.dtype)
            scores_lo = (grad_scores - scores_hi.to(tl.float32)).to(key.dtype)
            dq += tl.dot(scores_hi, key)
            dq += tl.dot(scores_lo, key)
    tl.store(DQ + base, dq.to(DQ.dtype.element_ty), query_valid[:, None])


@triton.jit(do_not_specialize=["q_len", "kv_len", "words_per_row", "query_blocks_per_row"])
def _qsa_fused_bwd_dkv(
    Q,
    K,
    V,
    DO,
    DK,
    DV,
    Lse,
    Delta,
    Positions,
    DocIds,
    Bits32,
    QueryIds,
    QueryCounts,
    QueryIndices,
    q_len,
    kv_len,
    words_per_row,
    query_blocks_per_row,
    stride_qm,
    stride_qh,
    stride_kn,
    stride_kh,
    scale,
    HQ: tl.constexpr,
    HK: tl.constexpr,
    DIM: tl.constexpr,
    RATIO: tl.constexpr,
    BLOCK: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    HAS_QUERY_IDS: tl.constexpr,
):
    start_n = tl.program_id(0) * BN
    kv_head = tl.program_id(1)
    offs_n = start_n + tl.arange(0, BN)
    offs_d = tl.arange(0, DIM)
    key_valid = offs_n < kv_len
    kv_base = offs_n[:, None] * stride_kn + kv_head * stride_kh + offs_d[None, :]
    key = tl.load(K + kv_base, key_valid[:, None], other=0.0)
    value = tl.load(V + kv_base, key_valid[:, None], other=0.0)
    dk = tl.zeros((BN, DIM), tl.float32)
    dv = tl.zeros((BN, DIM), tl.float32)

    classifier_block = start_n // BLOCK
    count = tl.load(QueryCounts + classifier_block)
    for index in range(count):
        block = tl.load(QueryIndices + classifier_block * query_blocks_per_row + index)
        for sub in tl.static_range(BLOCK // BM):
            offs_m = block * BLOCK + sub * BM + tl.arange(0, BM)
            query_valid, query_global, query_position, query_doc, tail_start = _query_metadata(
                Positions, DocIds, QueryIds, offs_m, q_len, RATIO, HAS_QUERY_IDS
            )
            key_position = tl.load(Positions + offs_n, key_valid, other=0).to(tl.int32)
            key_doc = tl.load(DocIds + offs_n, key_valid, other=-2).to(tl.int32)
            block_id = key_position // RATIO
            word = tl.load(
                Bits32 + query_global[:, None].to(tl.int64) * words_per_row + (block_id >> 5)[None, :],
                query_valid[:, None] & key_valid[None, :],
                other=0,
            )
            selected = ((word >> (block_id & 31)[None, :]) & 1) != 0
            admitted = (query_doc[:, None] == key_doc[None, :]) & (key_position[None, :] <= query_position[:, None])
            admitted &= selected | (key_position[None, :] >= tail_start[:, None])
            admitted &= query_valid[:, None] & key_valid[None, :]
            for group in tl.static_range(HQ // HK):
                head = kv_head * (HQ // HK) + group
                base = offs_m[:, None] * stride_qm + head * stride_qh + offs_d[None, :]
                query = tl.load(Q + base, query_valid[:, None], other=0.0)
                grad_out = tl.load(DO + base, query_valid[:, None], other=0.0)
                lse = tl.load(Lse + head * q_len + offs_m, query_valid, other=0.0)
                delta = tl.load(Delta + head * q_len + offs_m, query_valid, other=0.0)
                finite = lse != -float("inf")
                scores = tl.dot(query, tl.trans(key)) * scale
                probabilities = tl.where(
                    admitted & finite[:, None], tl.exp(scores - tl.where(finite, lse, 0.0)[:, None]), 0.0
                )
                dv += tl.dot(tl.trans(probabilities).to(grad_out.dtype), grad_out)
                grad_probabilities = tl.dot(grad_out, tl.trans(value))
                grad_scores = probabilities * (grad_probabilities - delta[:, None]) * scale
                dk += tl.dot(tl.trans(grad_scores).to(query.dtype), query)
    tl.store(DK + kv_base, dk.to(DK.dtype.element_ty), key_valid[:, None])
    tl.store(DV + kv_base, dv.to(DV.dtype.element_ty), key_valid[:, None])


@triton.jit(do_not_specialize=["inner"])
def _pack_touched(flags, counts, indices, inner, scan_width: tl.constexpr):
    """Compact the touched block ids of one classifier row, preserving
    order."""
    row = tl.program_id(0)
    columns = tl.arange(0, scan_width)
    valid = columns < inner
    touched = (tl.load(flags + row * inner + columns, mask=valid, other=0) != 0) & valid
    offsets = tl.cumsum(touched.to(tl.int32), 0) - 1
    tl.store(counts + row, tl.sum(touched.to(tl.int32), 0))
    tl.store(indices + row * inner + offsets, columns, mask=touched)


def _pack_both_directions(flags: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Per query block the touched KV blocks, and per KV block the touching
    query blocks."""
    query_blocks, kv_blocks = flags.shape
    kv_counts = torch.empty(query_blocks, device=flags.device, dtype=torch.int32)
    kv_indices = torch.zeros((query_blocks, kv_blocks), device=flags.device, dtype=torch.int32)
    _pack_touched[(query_blocks,)](
        flags, kv_counts, kv_indices, kv_blocks, triton.next_power_of_2(kv_blocks), num_warps=8
    )
    transposed = flags.t().contiguous()
    query_counts = torch.empty(kv_blocks, device=flags.device, dtype=torch.int32)
    query_indices = torch.zeros((kv_blocks, query_blocks), device=flags.device, dtype=torch.int32)
    _pack_touched[(kv_blocks,)](
        transposed, query_counts, query_indices, query_blocks, triton.next_power_of_2(query_blocks), num_warps=8
    )
    return kv_counts, kv_indices, query_counts, query_indices


def _select_tiles(dim: int, capability: tuple[int, int], dtype: torch.dtype) -> tuple[int, int, int, int]:
    """Query/key tiles for the forward-dq pass and for the dK/dV reduction.

    With 128 query rows, wide heads exceed Blackwell's tensor-memory budget
    even with one pipeline stage (D=256 forward and D=128 backward). Halve the
    query tile; dK/dV also needs the narrower key tile for its two
    accumulators.
    """
    # A 64x64 tile at DIM=256 requires 672 TMEM columns on SM103 (limit 512).
    if dim == 256 and capability == (10, 3) and dtype == torch.bfloat16:
        return 32, 32, 32, 32
    if dim == 128 and capability == (10, 3) and dtype == torch.bfloat16:
        return 64, 64, 64, 32
    if dim >= 256:
        return 128, 64, 64, 32
    return 128, 64, 128, 64


class _QSAFusedAttention(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        positions: Tensor,
        doc_ids: Tensor,
        bits32: Tensor,
        query_ids: Tensor | None,
        kv_counts: Tensor,
        kv_indices: Tensor,
        query_counts: Tensor,
        query_indices: Tensor,
        ratio: int,
        scale: float,
        save_residual: bool,
    ) -> Tensor:
        q_len, heads, dim = query.shape
        kv_len, kv_heads = key.shape[0], key.shape[1]
        ctx.tiles = _select_tiles(dim, torch.cuda.get_device_capability(query.device), query.dtype)
        forward_query_tile, forward_key_tile, dkv_query_tile, dkv_key_tile = ctx.tiles
        output = torch.empty_like(query)
        residual = torch.empty_like(query) if save_residual else None
        lse = torch.empty((heads, q_len), device=query.device, dtype=torch.float32)
        _qsa_fused_fwd[(triton.cdiv(q_len, forward_query_tile), heads)](
            query,
            key,
            value,
            output,
            lse,
            positions,
            doc_ids,
            bits32,
            query_ids,
            kv_counts,
            kv_indices,
            q_len,
            kv_len,
            bits32.shape[1],
            kv_indices.shape[1],
            heads * dim,
            dim,
            kv_heads * dim,
            dim,
            scale,
            heads,
            kv_heads,
            dim,
            ratio,
            _BLOCK,
            forward_query_tile,
            forward_key_tile,
            query_ids is not None,
            OutResidual=residual,
            num_warps=8,
            num_stages=_NUM_STAGES,
        )
        ctx.save_for_backward(
            query,
            key,
            value,
            positions,
            doc_ids,
            bits32,
            query_ids,
            kv_counts,
            kv_indices,
            query_counts,
            query_indices,
            output,
            residual,
            lse,
        )
        ctx.ratio = ratio
        ctx.scale = scale
        return output

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        (
            query,
            key,
            value,
            positions,
            doc_ids,
            bits32,
            query_ids,
            kv_counts,
            kv_indices,
            query_counts,
            query_indices,
            output,
            residual,
            lse,
        ) = ctx.saved_tensors
        if residual is None:
            raise RuntimeError("QSA backward requires the saved output residual")
        grad_output = grad_output.contiguous()
        q_len, heads, dim = query.shape
        kv_len, kv_heads = key.shape[0], key.shape[1]
        forward_query_tile, forward_key_tile, dkv_query_tile, dkv_key_tile = ctx.tiles
        delta = torch.empty_like(lse)
        _qsa_fused_delta[(triton.cdiv(q_len, forward_query_tile), heads)](
            output,
            grad_output,
            delta,
            q_len,
            heads * dim,
            dim,
            heads,
            dim,
            forward_query_tile,
            OutResidual=residual,
            num_warps=8,
        )
        dq = torch.empty_like(query)
        _qsa_fused_bwd_dq[(triton.cdiv(q_len, forward_query_tile), heads)](
            query,
            key,
            value,
            grad_output,
            dq,
            lse,
            delta,
            positions,
            doc_ids,
            bits32,
            query_ids,
            kv_counts,
            kv_indices,
            q_len,
            kv_len,
            bits32.shape[1],
            kv_indices.shape[1],
            heads * dim,
            dim,
            kv_heads * dim,
            dim,
            ctx.scale,
            heads,
            kv_heads,
            dim,
            ctx.ratio,
            _BLOCK,
            forward_query_tile,
            forward_key_tile,
            query_ids is not None,
            num_warps=8,
            num_stages=_NUM_STAGES,
        )
        dk = torch.empty_like(key)
        dv = torch.empty_like(value)
        _qsa_fused_bwd_dkv[(triton.cdiv(kv_len, dkv_key_tile), kv_heads)](
            query,
            key,
            value,
            grad_output,
            dk,
            dv,
            lse,
            delta,
            positions,
            doc_ids,
            bits32,
            query_ids,
            query_counts,
            query_indices,
            q_len,
            kv_len,
            bits32.shape[1],
            query_indices.shape[1],
            heads * dim,
            dim,
            kv_heads * dim,
            dim,
            ctx.scale,
            heads,
            kv_heads,
            dim,
            ctx.ratio,
            _BLOCK,
            dkv_query_tile,
            dkv_key_tile,
            query_ids is not None,
            num_warps=8,
            num_stages=_NUM_STAGES,
        )
        return dq, dk, dv, None, None, None, None, None, None, None, None, None, None, None


class QSAFusedPlan:
    """Routing metadata that only depends on the frozen selection, not on
    Q/K/V."""

    __slots__ = ("bits32", "kv_counts", "kv_indices", "query_counts", "query_indices", "ratio", "q_len")

    def __init__(self, bits32, kv_counts, kv_indices, query_counts, query_indices, ratio, q_len):
        self.bits32 = bits32
        self.kv_counts = kv_counts
        self.kv_indices = kv_indices
        self.query_counts = query_counts
        self.query_indices = query_indices
        self.ratio = ratio
        self.q_len = q_len


def build_plan(selection, sequence_length: int, query_ids: Tensor | None = None) -> QSAFusedPlan:
    """Classify 128x128 tiles once and compact them in both directions."""
    from relax.models.qwen3_8 import qsa_block_mask

    batch_size, selected_length = selection.positions.shape
    if batch_size != 1 or selected_length != sequence_length:
        raise ValueError("The fused QSA kernel expects a single packed row of the attention length")
    blocks_per_row = triton.cdiv(sequence_length, _BLOCK)
    q_len = sequence_length if query_ids is None else query_ids.numel()
    query_blocks = triton.cdiv(q_len, _BLOCK)
    flags = torch.empty((query_blocks, blocks_per_row), device=selection.positions.device, dtype=torch.uint8)
    qsa_block_mask._classify_blocks[(query_blocks * blocks_per_row,)](
        selection.doc_ids,
        selection.positions,
        selection.selected_bits,
        flags,
        sequence_length,
        blocks_per_row,
        selection.bits_per_row,
        _BLOCK,
        selection.compress_ratio,
        query_ids,
        q_len,
        query_ids is not None,
        num_warps=8,
    )
    kv_counts, kv_indices, query_counts, query_indices = _pack_both_directions(flags)
    bits = selection.selected_bits.view(-1, selection.bits_per_row)
    if selection.bits_per_row % 4:
        bits = torch.nn.functional.pad(bits, (0, -selection.bits_per_row % 4))
    bits32 = bits.contiguous().view(torch.int32)
    return QSAFusedPlan(bits32, kv_counts, kv_indices, query_counts, query_indices, selection.compress_ratio, q_len)


def qsa_fused_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    selection,
    plan: QSAFusedPlan,
    query_ids: Tensor | None,
    scale: float,
) -> Tensor:
    """Sparse QSA attention over packed THD ``[t, h, d]`` tensors."""
    if query.shape[0] != plan.q_len:
        raise ValueError("Fused QSA plan was built for a different local query length")
    if query.shape[-1] != key.shape[-1] or key.shape[:1] != value.shape[:1]:
        raise ValueError("Inconsistent fused QSA tensor shapes")
    if query.shape[1] % key.shape[1]:
        raise ValueError("Fused QSA requires the query heads to be a multiple of the KV heads")
    return _QSAFusedAttention.apply(
        query.contiguous(),
        key.contiguous(),
        value.contiguous(),
        selection.positions.view(-1),
        selection.doc_ids.view(-1),
        plan.bits32,
        query_ids,
        plan.kv_counts,
        plan.kv_indices,
        plan.query_counts,
        plan.query_indices,
        plan.ratio,
        scale,
        torch.is_grad_enabled() and any(t.requires_grad for t in (query, key, value)),
    )


def supports(selection, query, key, sparse_backend: str) -> bool:
    """The routed R=4 bitset contract this kernel is specialized for."""
    return (
        query.is_cuda
        and sparse_backend == "flex"
        and selection is not None
        and selection.selected_bits is not None
        and selection.compress_ratio == 4
        and selection.bits_per_row > 0
        and selection.positions.shape[0] == 1
        and query.shape[-1] in (64, 128, 256)
        and query.shape[-2] % key.shape[-2] == 0
    )
