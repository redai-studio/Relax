# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""V2 kernels adapted from the standalone QSA triton_v2 experiment.

Keep sub-tile and GQA-head loops rolled; broadcast selection words; compute dKV
with KV-major intermediates. Selection semantics and deterministic ownership
are preserved. Tile changes can change floating-point rounding. The baseline
source is intentionally untouched for side-by-side comparisons.
"""

import triton
import triton.language as tl

from relax.models.qwen3_8.ops.qsa_fused_attention import _query_metadata


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
    """Broadcast up to two selection words per row instead of a 2-D gather.

    A normal contiguous key tile spans one or two bitset words. Packed document
    boundaries with widely separated word IDs use the exact general gather. The
    fast path depends on observed word IDs, not document alignment.
    """
    offs_n = start_n + tl.arange(0, BN)
    key_valid = offs_n < kv_len
    key_position = tl.load(Positions + offs_n, key_valid, other=0).to(tl.int32)
    key_doc = tl.load(DocIds + offs_n, key_valid, other=-2).to(tl.int32)
    block_id = key_position // RATIO
    word_id = block_id >> 5
    low = tl.min(tl.where(key_valid, word_id, 2147483647), 0)
    high = tl.max(tl.where(key_valid, word_id, -1), 0)
    row = query_global.to(tl.int64) * words_per_row
    if high - low <= 1:
        first = tl.load(Bits32 + row + low, query_valid & (low < words_per_row), other=0)
        second = tl.load(Bits32 + row + low + 1, query_valid & (high > low) & (low + 1 < words_per_row), other=0)
        word = tl.where(word_id[None, :] == low, first[:, None], second[:, None])
    else:
        word = tl.load(Bits32 + row[:, None] + word_id[None, :], query_valid[:, None] & key_valid[None, :], other=0)
    selected = ((word >> (block_id & 31)[None, :]) & 1) != 0
    admitted = (query_doc[:, None] == key_doc[None, :]) & (key_position[None, :] <= query_position[:, None])
    admitted &= selected | (key_position[None, :] >= tail_start[:, None])
    return admitted & query_valid[:, None] & key_valid[None, :], offs_n, key_valid


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
        for sub in range(BLOCK // BN):
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
            # Retain the low part before cancellation-sensitive backward delta.
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
def _qsa_fused_delta_residual(O, OutResidual, DO, Delta, q_len, HQ: tl.constexpr, DIM: tl.constexpr, BM: tl.constexpr):
    offs_m = tl.program_id(0) * BM + tl.arange(0, BM)
    head = tl.program_id(1)
    offs_d = tl.arange(0, DIM)
    valid = offs_m < q_len
    offset = offs_m[:, None] * HQ * DIM + head * DIM + offs_d[None, :]
    output = tl.load(O + offset, valid[:, None], other=0.0).to(tl.float32)
    residual = tl.load(OutResidual + offset, valid[:, None], other=0.0).to(tl.float32)
    grad = tl.load(DO + offset, valid[:, None], other=0.0).to(tl.float32)
    tl.store(Delta + head * q_len + offs_m, tl.sum((output + residual) * grad, 1), valid)


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
        for sub in range(BLOCK // BN):
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
            # One BF16 rounding of dS can erase the small residual after dP - delta.
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
        for sub in range(BLOCK // BM):
            offs_m = block * BLOCK + sub * BM + tl.arange(0, BM)
            query_valid, query_global, query_position, query_doc, tail_start = _query_metadata(
                Positions, DocIds, QueryIds, offs_m, q_len, RATIO, HAS_QUERY_IDS
            )
            admitted, _, _ = _admitted(
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
            for group in range(HQ // HK):
                head = kv_head * (HQ // HK) + group
                base = offs_m[:, None] * stride_qm + head * stride_qh + offs_d[None, :]
                query = tl.load(Q + base, query_valid[:, None], other=0.0)
                grad_out = tl.load(DO + base, query_valid[:, None], other=0.0)
                lse = tl.load(Lse + head * q_len + offs_m, query_valid, other=0.0)
                delta = tl.load(Delta + head * q_len + offs_m, query_valid, other=0.0)
                finite = lse != -float("inf")
                # Keep score/probability tiles KV-major, matching dK/dV.
                scores = tl.dot(key, tl.trans(query)) * scale
                probabilities = tl.where(
                    tl.trans(admitted) & finite[None, :], tl.exp(scores - tl.where(finite, lse, 0.0)[None, :]), 0.0
                )
                dv += tl.dot(probabilities.to(grad_out.dtype), grad_out)
                grad_probabilities = tl.dot(value, tl.trans(grad_out))
                grad_scores = probabilities * (grad_probabilities - delta[None, :]) * scale
                dk += tl.dot(grad_scores.to(query.dtype), query)
    tl.store(DK + kv_base, dk.to(DK.dtype.element_ty), key_valid[:, None])
    tl.store(DV + kv_base, dv.to(DV.dtype.element_ty), key_valid[:, None])
