# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Bounded FP32 dS producer and consumers for the SM103 v3 backward path."""

import triton
import triton.language as tl

from relax.models.qwen3_8.ops.qsa_fused_attention import _query_metadata
from relax.models.qwen3_8.ops.qsa_triton_v2.kernels import _admitted


@triton.jit
def _prefix_counts(Counts, Prefix, rows: tl.constexpr, WIDTH: tl.constexpr):
    row = tl.arange(0, WIDTH)
    count = tl.load(Counts + row, row < rows, other=0).to(tl.int64)
    prefix = tl.cumsum(count, 0)
    tl.store(Prefix + row + 1, prefix, row < rows)
    tl.store(Prefix, 0)


@triton.jit
def _edge_lookup(Counts, Indices, Prefix, EdgeMap, columns: tl.constexpr, WIDTH: tl.constexpr):
    row = tl.program_id(0)
    slot = tl.arange(0, WIDTH)
    count = tl.load(Counts + row)
    block = tl.load(Indices + row.to(tl.int64) * columns + slot, slot < count, other=0)
    edge = tl.load(Prefix + row) + slot.to(tl.int64)
    tl.store(EdgeMap + row.to(tl.int64) * columns + block, edge.to(tl.int32), slot < count)


@triton.jit(do_not_specialize=["q_len", "kv_len", "words_per_row", "query_blocks_per_row"])
def _shared_dkv(
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
    DS,
    EdgeMap,
    Prefix,
    capacity_edges,
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
    query_blocks = tl.cdiv(q_len, BLOCK)
    kv_blocks = tl.cdiv(kv_len, BLOCK)
    fits = tl.load(Prefix + query_blocks) <= capacity_edges
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
                if fits:
                    edge = tl.load(EdgeMap + block.to(tl.int64) * kv_blocks + classifier_block).to(tl.int64)
                    address = (
                        (edge * HQ + head.to(tl.int64)) * BLOCK * BLOCK
                        + (offs_n[:, None] % BLOCK).to(tl.int64) * BLOCK
                        + (offs_m[None, :] % BLOCK).to(tl.int64)
                    )
                    tl.store(DS + address, grad_scores, key_valid[:, None] & query_valid[None, :])
    tl.store(DK + kv_base, dk.to(DK.dtype.element_ty), key_valid[:, None])
    tl.store(DV + kv_base, dv.to(DV.dtype.element_ty), key_valid[:, None])


@triton.jit(do_not_specialize=["q_len", "kv_len", "words_per_row", "blocks_per_row"])
def _fit_dq(
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
    DS,
    EdgeMap,
    Prefix,
    capacity_edges,
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
    query_blocks = tl.cdiv(q_len, BLOCK)
    fits = tl.load(Prefix + query_blocks) <= capacity_edges
    if fits:
        start_m = tl.program_id(0) * BM
        head = tl.program_id(1)
        kv_head = head // (HQ // HK)
        offs_m = start_m + tl.arange(0, BM)
        offs_d = tl.arange(0, DIM)
        query_valid = offs_m < q_len
        dq = tl.zeros((BM, DIM), tl.float32)
        classifier_block = start_m // BLOCK
        count = tl.load(BlockCounts + classifier_block)
        for index in range(count):
            block = tl.load(BlockIndices + classifier_block * blocks_per_row + index)
            edge = tl.load(EdgeMap + classifier_block.to(tl.int64) * blocks_per_row + block).to(tl.int64)
            for sub in range(BLOCK // BN):
                offs_n = block * BLOCK + sub * BN + tl.arange(0, BN)
                key_valid = offs_n < kv_len
                kv_base = offs_n[:, None].to(tl.int64) * stride_kn + kv_head * stride_kh + offs_d[None, :]
                key = tl.load(K + kv_base, key_valid[:, None], other=0.0)
                address = (
                    (edge * HQ + head.to(tl.int64)) * BLOCK * BLOCK
                    + (offs_n[:, None] % BLOCK).to(tl.int64) * BLOCK
                    + (offs_m[None, :] % BLOCK).to(tl.int64)
                )
                grad_scores = tl.trans(tl.load(DS + address, key_valid[:, None] & query_valid[None, :], other=0.0))
                scores_hi = grad_scores.to(key.dtype)
                scores_lo = (grad_scores - scores_hi.to(tl.float32)).to(key.dtype)
                dq += tl.dot(scores_hi, key)
                dq += tl.dot(scores_lo, key)
        base = offs_m[:, None].to(tl.int64) * stride_qm + head * stride_qh + offs_d[None, :]
        tl.store(DQ + base, dq.to(DQ.dtype.element_ty), query_valid[:, None])


@triton.jit(do_not_specialize=["q_len", "kv_len", "words_per_row", "blocks_per_row"])
def _fallback_dq(
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
    DS,
    EdgeMap,
    Prefix,
    capacity_edges,
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
    query_blocks = tl.cdiv(q_len, BLOCK)
    fits = tl.load(Prefix + query_blocks) <= capacity_edges
    if not fits:
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
                scores_hi = grad_scores.to(key.dtype)
                scores_lo = (grad_scores - scores_hi.to(tl.float32)).to(key.dtype)
                dq += tl.dot(scores_hi, key)
                dq += tl.dot(scores_lo, key)
        tl.store(DQ + base, dq.to(DQ.dtype.element_ty), query_valid[:, None])
