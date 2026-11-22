# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Screen with Tensor Core score intervals and refine with the radix scorer."""

from typing import NamedTuple

import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice as lib

from relax.models.qwen3_8.ops.qsa_cute.dosa_kernels import _safe, _scores, _up, key_norm
from relax.models.qwen3_8.ops.qsa_cute.triton_qsa_select import (
    _MAX_ID_CODE,
    ID_BITS,
    _cudnn_kth_bits,
    _has_cudnn_topk,
    _qsa_select_score_kernel,
    _qsa_select_topk_kernel,
)


class IntervalConfig(NamedTuple):
    prefix_threshold: bool = False
    coarse_rows: int = 64
    coarse_keys: int = 128
    chunk_rows: int = 512
    compact_tile: int = 1024
    refine_tile: int = 128
    capacity: int = 2048
    half_scores: bool = False
    small_topk: bool = False
    boundary_bins: int = 0
    boundary_fused_refine: bool = False
    boundary_reuse_topk: bool = False


@triton.jit
def _query_error(Q, MAX_KEY_NORM, ERROR, HALF: tl.constexpr, LOCAL_HALF: tl.constexpr = False):
    row = tl.program_id(0)
    h, d = tl.arange(0, 4), tl.arange(0, 128)
    q = tl.load(Q + row * 512 + h[:, None] * 128 + d[None, :]).to(tl.float32)
    norms = lib.sqrt_ru(tl.reduce(lib.mul_ru(q, q), 1, _up))
    finite = tl.sum((~_safe(q)).to(tl.int32)) == 0
    # Bound both FP32 dot paths, both head reductions and final scaling.
    # For safe BF16 operands, products are exactly representable in FP32.
    norm_product = lib.mul_ru(tl.reduce(norms, 0, _up), tl.load(MAX_KEY_NORM))
    score_bound = lib.mul_ru(norm_product, 128**-0.5)
    if HALF:
        # Rows that could overflow FP16 refine without pruning.
        finite &= score_bound < 65000.0
    error = lib.mul_ru(score_bound, 2.0**-12)
    if HALF and not LOCAL_HALF:
        # The ordered selector uses a uniform bound for FP16 scratch storage.
        error = lib.add_ru(lib.mul_ru(score_bound, 2.0**-10), 2.0**-24)
    tl.store(ERROR + row, tl.where(finite & (error < float("inf")), error, float("inf")))


@triton.jit
def _coarse_scores(Q, K, VISIBLE, SCORES, ROWS, BLOCKS, BM: tl.constexpr, BN: tl.constexpr):
    q_rows = tl.program_id(0) * BM + tl.arange(0, BM)
    ids = tl.program_id(1) * BN + tl.arange(0, BN)
    vis = tl.load(VISIBLE + q_rows // 4, q_rows < ROWS * 4, 0)
    if tl.program_id(1) * BN >= tl.max(vis, 0):
        return
    d = tl.arange(0, 32)
    acc = tl.full((BM, BN), 0, tl.float32)
    for start in range(0, 128, 32):
        q = tl.load(Q + q_rows[:, None] * 128 + start + d[None, :], q_rows[:, None] < ROWS * 4, 0)
        k = tl.load(K + ids[None, :] * 128 + start + d[:, None], ids[None, :] < BLOCKS, 0)
        acc = tl.dot(q, k, acc)
    head_scores = tl.maximum(acc, 0.0).reshape((BM // 4, 4, BN))
    scores = tl.sum(head_scores, 1) * (128**-0.5)
    rows = tl.program_id(0) * (BM // 4) + tl.arange(0, BM // 4)
    vis = tl.load(VISIBLE + rows, rows < ROWS, 0)
    scores = tl.where(scores < float("inf"), scores, 0.0)
    tl.store(SCORES + rows[:, None] * BLOCKS + ids[None, :], scores, ids[None, :] < vis[:, None])


@triton.jit
def _compact_intervals(
    SCORES, KTH_BITS, ERROR, VISIBLE, IDS, LENGTHS, FALLBACK, N, CAP, K, TILE: tl.constexpr, PREFIX: tl.constexpr
):
    row = tl.program_id(0)
    vis = tl.load(VISIBLE + row)
    error = tl.load(ERROR + row)
    cols = tl.arange(0, TILE)
    if PREFIX:
        # The lower edge of the approximate kth score's high-16-bit bin
        # is a sufficient witness. Only two radix passes are needed.
        prefix = vis * 0
        rank = prefix + K
        if (vis > K) & (error < float("inf")):
            bins = tl.arange(0, 256)
            for i in tl.static_range(2):
                shift = 24 - 8 * i
                histogram = tl.zeros((256,), tl.int32)
                for start in range(0, vis, TILE):
                    ids = start + cols
                    valid = ids < vis
                    bits = tl.load(SCORES + row * N + ids, valid, 0.0).to(tl.float32).to(tl.int32, bitcast=True)
                    if i:
                        valid = valid & ((bits >> 24) == (prefix >> 24))
                    histogram += tl.histogram((bits >> shift) & 255, 256, mask=valid)
                suffix = tl.cumsum(histogram, 0, reverse=True)
                digit = tl.max(tl.where(suffix >= rank, bins, 0))
                rank -= tl.sum(tl.where(bins > digit, histogram, 0))
                prefix |= digit << shift
        kth = prefix.to(tl.float32, bitcast=True)
    else:
        kth = tl.load(KTH_BITS + row).to(tl.float32, bitcast=True)
    tau = lib.add_rd(kth, -error)
    count = vis * 0
    fallback = (error == float("inf")) | (vis > CAP) & ((vis <= K) | (tau <= 0))
    if not fallback:
        for start in range(0, vis, TILE):
            if count <= CAP:
                ids = start + cols
                valid = ids < vis
                score = tl.load(SCORES + row * N + ids, valid, 0.0).to(tl.float32)
                live = valid & ((vis <= K) | (lib.add_ru(score, error) >= tau))
                pos = count + tl.cumsum(live.to(tl.int32)) - 1
                tl.store(IDS + row * CAP + pos, ids, live & (pos < CAP))
                count += tl.sum(live.to(tl.int32))
        fallback = (count > CAP) | (count < tl.minimum(K, vis))
    tl.store(FALLBACK + row, tl.where(fallback, vis, 0))
    tl.store(LENGTHS + row, tl.where(fallback, vis, count))


@triton.jit
def _refine_intervals(Q, K, IDS, LENGTHS, FALLBACK, SCORES, STRIDE, CAP, TILE: tl.constexpr):
    row, tile = tl.program_id(0), tl.program_id(1)
    length = tl.load(LENGTHS + row)
    if (tl.load(FALLBACK + row) == 0) & (tile * TILE < length):
        cols = tile * TILE + tl.arange(0, TILE)
        ids = tl.load(IDS + row * CAP + cols, cols < length, 0)
        scores = _scores(Q, K, row, ids, cols < length, TILE)
        tl.store(SCORES + row * STRIDE + cols, scores, cols < length)


@triton.jit
def _small_topk(
    SCORES,
    LENGTHS,
    FALLBACK,
    CODES,
    STRIDE,
    CAP: tl.constexpr,
    KP: tl.constexpr,
    IDB: tl.constexpr,
    MAX_ID: tl.constexpr,
):
    row = tl.program_id(0)
    if tl.load(FALLBACK + row) == 0:
        cols = tl.arange(0, CAP)
        valid = cols < tl.load(LENGTHS + row)
        scores = tl.load(SCORES + row * STRIDE + cols, valid, 0.0)
        bits = scores.to(tl.int32, bitcast=True).to(tl.int64)
        # Compaction follows key ID order, so ordinal ties are original ID ties.
        code = ((bits + 1) << IDB) | (MAX_ID - cols).to(tl.int64)
        best = tl.topk(tl.where(valid, code, 0), KP)
        tl.store(CODES + row * KP + tl.arange(0, KP), best)


@triton.jit
def _decode(CODES, IDS, FALLBACK, OUTPUT, CAP, K: tl.constexpr, KP: tl.constexpr, MAX_ID: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, KP)
    codes = tl.load(CODES + row * KP + cols)
    offset = (MAX_ID - (codes & MAX_ID)).to(tl.int32)
    if tl.load(FALLBACK + row) == 0:
        offset = tl.load(IDS + row * CAP + offset, (codes > 0) & (offset < CAP), -1)
    tl.store(OUTPUT + row * K + cols, tl.where(codes > 0, offset, -1), cols < K)


def select_interval_ids(
    query: torch.Tensor,
    keys: torch.Tensor,
    visible: torch.Tensor,
    budget: int,
    *,
    stats: dict[str, torch.Tensor] | None = None,
    config: IntervalConfig = IntervalConfig(),
    boundary_only: bool = False,
) -> torch.Tensor | None:
    """Screen score intervals and optionally return unordered top-k IDs."""
    bins = config.boundary_bins if boundary_only else 0
    reuse_topk = boundary_only and config.boundary_reuse_topk
    if (
        query.dtype != torch.bfloat16
        or keys.dtype != torch.bfloat16
        or query.shape[1:] != (4, 128)
        or keys.shape[1] != 128
        or not 4096 < keys.shape[0] < (1 << ID_BITS)
        or not 0 < budget <= 512
        or torch.version.hip is not None
        or torch.cuda.get_device_capability(query.device) != (10, 3)
        or (not config.prefix_threshold and not bins and not _has_cudnn_topk())
    ):
        return None
    if config.half_scores and not (config.prefix_threshold or bins or reuse_topk):
        raise ValueError("FP16 score storage requires a verified threshold")
    if bins not in (0, 512, 1024, 2048, 4096):
        raise ValueError("Unsupported DoSA boundary histogram size")
    if bins and reuse_topk:
        raise ValueError("Choose either a boundary histogram or cuDNN indices")
    if boundary_only and config.prefix_threshold:
        return None  # A truncated threshold cannot certify membership in both directions.
    if boundary_only:
        from relax.models.qwen3_8.ops.qsa_cute.dosa_boundary import compact_boundary, finish_boundary
    if reuse_topk:
        from cudnn import DSA

    rows, blocks = len(query), len(keys)
    # Match the existing radix backend's 512 MiB scratch limit.
    chunk = min(rows, config.chunk_rows, (1 << 27) // blocks)
    capacity = min(blocks, config.capacity)
    # IEEE refinement uses the same 64/128-wide accumulation as dense radix.
    fused_refine = boundary_only and config.boundary_fused_refine and 64 <= capacity <= 128
    kp = triton.next_power_of_2(budget)
    use_small_topk = (
        not boundary_only and config.small_topk and capacity >= kp and capacity == triton.next_power_of_2(capacity)
    )
    query, keys = query.contiguous(), keys.contiguous()
    visible = visible.int().contiguous()
    norms = torch.empty(blocks, device=keys.device)
    error = torch.empty(rows, device=keys.device)
    scores = torch.empty((chunk, blocks), device=keys.device)
    # The compacted IDs have no dependency on coarse scores after screening.
    # Reuse the same storage for final FP32 scores on the current stream.
    coarse = (
        scores.view(torch.float16).view(-1)[: chunk * blocks].view(chunk, blocks) if config.half_scores else scores
    )
    ids = torch.empty((chunk, capacity), dtype=torch.int32, device=keys.device)
    lengths = torch.empty(chunk, dtype=torch.int32, device=keys.device)
    fallback = torch.empty_like(lengths)
    codes = torch.empty((chunk, kp), dtype=torch.int64, device=keys.device)
    output = torch.empty((rows, budget), dtype=torch.int32, device=keys.device)
    if boundary_only:
        certified = torch.empty_like(lengths)
    if stats is not None:
        stats["refined"] = torch.empty_like(visible)
        stats["fallback"] = torch.empty_like(visible)
        if boundary_only:
            stats["certified"] = torch.empty_like(visible)

    key_norm[(triton.cdiv(blocks, 32),)](keys, norms, blocks, num_warps=4, enable_fp_fusion=False)
    _query_error[(rows,)](
        query, norms.max(), error, config.half_scores, boundary_only, num_warps=4, enable_fp_fusion=False
    )

    for start in range(0, rows, chunk):
        stop = min(start + chunk, rows)
        q, vis = query[start:stop], visible[start:stop]
        count = stop - start
        tiles = triton.cdiv(blocks, 128)
        _coarse_scores[(triton.cdiv(count * 4, config.coarse_rows), triton.cdiv(blocks, config.coarse_keys))](
            q, keys, vis, coarse, count, blocks, config.coarse_rows, config.coarse_keys
        )
        if reuse_topk:
            kth = DSA.indexer_top_k_wrapper(coarse[:count], vis, budget, return_val=False)["indices"]
        else:
            kth = vis if (config.prefix_threshold or bins) else _cudnn_kth_bits(scores[:count], vis, budget)
        if boundary_only:
            compact_boundary[(count,)](
                coarse,
                kth,
                error[start:stop],
                vis,
                ids,
                lengths,
                fallback,
                certified,
                output[start:stop],
                blocks,
                capacity,
                budget,
                kp,
                config.compact_tile,
                max(0, bins.bit_length() - 1),
                config.half_scores,
                reuse_topk,
                num_warps=4,
                enable_fp_fusion=False,
            )
        else:
            _compact_intervals[(count,)](
                coarse,
                kth,
                error[start:stop],
                vis,
                ids,
                lengths,
                fallback,
                blocks,
                capacity,
                budget,
                config.compact_tile,
                config.prefix_threshold,
                num_warps=4,
                enable_fp_fusion=False,
            )
        if not fused_refine:
            _refine_intervals[(count, triton.cdiv(capacity, config.refine_tile))](
                q,
                keys,
                ids,
                lengths,
                fallback,
                scores,
                blocks,
                capacity,
                config.refine_tile,
                num_warps=4,
                enable_fp_fusion=False,
            )
        _qsa_select_score_kernel.fn[(triton.cdiv(count, 64), tiles)](
            q,
            keys,
            scores,
            fallback,
            vis,
            count,
            blocks,
            blocks,
            128**-0.5,
            NH=4,
            DI=128,
            BM=64,
            BN=128,
            BK=16,
            HAS_LO=False,
            num_warps=4,
            enable_fp_fusion=False,
        )
        selection_lengths = fallback if (boundary_only or use_small_topk) else lengths
        _qsa_select_topk_kernel[(count,)](
            scores,
            selection_lengths,
            codes,
            selection_lengths,
            selection_lengths,
            budget,
            blocks,
            _MAX_ID_CODE,
            TILE=2048,
            KP=kp,
            IDB=ID_BITS,
            HAS_LO=False,
            GIVEN_THR=False,
            num_warps=8,
        )
        if boundary_only:
            finish_boundary[(count,)](
                q,
                keys,
                scores,
                codes,
                ids,
                lengths,
                fallback,
                certified,
                output[start:stop],
                blocks,
                capacity,
                budget,
                kp,
                triton.next_power_of_2(capacity),
                ID_BITS,
                _MAX_ID_CODE,
                fused_refine,
                num_warps=4,
                enable_fp_fusion=False,
            )
        else:
            if use_small_topk:
                _small_topk[(count,)](
                    scores, lengths, fallback, codes, blocks, capacity, kp, ID_BITS, _MAX_ID_CODE, num_warps=8
                )
            _decode[(count,)](codes, ids, fallback, output[start:stop], capacity, budget, kp, _MAX_ID_CODE)
        if stats is not None:
            stats["refined"][start:stop] = lengths[:count] + fallback[:count] if boundary_only else lengths[:count]
            stats["fallback"][start:stop] = fallback[:count]
            if boundary_only:
                stats["certified"][start:stop] = certified[:count]
    return output
