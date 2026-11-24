# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Top-k membership certificates for consumers that accept unordered IDs."""

import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice as lib

from relax.models.qwen3_8.ops.qsa_cute.score_bound_kernels import _scores


@triton.jit
def _histogram_bounds(SCORES, vis, error, K, TILE: tl.constexpr, LOG_BINS: tl.constexpr):
    # A binary bucket width makes the score-to-bucket scaling exact for
    # normal scores. The error already has a per-row estimate of score scale.
    BINS: tl.constexpr = 1 << LOG_BINS
    scale = error * 4096.0
    exponent = (scale.to(tl.int32, bitcast=True) >> 23) + 1 - LOG_BINS
    lo, hi = 0.0, 0.0
    usable = (exponent > 0) & (exponent < 254)
    if usable:
        width = (exponent << 23).to(tl.float32, bitcast=True)
        inverse = ((254 - exponent) << 23).to(tl.float32, bitcast=True)
        hist = tl.zeros((BINS,), tl.int32)
        cols = tl.arange(0, TILE)
        for tile in range(0, vis, TILE):
            idx = tile + cols
            valid = idx < vis
            score = tl.load(SCORES + idx, valid, 0.0).to(tl.float32)
            bucket = tl.minimum(score * inverse, BINS - 1).to(tl.int32)
            hist += tl.histogram(bucket, BINS, mask=valid)
        suffix = tl.cumsum(hist, 0, reverse=True)
        selected = tl.max(tl.where(suffix >= K, tl.arange(0, BINS), -1))
        # The top bucket also contains values clipped to the histogram range.
        usable = (selected >= 0) & (selected < BINS - 1)
        lo = selected.to(tl.float32) * width
        hi = (selected + 1).to(tl.float32) * width
    return lo, hi, usable


@triton.jit
def _coarse_cutoffs(low_score, high_score, error, HALF: tl.constexpr):
    # Convert certified bounds on the exact kth score into cutoffs on the
    # approximate scores. Strict comparisons keep every possible exact tie.
    if HALF:
        # FP16 rounding is bounded by one ulp of the stored score, with an
        # absolute allowance for subnormals. Its affine envelopes are monotone.
        radius = lib.add_ru(error, 2.0**-24)
        small: tl.constexpr = 1.0 - 2.0**-10
        large: tl.constexpr = 1.0 + 2.0**-10
        lower = lib.add_rd(lib.mul_rd(low_score, small), -radius)
        upper = lib.add_ru(lib.mul_ru(high_score, large), radius)
        drop = lib.div_rd(lib.add_rd(lower, -radius), large)
        keep = lib.div_ru(lib.add_ru(upper, radius), small)
    else:
        drop = lib.add_rd(lib.add_rd(low_score, -error), -error)
        keep = lib.add_ru(lib.add_ru(high_score, error), error)
    return drop, keep


@triton.jit
def _reuse_certified_ids(
    SCORES,
    TOPK_IDS,
    OUTPUT,
    vis,
    error,
    K: tl.constexpr,
    KP: tl.constexpr,
    HALF: tl.constexpr,
):
    cols = tl.arange(0, KP)
    ids = tl.load(TOPK_IDS + cols, cols < K, -1).to(tl.int32)
    valid = (ids >= 0) & (ids < vis) & (cols < K)
    scores = tl.load(SCORES + ids, valid, float("inf")).to(tl.float32)
    kth = tl.min(scores, 0)
    drop, keep = _coarse_cutoffs(kth, kth, error, HALF)
    sure = valid & (scores > keep)
    ordered = tl.sort(tl.where(sure, ids, vis), descending=False)
    size = tl.sum(sure.to(tl.int32), 0)
    next_id = tl.gather(ordered, tl.minimum(cols + 1, KP - 1), 0)
    duplicate = tl.sum(((cols + 1 < size) & (ordered == next_id)).to(tl.int32), 0) > 0
    usable = (tl.sum(valid.to(tl.int32), 0) == K) & (kth >= 0.0) & (keep < float("inf")) & ~duplicate
    tl.store(OUTPUT + cols, ordered, usable & (cols < size))
    return kth, drop, keep, size, usable


@triton.jit
def compact_boundary(
    SCORES,
    THRESHOLD_INPUT,
    ERROR,
    VISIBLE,
    IDS,
    LENGTHS,
    FALLBACK,
    CERTIFIED,
    OUTPUT,
    N,
    CAP,
    K: tl.constexpr,
    KP: tl.constexpr,
    TILE: tl.constexpr,
    LOG_BINS: tl.constexpr,
    HALF: tl.constexpr,
    REUSE_TOPK: tl.constexpr,
):
    row = tl.program_id(0)
    vis = tl.load(VISIBLE + row)
    error = tl.load(ERROR + row)
    certain, border, above, at_least = 0, 0, 0, 0
    fallback = error == float("inf")
    if not fallback:
        if vis <= K:
            output_cols = tl.arange(0, KP)
            tl.store(OUTPUT + row * K + output_cols, output_cols, output_cols < vis)
            certain = vis
        else:
            if LOG_BINS:
                low_score, high_score, usable = _histogram_bounds(SCORES + row * N, vis, error, K, TILE, LOG_BINS)
                drop, keep = _coarse_cutoffs(low_score, high_score, error, HALF)
                fallback = ~usable
            elif REUSE_TOPK:
                kth, drop, keep, from_topk, usable = _reuse_certified_ids(
                    SCORES + row * N, THRESHOLD_INPUT + row * K, OUTPUT + row * K, vis, error, K, KP, HALF
                )
                fallback = ~usable
            else:
                kth = tl.load(THRESHOLD_INPUT + row).to(tl.float32, bitcast=True)
                drop, keep = _coarse_cutoffs(kth, kth, error, HALF)
                fallback = kth < 0.0
            fallback |= ~(keep < float("inf")) | ((drop <= 0.0) & (vis > CAP + K))
            cols = tl.arange(0, TILE)
            if not fallback:
                for tile in range(0, vis, TILE):
                    if (border <= CAP) & (certain <= K):
                        ids = tile + cols
                        valid = ids < vis
                        scores = tl.load(SCORES + row * N + ids, valid, 0.0).to(tl.float32)
                        # The exact kth score lies inside the kth coarse interval.
                        # Equality at either end stays eligible for exact tie-breaking.
                        sure = valid & (scores > keep)
                        ambiguous = valid & ~sure & (scores >= drop)
                        sure_i, ambiguous_i = sure.to(tl.int32), ambiguous.to(tl.int32)
                        if not REUSE_TOPK:
                            sure_pos = certain + tl.cumsum(sure_i, 0) - 1
                            tl.store(OUTPUT + row * K + sure_pos, ids, sure & (sure_pos < K))
                        border_pos = border + tl.cumsum(ambiguous_i, 0) - 1
                        tl.store(IDS + row * CAP + border_pos, ids, ambiguous & (border_pos < CAP))
                        certain += tl.sum(sure_i, 0)
                        border += tl.sum(ambiguous_i, 0)
                        if not LOG_BINS:
                            above += tl.sum((valid & (scores > kth)).to(tl.int32), 0)
                            at_least += tl.sum((valid & (scores >= kth)).to(tl.int32), 0)
                fallback = (certain > K) | (border > CAP) | (certain + border < K)
                if not LOG_BINS:
                    # Verify the threshold hint against all visible coarse scores.
                    fallback |= (above >= K) | (at_least < K)
                if REUSE_TOPK:
                    # Only keep the emitted IDs when they cover every proven winner.
                    fallback |= from_topk != certain
    tl.store(CERTIFIED + row, tl.where(fallback, 0, certain))
    tl.store(LENGTHS + row, tl.where(fallback, 0, border))
    tl.store(FALLBACK + row, tl.where(fallback, vis, 0))


@triton.jit
def _finish_with_scores(
    QUERY,
    KEYS,
    SCORES,
    IDS,
    OUTPUT,
    row,
    length,
    certain,
    N,
    CAP,
    K: tl.constexpr,
    WIDTH: tl.constexpr,
    IDB: tl.constexpr,
    MAX_ID: tl.constexpr,
    FUSED: tl.constexpr,
):
    cols = tl.arange(0, WIDTH)
    valid = cols < length
    if FUSED:
        ids = tl.load(IDS + row * CAP + cols, valid, 0)
        scores = _scores(QUERY, KEYS, row, ids, valid, WIDTH)
    else:
        scores = tl.load(SCORES + row * N + cols, valid, 0.0)
    bits = scores.to(tl.int32, bitcast=True).to(tl.int64)
    # Boundary IDs were compacted in ascending order. Use the same score/ID
    # tie-break as dense radix when the boundary contains more than K-certain.
    codes = ((bits + 1) << IDB) | (MAX_ID - cols).to(tl.int64)
    best = tl.sort(tl.where(valid, codes, 0), descending=True)
    offset = (MAX_ID - (best & MAX_ID)).to(tl.int32)
    write = (cols + certain < K) & (best > 0)
    ids = tl.load(IDS + row * CAP + offset, write, -1)
    tl.store(OUTPUT + row * K + certain + cols, ids, write)


@triton.jit
def finish_boundary(
    QUERY,
    KEYS,
    SCORES,
    CODES,
    IDS,
    LENGTHS,
    FALLBACK,
    CERTIFIED,
    OUTPUT,
    N,
    CAP,
    K: tl.constexpr,
    KP: tl.constexpr,
    CP: tl.constexpr,
    IDB: tl.constexpr,
    MAX_ID: tl.constexpr,
    FUSED: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, KP)
    if tl.load(FALLBACK + row) > 0:
        codes = tl.load(CODES + row * KP + cols)
        ids = (MAX_ID - (codes & MAX_ID)).to(tl.int32)
        tl.store(OUTPUT + row * K + cols, tl.where(codes > 0, ids, -1), cols < K)
    else:
        certain = tl.load(CERTIFIED + row)
        length = tl.load(LENGTHS + row)
        if length > 0:
            # Most rows have only a few ambiguous candidates.
            if length <= 64:
                _finish_with_scores(
                    QUERY, KEYS, SCORES, IDS, OUTPUT, row, length, certain, N, CAP, K, 64, IDB, MAX_ID, FUSED
                )
            else:
                _finish_with_scores(
                    QUERY, KEYS, SCORES, IDS, OUTPUT, row, length, certain, N, CAP, K, CP, IDB, MAX_ID, FUSED
                )
        else:
            tl.store(OUTPUT + row * K + cols, -1, (cols >= certain) & (cols < K))
