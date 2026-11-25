# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""QSA's positive-head DoSA certificate and bounded candidate refinement.

Adapted from Keye's directed-rounding certificate (DoSA Eqs. 4–7). Scores
retain PR7553's FP32 IEEE tile/head-reduction order, without Keye's BF16 score
rounding. All candidate counts and fallback decisions stay on GPU.
"""

import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice as lib


@triton.jit
def _up(a, b):
    return lib.add_ru(a, b)


@triton.jit
def _down(a, b):
    return lib.add_rd(a, b)


@triton.jit
def _norm(x):
    return lib.sqrt_ru(tl.reduce(lib.mul_ru(x, x), 0, _up))


@triton.jit
def _safe(x):
    a = tl.abs(x.to(tl.float32))
    return (a == 0) | ((a >= 2.0**-32) & (a <= 2.0**16))


@triton.jit
def prepare(Q, ANCHOR, FOLD, ERROR):
    row = tl.program_id(0)
    h, d = tl.arange(0, 4), tl.arange(0, 128)
    q = tl.load(Q + row * 512 + h[:, None] * 128 + d[None, :]).to(tl.float32)
    order = tl.sum(q * q, 1)
    anchor_id = tl.argmax(order, 0, tie_break_left=True)
    anchor = tl.sum(tl.where((h == anchor_id)[:, None], q, 0), 0)
    denominator = tl.sum(anchor * anchor, 0)
    # Projection quality affects pruning only. Certify the actual rounded beta.
    beta = tl.minimum(tl.maximum(tl.sum(q * anchor[None, :], 1), 0) / tl.maximum(denominator, 2.0**-64), 65536.0)
    beta = tl.where(h == anchor_id, 0, beta)
    lo = lib.add_rd(q, -lib.mul_ru(beta[:, None], anchor[None, :]))
    hi = lib.add_ru(q, -lib.mul_rd(beta[:, None], anchor[None, :]))
    lo = tl.where((h == anchor_id)[:, None], 0, lo)
    hi = tl.where((h == anchor_id)[:, None], 0, hi)
    envelope = tl.maximum(tl.abs(lo), tl.abs(hi))
    triangle = tl.reduce(lib.sqrt_ru(tl.reduce(lib.mul_ru(envelope, envelope), 1, _up)), 0, _up)
    center = _norm(tl.maximum(tl.abs(tl.reduce(lo, 0, _down)), tl.abs(tl.reduce(hi, 0, _up))))
    gram = tl.full((), 0, tl.float32)
    for i in tl.static_range(4):
        a = tl.sum(tl.where((h == i)[:, None], lo, 0), 0)
        b = tl.sum(tl.where((h == i)[:, None], hi, 0), 0)
        products_lo = tl.minimum(
            tl.minimum(lib.mul_rd(lo, a), lib.mul_rd(lo, b)),
            tl.minimum(lib.mul_rd(hi, a), lib.mul_rd(hi, b)),
        )
        products_hi = tl.maximum(
            tl.maximum(lib.mul_ru(lo, a), lib.mul_ru(lo, b)),
            tl.maximum(lib.mul_ru(hi, a), lib.mul_ru(hi, b)),
        )
        dot_lo = tl.reduce(products_lo, 1, _down)
        dot_hi = tl.reduce(products_hi, 1, _up)
        gram = lib.add_ru(gram, tl.reduce(tl.maximum(tl.abs(dot_lo), tl.abs(dot_hi)), 0, _up))
    joint = lib.mul_ru(lib.add_ru(center, lib.sqrt_ru(gram)), 0.5)
    fold = lib.add_ru(1.0, tl.reduce(beta, 0, _up))
    norms = lib.sqrt_ru(tl.reduce(lib.mul_ru(q, q), 1, _up))
    # gamma_128 < 2^-15 (even using a full FP32 ulp). The 2^-14
    # reference allowance also covers the four-head FP32 sum. The second
    # term encloses the BF16 tensor-core screening dot, not BF16 scores.
    arithmetic = lib.mul_ru(tl.reduce(norms, 0, _up), 2.0**-14)
    screening = lib.mul_ru(lib.mul_ru(fold, _norm(anchor)), 2.0**-15)
    error = lib.add_ru(tl.minimum(triangle, joint), lib.add_ru(arithmetic, screening))
    ok = (tl.sum((~_safe(q)).to(tl.int32)) == 0) & (tl.sum((~_safe(beta)).to(tl.int32)) == 0)
    ok &= (error < float("inf")) & (fold < float("inf"))
    tl.store(ANCHOR + row * 128 + d, anchor)
    tl.store(FOLD + row, fold)
    tl.store(ERROR + row, tl.where(ok, error, float("inf")))


@triton.jit
def key_norm(K, NORMS, N):
    ids = tl.program_id(0) * 32 + tl.arange(0, 32)
    d = tl.arange(0, 128)
    k = tl.load(K + ids[:, None] * 128 + d[None, :], ids[:, None] < N, 0).to(tl.float32)
    norm = lib.sqrt_ru(tl.reduce(lib.mul_ru(k, k), 1, _up))
    safe = tl.sum((~_safe(k)).to(tl.int32), 1) == 0
    tl.store(NORMS + ids, tl.where(safe, norm, float("inf")), ids < N)


@triton.jit
def _scores(Q, K, row, ids, valid, B: tl.constexpr):
    """Match the radix scorer's IEEE dot layout, followed by the head sum.

    On SM103 the 64-row tile reproduces every existing radix autotune
    configuration. A 16-row tile changes the compiler's FP32 accumulation order
    and can swap nearly tied IDs even with identical BF16 operands.
    """
    h, d = tl.arange(0, 64), tl.arange(0, 16)
    acc = tl.full((64, B), 0, tl.float32)
    for first in range(0, 128, 16):
        q = tl.load(Q + row * 512 + h[:, None] * 128 + first + d[None, :], h[:, None] < 4, 0)
        k = tl.load(K + ids[None, :] * 128 + first + d[:, None], valid[None, :], 0)
        acc = tl.dot(q.to(tl.float32), k.to(tl.float32), acc, input_precision="ieee")
    score = tl.full((B,), 0, tl.float32)
    for head in tl.static_range(4):
        score += tl.maximum(tl.sum(tl.where(h[:, None] == head, acc, 0), 0), 0)
    return score * (128**-0.5)


@triton.jit
def screen(
    K,
    NORMS,
    ANCHOR,
    FOLD,
    ERROR,
    VISIBLE,
    TAU,
    FLAGS,
    COUNTS,
    ROWS,
    N,
    BOUNDS=None,
    DUMP: tl.constexpr = False,
):
    rows = tl.program_id(0) * 16 + tl.arange(0, 16)
    ids = tl.program_id(1) * 128 + tl.arange(0, 128)
    visible = tl.load(VISIBLE + rows, rows < ROWS, 0)
    if tl.program_id(1) * 128 >= tl.max(visible, 0):
        live = tl.full((16, 128), False, tl.int1)
        upper = tl.full((16, 128), float("inf"), tl.float32)
    else:
        d = tl.arange(0, 128)
        q = tl.load(ANCHOR + rows[:, None] * 128 + d[None, :], rows[:, None] < ROWS, 0)
        k = tl.load(K + ids[None, :] * 128 + d[:, None], ids[None, :] < N, 0)
        dot = tl.dot(q, k, input_precision="ieee")
        fold = tl.load(FOLD + rows, rows < ROWS, 0)
        error = tl.load(ERROR + rows, rows < ROWS, float("inf"))
        norm = tl.load(NORMS + ids, ids < N, float("inf"))
        upper = lib.add_ru(lib.mul_ru(fold[:, None], tl.maximum(dot, 0)), lib.mul_ru(error[:, None], norm))
        upper = lib.mul_ru(upper, 128**-0.5)
        # NaN, infinity, underflow-domain inputs, and threshold equality
        # must all survive. The dense refinement fallback handles them.
        upper = tl.where(tl.abs(upper) < float("inf"), upper, float("inf"))
        tau = tl.load(TAU + rows, rows < ROWS, -float("inf"))
        live = (ids[None, :] < visible[:, None]) & ~(upper < tau[:, None])
    if DUMP:
        tl.store(BOUNDS + rows[:, None] * N + ids[None, :], upper, (rows[:, None] < ROWS) & (ids[None, :] < N))
    tl.store(FLAGS + rows[:, None] * N + ids[None, :], live, (rows[:, None] < ROWS) & (ids[None, :] < N))
    tl.store(COUNTS + rows * tl.cdiv(N, 128) + tl.program_id(1), tl.sum(live.to(tl.int32), 1), rows < ROWS)


@triton.jit
def compact(FLAGS, PREFIX, IDS, N, CAP):
    row, tile = tl.program_id(0), tl.program_id(1)
    ids = tile * 128 + tl.arange(0, 128)
    live = tl.load(FLAGS + row * N + ids, ids < N, False)
    start = tl.load(PREFIX + row * tl.cdiv(N, 128) + tile - 1, tile > 0, 0)
    dest = start + tl.cumsum(live.to(tl.int32), 0) - 1
    tl.store(IDS + row * CAP + dest, ids, live & (dest < CAP))


@triton.jit
def refine(Q, K, IDS, PREFIX, VISIBLE, SCORES, LENGTHS, N, CAP):
    row, tile = tl.program_id(0), tl.program_id(1)
    count = tl.load(PREFIX + row * tl.cdiv(N, 128) + tl.cdiv(N, 128) - 1)
    overflow = count > CAP
    count = tl.where(overflow, tl.load(VISIBLE + row), count)
    if tile == 0:
        tl.store(LENGTHS + row, count)
    # Overflow rows are scored by the existing dense radix kernel, shared
    # across query rows. Avoid the much slower per-query dense gather path.
    if (tile * 128 < count) & ~overflow:
        t = tile * 128 + tl.arange(0, 128)
        ids = tl.load(IDS + row * CAP + t, t < count, 0)
        scores = _scores(Q, K, row, ids, t < count, 128)
        tl.store(SCORES + row * N + t, scores, t < count)


@triton.jit
def restore(CODES, IDS, PREFIX, OUT, N, CAP, KP: tl.constexpr, BUDGET: tl.constexpr):
    row = tl.program_id(0)
    t = tl.arange(0, KP)
    codes = tl.load(CODES + row * KP + t)
    rank = ((1 << 21) - 1 - (codes & ((1 << 21) - 1))).to(tl.int32)
    count = tl.load(PREFIX + row * tl.cdiv(N, 128) + tl.cdiv(N, 128) - 1)
    if count > CAP:
        original = rank
    else:
        original = tl.load(IDS + row * CAP + rank, (codes > 0) & (rank < CAP), -1)
    tl.store(OUT + row * BUDGET + t, tl.where(codes > 0, original, -1), t < BUDGET)
