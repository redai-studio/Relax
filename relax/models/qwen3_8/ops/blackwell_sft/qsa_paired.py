# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Group adjacent GQA queries while preserving each row's complete selected-
token set."""

import torch
import triton
import triton.language as tl


@triton.jit(do_not_specialize=["N"])
def _merge(Ids, Positions, Rows, Union, Counts, N, TOP: tl.constexpr, TILE: tl.constexpr, STRIDE: tl.constexpr):
    pair = tl.program_id(0)
    lane = tl.arange(0, TILE)
    which = lane // TOP
    physical = tl.load(Rows + pair * 2 + which, mask=(which < 2) & (pair * 2 + which < N), other=0)
    pos = tl.load(Positions + physical)
    doc = physical - pos
    raw = tl.load(Ids + physical * TOP + lane % TOP, mask=(which < 2) & (pair * 2 + which < N), other=-1)
    valid = (raw >= 0) & (which < 2) & (pair * 2 + which < N)
    key = tl.where(valid, (doc + 4 * raw) * 2 + which, 0x7FFFFFFE)
    sorted_keys = tl.sort(key, descending=False)
    addr = sorted_keys // 2
    prev = tl.gather(addr, tl.maximum(lane - 1, 0), 0)
    after = tl.gather(sorted_keys, tl.minimum(lane + 1, TILE - 1), 0)
    first = (sorted_keys != 0x7FFFFFFE) & ((lane == 0) | (addr != prev))
    shared = (lane + 1 < TILE) & (after // 2 == addr)
    # Two 3-bit fields give the number of valid tokens for each query.
    both = tl.where(shared, 36, tl.where(sorted_keys % 2 == 0, 32, 4))
    compact = tl.cumsum(first.to(tl.int32), 0) - 1
    total = tl.sum(first.to(tl.int32), 0)
    tl.store(Union + pair * STRIDE + compact, addr * 64 + both, first)
    tl.store(Counts + pair, total + 2)
    for q in tl.static_range(2):
        ix = pair * 2 + q
        row = tl.load(Rows + ix, ix < N, other=0)
        position = tl.load(Positions + row)
        n = tl.where(ix < N, (position + 1) % 4, 0)
        absolute = row - position + position // 4 * 4
        tl.store(Union + pair * STRIDE + total + q, absolute * 64 + tl.where(q == 0, n * 8, n))


@triton.jit(do_not_specialize=["N"])
def _pair(
    Q,
    K,
    V,
    Union,
    Counts,
    O,
    L,
    DO,
    DQ,
    DK,
    DV,
    N,
    HQ: tl.constexpr,
    HK: tl.constexpr,
    G: tl.constexpr,
    D: tl.constexpr,
    GH: tl.constexpr,
    BK: tl.constexpr,
    STRIDE: tl.constexpr,
    SCALE: tl.constexpr,
    BACKWARD: tl.constexpr,
):
    pid, kh = tl.program_id(0), tl.program_id(1)
    h = tl.arange(0, 2 * GH)
    d = tl.arange(0, D)
    b = tl.arange(0, BK)
    which = h // GH
    qidx = pid * 2 + which
    validq = (qidx < N) & (h % GH < G)
    hl = qidx * HQ + kh * G + h % GH
    p = hl[:, None] * D + d[None, :]
    q = tl.load(Q + p, validq[:, None], other=0)
    count = tl.load(Counts + pid)
    acc = tl.full((2 * GH, D), 0, tl.float32)
    if BACKWARD:
        do = tl.load(DO + p, validq[:, None], other=0)
        out = tl.load(O + p, validq[:, None], other=0)
        lse = tl.load(L + hl, validq, other=-float("inf"))
        delta = tl.sum(do.to(tl.float32) * out, 1)
    else:
        maximum = tl.full((2 * GH,), -float("inf"), tl.float32)
        denom = tl.full((2 * GH,), 0, tl.float32)
    for tile in range(tl.cdiv(count * 4, BK)):
        off = tile * BK + b
        packed = tl.load(Union + pid * STRIDE + off // 4, off // 4 < count, other=0)
        la = (packed // 8) % 8
        lb = packed % 8
        a = off % 4 < la
        z = off % 4 < lb
        valid = a | z
        ptr = ((packed // 64 + off % 4)[:, None] * HK + kh) * D + d[None, :]
        k = tl.load(K + ptr, valid[:, None], other=0)
        v = tl.load(V + ptr, valid[:, None], other=0)
        mask = validq[:, None] & tl.where(which[:, None] == 0, a[None, :], z[None, :])
        scores = tl.dot(q, tl.trans(k)) * SCALE
        if BACKWARD:
            probs = tl.where(mask, tl.exp(scores - tl.where(lse == -float("inf"), 0.0, lse)[:, None]), 0.0)
            dp = tl.dot(do, tl.trans(v))
            ds = probs * (dp - delta[:, None]) * SCALE
            acc += tl.dot(ds.to(k.dtype), k)
            dk = tl.dot(tl.trans(ds.to(q.dtype)), q)
            dv = tl.dot(tl.trans(probs.to(do.dtype)), do)
            tl.atomic_add(DK + ptr, dk, valid[:, None], sem="relaxed")
            tl.atomic_add(DV + ptr, dv, valid[:, None], sem="relaxed")
        else:
            scores = tl.where(mask, scores, -float("inf"))
            upper = tl.maximum(maximum, tl.max(scores, 1))
            safe = tl.where(upper == -float("inf"), 0.0, upper)
            probs = tl.exp(scores - safe[:, None])
            rescale = tl.where(maximum == -float("inf"), 0.0, tl.exp(maximum - safe))
            denom = denom * rescale + tl.sum(probs, 1)
            acc = acc * rescale[:, None] + tl.dot(probs.to(v.dtype), v)
            maximum = upper
    if BACKWARD:
        tl.store(DQ + p, acc, validq[:, None])
    else:
        div = tl.where(denom == 0, 1.0, denom)
        tl.store(O + p, acc / div[:, None], validq[:, None])
        tl.store(L + hl, tl.where(denom == 0, -float("inf"), maximum + tl.log(div)), validq)


class Paired(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, ids, pos, rows, scale):
        q, k, v, ids, pos, rows = (x.contiguous() for x in (q, k, v, ids, pos, rows))
        n, h, d = q.shape
        hk = k.shape[1]
        g = h // hk
        top = ids.shape[1]
        assert d == 256 and g == 12 and n and top == triton.next_power_of_2(top) and k.shape == v.shape
        assert ids.shape[0] <= 131072 and q.dtype == k.dtype == v.dtype == torch.bfloat16
        pairs = triton.cdiv(n, 2)
        stride = 2 * top + 2
        union = torch.empty((pairs, stride), device=q.device, dtype=torch.int32)
        counts = torch.empty((pairs,), device=q.device, dtype=torch.int32)
        _merge[(pairs,)](ids, pos, rows, union, counts, N=n, TOP=top, TILE=2 * top, STRIDE=stride, num_warps=4)
        out = torch.empty_like(q, dtype=torch.float32)
        lse = torch.empty((n, h), device=q.device, dtype=torch.float32)
        opts = dict(N=n, HQ=h, HK=hk, G=g, D=d, GH=16, STRIDE=stride, SCALE=scale)
        _pair[(pairs, hk)](
            q, k, v, union, counts, out, lse, q, q, k, v, **opts, BK=64, BACKWARD=False, num_warps=4, num_stages=2
        )
        ctx.opts = opts
        ctx.save_for_backward(q, k, v, union, counts, out, lse)
        return out.to(q.dtype)

    @staticmethod
    def backward(ctx, gradient):
        q, k, v, union, counts, out, lse = ctx.saved_tensors
        dq = torch.empty_like(q, dtype=torch.float32)
        dk = torch.zeros_like(k, dtype=torch.float32)
        dv = torch.zeros_like(v, dtype=torch.float32)
        _pair[(union.shape[0], k.shape[1])](
            q,
            k,
            v,
            union,
            counts,
            out,
            lse,
            gradient.contiguous(),
            dq,
            dk,
            dv,
            **ctx.opts,
            BK=32,
            BACKWARD=True,
            num_warps=4,
            num_stages=2,
        )
        return dq.to(q.dtype), dk.to(k.dtype), dv.to(v.dtype), None, None, None, None


def install():
    from relax.models.qwen3_8.ops.blackwell_sft import qsa_direct
    from relax.utils.logging_utils import get_logger

    native = qsa_direct.DirectGrouped

    class Selected:
        reported = False

        @staticmethod
        def apply(q, k, v, ids, positions, rows, scale):
            if (
                q.ndim != 3
                or q.shape[1:] != (12, 256)
                or k.shape[1:] != (1, 256)
                or k.shape != v.shape
                or q.dtype != torch.bfloat16
                or k.dtype != q.dtype
                or v.dtype != q.dtype
                or not 0 < q.shape[0] <= 131072
                or ids.shape[-1] != 512
                or ids.shape[0] > 131072
            ):
                return native.apply(q, k, v, ids, positions, rows, scale)
            if not Selected.reported:
                get_logger(__name__).info(
                    "Dynamic-row paired QSA with independent causal masks active on TP2 BF16 12Q/1KV/D256/top512"
                )
                Selected.reported = True
            return Paired.apply(q, k, v, ids, positions, rows, scale)

    qsa_direct.DirectGrouped = Selected
