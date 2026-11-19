# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""TP4 Qwen gated QKV pair-gather extraction and rank-major SUM-RS packing."""

import torch
import torch.distributed as dist
import triton
import triton.language as tl


@triton.jit
def _extract(X, Q, K, V, G, N, RANK: tl.constexpr, BLOCK: tl.constexpr):
    o = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    row = o // 1024
    d = o % 1024
    valid = row < N
    qc = RANK * 1024 + d
    gc = 2048 + RANK * 1024 + d
    qi = (qc // 2304 * N + row) * 2304 + qc % 2304
    gi = (gc // 2304 * N + row) * 2304 + gc % 2304
    q = tl.load(X + qi, valid, 0)
    g = tl.load(X + gi, valid, 0)
    tl.store(Q + o, q, valid)
    tl.store(G + o, g, valid)
    kvvalid = valid & (d < 256)
    k = tl.load(X + (N + row) * 2304 + 1792 + d, kvvalid, 0)
    v = tl.load(X + (N + row) * 2304 + 2048 + d, kvvalid, 0)
    tl.store(K + row * 256 + d, k, kvvalid)
    tl.store(V + row * 256 + d, v, kvvalid)


@triton.jit
def _pack(
    DQ,
    DK,
    DV,
    DG,
    Y,
    N,
    RANK: tl.constexpr,
    QS: tl.constexpr,
    QH: tl.constexpr,
    QD: tl.constexpr,
    KS: tl.constexpr,
    KD: tl.constexpr,
    VS: tl.constexpr,
    VD: tl.constexpr,
    GS: tl.constexpr,
    GH: tl.constexpr,
    GD: tl.constexpr,
    HASQ: tl.constexpr,
    HASK: tl.constexpr,
    HASV: tl.constexpr,
    HASG: tl.constexpr,
    BLOCK: tl.constexpr,
):
    local_o = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = local_o < N * 2304
    owner = tl.program_id(1)
    o = owner * N * 2304 + local_o
    row = local_o // 2304
    col = owner * 2304 + local_o % 2304
    qd = col - RANK * 1024
    gd = col - 2048 - RANK * 1024
    kd = col - 4096
    vd = col - 4352
    qmask = valid & (col < 2048) & (qd >= 0) & (qd < 1024)
    gmask = valid & (col >= 2048) & (col < 4096) & (gd >= 0) & (gd < 1024)
    kmask = valid & (col >= 4096) & (col < 4352)
    vmask = valid & (col >= 4352)
    y = tl.full((BLOCK,), 0, tl.bfloat16)
    if HASQ:
        q = tl.load(DQ + row * QS + (qd // 256) * QH + (qd % 256) * QD, qmask, 0)
        y = tl.where(qmask, q, y)
    if HASG:
        g = tl.load(DG + row * GS + (gd // 256) * GH + (gd % 256) * GD, gmask, 0)
        y = tl.where(gmask, g, y)
    if HASK:
        k = tl.load(DK + row * KS + kd * KD, kmask, 0)
        y = tl.where(kmask, k, y)
    if HASV:
        v = tl.load(DV + row * VS + vd * VD, vmask, 0)
        y = tl.where(vmask, v, y)
    tl.store(Y + o, y, valid)


class Extract(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, group):
        assert x.is_cuda and x.dtype == torch.bfloat16 and x.ndim == 3 and x.shape[1:] == (1, 2304)
        assert x.is_contiguous() and dist.get_world_size(group) == 2
        n = x.shape[0]
        rank = dist.get_rank(group)
        assert n > 0 and rank in (0, 1)
        gathered = torch.empty((2 * n, 1, 2304), device=x.device, dtype=x.dtype)
        dist.all_gather_into_tensor(gathered, x, group=group)
        q = torch.empty((n, 1, 4, 256), device=x.device, dtype=x.dtype)
        g = torch.empty_like(q)
        k = torch.empty((n, 1, 1, 256), device=x.device, dtype=x.dtype)
        v = torch.empty_like(k)
        _extract[(triton.cdiv(n * 1024, 1024),)](gathered, q, k, v, g, n, rank, 1024)
        ctx.meta = (n, rank, x.device)
        ctx.group = group
        ctx.set_materialize_grads(False)
        return q, k, v, g

    @staticmethod
    def backward(ctx, dq, dk, dv, dg):
        n, rank, device = ctx.meta
        grads = [dq, dk, dv, dg]
        present = [x is not None for x in grads]
        fallback = next(x for x in grads if x is not None)
        assert all(x is None or (x.dtype == torch.bfloat16 and x.device == device) for x in grads)
        q, k, v, g = [x if x is not None else fallback for x in grads]

        def stride(x, which):
            return x.stride(which)

        packed = torch.empty((2 * n, 1, 2304), device=device, dtype=torch.bfloat16)
        _pack[(triton.cdiv(n * 2304, 1024), 2)](
            q,
            k,
            v,
            g,
            packed,
            n,
            rank,
            q.stride(0),
            q.stride(-2),
            q.stride(-1),
            k.stride(0),
            k.stride(-1),
            v.stride(0),
            v.stride(-1),
            g.stride(0),
            g.stride(-2),
            g.stride(-1),
            *present,
            1024,
        )
        dx = torch.empty((n, 1, 2304), device=device, dtype=torch.bfloat16)
        dist.reduce_scatter_tensor(dx, packed, op=dist.ReduceOp.SUM, group=ctx.group)
        return dx, None


def method(self, hidden_states, key_value_states=None, output_gate=False, head_wise_gate=False, split_qkv=True):
    assert output_gate and not head_wise_gate and split_qkv and key_value_states is None
    mixed, _ = self.linear_qkv(hidden_states)
    q, k, v, g = Extract.apply(mixed, self._qwen_kv_pair_group)
    if self.q_layernorm is not None:
        q = self.q_layernorm(q)
    if self.k_layernorm is not None:
        k = self.k_layernorm(k)
    return q, k, v, g
