# Copyright (c) 2026 Relax Authors. All Rights Reserved.


from __future__ import annotations

import torch

from relax.backends.megatron.chunkflow.flash_backends import flash3_varlen_ops, flash_backend_name
from relax.backends.megatron.chunkflow.state_store import ChunkState, StateStore


def _manual_attn_forward(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, causal: bool, softmax_scale: float
) -> tuple[torch.Tensor, torch.Tensor]:
    tq, h, _d = q.shape
    tk = k.shape[0]
    k, v, _repeat = _expand_kv_heads(k, v, h)
    q_h = q.permute(1, 0, 2)
    k_h = k.permute(1, 0, 2)
    v_h = v.permute(1, 0, 2)

    scores = torch.matmul(q_h, k_h.transpose(-1, -2)) * softmax_scale
    if causal:
        scores = scores.masked_fill(_causal_mask(tq, tk, q.device), float("-inf"))

    lse_h = torch.logsumexp(scores, dim=-1)
    probs = torch.exp(scores - lse_h.unsqueeze(-1))
    out_h = torch.matmul(probs, v_h)

    out = out_h.permute(1, 0, 2).contiguous()
    lse = lse_h.permute(1, 0).contiguous()
    return out, lse


def _manual_attn_backward(
    dout: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    lse: torch.Tensor,
    causal: bool,
    softmax_scale: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    tq, h, _d = q.shape
    tk = k.shape[0]
    kv_heads = k.shape[1]
    k, v, repeat = _expand_kv_heads(k, v, h)
    q_h = q.permute(1, 0, 2)
    k_h = k.permute(1, 0, 2)
    v_h = v.permute(1, 0, 2)

    scores = torch.matmul(q_h, k_h.transpose(-1, -2)) * softmax_scale
    if causal:
        scores = scores.masked_fill(_causal_mask(tq, tk, q.device), float("-inf"))
    lse_h = lse.permute(1, 0)
    probs = torch.exp(scores - lse_h.unsqueeze(-1))

    dout_h = dout.permute(1, 0, 2)
    dv_h = torch.matmul(probs.transpose(-1, -2), dout_h)
    dprobs = torch.matmul(dout_h, v_h.transpose(-1, -2))
    dscores = probs * (dprobs - (dprobs * probs).sum(dim=-1, keepdim=True))
    draw = dscores * softmax_scale

    dq_h = torch.matmul(draw, k_h)
    dk_h = torch.matmul(draw.transpose(-1, -2), q_h)

    dq = dq_h.permute(1, 0, 2).contiguous()
    dk = dk_h.permute(1, 0, 2).contiguous()
    dv = dv_h.permute(1, 0, 2).contiguous()
    if repeat > 1:
        dk = dk.reshape(tk, kv_heads, repeat, _d).sum(dim=2)
        dv = dv.reshape(tk, kv_heads, repeat, _d).sum(dim=2)
    return dq, dk, dv


def _expand_kv_heads(k: torch.Tensor, v: torch.Tensor, query_heads: int) -> tuple[torch.Tensor, torch.Tensor, int]:
    kv_heads = k.shape[1]
    if kv_heads == query_heads:
        return k, v, 1
    if query_heads % kv_heads != 0:
        raise ValueError(f"query heads ({query_heads}) must be divisible by KV heads ({kv_heads})")
    repeat = query_heads // kv_heads
    return k.repeat_interleave(repeat, dim=1), v.repeat_interleave(repeat, dim=1), repeat


def _causal_mask(tq: int, tk: int, device: torch.device) -> torch.Tensor:
    offset = tk - tq
    q_pos = torch.arange(tq, device=device).unsqueeze(1) + offset
    k_pos = torch.arange(tk, device=device).unsqueeze(0)
    return k_pos > q_pos


def _flash_varlen_ops():
    if flash_backend_name() == "fa3":
        return flash3_varlen_ops()
    try:
        from flash_attn.flash_attn_interface import (
            _flash_attn_varlen_backward,
            _flash_attn_varlen_forward,
        )
    except ImportError:
        return None
    return _flash_attn_varlen_forward, _flash_attn_varlen_backward


def _use_flash(q: torch.Tensor) -> bool:
    return q.is_cuda and q.dtype in (torch.float16, torch.bfloat16) and _flash_varlen_ops() is not None


def _cu_seqlens(total: int, device: torch.device) -> torch.Tensor:
    return torch.tensor([0, total], dtype=torch.int32, device=device)


class ChunkCacheAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, causal, softmax_scale, num_chunks, *kv_chunks):
        ks, vs = kv_chunks[:num_chunks], kv_chunks[num_chunks:]
        k_all = torch.cat(ks, dim=0)
        v_all = torch.cat(vs, dim=0)
        flash = _use_flash(q)
        ctx.flash_ops = _flash_varlen_ops() if flash else None
        if flash:
            out, lse = _flash_varlen(
                q,
                k_all,
                v_all,
                _cu_seqlens(q.shape[0], q.device),
                _cu_seqlens(k_all.shape[0], k_all.device),
                causal,
                softmax_scale,
                max_seqlens=(q.shape[0], k_all.shape[0]),
                ops=ctx.flash_ops,
            )
        else:
            out, lse = _manual_attn_forward(q, k_all, v_all, causal, softmax_scale)
        # Save KV fragments; rebuild concatenations only during backward.
        ctx.save_for_backward(q, lse, *((out,) if flash else ()), *kv_chunks)
        ctx.causal = causal
        ctx.softmax_scale = softmax_scale
        ctx.num_chunks = num_chunks
        ctx.flash = flash
        ctx.chunk_lengths = [k.shape[0] for k in ks]
        return out, lse

    @staticmethod
    def backward(ctx, dout, _dlse):
        saved = list(ctx.saved_tensors)
        q, lse = saved.pop(0), saved.pop(0)
        out = saved.pop(0) if ctx.flash else None
        kv_chunks = saved
        n = ctx.num_chunks
        k_all = torch.cat(kv_chunks[:n], dim=0)
        v_all = torch.cat(kv_chunks[n:], dim=0)
        if ctx.flash:
            dq, dk_all, dv_all = _flash_varlen_backward(
                dout.contiguous(),
                q,
                k_all,
                v_all,
                out,
                lse,
                _cu_seqlens(q.shape[0], q.device),
                _cu_seqlens(k_all.shape[0], k_all.device),
                ctx.causal,
                ctx.softmax_scale,
                max_seqlens=(q.shape[0], k_all.shape[0]),
                ops=ctx.flash_ops,
            )
        else:
            dq, dk_all, dv_all = _manual_attn_backward(dout, q, k_all, v_all, lse, ctx.causal, ctx.softmax_scale)
        dks = torch.split(dk_all, ctx.chunk_lengths, dim=0)
        dvs = torch.split(dv_all, ctx.chunk_lengths, dim=0)
        return (dq, None, None, None, *dks, *dvs)


def chunk_attention_forward(
    chunk_id: int,
    q: torch.Tensor,
    new_k: torch.Tensor,
    new_v: torch.Tensor,
    store: StateStore,
    causal: bool = True,
    softmax_scale: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    store.put(chunk_id, ChunkState({"k": new_k, "v": new_v}))
    states = store.lookup(chunk_id)
    ks = [s.tensors["k"] for s in states]
    vs = [s.tensors["v"] for s in states]
    scale = softmax_scale if softmax_scale is not None else q.shape[-1] ** -0.5
    return ChunkCacheAttention.apply(q, causal, scale, len(ks), *ks, *vs)


def _split_sizes_by_cu(cu: torch.Tensor) -> list[int]:
    return [int(cu[i + 1] - cu[i]) for i in range(cu.numel() - 1)]


class UnitCacheAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, causal, softmax_scale, cu_q, cu_kv, *kv):
        ctx.has_max_seqlens = bool(kv and isinstance(kv[-1], tuple))
        max_seqlens = kv[-1] if ctx.has_max_seqlens else None
        if ctx.has_max_seqlens:
            kv = kv[:-1]
            if len(max_seqlens) != 2 or any(type(length) is not int or length <= 0 for length in max_seqlens):
                raise ValueError("max_seqlens must contain two positive Python integers")
        n = len(kv) // 2
        k_all = kv[0].contiguous() if n == 1 else torch.cat(kv[:n], dim=0)
        v_all = kv[1].contiguous() if n == 1 else torch.cat(kv[n:], dim=0)
        flash = _use_flash(q)
        ctx.flash_ops = _flash_varlen_ops() if flash else None
        if flash:
            if max_seqlens is None:
                max_seqlens = (_max_seqlen(cu_q), _max_seqlen(cu_kv))
            out, lse = _flash_varlen(
                q,
                k_all,
                v_all,
                cu_q,
                cu_kv,
                causal,
                softmax_scale,
                max_seqlens=max_seqlens,
                ops=ctx.flash_ops,
            )
        else:
            out, lse = _manual_varlen_forward(q, k_all, v_all, cu_q, cu_kv, causal, softmax_scale)

        ctx.save_for_backward(q, lse, cu_q, cu_kv, *((out,) if flash else ()), *kv)
        ctx.causal = causal
        ctx.softmax_scale = softmax_scale
        ctx.flash = flash
        ctx.n = n
        ctx.kv_lengths = [k.shape[0] for k in kv[:n]]
        ctx.max_seqlens = max_seqlens
        return out, lse

    @staticmethod
    def backward(ctx, dout, _dlse):
        saved = list(ctx.saved_tensors)
        q, lse, cu_q, cu_kv = saved[:4]
        rest = saved[4:]
        out = rest.pop(0) if ctx.flash else None
        n = ctx.n
        k_all = rest[0].contiguous() if n == 1 else torch.cat(rest[:n], dim=0)
        v_all = rest[1].contiguous() if n == 1 else torch.cat(rest[n:], dim=0)
        if ctx.flash:
            dq, dk_all, dv_all = _flash_varlen_backward(
                dout.contiguous(),
                q,
                k_all,
                v_all,
                out,
                lse,
                cu_q,
                cu_kv,
                ctx.causal,
                ctx.softmax_scale,
                max_seqlens=ctx.max_seqlens,
                ops=ctx.flash_ops,
            )
        else:
            dq, dk_all, dv_all = _manual_varlen_backward(
                dout, q, k_all, v_all, lse, cu_q, cu_kv, ctx.causal, ctx.softmax_scale
            )
        dks = torch.split(dk_all, ctx.kv_lengths, dim=0)
        dvs = torch.split(dv_all, ctx.kv_lengths, dim=0)
        gradients = (dq, None, None, None, None, *dks, *dvs)
        return (*gradients, None) if ctx.has_max_seqlens else gradients


def _manual_varlen_forward(q, k, v, cu_q, cu_kv, causal, softmax_scale):
    outs, lses = [], []
    qs = _split_sizes_by_cu(cu_q)
    ks = _split_sizes_by_cu(cu_kv)
    qo = ko = 0
    for ql, kl in zip(qs, ks):
        out, lse = _manual_attn_forward(
            q[qo : qo + ql], k[ko : ko + kl], v[ko : ko + kl], causal=causal, softmax_scale=softmax_scale
        )
        outs.append(out)
        lses.append(lse)
        qo += ql
        ko += kl
    return torch.cat(outs, dim=0), torch.cat(lses, dim=0)


def _manual_varlen_backward(dout, q, k, v, lse, cu_q, cu_kv, causal, softmax_scale):
    qs = _split_sizes_by_cu(cu_q)
    ks = _split_sizes_by_cu(cu_kv)
    dqs, dks, dvs = [], [], []
    qo = ko = 0
    for ql, kl in zip(qs, ks):
        dq, dk, dv = _manual_attn_backward(
            dout[qo : qo + ql],
            q[qo : qo + ql],
            k[ko : ko + kl],
            v[ko : ko + kl],
            lse[qo : qo + ql],
            causal=causal,
            softmax_scale=softmax_scale,
        )
        dqs.append(dq)
        dks.append(dk)
        dvs.append(dv)
        qo += ql
        ko += kl
    return torch.cat(dqs, dim=0), torch.cat(dks, dim=0), torch.cat(dvs, dim=0)


def _max_seqlen(cu: torch.Tensor) -> int:
    return max(_split_sizes_by_cu(cu))


def _flash_varlen(
    q, k, v, cu_q, cu_kv, causal, softmax_scale, *, max_seqlens: tuple[int, int] | None = None, ops=None
):
    fwd, _bwd = ops if ops is not None else _flash_varlen_ops()
    max_q, max_kv = max_seqlens if max_seqlens is not None else (_max_seqlen(cu_q), _max_seqlen(cu_kv))
    out, lse, _s, _rng = fwd(
        q,
        k,
        v,
        cu_q.to(torch.int32),
        cu_kv.to(torch.int32),
        max_q,
        max_kv,
        0.0,
        softmax_scale,
        causal,
    )[:4]
    return out, lse


def _flash_varlen_backward(
    dout,
    q,
    k,
    v,
    out,
    lse,
    cu_q,
    cu_kv,
    causal,
    softmax_scale,
    *,
    max_seqlens: tuple[int, int] | None = None,
    ops=None,
):
    _fwd, bwd = ops if ops is not None else _flash_varlen_ops()
    max_q, max_kv = max_seqlens if max_seqlens is not None else (_max_seqlen(cu_q), _max_seqlen(cu_kv))
    dq, dk, dv = torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)
    bwd(
        dout,
        q,
        k,
        v,
        out,
        lse,
        dq,
        dk,
        dv,
        cu_q.to(torch.int32),
        cu_kv.to(torch.int32),
        max_q,
        max_kv,
        0.0,
        softmax_scale,
        causal,
        -1,
        -1,
        0.0,
        None,
        False,
    )
    return dq, dk, dv
