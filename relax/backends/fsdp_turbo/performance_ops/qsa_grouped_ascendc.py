# Copyright (c) 2026 Relax Authors. All Rights Reserved.
# Adapted from MindSpeed-Bridge, Apache-2.0; pinned revision 1da5bf7.
"""Packed, per-row-exact grouped QSA for BF16 A3."""

from dataclasses import dataclass
from functools import lru_cache

import torch
from qwen38_fused_train_ops.grouped_qsa import binding
from torch.autograd.function import once_differentiable


@dataclass(frozen=True)
class Layout:
    descriptors: torch.Tensor
    cu_q: torch.Tensor
    used_q: torch.Tensor
    probe: torch.Tensor | None
    group_size: int
    max_q: int
    max_kv: int
    max_blocks: int
    capacity: int
    needs_capacity_check: bool
    tail_padding: int


@lru_cache(maxsize=24)
def layout(q_offsets, kv_offsets, topk, group_size, device):
    if len(q_offsets) != len(kv_offsets):
        raise ValueError("Q and KV must have the same number of continuous intervals")
    used, desc, probes = [], [], []
    max_kv = max(b - a for a, b in zip(kv_offsets, kv_offsets[1:]))
    max_blocks = (max_kv + 3) // 4
    max_possible = min(max_blocks, (topk + 1) * group_size)
    capacity = min(4096, (max_possible + 63) // 64 * 64)
    for qa, qb, ka, kb in zip(q_offsets[:-1], q_offsets[1:], kv_offsets[:-1], kv_offsets[1:], strict=True):
        q_len, kv_len = qb - qa, kb - ka
        if not 0 < q_len <= kv_len:
            raise ValueError("Grouped QSA requires a right-aligned Q interval in its KV prefix")
        first_group = len(desc)
        for first in range(qa, qb, group_size):
            n = min(group_size, qb - first)
            desc.append((first, n, kv_len - q_len + first - qa, (kv_len + 3) // 4))
        group_count = len(desc) - first_group
        if (kv_len + 3) // 4 > 4096:
            # A partial last group exaggerates overlap. Probe the last
            # full group and two interior groups before allocating a full plan.
            last = first_group + max(0, q_len // group_size - 1)
            probes.extend((last, max(first_group, last - group_count // 8), max(first_group, last - group_count // 2)))
        used.append(q_len)

    def npu(values, dtype):
        return torch.tensor(values, dtype=dtype).to(device, non_blocking=True)

    return Layout(
        npu(desc, torch.int32),
        npu(q_offsets, torch.int64),
        npu(used, torch.int32),
        npu([desc[i] for i in sorted(set(probes))], torch.int32) if probes and max_possible > capacity else None,
        group_size,
        max(used),
        max_kv,
        max_blocks,
        capacity,
        max_possible > capacity,
        (-(kv_offsets[-1] - kv_offsets[-2])) % 4,
    )


def prepare(carrier, rows, cfg):
    indices, sizes, mask = binding().plan(
        carrier,
        rows,
        cfg.descriptors,
        cfg.group_size,
        cfg.capacity,
        cfg.max_blocks,
    )
    return indices, sizes, mask


def pad_kv(tensor, cfg):
    if cfg.tail_padding == 0:
        return tensor
    return torch.cat((tensor, tensor.new_zeros((cfg.tail_padding, *tensor.shape[1:]))), dim=0)


def forward(q, k, v, plan, cfg, cu_kv, scale, *, preserve_fp32_output=False):
    indices, sizes, mask = plan
    fn = binding().forward_fp32 if preserve_fp32_output else binding().forward
    return fn(
        q,
        pad_kv(k, cfg),
        pad_kv(v, cfg),
        indices,
        sizes,
        mask,
        cfg.cu_q,
        cu_kv,
        cfg.used_q,
        cfg.group_size,
        cfg.max_q,
        cfg.max_kv,
        scale,
    )


def backward(q, k, v, dout, out, lse, plan, cfg, cu_q, cu_kv, scale):
    indices, sizes, mask = plan
    dq, dk, dv = binding().backward(
        q,
        pad_kv(k, cfg),
        pad_kv(v, cfg),
        dout,
        out,
        lse,
        indices.unsqueeze(0),
        sizes.unsqueeze(0),
        mask,
        cu_q,
        cu_kv,
        cfg.used_q,
        cfg.group_size,
        scale,
    )
    return dq, dk[: k.shape[0]], dv[: v.shape[0]]


def adaptive_plan(carrier, rows, q_offsets, kv_offsets, device, q_heads_per_kv):
    topk = carrier.size(-1) - 1
    # Keep at most 128 Q/head rows per task so Grad can use the 128-token KV tile.
    for group_size in (16, 8, 4):
        if group_size * q_heads_per_kv > 128:
            continue
        cfg = layout(q_offsets, kv_offsets, topk, group_size, device)
        # Select a provably sufficient capacity from CPU shape metadata.
        # Avoid a per-layer device-to-host capacity probe in training.
        if cfg.needs_capacity_check:
            continue
        return cfg, prepare(carrier, rows, cfg)
    return None, None


class _Grouped(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, indices, sizes, mask, cu_q, cu_kv, cfg, scale):
        out, lse = forward(q, k, v, (indices, sizes, mask), cfg, cu_kv, scale, preserve_fp32_output=True)
        # Save only the native output used in Delta. The BF16 result is needed
        # by the rest of the model, but must not also be retained by this op.
        ctx.save_for_backward(q, k, v, out, lse, indices, sizes, mask, cu_q, cu_kv)
        ctx.cfg = cfg
        ctx.scale = scale
        return out.to(dtype=q.dtype)

    @staticmethod
    @once_differentiable
    def backward(ctx, dout):
        q, k, v, out, lse, indices, sizes, mask, cu_q, cu_kv = ctx.saved_tensors
        dq, dk, dv = backward(
            q,
            k,
            v,
            dout.contiguous(),
            out,
            lse,
            (indices, sizes, mask),
            ctx.cfg,
            cu_q,
            cu_kv,
            ctx.scale,
        )
        return dq, dk, dv, None, None, None, None, None, None, None


def _apply_group(q, k, v, indices, sizes, mask, cu_q, cu_kv, cfg, scale):
    q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
    if torch.is_grad_enabled() and (q.requires_grad or k.requires_grad or v.requires_grad):
        return _Grouped.apply(q, k, v, indices, sizes, mask, cu_q, cu_kv, cfg, scale)
    # Activation checkpointing first runs a no-grad forward. Allocate saved
    # FP32 output only when the forward actually builds a backward graph.
    return forward(q, k, v, (indices, sizes, mask), cfg, cu_kv, scale)[0]


def grouped_attention(q, k, v, carrier, rows, cu_q, cu_kv, q_offsets, kv_offsets, scale):
    kv_heads = k.size(1)
    q_heads_per_kv = q.size(1) // kv_heads
    cfg, plan = adaptive_plan(carrier, rows, q_offsets, kv_offsets, q.device, q_heads_per_kv)
    if plan is None:
        raise RuntimeError("Grouped QSA could not construct a valid plan even with four Q rows per group")

    args = (*plan, cu_q, cu_kv, cfg, scale)
    if kv_heads == 1:
        return _apply_group(q, k, v, *args)

    # The QSA indexer shares its per-token selection across attention heads.
    # Each KV head uses the same plan with its contiguous slice of Q heads.
    return torch.cat(
        [
            _apply_group(qh, kh, vh, *args)
            for qh, kh, vh in zip(q.split(q_heads_per_kv, dim=1), k.split(1, dim=1), v.split(1, dim=1), strict=True)
        ],
        dim=1,
    )
