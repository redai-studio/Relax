# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from functools import lru_cache
from itertools import accumulate

import torch
import triton


@lru_cache(maxsize=96)
def layout(bounds, limit, device):
    prefixes = []
    suffixes = []
    for start, end in zip(bounds, bounds[1:]):
        if end == start:
            continue
        pivot = min(end, start + limit)
        prefixes.append((start, pivot))
        suffixes.append((pivot, end))
    prefix_lengths = [b - a for a, b in prefixes]
    cu = torch.tensor([0, *accumulate(prefix_lengths)], device=device, dtype=torch.int32)
    return prefixes, suffixes, cu, max(prefix_lengths)


class DirectGrouped(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, block_ids, positions, rows, scale):
        from relax.models.qwen3_8.ops.blackwell_sft.direct_kernels import _forward

        q, k, v, block_ids, positions, rows = [t.contiguous() for t in (q, k, v, block_ids, positions, rows)]
        tokens, heads, width = q.shape
        kv_heads = k.shape[1]
        if q.dtype != torch.bfloat16 or width != 256 or heads % kv_heads or k.shape != v.shape:
            raise ValueError("Direct grouped QSA requires integral BF16 GQA/D256")
        opts = dict(
            HQ=heads,
            HK=kv_heads,
            D=width,
            S=4 * block_ids.shape[-1] + 4,
            G=heads // kv_heads,
            BH=max(16, triton.next_power_of_2(heads // kv_heads)),
            SCALE=scale,
            BF16_PRODUCTS=True,
        )
        output = torch.empty_like(q, dtype=torch.float32)
        lse = torch.empty((tokens, heads), device=q.device, dtype=torch.float32)
        _forward[(tokens, kv_heads)](
            q, k, v, block_ids, positions, rows, output, lse, **opts, BK=64, num_warps=4, num_stages=1
        )
        ctx.opts = opts
        ctx.save_for_backward(q, k, v, block_ids, positions, rows, output, lse)
        return output.to(q.dtype)

    @staticmethod
    def backward(ctx, dout):
        from relax.models.qwen3_8.ops.blackwell_sft.direct_kernels import _backward

        q, k, v, ids, positions, rows, out, lse = ctx.saved_tensors
        dq = torch.empty_like(q, dtype=torch.float32)
        dk = torch.zeros_like(k, dtype=torch.float32)
        dv = torch.zeros_like(v, dtype=torch.float32)
        _backward[(q.shape[0], k.shape[1])](
            q,
            k,
            v,
            ids,
            positions,
            rows,
            out,
            lse,
            dout.contiguous(),
            dq,
            dk,
            dv,
            **ctx.opts,
            BK=32,
            num_warps=4,
            num_stages=3,
        )
        return dq.to(q.dtype), dk.to(k.dtype), dv.to(v.dtype), None, None, None, None


def install():
    from functools import lru_cache

    from flash_attn import flash_attn_varlen_func

    from relax.models.qwen3_8.ops import qsa_mixed_ops
    from relax.utils.env import Envs
    from relax.utils.logging_utils import get_logger

    native = qsa_mixed_ops.attention_for_selection
    first = True

    @lru_cache(maxsize=80)
    def suffix_rows(bounds, limit, device):
        _, spans, _, _ = layout(bounds, limit, device)
        ranges = [torch.arange(a, b, dtype=torch.int32, device=device) for a, b in spans if b > a]
        return torch.cat(ranges) if ranges else None

    def attention(q, k, v, selection, scale, is_thd):
        nonlocal first
        batch, length, _, width = q.shape
        if (
            not is_thd
            or batch != 1
            or length == 0
            or width != 256
            or selection.compress_ratio != 4
            or selection.selected_ids is None
            or Envs.RELAX_QWEN38_QSA_MIXED_LONG_BACKEND != "grouped"
        ):
            return native(q, k, v, selection, scale, is_thd)
        if Envs.RELAX_QWEN38_QSA_MIXED_FLASH_MAX_LEN != -1 or Envs.RELAX_QWEN38_QSA_MIXED_FLEX_MAX_LEN != 0:
            raise RuntimeError("Direct QSA needs its audited document routing")
        ids = selection.selected_ids.view(length, -1)
        positions = selection.positions.view(-1)
        bounds = getattr(selection, "_relax_qsa_mixed_bounds", None)
        if not bounds or bounds[0] != 0 or bounds[-1] != length:
            raise ValueError("Missing producer document layout")
        limit = 4 * ids.shape[-1] + 3
        p, s, cu, maximum = layout(bounds, limit, q.device)

        def take(t, spans):
            return torch.cat([t[0, a:b] for a, b in spans if b > a], dim=0)

        dense = flash_attn_varlen_func(
            take(q, p), take(k, p), take(v, p), cu, cu, maximum, maximum, softmax_scale=scale, causal=True
        )
        rows = suffix_rows(bounds, limit, q.device)
        if rows is None:
            return dense.unsqueeze(0)
        suffix = DirectGrouped.apply(take(q, s), k[0], v[0], ids, positions, rows, scale)
        pieces = []
        pi = si = 0
        for (a, b), (_, end) in zip(p, s):
            n = b - a
            pieces.append(dense[pi : pi + n])
            pi += n
            if end > b:
                n = end - b
                pieces.append(suffix[si : si + n])
                si += n
        if first:
            get_logger(__name__).info(
                "QSA compressed block kernels active: rows=%d suffix=%d topk=%d", length, rows.numel(), ids.shape[-1]
            )
            first = False
        return torch.cat(pieces).unsqueeze(0)

    qsa_mixed_ops.attention_for_selection = attention
    return native, attention
