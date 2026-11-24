# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Shared sparse GQA forward, with standard masked SDPA gradients."""

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from torch.autograd.function import FunctionCtx


@triton.jit
def _pack_slots(SLOTS, COUNTS, PTR, OUTPUT, TOPK: tl.constexpr):
    row = tl.program_id(0)
    offset = tl.arange(0, TOPK)
    count = tl.load(COUNTS + row)
    start = tl.load(PTR + row)
    slots = tl.load(SLOTS + row * TOPK + offset, mask=offset < count, other=0)
    tl.store(OUTPUT + start + offset, slots, mask=offset < count)


def compact_indices(indices: torch.Tensor, valid: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Put valid selections in token order, followed by unused slots."""
    sentinel = torch.iinfo(torch.int32).max
    ordered = indices.masked_fill(~valid, sentinel).sort(-1).values
    counts = valid.sum(-1, dtype=torch.int32)
    ordered = ordered.masked_fill(ordered == sentinel, 0).int()
    if ordered.shape[-1] < 2048:
        ordered = F.pad(ordered, (0, 2048 - ordered.shape[-1]))
    return ordered.contiguous(), counts.contiguous()


class _SparseAttention(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: FunctionCtx,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        slots: torch.Tensor,
        counts: torch.Tensor,
        support: torch.Tensor | None,
    ) -> torch.Tensor:
        from sglang.kernels.ops.attention.decode_attention import decode_attention_fwd

        query, key, value = query.contiguous(), key.contiguous(), value.contiguous()
        output = torch.empty_like(query)
        # A padded query needs a valid CSR interval for the native kernel.
        # Its result is zeroed before leaving this operation.
        safe_counts = counts.clamp_min(1)
        indptr = torch.cat((torch.zeros(1, device=query.device, dtype=torch.int32), safe_counts.cumsum(0).int()))
        indices = torch.empty(query.shape[0] * 2048, device=query.device, dtype=torch.int32)
        _pack_slots[(query.shape[0],)](slots, safe_counts, indptr, indices, 2048)
        partial = torch.empty((*query.shape[:2], 4, value.shape[-1]), device=query.device, dtype=torch.float32)
        lse = torch.empty((*query.shape[:2], 4), device=query.device, dtype=torch.float32)
        splits = torch.full((query.shape[0],), 4, device=query.device, dtype=torch.int32)
        decode_attention_fwd(
            query, key, value, output, indptr, indices, partial, lse, splits, 4, query.shape[-1] ** -0.5, 1.0, 1.0
        )
        output.masked_fill_((counts == 0)[:, None, None], 0)
        ctx.save_for_backward(query, key, value, *((support,) if support is not None else ()))
        return output

    @staticmethod
    def backward(ctx: FunctionCtx, grad: torch.Tensor) -> tuple:
        query, key, value, support = ctx.saved_tensors
        needs = ctx.needs_input_grad[:3]
        with torch.enable_grad():
            inputs = [tensor.detach().requires_grad_(need) for tensor, need in zip((query, key, value), needs)]
            q, k, v = inputs
            repeats = q.shape[1] // k.shape[1]
            # Recompute the same masked attention equations for gradients;
            # no inference-only kernel is asked to implement backward.
            reference = F.scaled_dot_product_attention(
                q.transpose(0, 1)[None],
                k.repeat_interleave(repeats, 1).transpose(0, 1)[None],
                v.repeat_interleave(repeats, 1).transpose(0, 1)[None],
                attn_mask=support[None, None],
            )[0].transpose(0, 1)
            gradients = torch.autograd.grad(reference, [tensor for tensor, need in zip(inputs, needs) if need], grad)
        iterator = iter(gradients)
        result = [next(iterator) if need else None for need in needs]
        return result[0], result[1], result[2], None, None, None


def sparse_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    slots: torch.Tensor,
    counts: torch.Tensor,
    support: torch.Tensor | None = None,
) -> torch.Tensor:
    """Use ordered physical KV slots; supply the dense mask for training
    backward."""
    return _SparseAttention.apply(query, key, value, slots, counts, support)
