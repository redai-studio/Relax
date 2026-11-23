# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Autograd operations for expanded DeepEP dispatch and combine."""

import torch


def _event_args(buffer, asynchronous, allocate):
    return dict(
        previous_event=buffer.capture() if asynchronous and allocate else None,
        async_with_compute_stream=asynchronous,
        allocate_on_comm_stream=allocate,
    )


class ExpandedDispatch(torch.autograd.Function):
    @staticmethod
    def forward(ctx, buffer, x, indices, probs, experts, sms, asynchronous, allocate, capacity, topk):
        assert x.ndim == 2 and x.shape[0] <= capacity
        assert x.dtype == torch.bfloat16 and probs.dtype == torch.float32
        assert indices.shape == probs.shape and indices.shape == (x.shape[0], topk)
        out, _, out_probs, handle, event = buffer.dispatch(
            x.contiguous(),
            topk_idx=indices,
            topk_weights=probs,
            num_experts=experts,
            num_max_tokens_per_rank=capacity,
            expert_alignment=1,
            num_sms=sms,
            do_expand=True,
            do_cpu_sync=True,
            **_event_args(buffer, asynchronous, allocate),
        )
        if asynchronous:
            event.current_stream_wait()
        counts = tuple(handle.num_recv_tokens_per_expert_list)
        assert handle.do_expand and handle.num_max_tokens_per_rank == capacity
        assert out_probs.ndim == 1 and out.shape[0] == out_probs.shape[0] == sum(counts)
        ctx.buffer, ctx.handle = buffer, handle
        ctx.asynchronous, ctx.allocate = asynchronous, allocate
        ctx.set_materialize_grads(False)
        return out, out_probs, handle, counts

    @staticmethod
    def backward(ctx, dx, dp, _handle, _counts):
        # The actual expert path consumes both outputs; fail closed on an untested branch.
        assert dx is not None and dp is not None
        assert dp.ndim == 1
        gx, gp, event = ctx.buffer.combine(
            dx.contiguous(),
            ctx.handle,
            topk_weights=dp.float().contiguous(),
            num_sms=ctx.handle.num_sms,
            **_event_args(ctx.buffer, ctx.asynchronous, ctx.allocate),
        )
        if ctx.asynchronous:
            event.current_stream_wait()
        assert gp.shape == ctx.handle.topk_idx.shape
        return None, gx, None, gp, None, None, None, None, None, None


class ExpandedCombine(torch.autograd.Function):
    @staticmethod
    def forward(ctx, buffer, x, handle, asynchronous, allocate):
        assert handle.do_expand
        out, _, event = buffer.combine(
            x.contiguous(), handle, num_sms=handle.num_sms, **_event_args(buffer, asynchronous, allocate)
        )
        if asynchronous:
            event.current_stream_wait()
        ctx.buffer, ctx.handle = buffer, handle
        ctx.asynchronous, ctx.allocate = asynchronous, allocate
        return out

    @staticmethod
    def backward(ctx, grad):
        out, _, _, _, event = ctx.buffer.dispatch(
            grad.contiguous(),
            handle=ctx.handle,
            do_expand=True,
            num_sms=ctx.handle.num_sms,
            **_event_args(ctx.buffer, ctx.asynchronous, ctx.allocate),
        )
        if ctx.asynchronous:
            event.current_stream_wait()
        return None, out, None, None, None
