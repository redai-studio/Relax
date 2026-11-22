# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Expanded DeepEP manager with native dispatch and expert weight-gradient
overlap."""

import os

import torch
from megatron.core.transformer.moe import token_dispatcher as td


CAPACITY = 65536


_owned_buffer = None


def get_buffer(group):
    global _owned_buffer
    if _owned_buffer is None:
        from _relax_deep_ep_v2 import ElasticBuffer

        kwargs = dict(num_max_tokens_per_rank=CAPACITY, hidden=2560, num_topk=10)
        # PyTorch lazily creates an NCCL communicator on the first collective.
        # DeepEP reads its raw handle immediately inside get_buffer_size_hint.
        token = torch.ones(1, device=torch.cuda.current_device(), dtype=torch.int32)
        torch.distributed.all_reduce(token, group=group)
        torch.cuda.current_stream().synchronize()
        backend = group._get_backend(torch.device("cuda"))
        if hasattr(backend, "_comm_ptr") and not backend._comm_ptr():
            raise RuntimeError("EP NCCL group has no usable communicator after initialization")
        size = ElasticBuffer.get_buffer_size_hint(group, **kwargs)
        _owned_buffer = ElasticBuffer(group, num_bytes=size, explicitly_destroy=True, **kwargs)
    elif _owned_buffer.group is not group:
        raise RuntimeError("Expanded dispatch changed its EP group")
    return _owned_buffer


def release():
    global _owned_buffer
    if _owned_buffer is None:
        return
    from relax.utils.logging_utils import get_logger

    torch.cuda.synchronize()
    _owned_buffer.destroy()
    _owned_buffer = None
    get_logger(__name__).info("Released DeepEPv2 elastic buffer before EP communicator teardown")


def _event_args(buffer, asynchronous, allocate):
    return dict(
        previous_event=buffer.capture() if asynchronous and allocate else None,
        async_with_compute_stream=asynchronous,
        allocate_on_comm_stream=allocate,
    )


class ExpandedDispatch(torch.autograd.Function):
    @staticmethod
    def forward(ctx, buffer, x, indices, probs, experts, sms, asynchronous, allocate):
        assert x.ndim == 2 and x.shape[0] <= CAPACITY and x.shape[1] == 2560
        assert x.dtype == torch.bfloat16 and probs.dtype == torch.float32
        assert indices.shape == probs.shape and indices.shape == (x.shape[0], 10)
        out, _, out_probs, handle, event = buffer.dispatch(
            x.contiguous(),
            topk_idx=indices,
            topk_weights=probs,
            num_experts=experts,
            num_max_tokens_per_rank=CAPACITY,
            expert_alignment=1,
            num_sms=sms,
            do_expand=True,
            do_cpu_sync=True,
            **_event_args(buffer, asynchronous, allocate),
        )
        if asynchronous:
            event.current_stream_wait()
        counts = tuple(handle.num_recv_tokens_per_expert_list)
        assert handle.do_expand and handle.num_max_tokens_per_rank == CAPACITY
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
        return None, gx, None, gp, None, None, None, None


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


class ExpandedManager(td._DeepepV2Manager):
    def __init__(self, group, num_local_experts, router_topk, num_experts, config):
        assert group.size() == 8 and num_experts == 512 and num_local_experts == 64
        assert router_topk == 10 and config.moe_router_dtype == "fp32"
        assert not config.moe_router_padding_for_quantization
        assert not config.fp8 and not config.fp4 and not config.use_transformer_engine_op_fuser
        assert config.moe_flex_dispatcher_num_sms in (64, 80)
        assert not getattr(config, "moe_combine_in_fp32", False)
        assert not getattr(config, "moe_apply_probs_on_input", False)
        assert config.moe_expert_capacity_factor is None
        super().__init__(group, num_local_experts, router_topk, num_experts, config)

    def setup_metadata(self, routing_map, probs):
        if routing_map.dtype != torch.bool:
            idx = routing_map.reshape(-1, self.router_topk).long()
            scores = probs.reshape(-1, self.num_experts)
            self.token_indices = idx
            self.token_probs = scores.gather(1, idx.clamp_min(0)).masked_fill(idx < 0, 0.0)
            return
        route = routing_map.reshape(-1, self.num_experts)
        p = probs.reshape_as(route)
        selected, indices = torch.topk(p.masked_fill(~route, -torch.inf), self.router_topk, dim=1)
        present = route.gather(1, indices)
        self.token_indices = indices.masked_fill(~present, -1)
        self.token_probs = selected.masked_fill(~present, 0.0)

    def _get_buffer(self, hidden_states):
        assert hidden_states.shape[0] <= CAPACITY and hidden_states.shape[1] == 2560
        self.buffer = get_buffer(self.group)
        physical = tuple(self.buffer.get_physical_domain_size())
        logical = tuple(self.buffer.get_logical_domain_size())
        assert physical == logical == (1, 8), (physical, logical)
        assert os.environ.get("EP_DISABLE_GIN") == "1"
        if not getattr(self, "_nvlink_domain_reported", False):
            import json

            from relax.utils.logging_utils import get_logger

            get_logger(__name__).info(
                "EXPANDED_NVLINK_ONLY_DOMAIN "
                + json.dumps(
                    dict(
                        rank=self.group.rank(),
                        physical=physical,
                        logical=logical,
                        ep_size=self.group.size(),
                        only_intra_node=True,
                    )
                ),
            )
            self._nvlink_domain_reported = True
        return self.buffer

    def dispatch(self, hidden_states, async_finish=False, allocate_on_comm_stream=False):
        assert self.token_probs.dtype == torch.float32
        hidden, self.dispatched_probs, self.handle, counts = ExpandedDispatch.apply(
            self._get_buffer(hidden_states),
            hidden_states,
            self.token_indices,
            self.token_probs,
            self.num_experts,
            self.num_sms,
            async_finish,
            allocate_on_comm_stream,
        )
        self.tokens_per_expert = torch.tensor(counts, dtype=torch.long)
        return hidden

    def get_permuted_hidden_states_by_experts(self, hidden_states):
        assert self.dispatched_probs.ndim == 1
        return hidden_states, self.dispatched_probs

    def get_restored_hidden_states_by_experts(self, hidden_states):
        return hidden_states

    def combine(self, hidden_states, async_finish=False, allocate_on_comm_stream=False):
        out = ExpandedCombine.apply(self.buffer, hidden_states, self.handle, async_finish, allocate_on_comm_stream)
        self.handle = None
        self.dispatched_probs = None
        return out


def install(args):
    from functools import wraps

    from relax.backends.megatron import actor, model

    native = model.train
    if actor.train is not native:
        raise RuntimeError("Expected a shared Megatron training entrypoint")

    @wraps(native)
    def train(rollout_id, *positional, **kwargs):
        result = native(rollout_id, *positional, **kwargs)
        if rollout_id + 1 == args.num_rollout:
            release()
        return result

    actor.train = model.train = train
    os.environ["EP_DISABLE_GIN"] = "1"
    td._DeepepV2Manager = ExpandedManager
