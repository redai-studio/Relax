# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Expanded DeepEP manager with native dispatch and expert weight-gradient
overlap."""

import os

import torch
from megatron.core.transformer.moe import token_dispatcher as td

from relax.models.common.deepep_expanded import ExpandedCombine, ExpandedDispatch


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
            CAPACITY,
            self.router_topk,
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


def install():
    from functools import wraps

    from relax.backends.megatron.actor import MegatronTrainRayActor

    native = MegatronTrainRayActor.train_actor

    @wraps(native)
    def train_actor(self, rollout_id, *positional, **kwargs):
        result = native(self, rollout_id, *positional, **kwargs)
        # The final actor step includes checkpointing and local SFT evaluation.
        if rollout_id + 1 == self.args.num_rollout:
            release()
        return result

    MegatronTrainRayActor.train_actor = train_actor
    os.environ["EP_DISABLE_GIN"] = "1"
    td._DeepepV2Manager = ExpandedManager
