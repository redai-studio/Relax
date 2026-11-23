# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Expanded DeepEP manager with native dispatch and expert weight-gradient
overlap."""

import os

import torch
from megatron.core.transformer.moe import fused_a2a as fa
from megatron.core.transformer.moe import token_dispatcher as td

from relax.models.common.deepep_expanded import ExpandedCombine, ExpandedDispatch


CAPACITY = 131072


class ExpandedManager(td._DeepepV2Manager):
    def __init__(self, group, num_local_experts, router_topk, num_experts, config):
        assert group.size() == 8 and num_experts == 256 and num_local_experts == 32
        assert router_topk == 8 and config.moe_router_dtype == "fp32"
        assert not config.moe_router_padding_for_quantization
        assert not config.fp8 and not config.fp4 and not config.use_transformer_engine_op_fuser
        assert config.moe_flex_dispatcher_num_sms in (64, 80)
        assert not getattr(config, "moe_combine_in_fp32", False)
        assert not getattr(config, "moe_apply_probs_on_input", False)
        assert config.moe_expert_capacity_factor is None
        super().__init__(group, num_local_experts, router_topk, num_experts, config)

    def _get_buffer(self, hidden_states):
        assert hidden_states.shape[0] <= CAPACITY and hidden_states.shape[1] == 2048
        self.buffer = fa.get_elastic_buffer(self.group, CAPACITY, 2048, 8)
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
