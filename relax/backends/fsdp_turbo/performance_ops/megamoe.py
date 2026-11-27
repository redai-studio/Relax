# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""MegaMoE through FSDPTurbo's existing custom expert dispatcher."""

import torch
from torch.distributed.tensor import DTensor

from relax.utils.logging_utils import get_logger


logger = get_logger(__name__)
_EXECUTORS = {}


def megamoe_forward(self, hidden_states, top_k_index, top_k_weights, *, ep_group, fixed_router=False):
    from fsdp_turbo.distributed.expert_parallel.utils import fixed_router_for_debug, normalize_expert_args
    from hyper_multicore import MegaMoeExperts

    top_k_index, top_k_weights = normalize_expert_args(top_k_index, top_k_weights)
    shape = hidden_states.shape
    hidden_states = hidden_states.reshape(-1, shape[-1])
    if fixed_router:
        top_k_index, top_k_weights = fixed_router_for_debug(top_k_index, top_k_weights, self.num_global_experts)
    gate_up = self.gate_up_proj.to_local() if isinstance(self.gate_up_proj, DTensor) else self.gate_up_proj
    down = self.down_proj.to_local() if isinstance(self.down_proj, DTensor) else self.down_proj
    key = (ep_group, hidden_states.device, shape[-1], down.shape[-1], self.num_global_experts, top_k_index.shape[-1])
    executor = _EXECUTORS.get(key)
    if executor is None:
        executor = MegaMoeExperts(
            hidden_size=shape[-1],
            intermediate_size=down.shape[-1],
            num_experts=self.num_global_experts,
            top_k=top_k_index.shape[-1],
            ep_group=ep_group,
            create_parameters=False,
            expert_weight_layout="EOI",
            dispatch_mode="push",
            apply_probs_before_fc2=False,
        )
        _EXECUTORS[key] = executor
        logger.info(
            f"[megamoe] rank={torch.distributed.get_rank(ep_group)} "
            f"gate_up={tuple(gate_up.shape)} down={tuple(down.shape)}"
        )
    return executor(
        hidden_states,
        top_k_index.contiguous(),
        top_k_weights.contiguous(),
        expert_weights=(gate_up, down),
    ).reshape(shape)
