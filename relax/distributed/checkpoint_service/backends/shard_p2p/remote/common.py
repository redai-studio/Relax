# Copyright (c) 2026 Relax Authors. All Rights Reserved.
from typing import Any

import torch


def validate_consistency_check(command: dict, *, audit_key: str) -> None:
    """All shard receivers reject a mismatched numerical-audit policy."""
    from relax.utils.env import Envs

    enabled = command.get("consistency_check")
    if type(enabled) is not bool or enabled != bool(Envs.RELAX_P2P_WEIGHT_CONSISTENCY_CHECK):
        raise ValueError("Shard weight consistency-check configuration differs between source and receiver")
    audit = command.get(audit_key, False)
    if type(audit) is not bool or audit and not enabled:
        raise ValueError("Shard numerical audit requires the consistency-check switch")


def layout_of(tensor: torch.Tensor) -> dict:
    if not tensor.is_contiguous():
        raise NotImplementedError("P2P currently requires contiguous target parameters")
    return dict(
        shape=list(tensor.shape),
        stride=list(tensor.stride()),
        dtype=str(tensor.dtype),
        nbytes=tensor.numel() * tensor.element_size(),
        address=tensor.data_ptr(),
    )


def create_agent() -> Any:
    import uuid

    from nixl._api import nixl_agent, nixl_agent_config, nixl_thread_sync_t

    agent = nixl_agent(
        f"relax-p2p-{uuid.uuid4()}",
        nixl_agent_config(backends=["UCX"], sync_mode=nixl_thread_sync_t.NIXL_THREAD_SYNC_STRICT),
    )
    if "UCX" not in agent.get_plugin_list():
        raise RuntimeError("NIXL UCX plugin is required for P2P")
    return agent
