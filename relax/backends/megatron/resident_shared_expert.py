# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Verify shared-gate parameter publication before retaining SFT all-
gathers."""

from __future__ import annotations

import os
from argparse import Namespace
from collections.abc import Sequence
from typing import Any

from relax.utils.env import Envs


def _verify_runtime_config(config: Any) -> None:
    if (
        config is None
        or getattr(config, "moe_token_dispatcher_type", None) != "flex"
        or getattr(config, "moe_flex_dispatcher_backend", None) != "deepep"
        or not getattr(config, "moe_shared_expert_overlap", False)
        or getattr(config, "cuda_graph_impl", "none") != "none"
        or getattr(config, "enable_cuda_graph", False)
    ):
        raise ValueError("Actual shared-expert configuration must use flex/DeepEP overlap without CUDA Graphs")


def _verify_gate_modules(
    model_chunks: Sequence[Any], ddp_type: type, shared_type: type, callback_attribute: str
) -> tuple[int, int]:
    """Inspect callback registration without dispatching a collective."""
    if not model_chunks:
        raise ValueError("Shared resident overlap requires at least one DDP model chunk")
    gates = trainable_gates = 0
    for chunk in model_chunks:
        if not isinstance(chunk, ddp_type) or any(
            not getattr(chunk.ddp_config, flag, False)
            for flag in ("overlap_param_gather", "overlap_grad_reduce", "use_distributed_optimizer")
        ):
            raise ValueError("Shared resident overlap requires DDP with parameter all-gather overlap")
        _verify_runtime_config(getattr(chunk, "config", None))
        for module in chunk.module.modules():
            if not isinstance(module, shared_type) or not module.use_shared_expert_gate:
                continue
            gates += 1
            _verify_runtime_config(getattr(module, "config", None))
            if getattr(type(module), "shared_gate_param_readiness_version", None) != 1:
                raise ValueError("Shared resident overlap requires the patched MCore shared-gate readiness capability")
            gate = module.gate_weight
            if gate.requires_grad:
                trainable_gates += 1
                if gate not in chunk.param_to_bucket_group or not callable(getattr(gate, callback_attribute, None)):
                    raise ValueError("A trainable shared gate has no DDP bucket readiness callback")
    return gates, trainable_gates


def verify_resident_shared_readiness(args: Namespace, model_chunks: Sequence[Any]) -> None:
    """Run after freezing and DDP setup, before any resident training call."""
    args._relax_shared_expert_gate_ready_verified = False
    if not Envs.RELAX_SFT_RESIDENT_PARAM_OVERLAP:
        return
    shared = getattr(args, "moe_shared_expert_overlap", False)
    if any(
        getattr(getattr(chunk, "config", None), "moe_shared_expert_overlap", False) != shared for chunk in model_chunks
    ):
        raise ValueError("Shared resident overlap args and actual model configuration disagree")
    if not shared:
        return
    if (
        getattr(args, "moe_token_dispatcher_type", None) != "flex"
        or getattr(args, "moe_flex_dispatcher_backend", None) != "deepep"
        or os.environ.get("CUDA_DEVICE_MAX_CONNECTIONS") != "32"
        or getattr(args, "cuda_graph_impl", "none") != "none"
        or getattr(args, "enable_cuda_graph", False)
    ):
        raise ValueError("Shared resident overlap is gated to flex/DeepEP, CUDA connections32 and no CUDA Graphs")
    from megatron.core.distributed import DistributedDataParallel
    from megatron.core.transformer.moe.shared_experts import SharedExpertMLP
    from megatron.core.utils import PARAM_READY_CALLBACK_ATTR

    counts = _verify_gate_modules(model_chunks, DistributedDataParallel, SharedExpertMLP, PARAM_READY_CALLBACK_ATTR)
    args._relax_shared_expert_gate_readiness_counts = counts
    args._relax_shared_expert_gate_ready_verified = True
