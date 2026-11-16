# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from typing import Any

import torch
import torch.distributed as dist


def mark_expert_adapter_optimizer_replicas(optimizer: Any, *, expert_group: dist.ProcessGroup) -> int:
    """Exclude EP copies of shared adapters from distributed optimizer norms.

    Only optimizer-owned views may be marked: Bridge's model parameter-norm
    routine requires expert model parameters themselves to remain unshared.
    Expert-DP shards on every EP-rank-zero participant still contribute, and
    clipping/updates continue to visit all replicas. Non-distributed optimizers
    without a model-to-owned-shard mapping are outside this helper's scope.
    """
    if dist.get_rank(group=expert_group) == 0:
        return 0
    marked = 0
    for member in getattr(optimizer, "chained_optimizers", [optimizer]):
        for model_param, (group_index, group_order) in getattr(member, "model_param_group_index_map", {}).items():
            if not getattr(model_param, "expert_parallel_replicated", False):
                continue
            shard = member.optimizer.param_groups[group_index]["params"][group_order]
            if shard is model_param:
                raise RuntimeError("Shared expert norm exclusion requires an optimizer-owned parameter view")
            if not getattr(shard, "shared", False):
                shard.shared = True
                marked += 1
    return marked


def exclude_replicated_expert_adapter_grads_from_norm(optimizer: Any, *, ep_rank: int) -> None:
    """Count EP-replicated adapters once in DistributedOptimizer gradient
    norms.

    Mark only optimizer shards: Bridge's model parameter norm requires expert
    parameters to retain their original ``shared`` metadata.
    """
    if ep_rank == 0:
        return
    for child in getattr(optimizer, "chained_optimizers", (optimizer,)):
        for param, (group, index) in getattr(child, "model_param_group_index_map", {}).items():
            if getattr(param, "expert_parallel_replicated", False):
                child.optimizer.param_groups[group]["params"][index].shared = True


def inherit_lora_tp_grad_sync(model: torch.nn.Module) -> int:
    """Inherit replicated linear TP gradient reduction on newly injected
    adapters.

    Bridge marks existing HF parameters before LoRA injection, but its
    instance-level ``__setattr__`` hook does not intercept normal submodule
    replacement. Copy the base linear's explicit averaging or sequence-parallel
    SUM contract after injection and before DDP construction. Only Bridge's
    exact, unsharded LinearAdapter is eligible; parallel/custom adapters and
    parameters with a conflicting explicit contract retain their existing rules.

    Returns the number of parameter tensors newly marked; repeated calls are
    idempotent and return zero once every eligible adapter is marked.
    """
    try:
        from megatron.bridge.peft.lora_layers import LinearAdapter
    except ImportError:
        # Older Bridge versions do not expose the plain HF adapter class.
        return 0

    marked = 0
    for module in model.modules():
        base = getattr(module, "to_wrap", None)
        adapter = getattr(module, "adapter", None)
        if not isinstance(base, torch.nn.Linear) or type(adapter) is not LinearAdapter:
            continue
        if getattr(base.weight, "average_gradients_across_tp_domain", False):
            sync_flag = "average_gradients_across_tp_domain"
        elif getattr(base.weight, "sequence_parallel", False):
            sync_flag = "sequence_parallel"
        else:
            continue
        parameters = tuple(adapter.parameters())
        if any(
            getattr(parameter, flag, False)
            for parameter in parameters
            for flag in (
                "tensor_model_parallel",
                "sequence_parallel",
                "sum_gradients_across_tp_domain",
                "average_gradients_across_tp_domain",
            )
            if flag != sync_flag
        ):
            continue
        for parameter in parameters:
            if not getattr(parameter, sync_flag, False):
                setattr(parameter, sync_flag, True)
                marked += 1
    return marked
