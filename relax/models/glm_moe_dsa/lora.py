# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""LoRA accessors for GLM's absorbed KV projection and fused norms."""

import torch
import torch.distributed as dist
from torch import nn


def fused_norm_weight(projection: nn.Module) -> torch.Tensor:
    """Read the frozen fused norm through a Bridge adapter wrapper."""
    return getattr(projection, "to_wrap", projection).layer_norm_weight


def absorbed_kv_weight(projection: nn.Module) -> torch.Tensor:
    """Form the column-parallel KV weight without detaching adapter gradients.

    Bridge's export-oriented weight merge uses a non-differentiable all-gather.
    Absorbed MLA uses this weight during training, so A's backward must sum
    contributions from every TP head shard before returning its local shard.
    """
    if not hasattr(projection, "adapter"):
        return projection.weight
    base = projection.to_wrap.weight
    if not projection._adapter_enabled:
        return base
    adapter = projection.adapter
    if projection.training and getattr(adapter.dropout, "p", 0.0):
        raise ValueError("GLM absorbed KV LoRA requires --lora-dropout 0 when targeting linear_kv_up_proj.")

    a = adapter.linear_in.weight
    b = adapter.linear_out.weight
    group = getattr(adapter, "tp_group", None)
    if a.shape[0] != adapter.dim:
        if group is None or a.shape[0] * dist.get_world_size(group) != adapter.dim:
            raise ValueError("GLM KV LoRA A shard does not match its explicit tensor-parallel group.")
        from torch.distributed.nn.functional import all_gather

        a = torch.cat(all_gather(a, group=group), dim=0)
    delta = (b @ a) * (adapter.alpha / adapter.dim)
    if delta.shape != base.shape:
        raise ValueError(f"GLM KV LoRA delta shape {tuple(delta.shape)} != base shape {tuple(base.shape)}.")
    return (base + delta).to(dtype=base.dtype)
