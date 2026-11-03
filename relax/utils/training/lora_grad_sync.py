# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import torch


def inherit_lora_tp_grad_sync(model: torch.nn.Module) -> int:
    """Inherit replicated HF linear TP averaging on newly injected adapters.

    Bridge marks existing HF parameters before LoRA injection, but its
    instance-level ``__setattr__`` hook does not intercept normal submodule
    replacement. Copy the base linear's explicit averaging contract after
    injection and before DDP construction. Only Bridge's exact, unsharded
    LinearAdapter is eligible; parallel/custom adapters and parameters with
    an explicit TP-sharding or SP/SUM contract retain their existing rules.

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
        if not getattr(base.weight, "average_gradients_across_tp_domain", False):
            continue
        parameters = tuple(adapter.parameters())
        if any(
            getattr(parameter, flag, False)
            for parameter in parameters
            for flag in ("tensor_model_parallel", "sequence_parallel", "sum_gradients_across_tp_domain")
        ):
            continue
        for parameter in parameters:
            if not getattr(parameter, "average_gradients_across_tp_domain", False):
                parameter.average_gradients_across_tp_domain = True
                marked += 1
    return marked
