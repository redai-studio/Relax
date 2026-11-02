# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Fuse the shared expert output gate while preserving overlap state and
streams."""

_installed = False


def install(model, gate_function, warps=4):
    global _installed
    if _installed:
        return
    import torch
    from megatron.core.transformer.moe.shared_experts import SharedExpertMLP

    assert warps in (4, 8)
    modules = [m for chunk in model for m in chunk.modules() if isinstance(m, SharedExpertMLP)]
    assert len(modules) == 41
    assert all(
        m.config.moe_shared_expert_overlap
        and m.use_shared_expert_gate
        and m.config.hidden_size == 2048
        and m.gate_weight.dtype == torch.bfloat16
        for m in modules
    )

    _installed = True


def apply(x, gate, warps=4):
    import torch

    from .ops.shared_gate import fused_gate as gate_function

    assert x.shape[-1] == 2048 and gate.shape == (*x.shape[:-1], 1)
    rows = x.numel() // x.shape[-1]
    if not (x.is_contiguous() and gate.is_contiguous() and 32768 <= rows <= 131072):
        return x * gate
    result = gate_function(x, gate, warps)
    if torch.is_grad_enabled():
        assert result.grad_fn is not None and type(result.grad_fn).__name__ == "GateBackward"
    return result
