# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Keep Qwen4 MoE routing in FP32 without changing FSDP parameter storage."""

from __future__ import annotations

from types import MethodType

import torch
import torch.nn.functional as F


def _qwen4_router_forward_fp32(
    self: torch.nn.Module, hidden_states: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    hidden_states = hidden_states.reshape(-1, self.hidden_dim)
    # FP32 softmax alone cannot recover expert ordering lost in BF16 logits.
    # Disable outer autocast as well: it otherwise lowers this linear to BF16.
    with torch.autocast(device_type=hidden_states.device.type, enabled=False):
        router_logits = F.linear(hidden_states.float(), self.weight.float())
        router_probs = F.softmax(router_logits, dim=-1, dtype=torch.float32)
        router_scores, router_indices = torch.topk(router_probs, self.top_k, dim=-1)
        if self.norm_topk_prob:
            router_scores = router_scores / router_scores.sum(dim=-1, keepdim=True)
    return router_logits, router_scores, router_indices


def enable_qwen4_router_fp32(model: torch.nn.Module) -> int:
    """Patch this model's routers before FSDP wrapping; preserve parameter
    names/dtypes."""
    from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpTextTopKRouter

    patched = 0
    for module in model.modules():
        if isinstance(module, Qwen4ExpTextTopKRouter):
            module.forward = MethodType(_qwen4_router_forward_fp32, module)
            patched += 1
    if not patched:
        raise RuntimeError("Qwen4 FP32 routing requested, but no Qwen4ExpTextTopKRouter modules were found")
    return patched
