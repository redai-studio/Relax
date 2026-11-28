# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Shared router selection for Qwen3.8 bitwise training and serving."""

import torch


def stable_topk(logits: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    # Stable descending order breaks ties by ascending expert ID and fixes softmax's input order.
    values, indices = torch.sort(logits, dim=-1, descending=True, stable=True)
    return values[..., :k].contiguous(), indices[..., :k].contiguous()


def serving_topk(
    hidden_states: torch.Tensor, gating_output: torch.Tensor, topk: int, renormalize: bool
) -> tuple[torch.Tensor, torch.Tensor]:
    logits = gating_output.float()
    values, indices = stable_topk(logits, topk)
    if renormalize:
        weights = torch.softmax(values, dim=-1, dtype=torch.float32)
    else:
        weights = torch.softmax(logits, dim=-1, dtype=torch.float32).gather(-1, indices)
    return weights, indices.to(torch.int32)
