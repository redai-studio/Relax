# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Shared DSA selection with the same boundary tie-break for train and
decode."""

import torch


def dsa_topk_indices(scores: torch.Tensor) -> torch.Tensor:
    """Select up to 2048 keys, preferring earlier tokens when scores tie."""
    width = scores.shape[-1]
    if width <= 2048:
        return torch.arange(width, device=scores.device).expand(scores.shape[0], -1)

    from flashinfer import top_k

    return top_k(scores.detach().contiguous(), 2048, deterministic=True, tie_break=1, dsa_graph_safe=True)[1]
