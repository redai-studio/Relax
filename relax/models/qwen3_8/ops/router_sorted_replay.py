# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Restrict deterministic checkpoint routing order to the audited Qwen38
case."""

from __future__ import annotations

from typing import Any

import torch

from relax.utils.env import Envs


def use_sorted_topk(
    logits: torch.Tensor,
    topk: int,
    use_pre_softmax: bool,
    num_groups: int | None,
    group_topk: int | None,
    scaling_factor: float | None,
    score_function: str,
    expert_bias: torch.Tensor | None,
    fused: bool,
    router_replay: Any,
    dense_output: bool,
    qb_histogram: Any = None,
    qb_bin_bounds: Any = None,
    topk_indices: Any = None,
) -> bool:
    return (
        not torch.is_grad_enabled()
        and not Envs.ENABLE_ROUTING_REPLAY
        and qb_histogram is None
        and qb_bin_bounds is None
        and topk_indices is None
        and router_replay is None
        and not fused
        and not dense_output
        and not use_pre_softmax
        and not num_groups
        and not group_topk
        and score_function == "softmax"
        and expert_bias is None
        and scaling_factor is None
        and topk == 10
        and logits.ndim == 2
        and logits.shape[1] == 512
        and logits.dtype == torch.float32
        and logits.is_cuda
    )
