# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Rectangular sequence dimensions for the shared V4.1 carry codec."""

from __future__ import annotations

from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from ..runtime import V41CarryPlan


def carry_dimensions(plan: V41CarryPlan, s: int, b: int) -> tuple[int, dict[int, int], int]:
    local_length = s * plan.tp_size if plan.sequence_parallel else s
    global_length = local_length * plan.cp_size
    compressed_lengths = {}
    for key in (*plan.latent_keys, *plan.index_k_keys):
        ratio = plan.compress_ratios[key]
        if plan.cp_size > 1 and local_length % ratio:
            raise ValueError("DeepSeek V4.1: CP chunk length must be divisible by its compression ratio.")
        compressed_lengths[key] = global_length // ratio
    candidate_width = global_length // plan.candidate_ratio if plan.candidates and plan.candidate_ratio > 0 else 0
    return local_length, compressed_lengths, candidate_width
