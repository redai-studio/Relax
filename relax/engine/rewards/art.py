# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Raw scalar rewards for the canonical image ART profile."""

from __future__ import annotations

import math
from numbers import Real
from typing import Mapping, Sequence

import torch


def scalarize_art_rewards(
    components: Mapping[str, Sequence[float]],
    weights: Mapping[str, float],
    *,
    required: Sequence[str],
    valid_mask: Sequence[bool],
) -> list[float]:
    """Weight raw components BEFORE covariance, without centering or clipping.

    Invalid judges fail the acquisition, not a partially filtered K-group. An
    explicit weight mapping is required so adding a scorer cannot silently
    alter the objective. Zero/negative finite component weights are supported.
    """
    count = len(valid_mask)
    if not count or not all(valid_mask):
        raise ValueError("ART requires complete valid reward groups; a failed judge is not a zero score.")
    if not weights or set(weights) != set(components) or not set(required) <= set(components):
        raise ValueError("ART requires exact component/weight keys and every required reward component.")
    result = torch.zeros(count, dtype=torch.float32)
    # Match upstream weighted_score: stable component order and separate FP32
    # multiplication/addition, not add_(alpha=weight)'s fused rounding.
    for name in sorted(components):
        values = components[name]
        tensor = torch.tensor(values, dtype=torch.float32)
        if tensor.shape != (count,) or not bool(torch.isfinite(tensor).all()):
            raise ValueError(f"ART component {name!r} must contain one finite score per candidate.")
        weight = weights[name]
        if isinstance(weight, bool) or not isinstance(weight, Real) or not math.isfinite(weight):
            raise ValueError(f"ART component {name!r} has a non-finite weight.")
        result = result + float(weight) * tensor
    if not bool(torch.isfinite(result).all()):
        raise ValueError("ART scalar reward overflowed FP32.")
    return result.tolist()
