# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Explicit tensor-payload budgets for synchronous ART acquisition/replay."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import torch


def tensor_payload_bytes(batch: Mapping[str, torch.Tensor]) -> int:
    # Count views conservatively, rather than assuming shared backing storage.
    return sum(value.numel() * value.element_size() for value in batch.values() if isinstance(value, torch.Tensor))


def art_cache_limits(sampling: Mapping[str, Any]) -> tuple[int, int]:
    config = sampling.get("art", {})
    limits = tuple(config.get(key) for key in ("max_cache_bytes", "max_replay_bytes"))
    if any(type(value) is not int or value < 1 for value in limits):
        raise ValueError("ART requires positive integer max_cache_bytes and max_replay_bytes tensor budgets.")
    return limits


def check_art_cache_budget(sampling: Mapping[str, Any], sizes: Sequence[int], guidance_sizes: Sequence[int]) -> None:
    """Check publisher-reported sizes BEFORE loading a group's tensors.

    Replay reserves all group directions plus eight times the largest group for
    the full cache, B1 selection and FP32 covariance working tensors. This is a
    conservative logical tensor budget, not an RSS, Plasma spill, allocator,
    CUDA workspace or whole-service memory guarantee.
    """
    cache_limit, replay_limit = art_cache_limits(sampling)
    if not sizes or len(sizes) != len(guidance_sizes) or any(value <= 0 for value in [*sizes, *guidance_sizes]):
        raise ValueError("ART cache byte descriptors must be positive and complete.")
    if sum(sizes) > cache_limit:
        raise ValueError("ART acquisition exceeds max_cache_bytes; reduce candidates/resolution/selected steps.")
    if sum(guidance_sizes) + 8 * max(sizes) > replay_limit:
        raise ValueError("ART covariance/replay tensor working estimate exceeds max_replay_bytes.")
