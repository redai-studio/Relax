# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Small SFT control-plane collectives without Ray/actor imports."""

from __future__ import annotations

from typing import Any

import torch.distributed as dist


def collect_rank_objects(value: Any) -> list[Any]:
    if not dist.is_initialized() or dist.get_world_size(group=dist.group.WORLD) <= 1:
        return [value]
    from relax.utils.distributed_utils import get_gloo_group

    group = get_gloo_group()
    values = [None] * dist.get_world_size(group=group)
    dist.all_gather_object(values, value, group=group)
    return values


def collect_weight_sync_errors(local_error: str | None) -> list[tuple[int, str]]:
    return [(rank, error) for rank, error in enumerate(collect_rank_objects(local_error)) if error is not None]
