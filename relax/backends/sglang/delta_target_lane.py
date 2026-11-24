# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Bind format-1 lane coordinates to observed ordinary FULL copies."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import torch

from relax.distributed.weight_sync.target_shard import TP_SIZE, split_axis

from .delta_shard_layout import BroadcastTargets, validate_shard_storage


@dataclass
class ExpertTargets:
    rank: int
    bindings: BroadcastTargets
    targets: dict[str, tuple[torch.Tensor, int, int]]
    broadcast = False

    @classmethod
    def build(cls, receiver: Any) -> ExpertTargets | None:
        bindings = receiver._broadcast_plan
        if bindings is None or receiver.force_full_seed:
            return None
        rank = receiver.active["control"]["target_shard_rank"]
        targets = {}
        slices = defaultdict(list)
        for name, (shape, dtype) in receiver.schema.items():
            axis = split_axis(name, shape)
            if axis is None:
                continue
            region = bindings.targets.get(name)
            if region is None or name in receiver.full_only or dtype != "bfloat16":
                return None
            rows, cols = shape
            length = rows * cols // TP_SIZE
            if axis == 0 or rows == 1:
                expected = (rank * length, 1, length, length)
            else:
                width = cols // TP_SIZE
                expected = (rank * width, rows, cols, width)
            if (region.offset, region.rows, region.stride, region.width) != expected:
                return None
            targets[name] = (region.weight, region.base, length)
            slices[region.weight].append((region.base, region.base + length))
        # Do not acknowledge a partial set of HF inputs for a packed Parameter.
        # Compare actual physical ranges, independent of expert count and order.
        for weight, intervals in slices.items():
            end = 0
            for start, stop in sorted(intervals):
                if start != end:
                    return None
                end = stop
            if end != weight.numel():
                return None
        if not targets or not validate_shard_storage(receiver.model, receiver.destinations, ()):
            return None
        return cls(rank, bindings, targets)

    def check_run(self, model: Any, batch: list) -> None:
        self.bindings.check_run(model, batch)

    def current(self, receiver: Any, raw: torch.Tensor, status: torch.Tensor) -> bool:
        return self.bindings.current(receiver, raw, status)

    def check(self, receiver: Any, raw: torch.Tensor, status: torch.Tensor) -> None:
        if not self.current(receiver, raw, status):
            raise ValueError("Changed or aliased target shard layout")
