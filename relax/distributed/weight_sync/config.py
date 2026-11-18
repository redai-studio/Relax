# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Opt-in lossless delta settings; execution lifecycle belongs to Actor/DCS."""

from __future__ import annotations

import json
from dataclasses import dataclass, fields
from pathlib import Path


@dataclass(frozen=True)
class DeltaSyncConfig:
    stream: str
    bucket_bytes: int = 512 << 20
    gpu_budget_bytes: int = 16 << 30
    host_budget_bytes: int = 8 << 30
    receiver_gpu_budget_bytes: int = 12 << 30
    staging_budget_bytes: int = 512 << 20
    sync_timeout_s: int = 120
    anchor_interval: int = 20
    full_only: bool = False
    transport: str = "tcp"

    @classmethod
    def load(cls, path: str | Path) -> DeltaSyncConfig:
        def unique(pairs: list[tuple]) -> dict:
            result = {}
            for key, value in pairs:
                if key in result:
                    raise ValueError(f"Duplicate delta configuration key: {key}")
                result[key] = value
            return result

        raw = Path(path).read_bytes()
        if len(raw) > 65536:
            raise ValueError("Delta configuration exceeds its bounded control size")
        value = json.loads(raw, object_pairs_hook=unique)
        if not isinstance(value, dict) or set(value) - {field.name for field in fields(cls)}:
            raise ValueError("Unknown delta settings; model layout selection is owned by the existing converter")
        config = cls(**value)
        if not isinstance(config.stream, str) or not 0 < len(config.stream) <= 256:
            raise ValueError("Delta configuration requires a bounded nonempty stream identity")
        for name in (
            "bucket_bytes",
            "gpu_budget_bytes",
            "host_budget_bytes",
            "receiver_gpu_budget_bytes",
            "staging_budget_bytes",
            "sync_timeout_s",
            "anchor_interval",
        ):
            if type(getattr(config, name)) is not int or getattr(config, name) <= 0:
                raise ValueError(f"Delta setting {name} must be a positive integer")
        if type(config.full_only) is not bool or config.transport not in ("tcp", "nccl"):
            raise ValueError("Delta full_only must be boolean and transport must be tcp or nccl")
        return config
