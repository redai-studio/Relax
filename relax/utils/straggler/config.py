# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class StragglerConfig:
    enabled: bool = False
    interval: int = 10
    relative_threshold: float = 0.10
    absolute_ms_threshold: float = 5.0
    persist_windows: int = 3
    enable_module_stages: bool = False
    max_pending_events: int = 16384

    @classmethod
    def from_args(cls, args) -> StragglerConfig:
        return cls(
            enabled=bool(getattr(args, "straggler_analysis", False)),
            interval=int(getattr(args, "straggler_interval", 10) or 10),
            relative_threshold=float(getattr(args, "straggler_relative_threshold", 0.10)),
            absolute_ms_threshold=float(getattr(args, "straggler_absolute_ms_threshold", 5.0)),
            persist_windows=int(getattr(args, "straggler_persist_windows", 3) or 3),
            enable_module_stages=bool(getattr(args, "straggler_enable_module_stages", False)),
        )
