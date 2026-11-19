# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""SenseNova image training adapters."""

from typing import Any


__all__ = ["SenseNovaU1ImageEditAdapter"]


def __getattr__(name: str) -> Any:
    if name == "SenseNovaU1ImageEditAdapter":
        from .rl_adapter import SenseNovaU1ImageEditAdapter

        return SenseNovaU1ImageEditAdapter
    raise AttributeError(name)
