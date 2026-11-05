# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""One operation budget shared by synchronous preparation and installation."""

import math
import time
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Iterator


_deadline: ContextVar[float | None] = ContextVar("weight_operation_deadline", default=None)


def validate_timeout(seconds: float) -> float:
    if not math.isfinite(seconds) or seconds <= 0:
        raise ValueError("Operation timeout must be positive and finite")
    return seconds


@contextmanager
def time_budget(seconds: float, *, inherit: bool = True) -> Iterator[None]:
    validate_timeout(seconds)
    end = time.monotonic() + seconds
    parent = _deadline.get()
    if inherit and parent is not None:
        end = min(end, parent)
    token = _deadline.set(end)
    try:
        yield
    finally:
        _deadline.reset(token)


def remaining(maximum: float, stage: str = "weight operation") -> float:
    end = _deadline.get()
    budget = maximum if end is None else min(maximum, end - time.monotonic())
    if budget <= 0:
        raise TimeoutError(
            f"Operation deadline exceeded during {stage}; resume the same target or recover its checkpoint"
        )
    return budget


def operation_remaining(default: float, stage: str = "weight operation") -> float:
    """Use the caller's operation deadline, or a default for standalone
    calls."""
    validate_timeout(default)
    return remaining(default if _deadline.get() is None else float("inf"), stage)


def checkpoint(stage: str) -> None:
    remaining(float("inf"), stage)
