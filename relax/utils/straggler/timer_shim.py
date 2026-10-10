# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Megatron timer object that records CUDA events and never synchronizes."""

from __future__ import annotations

import time
from typing import Any

import torch

from relax.utils.logging_utils import get_logger
from relax.utils.straggler.stages import STAGES, stage_of


def _new_event():
    try:
        from relax.utils import device as device_utils

        factory = getattr(device_utils, "Event", None)
        if factory is not None:
            return factory(enable_timing=True)
    except Exception:
        pass
    return torch.cuda.Event(enable_timing=True)


logger = get_logger(__name__)


class _Span:
    def __init__(self, owner: NonBlockingTimers, name: str):
        self._owner = owner
        self._name = name

    def start(self, barrier: bool = False) -> None:
        del barrier
        self._owner.start(self._name)

    def stop(self, barrier: bool = False) -> None:
        del barrier
        self._owner.stop(self._name)

    def reset(self) -> None:
        return None

    def elapsed(self, reset: bool = True, barrier: bool = False) -> float:
        del reset, barrier
        return 0.0

    def __enter__(self) -> _Span:
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop()


class NonBlockingTimers:
    """Drop-in for ``config.timers`` without ``cuda.synchronize``."""

    def __init__(self, module_stages: bool = False, max_pending: int = 16384):
        self.module_stages = module_stages
        self.max_pending = max_pending
        self._open: dict[str, tuple[Any, float]] = {}
        self._pending: list[tuple[str, Any, Any]] = []
        self._sums = {name: 0.0 for name in STAGES}
        self.dropped_events = 0
        self._cuda = torch.cuda.is_available()

    def __call__(self, name: str, log_level: int | None = None) -> _Span:
        del log_level
        return _Span(self, name)

    def start(self, name: str) -> None:
        if self._skip():
            return
        stage = stage_of(name, module_stages=self.module_stages)
        if stage is None:
            return
        if not self._cuda:
            self._open[name] = (None, time.perf_counter())
            return
        try:
            event = _new_event()
            event.record()
        except Exception as exc:
            logger.warning("straggler timer start skipped: %s", exc)
            return
        self._open[name] = (event, time.perf_counter())

    def stop(self, name: str) -> None:
        opened = self._open.pop(name, None)
        if opened is None or self._skip():
            return
        start_event, start_ts = opened
        if start_event is None:
            elapsed_ms = (time.perf_counter() - start_ts) * 1000.0
            self._add(name, elapsed_ms)
            return
        try:
            end = _new_event()
            end.record()
        except Exception as exc:
            logger.warning("straggler timer stop skipped: %s", exc)
            return
        self._pending.append((name, start_event, end))
        if len(self._pending) > self.max_pending:
            self._pending.pop(0)
            self.dropped_events += 1

    def drain(self) -> dict[str, float]:
        """Read events that have already completed.

        Does not synchronize.
        """
        still: list[tuple[str, Any, Any]] = []
        for name, start, end in self._pending:
            try:
                ready = bool(end.query())
            except Exception:
                self.dropped_events += 1
                continue
            if not ready:
                still.append((name, start, end))
                continue
            try:
                self._add(name, float(start.elapsed_time(end)))
            except Exception:
                self.dropped_events += 1
        self._pending = still
        totals = dict(self._sums)
        self._sums = {name: 0.0 for name in STAGES}
        return totals

    def log(self, names, rank=None, normalizer=1.0, reset=True, barrier=False) -> None:
        del names, rank, normalizer, reset, barrier
        return None

    def write(self, names, writer, iteration, normalizer=1.0, reset=False, barrier=False) -> None:
        del names, writer, iteration, normalizer, reset, barrier
        return None

    def _add(self, name: str, elapsed_ms: float) -> None:
        stage = stage_of(name, module_stages=self.module_stages)
        if stage is None:
            return
        self._sums[stage] = self._sums.get(stage, 0.0) + max(0.0, elapsed_ms)

    @staticmethod
    def _skip() -> bool:
        try:
            return bool(torch.cuda.is_available() and torch.cuda.is_current_stream_capturing())
        except Exception:
            return False


_TIMERS: NonBlockingTimers | None = None


def get_nonblocking_timers(module_stages: bool = False, max_pending: int = 16384) -> NonBlockingTimers:
    global _TIMERS
    if _TIMERS is None:
        _TIMERS = NonBlockingTimers(module_stages=module_stages, max_pending=max_pending)
    return _TIMERS


def reset_timers_for_tests() -> None:
    global _TIMERS
    _TIMERS = None
