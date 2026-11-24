# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""The profiler's own health: ``active`` -> ``degraded`` -> ``disabled``.

Every failure inside the profiler (a timer bracket that cannot record, an event
readout that raises, a gather that fails, an analysis that crashes) is caught
and counted here instead of reaching the training loop. A rank that keeps
failing asks for the profiler to be switched off; the request travels as the
``health`` field of the rank's window vector, so every rank sees the same
gathered table and turns off at the same window. Deciding locally would leave
the other ranks blocked in the next window's gather.

A lasting gap in what a rank measures (the framework calls the timers in a way
the profiler does not know, or never calls one it relies on) keeps the rank
``degraded`` from then on without counting towards a switch-off: the rest of
what it measures is still right.

``disabled`` is terminal: a profiler that flaps on and off leaves holes in the
evidence that are worse than the profiler simply being off.
"""

from __future__ import annotations

import logging
import threading
from collections import deque
from dataclasses import dataclass
from typing import Any, Callable


logger = logging.getLogger(__name__)

# Order is part of the metric contract (``straggler/health/state``).
HEALTH_ACTIVE, HEALTH_DEGRADED, HEALTH_DISABLE = 0, 1, 2


@dataclass(frozen=True)
class HealthConfig:
    """When a rank asks for the profiler to be switched off."""

    # Caught exceptions of any kind within the last ``error_windows`` windows: a rank that keeps failing is
    # switched off within a few windows, while failures spread over a long run (a rank that now and then
    # reaches a gather later than its timeout) are not. A switched-off rank still joins every gather, so
    # switching off would not avoid what such a failure costs, only lose the data after it.
    max_errors: int = 3
    error_windows: int = 10
    # Consecutive windows with dropped event pairs or caught exceptions.
    max_degraded_windows: int = 3


class ProfilerHealth:
    """Count this rank's profiler failures and derive its per-window health
    code.

    ``record_error`` may be called from the analysis worker thread as well as
    the training thread, hence the lock around the counters.
    """

    def __init__(self, config: HealthConfig | None = None) -> None:
        self.config = config or HealthConfig()
        self.state = "active"
        self.errors_total = 0
        self.disabled_reason = ""
        self._errors_window = 0
        self._recent_errors: deque[int] = deque(maxlen=self.config.error_windows)
        self._degraded_streak = 0
        self._logged: set[str] = set()
        self._gaps: set[str] = set()
        self._lock = threading.Lock()

    @property
    def disabled(self) -> bool:
        return self.state == "disabled"

    @property
    def gaps(self) -> frozenset[str]:
        return frozenset(self._gaps)

    def mark_degraded(self, reason: str) -> None:
        """Record a lasting gap in what this rank measures; the first report of
        each ``reason`` is logged."""
        with self._lock:
            first = reason not in self._gaps
            self._gaps.add(reason)
        if first:
            logger.warning(
                "[straggler] %s; training continues, the rest is still measured and this rank reports health degraded",
                reason,
            )

    def record_error(self, where: str, exc: BaseException) -> None:
        """Count one caught exception; the first one per ``where`` is logged
        with its message, later ones only counted."""
        with self._lock:
            self.errors_total += 1
            self._errors_window += 1
            first = where not in self._logged
            self._logged.add(where)
        if first:
            logger.warning(
                "[straggler] %s failed (%s: %s); training continues and the profiler counts the failure "
                "(it switches itself off after %d errors within %d windows)",
                where,
                type(exc).__name__,
                exc,
                self.config.max_errors,
                self.config.error_windows,
            )

    def guarded(self, where: str, call: Callable[..., Any], *args: Any, default: Any = None) -> Any:
        """``call(*args)``, or ``default`` once its exception is counted under
        ``where``."""
        try:
            return call(*args)
        except Exception as exc:
            self.record_error(where, exc)
            return default

    def close_window(self, dropped: float) -> tuple[float, float]:
        """Return ``(errors, health_code)`` for this rank's closing window and
        start counting the next one."""
        with self._lock:
            errors = self._errors_window
            self._errors_window = 0
        self._recent_errors.append(errors)
        recent = sum(self._recent_errors)
        bad = errors > 0 or dropped > 0
        self._degraded_streak = self._degraded_streak + 1 if bad else 0
        degraded = bad or bool(self._gaps)
        if recent >= self.config.max_errors or self._degraded_streak >= self.config.max_degraded_windows:
            code = HEALTH_DISABLE
        else:
            code = HEALTH_DEGRADED if degraded else HEALTH_ACTIVE
        if not self.disabled:
            self.state = "degraded" if degraded else "active"
        return float(errors), float(code)

    def request_reason(self) -> str:
        """Why this rank's last ``close_window`` asked for a switch-off."""
        recent = sum(self._recent_errors)
        if recent >= self.config.max_errors:
            return (
                f"{recent} errors caught inside the profiler within {self.config.error_windows} windows "
                f"(limit {self.config.max_errors})"
            )
        return (
            f"{self._degraded_streak} consecutive windows with dropped event pairs or caught errors "
            f"(limit {self.config.max_degraded_windows})"
        )

    def disable(self, reason: str) -> None:
        self.state = "disabled"
        self.disabled_reason = reason
