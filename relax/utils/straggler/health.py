# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""The profiler's own health: ``active`` -> ``degraded`` -> ``disabled``.

Every failure inside the profiler (a timer bracket that cannot record, an event
readout that raises, a gather that fails, an analysis that crashes) is caught
and counted here instead of reaching the training loop. A rank that keeps
failing asks for the profiler to be switched off; the request travels as the
``health`` field of the rank's window vector, so every rank sees the same
gathered table and turns off at the same window. Deciding locally would leave
the other ranks blocked in the next window's gather.

``disabled`` is terminal: a profiler that flaps on and off leaves holes in the
evidence that are worse than the profiler simply being off.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass

from relax.utils.logging_utils import get_logger


logger = get_logger(__name__)

# Order is part of the metric contract (``straggler/health/state``).
HEALTH_ACTIVE, HEALTH_DEGRADED, HEALTH_DISABLE = 0, 1, 2
HEALTH_STATES: tuple[str, ...] = ("active", "degraded", "disabled")


@dataclass(frozen=True)
class HealthConfig:
    """When a rank asks for the profiler to be switched off."""

    # Caught exceptions of any kind, counted over the whole run.
    max_errors: int = 3
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
        self._degraded_streak = 0
        self._logged: set[str] = set()
        self._lock = threading.Lock()

    @property
    def disabled(self) -> bool:
        return self.state == "disabled"

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
                "(it switches itself off after %d errors)",
                where,
                type(exc).__name__,
                exc,
                self.config.max_errors,
            )

    def close_window(self, dropped: float) -> tuple[float, float]:
        """Return ``(errors, health_code)`` for this rank's closing window and
        start counting the next one."""
        with self._lock:
            errors = self._errors_window
            self._errors_window = 0
            errors_total = self.errors_total
        bad = errors > 0 or dropped > 0
        self._degraded_streak = self._degraded_streak + 1 if bad else 0
        if errors_total >= self.config.max_errors or self._degraded_streak >= self.config.max_degraded_windows:
            code = HEALTH_DISABLE
        else:
            code = HEALTH_DEGRADED if bad else HEALTH_ACTIVE
        if not self.disabled:
            self.state = "degraded" if bad else "active"
        return float(errors), float(code)

    def request_reason(self) -> str:
        """Why this rank's last ``close_window`` asked for a switch-off."""
        if self.errors_total >= self.config.max_errors:
            return f"{self.errors_total} errors caught inside the profiler (limit {self.config.max_errors})"
        return (
            f"{self._degraded_streak} consecutive windows with dropped event pairs or caught errors "
            f"(limit {self.config.max_degraded_windows})"
        )

    def disable(self, reason: str) -> None:
        self.state = "disabled"
        self.disabled_reason = reason
