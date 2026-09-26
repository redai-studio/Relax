# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Runtime configuration for the Task 11 straggler profiler.

Every knob is declared once in :class:`relax.utils.env.Envs` and resolved
lazily here, so a recipe turns the profiler on with environment variables alone
and the default deployment keeps it off.

The profiler is a *pure observer*: nothing in this package may abort a training
step, raise into Megatron's schedule, or add a collective. Values that would
break that contract are therefore clamped and recorded in
:attr:`StragglerConfig.clamped` instead of rejected.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from relax.utils.env import Envs


@dataclass(frozen=True)
class StragglerConfig:
    """Resolved profiler settings.

    Attributes:
        enabled: master switch; ``False`` keeps ``config.timers`` at ``None`` and
            leaves Megatron's upstream (disabled) behaviour untouched.
        timer_log_level: highest Megatron timer log level to capture. Megatron
            uses 1 for coarse phases and 2 for fine ones such as
            ``forward-compute``; the default 2 captures every stage so a
            straggler can be attributed to a specific phase.
        event_pool: CUDA events preallocated per rank. One interval needs two,
            so exhaustion degrades that interval to host timing only.
        queue_max: bound of the per-rank envelope queue between the observer
            thread and the sender.
        output_dir: directory for per-rank JSONL streams; ``None`` keeps the data
            in memory only (still counted and summarised).
        warmup_windows: windows at the start of a run that are never judged.
            The first intervals carry lazy CUDA context/event allocation, the
            first data batch and allocator growth, which look like a slow rank;
            measured startup outliers reached +23% on an otherwise identical
            rank, so the opening windows are excluded and the exclusion is
            counted rather than silently applied.
        work_tolerance: relative work-difference tolerance the detector uses
            before calling two ranks' windows different amounts of work.
        min_stage_ms: absolute magnitude floor, in milliseconds. A stage whose
            observed or peer-fastest magnitude is below this, or whose absolute
            gap is not above it, is classified ``uncertain`` rather than
            ``straggler``: below the floor the relative deviation is dominated by
            host/launch jitter, so the measurement cannot distinguish a slow rank
            from noise. Measured on the DP4 SFT recipe, metadata stages run
            0.06-4.0 ms while the real compute stages run ~81 ms
            (backward-compute) and ~134 ms (forward-compute).
        window_seconds: duration of one aggregation window.
        persist_windows: consecutive anomalous windows required before a rank is
            reported as a straggler.
        min_cohort_size: cohorts smaller than this are reported as ``uncertain``
            rather than judged.
        report_interval_seconds: cadence of the collector's summary log line.
        collector_addr: ``host:port`` of the same-host collector socket; ``None``
            keeps observation rank-local.
    """

    enabled: bool = False
    timer_log_level: int = 2
    event_pool: int = 512
    queue_max: int = 4096
    output_dir: Optional[str] = None
    warmup_windows: int = 2
    work_tolerance: float = 0.05
    min_stage_ms: float = 5.0
    window_seconds: float = 5.0
    persist_windows: int = 3
    min_cohort_size: int = 2
    report_interval_seconds: float = 10.0
    collector_addr: Optional[str] = None
    clamped: Dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_env(cls) -> "StragglerConfig":
        """Resolve the configuration from the ``RELAX_STRAGGLER_*``
        variables."""
        clamped: Dict[str, str] = {}

        def integer(name: str, env: Any, minimum: int, fallback: int, maximum: Optional[int] = None) -> int:
            raw = env
            try:
                value = int(raw)
            except (TypeError, ValueError):
                value = fallback
            if value < minimum:
                value = fallback
            if maximum is not None and value > maximum:
                value = maximum
            if value != raw:
                clamped[name] = f"{raw!r} -> {value!r}"
            return value

        def real(name: str, env: Any, minimum: float, fallback: float) -> float:
            raw = env
            try:
                value = float(raw)
            except (TypeError, ValueError):
                value = fallback
            if value < minimum:
                value = fallback
            if value != raw:
                clamped[name] = f"{raw!r} -> {value!r}"
            return value

        config = cls(
            enabled=bool(Envs.RELAX_STRAGGLER_ENABLE),
            timer_log_level=integer("timer_log_level", Envs.RELAX_STRAGGLER_TIMER_LOG_LEVEL, 1, 2, maximum=2),
            event_pool=integer("event_pool", Envs.RELAX_STRAGGLER_EVENT_POOL, 2, 512),
            queue_max=integer("queue_max", Envs.RELAX_STRAGGLER_QUEUE_MAX, 1, 4096),
            output_dir=Envs.RELAX_STRAGGLER_OUTPUT_DIR,
            warmup_windows=integer("warmup_windows", Envs.RELAX_STRAGGLER_WARMUP_WINDOWS, 0, 10),
            work_tolerance=real("work_tolerance", Envs.RELAX_STRAGGLER_WORK_TOLERANCE, 0.0, 0.05),
            min_stage_ms=real("min_stage_ms", Envs.RELAX_STRAGGLER_MIN_STAGE_MS, 0.0, 5.0),
            window_seconds=real("window_seconds", Envs.RELAX_STRAGGLER_WINDOW_S, 0.1, 5.0),
            persist_windows=integer("persist_windows", Envs.RELAX_STRAGGLER_PERSIST_WINDOWS, 1, 3),
            min_cohort_size=integer("min_cohort_size", Envs.RELAX_STRAGGLER_MIN_COHORT, 2, 2),
            report_interval_seconds=real("report_interval_seconds", Envs.RELAX_STRAGGLER_REPORT_INTERVAL_S, 0.1, 10.0),
            collector_addr=Envs.RELAX_STRAGGLER_COLLECTOR_ADDR,
            clamped=clamped,
        )
        return config


__all__ = ["StragglerConfig"]
