# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Platform-facing metrics built from the straggler profiler's evidence.

Contract:

* this module never calls HTTP, never opens a socket and never blocks the
  training thread; it only reads in-process counters and drains the verdicts the
  runtime already holds;
* every failure is counted here, never raised into the caller, so a broken
  profiler can never affect normal training metrics;
* the profiler is off by default, in which case :func:`report_once` returns an
  empty mapping without touching the runtime.

The emitted keys are fixed by :data:`STRAGGLER_METRIC_KEYS`, all under
:data:`STRAGGLER_METRIC_PREFIX`, and every value is a scalar. A value that
cannot be measured is omitted rather than reported as zero, so a missing key
reads as "not measured" and never as a false "none observed".
"""

from typing import Any, Dict, Mapping, Optional, Tuple


STRAGGLER_METRIC_PREFIX = "perf/straggler/"

#: Emitted metric names without the prefix, in a stable documented order.
#: ``active_stragglers``/``verdicts``/``windows_closed``/``coverage``/
#: ``dropped`` come from the runtime summary, ``worst_deviation``/
#: ``worst_rank`` from the drained verdicts, and the remaining three from the
#: published training context when one exists.
STRAGGLER_METRIC_KEYS: Tuple[str, ...] = (
    "active_stragglers",
    "verdicts",
    "windows_closed",
    "coverage",
    "dropped",
    "worst_deviation",
    "worst_rank",
    "rollout_id",
    "optimizer_step",
    "global_step",
)

#: Drop reasons summed into ``perf/straggler/dropped``. The first is an
#: aggregate a runtime may already provide; the rest are per-component counters.
_DROPPED_FIELDS: Tuple[str, ...] = (
    "dropped",
    "dropped_queue_full",
    "dropped_pending_full",
    "dropped_output_full",
)

_FAILURES = 0


def error_count() -> int:
    """Number of failures this module has swallowed."""
    return _FAILURES


def reset_for_tests() -> None:
    """Zero the swallowed-failure counter."""
    global _FAILURES
    _FAILURES = 0


def _count_failure() -> None:
    """Count one swallowed failure; the counter is a bounded scalar."""
    global _FAILURES
    _FAILURES += 1


def _is_number(value: Any) -> bool:
    """Whether ``value`` is a real scalar (``bool`` is not a metric)."""
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _number(value: Any) -> Optional[float]:
    """Coerce a report field to a float, or ``None`` when unmeasurable."""
    if not _is_number(value):
        return None
    return float(value)


def _mapping(value: Any) -> Mapping[str, Any]:
    """Return ``value`` as a mapping, or an empty one."""
    if isinstance(value, Mapping):
        return value
    return {}


def _count(value: Any) -> Optional[float]:
    """Read a count that may be reported as a number or as a list of items."""
    number = _number(value)
    if number is not None:
        return number
    if isinstance(value, (list, tuple, set, frozenset)):
        return float(len(value))
    return None


def _coverage(summary: Mapping[str, Any]) -> Optional[float]:
    """Return cohort coverage, preferring an explicit ratio.

    When the runtime does not publish one, fall back to the fraction of
    received envelopes the collector actually judged, which is the closest in-
    process proxy; ``None`` when neither can be measured.
    """
    for key in ("coverage_ratio", "coverage"):
        number = _number(summary.get(key))
        if number is not None:
            return number
    envelopes = _number(summary.get("envelopes"))
    judged = _number(summary.get("judged_packets"))
    if envelopes is not None and judged is not None and envelopes > 0.0:
        return judged / envelopes
    return None


def _dropped(summary: Mapping[str, Any]) -> Optional[float]:
    """Sum every drop counter the runtime reports, or ``None`` when none
    does."""
    explicit = _number(summary.get("dropped"))
    if explicit is not None:
        return explicit
    total = 0.0
    seen = False
    sources = (summary, summary.get("sender"), summary.get("observer"), summary.get("collector"))
    for source in sources:
        if not isinstance(source, Mapping):
            continue
        for key in _DROPPED_FIELDS[1:]:
            number = _number(source.get(key))
            if number is not None:
                total += number
                seen = True
    return total if seen else None


def _verdict_field(verdict: Any, name: str) -> Any:
    """Read a verdict field from an object or a mapping."""
    if isinstance(verdict, Mapping):
        return verdict.get(name)
    return getattr(verdict, name, None)


def _worst_verdict(verdicts: Any) -> Tuple[Optional[float], Optional[float]]:
    """Return ``(deviation, rank)`` of the largest measured slowdown."""
    worst_deviation: Optional[float] = None
    worst_rank: Optional[float] = None
    for verdict in verdicts:
        deviation = _number(_verdict_field(verdict, "deviation"))
        if deviation is None:
            continue
        if worst_deviation is None or deviation > worst_deviation:
            worst_deviation = deviation
            worst_rank = _number(_verdict_field(verdict, "rank"))
    return worst_deviation, worst_rank


def _training_context() -> Mapping[str, Any]:
    """Read the published training context; empty when unavailable."""
    try:
        from relax.utils.straggler.context import snapshot
    except Exception:
        _count_failure()
        return {}
    try:
        value = snapshot()
    except Exception:
        _count_failure()
        return {}
    return value if isinstance(value, Mapping) else {}


def _runtime_summary(runtime: Any) -> Any:
    """Read the runtime summary without flushing persistence.

    ``report_once`` runs on the training thread, and the real runtime's
    ``report()`` flushes buffered JSONL lines. The non-flushing ``summary()``
    is preferred whenever the runtime exposes one (the observer/collector path
    does); a stub that only implements ``report()`` keeps working because the
    reporter is a read-only adapter.
    """
    summary = getattr(runtime, "summary", None)
    if callable(summary):
        return summary()
    return runtime.report()


def build_metrics(runtime: Any) -> Dict[str, float]:
    """Build the flat metric mapping for one training rollout.

    ``runtime`` is anything exposing ``report()`` and ``drain_verdicts()`` (the
    real :class:`~relax.utils.straggler.runtime.StragglerRuntime` or a stub). A
    runtime whose methods raise yields ``{}`` and bumps the error counter; this
    function itself never raises. Calling it drains the runtime's pending
    verdicts, which is exactly what the per-rollout report wants.
    """
    try:
        summary = _mapping(_runtime_summary(runtime))
        drained = runtime.drain_verdicts()
    except Exception:
        _count_failure()
        return {}
    verdicts = drained if isinstance(drained, (list, tuple)) else []

    try:
        values: Dict[str, Optional[float]] = {
            "active_stragglers": _count(summary.get("active_stragglers")),
            "verdicts": _number(summary.get("verdicts")),
            "windows_closed": _number(summary.get("windows_closed")),
            "coverage": _coverage(summary),
            "dropped": _dropped(summary),
        }
        worst_deviation, worst_rank = _worst_verdict(verdicts)
        values["worst_deviation"] = worst_deviation
        values["worst_rank"] = worst_rank
        training_context = _training_context()
        for key in ("rollout_id", "optimizer_step", "global_step"):
            values[key] = _number(training_context.get(key))
        metrics: Dict[str, float] = {}
        for key in STRAGGLER_METRIC_KEYS:
            value = values.get(key)
            if value is not None:
                metrics[f"{STRAGGLER_METRIC_PREFIX}{key}"] = value
    except Exception:
        _count_failure()
        return {}
    return metrics


def report_once(args: Any, rollout_id: int) -> Dict[str, float]:
    """Return the straggler metrics to merge into the training log.

    Performs no I/O and never raises. ``args`` and ``rollout_id`` are accepted
    so the call site stays uniform with the rest of the training metrics and so
    future knobs can be read from ``args``; the rollout id is already carried
    by the training context.
    """
    try:
        from relax.utils.straggler import get_straggler_runtime, is_straggler_profiler_enabled
    except Exception:
        _count_failure()
        return {}
    try:
        if not is_straggler_profiler_enabled():
            return {}
        runtime = get_straggler_runtime()
        if runtime is None:
            return {}
        return build_metrics(runtime)
    except Exception:
        _count_failure()
        return {}


__all__ = [
    "STRAGGLER_METRIC_KEYS",
    "STRAGGLER_METRIC_PREFIX",
    "build_metrics",
    "error_count",
    "report_once",
    "reset_for_tests",
]
