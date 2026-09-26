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

The drained verdicts are also logged, one line each, with the raw Megatron
timer name and its coarse stage group next to each other. The group is derived
from the stage the verdict already carries, so a reader can localise a
measurement without the wire envelope growing a field and without any metrics
backend having to accept a non-numeric value.

Two keys deserve a precise reading:

* ``coverage`` is cohort coverage and is emitted **only** when the summary
  publishes it. The fallback this module used to compute was
  ``judged_packets / envelopes`` --- a different quantity --- and is now its own
  key, ``judged_fraction``, so neither can be misread as the other.
* ``collector_status_available`` is ``1`` when this process owns the collector
  and ``0`` when it does not. The platform merges metrics only on the Megatron
  primary rank (tp0, pipeline-last, dp0), which coincides with the collector
  rank (global rank 0) only when ``pp_size == 1``; under PP>1 the exporting rank
  owns no collector, and this ``0`` is the explicit "straggler counters
  unavailable here" marker instead of silent omission.

Known limits (documented, not fixed here): ``RELAX_STRAGGLER_TOPOLOGY_EPOCH`` is
inert --- nothing derives or updates it, so a re-shard is invisible to the
cohort key unless an operator sets the variable by hand --- and there is no
rollout or topology reset, so a stall spanning a rollout boundary is drained
into the next rollout's perf log.
"""

from typing import Any, Dict, Mapping, Optional, Tuple

from relax.utils.logging_utils import get_logger
from relax.utils.straggler.stages import with_stage_group


logger = get_logger(__name__)


STRAGGLER_METRIC_PREFIX = "perf/straggler/"

#: Emitted metric names without the prefix, in a stable documented order.
#: ``active_stragglers``/``verdicts``/``windows_closed``/``coverage``/
#: ``judged_fraction``/``dropped`` come from the runtime summary;
#: ``worst_deviation``/``worst_rank`` from the drained verdicts;
#: ``rollout_id``/``optimizer_step``/``step_ordinal`` from the published
#: training context; ``workload_incomparable_windows``/
#: ``workload_missing_windows`` from the detector counters in the summary;
#: ``workload_publish_skipped``/``workload_publish_errors`` from the training
#: context counters; and ``collector_status_available`` from the runtime's role.
STRAGGLER_METRIC_KEYS: Tuple[str, ...] = (
    "active_stragglers",
    "verdicts",
    "windows_closed",
    "coverage",
    "judged_fraction",
    "dropped",
    "worst_deviation",
    "worst_rank",
    "rollout_id",
    "optimizer_step",
    "step_ordinal",
    "workload_incomparable_windows",
    "workload_missing_windows",
    "workload_publish_skipped",
    "workload_publish_errors",
    "collector_status_available",
)

#: Drop reasons summed into ``perf/straggler/dropped``. The first is an
#: aggregate a runtime may already provide; the rest are per-component
#: counters. Every bounded structure in the detector reports its evictions, so
#: they are all published here: a cap that dropped data must never be invisible
#: behind a ``dropped`` key that stays absent.
_DROPPED_FIELDS: Tuple[str, ...] = (
    "dropped",
    "dropped_queue_full",
    "dropped_pending_full",
    "dropped_output_full",
    "pending_line_drops",
    # Detector caps (``relax/utils/straggler/detector.py``).
    "sample_evictions",
    "pair_evictions",
    "rank_evictions",
    "verdict_evictions",
    "streak_evictions",
    "label_evictions",
    "epoch_evictions",
    "cohort_epoch_evictions",
    "active_evictions",
    # Malformed input: kept as measurements rather than silently skipped.
    "invalid_samples",
    "invalid_device_samples",
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
    """Return cohort coverage, only when the summary publishes it.

    The name means the fraction of the expected cohort that reported, so it is
    emitted only from an explicit ``coverage_ratio``/``coverage``. The
    ``judged_packets / envelopes`` fallback measures something else entirely
    and now has its own key; reporting it as coverage read 1.0 on a run whose
    real cohort coverage was far lower.
    """
    for key in ("coverage_ratio", "coverage"):
        number = _number(summary.get(key))
        if number is not None:
            return number
    return None


def _judged_fraction(summary: Mapping[str, Any]) -> Optional[float]:
    """Return the fraction of received envelopes the collector judged.

    This is deliberately a separate key from ``coverage``: it is an in-process
    transport ratio, not cohort coverage.
    """
    envelopes = _number(summary.get("envelopes"))
    judged = _number(summary.get("judged_packets"))
    if envelopes is not None and judged is not None and envelopes > 0.0:
        return judged / envelopes
    return None


def _context_counters() -> Mapping[str, Any]:
    """Read the training-context counters; empty when unavailable.

    The publish counters live in
    :mod:`relax.utils.straggler.context`, which used to have no production
    caller at all. This read is in-process and allocation-light, so it is safe on
    the training thread; it adds no request, no socket and no collective.
    """
    try:
        from relax.utils.straggler.context import training_context_stats
    except Exception:
        _count_failure()
        return {}
    try:
        value = training_context_stats()
    except Exception:
        _count_failure()
        return {}
    return value if isinstance(value, Mapping) else {}


def _collector_status_available(runtime: Any) -> Optional[float]:
    """Return 1 when this process owns a collector, 0 when it does not.

    The platform merges metrics only on the Megatron primary rank, which owns
    the collector only when ``pp_size == 1``. Under PP>1 the exporting rank
    owns none, and this ``0`` is the explicit unavailable marker. A stub
    runtime without a ``collector`` attribute cannot be classified, so nothing
    is emitted for it rather than inventing a ``0``.
    """
    if not hasattr(runtime, "collector"):
        return None
    return 1.0 if getattr(runtime, "collector", None) is not None else 0.0


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


def _stage_name(verdict: Any) -> Optional[str]:
    """Return the raw timer name a verdict was judged on, or ``None``.

    ``facts["stage"]`` is the canonical location; the legacy top-level ``name``
    field is the fallback, so a stub or an older verdict still labels.
    """
    facts = _mapping(_verdict_field(verdict, "facts"))
    stage = facts.get("stage")
    if isinstance(stage, str) and stage:
        return stage
    name = _verdict_field(verdict, "name")
    return name if isinstance(name, str) and name else None


def _stage_facts(verdict: Any) -> Mapping[str, Any]:
    """Return a verdict's facts with the raw ``stage`` guaranteed present."""
    facts = _mapping(_verdict_field(verdict, "facts"))
    stage = _stage_name(verdict)
    if stage is None:
        return {}
    if facts.get("stage") != stage:
        return {**facts, "stage": stage}
    return facts


def _log_stage_groups(verdicts: Any) -> None:
    """Log every verdict's raw stage next to its coarse group.

    Presentation only: the detector's decision is already made, and the group
    is derived from the same facts that carry the raw stage, so a reader can
    localise a measurement without the wire envelope growing a field. A name
    outside the taxonomy is logged as ``other`` rather than guessed.
    """
    for verdict in verdicts:
        labelled = with_stage_group(_stage_facts(verdict))
        if "stage" not in labelled:
            continue
        logger.info("straggler stage: name=%s coarse_stage=%s", labelled["stage"], labelled["stage_group"])


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
            "judged_fraction": _judged_fraction(summary),
            "dropped": _dropped(summary),
            # Detector counters that were only reachable through the status
            # files; surfaced here so the workload gate is observable in the
            # training log too.
            "workload_incomparable_windows": _number(summary.get("workload_incomparable_windows")),
            "workload_missing_windows": _number(summary.get("workload_missing_windows")),
            "collector_status_available": _collector_status_available(runtime),
        }
        worst_deviation, worst_rank = _worst_verdict(verdicts)
        values["worst_deviation"] = worst_deviation
        values["worst_rank"] = worst_rank
        training_context = _training_context()
        for key in ("rollout_id", "optimizer_step", "step_ordinal"):
            values[key] = _number(training_context.get(key))
        # The publish counters are only meaningful once training has published a
        # context; before that they are omitted rather than reported as a
        # reassuring zero.
        context_counters = _context_counters()
        updates = _number(context_counters.get("updates"))
        if updates is not None and updates > 0.0:
            values["workload_publish_skipped"] = _number(context_counters.get("workload_publish_skipped"))
            values["workload_publish_errors"] = _number(context_counters.get("workload_publish_errors"))
        metrics: Dict[str, float] = {}
        for key in STRAGGLER_METRIC_KEYS:
            value = values.get(key)
            if value is not None:
                metrics[f"{STRAGGLER_METRIC_PREFIX}{key}"] = value
    except Exception:
        _count_failure()
        return {}
    try:
        _log_stage_groups(verdicts)
    except Exception:
        # A logging failure must not cost the caller its metrics.
        _count_failure()
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
            # Enabled but no runtime: say so explicitly instead of returning an
            # empty mapping that reads as "nothing to report".
            return {f"{STRAGGLER_METRIC_PREFIX}collector_status_available": 0.0}
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
