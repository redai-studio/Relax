# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""The profiler inside Relax: its settings (``RELAX_STRAGGLER_*`` in ``Envs``),
the process-wide profiler the Megatron actor holds, and what it needs from
Relax and torch: device events, the Gloo gathers (``collectives``), the metrics
service, the CPU timeline and the rollout step numbers.

The training code calls only the functions in ``__init__``; everything that
touches the accelerator (through ``relax.utils.device``, so CUDA and NPU alike)
or the process groups is here, so the collection and the analysis run (and are
tested) on CPU with fakes.
"""

from __future__ import annotations

import os
import socket
from dataclasses import replace
from functools import partial
from typing import Any, Callable, Sequence

import torch
import torch.distributed as dist

from relax.utils import device
from relax.utils.env import Envs
from relax.utils.logging_utils import get_logger
from relax.utils.metrics.metric_utils import compute_rollout_step
from relax.utils.straggler import megatron_adapter
from relax.utils.straggler.analysis.control import LevelPoller
from relax.utils.straggler.analysis.detector import DetectorConfig, WindowReport
from relax.utils.straggler.analysis.report import (
    WindowAnalyzer,
    log_window,
    reported_scalars,
    switched_off,
    timeline_bars,
)
from relax.utils.straggler.analysis.trace import write_window_trace
from relax.utils.straggler.collectives import (
    gloo_all_gather,
    gloo_all_gather_objects,
    gloo_gather_objects_to,
    gloo_primary_rank,
    init_straggler_gather_group,
)
from relax.utils.straggler.core.collector import StragglerCollector
from relax.utils.straggler.core.stats import LEVELS, RankMeta
from relax.utils.straggler.core.timers import StragglerTimers
from relax.utils.timer import TimelineEvent, Timer


logger = get_logger(__name__)

# Only the role that runs training steps closes them; critic / reference /
# actor_fwd actors would only fill the pending queue and never drain it.
PROFILED_ROLES: frozenset[str] = frozenset({"actor"})


# ---------------------------------------------------------------------------
# Collection: device events, rank identity, CPU timeline (the gathers are in ``collectives``)
# ---------------------------------------------------------------------------


def timing_event_factory() -> Callable[[], Any]:
    """Build timing events of the active accelerator.

    On CUDA they are ``torch.Event``, whose ``record`` finds the current stream
    in C++; ``torch.cuda.Event.record`` looks it up in Python, several
    microseconds more per record.
    """
    if device.get_device_name() == "cuda" and hasattr(torch, "Event"):
        return partial(torch.Event, device="cuda", enable_timing=True)
    return partial(device.device_module.Event, enable_timing=True)


def stream_capture_check() -> Callable[[], bool]:
    """The active accelerator's "is the current stream being captured" check,
    resolved once because the timers call it on every bracket.

    A backend without one cannot be capturing through this API.
    """
    if not device.is_available():
        return _never
    return getattr(device.device_module, "is_current_stream_capturing", None) or _never


def _never() -> bool:
    return False


def device_synchronize() -> Callable[[], None]:
    """The active accelerator's synchronize, which anchors a sampled step."""
    return device.device_module.synchronize if device.is_available() else (lambda: None)


def local_rank_meta() -> RankMeta:
    return RankMeta(
        # Indexes the gathered table: the straggler group spans every global rank.
        rank=dist.get_rank(),
        **megatron_adapter.parallel_ranks(),
        host=socket.gethostname(),
        device=device.device_module.current_device() if device.is_available() else -1,
    )


def _cpu_events(start: float, end: float) -> list[dict[str, Any]]:
    """This process's finished ``Timer`` events that overlap the wall-clock
    span ``[start, end]``, without their call stacks; timers still running when
    the step ended are not among them."""
    events = []
    for record in reversed(Timer().records):
        if record.end_ts < start:
            break
        if record.start_ts <= end and not record.name.startswith("straggler/"):
            event = record.to_trace_event()
            event.get("args", {}).pop("call_stack", None)
            events.append(event)
    return events


# ---------------------------------------------------------------------------
# Analysis and reporting: detector thresholds, log lines, timeline bars, trace files
# ---------------------------------------------------------------------------


def window_analyzer(args: Any, traces_to: str) -> WindowAnalyzer:
    """What becomes of the primary rank's closed windows: the detector with the
    ``Envs`` thresholds, the sampled-step trace files under ``traces_to`` (none
    when it is empty) and the window's log lines."""
    return WindowAnalyzer(
        DetectorConfig(
            z_threshold=Envs.RELAX_STRAGGLER_Z_THRESHOLD,
            rel_threshold=Envs.RELAX_STRAGGLER_REL_THRESHOLD,
            persist_windows=Envs.RELAX_STRAGGLER_PERSIST_WINDOWS,
            recover_windows=Envs.RELAX_STRAGGLER_RECOVER_WINDOWS,
        ),
        trace_sink=partial(
            write_window_trace,
            traces_to,
            Envs.RELAX_STRAGGLER_TRACE_KEEP,
            sync_names=megatron_adapter.GRAD_SYNC_TIMERS,
            module_names=megatron_adapter.MODULE_NAMES,
        )
        if traces_to
        else None,
        on_report=partial(log_straggler_window, args),
    )


def report_straggler_window(args: Any, report: WindowReport) -> dict[str, float]:
    """Put one window's bars on the timeline and return its scalars for the
    step's ``log_perf_data`` (primary rank, training thread), which also
    flushes the bars and stamps them with the step."""
    if getattr(args, "timeline_dump_dir", None) and not switched_off(report):
        Timer().records.extend(
            TimelineEvent(bar.name, bar.start, bar.end, pid=bar.pid, tid=bar.tid) for bar in timeline_bars(report)
        )
    return reported_scalars(report)


def log_straggler_window(args: Any, report: WindowReport) -> None:
    """Log one window at the step of its last rollout; the analysis thread runs
    it as soon as the analysis ends."""
    log_window(report, compute_rollout_step(args, report.last_rollout))


def trace_dir(args: Any) -> str:
    """Where the primary rank writes sampled-step traces (empty: no tracing).

    ``RELAX_STRAGGLER_TRACE_DIR``, else ``straggler/`` under the timeline dump
    directory.
    """
    timeline_dir = getattr(args, "timeline_dump_dir", None)
    return Envs.RELAX_STRAGGLER_TRACE_DIR or (os.path.join(timeline_dir, "straggler") if timeline_dir else "")


# ---------------------------------------------------------------------------
# The process-wide profiler and the training code's entry points
# ---------------------------------------------------------------------------


class StragglerProfiler:
    """One collector, the ``config.timers`` of each phase that feed it, and the
    per-step reporting around ``log_perf_data``.

    ``rollout_end`` never raises: a failing reporter is counted by the
    collector's health like any other profiler failure.
    """

    def __init__(self, collector: StragglerCollector, args: Any) -> None:
        self.collector = collector
        self.timers = megatron_adapter.phase_timers(collector.recorder)
        self._args = args
        self._model_seen = False
        self._unchecked_phases = set(megatron_adapter.REQUIRED_TIMER_NAMES)

    def see_model(self, model_chunks: Sequence[Any]) -> None:
        """Take the model chunks the first time a forward-backward call names
        them: their count, and their blocks for the ``modules`` level, which
        are timed only while a sampled step at that level runs."""
        if self._model_seen:
            return
        self._model_seen = True
        recorder = self.collector.recorder
        recorder.model_chunks = max(len(model_chunks), 1)
        try:
            timers = megatron_adapter.module_timers(model_chunks, recorder)
        except Exception as exc:
            recorder.on_error("modules", exc)
            return
        recorder.module_timers = timers
        if self.collector.is_primary:
            counts = timers.counts()
            logger.info(
                "[straggler] modules level: %d blocks on this rank (%s)",
                counts.total(),
                ", ".join(f"{name} {count}" for name, count in counts.items()),
            )

    def rollout_end(self, rollout_id: int) -> dict[str, float] | None:
        """Close this step (all ranks; a Gloo collective every window) and
        return the scalars of the latest window whose analysis has finished,
        for this step's ``log_perf_data``.

        If several windows are ready at once, each still reaches the timeline
        and logs, but only the scalars of the latest one whose reporter
        succeeded are returned (one metrics call per step carries one value per
        key). Each window it supersedes is counted as a profiler error, except
        at the last step, whose flush makes the last window ready together with
        the one before it (always so with one-rollout windows); a window whose
        reporter fails is counted once, as that failure.
        """
        try:
            self._check_timer_points()
        except Exception as exc:
            self.collector.on_error("self-check", exc)
        final = rollout_id + 1 == self._args.num_rollout
        reports = self.collector.end_step(rollout_id, final=final)
        reported: list[tuple[WindowReport, dict[str, float]]] = []
        for report in reports:
            try:
                reported.append((report, report_straggler_window(self._args, report)))
            except Exception as exc:
                self.collector.on_error("report", exc)
        if not reported:
            return None
        *superseded, (latest, metrics) = reported
        if final:
            return metrics
        for dropped, _ in superseded:
            self.collector.on_error(
                "report",
                RuntimeError(
                    f"scalars of rollouts {dropped.first_rollout}-{dropped.last_rollout} superseded by those of "
                    f"rollouts {latest.first_rollout}-{latest.last_rollout}, ready at the same step"
                ),
            )
        return metrics

    def _check_timer_points(self) -> None:
        """At the end of the first rollout that ran a phase, report each timer
        that phase always calls but Megatron did not: its time would be missing
        from every window."""
        for phase in sorted(self._unchecked_phases):
            seen = self.timers[phase].names_seen()
            if not seen:
                continue
            self._unchecked_phases.discard(phase)
            missing = sorted(megatron_adapter.REQUIRED_TIMER_NAMES[phase] - seen)
            if missing:
                self.collector.on_degraded(
                    f"the first rollout with {phase} steps never called the Megatron timers {', '.join(missing)}, "
                    "so their time is not measured"
                )


_PROFILER: StragglerProfiler | None = None


def install_straggler_profiler(role: str, args: Any) -> StragglerProfiler | None:
    """Create the process-wide profiler if it is enabled and ``role`` runs
    training steps; ``None`` otherwise.

    Must run after Megatron's parallel state exists and before the optimizer is
    built (its config takes ``straggler_timers``).
    """
    global _PROFILER
    if not Envs.RELAX_STRAGGLER_PROFILER or role not in PROFILED_ROLES:
        return None
    if _PROFILER is not None:
        return _PROFILER

    init_straggler_gather_group(Envs.RELAX_STRAGGLER_GATHER_TIMEOUT_S)

    # The window scalars are logged through ``log_perf_data``, which drops them
    # on every other rank.
    is_primary = megatron_adapter.is_metrics_primary_rank()
    traces_to = trace_dir(args)
    # Only a tracing run gathers samples; every rank takes this branch alike, so the all-reduce is collective.
    primary_rank = gloo_primary_rank(is_primary) if traces_to else None
    poller: LevelPoller | None = None
    collector = StragglerCollector(
        rank_meta=replace(local_rank_meta(), primary=is_primary),
        is_primary=is_primary,
        report_interval=Envs.RELAX_STRAGGLER_REPORT_INTERVAL,
        event_factory=timing_event_factory(),
        gather=gloo_all_gather,
        gather_objects=gloo_all_gather_objects,
        is_capturing=stream_capture_check(),
        analyze=window_analyzer(args, traces_to),
        trace=bool(traces_to),
        synchronize=device_synchronize(),
        gather_to=gloo_gather_objects_to,
        primary_rank=primary_rank,
        cpu_events=_cpu_events,
        level=initial_level(bool(traces_to), dist.get_world_size()),
        # Read at window closes; the poller is created below, once there is a collector to report.
        level_request=lambda: poller.requested if poller is not None else None,
        interleaved=(getattr(args, "virtual_pipeline_model_parallel_size", None) or 1) > 1,
    )
    if is_primary and getattr(args, "use_metrics_service", False):
        from relax.utils.utils import get_serve_url

        poller = LevelPoller(
            get_serve_url(route_prefix="/metrics"), collector.status, Envs.RELAX_STRAGGLER_CONTROL_POLL_S
        )
    _PROFILER = StragglerProfiler(collector, args)
    logger.info(
        "Straggler profiler enabled: report_interval=%d primary=%s level=%s control=%s trace_dir=%s meta=%s",
        collector.report_interval,
        is_primary,
        LEVELS[collector.level],
        "metrics service" if poller is not None else "-",
        traces_to or "-",
        collector.rank_meta,
    )
    return _PROFILER


# The primary builds one trace of every rank per window: at 2048 ranks about 10 MB and a quarter of a second
# of CPU (level 3: 80 MB, 2 s), so a larger job starts without it unless asked.
_TRACE_DEFAULT_MAX_RANKS = 256


def initial_level(trace_on: bool, world_size: int) -> int:
    """The level a run starts at: ``RELAX_STRAGGLER_LEVEL``, else detail when
    traces have somewhere to go and the job has at most 256 ranks, coarse
    otherwise; never ``modules`` unless asked for."""
    level = Envs.RELAX_STRAGGLER_LEVEL
    if level >= 0:
        return level
    return 2 if trace_on and world_size <= _TRACE_DEFAULT_MAX_RANKS else 1


def straggler_timers(
    phase: str, model: Sequence[Any] | None = None, num_microbatches: int | Sequence[int] | None = None
) -> StragglerTimers | None:
    """The ``config.timers`` value for ``phase``: ``None`` without a profiler,
    which is what Relax assigns anyway.

    Set right before Megatron's forward-backward call, ``model`` and
    ``num_microbatches`` describe that call (a list: the calls of one
    ``forward_only``, in order): the first call hands the model to the
    profiler, and under interleaved PP each call names the micro-batch and
    model chunk of its brackets.
    """
    if _PROFILER is None:
        return None
    timers = _PROFILER.timers.get(phase)
    if timers is None:
        raise ValueError(f"unknown straggler phase {phase!r}; expected one of {sorted(_PROFILER.timers)}")
    if model is not None:
        _PROFILER.see_model(model)
        if num_microbatches is not None:
            forward_only = phase == "forward_only"
            _PROFILER.collector.recorder.note_schedule(
                partial(megatron_adapter.interleaved_schedule_table, model, num_microbatches, forward_only),
                forward_only,
            )
    return timers


def straggler_rollout_end(rollout_id: int) -> dict[str, float] | None:
    """Close this rollout's step and return the scalars of a finished window
    for its ``log_perf_data``; ``None`` without a profiler or without a
    finished window.

    Every rank of the actor group must call it once per rollout: it joins a
    Gloo gather every ``RELAX_STRAGGLER_REPORT_INTERVAL`` rollouts.
    """
    return _PROFILER.rollout_end(rollout_id) if _PROFILER is not None else None


def note_microbatch_tokens(total_lengths: Any) -> None:
    """Report the sample lengths of the training micro-batch whose forward
    bracket is open; a no-op without a profiler.

    Only a host-side list is summed: summing a device tensor would synchronize.
    """
    if _PROFILER is not None and isinstance(total_lengths, (list, tuple)):
        _PROFILER.collector.recorder.note_tokens(int(sum(total_lengths)))
