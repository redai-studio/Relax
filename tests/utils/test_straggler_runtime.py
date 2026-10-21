# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""CPU tests for the per-step reporting the Megatron actor drives, and for when
the profiler gets installed."""

from argparse import Namespace

import pytest

from relax.utils.straggler import runtime
from relax.utils.straggler.detector import WindowReport
from relax.utils.straggler.runtime import StragglerProfiler, install_straggler_profiler
from relax.utils.training import train_metric_utils
from tests.utils.straggler_helpers import FakeEvent, bracket, make_collector, meta


def _args(num_rollout=100):
    return Namespace(num_rollout=num_rollout, wandb_always_use_train_step=False, timeline_dump_dir=None)


def _profiler(interval=2, num_rollout=100, **kwargs):
    return StragglerProfiler(make_collector(interval=interval, **kwargs), _args(num_rollout))


def _step(profiler, rollout_id):
    bracket(profiler.collector)
    return profiler.step_metrics(rollout_id, tokens=1000)


def test_scalars_arrive_when_a_window_closes_and_carry_its_rollout_range():
    profiler = _profiler(interval=2)
    assert _step(profiler, 0) is None
    metrics = _step(profiler, 1)
    assert metrics["straggler/window/first_rollout"] == 0.0 and metrics["straggler/window/last_rollout"] == 1.0
    assert metrics["straggler/tokens/median"] == pytest.approx(1000.0)


def test_a_failing_reporter_is_counted_and_never_raises(monkeypatch):
    def boom(*args, **kwargs):
        raise RuntimeError("timeline broke")

    monkeypatch.setattr(runtime, "report_straggler_window", boom)
    profiler = _profiler(interval=1)
    assert _step(profiler, 0) is None
    assert profiler.collector.health.errors_total == 1


def test_last_rollout_of_the_run_waits_for_its_own_window():
    profiler = _profiler(interval=1, num_rollout=5, background=True)
    try:
        assert _step(profiler, 4)["straggler/window/last_rollout"] == 4.0
    finally:
        profiler.collector.close()


def test_install_is_a_no_op_when_disabled_or_for_non_training_roles(monkeypatch):
    monkeypatch.setenv("RELAX_STRAGGLER_PROFILER", "0")
    assert install_straggler_profiler("actor", _args()) is None
    monkeypatch.setenv("RELAX_STRAGGLER_PROFILER", "1")
    # These roles never report a training step, so they must not get a profiler
    # (checked before any Megatron parallel state is touched).
    for role in ("critic", "reference", "actor_fwd"):
        assert install_straggler_profiler(role, _args()) is None


class _FakeBackend:
    """Stands in for ``relax.utils.device.device_module`` on a non-CUDA
    accelerator."""

    def __init__(self, capturing=None):
        self.events = []
        if capturing is not None:
            self.is_current_stream_capturing = lambda: capturing

    def Event(self, **kwargs):
        self.events.append(kwargs)
        return object()


def _use_backend(monkeypatch, backend, available=True):
    from relax.utils import device

    monkeypatch.setattr(device, "device_module", backend)
    monkeypatch.setattr(device, "is_available", lambda: available)


def test_timing_events_come_from_the_active_accelerator(monkeypatch):
    backend = _FakeBackend()
    _use_backend(monkeypatch, backend)
    runtime.timing_event_factory()()
    assert backend.events == [{"enable_timing": True}]


@pytest.mark.parametrize(
    ("capturing", "available", "expected"), [(True, True, True), (False, True, False), (True, False, False)]
)
def test_capture_check_uses_the_backend_function_when_there_is_one(monkeypatch, capturing, available, expected):
    _use_backend(monkeypatch, _FakeBackend(capturing=capturing), available=available)
    assert runtime.stream_capture_check()() is expected


def test_capture_check_is_false_when_the_backend_has_none(monkeypatch):
    _use_backend(monkeypatch, _FakeBackend())
    assert runtime.stream_capture_check()() is False


def test_timers_are_none_without_a_profiler():
    assert runtime.straggler_timers("train") is None
    with pytest.raises(ValueError):
        runtime._timers_for(_profiler().collector, "eval")


def test_only_the_latest_of_several_ready_windows_reaches_the_metrics(monkeypatch):
    def window(n):
        return WindowReport(
            metrics={"straggler/window/last_rollout": float(n)}, alerts=[], rows=[], first_rollout=n, last_rollout=n
        )

    reported, warnings = [], []
    monkeypatch.setattr(
        runtime, "report_straggler_window", lambda args, report: reported.append(report) or dict(report.metrics)
    )
    monkeypatch.setattr(runtime.logger, "warning", lambda msg, *a: warnings.append(msg % a))
    profiler = _profiler(interval=1)
    monkeypatch.setattr(profiler.collector, "end_step", lambda rollout_id, final=False: [window(0), window(1)])

    metrics = profiler.step_metrics(1, tokens=0)
    assert metrics == {"straggler/window/last_rollout": 1.0}
    assert [r.last_rollout for r in reported] == [0, 1], "every window still reaches the timeline"
    assert len(warnings) == 1 and "2 windows ready at once" in warnings[0]
    assert profiler.collector.health.errors_total == 1


def _ready_windows(monkeypatch, profiler, rollouts, failing):
    """Make ``step_metrics`` see the windows of ``rollouts`` ready at once,
    with the reporter failing on ``failing``; returns the WARNING lines."""

    def report(args, window):
        if window.last_rollout == failing:
            raise RuntimeError("timeline broke")
        return dict(window.metrics)

    def window(n):
        return WindowReport(
            metrics={"straggler/window/last_rollout": float(n)}, alerts=[], rows=[], first_rollout=n, last_rollout=n
        )

    warnings = []
    monkeypatch.setattr(runtime, "report_straggler_window", report)
    monkeypatch.setattr(runtime.logger, "warning", lambda msg, *a: warnings.append(msg % a))
    monkeypatch.setattr(profiler.collector, "end_step", lambda rollout_id, final=False: [window(n) for n in rollouts])
    return warnings


def test_a_failing_latest_window_falls_back_to_the_previous_one_and_counts_one_error(monkeypatch):
    profiler = _profiler(interval=1)
    warnings = _ready_windows(monkeypatch, profiler, rollouts=[1, 2], failing=2)
    for errors in (1, 2):
        assert profiler.step_metrics(2, tokens=0) == {"straggler/window/last_rollout": 1.0}
        assert profiler.collector.health.errors_total == errors, "only the failing window is lost"
    assert warnings == [], "the returned window superseded nothing"


def test_the_superseded_warning_names_the_window_that_is_returned(monkeypatch):
    profiler = _profiler(interval=1)
    warnings = _ready_windows(monkeypatch, profiler, rollouts=[0, 1, 2], failing=2)
    assert profiler.step_metrics(2, tokens=0) == {"straggler/window/last_rollout": 1.0}
    assert len(warnings) == 1 and "only rollouts 1-1 reach the metrics" in warnings[0]
    assert profiler.collector.health.errors_total == 2, "window 0 superseded, window 2 failed"


@pytest.mark.parametrize("primary", [True, False])
def test_install_uses_the_shared_metrics_primary_rank(monkeypatch, primary):
    monkeypatch.setenv("RELAX_STRAGGLER_PROFILER", "1")
    monkeypatch.setattr(train_metric_utils, "is_metrics_primary_rank", lambda: primary)
    monkeypatch.setattr(runtime, "local_rank_meta", lambda: meta(1)[0])
    monkeypatch.setattr(runtime, "timing_event_factory", lambda: FakeEvent)
    monkeypatch.setattr(runtime, "stream_capture_check", lambda: runtime._never)
    # The gather group is a collective; its creation is covered separately.
    monkeypatch.setattr(runtime, "init_straggler_gather_group", lambda timeout_s: None)
    monkeypatch.setattr(runtime, "_PROFILER", None)
    profiler = install_straggler_profiler("actor", _args())
    try:
        assert profiler.collector.is_primary is primary
    finally:
        profiler.collector.close()
