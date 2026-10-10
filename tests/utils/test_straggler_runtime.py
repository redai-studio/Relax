# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""CPU tests for the per-step reporting the Megatron actor drives, and for when
the profiler gets installed."""

from argparse import Namespace

import pytest

from relax.utils.straggler import reporter, runtime
from relax.utils.straggler.runtime import StragglerProfiler, install_straggler_profiler
from tests.utils.straggler_helpers import bracket, make_collector


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


def test_delivery_is_logged_and_carried_by_the_next_report(monkeypatch):
    lines = []
    monkeypatch.setattr(reporter.logger, "info", lambda msg, *a: lines.append(msg % a))
    profiler = _profiler(interval=1)
    first = _step(profiler, 0)
    assert "straggler/latency/prev_delivered_ms" not in first
    profiler.delivered(0)
    assert any("delivered +" in line for line in lines)
    assert _step(profiler, 1)["straggler/latency/prev_delivered_ms"] >= 0.0
    profiler.delivered(1)
    profiler.delivered(1)  # nothing left to deliver: no second line
    assert sum("delivered +" in line for line in lines) == 2


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
