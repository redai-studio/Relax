# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Unit tests for the non-blocking Megatron ``config.timers`` replacement.

The tests pin the two properties Task 11 depends on: the shim satisfies every
call shape Megatron's schedule uses, and it never synchronises the device,
never issues a collective, and never lets a failure escape into training.
"""

import ast
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest
import torch

import relax.utils.straggler as straggler
from relax.utils.straggler import megatron_timer_shim as shim_mod
from relax.utils.straggler.config import StragglerConfig
from relax.utils.straggler.megatron_timer_shim import (
    MAX_LOG_LEVEL,
    NullTimerSink,
    StragglerTimers,
    _NoopTimer,
)


class FakeClock:
    """Deterministic monotonic clock."""

    def __init__(self, start: float = 0.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now


class RecordingSink:
    """Captures completed intervals and can be told to fail."""

    def __init__(self, fail_acquire: bool = False, fail_complete: bool = False) -> None:
        self.intervals: List[Dict[str, Any]] = []
        self.outstanding = 0
        self.fail_acquire = fail_acquire
        self.fail_complete = fail_complete

    def acquire_interval(self, name: str, log_level: int) -> Optional[Dict[str, Any]]:
        if self.fail_acquire:
            raise RuntimeError("acquire exploded")
        self.outstanding += 1
        return {"name": name, "log_level": log_level, "seq": self.outstanding}

    def complete_interval(
        self,
        token: Any,
        name: str,
        log_level: int,
        host_start: float,
        host_end: float,
        barrier: bool,
    ) -> None:
        if self.fail_complete:
            raise RuntimeError("complete exploded")
        self.outstanding -= 1
        self.intervals.append(
            {
                "token": token,
                "name": name,
                "log_level": log_level,
                "host_start": host_start,
                "host_end": host_end,
                "barrier": barrier,
            }
        )

    def stats(self) -> Dict[str, int]:
        return {"outstanding": self.outstanding}


class _OwnerStub:
    """Stands in for StragglerTimers, which reaches the rank through its sink."""

    def __init__(self, rank: int) -> None:
        self.identity = _IdentityStub(rank)


class _IdentityStub:
    """Minimal stand-in for RuntimeIdentity, which only needs `.rank` here."""

    def __init__(self, rank: int) -> None:
        self.rank = rank


def make_timers(
    sink: Optional[RecordingSink] = None,
    clock: Optional[FakeClock] = None,
    log_level: int = 2,
) -> tuple:
    config = StragglerConfig(enabled=True, timer_log_level=log_level)
    fake_clock = clock if clock is not None else FakeClock()
    timers = StragglerTimers(config, sink=sink, clock=fake_clock)
    return timers, fake_clock


def test_disabled_by_default_returns_none(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("RELAX_STRAGGLER_ENABLE", raising=False)
    straggler.reset_straggler_state_for_tests()

    assert straggler.get_straggler_timers() is None
    assert straggler.is_straggler_profiler_enabled() is False


def test_enabled_env_returns_singleton(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RELAX_STRAGGLER_ENABLE", "1")
    straggler.reset_straggler_state_for_tests()

    first = straggler.get_straggler_timers()
    second = straggler.get_straggler_timers()

    assert isinstance(first, StragglerTimers)
    assert first is second


def test_malformed_knob_disables_profiler_without_raising(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RELAX_STRAGGLER_ENABLE", "ture")
    straggler.reset_straggler_state_for_tests()

    assert straggler.get_straggler_timers() is None  # fail-open, never raises


def test_failed_construction_disables_profiler_without_raising(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RELAX_STRAGGLER_ENABLE", "1")

    def explode(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError("boom")

    monkeypatch.setattr(straggler, "StragglerRuntime", explode)
    straggler.reset_straggler_state_for_tests()

    assert straggler.get_straggler_timers() is None


def test_megatron_call_shapes_are_supported() -> None:
    """Every shape observed in megatron/core must work unchanged."""
    timers, _ = make_timers(RecordingSink())

    timers("forward-backward", log_level=1).start(barrier=True)
    timers("forward-backward").stop()
    timers("forward-compute", log_level=2).start()
    timers("forward-compute").stop()
    timers("optimizer-inner-step", log_level=1).start(barrier=False)
    timers("optimizer-inner-step").stop()
    timers("params-all-gather", log_level=1).start(barrier=True)
    timers("params-all-gather").stop()

    assert timers.stats()["intervals"] == 4
    assert timers.stats()["timer_names"] == 4


def test_repeated_access_returns_same_handle() -> None:
    timers, fake_clock = make_timers(RecordingSink())

    handle = timers("forward-compute", log_level=2)
    fake_clock.now = 0.5
    handle.start()
    fake_clock.now = 1.5
    handle.stop()

    assert timers("forward-compute", log_level=2) is handle
    assert timers("forward-compute").elapsed(reset=False) == pytest.approx(1.0)


def test_level_mismatch_does_not_raise() -> None:
    timers, _ = make_timers(RecordingSink())

    timers("forward-compute", log_level=2)
    timers("forward-compute", log_level=1)  # Megatron asserts here; we count

    assert timers.stats()["level_mismatch"] == 1


def test_log_level_filtering_returns_noop_above_configured_level() -> None:
    timers, _ = make_timers(RecordingSink(), log_level=1)

    filtered = timers("forward-compute", log_level=2)
    captured = timers("forward-backward", log_level=1)

    assert isinstance(filtered, _NoopTimer)
    assert not isinstance(captured, _NoopTimer)
    filtered.start(barrier=True)
    filtered.stop()
    assert timers.stats()["intervals"] == 0


def test_default_log_level_is_maximum() -> None:
    timers, _ = make_timers(RecordingSink())

    assert timers("forward-compute") is not timers("forward-backward")
    assert timers.log_level == MAX_LOG_LEVEL


def test_noop_timer_raises_on_elapsed_like_megatron() -> None:
    timers, _ = make_timers(RecordingSink(), log_level=1)

    filtered = timers("forward-compute", log_level=2)

    with pytest.raises(RuntimeError):
        filtered.elapsed()
    with pytest.raises(RuntimeError):
        filtered.active_time()


def test_interval_records_host_bounds_and_barrier_request() -> None:
    sink = RecordingSink()
    timers, fake_clock = make_timers(sink)

    fake_clock.now = 10.0
    timers("optimizer-inner-step", log_level=1).start(barrier=True)
    fake_clock.now = 10.25
    timers("optimizer-inner-step").stop(barrier=True)

    assert len(sink.intervals) == 1
    interval = sink.intervals[0]
    assert interval["name"] == "optimizer-inner-step"
    assert interval["log_level"] == 1
    assert interval["host_start"] == pytest.approx(10.0)
    assert interval["host_end"] == pytest.approx(10.25)
    assert interval["barrier"] is True
    assert interval["token"] == {"name": "optimizer-inner-step", "log_level": 1, "seq": 1}
    assert timers.stats()["ignored_barriers"] == 2


def test_active_time_accumulates_across_intervals() -> None:
    sink = RecordingSink()
    timers, fake_clock = make_timers(sink)
    handle = timers("forward-backward", log_level=1)

    for duration in (0.1, 0.2, 0.3):
        fake_clock.now += duration
        handle.start()
        fake_clock.now += duration
        handle.stop()

    assert handle.active_time() == pytest.approx(0.6)
    assert sink.outstanding == 0


def test_reset_closes_running_interval_and_returns_token() -> None:
    sink = RecordingSink()
    timers, fake_clock = make_timers(sink)
    handle = timers("forward-compute", log_level=2)

    fake_clock.now = 0.0
    handle.start()
    fake_clock.now = 1.0
    handle.reset()

    assert sink.outstanding == 0
    assert handle.elapsed(reset=False) == 0.0
    assert handle.active_time() == pytest.approx(1.0)


def test_elapsed_does_not_touch_a_running_interval() -> None:
    sink = RecordingSink()
    timers, fake_clock = make_timers(sink)
    handle = timers("forward-compute", log_level=2)

    fake_clock.now = 0.0
    handle.start()
    fake_clock.now = 2.0
    assert handle.elapsed() == 0.0  # no completed interval yet

    fake_clock.now = 3.0
    handle.stop()
    assert handle.elapsed(reset=False) == pytest.approx(3.0)
    assert sink.outstanding == 0


def test_unbalanced_start_and_stop_are_counted_not_raised() -> None:
    timers, _ = make_timers(RecordingSink())
    handle = timers("forward-compute", log_level=2)

    handle.stop()  # never started
    handle.start()
    handle.start()  # already started
    handle.stop()

    stats = timers.stats()
    assert stats["unbalanced_stop"] == 1
    assert stats["unbalanced_start"] == 1
    assert stats["intervals"] == 1


def test_sink_acquire_failure_degrades_to_host_only() -> None:
    sink = RecordingSink(fail_acquire=True)
    timers, fake_clock = make_timers(sink)

    handle = timers("forward-compute", log_level=2)
    fake_clock.now = 0.0
    handle.start()
    fake_clock.now = 0.5
    handle.stop()

    stats = timers.stats()
    assert stats["sink_errors"] == 1
    assert stats["intervals"] == 1
    assert handle.elapsed(reset=False) == pytest.approx(0.5)


def test_sink_complete_failure_does_not_escape() -> None:
    sink = RecordingSink(fail_complete=True)
    timers, fake_clock = make_timers(sink)

    handle = timers("forward-compute", log_level=2)
    handle.start()
    fake_clock.now = 0.5
    handle.stop()

    assert timers.stats()["sink_errors"] == 1
    assert handle.active_time() == pytest.approx(0.5)


def test_sink_without_stats_is_tolerated() -> None:
    class BareSink:
        def acquire_interval(self, name: str, log_level: int) -> None:
            return None

        def complete_interval(
            self, token: Any, name: str, log_level: int, host_start: float, host_end: float, barrier: bool
        ) -> None:
            return None

    timers = StragglerTimers(StragglerConfig(enabled=True), sink=BareSink(), clock=FakeClock())

    assert timers.stats()["sink"] == {}


def test_null_sink_is_the_default() -> None:
    timers = StragglerTimers(StragglerConfig(enabled=True), clock=FakeClock())

    assert isinstance(timers._sink, NullTimerSink)
    handle = timers("forward-compute", log_level=2)
    handle.start()
    handle.stop()
    assert timers.stats()["intervals"] == 1


def test_set_barrier_group_and_set_elapsed_are_accepted() -> None:
    timers, _ = make_timers(RecordingSink())
    handle = timers("params-all-gather", log_level=1)

    handle.set_barrier_group(object())
    handle.set_elapsed(1.25)

    assert handle.elapsed() == pytest.approx(1.25)


def test_start_stop_never_synchronize_or_barrier(monkeypatch: pytest.MonkeyPatch) -> None:
    """A suppressed sync call would be swallowed by the fail-open handler, so
    the counters must stay clean as well."""
    timers, _ = make_timers(RecordingSink())

    def forbidden(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("the profiler must not synchronise")

    monkeypatch.setattr(torch.cuda, "synchronize", forbidden)
    monkeypatch.setattr(torch.distributed, "barrier", forbidden)

    handle = timers("forward-backward", log_level=1)
    handle.start(barrier=True)
    handle.stop(barrier=True)

    stats = timers.stats()
    assert stats["intervals"] == 1
    assert stats["sink_errors"] == 0
    assert stats["ignored_barriers"] == 2


def test_profiler_sources_contain_no_sync_or_barrier_calls() -> None:
    """Static guard: the package must not call these APIs anywhere.

    Only *calls* are flagged: ``barrier`` is a legitimate field name on the
    interval and envelope records.
    """
    forbidden = {"synchronize", "barrier", "all_reduce", "all_gather"}
    package_dir = Path(straggler.__file__).parent
    offenders: List[str] = []
    for path in sorted(package_dir.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if isinstance(func, ast.Attribute) and func.attr in forbidden:
                offenders.append(f"{path.name}:{node.lineno} call .{func.attr}()")
            elif isinstance(func, ast.Name) and func.id in forbidden:
                offenders.append(f"{path.name}:{node.lineno} call {func.id}()")

    assert offenders == []


def test_timers_survive_megatron_style_deepcopy_and_asdict() -> None:
    """Megatron deep-copies and asdicts configs while building modules."""
    import copy
    import dataclasses

    @dataclasses.dataclass
    class FakeModelConfig:
        timers: Any = None
        barrier_with_L1_time: bool = True

    timers, _ = make_timers(RecordingSink())
    config = FakeModelConfig(timers=timers)

    assert copy.deepcopy(config).timers is timers
    assert dataclasses.asdict(config)["timers"] is timers


def _set_injection(monkeypatch: pytest.MonkeyPatch, delay_ms: float, rank: int, stage: str) -> None:
    """Point the test-only injection knobs at fixed values."""
    monkeypatch.setattr(shim_mod, "_debug_settings", (delay_ms, rank, stage))


def test_injection_is_inert_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    _set_injection(monkeypatch, 0.0, -1, "")
    timers, _ = make_timers()
    handle = timers("forward-compute")

    assert shim_mod._injected_delay_s(handle._owner, "forward-compute") == 0.0


def test_injection_targets_only_the_named_rank_and_stage(monkeypatch: pytest.MonkeyPatch) -> None:
    _set_injection(monkeypatch, 12.0, 1, "")
    assert shim_mod._injected_delay_s(_OwnerStub(rank=1), "forward-compute") == pytest.approx(0.012)

    _set_injection(monkeypatch, 12.0, 0, "")
    assert shim_mod._injected_delay_s(_OwnerStub(rank=1), "forward-compute") == 0.0

    _set_injection(monkeypatch, 12.0, 1, "backward-compute")
    assert shim_mod._injected_delay_s(_OwnerStub(rank=1), "forward-compute") == 0.0

    _set_injection(monkeypatch, 12.0, -1, "")
    assert shim_mod._injected_delay_s(None, "forward-compute") == pytest.approx(0.012)


def test_injection_delay_is_applied_inside_the_measured_interval(monkeypatch: pytest.MonkeyPatch) -> None:
    """The sleep must happen between the two clock reads, like a real slow stage."""
    slept: list = []
    monkeypatch.setattr(shim_mod.time, "sleep", lambda seconds: slept.append(seconds))
    sink = RecordingSink()
    sink.identity = _IdentityStub(rank=0)
    timers, clock = make_timers(sink=sink)
    handle = timers("forward-compute")
    _set_injection(monkeypatch, 25.0, 0, "forward-compute")

    handle.start()
    clock.now += 0.5
    handle.stop()

    assert slept == [pytest.approx(0.025)]
    assert sink.intervals[0]["host_end"] - sink.intervals[0]["host_start"] == pytest.approx(0.5)
