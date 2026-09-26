# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Unit tests for turning straggler evidence into platform metrics."""

from argparse import Namespace
from typing import Any, Dict, List, Optional
from unittest.mock import Mock

import pytest

import relax.utils.straggler as straggler_package
from relax.utils.straggler import context, reporter
from relax.utils.straggler.reporter import STRAGGLER_METRIC_KEYS, STRAGGLER_METRIC_PREFIX
from relax.utils.training import train_metric_utils


class StubRuntime:
    """A runtime exposing only the two methods the reporter reads."""

    def __init__(
        self,
        summary: Dict[str, Any],
        verdicts: Optional[List[Any]] = None,
        report_error: Optional[Exception] = None,
    ) -> None:
        self._summary = summary
        self._verdicts = list(verdicts or [])
        self._report_error = report_error

    def report(self) -> Dict[str, Any]:
        if self._report_error is not None:
            raise self._report_error
        return self._summary

    def drain_verdicts(self) -> List[Any]:
        return list(self._verdicts)


class VerdictStub:
    def __init__(self, rank: int, deviation: float) -> None:
        self.rank = rank
        self.deviation = deviation


@pytest.fixture(autouse=True)
def _clean_state():
    reporter.reset_for_tests()
    context.reset_training_context_for_tests()
    yield
    reporter.reset_for_tests()
    context.reset_training_context_for_tests()


def _prefixed(name: str) -> str:
    return f"{STRAGGLER_METRIC_PREFIX}{name}"


def test_build_metrics_emits_only_prefixed_scalars() -> None:
    runtime = StubRuntime(
        {
            "active_stragglers": [{"rank": 1}, {"rank": 2}],
            "verdicts": 4,
            "windows_closed": 7,
            "coverage_ratio": 0.5,
            "dropped_queue_full": 2,
        },
        verdicts=[VerdictStub(2, 0.4), VerdictStub(1, 0.1)],
    )

    metrics = reporter.build_metrics(runtime)

    assert metrics
    assert all(key.startswith(STRAGGLER_METRIC_PREFIX) for key in metrics)
    assert all(isinstance(value, (int, float)) and not isinstance(value, bool) for value in metrics.values())
    assert set(metrics) <= {_prefixed(name) for name in STRAGGLER_METRIC_KEYS}
    assert metrics[_prefixed("active_stragglers")] == 2
    assert metrics[_prefixed("verdicts")] == 4
    assert metrics[_prefixed("windows_closed")] == 7
    assert metrics[_prefixed("coverage")] == 0.5
    assert metrics[_prefixed("dropped")] == 2
    assert metrics[_prefixed("worst_deviation")] == 0.4
    assert metrics[_prefixed("worst_rank")] == 2


def test_build_metrics_omits_values_that_cannot_be_measured() -> None:
    metrics = reporter.build_metrics(StubRuntime({}))

    assert metrics == {}


def test_judged_fraction_is_not_mislabelled_as_coverage() -> None:
    """``judged/envelopes`` is a transport ratio, never cohort coverage.

    The collector's real cohort coverage is not in the summary here, so
    ``coverage`` must be absent; the in-process ratio is published under its own
    key. Reading it as coverage used to show 1.0 on a run whose cohort coverage
    was far lower.
    """
    metrics = reporter.build_metrics(StubRuntime({"envelopes": 10, "judged_packets": 4}))

    assert _prefixed("coverage") not in metrics
    assert metrics[_prefixed("judged_fraction")] == 0.4


def test_coverage_is_emitted_only_from_an_explicit_cohort_ratio() -> None:
    metrics = reporter.build_metrics(StubRuntime({"coverage_ratio": 0.75, "envelopes": 10, "judged_packets": 10}))

    assert metrics[_prefixed("coverage")] == 0.75
    assert metrics[_prefixed("judged_fraction")] == 1.0


def test_every_drop_counter_is_published_in_dropped() -> None:
    """The detector's capped structures must not be invisible.

    ``MAX_SAMPLES_PER_RANK`` and the other caps report their evictions through
    the detector counters; the aggregate ``dropped`` key used to omit them, so a
    window that dropped 512 samples published no drop at all.
    """
    metrics = reporter.build_metrics(
        StubRuntime(
            {
                "sample_evictions": 512,
                "pair_evictions": 3,
                "rank_evictions": 2,
                "active_evictions": 1,
                "pending_line_drops": 4,
                "invalid_samples": 5,
            }
        )
    )

    assert metrics[_prefixed("dropped")] == 512 + 3 + 2 + 1 + 4 + 5


def test_build_metrics_never_raises_when_report_fails() -> None:
    runtime = StubRuntime({}, report_error=RuntimeError("report exploded"))
    before = reporter.error_count()

    metrics = reporter.build_metrics(runtime)

    assert metrics == {}
    assert reporter.error_count() == before + 1


def test_build_metrics_includes_the_training_context_when_set() -> None:
    context.set_training_context(5, 2, sample_seq=1, num_steps_per_rollout=4)

    metrics = reporter.build_metrics(StubRuntime({}))

    assert metrics[_prefixed("rollout_id")] == 5
    assert metrics[_prefixed("optimizer_step")] == 2
    # The monotonic profiler ordinal replaces the fabricated global step.
    assert metrics[_prefixed("step_ordinal")] == 1
    assert _prefixed("global_step") not in metrics
    # The publish counters are now on a surface that runs, not just a test helper.
    assert metrics[_prefixed("workload_publish_skipped")] == 0
    assert metrics[_prefixed("workload_publish_errors")] == 0


def test_reporter_omits_the_global_step_metric() -> None:
    """The in-rollout index is never exported as a run-wide step.

    ``record_optimizer_step`` with no run-wide value leaves ``global_step``
    ``None``; the reporter must omit it (it emits ``optimizer_step`` and the
    monotonic ``step_ordinal`` instead).
    """
    context.set_training_context(4, 9)

    metrics = reporter.build_metrics(StubRuntime({}))

    assert _prefixed("global_step") not in metrics
    assert metrics[_prefixed("optimizer_step")] == 9
    assert metrics[_prefixed("step_ordinal")] == 1


def test_build_metrics_surfaces_the_workload_counters() -> None:
    context.set_training_context(4, 9)
    context.count_workload_publish_skipped()
    context.count_workload_publish_skipped()
    context.count_workload_publish_error()

    metrics = reporter.build_metrics(StubRuntime({"workload_incomparable_windows": 3, "workload_missing_windows": 2}))

    assert metrics[_prefixed("workload_publish_skipped")] == 2
    assert metrics[_prefixed("workload_publish_errors")] == 1
    assert metrics[_prefixed("workload_incomparable_windows")] == 3
    assert metrics[_prefixed("workload_missing_windows")] == 2


def test_collector_status_available_marks_the_pp_gt_one_exporting_rank() -> None:
    """The PP>1 mismatch must be an explicit marker, not silence.

    The platform exports straggler metrics only on the Megatron primary rank,
    which owns the collector only when ``pp_size == 1``. A runtime without a
    collector (the PP>1 exporting rank) must say so.
    """
    from relax.utils.straggler.collector import TimingCollector
    from relax.utils.straggler.config import StragglerConfig
    from relax.utils.straggler.identity import RuntimeIdentity
    from relax.utils.straggler.runtime import StragglerRuntime

    config = StragglerConfig(enabled=True, window_seconds=1.0)
    sender = StragglerRuntime(
        config,
        identity=RuntimeIdentity(run_id="run-1", rank=3, world_size=8),
        register_atexit=False,
    )
    assert sender.collector is None

    metrics = reporter.build_metrics(sender)

    assert metrics[_prefixed("collector_status_available")] == 0.0

    owner = StragglerRuntime(
        config,
        identity=RuntimeIdentity(run_id="run-1", rank=0, world_size=8),
        register_atexit=False,
    )
    owner._collector = TimingCollector(config, identity=owner.identity)

    assert reporter.build_metrics(owner)[_prefixed("collector_status_available")] == 1.0


def test_report_once_marks_an_enabled_run_without_a_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(straggler_package, "is_straggler_profiler_enabled", lambda: True)
    monkeypatch.setattr(straggler_package, "get_straggler_runtime", lambda: None)

    metrics = reporter.report_once(Namespace(), 3)

    assert metrics == {_prefixed("collector_status_available"): 0.0}


def test_report_once_returns_nothing_when_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(straggler_package, "is_straggler_profiler_enabled", lambda: False)

    assert reporter.report_once(Namespace(), 3) == {}


def test_report_once_reads_the_runtime_when_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(straggler_package, "is_straggler_profiler_enabled", lambda: True)
    monkeypatch.setattr(straggler_package, "get_straggler_runtime", lambda: StubRuntime({"verdicts": 1}))

    metrics = reporter.report_once(Namespace(), 3)

    assert metrics == {_prefixed("verdicts"): 1.0}


def test_build_metrics_never_flushes_persistence(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """The metrics path runs on the training thread and must not write files.

    ``report_once``/``build_metrics`` read the runtime once per rollout. The
    collector's explicit ``report()`` flushes buffered JSONL lines, so the
    reporter must go through the non-flushing ``summary()`` accessor.
    """
    from pathlib import Path

    from relax.utils.straggler.collector import TimingCollector
    from relax.utils.straggler.config import StragglerConfig
    from relax.utils.straggler.identity import RuntimeIdentity
    from relax.utils.straggler.observer import TimingEnvelope
    from relax.utils.straggler.runtime import StragglerRuntime

    identity = RuntimeIdentity(run_id="run-1", rank=0, world_size=4)
    config = StragglerConfig(
        enabled=True,
        window_seconds=1.0,
        warmup_windows=0,
        persist_windows=1,
        output_dir=str(tmp_path),
        report_interval_seconds=3600.0,
    )
    runtime = StragglerRuntime(config, identity=identity, register_atexit=False)
    collector = TimingCollector(config, identity=identity)
    runtime._collector = collector
    collector.ingest(
        TimingEnvelope(
            run_id="run-1",
            rank=0,
            cohort="0:0:0:0:0",
            label="rank0/tp0/pp0",
            world_size=4,
            name="forward-compute",
            log_level=2,
            seq=1,
            host_start=0.1,
            host_end=0.2,
            device_ms=None,
            barrier=False,
            reason="no_event_pair",
        )
    )

    envelope_path = Path(tmp_path) / "straggler_envelopes.jsonl"
    assert envelope_path.read_text(encoding="utf-8") == ""
    flushes: List[Any] = []
    monkeypatch.setattr(collector, "_flush_path", lambda path: flushes.append(path))

    metrics = reporter.build_metrics(runtime)

    assert flushes == []
    assert envelope_path.read_text(encoding="utf-8") == ""
    assert collector.status()["pending_lines"][str(envelope_path)] == 1
    assert isinstance(metrics, dict)

    # The explicit diagnostic/close path keeps flushing.
    collector.report()
    assert flushes == [str(envelope_path)]


class FakeTimer:
    """Mirrors tests/utils/test_train_metric_utils.py."""

    def __init__(self) -> None:
        self.seq_lens = [100, 200]
        self.response_lens = [50, 100]

    def log_dict(self) -> Dict[str, float]:
        return {
            "actor_train": 2.0,
            "log_probs": 1.0,
            "ref_log_probs": 1.0,
            "train_wait": 1.0,
            "train": 2.0,
        }

    def reset(self) -> None:
        pass


def _run_log_perf_data_raw(
    monkeypatch: pytest.MonkeyPatch,
    enabled: bool,
    straggler_metrics: Dict[str, float],
    merge: Optional[Any] = None,
) -> Dict[str, float]:
    timer = FakeTimer()
    flops_counter = Mock()
    flops_counter.estimate.return_value = (300.0, 100.0)
    logged_metrics: Dict[str, float] = {}

    def default_merge(_args: Any, _rollout_id: int) -> Dict[str, float]:
        return dict(straggler_metrics)

    monkeypatch.setattr(train_metric_utils, "Timer", lambda: timer)
    monkeypatch.setattr(
        train_metric_utils.tracking_utils,
        "log",
        lambda _args, metrics, step_key: logged_metrics.update(metrics),
    )
    monkeypatch.setattr(train_metric_utils, "is_straggler_profiler_enabled", lambda: enabled)
    monkeypatch.setattr(train_metric_utils, "report_once", default_merge if merge is None else merge)

    train_metric_utils.log_perf_data_raw(
        rollout_id=3,
        args=Namespace(wandb_always_use_train_step=False),
        is_primary_rank=True,
        flops_counter=flops_counter,
        world_size=2,
    )

    return logged_metrics


def test_log_perf_data_raw_merges_straggler_metrics_when_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    metrics = _run_log_perf_data_raw(
        monkeypatch,
        enabled=True,
        straggler_metrics={_prefixed("verdicts"): 1.0},
    )

    assert metrics[_prefixed("verdicts")] == 1.0


def test_log_perf_data_raw_adds_no_straggler_keys_when_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    metrics = _run_log_perf_data_raw(
        monkeypatch,
        enabled=False,
        straggler_metrics={_prefixed("verdicts"): 1.0},
    )

    assert not any(key.startswith(STRAGGLER_METRIC_PREFIX) for key in metrics)


def test_log_perf_data_raw_keeps_normal_perf_keys_in_both_modes(monkeypatch: pytest.MonkeyPatch) -> None:
    reference = _run_log_perf_data_raw(monkeypatch, enabled=False, straggler_metrics={})
    normal_keys = {key for key in reference if not key.startswith(STRAGGLER_METRIC_PREFIX)}
    assert normal_keys

    merged = _run_log_perf_data_raw(
        monkeypatch,
        enabled=True,
        straggler_metrics={_prefixed("verdicts"): 9.0},
    )

    assert {key for key in merged if not key.startswith(STRAGGLER_METRIC_PREFIX)} == normal_keys
    for key in normal_keys:
        assert merged[key] == reference[key]


def test_log_perf_data_raw_survives_a_failing_straggler_merge(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(_args: Any, _rollout_id: int) -> Dict[str, float]:
        raise RuntimeError("straggler merge exploded")

    metrics = _run_log_perf_data_raw(monkeypatch, enabled=True, straggler_metrics={}, merge=boom)

    assert "perf/actor_train_time" in metrics
    assert not any(key.startswith(STRAGGLER_METRIC_PREFIX) for key in metrics)
