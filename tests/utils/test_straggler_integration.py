# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""CPU tests for the profiler inside Relax: when it gets installed, what the
training code's calls hand it, and the per-step reporting."""

import ast
import itertools
import sys
from argparse import Namespace
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from relax.utils.straggler import megatron_adapter
from relax.utils.straggler import relax_integration as integration
from relax.utils.straggler.analysis.detector import WindowReport
from relax.utils.straggler.analysis.report import REPORTED_METRICS
from relax.utils.straggler.relax_integration import StragglerProfiler, install_straggler_profiler
from relax.utils.timer import TimelineEvent, Timer
from tests.utils.straggler_helpers import FakeEvent, bracket, make_collector, meta


def _args(num_rollout=100):
    return Namespace(num_rollout=num_rollout, wandb_always_use_train_step=False, timeline_dump_dir=None)


def _profiler(interval=2, num_rollout=100, **kwargs):
    return StragglerProfiler(make_collector(interval=interval, **kwargs), _args(num_rollout))


def _step(profiler, rollout_id):
    bracket(profiler.collector)
    return profiler.rollout_end(rollout_id)


def test_scalars_arrive_when_a_window_closes_and_carry_its_rollout_range():
    profiler = _profiler(interval=2)
    assert _step(profiler, 0) is None
    metrics = _step(profiler, 1)
    assert metrics["straggler/window/first_rollout"] == 0.0 and metrics["straggler/window/last_rollout"] == 1.0
    assert set(metrics) <= REPORTED_METRICS


def test_a_failing_reporter_is_counted_and_never_raises(monkeypatch):
    def boom(*args, **kwargs):
        raise RuntimeError("timeline broke")

    monkeypatch.setattr(integration, "report_straggler_window", boom)
    profiler = _profiler(interval=1)
    assert _step(profiler, 0) is None
    assert profiler.collector.health.errors_total == 1


def _call_timers(profiler, phase, names):
    for name in names:
        timer = profiler.timers[phase](name)
        timer.start()
        timer.stop()


def test_a_rollout_that_calls_every_core_timer_leaves_the_rank_active():
    profiler = _profiler(interval=1)
    for phase, names in megatron_adapter.REQUIRED_TIMER_NAMES.items():
        _call_timers(profiler, phase, names)
    profiler.rollout_end(0)
    assert profiler.collector.health.gaps == frozenset() and profiler.collector.health.state == "active"


def test_core_timers_missing_from_the_first_rollout_of_a_phase_degrade_the_rank_for_good():
    profiler = _profiler(interval=1)
    profiler.rollout_end(0)
    assert profiler.collector.health.gaps == frozenset(), "no phase ran yet, so nothing is checked"
    _call_timers(profiler, "train", ["forward-compute", "backward-compute"])
    profiler.rollout_end(1)
    (gap,) = profiler.collector.health.gaps
    assert gap.startswith(
        "the first rollout with train steps never called the Megatron timers all-grads-sync, "
        "optimizer-copy-to-main-grad"
    )
    _call_timers(profiler, "train", megatron_adapter.REQUIRED_TIMER_NAMES["train"])
    _call_timers(profiler, "forward_only", ["forward-compute"])
    profiler.rollout_end(2)
    assert profiler.collector.health.gaps == {gap} and profiler.collector.health.state == "degraded"


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
    monkeypatch.setattr(device, "get_device_name", lambda: "npu")


def test_timing_events_come_from_the_active_accelerator(monkeypatch):
    backend = _FakeBackend()
    _use_backend(monkeypatch, backend)
    integration.timing_event_factory()()
    assert backend.events == [{"enable_timing": True}]


def test_timing_events_on_cuda_are_device_generic_events(monkeypatch):
    torch = pytest.importorskip("torch")
    from relax.utils import device

    built = []
    monkeypatch.setattr(device, "get_device_name", lambda: "cuda")
    monkeypatch.setattr(torch, "Event", lambda **kwargs: built.append(kwargs))
    integration.timing_event_factory()()
    assert built == [{"device": "cuda", "enable_timing": True}]


@pytest.mark.parametrize(
    ("capturing", "available", "expected"), [(True, True, True), (False, True, False), (True, False, False)]
)
def test_capture_check_uses_the_backend_function_when_there_is_one(monkeypatch, capturing, available, expected):
    _use_backend(monkeypatch, _FakeBackend(capturing=capturing), available=available)
    assert integration.stream_capture_check()() is expected


def test_capture_check_is_false_when_the_backend_has_none(monkeypatch):
    _use_backend(monkeypatch, _FakeBackend())
    assert integration.stream_capture_check()() is False


def test_timers_are_none_without_a_profiler():
    assert integration.straggler_timers("train") is None


def test_an_unknown_phase_is_refused(monkeypatch):
    monkeypatch.setattr(integration, "_PROFILER", _profiler())
    with pytest.raises(ValueError, match="unknown straggler phase 'eval'"):
        integration.straggler_timers("eval")


class _DeviceLengths:
    def __iter__(self):
        raise AssertionError("summing device lengths would synchronize")


def test_micro_batch_tokens_reach_the_profiler_only_as_host_lists(monkeypatch):
    integration.note_microbatch_tokens([1, 2])  # no profiler: nothing happens
    profiler = _profiler()
    noted = []
    monkeypatch.setattr(integration, "_PROFILER", profiler)
    monkeypatch.setattr(profiler.collector.recorder, "note_tokens", noted.append)
    for lengths in ([100, 250], (7,), None, _DeviceLengths()):
        integration.note_microbatch_tokens(lengths)
    assert noted == [350, 7]


@pytest.mark.parametrize(
    ("env", "trace_on", "world", "expected"),
    [
        (None, True, 256, 2),
        (None, True, 257, 1),
        (None, False, 8, 1),
        ("0", True, 8, 0),
        ("1", True, 8, 1),
        ("2", True, 4096, 2),
        ("3", True, 8, 3),
    ],
)
def test_the_start_level_comes_from_the_env_or_from_whether_traces_can_be_written_and_the_job_size(
    monkeypatch, env, trace_on, world, expected
):
    if env is None:
        monkeypatch.delenv("RELAX_STRAGGLER_LEVEL", raising=False)
    else:
        monkeypatch.setenv("RELAX_STRAGGLER_LEVEL", env)
    assert integration.initial_level(trace_on, world) == expected


class _Chunk:
    def modules(self):
        return []


def test_the_first_call_with_a_model_hands_it_to_the_collector(monkeypatch):
    monkeypatch.setattr(integration, "_PROFILER", None)
    assert integration.straggler_timers("train", [_Chunk()], 4) is None
    profiler = _profiler()
    monkeypatch.setattr(integration, "_PROFILER", profiler)
    assert integration.straggler_timers("train") is profiler.timers["train"]
    assert profiler.collector.recorder.module_timers is None, "the optimizer config's call names no model"
    assert integration.straggler_timers("forward_only", [_Chunk(), _Chunk()]) is profiler.timers["forward_only"]
    first = profiler.collector.recorder.module_timers
    assert not first.counts() and profiler.collector.recorder.model_chunks == 2
    integration.straggler_timers("train", [object()])
    assert profiler.collector.recorder.module_timers is first, "only the first model is taken"


def test_a_model_without_modules_is_a_profiler_error(monkeypatch):
    profiler = _profiler()
    monkeypatch.setattr(integration, "_PROFILER", profiler)
    integration.straggler_timers("train", [object()])
    assert profiler.collector.health.errors_total == 1
    assert profiler.collector.recorder.module_timers is None


def test_traces_go_to_the_env_dir_or_under_the_timeline_dir(monkeypatch):
    monkeypatch.delenv("RELAX_STRAGGLER_TRACE_DIR", raising=False)
    assert integration.trace_dir(Namespace(timeline_dump_dir=None)) == ""
    assert integration.trace_dir(Namespace(timeline_dump_dir="/runs/x/timeline")) == "/runs/x/timeline/straggler"
    monkeypatch.setenv("RELAX_STRAGGLER_TRACE_DIR", "/traces")
    assert integration.trace_dir(Namespace(timeline_dump_dir="/runs/x/timeline")) == "/traces"


@pytest.fixture
def timer_records():
    Timer().records.clear()
    yield Timer().records
    Timer().records.clear()


def test_cpu_events_are_the_finished_timers_overlapping_the_step_without_call_stacks(timer_records):
    timer_records.extend(
        [
            TimelineEvent("train_wait", 1.0, 2.0, pid=1, tid=1),
            TimelineEvent("get_data_batch_a", 9.0, 11.0, pid=1, tid=1, call_stack="  File ..."),
            TimelineEvent("straggler/fwd rank0_dp0_tp0_pp0", 10.0, 12.0, pid=1, tid=1),
            TimelineEvent("next_step", 21.0, 22.0, pid=1, tid=1),
        ]
    )
    events = integration._cpu_events(10.0, 20.0)
    assert [event["name"] for event in events] == ["get_data_batch_a"]
    assert events[0]["args"] == {}


def test_only_the_latest_of_several_ready_windows_reaches_the_metrics(monkeypatch):
    def window(n):
        return WindowReport(
            metrics={"straggler/window/last_rollout": float(n)}, alerts=[], rows=[], first_rollout=n, last_rollout=n
        )

    reported = []
    monkeypatch.setattr(
        integration, "report_straggler_window", lambda args, report: reported.append(report) or dict(report.metrics)
    )
    profiler = _profiler(interval=1)
    monkeypatch.setattr(profiler.collector, "end_step", lambda rollout_id, final=False: [window(0), window(1)])

    metrics = profiler.rollout_end(1)
    assert metrics == {"straggler/window/last_rollout": 1.0}
    assert [r.last_rollout for r in reported] == [0, 1], "every window still reaches the timeline"
    assert profiler.collector.health.errors_total == 1


def _ready_windows(monkeypatch, profiler, rollouts, failing):
    """Make ``rollout_end`` see the windows of ``rollouts`` ready at once, with
    the reporter failing on ``failing``."""

    def report(args, window):
        if window.last_rollout == failing:
            raise RuntimeError("timeline broke")
        return dict(window.metrics)

    def window(n):
        return WindowReport(
            metrics={"straggler/window/last_rollout": float(n)}, alerts=[], rows=[], first_rollout=n, last_rollout=n
        )

    monkeypatch.setattr(integration, "report_straggler_window", report)
    monkeypatch.setattr(profiler.collector, "end_step", lambda rollout_id, final=False: [window(n) for n in rollouts])


def test_a_failing_latest_window_falls_back_to_the_previous_one_and_counts_one_error(monkeypatch):
    profiler = _profiler(interval=1)
    _ready_windows(monkeypatch, profiler, rollouts=[1, 2], failing=2)
    for errors in (1, 2):
        assert profiler.rollout_end(2) == {"straggler/window/last_rollout": 1.0}
        assert profiler.collector.health.errors_total == errors, "only the failing window is lost"


def test_a_superseded_and_a_failed_window_count_one_error_each(monkeypatch):
    profiler = _profiler(interval=1)
    _ready_windows(monkeypatch, profiler, rollouts=[0, 1, 2], failing=2)
    assert profiler.rollout_end(2) == {"straggler/window/last_rollout": 1.0}
    assert profiler.collector.health.errors_total == 2, "window 0 superseded, window 2 failed"


def test_a_window_superseded_at_the_last_step_is_not_an_error(monkeypatch):
    profiler = _profiler(interval=1, num_rollout=100)
    _ready_windows(monkeypatch, profiler, rollouts=[98, 99], failing=-1)
    assert profiler.rollout_end(99) == {"straggler/window/last_rollout": 99.0}
    assert profiler.collector.health.errors_total == 0


@pytest.mark.parametrize(("primary", "vpp"), [(True, None), (False, 2)])
def test_install_uses_the_metrics_primary_rank_and_names_brackets_by_schedule_only_under_vpp(
    monkeypatch, primary, vpp
):
    monkeypatch.setenv("RELAX_STRAGGLER_PROFILER", "1")
    monkeypatch.setattr(megatron_adapter, "is_metrics_primary_rank", lambda: primary)
    monkeypatch.setattr(integration, "local_rank_meta", lambda: meta(1)[0])
    monkeypatch.setattr(integration.dist, "get_world_size", lambda group=None: 1)
    monkeypatch.setattr(integration, "timing_event_factory", lambda: FakeEvent)
    monkeypatch.setattr(integration, "stream_capture_check", lambda: integration._never)
    # The gather group is a collective; its creation is covered separately.
    monkeypatch.setattr(integration, "init_straggler_gather_group", lambda timeout_s: None)
    monkeypatch.setattr(integration, "_PROFILER", None)
    args = _args()
    args.virtual_pipeline_model_parallel_size = vpp
    profiler = install_straggler_profiler("actor", args)
    try:
        assert profiler.collector.is_primary is primary
        assert profiler.collector.recorder._cpu_events_source is (integration._cpu_events if primary else None)
        assert profiler.collector.recorder._interleaved is (vpp is not None)
    finally:
        profiler.collector.close()


# Megatron's schedule table for 4 micro-batches over 2 model chunks, in groups of 2.
_TABLE_4 = [(0, 0), (1, 0), (0, 1), (1, 1), (2, 0), (3, 0), (2, 1), (3, 1)]


@pytest.mark.parametrize(
    ("forward_only", "combined", "expected"), [(False, False, _TABLE_4), (True, True, _TABLE_4), (False, True, [])]
)
def test_the_schedule_table_comes_from_the_model_config(monkeypatch, forward_only, combined, expected):
    profiler = _profiler(interleaved=True)
    monkeypatch.setattr(integration, "_PROFILER", profiler)
    integration.straggler_timers("forward_only" if forward_only else "train", _vpp_model(combined), 4)
    _, table, _ = profiler.collector.recorder._schedule
    assert list(table) == expected


def _vpp_model(combined=False):
    # The launch arguments may ask for groups of 3; Megatron's schedule reads the wrapped model's config.
    config = Namespace(microbatch_group_size_per_vp_stage=2, overlap_moe_expert_parallel_comm=combined)
    chunk = Namespace(module=Namespace(config=config), modules=lambda: [])
    return [chunk, chunk]


def test_the_calls_of_one_forward_only_number_their_micro_batches_in_order(monkeypatch):
    profiler = _profiler(interleaved=True)
    monkeypatch.setattr(integration, "_PROFILER", profiler)
    integration.straggler_timers("forward_only", _vpp_model(), [4, 2])
    _, table, _ = profiler.collector.recorder._schedule
    assert list(table) == _TABLE_4 + [(4, 0), (5, 0), (4, 1), (5, 1)]


def test_without_a_count_no_schedule_is_noted(monkeypatch):
    profiler = _profiler(interleaved=True)
    monkeypatch.setattr(integration, "_PROFILER", profiler)
    integration.straggler_timers("train", _vpp_model())
    assert profiler.collector.recorder._schedule is None


@pytest.mark.parametrize(("tp", "last_pp", "dp"), list(itertools.product([0, 1], [True, False], [0, 1])))
def test_the_primary_rank_is_tp0_last_pp_stage_dp0(monkeypatch, tp, last_pp, dp):
    core = ModuleType("megatron.core")
    core.mpu = SimpleNamespace(
        get_tensor_model_parallel_rank=lambda: tp,
        is_pipeline_last_stage=lambda: last_pp,
        get_data_parallel_rank=lambda with_context_parallel=False: dp if with_context_parallel else 0,
    )
    monkeypatch.setitem(sys.modules, "megatron", ModuleType("megatron"))
    monkeypatch.setitem(sys.modules, "megatron.core", core)

    assert megatron_adapter.is_metrics_primary_rank() is (tp == 0 and last_pp and dp == 0)


def _function(path, name):
    tree = ast.parse(Path(path).read_text())
    return next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == name)


def test_the_primary_rank_rule_is_the_one_log_perf_data_logs_on():
    # The window scalars reach the metrics only through that rank's log_perf_data.
    repo = Path(__file__).resolve().parents[2]
    log_perf_data = _function(repo / "relax/backends/megatron/data.py", "log_perf_data")
    (logged_on,) = (
        keyword.value
        for node in ast.walk(log_perf_data)
        if isinstance(node, ast.Call)
        for keyword in node.keywords
        if keyword.arg == "is_primary_rank"
    )
    (rule,) = (
        node.value
        for node in ast.walk(_function(Path(megatron_adapter.__file__), "is_metrics_primary_rank"))
        if isinstance(node, ast.Return)
    )
    assert ast.dump(rule) == ast.dump(logged_on)
