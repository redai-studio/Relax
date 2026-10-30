# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Execute the real Controller and Service definitions with CPU dependencies.

The classes and their control flow come from production source. Model
deployment, placement groups, HTTP/Ray transport and process exit are explicit
substitutes. No GPU initialization or Ray cluster is started by these local
recovery tests.
"""

import ast
import sys
import threading
import time
from dataclasses import asdict
from enum import Enum, IntEnum
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest

from relax.utils.fault_events.models import FaultContext, RecoveryEvidence, new_id
from relax.utils.fault_events.recovery import expected_evidence
from relax.utils.fault_events.runtime import FaultEventRuntime


try:
    from enum import StrEnum
except ImportError:
    # Match the production registry's string-enum contract on Python 3.10.
    class StrEnum(str, Enum):
        def __str__(self) -> str:
            return self.value


ROOT = Path(__file__).resolve().parents[2]


class Roles(StrEnum):
    actor = "actor"
    actor_fwd = "actor_fwd"
    rollout = "rollout"
    critic = "critic"
    reference = "reference"
    advantages = "advantages"


def load_classes(path, names, namespace):
    tree = ast.parse((ROOT / path).read_text())
    selected = [node for node in tree.body if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in names]
    module = ast.Module(body=ast.parse("from __future__ import annotations").body + selected, type_ignores=[])
    namespace = {
        "__name__": "fault_local_cpu",
        "Enum": Enum,
        "IntEnum": IntEnum,
        "RecoveryEvidence": RecoveryEvidence,
        "expected_evidence": expected_evidence,
        **namespace,
    }
    exec(compile(ast.fix_missing_locations(module), str(ROOT / path), "exec"), namespace)
    return namespace


class Remote:
    def __init__(self, function):
        self.remote = function


@pytest.fixture(autouse=True)
def cpu_algorithm_registry(monkeypatch):
    # Controller recovery resolves roles through the registry, whose real
    # module imports model components; substitute the same CPU role set.
    registry = ModuleType("relax.core.registry")
    registry.resolve_algorithm = lambda config: {
        role: object for role in (Roles.actor, Roles.actor_fwd, Roles.advantages)
    }
    monkeypatch.setitem(sys.modules, "relax.core.registry", registry)


@pytest.fixture
def runtime():
    value = FaultEventRuntime()
    value.bind_source()
    yield value
    value.close()
    value._consumer.join(3)
    assert not value._consumer.is_alive()


def observe(runtime, roles=(Roles.advantages,)):
    faults = [
        FaultContext(str.__str__(role), new_id(), runtime.generation, new_id(), "reported_error") for role in roles
    ]
    runtime.observe_snapshot(
        {
            "source_generation": runtime.generation,
            "roles": list(roles),
            "faults": [asdict(fault) for fault in faults],
            "observation_error": False,
        }
    )
    return faults


def records(runtime):
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        status = runtime.diagnostics()
        if not status["queued"] and not status["inflight"]:
            return runtime.query(limit=100)["events"]
        time.sleep(0.001)
    pytest.fail("event consumer failed to drain")


def make_service(trace, *, failure=None, observer_log_failure=False):
    logger = Mock()
    if observer_log_failure:

        def warning(message, *args, **kwargs):
            if message == "Fault observation unavailable":
                raise RuntimeError("observation log failure")

        logger.warning.side_effect = warning
    namespace = {
        "logger": logger,
        "time": SimpleNamespace(sleep=lambda seconds: trace.append(("sleep", seconds))),
        "serve": SimpleNamespace(delete=lambda role: trace.append(("delete", role))),
        "ray": SimpleNamespace(get=lambda value: value),
        "recovery_load_path": lambda config: trace.append("load_path"),
    }
    cls = load_classes("relax/core/service.py", {"Service", "_observe_recovery"}, namespace)["Service"]
    service = object.__new__(cls)
    service.role = Roles.advantages
    service.config = SimpleNamespace()
    service._task_ref = None
    service._stop_heartbeat_thread = lambda: trace.append("stop_heartbeat")
    service._start_heartbeat = lambda: trace.append("start_heartbeat")
    service._ensure_placement_group = lambda: "cpu_pg"

    def deploy(pg):
        trace.append(("deploy", pg))
        if failure == "deploy":
            raise RuntimeError("injected deployment failure")

    def run():
        trace.append("submit_task")
        if failure == "run":
            raise RuntimeError("injected submission failure")
        return None if failure == "no_task" else object()

    def http(role, path, **kwargs):
        trace.append(("http", role, path))
        if failure == "step" and path == "/set_step":
            raise RuntimeError("SENSITIVE_HTTP_ERROR")
        if path == "/get_step":
            return {"step": 7}
        return {"status": "ok"}

    service._deploy = deploy
    service._http_call = http
    service.handle = SimpleNamespace(run=Remote(run))
    service.healthy = SimpleNamespace(
        set_task_status=Remote(lambda role, running: trace.append(("task_status", role, running))),
        get_current_step=Remote(lambda **kwargs: 7),
    )
    return service


def make_controller(runtime, services, trace, *, count=1, max_global_restart=3, fully_async=True, debug=False):
    logger = Mock()

    def exit_process(code):
        trace.append(("exit", code))
        raise SystemExit(code)

    namespace = {
        "logger": logger,
        "threading": threading,
        "ROLES": Roles,
        "GENRM_ROLE": "genrm",
        "ALGOS": {"cpu": {}},
        "resolve_sft_algo_key": lambda config: "cpu",
        "register_extra_roles": lambda config, algo: None,
        "os": SimpleNamespace(_exit=exit_process),
    }
    cls = load_classes("relax/core/controller.py", {"Controller"}, namespace)["Controller"]
    controller = object.__new__(cls)
    controller.config = SimpleNamespace(
        fully_async=fully_async, hybrid=False, debug_train_only=debug, debug_rollout_only=False
    )
    controller.serve_dict = services
    controller._fault_events = runtime
    controller._fault_recovery = None
    controller._restart_consumed_event = threading.Event()
    controller._restart_consumed_event.set()
    controller._restart_lock = threading.RLock()
    controller._restart_done_event = threading.Event()
    controller._restart_error = None
    controller._restart_mode = None
    controller._restarting = False
    controller._global_restart_count = 0
    controller._max_global_restart = max_global_restart
    controller._metrics_service_enabled = False
    controller._pending_task_refs = []
    controller._pending_task_refs_lock = threading.Lock()
    controller._health_manager = SimpleNamespace(
        increment_restart_count=lambda role: count,
        mark_healthy=lambda role: trace.append(("healthy", role)),
        _checker=None,
    )
    return controller, logger


@pytest.mark.parametrize("failure", [None, "step", "deploy", "run", "no_task"])
def test_actual_local_restart_preserves_business_trace_and_uses_real_step_evidence(runtime, failure):
    observe(runtime)
    trace, baseline_trace = [], []
    service = make_service(trace, failure=failure)
    controller, _ = make_controller(runtime, {Roles.advantages: service}, trace)
    baseline_service = make_service(baseline_trace, failure=failure)
    baseline, _ = make_controller(None, {Roles.advantages: baseline_service}, baseline_trace)
    baseline.restart_serve(Roles.advantages)
    controller.restart_serve(Roles.advantages)
    assert trace == baseline_trace
    assert controller._restart_mode == baseline._restart_mode == "local"
    assert controller._restart_done_event.is_set()
    assert type(controller._restart_error) is type(baseline._restart_error)
    events = records(runtime)
    terminals = [event for event in events if event["event_type"] in ("recovery_succeeded", "recovery_failed")]
    if failure == "no_task":
        assert not terminals
        assert "OUTCOME_UNOBSERVED" in runtime.diagnostics()["unknown_reasons"]
        assert not runtime._recoveries
    else:
        assert len(terminals) == 1
        terminal = terminals[0]
        assert terminal["event_type"] == ("recovery_succeeded" if failure is None else "recovery_failed")
        assert terminal["expected_evidence"] == 7
        if failure is None:
            assert terminal["observed_evidence"] == 7 and terminal["failed_evidence"] == 0
        elif failure == "step":
            assert controller._restart_error is None
            assert terminal["reason_code"] == "step_restore_failed" and terminal["failed_evidence"] == 2
            assert "submit_task" in trace
        else:
            assert terminal["reason_code"] == "restart_exception"
    assert "SENSITIVE_HTTP_ERROR" not in str(events)
    assert runtime.generation == 0


def test_observer_and_its_log_failure_do_not_change_actual_service_restart():
    trace, baseline = [], []
    observed_service = make_service(trace, observer_log_failure=True)
    baseline_service = make_service(baseline)

    def fail_observation(**kwargs):
        raise RuntimeError("observation failure")

    baseline_service.restart()
    observed_service.restart(fault_observer=fail_observation)
    assert trace == baseline


def test_default_off_never_evaluates_recovery_plan():
    trace = []
    service = make_service(trace)
    controller, _ = make_controller(None, {Roles.advantages: service}, trace)
    plan = Mock(side_effect=AssertionError("disabled observation must not evaluate a plan"))
    type(controller)._restart_serve.__globals__["expected_evidence"] = plan
    controller.restart_serve(Roles.advantages)
    assert "submit_task" in trace and controller._restart_error is None
    plan.assert_not_called()


@pytest.mark.parametrize(
    "mode,roles,options,expected",
    [
        ("local", (), {}, 7),
        ("global", (), {}, 5),
        ("global", (Roles.actor_fwd, Roles.reference), {"fully_async": True}, 5),
        ("global", (Roles.actor,), {}, 7),
        ("global", (Roles.actor,), {"fully_async": True}, 15),
        ("global", (Roles.actor, Roles.actor_fwd), {"fully_async": True}, 31),
        ("global", (Roles.actor, Roles.reference), {"fully_async": True}, 47),
        ("global", ("actor", "actor_fwd", "reference"), {"fully_async": True}, 63),
        ("global", (Roles.actor,), {"fully_async": True, "hybrid": True}, 7),
        ("global", (Roles.actor,), {"debug_train_only": True}, 5),
        ("global", (Roles.actor,), {"debug_rollout_only": True}, 5),
    ],
)
def test_recovery_plan_preserves_role_and_configuration_rules(mode, roles, options, expected):
    config = SimpleNamespace(**options)
    plan = expected_evidence(mode, config, dict.fromkeys(roles))
    assert type(plan) is int and plan == expected


def test_local_recovery_context_stays_owned_until_restart_is_complete_and_consumed(runtime):
    observe(runtime)
    trace = []
    service = make_service(trace)
    restart = service.restart
    entered, release, second_started = threading.Event(), threading.Event(), threading.Event()
    calls = []

    def blocked_restart(fault_observer=None):
        calls.append(fault_observer)
        entered.set()
        assert release.wait(3)
        restart(fault_observer=fault_observer)

    service.restart = blocked_restart
    controller, _ = make_controller(runtime, {Roles.advantages: service}, trace)
    first = threading.Thread(target=controller.restart_serve, args=(Roles.advantages,))

    def next_restart():
        second_started.set()
        controller.restart_serve(Roles.advantages)

    second = threading.Thread(target=next_restart)
    first.start()
    assert entered.wait(3)
    recovery = controller._fault_recovery
    second.start()
    try:
        assert second_started.wait(3)
        assert controller._fault_recovery is recovery and len(calls) == 1
        release.set()
        first.join(3)
        assert not first.is_alive()
        assert recovery.outcome == "recovery_succeeded"
        assert controller._fault_recovery is recovery and len(calls) == 1
        assert second.is_alive() and not controller._restart_consumed_event.is_set()
        controller._consume_restart_cycle()
        second.join(3)
        assert not second.is_alive() and len(calls) == 2
    finally:
        release.set()
        first.join(3)
        controller._restart_consumed_event.set()
        second.join(3)
        assert not first.is_alive() and not second.is_alive()


def test_local_completion_follows_all_original_controller_success_actions(runtime):
    observe(runtime)
    trace = []
    controller, logger = make_controller(runtime, {Roles.advantages: make_service(trace)}, trace)

    def fail_final_log(message, *args, **kwargs):
        if "restarted successfully" in message:
            raise RuntimeError("original final action failed")

    logger.info.side_effect = fail_final_log
    controller.restart_serve(Roles.advantages)
    assert isinstance(controller._restart_error, RuntimeError)
    terminals = [row for row in records(runtime) if row["event_type"].startswith("recovery_")]
    assert [row["event_type"] for row in terminals] == ["recovery_failed"]


def test_unregistered_role_records_skip_and_preserves_upstream_failure_handshake(runtime):
    observe(runtime, (Roles.reference,))
    controller, _ = make_controller(runtime, {}, [])
    controller.restart_serve(Roles.reference)
    assert not controller._restart_consumed_event.is_set()
    assert controller._restart_done_event.is_set()
    assert isinstance(controller._restart_error, RuntimeError)
    assert "No class registered" in str(controller._restart_error)
    assert controller._fault_recovery is None
    assert [event["event_type"] for event in records(runtime)] == ["fault_detected", "handling_skipped"]


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("failure", ["unknown_role", "counter"])
def test_early_recovery_error_stops_checker_and_cancels_without_reusing_previous_context(runtime, enabled, failure):
    observe(runtime, (Roles.reference if failure == "unknown_role" else Roles.advantages,))
    trace = []
    controller, _ = make_controller(runtime if enabled else None, {}, trace)
    controller._fault_recovery = object()
    checker_stop = threading.Event()
    controller._health_manager._checker = SimpleNamespace(_stop_event=checker_stop)
    controller._cancel_pending_tasks = lambda: trace.append("cancel")
    error = ValueError("counter failed")
    if failure == "counter":
        controller._health_manager.increment_restart_count = Mock(side_effect=error)

    controller.restart_serve(Roles.reference if failure == "unknown_role" else Roles.advantages)

    assert controller._restart_done_event.is_set()
    assert not controller._restart_consumed_event.is_set()
    assert checker_stop.is_set() and trace == ["cancel"]
    assert controller._fault_recovery is None
    if failure == "counter":
        assert controller._restart_error is error
    else:
        assert isinstance(controller._restart_error, RuntimeError)
    assert not any(event["event_type"].startswith("recovery_") for event in records(runtime))


@pytest.mark.parametrize("raises", [False, True])
def test_global_error_return_or_exception_is_recorded_without_false_success(runtime, raises):
    faults = observe(runtime, (Roles.actor, Roles.actor_fwd))
    controller, _ = make_controller(runtime, {Roles.actor: object(), Roles.actor_fwd: object()}, [])
    error = RuntimeError("injected global failure")

    def fail_global():
        if raises:
            raise error
        controller._restart_error = error

    controller._run_global_restart = fail_global
    controller.restart_serve(Roles.actor)
    assert controller._restart_error is error and controller._restart_done_event.is_set()
    failed = [event for event in records(runtime) if event["event_type"] == "recovery_failed"]
    assert {event["fault_id"] for event in failed} == {fault.fault_id for fault in faults}
    assert len({event["recovery_id"] for event in failed}) == 1
    assert {event["expected_evidence"] for event in failed} == {31}
    assert not any(event["event_type"] == "recovery_succeeded" for event in records(runtime))


def test_actual_global_budget_exit_terminates_all_members_once(runtime):
    faults = observe(runtime, (Roles.actor, Roles.actor_fwd))
    trace = []
    controller, _ = make_controller(
        runtime, {Roles.actor: object(), Roles.actor_fwd: object()}, trace, max_global_restart=0
    )
    controller.restart_serve(Roles.actor)
    assert trace == [("exit", 1)]
    assert isinstance(controller._restart_error, SystemExit)
    assert controller._restart_done_event.is_set()
    events = records(runtime)
    terminal = [event for event in events if event["event_type"] == "termination_requested"]
    assert {event["fault_id"] for event in terminal} == {fault.fault_id for fault in faults}
    assert {event["expected_evidence"] for event in terminal} == {31}
    assert len({event["recovery_id"] for event in terminal}) == 1
    assert not runtime._recoveries
    assert not any(event["event_type"] in ("recovery_succeeded", "recovery_failed") for event in events)


def test_successful_global_rebuild_remains_unverified_in_this_batch(runtime):
    observe(runtime, (Roles.actor,))
    controller, _ = make_controller(runtime, {Roles.actor: object()}, [])
    controller._run_global_restart = lambda: None
    controller.restart_serve(Roles.actor)
    events = records(runtime)
    assert [event["event_type"] for event in events] == ["fault_detected", "handling_started"]
    assert controller._fault_recovery.mode == "global" and controller._fault_recovery.expected == 15
    assert controller._fault_recovery.outcome is None


def test_same_fault_can_retry_in_later_actual_controller_round(runtime):
    faults = observe(runtime)
    controller, _ = make_controller(runtime, {Roles.advantages: make_service([], failure="step")}, [])
    controller.restart_serve(Roles.advantages)
    first = controller._fault_recovery
    controller._consume_restart_cycle()
    runtime.observe_snapshot(
        {
            "source_generation": 0,
            "roles": [Roles.advantages],
            "faults": [asdict(faults[0])],
            "observation_error": False,
        }
    )
    controller.serve_dict[Roles.advantages] = make_service([])
    controller.restart_serve(Roles.advantages)
    second = controller._fault_recovery
    assert first.recovery_id != second.recovery_id
    events = records(runtime)
    assert sum(event["event_type"] == "fault_detected" for event in events) == 1
    assert [
        event["event_type"] for event in events if event["event_type"] in ("recovery_failed", "recovery_succeeded")
    ] == ["recovery_failed", "recovery_succeeded"]


def test_actual_local_end_releases_missing_evidence_without_blocking_next_recovery(runtime, monkeypatch):
    observe(runtime, (Roles.actor,))
    pending_global = runtime.start_recovery(Roles.actor, "global", 15)
    runtime.bind_source()
    observe(runtime)
    evidence = runtime.evidence

    def contend_on_step(recovery, **kwargs):
        if kwargs.get("observed") == 2:
            with runtime._lock:
                evidence(recovery, **kwargs)
        else:
            evidence(recovery, **kwargs)

    monkeypatch.setattr(runtime, "evidence", contend_on_step)
    controller, _ = make_controller(runtime, {Roles.advantages: make_service([])}, [])
    controller.restart_serve(Roles.advantages)
    incomplete = controller._fault_recovery
    assert controller._restart_error is None
    assert incomplete.outcome is None and incomplete.recovery_id not in runtime._recoveries
    assert runtime._recoveries == {pending_global.recovery_id: pending_global}
    assert "OUTCOME_UNOBSERVED" in runtime.diagnostics()["unknown_reasons"]
    assert not any(row["event_type"] == "recovery_succeeded" for row in records(runtime))

    controller._consume_restart_cycle()
    monkeypatch.setattr(runtime, "evidence", evidence)
    observe(runtime)
    controller.serve_dict[Roles.advantages] = make_service([])
    controller.restart_serve(Roles.advantages)
    assert controller._fault_recovery.outcome == "recovery_succeeded"
    assert pending_global.outcome is None and runtime.generation == 1


def test_actual_global_teardown_retires_old_receiver_even_when_reinit_fails(runtime, monkeypatch):
    first = observe(runtime, (Roles.actor,))[0]
    pending = runtime.start_recovery(Roles.actor, "global", 15)
    runtime.bind_source()
    second = observe(runtime, (Roles.actor,))[0]
    trace = []
    service = SimpleNamespace(_stop_heartbeat_thread=lambda: trace.append("heartbeat_stop"))
    controller, _ = make_controller(runtime, {Roles.actor: service}, trace)
    controller.runtime_env = {}
    controller.config.use_agentic_rollout = False
    controller._autoscaler_config = None
    controller._health_manager.stop = lambda **kwargs: trace.append("health_stop")
    controller._tq_legacy_init = True
    namespace = type(controller)._run_global_restart.__globals__

    def fail_init(**kwargs):
        trace.append("ray_init")
        raise RuntimeError("injected reinitialization failure")

    namespace.update(
        ray=SimpleNamespace(shutdown=lambda: trace.append("ray_shutdown"), init=fail_init),
        serve=SimpleNamespace(
            delete=lambda role: trace.append(("delete", role)), shutdown=lambda: trace.append("serve_shutdown")
        ),
        recovery_load_path=lambda config: None,
        tq=SimpleNamespace(close=lambda: trace.append("tq_close")),
        shutdown_async_loop=lambda: trace.append("loop_shutdown"),
        time=SimpleNamespace(sleep=lambda seconds: trace.append(("sleep", seconds))),
    )
    router = ModuleType("relax.distributed.ray.rollout")
    router.stop_launched_routers = lambda: 0
    monkeypatch.setitem(sys.modules, "relax.distributed.ray.rollout", router)
    controller.restart_serve(Roles.actor)

    assert trace.index("ray_shutdown") < trace.index("ray_init")
    assert pending.outcome == "recovery_interrupted"
    assert controller._fault_recovery.outcome == "recovery_failed"
    assert controller._restart_done_event.is_set() and isinstance(controller._restart_error, RuntimeError)
    terminals = [row for row in records(runtime) if row["event_type"] in ("recovery_interrupted", "recovery_failed")]
    assert [(row["fault_id"], row["event_type"]) for row in terminals] == [
        (first.fault_id, "recovery_interrupted"),
        (second.fault_id, "recovery_failed"),
    ]
