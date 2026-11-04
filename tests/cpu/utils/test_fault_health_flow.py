# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Real health/source methods and Checker loops with an explicit local Ray
transport double."""

import asyncio
import enum
import threading
import time
from enum import Enum
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from relax.utils import async_utils, health_system
from relax.utils.fault_events.models import MAX_UINT, ROLES, EventDraft, FaultContext, new_id, service_role
from relax.utils.fault_events.source import FaultSource


# This is Ray's original decorated implementation, also used by existing actor
# CPU tests. Only scheduling/serialization are replaced; health logic is real.
HealthStatusImpl = health_system.HealthStatus.__ray_metadata__.modified_class


class LocalStatusRPC:
    def __init__(self, status) -> None:
        self.status = status
        self.calls = []
        self.trace = []
        self.before = {}
        self.after = {}
        for name in ("get_unhealthy_services", "get_service_health", "get_stale_services", "mark_unhealthy"):
            setattr(self, name, SimpleNamespace(remote=self._remote(name)))

    def _remote(self, name):
        def call(*args, **kwargs):
            self.calls.append((name, args, kwargs))
            self.trace.append((name, args))
            if name in self.before:
                self.before[name](*args, **kwargs)
            result = getattr(self.status, name)(*args, **kwargs)
            if name in self.after:
                self.after[name](result)
            return result

        return call


class ObserverSpy:
    def __init__(self) -> None:
        self.calls = []

    def observe_snapshot(self, payload):
        self.calls.append(("snapshot", payload))

    def observe_fault(self, payload, role=None, trace_contexts=None):
        self.calls.append(("fault", payload, role))

    def special(self, *args):
        self.calls.append(("special", *args))

    def note_unknown(self):
        self.calls.append(("unknown",))


@pytest.fixture(autouse=True)
def local_ray_transport(monkeypatch):
    monkeypatch.setattr(health_system.ray, "get", lambda result: result)


def wait_until(predicate, timeout=3) -> None:
    deadline = time.monotonic() + timeout
    while not predicate():
        assert time.monotonic() < deadline, "test barrier did not complete"
        time.sleep(0.005)


@pytest.fixture
def runtime(request):
    from relax.utils.fault_events.runtime import FaultEventRuntime

    instance = FaultEventRuntime(queue_capacity=getattr(request, "param", 1))
    yield instance
    instance.close()
    instance._consumer.join(3)
    assert not instance._consumer.is_alive()


def test_fault_source_keeps_discovery_and_first_upgrade_identity():
    source = FaultSource(0)
    source.error("actor", "heartbeat_timeout")
    discovery = source.get("actor")
    source.error("actor", "reported_error")
    assert source.get("actor") == discovery
    source.error("actor", "reported_error", fatal=True)
    upgraded = source.get("actor")
    assert upgraded["event_id"] == discovery["event_id"]
    assert upgraded["fault_id"] == discovery["fault_id"]
    assert upgraded["reason"] == "heartbeat_timeout" and upgraded["fatal"]
    assert upgraded["fatal_event_id"] not in (None, discovery["event_id"])
    for fatal in (True, False, True):
        source.error("actor", "reported_error", fatal=fatal)
        assert source.get("actor") == upgraded
    FaultContext.from_dict(upgraded)
    source.clear("actor")
    assert source.get("actor") is None
    source.error("actor", "reported_error")
    fresh = source.get("actor")
    assert fresh["fault_id"] != discovery["fault_id"] and fresh["event_id"] != discovery["event_id"]
    assert fresh["fatal_event_id"] is None


def test_fault_source_initial_fatal_has_only_discovery_identity():
    source = FaultSource(MAX_UINT)
    source.error("actor", "reported_error", fatal=True)
    fault = source.get("actor")
    assert fault["reason"] == "fatal_error" and fault["fatal"] and fault["fatal_event_id"] is None
    source.error("actor", "heartbeat_timeout", fatal=True)
    assert source.get("actor") == fault
    FaultContext.from_dict(fault)


def test_fault_source_accepts_underlying_string_without_enum_formatting():
    class Role(str, Enum):
        actor = "actor"

        def __str__(self):
            raise AssertionError("role formatting must not be invoked")

    source = FaultSource(0)
    source.error(Role.actor, "reported_error")
    fault = source.get(Role.actor)
    assert type(fault["role"]) is str and fault["role"] == "actor"
    roles = [Role.actor]
    snapshot = source.snapshot(roles)
    assert snapshot["roles"] is roles and snapshot["roles"][0] is Role.actor
    source.clear(Role.actor)
    assert source.get("actor") is None


@pytest.mark.parametrize("formatted", [False, True])
def test_service_role_rejects_non_enum_string_subclasses(formatted):
    class PlainRole(str):
        pass

    class FormattedRole(str):
        def __str__(self):
            raise AssertionError("role formatting must not be invoked")

    role = FormattedRole("actor") if formatted else PlainRole("actor")
    with pytest.raises(ValueError, match="^INVALID_ROLE: role must be a supported service role$"):
        service_role(role)


@pytest.mark.parametrize("generation", [-1, MAX_UINT + 1, True, "0", {}, 1.5])
def test_fault_source_rejects_invalid_generation(generation):
    with pytest.raises(ValueError, match="^INVALID_GENERATION: "):
        FaultSource(generation)
    status = HealthStatusImpl(fault_generation=generation)
    status.report_error("actor", "business failure")
    snapshot = status.get_unhealthy_services(include_faults=True)
    assert snapshot["roles"] == ["actor"]
    assert snapshot["source_generation"] is None and not snapshot["faults"]
    assert snapshot["observation_error"] and status._fault_source is None


@pytest.mark.parametrize(
    "args",
    [
        (None, "reported_error", False),
        ("SECRET", "reported_error", False),
        ("actor", "SECRET", False),
        ("actor", {}, False),
        ("actor", "reported_error", 1),
        ("actor", "reported_error", "true"),
        ("actor", "fatal_error", False),
    ],
)
def test_fault_source_invalid_observation_preserves_existing_identity(args):
    source = FaultSource(0)
    source.error("actor", "reported_error")
    before = source.get("actor")
    source.error(*args)
    assert source.get("actor") == before
    assert source.snapshot(["actor"])["observation_error"]


def test_fault_source_snapshot_is_bounded_and_never_restores_cleared_roles():
    source = FaultSource(0)
    roles = sorted(ROLES)
    for role in roles:
        source.error(role, "reported_error")
    assert len(source.faults) == len(source.snapshot(roles)["faults"]) == 9
    source.error("unsupported", "reported_error")
    assert len(source.faults) == 9
    original_roles = roles + ["unsupported"]
    snapshot = source.snapshot(original_roles)
    assert snapshot["roles"] is original_roles and len(snapshot["faults"]) <= 9
    assert snapshot["observation_error"]
    source.clear("actor")
    empty = source.snapshot([])
    assert empty["roles"] == [] and empty["faults"] == [] and empty["source_generation"] == 0
    assert set(empty) == {"roles", "faults", "source_generation", "observation_error"}


def test_health_status_default_shapes_and_opt_in_context_keep_original_state():
    plain = HealthStatusImpl()
    observed = HealthStatusImpl(fault_generation=0)
    for status in (plain, observed):
        status.report_error("actor", "SECRET")
        status.update_heartbeat("actor", 7)
        status.set_task_status("actor", True)
    assert plain.get_unhealthy_services() == observed.get_unhealthy_services() == ["actor"]
    expected = {"healthy", "error", "last_heartbeat", "current_step", "task_running", "fatal"}
    assert set(plain.get_service_health("actor")) == set(observed.get_service_health("actor")) == expected
    details = observed.get_service_health("actor", include_fault=True)
    assert details["error"] == "SECRET" and details["current_step"] == 7 and details["task_running"]
    assert "SECRET" not in str(details["fault"])
    assert not details["observation_error"]
    assert plain.mark_unhealthy("actor", "new error") is None
    assert observed.mark_unhealthy("actor", "new error") is None


def test_health_status_preserves_sticky_fatal_when_healthy_starts_new_fault():
    status = HealthStatusImpl(fault_generation=0)
    status.report_error("actor", "fatal", fatal=True)
    old = status.get_service_health("actor", include_fault=True)["fault"]
    status.mark_healthy("actor")
    details = status.get_service_health("actor", include_fault=True)
    assert details["healthy"] and details["fatal"] and details["fault"] is None
    assert status.get_unhealthy_services(include_faults=True)["faults"] == []
    status.report_error("actor", "ordinary", fatal=False)
    details = status.get_service_health("actor", include_fault=True)
    fresh = details["fault"]
    assert not details["healthy"] and details["fatal"]
    assert fresh["fault_id"] != old["fault_id"] and fresh["event_id"] != old["event_id"]
    assert fresh["reason"] == "fatal_error" and fresh["fatal_event_id"] is None


def test_health_status_invalid_observation_type_does_not_undo_business_update():
    status = HealthStatusImpl(fault_generation=0)
    result = status.mark_unhealthy("actor", "SECRET", return_fault=True, reason_code={"SECRET": True})
    assert result is None
    assert status.get_service_health("actor")["error"] == "SECRET"
    assert not status.get_service_health("actor")["healthy"]
    assert status.get_unhealthy_services(include_faults=True)["observation_error"]
    status.mark_healthy("actor")
    assert status.get_service_health("actor")["healthy"]


@pytest.mark.parametrize("failure", ["constructor", "clear", "snapshot"])
def test_health_status_observer_failure_keeps_business_results(monkeypatch, failure):
    def fail(*args, **kwargs):
        raise RuntimeError("SECRET")

    if failure == "constructor":
        monkeypatch.setattr("relax.utils.fault_events.source.FaultSource", fail)
    status = HealthStatusImpl(fault_generation=0)
    status.report_error("actor", "business failure")
    if failure == "clear":
        monkeypatch.setattr(status._fault_source, "clear", fail)
        status.mark_healthy("actor")
        assert status.get_service_health("actor")["healthy"]
        status.report_error("actor", "second failure")
        assert status.get_service_health("actor", include_fault=True)["fault"] is None
    elif failure == "snapshot":

        def mutate_then_fail(roles):
            roles.clear()
            fail()

        monkeypatch.setattr(status._fault_source, "snapshot", mutate_then_fail)
    snapshot = status.get_unhealthy_services(include_faults=True)
    assert snapshot["roles"] == ["actor"] and snapshot["source_generation"] == 0
    assert snapshot["faults"] == [] and snapshot["observation_error"]
    assert status.get_unhealthy_services() == ["actor"]


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("global_restart", [False, True])
def test_real_checker_preserves_unhealthy_callback_then_stale_order(enabled, global_restart):
    status = HealthStatusImpl(fault_generation=0 if enabled else None)
    status.report_error("actor", "failure")
    status.mark_healthy("rollout")
    status.set_task_status("rollout", True)
    status.state["rollout"].last_heartbeat = 0
    rpc = LocalStatusRPC(status)
    observer = ObserverSpy() if enabled else None
    actions = []

    def recover(role):
        rpc.trace.append(("callback", (role,)))
        actions.append(role)
        status.update_heartbeat("rollout")
        if global_restart:
            checker._stop_event.set()

    checker = health_system.HealthChecker(rpc, recover, check_interval=0, fault_events=observer)
    rpc.after["get_stale_services"] = lambda result: checker._stop_event.set()
    asyncio.run(checker._check_loop())
    assert actions == ["actor"]
    assert [name for name, _, _ in rpc.calls] == ["get_unhealthy_services", "get_service_health"] + (
        [] if global_restart else ["get_stale_services"]
    )
    assert rpc.calls[0][2] == ({"include_faults": True} if enabled else {})
    assert rpc.calls[1][2] == ({"include_fault": True} if enabled else {})
    assert rpc.trace == [
        ("get_unhealthy_services", ()),
        ("get_service_health", ("actor",)),
        ("callback", ("actor",)),
    ] + ([] if global_restart else [("get_stale_services", ())])
    if enabled:
        assert observer.calls[0][0] == "snapshot" and observer.calls[1][0] == "fault"


def test_real_checker_observes_all_confirmed_faults_before_first_callback():
    status = HealthStatusImpl(fault_generation=0)
    for role in ("actor", "critic"):
        status.report_error(role, "failure")
    rpc, observer = LocalStatusRPC(status), ObserverSpy()
    actions = []

    def recover(role):
        assert [item["role"] for item in observer.calls[0][1]["faults"]] == ["actor", "critic"]
        rpc.trace.append(("callback", (role,)))
        actions.append(role)

    checker = health_system.HealthChecker(rpc, recover, check_interval=0, fault_events=observer)
    rpc.after["get_stale_services"] = lambda result: checker._stop_event.set()
    asyncio.run(checker._check_loop())
    assert actions == ["actor", "critic"]
    assert [name for name, _, _ in rpc.calls] == [
        "get_unhealthy_services",
        "get_service_health",
        "get_service_health",
        "get_stale_services",
    ]
    assert rpc.trace == [
        ("get_unhealthy_services", ()),
        ("get_service_health", ("actor",)),
        ("callback", ("actor",)),
        ("get_service_health", ("critic",)),
        ("callback", ("critic",)),
        ("get_stale_services", ()),
    ]


@pytest.mark.parametrize("enabled", [False, True])
def test_real_checker_stale_identity_returns_in_original_mark_rpc(enabled):
    status = HealthStatusImpl(fault_generation=0 if enabled else None)
    status.mark_healthy("rollout")
    status.set_task_status("rollout", True)
    status.state["rollout"].last_heartbeat = 0
    rpc, observer = LocalStatusRPC(status), ObserverSpy() if enabled else None
    actions = []

    def recover(role):
        actions.append(role)
        checker._stop_event.set()

    checker = health_system.HealthChecker(rpc, recover, check_interval=0, fault_events=observer)
    asyncio.run(checker._check_loop())
    assert actions == ["rollout"]
    assert [name for name, _, _ in rpc.calls] == ["get_unhealthy_services", "get_stale_services", "mark_unhealthy"]
    assert rpc.calls[-1][2] == (
        {"return_fault": True, "reason_code": "heartbeat_timeout", "include_trace_context": True} if enabled else {}
    )
    if enabled:
        assert observer.calls[-1][0] == "fault" and observer.calls[-1][1]["reason"] == "heartbeat_timeout"
        assert observer.calls[-1][2] == "rollout"


@pytest.mark.parametrize("enabled", [False, True])
def test_real_checker_runtime_observation_failures_do_not_suppress_callback(monkeypatch, caplog, enabled, runtime):
    generation = runtime.bind_source()[0] if enabled else None
    status = HealthStatusImpl(fault_generation=generation)
    status.report_error("actor", "failure")
    rpc = LocalStatusRPC(status)

    def fail(*args, **kwargs):
        raise RuntimeError("SECRET")

    monkeypatch.setattr(runtime, "_observe_snapshot", fail)
    monkeypatch.setattr(runtime, "_observe_fault", fail)
    observer = runtime if enabled else None
    actions = []

    def recover(role):
        actions.append(role)
        checker._stop_event.set()

    checker = health_system.HealthChecker(rpc, recover, check_interval=0, fault_events=observer)
    asyncio.run(checker._check_loop())
    assert actions == ["actor"]
    assert [name for name, _, _ in rpc.calls] == ["get_unhealthy_services", "get_service_health"]
    assert runtime.diagnostics()["coverage_unknown"] is enabled
    assert "SECRET" not in caplog.text


@pytest.mark.parametrize("enabled", [False, True])
def test_real_checker_fatal_preserves_callback_and_exit_despite_runtime_failure(monkeypatch, caplog, enabled, runtime):
    generation = runtime.bind_source()[0] if enabled else None
    status = HealthStatusImpl(fault_generation=generation)
    status.report_error("actor", "fatal business error", fatal=True)
    rpc, observer = LocalStatusRPC(status), runtime if enabled else None
    actions = []

    special = Mock(side_effect=RuntimeError("SECRET"))
    monkeypatch.setattr(runtime, "_special", special)

    def on_fatal(role, error):
        actions.append((role, error))
        raise RuntimeError("callback failure")

    def exit_now(code):
        raise SystemExit(code)

    monkeypatch.setattr(health_system.os, "_exit", exit_now)
    checker = health_system.HealthChecker(
        rpc, lambda role: pytest.fail("fatal entered restart"), on_fatal=on_fatal, fault_events=observer
    )
    with pytest.raises(SystemExit) as exc:
        asyncio.run(checker._check_loop())
    assert exc.value.code == 1 and actions == [("actor", "fatal business error")]
    assert [name for name, _, _ in rpc.calls] == ["get_unhealthy_services", "get_service_health"]
    if enabled:
        special.assert_called_once()
        assert special.call_args.args[:3] == ("actor", "termination_requested", "fatal_error")
        assert runtime.diagnostics()["coverage_unknown"]
    else:
        special.assert_not_called()
    assert "SECRET" not in caplog.text


@pytest.mark.parametrize("binding", ["disabled", "valid", "raises", "invalid"])
def test_health_manager_observer_binding_keeps_health_creation_available(monkeypatch, binding, runtime):
    created = []

    class ActorClass:
        @classmethod
        def options(cls, **kwargs):
            return cls

        @staticmethod
        def remote(*args, **kwargs):
            created.append((args, kwargs))
            return HealthStatusImpl(**kwargs)

    monkeypatch.setattr(health_system, "HealthStatus", ActorClass)
    bind = Mock(wraps=runtime.bind_source)
    if binding == "raises":
        monkeypatch.setattr(runtime, "_bind_source", Mock(side_effect=RuntimeError("SECRET")))
    elif binding == "invalid":
        # The actor remains the validation boundary even for a malformed peer.
        bind = Mock(return_value=(True, None))
    monkeypatch.setattr(runtime, "bind_source", bind)
    observer = None if binding == "disabled" else runtime
    manager = health_system.HealthManager(fault_events=observer)
    assert manager.status is not None
    generation = {"valid": 0, "invalid": True}.get(binding)
    assert created == [((), {"fault_generation": generation})]
    assert bind.call_count == int(binding != "disabled")
    assert manager.status._fault_observation_error is (binding == "invalid")
    if binding == "raises":
        assert runtime.diagnostics()["coverage_unknown"]
    monkeypatch.setattr(health_system.HealthChecker, "start", lambda self: None)
    manager.start(lambda role: None)
    assert manager._checker.health_status is manager.status
    assert manager._checker.fault_events is observer


def test_real_health_thread_with_observer_can_stop_shared_loop_and_itself(runtime):
    generation, _ = runtime.bind_source()
    status = HealthStatusImpl(fault_generation=generation)
    status.report_error("actor", "failure")
    rpc = LocalStatusRPC(status)
    shared = async_utils.get_async_loop()
    done = threading.Event()
    seen = []

    def recover(role):
        seen.append((role, threading.current_thread()))
        async_utils.shutdown_async_loop()
        checker.stop()
        done.set()

    checker = health_system.HealthChecker(rpc, recover, check_interval=0, fault_events=runtime)
    try:
        checker.start()
        assert done.wait(3)
        thread = checker._thread
        thread.join(3)
        assert not thread.is_alive() and not shared._thread.is_alive()
        assert seen == [("actor", thread)] and thread is not shared._thread
        assert [name for name, _, _ in rpc.calls] == ["get_unhealthy_services", "get_service_health"]
    finally:
        checker.stop()
        async_utils.shutdown_async_loop()


def test_real_checker_runtime_retries_upgrade_after_healthy_without_extra_rpc_or_callback(runtime, monkeypatch):
    generation, _ = runtime.bind_source()
    status = HealthStatusImpl(fault_generation=generation)
    status.report_error("actor", "failure")
    runtime.observe_snapshot(status.get_unhealthy_services(include_faults=True))
    discovery = status.get_service_health("actor", include_fault=True)["fault"]
    wait_until(lambda: runtime.query()["events"])
    blocker = EventDraft(runtime.run_id, new_id(), new_id(), "critic", generation, "fault_detected", "reported_error")
    filler = EventDraft(runtime.run_id, new_id(), new_id(), "rollout", generation, "fault_detected", "reported_error")
    entered, release = threading.Event(), threading.Event()
    append = runtime.store._append_committed

    def blocked_append(draft):
        if draft.event_id == blocker.event_id:
            entered.set()
            assert release.wait(3)
        return append(draft)

    monkeypatch.setattr(runtime.store, "_append_committed", blocked_append)
    try:
        assert runtime.record(blocker) and entered.wait(3)
        assert runtime.record(filler)
        status.report_error("actor", "fatal escalation", fatal=True)
        upgraded = status.get_service_health("actor", include_fault=True)["fault"]
        # Seed the previous query's rejected U using actual source/runtime methods.
        runtime.observe_snapshot(status.get_unhealthy_services(include_faults=True))
        assert runtime.diagnostics()["admission_rejections"] == 1
        status.mark_healthy("actor")
        assert status.get_service_health("actor", include_fault=True)["fault"] is None
        release.set()
        wait_until(lambda: runtime.diagnostics()["queued"] == runtime.diagnostics()["inflight"] == 0)
        rpc = LocalStatusRPC(status)
        checker = health_system.HealthChecker(
            rpc,
            lambda role: pytest.fail("pending publication became recovery candidate"),
            check_interval=0,
            fault_events=runtime,
        )
        rpc.after["get_stale_services"] = lambda result: checker._stop_event.set()
        asyncio.run(checker._check_loop())
        wait_until(lambda: any(event["event_id"] == upgraded["fatal_event_id"] for event in runtime.query()["events"]))
        events = runtime.query()["events"]
        assert sum(event["event_id"] == discovery["event_id"] for event in events) == 1
        assert sum(event["event_id"] == upgraded["fatal_event_id"] for event in events) == 1
        assert runtime.diagnostics()["known_lost_events"] == 0
        assert runtime.diagnostics()["admission_rejections"] == 1
        assert runtime._snapshot == ()
        assert [name for name, _, _ in rpc.calls] == ["get_unhealthy_services", "get_stale_services"]
    finally:
        release.set()


def test_real_checker_runtime_ignores_old_generation_empty_snapshot(runtime):
    old_generation, _ = runtime.bind_source()
    old = HealthStatusImpl(fault_generation=old_generation)
    generation, _ = runtime.bind_source()
    current = HealthStatusImpl(fault_generation=generation)
    current.report_error("critic", "failure")
    runtime.observe_snapshot(current.get_unhealthy_services(include_faults=True))
    before = runtime._snapshot
    assert len(before) == 1
    rpc = LocalStatusRPC(old)
    checker = health_system.HealthChecker(
        rpc, lambda role: pytest.fail("unexpected callback"), check_interval=0, fault_events=runtime
    )
    rpc.after["get_stale_services"] = lambda result: checker._stop_event.set()
    asyncio.run(checker._check_loop())
    assert runtime._snapshot == before
    assert [name for name, _, _ in rpc.calls] == ["get_unhealthy_services", "get_stale_services"]


@pytest.mark.parametrize("corruption", ["missing", "changed_id"])
def test_real_checker_runtime_invalid_role_context_cannot_reuse_snapshot_candidate(runtime, monkeypatch, corruption):
    generation, _ = runtime.bind_source()
    status = HealthStatusImpl(fault_generation=generation)
    status.report_error("actor", "failure")
    original = status.get_service_health("actor", include_fault=True)["fault"]
    rpc = LocalStatusRPC(status)

    def corrupt_after_snapshot(result):
        invalid = None if corruption == "missing" else {**original, "event_id": new_id()}
        monkeypatch.setattr(status._fault_source, "get", lambda role: invalid)

    rpc.after["get_unhealthy_services"] = corrupt_after_snapshot
    actions = []

    def recover(role):
        actions.append(role)
        assert runtime._snapshot == ()
        checker._stop_event.set()

    checker = health_system.HealthChecker(rpc, recover, check_interval=0, fault_events=runtime)
    asyncio.run(checker._check_loop())
    assert actions == ["actor"] and runtime.diagnostics()["coverage_unknown"]


def test_real_checker_runtime_preserves_enum_callback_role_and_records_plain_role(runtime):
    if hasattr(enum, "StrEnum"):
        Role = enum.StrEnum("Role", {"actor": "actor"})
    else:
        # Same fallback as registry.py on Python 3.10.
        class Role(str, Enum):
            actor = "actor"

            def __str__(self):
                return self.value

    generation, _ = runtime.bind_source()
    status = HealthStatusImpl(fault_generation=generation)
    status.report_error(Role.actor, "failure")
    rpc = LocalStatusRPC(status)
    actions = []

    def recover(role):
        actions.append(role)
        checker._stop_event.set()

    checker = health_system.HealthChecker(rpc, recover, check_interval=0, fault_events=runtime)
    asyncio.run(checker._check_loop())
    wait_until(lambda: runtime.query()["events"])
    event = runtime.query()["events"][0]
    assert actions[0] is Role.actor
    assert event["role"] == "actor" and type(event["role"]) is str
    assert not runtime.diagnostics()["coverage_unknown"]


@pytest.mark.parametrize("runtime", [512], indirect=True)
@pytest.mark.parametrize("escalate", [False, True])
def test_real_checker_runtime_fatal_records_only_discovery_upgrade_and_termination(runtime, monkeypatch, escalate):
    generation, _ = runtime.bind_source()
    status = HealthStatusImpl(fault_generation=generation)
    status.report_error("actor", "business failure", fatal=not escalate)
    original = status.get_service_health("actor", include_fault=True)["fault"]
    rpc = LocalStatusRPC(status)
    if escalate:
        rpc.before["get_service_health"] = lambda *args, **kwargs: status.report_error("actor", "fatal", fatal=True)

    def exit_now(code):
        raise SystemExit(code)

    monkeypatch.setattr(health_system.os, "_exit", exit_now)
    fatal_roles = []
    checker = health_system.HealthChecker(
        rpc,
        lambda role: pytest.fail("fatal entered recovery"),
        on_fatal=lambda role, error: fatal_roles.append(role),
        fault_events=runtime,
    )
    with pytest.raises(SystemExit) as exc:
        asyncio.run(checker._check_loop())
    assert exc.value.code == 1 and fatal_roles == ["actor"]
    wait_until(lambda: len(runtime.query()["events"]) == (3 if escalate else 2))
    events = runtime.query()["events"]
    assert [event["event_type"] for event in events] == ["fault_detected"] + (
        ["fatal_escalated"] if escalate else []
    ) + ["termination_requested"]
    assert events[0]["event_id"] == original["event_id"]
    assert events[0]["reason_code"] == ("reported_error" if escalate else "fatal_error")
    if escalate:
        assert (
            events[1]["event_id"] == status.get_service_health("actor", include_fault=True)["fault"]["fatal_event_id"]
        )
    assert [name for name, _, _ in rpc.calls] == ["get_unhealthy_services", "get_service_health"]


@pytest.mark.parametrize(
    "field,value,constraint",
    [
        ("role", "PRIVATE_ROLE", "role must be a supported service role"),
        ("source_generation", -1, "source_generation must be an unsigned 64-bit int"),
        ("reason", "PRIVATE_REASON", "reason must be a fault_detected reason code"),
        ("fatal_event_id", new_id(), "a nonfatal fault has no fatal_event_id"),
    ],
)
def test_fault_context_error_names_constraint_without_value(field, value, constraint):
    source = FaultSource(0)
    source.error("actor", "reported_error")
    payload = {**source.get("actor"), field: value}
    with pytest.raises(ValueError) as raised:
        FaultContext.from_dict(payload)
    assert raised.value.args == (f"INVALID_FAULT: {constraint}",)
