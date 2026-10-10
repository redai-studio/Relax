import threading
from types import SimpleNamespace
from unittest.mock import Mock

import relax.core.controller as controller_module
from relax.core.controller import Controller


def make_store():
    return SimpleNamespace(
        append=SimpleNamespace(remote=Mock()),
    )


def recorded_events(store):
    return [call.args[0] for call in store.append.remote.call_args_list]


def make_controller(role=None, service=None):
    controller = Controller.__new__(Controller)
    controller.config = SimpleNamespace()
    controller._failure_event_store = make_store()
    controller._restart_consumed_event = threading.Event()
    controller._restart_consumed_event.set()
    controller._restart_done_event = threading.Event()
    controller._restart_error = None
    controller._restart_mode = None
    controller._restarting = False
    controller.serve_dict = {}

    controller._health_manager = SimpleNamespace(
        increment_restart_count=lambda role: 1,
        mark_healthy=lambda role: None,
    )

    if role is not None and service is not None:
        controller.serve_dict[role] = service

    return controller


def configure_algo(monkeypatch, role):
    monkeypatch.setattr(
        controller_module,
        "resolve_sft_algo_key",
        lambda config: "test",
    )
    monkeypatch.setattr(
        controller_module,
        "register_extra_roles",
        lambda config, algo: None,
    )
    monkeypatch.setattr(
        controller_module,
        "ALGOS",
        {"test": {role: object}},
    )


def test_detected_event_is_recorded_before_restart():
    controller = make_controller()
    controller.restart_serve = Mock()

    controller._on_service_unhealthy(
        "actor",
        "fault-001",
        42,
        "heartbeat_timeout",
    )

    events = recorded_events(controller._failure_event_store)

    assert len(events) == 1
    assert events[0].fault_id == "fault-001"
    assert events[0].phase == "detected"
    assert events[0].reason == "heartbeat_timeout"
    assert events[0].step == 42

    controller.restart_serve.assert_called_once_with(
        "actor",
        fault_id="fault-001",
        step=42,
        reason="heartbeat_timeout",
    )


def test_local_restart_success_events(monkeypatch):
    role = "test_worker"

    class Service:
        def restart(self):
            pass

    configure_algo(monkeypatch, role)
    controller = make_controller(role, Service())

    controller.restart_serve(
        role,
        fault_id="fault-local",
        step=10,
        reason="reported_error",
    )

    events = recorded_events(controller._failure_event_store)

    assert [event.phase for event in events] == [
        "handling_started",
        "recovery_succeeded",
    ]
    assert all(event.action == "local_restart" for event in events)


def test_local_restart_failure_events(monkeypatch):
    role = "test_worker"

    class Service:
        def restart(self):
            raise RuntimeError("restart failed")

    configure_algo(monkeypatch, role)
    controller = make_controller(role, Service())

    controller.restart_serve(
        role,
        fault_id="fault-local-fail",
        step=11,
        reason="reported_error",
    )

    events = recorded_events(controller._failure_event_store)

    assert [event.phase for event in events] == [
        "handling_started",
        "recovery_failed",
    ]
    assert events[-1].reason == "restart_exception"


def test_global_restart_success_events(monkeypatch):
    role = controller_module.ROLES.actor

    configure_algo(monkeypatch, role)
    controller = make_controller()

    controller._global_restart = lambda: None

    controller.restart_serve(
        role,
        fault_id="fault-global",
        step=12,
        reason="reported_error",
    )

    events = recorded_events(controller._failure_event_store)

    assert [event.phase for event in events] == [
        "handling_started",
        "recovery_succeeded",
    ]
    assert all(event.action == "global_restart" for event in events)


def test_global_restart_failure_events(monkeypatch):
    role = controller_module.ROLES.actor

    configure_algo(monkeypatch, role)
    controller = make_controller()

    def fail_restart():
        controller._restart_error = RuntimeError("restart failed")

    controller._global_restart = fail_restart

    controller.restart_serve(
        role,
        fault_id="fault-global-fail",
        step=13,
        reason="reported_error",
    )

    events = recorded_events(controller._failure_event_store)

    assert [event.phase for event in events] == [
        "handling_started",
        "recovery_failed",
    ]
    assert events[-1].reason == "restart_exception"


def test_fatal_failure_records_detection_and_termination():
    controller = make_controller()
    controller._report_error_to_metrics_service = Mock()

    controller._on_service_fatal(
        "actor",
        "fatal error",
        "fault-fatal",
        14,
    )

    events = recorded_events(controller._failure_event_store)

    assert [event.phase for event in events] == [
        "detected",
        "handling_started",
    ]
    assert events[0].reason == "fatal_error"
    assert events[1].reason == "fatal_error"
    assert events[1].action == "terminate"
