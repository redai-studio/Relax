# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import asyncio
import sys
import threading
from types import SimpleNamespace
from unittest.mock import Mock

import pytest


try:
    import relax.core.controller as controller_module

    Controller = controller_module.Controller
except (ImportError, AssertionError) as exc:
    pytest.skip(f"relax.core.controller requires the full runtime environment: {exc}", allow_module_level=True)


def test_global_restart_signals_failure_from_any_phase():
    controller = object.__new__(Controller)
    controller._restart_done_event = threading.Event()
    controller._restart_error = None

    def fail_before_reinit():
        raise RuntimeError("ray init failed")

    controller._run_global_restart = fail_before_reinit

    controller._global_restart()

    assert controller._restart_done_event.is_set()
    assert isinstance(controller._restart_error, RuntimeError)
    assert str(controller._restart_error) == "ray init failed"


def test_global_restart_signals_success_on_same_event():
    controller = object.__new__(Controller)
    restart_done_event = threading.Event()
    controller._restart_done_event = restart_done_event
    controller._restart_error = None
    controller._run_global_restart = lambda: None

    controller._global_restart()

    assert controller._restart_done_event is restart_done_event
    assert restart_done_event.is_set()
    assert controller._restart_error is None


def test_restart_cycle_ack_is_published_after_old_state_is_cleared():
    controller = object.__new__(Controller)
    controller._restart_done_event = threading.Event()
    controller._restart_done_event.set()
    controller._restart_mode = "local"
    controller._restart_error = None
    controller._restarting = True

    class StartNextCycleOnAck:
        def set(self):
            assert controller._restarting is False
            assert controller._restart_mode is None
            controller._restarting = True
            controller._restart_mode = "global"

    controller._restart_consumed_event = StartNextCycleOnAck()

    restart_mode, restart_error = controller._consume_restart_cycle()

    assert restart_mode == "local"
    assert restart_error is None
    assert controller._restarting is True
    assert controller._restart_mode == "global"


@pytest.mark.parametrize(
    "driver_stopped, actor_fails, manager_pending, initial_router_ip, expected_router_ip",
    [
        (1, False, False, "127.0.0.1", None),
        (0, False, False, None, None),
        (1, True, False, "127.0.0.1", None),
        (1, False, True, "127.0.0.1", None),
        (0, False, False, "external-router.example", "external-router.example"),
    ],
    ids=[
        "agentic-driver",
        "manager",
        "manager-unavailable",
        "manager-query-pending",
        "external-primary-local-auxiliary",
    ],
)
def test_global_restart_cleans_both_router_owners(
    monkeypatch, driver_stopped, actor_fails, manager_pending, initial_router_ip, expected_router_ip
):
    events = []
    dispose_ref = object()
    dispose_remote = Mock(side_effect=lambda: events.append("dispose manager") or dispose_ref)
    manager = SimpleNamespace(dispose=SimpleNamespace(remote=dispose_remote))

    async def get_rollout_manager():
        if manager_pending:
            pending = asyncio.get_running_loop().create_future()
            # Release on regressions so a missing timeout fails instead of hanging pytest.
            watchdog = asyncio.get_running_loop().call_later(1, pending.set_result, None)
            try:
                await pending
            except asyncio.CancelledError:
                events.append("manager query cancelled")
                raise
            finally:
                watchdog.cancel()
        return manager

    service = SimpleNamespace(
        _stop_heartbeat_thread=Mock(),
        get_rollout_manager=get_rollout_manager,
    )
    config = SimpleNamespace(
        sglang_router_ip=initial_router_ip, sglang_router_port=3000 if initial_router_ip else None
    )
    controller = object.__new__(Controller)
    controller.config = config
    controller.runtime_env = {}
    controller._global_restart_count = 0
    controller._max_global_restart = 1
    controller._health_manager = SimpleNamespace(_checker=None, stop=Mock())
    controller._cancel_pending_tasks = Mock()
    controller.serve_dict = {controller_module.ROLES.rollout: service}
    controller._metrics_service_enabled = False
    controller._autoscaler_config = None
    controller._shutdown_agentic_rollout_services = Mock()
    controller._restart_error = None
    controller._restart_done_event = threading.Event()

    if manager_pending:
        wait_for = asyncio.wait_for

        async def short_wait_for(awaitable, *, timeout):
            assert timeout == 5.0
            return await wait_for(awaitable, timeout=0.01)

        monkeypatch.setattr(controller_module.asyncio, "wait_for", short_wait_for)

    def get(ref, *, timeout):
        assert ref is dispose_ref and timeout == 35.0
        if actor_fails:
            raise RuntimeError("Manager unavailable")
        return None

    stop_driver = Mock(side_effect=lambda: events.append("stop driver router") or driver_stopped)
    monkeypatch.setitem(
        sys.modules, "relax.distributed.ray.rollout", SimpleNamespace(stop_launched_routers=stop_driver)
    )

    monkeypatch.setattr(controller_module, "recovery_load_path", Mock())
    monkeypatch.setattr(controller_module.ray, "get", get)
    monkeypatch.setattr(controller_module.ray, "shutdown", lambda: events.append("ray shutdown"))
    monkeypatch.setattr(controller_module.ray, "init", Mock())
    monkeypatch.setattr(controller_module.serve, "delete", lambda role: events.append(f"delete {role}"))
    monkeypatch.setattr(controller_module.serve, "shutdown", Mock())
    monkeypatch.setattr(controller_module.serve, "start", Mock())
    monkeypatch.setattr(controller_module.tq, "close", Mock())
    monkeypatch.setattr(controller_module.time, "sleep", Mock())

    def reinit(self, config, runtime_env):
        events.append("reinit")
        assert config.sglang_router_ip == expected_router_ip
        assert config.sglang_router_port == (3000 if expected_router_ip else None)

    monkeypatch.setattr(Controller, "__init__", reinit)

    controller._global_restart()

    if manager_pending:
        dispose_remote.assert_not_called()
        assert events.index("manager query cancelled") < events.index(f"delete {controller_module.ROLES.rollout}")
    else:
        dispose_remote.assert_called_once_with()
        assert events.index("dispose manager") < events.index(f"delete {controller_module.ROLES.rollout}")
    stop_driver.assert_called_once_with()
    assert events.index("stop driver router") < events.index("ray shutdown") < events.index("reinit")
    assert controller._restart_error is None
    assert controller._restart_done_event.is_set()
