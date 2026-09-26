# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import threading
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest


try:
    from relax.core.controller import Controller
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


def _patched_restart_controller(monkeypatch, events):
    import sys
    import types

    import relax.core.controller as module

    # The controller imports stop_launched_routers lazily; a stub module keeps
    # the test off sglang, which the real rollout module imports at load time.
    rollout_module = types.ModuleType("relax.distributed.ray.rollout")
    monkeypatch.setitem(sys.modules, "relax.distributed.ray.rollout", rollout_module)

    controller = object.__new__(Controller)
    controller.config = SimpleNamespace(
        use_agentic_rollout=False, sglang_router_ip="external", sglang_router_port=9000
    )
    controller.runtime_env = {}
    controller._global_restart_count = 0
    controller._max_global_restart = 1
    controller._health_manager = MagicMock()
    controller._metrics_service_enabled = False
    controller._autoscaler_config = None
    controller._cancel_pending_tasks = lambda: events.append("cancel")
    controller._inference_manager_handle = MagicMock()
    controller._inference_manager_handle.shutdown_all.remote.side_effect = lambda: events.append("engines")
    controller.serve_dict = {
        "rollout": SimpleNamespace(
            _gateway_name="rollout_gateway", _backend_name="rollout_backend", _stop_heartbeat_thread=lambda: None
        ),
        "actor": SimpleNamespace(_gateway_name=None, _backend_name="actor", _stop_heartbeat_thread=lambda: None),
    }
    monkeypatch.setattr(module, "recovery_load_path", lambda config: None)
    monkeypatch.setattr(module.serve, "delete", lambda name: events.append(name))
    monkeypatch.setattr(module.serve, "shutdown", lambda: events.append("serve_shutdown"))
    monkeypatch.setattr(module.serve, "start", lambda **kwargs: None)
    monkeypatch.setattr(module.ray, "get", lambda ref: ref)
    monkeypatch.setattr(module.ray, "shutdown", lambda: events.append("ray_shutdown"))
    monkeypatch.setattr(module.ray, "init", lambda **kwargs: events.append("ray_init"))
    monkeypatch.setattr(module.tq, "close", lambda: events.append("tq"))
    monkeypatch.setattr(module, "shutdown_async_loop", lambda: events.append("async_loop"))
    monkeypatch.setattr(module.time, "sleep", lambda seconds: None)
    rollout_module.stop_launched_routers = lambda: 0
    monkeypatch.setattr(Controller, "__init__", lambda *args: None)
    return controller


def test_global_restart_cleans_owned_inference_before_ray(monkeypatch):
    events = []
    controller = _patched_restart_controller(monkeypatch, events)

    controller._run_global_restart()

    assert events[:4] == ["cancel", "rollout_gateway", "rollout_backend", "actor"]
    assert events.index("engines") < events.index("dcs_coordinator") < events.index("ray_shutdown")
    assert events.index("async_loop") < events.index("ray_shutdown")
    assert controller.config.sglang_router_ip == "external"
    assert controller.config.sglang_router_port == 9000


def test_global_restart_continues_when_inference_shutdown_fails(monkeypatch):
    """A stuck engine or dead manager must not abort the recovery path."""
    events = []
    controller = _patched_restart_controller(monkeypatch, events)
    controller._inference_manager_handle.shutdown_all.remote.side_effect = RuntimeError("engine stop failed")

    controller._run_global_restart()

    assert events.index("dcs_coordinator") < events.index("ray_shutdown") < events.index("ray_init")
