# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import asyncio
import threading
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import pytest


try:
    from relax.core import controller as controller_module

    Controller = controller_module.Controller
except (ImportError, AssertionError) as exc:
    pytest.skip(f"relax.core.controller requires the full runtime environment: {exc}", allow_module_level=True)


def _controller_shell():
    controller = object.__new__(Controller)
    with patch.object(
        controller_module, "require_control_plane_resource", side_effect=RuntimeError("stop before runtime startup")
    ):
        with pytest.raises(RuntimeError, match="stop before runtime startup"):
            controller.__init__(SimpleNamespace(), {})
    controller._restart_done_event = threading.Event()
    controller._restart_lock = threading.Lock()
    controller._training_wait_done = threading.Event()
    controller._training_wait_done.set()
    controller._local_recovery_thread = None
    controller._local_recovery_role = None
    controller._inference_recovery = None
    controller._inference_recovery_timeout = 2.0
    controller._restart_consumed_event = threading.Event()
    controller._restart_consumed_event.set()
    controller._restart_error = None
    controller._restart_mode = None
    controller._restarting = False
    controller._pending_task_refs = []
    controller._pending_task_refs_lock = threading.Lock()
    controller._global_restart_count = 0
    controller._max_global_restart = 3
    controller._health_manager = SimpleNamespace(_checker=None)
    return controller


def _patch_global_restart_cleanup(monkeypatch):
    monkeypatch.setattr(controller_module, "recovery_load_path", Mock())
    monkeypatch.setattr(controller_module.serve, "shutdown", Mock())
    monkeypatch.setattr(controller_module.ray, "shutdown", Mock())
    monkeypatch.setattr(controller_module.ray, "init", Mock())
    monkeypatch.setattr(controller_module.tq, "close", Mock())
    monkeypatch.setattr(controller_module, "shutdown_async_loop", Mock())


def test_global_restart_signals_failure_from_any_phase():
    controller = _controller_shell()
    error = RuntimeError("ray init failed")
    controller._restart_serve = Mock(side_effect=error)

    controller.restart_serve("actor")

    assert controller._restart_done_event.is_set()
    assert controller._restart_error is error


def test_global_restart_signals_success_on_same_event():
    controller = _controller_shell()
    event = threading.Event()
    controller._restart_done_event = event
    controller._restart_serve = Mock()

    controller.restart_serve("actor")

    assert controller._restart_done_event is event
    assert event.is_set()
    assert controller._restart_error is None


def test_default_global_restart_reinitializes_after_legacy_manager_death(monkeypatch):
    from relax.core import service as service_module

    controller = _controller_shell()
    controller.config, controller.runtime_env = SimpleNamespace(), {}
    controller._health_manager = SimpleNamespace(_checker=None, stop=Mock())
    controller._metrics_service_enabled, controller._autoscaler_config = False, None
    controller._shutdown_agentic_rollout_services = Mock()
    legacy = object.__new__(service_module.Service)
    legacy.role, legacy.pgs = "rollout", None
    legacy._is_shared_pgs = False
    legacy.handle, legacy._stop_heartbeat_thread = Mock(), Mock()
    controller.serve_dict = {"rollout": legacy}
    _patch_global_restart_cleanup(monkeypatch)
    monkeypatch.setattr(controller_module.serve, "delete", Mock())
    monkeypatch.setattr(controller_module.serve, "start", Mock())
    monkeypatch.setattr(controller_module.serve, "status", Mock(side_effect=RuntimeError("Serve unavailable")))
    monkeypatch.setattr(controller_module.time, "sleep", Mock())
    monkeypatch.setattr("relax.distributed.ray.rollout.stop_launched_routers", Mock(return_value=0))
    initialize = Mock()
    monkeypatch.setattr(Controller, "__init__", initialize)

    controller._run_global_restart()

    controller_module.ray.shutdown.assert_called_once_with()
    controller_module.ray.init.assert_called_once_with(runtime_env=controller.runtime_env)
    initialize.assert_called_once_with(controller.config, controller.runtime_env)


def test_global_restart_count_limit_remains_terminal(monkeypatch):
    controller = _controller_shell()
    controller._global_restart_count = 0
    controller._max_global_restart = 0
    controller._health_manager = SimpleNamespace(_checker=None)
    controller._report_error_to_metrics_service = Mock()
    exit_process = Mock(side_effect=SystemExit(1))
    monkeypatch.setattr(controller_module.os, "_exit", exit_process)

    with pytest.raises(SystemExit) as raised:
        controller._run_global_restart()

    assert raised.value.code == 1
    assert controller._global_restart_count == 1
    exit_process.assert_called_once_with(1)
    controller._report_error_to_metrics_service.assert_called_once()


def test_cancel_pending_tasks_cancels_owned_refs_once():
    controller = _controller_shell()
    ref = Mock()
    controller._pending_task_refs = [ref]

    controller._cancel_pending_tasks()

    assert controller._pending_task_refs == []
    ref.cancel.assert_called_once_with()


def _patch_restart_entry(monkeypatch, controller):
    monkeypatch.setattr(controller_module, "resolve_sft_algo_key", lambda config: "grpo")
    monkeypatch.setattr(controller_module, "ALGOS", {"grpo": {}})
    monkeypatch.setattr(controller_module, "register_extra_roles", Mock())
    monkeypatch.setattr(
        "relax.core.registry.resolve_algorithm",
        lambda config: {"actor": object(), "rollout": object(), "sft": object()},
    )
    controller._health_manager = SimpleNamespace(increment_restart_count=Mock(return_value=1))


@pytest.mark.parametrize("role", ["rollout", "actor_fwd"])
def test_unhealthy_inference_without_local_owner_keeps_original_restart(role):
    controller = _controller_shell()
    controller._inference_recovery = object()
    controller.restart_serve = Mock()

    controller._on_service_unhealthy(role)

    controller.restart_serve.assert_called_once_with(role, expected_state=controller._inference_recovery)


def test_unhealthy_inference_defers_only_same_live_local_worker():
    controller = _controller_shell()
    controller.restart_serve = Mock()
    release = threading.Event()
    worker = threading.Thread(target=release.wait)
    controller._local_recovery_thread, controller._local_recovery_role = worker, "rollout"
    worker.start()
    try:
        controller._on_service_unhealthy("rollout")
        controller.restart_serve.assert_not_called()
        controller._on_service_unhealthy("actor")
        controller.restart_serve.assert_called_once_with("actor", expected_state=None)
    finally:
        release.set()
        worker.join(timeout=2)
    controller.restart_serve.reset_mock()
    controller._on_service_unhealthy("rollout")
    controller.restart_serve.assert_called_once_with("rollout", expected_state=None)


def test_restart_requests_claim_only_one_cycle(monkeypatch):
    controller = _controller_shell()
    _patch_restart_entry(monkeypatch, controller)
    entered, release = threading.Event(), threading.Event()

    def restart_once(role, worker):
        entered.set()
        assert release.wait(2)
        controller._restart_done_event.set()

    controller._inference_recovery = object()
    controller._restart_serve = Mock(side_effect=restart_once)
    first = threading.Thread(target=controller.restart_serve, args=("actor",))
    first.start()
    try:
        assert entered.wait(2)
        controller.restart_serve("rollout", expected_state=controller._inference_recovery)
        controller._restart_serve.assert_called_once_with("actor", None)
        assert not controller._restart_done_event.is_set()
    finally:
        release.set()
        first.join(timeout=2)
    assert not first.is_alive()
    assert controller._restart_done_event.is_set()


def test_restart_rejects_previous_recovery_state(monkeypatch):
    controller = _controller_shell()
    _patch_restart_entry(monkeypatch, controller)
    controller._inference_recovery = object()
    controller._restart_serve = Mock()

    controller.restart_serve("rollout", expected_state=object())

    controller._restart_serve.assert_not_called()
    assert not controller._restarting
    assert controller._restart_consumed_event.is_set()


def test_global_takeover_waits_for_actual_worker_exit(monkeypatch):
    controller = _controller_shell()
    _patch_restart_entry(monkeypatch, controller)
    release, joining, exited = threading.Event(), threading.Event(), threading.Event()

    def old_work():
        assert release.wait(2)
        exited.set()

    worker = threading.Thread(target=old_work)
    actual_join = worker.join

    def observe_join(*args, **kwargs):
        joining.set()
        return actual_join(*args, **kwargs)

    monkeypatch.setattr(worker, "join", observe_join)
    controller._local_recovery_thread, controller._local_recovery_role = worker, "rollout"

    def teardown():
        assert exited.is_set() and not worker.is_alive()

    controller._run_global_restart = Mock(side_effect=teardown)
    worker.start()
    restart = threading.Thread(target=controller.restart_serve, args=("actor",))
    restart.start()
    try:
        assert joining.wait(2)
        controller._run_global_restart.assert_not_called()
    finally:
        release.set()
        actual_join(timeout=2)
        restart.join(timeout=2)
    assert not restart.is_alive()
    assert controller._restart_error is None
    controller._run_global_restart.assert_called_once_with()


def test_global_takeover_timeout_terminates_before_reinitialization(monkeypatch):
    controller = _controller_shell()
    _patch_restart_entry(monkeypatch, controller)
    controller._inference_recovery_timeout = 0.01
    release = threading.Event()
    worker = threading.Thread(target=release.wait)
    controller._local_recovery_thread = worker
    controller._run_global_restart = Mock()
    exit_process = Mock(side_effect=SystemExit(1))
    monkeypatch.setattr(controller_module.os, "_exit", exit_process)
    worker.start()
    try:
        controller.restart_serve("actor")
        exit_process.assert_called_once_with(1)
        controller._run_global_restart.assert_not_called()
    finally:
        release.set()
        worker.join(timeout=2)


def test_global_handoff_waits_for_main_service_wait_to_return():
    controller = _controller_shell()
    controller._training_wait_done.clear()
    cancelled, continued = threading.Event(), threading.Event()
    controller._cancel_pending_tasks = Mock(side_effect=cancelled.set)

    def handoff():
        controller._wait_recovery_exit(None)
        continued.set()

    restart = threading.Thread(target=handoff)
    restart.start()
    try:
        assert cancelled.wait(2)
        assert not continued.is_set()
    finally:
        controller._training_wait_done.set()
        restart.join(timeout=2)
    assert not restart.is_alive()
    assert continued.is_set()


def test_original_in_place_restart_keeps_other_service_responses(monkeypatch):
    controller = _controller_shell()
    _patch_restart_entry(monkeypatch, controller)
    controller._health_manager.mark_healthy = Mock()
    service = SimpleNamespace(restart=Mock())
    controller.serve_dict = {"sft": service}
    response = Mock()
    controller._pending_task_refs = [response]
    controller._training_wait_done.clear()

    controller.restart_serve("sft")

    service.restart.assert_called_once_with()
    response.cancel.assert_not_called()
    assert controller._pending_task_refs == [response]
    assert controller._restart_mode == "local"
    assert controller._restart_done_event.is_set()


@pytest.mark.asyncio
async def test_supervisor_failure_requests_global_outside_async_loop():
    controller = _controller_shell()
    controller._inference_recovery = object()
    controller._recover_inference = Mock(side_effect=RuntimeError("cannot rebuild"))
    controller.restart_serve = Mock()

    async def failed_response():
        raise ValueError("inference failed")

    with pytest.raises(controller_module._GlobalRestartRequired) as raised:
        await controller._supervise_inference("rollout", failed_response(), asyncio.Lock())

    assert raised.value.role == "rollout"
    assert raised.value.state is controller._inference_recovery
    assert isinstance(raised.value.__cause__, RuntimeError)
    controller.restart_serve.assert_not_called()
    assert controller._local_recovery_thread is None


def test_training_loop_requests_global_after_service_wait_returns(monkeypatch):
    controller = _controller_shell()
    state = controller._inference_recovery = object()
    stages = []

    def service_wait(coroutine):
        coroutine.close()
        stages.append("wait")
        if len(stages) == 1:
            raise controller_module._GlobalRestartRequired("rollout", state)

    def restart(role, *, expected_state):
        assert controller._training_wait_done.is_set()
        assert expected_state is state
        stages.append("global")
        controller._restarting = True
        controller._restart_mode = "global"
        controller._restart_done_event.set()

    monkeypatch.setattr(controller_module, "run", service_wait)
    controller.restart_serve = Mock(side_effect=restart)

    controller.training_loop()

    assert stages == ["wait", "global", "wait"]
    assert not controller._restarting
    assert controller._restart_consumed_event.is_set()


def test_old_global_owner_does_not_publish_completion_into_next_cycle(monkeypatch):
    controller = _controller_shell()
    _patch_restart_entry(monkeypatch, controller)
    controller._run_global_restart = Mock()

    class AcknowledgeAndBeginNextCycle(threading.Event):
        def __init__(self):
            super().__init__()
            self.publications = 0

        def set(self):
            self.publications += 1
            super().set()
            if self.publications == 1:
                controller._consume_restart_cycle()
                controller._restarting = True
                controller._restart_mode = "global"
                controller._restart_consumed_event.clear()
                self.clear()

    event = controller._restart_done_event = AcknowledgeAndBeginNextCycle()

    controller.restart_serve("actor")

    assert event.publications == 1
    assert not event.is_set()


@pytest.mark.asyncio
@pytest.mark.parametrize("global_takeover", [False, True])
async def test_activation_publishes_and_registers_only_current_recovery(monkeypatch, global_takeover):
    controller = _controller_shell()
    controller.config = SimpleNamespace(fully_async=True)
    controller._health_manager = SimpleNamespace(mark_healthy=Mock())
    old_response, replacement = object(), object()
    controller._pending_task_refs = [old_response]

    def resume(step):
        if global_takeover:

            def claim_global():
                with controller._restart_lock:
                    controller._restarting = True

            health_thread = threading.Thread(target=claim_global)
            health_thread.start()
            health_thread.join(timeout=2)
            assert not health_thread.is_alive()
        return 1

    state = controller._inference_recovery = SimpleNamespace(
        begin_sync=SimpleNamespace(remote=Mock(return_value=True)),
        end_sync=SimpleNamespace(remote=Mock()),
        resume=SimpleNamespace(remote=Mock(side_effect=resume)),
    )
    failed = SimpleNamespace(
        set_step=AsyncMock(),
        handle=SimpleNamespace(recv_weight_fully_async=SimpleNamespace(remote=AsyncMock())),
        run=Mock(return_value=replacement),
    )
    controller.serve_dict = {
        "actor": SimpleNamespace(
            handle=SimpleNamespace(
                reset_inference_transport=SimpleNamespace(remote=AsyncMock()),
                update_weights_fully_async=SimpleNamespace(remote=AsyncMock()),
            )
        ),
        "rollout": SimpleNamespace(set_step=AsyncMock()),
        "actor_fwd": failed,
    }
    monkeypatch.setattr(controller_module.ray, "get", lambda value: value)

    activation = controller._activate_inference_recovery(
        "actor_fwd", state, step=3, attempt=1, started=controller_module.time.monotonic()
    )
    if global_takeover:
        with pytest.raises(RuntimeError, match="Global restart has taken over"):
            await activation
        failed.run.assert_not_called()
        controller._health_manager.mark_healthy.assert_not_called()
        assert controller._pending_task_refs == [old_response]
    else:
        assert await activation is replacement
        failed.run.assert_called_once_with()
        controller._health_manager.mark_healthy.assert_called_once_with("actor_fwd")
        assert controller._pending_task_refs == [old_response, replacement]
    state.resume.remote.assert_called_once_with(3)
