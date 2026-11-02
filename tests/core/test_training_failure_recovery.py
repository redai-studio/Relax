# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Exercise production control flow without a Ray cluster or GPU workers."""

import ast
import asyncio
import logging
import os
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Optional
from unittest.mock import Mock

import pytest

from relax.components.background_loop import BackgroundLoopRunner


ROOT = Path(__file__).resolve().parents[2]


def load_methods(path, class_name, names, namespace):
    tree = ast.parse((ROOT / path).read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name)
    methods = [n for n in cls.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name in names]
    for method in methods:
        method.decorator_list = []
    module = ast.Module(body=methods, type_ignores=[])
    exec(compile(module, str(ROOT / path), "exec"), namespace)
    return type(class_name, (), {name: namespace[name] for name in names})


def actor_fixture(health=True, fail=True, report_fail=False):
    reported = threading.Event()
    error = ValueError("training failed")

    def report(*args):
        return object()

    def get(ref, timeout):
        if report_fail:
            raise TimeoutError("health unavailable")
        reported.set()

    ns = dict(
        asyncio=asyncio,
        threading=threading,
        ray=SimpleNamespace(get=get),
        sft_partition_ids=lambda *args: [],
        run=asyncio.run,
    )
    cls = load_methods("relax/components/actor.py", "Actor", {"run", "_background_run", "_reset_consumption"}, ns)
    actor = cls()
    actor.config = SimpleNamespace(num_rollout=2, fully_async=True, use_health_check=health)
    actor.step = 0
    actor._loop_runner = BackgroundLoopRunner()
    actor._lock = threading.Lock()
    actor._stop_event = threading.Event()
    actor._logger = logging.getLogger(__name__)
    actor.healthy = SimpleNamespace(report_error=SimpleNamespace(remote=report), update_heartbeat=Mock())
    actor._execute_training = Mock(side_effect=error if fail else None, return_value=False)
    return actor, reported, error


def test_actor_failure_waits_for_recovery_and_reaches_all_callers():
    actor, reported, error = actor_fixture()

    async def check():
        first = asyncio.create_task(actor.run())
        assert await asyncio.to_thread(reported.wait, 2)
        second = asyncio.create_task(actor.run())
        await asyncio.sleep(0.02)
        assert not first.done() and not second.done()
        actor._stop_event.set()
        results = await asyncio.wait_for(asyncio.gather(first, second, return_exceptions=True), 2)
        assert all(result is error for result in results)

    asyncio.run(check())


@pytest.mark.parametrize("health,report_fail", [(False, False), (True, True)])
def test_actor_failure_without_recovery_propagates_original(health, report_fail):
    actor, _, error = actor_fixture(health=health, report_fail=report_fail)
    with pytest.raises(ValueError) as caught:
        asyncio.run(actor.run())
    assert caught.value is error


def test_actor_only_completes_after_all_steps():
    actor, _, _ = actor_fixture(fail=False)
    asyncio.run(actor.run())
    assert actor.step == 2
    assert actor._execute_training.call_count == 2


@pytest.mark.parametrize("health", [False, True])
def test_actor_early_stop_is_not_success_or_health_failure(health):
    actor, reported, _ = actor_fixture(health=health, fail=False)
    actor._stop_event.set()
    with pytest.raises(RuntimeError, match="before all training"):
        asyncio.run(actor.run())
    assert not reported.is_set()
    actor._execute_training.assert_not_called()


def controller_fixture():
    ns = dict(logger=logging.getLogger(__name__), run=asyncio.run, asyncio=asyncio, Optional=Optional)
    cls = load_methods(
        "relax/core/controller.py", "Controller", {"restart_serve", "training_loop", "_consume_restart_cycle"}, ns
    )
    ctrl = cls()
    ctrl._restart_done_event = threading.Event()
    ctrl._restart_lock = threading.RLock()
    ctrl._restart_consumed_event = threading.Event()
    ctrl._restart_consumed_event.set()
    ctrl._restart_mode = None
    ctrl._health_manager = SimpleNamespace(_checker=SimpleNamespace(_stop_event=threading.Event()))
    ctrl._restart_error = None
    ctrl._restarting = False
    ctrl._pending_task_refs_lock = threading.Lock()
    ctrl._pending_task_refs = []
    ctrl.config = SimpleNamespace(debug_train_only=True)
    ctrl._cleanup_s3_model_weights_after_init = Mock()
    ctrl._cancel_pending_tasks = Mock()
    ctrl._report_error_to_metrics_service = Mock()
    return ctrl


@pytest.mark.parametrize("failure", [False, True])
def test_restart_publishes_state_and_signals_on_every_exit(failure):
    ctrl = controller_fixture()
    error = ValueError("checkpoint or teardown failed")

    def restart(role):
        assert ctrl._restarting
        assert not ctrl._restart_done_event.is_set()
        if failure:
            raise error

    ctrl._restart_serve = restart
    ctrl.restart_serve("actor")
    assert ctrl._restarting
    assert ctrl._restart_done_event.is_set()
    assert ctrl._restart_error is (error if failure else None)
    assert ctrl._cancel_pending_tasks.call_count == int(failure)


@pytest.mark.parametrize("recovery_fails", [False, True])
@pytest.mark.parametrize("old_returns", [False, True])
def test_controller_retries_recovery_but_never_reports_failed_recovery_as_success(recovery_fails, old_returns):
    ctrl = controller_fixture()
    calls = []
    error = ValueError("reinitialization failed")

    def restart(role):
        if recovery_fails:
            raise error

    ctrl._restart_serve = restart

    async def service_run():
        calls.append(1)
        if len(calls) == 1:
            ctrl.restart_serve("actor")
            if not old_returns:
                raise RuntimeError("old replica deleted")

    ctrl.serve_dict = {"actor": SimpleNamespace(run=service_run, role="actor")}
    if recovery_fails:
        with pytest.raises(RuntimeError, match="global restart failed"):
            ctrl.training_loop()
        assert len(calls) == 1
        ctrl._report_error_to_metrics_service.assert_called_once_with(error)
    else:
        ctrl.training_loop()
        assert len(calls) == 2
        assert not ctrl._restarting
        assert not ctrl._restart_done_event.is_set()


def test_controller_unhandled_service_failure_propagates():
    ctrl = controller_fixture()

    async def service_run():
        raise ValueError("worker failed")

    ctrl.serve_dict = {"actor": SimpleNamespace(run=service_run, role="actor")}
    with pytest.raises(RuntimeError, match="worker failed"):
        ctrl.training_loop()
    ctrl._report_error_to_metrics_service.assert_called_once()


def test_next_recovery_waits_until_previous_error_is_consumed():
    ctrl = controller_fixture()
    error = ValueError("first recovery failed")
    ctrl._restart_serve = Mock(side_effect=error)
    ctrl.restart_serve("actor")
    next_cycle = threading.Thread(target=ctrl.restart_serve, args=("another role",), daemon=True)
    next_cycle.start()
    next_cycle.join(0.05)
    assert next_cycle.is_alive()
    assert ctrl._restart_error is error
    ctrl._restart_serve = Mock()
    mode, failure = ctrl._consume_restart_cycle()
    assert mode == "global" and failure is error
    next_cycle.join(2)
    assert not next_cycle.is_alive()
    ctrl._restart_serve.assert_called_once_with("another role")
    assert ctrl._restart_error is None


def test_successful_recovery_allows_later_independent_recovery():
    ctrl = controller_fixture()
    ctrl._restart_serve = Mock()
    calls = []

    async def service_run():
        calls.append(1)
        if len(calls) in (1, 3):
            ctrl.restart_serve("actor")
            raise RuntimeError("replica deleted")

    ctrl.serve_dict = {"actor": SimpleNamespace(run=service_run, role="actor")}
    ctrl.training_loop()
    ctrl.training_loop()
    assert ctrl._restart_serve.call_count == 2
    assert len(calls) == 4


def test_reinitialization_preserves_recovery_synchronization():
    tree = ast.parse((ROOT / "relax/core/controller.py").read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Controller")
    init = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "__init__")
    blocks = [
        n
        for n in init.body
        if isinstance(n, ast.If)
        and any(
            isinstance(c, ast.Constant)
            and c.value
            in {
                "_restart_done_event",
                "_restart_lock",
                "_restart_consumed_event",
                "_restart_error",
                "_restarting",
                "_restart_mode",
            }
            for c in ast.walk(n.test)
        )
    ]
    code = compile(ast.Module(body=blocks, type_ignores=[]), "controller.py", "exec")
    ctrl = SimpleNamespace()
    scope = {"self": ctrl, "threading": threading}
    exec(code, scope)
    event, lock = ctrl._restart_done_event, ctrl._restart_lock
    ctrl._restarting = True
    error = RuntimeError("saved error")
    ctrl._restart_error = error
    exec(code, scope)
    assert ctrl._restart_done_event is event
    assert ctrl._restart_lock is lock
    assert ctrl._restarting
    assert ctrl._restart_error is error


@pytest.mark.parametrize("fails,expected", [(False, 0), (True, 1)])
def test_entrypoint_exits_nonzero_for_training_failure(fails, expected):
    tree = ast.parse((ROOT / "relax/entrypoints/train.py").read_text())
    main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")
    start = next(
        i
        for i, n in enumerate(main.body)
        if isinstance(n, ast.Try)
        and any(isinstance(c, ast.Attribute) and c.attr == "training_loop" for c in ast.walk(n))
    )
    code = compile(ast.Module(body=main.body[start:], type_ignores=[]), "train.py", "exec")

    def shutdown(*, exit_code):
        raise SystemExit(exit_code)

    class Controller:
        def __init__(self, *args):
            pass

        def training_loop(self):
            if fails:
                raise ValueError("failed")

    scope = {
        "Controller": Controller,
        "os": SimpleNamespace(path=os.path, environ={}),
        "cur_file_dir": str(ROOT),
        "yaml": SimpleNamespace(safe_load=lambda file: {}),
        "args": SimpleNamespace(),
        "post_process_env": lambda args, env: env,
        "ray": SimpleNamespace(is_initialized=lambda: True),
        "init_tracking": Mock(),
        "maybe_pin_baseline_to_stable": Mock(),
        "logger": logging.getLogger(__name__),
        "_graceful_shutdown": shutdown,
    }
    with pytest.raises(SystemExit) as caught:
        exec(code, scope)
    assert caught.value.code == expected


@pytest.mark.parametrize("native_ref", [False, True])
def test_cancel_uses_the_matching_ray_api(native_ref):
    ray = SimpleNamespace(cancel=Mock(), ObjectRef=type("ObjectRef", (), {}))
    cls = load_methods(
        "relax/core/controller.py",
        "Controller",
        {"_cancel_pending_tasks"},
        {"ray": ray, "logger": logging.getLogger(__name__)},
    )
    ctrl = cls()
    ref = ray.ObjectRef() if native_ref else SimpleNamespace(cancel=Mock())
    ctrl._pending_task_refs = [ref]
    ctrl._pending_task_refs_lock = threading.Lock()
    ctrl._cancel_pending_tasks()
    assert ctrl._pending_task_refs == []
    if native_ref:
        ray.cancel.assert_called_once_with(ref, force=False)
    else:
        ref.cancel.assert_called_once_with()
        ray.cancel.assert_not_called()


def test_failed_recovery_cancels_a_pending_serve_response_and_unblocks_training():
    ctrl = controller_fixture()
    cls = load_methods(
        "relax/core/controller.py",
        "Controller",
        {"_cancel_pending_tasks"},
        {"logger": logging.getLogger(__name__), "ray": SimpleNamespace(ObjectRef=type("ObjectRef", (), {}))},
    )
    ctrl._cancel_pending_tasks = cls._cancel_pending_tasks.__get__(ctrl)
    error = ValueError("failed before replica deletion")
    ctrl._restart_serve = Mock(side_effect=error)

    class Response:
        def __await__(self):
            async def wait():
                self.future = asyncio.get_running_loop().create_future()
                asyncio.get_running_loop().call_soon(ctrl.restart_serve, "actor")
                await self.future

            return wait().__await__()

        def cancel(self):
            self.future.cancel()

    response = Response()
    ctrl.serve_dict = {"actor": SimpleNamespace(run=lambda: response, role="actor")}
    with pytest.raises(RuntimeError, match="global restart failed"):
        ctrl.training_loop()
    ctrl._report_error_to_metrics_service.assert_called_once_with(error)


@pytest.mark.parametrize("failure", ["unknown_role", "counter"])
def test_early_recovery_failure_signals_and_cancels(failure):
    ctrl = controller_fixture()
    ns = dict(
        logger=logging.getLogger(__name__),
        resolve_sft_algo_key=lambda config: "sft",
        ALGOS={"sft": {"actor": object}},
        register_extra_roles=lambda config, algo: None,
    )
    cls = load_methods("relax/core/controller.py", "Controller", {"_restart_serve"}, ns)
    ctrl._restart_serve = cls._restart_serve.__get__(ctrl)
    ctrl._health_manager.increment_restart_count = Mock(side_effect=ValueError("counter failed"))
    ctrl.restart_serve("missing" if failure == "unknown_role" else "actor")
    assert ctrl._restart_done_event.is_set()
    assert ctrl._restart_error is not None
    assert ctrl._health_manager._checker._stop_event.is_set()
    ctrl._cancel_pending_tasks.assert_called_once()


def test_local_recovery_resumes_existing_task_without_restarting_service():
    ctrl = controller_fixture()
    consumed = []

    async def resumed():
        consumed.append(True)

    def restart(role):
        ctrl._restart_mode = "local"
        service._task_ref = resumed()

    async def first():
        ctrl.restart_serve("sft")
        raise RuntimeError("old replica deleted")

    service = SimpleNamespace(run=Mock(side_effect=first), role="sft")
    ctrl._restart_serve = restart
    ctrl.serve_dict = {"sft": service}
    ctrl.training_loop()
    assert consumed == [True]
    service.run.assert_called_once()


@pytest.mark.parametrize("role,global_restart", [("actor", True), ("rollout", True), ("sft", False)])
def test_recovery_dispatch_signals_only_after_work_finishes(role, global_restart, monkeypatch):
    from relax.core import registry

    monkeypatch.setattr(registry, "resolve_algorithm", lambda config: {role: object})
    ctrl = controller_fixture()
    roles = SimpleNamespace(**{name: name for name in ("actor", "rollout", "actor_fwd", "critic", "reference")})
    ns = dict(
        logger=logging.getLogger(__name__),
        resolve_sft_algo_key=lambda config: "sft",
        ALGOS={"sft": {role: object}},
        register_extra_roles=lambda config, algo: None,
        ROLES=roles,
        GENRM_ROLE="genrm",
    )
    cls = load_methods("relax/core/controller.py", "Controller", {"_restart_serve"}, ns)
    ctrl._restart_serve = cls._restart_serve.__get__(ctrl)
    ctrl._health_manager.increment_restart_count = Mock(return_value=1)
    ctrl._health_manager.mark_healthy = Mock()

    def work():
        assert ctrl._restarting
        assert not ctrl._restart_done_event.is_set()

    ctrl._run_global_restart = Mock(side_effect=work)
    service = SimpleNamespace(restart=Mock(side_effect=work))
    ctrl.serve_dict = {role: service}
    ctrl.restart_serve(role)
    assert ctrl._restart_error is None
    assert ctrl._restart_done_event.is_set()
    assert ctrl._run_global_restart.call_count == int(global_restart)
    assert service.restart.call_count == int(not global_restart)
    assert ctrl._restart_mode == ("global" if global_restart else "local")


def test_real_controller_initialization_wires_health_callback_and_preserves_restart_events(monkeypatch):
    import transfer_queue

    from tests.core.controller_test_utils import load_controller_with_stubbed_dependencies

    # Data-system initialization is a cluster boundary in this contract. Older
    # host TQ builds need only these import names; no sampler behavior is faked.
    for name in ("SeqlenBalancedSampler", "StreamingTokenBudgetSampler"):
        if not hasattr(transfer_queue, name):
            monkeypatch.setattr(transfer_queue, name, type(name, (), {}), raising=False)
    module = load_controller_with_stubbed_dependencies("_test_recovery_controller_contract")
    health = Mock()
    monkeypatch.setattr(module, "HealthManager", Mock(return_value=health))
    monkeypatch.setattr(module, "require_control_plane_resource", Mock())
    monkeypatch.setattr(module, "with_control_plane_affinity", lambda config, options: options)
    monkeypatch.setattr(module, "resolve_sft_num_rollout", Mock(return_value=None))
    monkeypatch.setattr(module, "create_dcs_deployment", Mock(return_value=(None, None)))
    monkeypatch.setattr(module.Controller, "_initialize_data_system", Mock())
    monkeypatch.setattr(module.Controller, "register_all_serve", Mock())
    config = SimpleNamespace(use_health_check=True, use_agentic_rollout=False)
    controller = module.Controller(config)
    health.start.assert_called_once_with(
        on_unhealthy=controller._on_service_unhealthy, on_fatal=controller._on_service_fatal
    )
    restart = Mock()
    monkeypatch.setattr(controller, "restart_serve", restart)
    health.start.call_args.kwargs["on_unhealthy"]("actor")
    restart.assert_called_once_with("actor")
    done, consumed = controller._restart_done_event, controller._restart_consumed_event
    controller._restart_mode = "global"
    controller._restarting = True
    consumed.clear()
    controller.__init__(config)
    assert controller._restart_done_event is done
    assert controller._restart_consumed_event is consumed
    done.set()
    assert controller._consume_restart_cycle() == ("global", None)
    assert not done.is_set()
    assert consumed.is_set()
    assert not controller._restarting
