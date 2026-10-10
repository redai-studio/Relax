# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""CPU regressions for startup failure propagation and bounded cleanup."""

import ast
import concurrent.futures
import os
import signal
import subprocess
import sys
import threading
from enum import Enum
from pathlib import Path
from types import SimpleNamespace
from typing import Optional
from unittest.mock import Mock

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[2]


def _load_definitions(path, names, namespace):
    # These orchestration functions only need mocked service APIs. Avoid importing
    # CUDA/Megatron dependencies just to exercise their actual control flow.
    tree = ast.parse((ROOT / path).read_text())
    tree.body = [
        node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names
    ]
    exec(compile(tree, str(ROOT / path), "exec"), namespace)
    return namespace


def _train_namespace():
    namespace = {
        "os": os,
        "sys": sys,
        "signal": Mock(SIGTERM=signal.SIGTERM, SIGINT=signal.SIGINT, Signals=signal.Signals),
        "threading": threading,
        "atexit": Mock(),
        "yaml": yaml,
        "cur_file_dir": ROOT,
        "logger": Mock(),
        "ray": Mock(),
        "serve": Mock(),
        "Controller": Mock(),
        "post_process_env": lambda args, env: env,
        "init_tracking": Mock(),
        "maybe_pin_baseline_to_stable": Mock(),
        "_ctrl": None,
        "_shutdown_done": False,
        "_kernel_cache_heartbeat": None,
        "_SHUTDOWN_TIMEOUT_SEC": 60,
        "_NORMAL_SHUTDOWN_TIMEOUT_SEC": 1800,
    }
    return _load_definitions("relax/entrypoints/train.py", {"main", "_hard_exit", "_graceful_shutdown"}, namespace)


@pytest.mark.parametrize("fail_at", ["startup", "training", None])
def test_train_cleans_up_startup_and_training(fail_at, monkeypatch):
    monkeypatch.delenv("RELAX_KERNEL_CACHE_DIR", raising=False)
    ns = _train_namespace()
    events = []
    controller = Mock()

    class FakeController:
        def __init__(self, *args):
            ns["signal"].signal.assert_any_call(signal.SIGTERM, ns["_graceful_shutdown"])
            ns["atexit"].register.assert_called_once_with(ns["_graceful_shutdown"])
            if fail_at == "startup":
                raise RuntimeError("Gloo connectFullMesh failed")

        shutdown = controller.shutdown
        training_loop = controller.training_loop

    def hard_exit(code):
        events.append(("exit", code))
        raise SystemExit(code)

    ns["Controller"] = FakeController
    ns["_hard_exit"] = hard_exit
    if fail_at == "training":
        controller.training_loop.side_effect = RuntimeError("training failed")
    ns["serve"].shutdown.side_effect = lambda: events.append("serve")
    ns["ray"].shutdown.side_effect = lambda: events.append("ray")
    with pytest.raises(SystemExit) as exc:
        ns["main"](SimpleNamespace())
    expected = 1 if fail_at else 0
    assert exc.value.code == expected
    assert events == ["serve", "ray", ("exit", expected)]
    controller.shutdown.assert_called_once()


def test_train_cleans_up_if_tracking_fails_before_controller(monkeypatch):
    monkeypatch.delenv("RELAX_KERNEL_CACHE_DIR", raising=False)
    ns = _train_namespace()
    events = []

    def hard_exit(code):
        events.append(("exit", code))
        raise SystemExit(code)

    ns["init_tracking"].side_effect = RuntimeError("tracking failed")
    ns["_hard_exit"] = hard_exit
    ns["serve"].shutdown.side_effect = lambda: events.append("serve")
    ns["ray"].shutdown.side_effect = lambda: events.append("ray")
    with pytest.raises(SystemExit) as exc:
        ns["main"](SimpleNamespace())

    assert exc.value.code == 1
    assert events == ["serve", "ray", ("exit", 1)]


@pytest.mark.parametrize("hang_at", ["serve", "controller"])
@pytest.mark.parametrize("exit_code", [0, 1, 143])
def test_train_shutdown_deadline_exits_blocked_process(hang_at, exit_code):
    script = f"""
import runpy
import threading
ns = runpy.run_path({str(Path(__file__).resolve())!r})["_train_namespace"]()
ns["_SHUTDOWN_TIMEOUT_SEC"] = 0.05
ns["_NORMAL_SHUTDOWN_TIMEOUT_SEC"] = 0.05
if {hang_at!r} == "controller":
    ns["_ctrl"] = ns["Controller"].return_value
    ns["_ctrl"].shutdown.side_effect = threading.Event().wait
else:
    ns["serve"].shutdown.side_effect = threading.Event().wait
ns["_graceful_shutdown"](exit_code={exit_code})
"""
    result = subprocess.run([sys.executable, "-c", script], timeout=10, capture_output=True, text=True)
    assert result.returncode == exit_code, result.stderr


class _FakePhase(str, Enum):
    ALLOCATE_RESOURCES = "allocate resources"
    DEPLOY = "deploy"


class _RecordingRemovePG:
    def __init__(self):
        self.calls = []

    def __call__(self, pg):
        self.calls.append(pg)


def _service_phase(fully_async, *, phase=_FakePhase.DEPLOY, serve_dict=None, remove_pg=None):
    tree = ast.parse((ROOT / "relax/core/controller.py").read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "Controller")
    wanted = {"_run_service_phase", "_abort_service_phase"}
    methods = [node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name in wanted]
    namespace = {
        "concurrent": SimpleNamespace(futures=concurrent.futures),
        "threading": threading,
        "logger": Mock(),
        "serve": Mock(),
        "ServiceStartupPhase": _FakePhase,
        "remove_placement_group": remove_pg or (lambda pg: None),
        "Any": object,
        "Optional": Optional,
        "Service": object,
    }
    exec(compile(ast.Module(body=methods, type_ignores=[]), "controller.py", "exec"), namespace)
    controller = SimpleNamespace(
        config=SimpleNamespace(fully_async=fully_async),
        serve_dict={} if serve_dict is None else serve_dict,
        _serve=namespace["serve"],
    )
    controller._abort_service_phase = lambda phase_, services, failed_services=None: namespace["_abort_service_phase"](
        controller, phase_, services, failed_services
    )
    return controller, lambda task, args: namespace["_run_service_phase"](controller, phase, task, args)


def test_controller_serial_startup_stops_at_first_failure():
    calls = []

    def task(role):
        calls.append(role)
        return role, None, "Gloo failed"

    _, run = _service_phase(False)
    with pytest.raises(RuntimeError, match="actor: Gloo failed"):
        run(task, [("actor",), ("rollout",)])
    assert calls == ["actor"]


def test_controller_parallel_startup_does_not_wait_for_blocked_sibling():
    started = threading.Event()
    release = threading.Event()
    finished = threading.Event()
    errors = []

    def task(role):
        if role == "rollout":
            started.set()
            release.wait()
            return role, object(), None
        assert started.wait(2)
        return role, None, "Gloo failed"

    controller, run = _service_phase(True)

    def run_phase():
        try:
            run(task, [("rollout",), ("actor",)])
        except RuntimeError as exc:
            errors.append(str(exc))
        finally:
            finished.set()

    thread = threading.Thread(target=run_phase, daemon=True)
    thread.start()
    try:
        assert finished.wait(3), "failure waited for the blocked sibling"
        assert errors == ["Failed to deploy service actor: Gloo failed"]
        controller._serve.delete.assert_called_once_with("rollout")
    finally:
        release.set()
        thread.join(3)


@pytest.mark.parametrize("fully_async", [False, True])
def test_controller_startup_returns_successful_services(fully_async):
    services = {"actor": object(), "rollout": object()}
    _, run = _service_phase(fully_async)
    assert run(lambda role: (role, services[role], None), [(r,) for r in services]) == services


def test_controller_deploy_phase_failure_keeps_successful_services_for_shutdown():
    # DEPLOY successes are alive Serve deployments; shutdown() needs them in
    # serve_dict to dispose SGLang engines and release handles.
    deployed = object()
    remove_pg = _RecordingRemovePG()
    controller, run = _service_phase(False, phase=_FakePhase.DEPLOY, remove_pg=remove_pg)

    def task(role):
        if role == "actor":
            return role, deployed, None
        return role, None, "Gloo failed"

    with pytest.raises(RuntimeError, match="rollout: Gloo failed"):
        run(task, [("actor",), ("rollout",)])
    assert controller.serve_dict == {"actor": deployed}
    assert remove_pg.calls == []


def test_controller_allocate_phase_failure_releases_pgs_without_touching_serve_dict():
    # ALLOCATE successes have placement groups but no handle. Adding them to
    # serve_dict would make shutdown() AttributeError on .handle; release the
    # PGs here so ray.shutdown() is not the only backstop.
    actor_pg = object()
    rollout_pg = object()
    prepared_actor = SimpleNamespace(pgs=(actor_pg, [0], ["gpu-0"]))
    failed_rollout = SimpleNamespace(pgs=(rollout_pg, [0], ["gpu-0"]))
    remove_pg = _RecordingRemovePG()
    controller, run = _service_phase(False, phase=_FakePhase.ALLOCATE_RESOURCES, remove_pg=remove_pg)

    def task(role):
        if role == "actor":
            return role, prepared_actor, None
        return role, failed_rollout, "resource unavailable"

    with pytest.raises(RuntimeError, match="rollout: resource unavailable"):
        run(task, [("actor",), ("rollout",)])
    assert controller.serve_dict == {}
    assert remove_pg.calls == [actor_pg, rollout_pg]


def test_controller_allocate_phase_abort_survives_pg_cleanup_failure():
    # A broken placement group during phase abort must not mask the original
    # startup failure.
    def raiser(_):
        raise RuntimeError("pg gone")

    controller, run = _service_phase(False, phase=_FakePhase.ALLOCATE_RESOURCES, remove_pg=raiser)
    prepared_actor = SimpleNamespace(pgs=(object(), [0], ["gpu-0"]))

    def task(role):
        if role == "actor":
            return role, prepared_actor, None
        return role, None, "resource unavailable"

    with pytest.raises(RuntimeError, match="rollout: resource unavailable"):
        run(task, [("actor",), ("rollout",)])
    assert controller.serve_dict == {}


@pytest.mark.parametrize("has_data_source", [False, True])
def test_service_constructor_failure_is_bounded_and_propagated(has_data_source):
    ns = {
        "Any": object,
        "Optional": Optional,
        "Namespace": SimpleNamespace,
        "logger": Mock(),
        "with_control_plane_affinity": lambda config, options: options,
        "serve": Mock(),
    }
    service_cls = _load_definitions("relax/core/service.py", {"Service"}, ns)["Service"]
    service = object.__new__(service_cls)
    service.cls = Mock()
    service.config = SimpleNamespace()
    service.runtime_env = None
    service.healthy = Mock()
    service.role = "actor"
    service.num_gpus = 0
    service.data_source = object() if has_data_source else None
    service.pgs = None
    service._deployed = False
    ns["serve"].run.side_effect = RuntimeError("constructor failed")
    with pytest.raises(RuntimeError, match="constructor failed"):
        service.deploy()
    assert service.cls.options.call_args.kwargs["max_constructor_retry_count"] == 1
    assert service._deployed is False


def test_service_deploy_failure_releases_owned_pg():
    remove_pg = Mock()
    ns = {
        "Any": object,
        "Optional": Optional,
        "Namespace": SimpleNamespace,
        "logger": Mock(),
        "with_control_plane_affinity": lambda config, options: options,
        "remove_placement_group": remove_pg,
        "serve": Mock(),
    }
    service_cls = _load_definitions("relax/core/service.py", {"Service"}, ns)["Service"]
    service = object.__new__(service_cls)
    service.cls = Mock()
    service.config = SimpleNamespace()
    service.runtime_env = None
    service.healthy = Mock()
    service.role = "actor"
    service.num_gpus = 1
    service.data_source = None
    pg = object()
    service.pgs = (pg, [0], ["gpu-0"])
    service._is_shared_pgs = False
    service._deployed = False
    ns["serve"].run.side_effect = RuntimeError("deploy failed")

    with pytest.raises(RuntimeError, match="deploy failed"):
        service.deploy()

    ns["serve"].delete.assert_called_once_with("actor")
    remove_pg.assert_called_once_with(pg)
    assert service.pgs is None


def test_train_normal_shutdown_arms_long_watchdog_and_cancels_it():
    ns = _train_namespace()
    fake_threading = Mock()
    timer = Mock()
    fake_threading.Timer.return_value = timer
    ns["threading"] = fake_threading
    ns["_hard_exit"] = Mock()
    ns["_graceful_shutdown"](exit_code=0)
    # Normal exit still installs a watchdog so a hanging shutdown cannot keep
    # the Ray Job alive forever; the timeout is the larger normal-path budget.
    fake_threading.Timer.assert_called_once()
    call = fake_threading.Timer.call_args
    assert call.args[0] == ns["_NORMAL_SHUTDOWN_TIMEOUT_SEC"]
    assert call.args[1] is ns["os"]._exit
    assert call.kwargs["args"] == (0,)
    timer.start.assert_called_once()
    timer.cancel.assert_called_once()
    ns["_hard_exit"].assert_called_once_with(0)


def test_train_failure_shutdown_uses_short_watchdog_budget():
    ns = _train_namespace()
    fake_threading = Mock()
    ns["threading"] = fake_threading
    ns["_hard_exit"] = Mock()
    ns["_graceful_shutdown"](exit_code=1)
    call = fake_threading.Timer.call_args
    assert call.args[0] == ns["_SHUTDOWN_TIMEOUT_SEC"]
    assert call.kwargs["args"] == (1,)


def test_controller_partial_startup_cleans_teacher_and_routers(monkeypatch):
    tree = ast.parse((ROOT / "relax/core/controller.py").read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "Controller")
    method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "shutdown")
    stop_routers = Mock(return_value=1)
    monkeypatch.setitem(
        sys.modules, "relax.distributed.ray.rollout", SimpleNamespace(stop_launched_routers=stop_routers)
    )
    stop_teacher = Mock()
    ns = {"logger": Mock(), "ROLES": SimpleNamespace(rollout="rollout"), "shutdown_managed_opd_teacher": stop_teacher}
    exec(compile(ast.Module(body=[method], type_ignores=[]), "controller.py", "exec"), ns)
    teacher = object()
    partial_controller = SimpleNamespace(
        serve_dict={},
        _teacher_manager=teacher,
        _shutdown_agentic_rollout_services=Mock(),
        _cleanup_s3_model_weights_after_init=Mock(),
    )
    ns["shutdown"](partial_controller)
    stop_teacher.assert_called_once_with(teacher)
    stop_routers.assert_called_once()
