# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""CPU regressions for startup failure propagation and bounded cleanup."""

import ast
import asyncio
import concurrent.futures
import json
import os
import signal
import subprocess
import sys
import threading
from enum import Enum
from pathlib import Path
from types import SimpleNamespace
from typing import Callable, Optional
from unittest.mock import AsyncMock, Mock

import pytest
import yaml

from relax.utils.agentic_obs.run_hooks import RunRecorder


ROOT = Path(__file__).resolve().parents[3]


def _load_definitions(path, names, namespace):
    # These orchestration functions only need mocked service APIs. Avoid importing
    # CUDA/Megatron dependencies just to exercise their actual control flow.
    namespace.setdefault("Callable", Callable)
    tree = ast.parse((ROOT / path).read_text())
    tree.body = [
        node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names
    ]
    exec(compile(tree, str(ROOT / path), "exec"), namespace)
    return namespace


def _train_namespace():
    namespace = {
        "asyncio": asyncio,
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
        "RunRecorder": RunRecorder,
        "post_process_env": lambda args, env: env,
        "init_tracking": Mock(),
        "quiesce_serve_routers": Mock(return_value=[]),
        "shutdown_serve_router_loops": Mock(),
        "shutdown_async_loop": Mock(),
        "maybe_pin_baseline_to_stable": Mock(),
        "_ctrl": None,
        "_shutdown_done": False,
        "_training_completed": False,
        "_kernel_cache_heartbeat": None,
        "_fault_runtime": None,
        "_run_recorder": None,
        "_unhandled_exit_code": 0,
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
    assert ns["_training_completed"] is (fail_at is None)
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


@pytest.mark.parametrize("failure_stage", [None, "startup", "training", "checkpoint"])
def test_train_records_run_outcome_before_remote_cleanup(tmp_path, monkeypatch, failure_stage):
    monkeypatch.delenv("RELAX_KERNEL_CACHE_DIR", raising=False)
    ns = _train_namespace()
    path = tmp_path / "run_test/stages/run.jsonl"
    before_cleanup = []

    class FakeController:
        def __init__(self, *args):
            if failure_stage == "startup":
                raise RuntimeError("startup failed")

        def training_loop(self):
            if failure_stage in ("training", "checkpoint"):
                error = OSError("Disk quota exceeded")
                if failure_stage == "checkpoint":
                    error._relax_run_failure_stage = "checkpoint"
                raise error

        def shutdown(self):
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            before_cleanup.extend(rows)

    def hard_exit(code):
        raise SystemExit(code)

    ns["Controller"] = FakeController
    ns["_hard_exit"] = hard_exit
    args = SimpleNamespace(dump_agentic_obs=str(tmp_path), agentic_obs_run_id="run_test")
    with pytest.raises(SystemExit) as caught:
        ns["main"](args)
    assert caught.value.code == (1 if failure_stage else 0)
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(before_cleanup) == 2 and before_cleanup[1]["is_terminal"]
    assert before_cleanup[1]["operation"] == (failure_stage or "training")
    assert before_cleanup[1]["status"] == ("failed" if failure_stage else "completed")
    assert rows[0]["stage_id"] == rows[1]["stage_id"]
    assert rows[1]["occurred_at"] >= rows[0]["occurred_at"]
    if failure_stage:
        assert "failed" in rows[1]["reason"] or "Disk quota exceeded" in rows[1]["reason"]


@pytest.mark.parametrize("sig", [signal.SIGTERM, signal.SIGINT])
@pytest.mark.parametrize("recorded", [None, "failed", "completed"])
def test_train_signal_preserves_known_run_outcome(tmp_path, sig, recorded):
    ns = _train_namespace()
    recorder = RunRecorder(SimpleNamespace(dump_agentic_obs=str(tmp_path), agentic_obs_run_id="run_test"))
    recorder.start()
    recorder.set_phase("training")
    if recorded is not None:
        recorder.record_outcome(recorded)
    ns["_run_recorder"] = recorder

    def hard_exit(code):
        raise SystemExit(code)

    ns["_hard_exit"] = hard_exit
    with pytest.raises(SystemExit) as caught:
        ns["_graceful_shutdown"](sig)
    assert caught.value.code == 128 + sig
    path = tmp_path / "run_test/stages/run.jsonl"
    start, end = [json.loads(line) for line in path.read_text().splitlines()]
    assert end["status"] == (recorded or "cancelled")
    assert end["is_terminal"] and start["stage_id"] == end["stage_id"]
    if recorded is None:
        assert signal.Signals(sig).name in end["reason"]


def test_train_atexit_does_not_infer_success_from_process_exit(tmp_path):
    ns = _train_namespace()
    recorder = RunRecorder(SimpleNamespace(dump_agentic_obs=str(tmp_path), agentic_obs_run_id="run_test"))
    recorder.start()
    ns["_run_recorder"] = recorder
    ns["_graceful_shutdown"]()
    path = tmp_path / "run_test/stages/run.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(rows) == 1 and rows[0]["status"] == "start"


@pytest.mark.parametrize(
    "error,status",
    [
        (KeyboardInterrupt(), "cancelled"),
        (asyncio.CancelledError(), "cancelled"),
        (SystemExit(0), "cancelled"),
        (SystemExit(2), "failed"),
    ],
)
def test_train_base_exception_records_outcome_without_replacing_error(tmp_path, monkeypatch, error, status):
    monkeypatch.delenv("RELAX_KERNEL_CACHE_DIR", raising=False)
    ns = _train_namespace()

    class FakeController:
        def __init__(self, *args):
            pass

        def training_loop(self):
            raise error

        def shutdown(self):
            pass

    ns["Controller"] = FakeController
    args = SimpleNamespace(dump_agentic_obs=str(tmp_path), agentic_obs_run_id="run_test")
    with pytest.raises(type(error)) as caught:
        ns["main"](args)
    assert caught.value is error
    path = tmp_path / "run_test/stages/run.jsonl"
    timer = Mock()
    ns["threading"] = SimpleNamespace(Timer=Mock(return_value=timer))
    recorder = ns["_run_recorder"]
    original_flush = recorder.flush

    def guarded_flush():
        timer.start.assert_called_once()
        original_flush()

    monkeypatch.setattr(recorder, "flush", guarded_flush)
    # Simulate the registered atexit callback after the original exception.
    ns["_graceful_shutdown"]()
    expected_code = (
        error.code if isinstance(error, SystemExit) else (130 if isinstance(error, KeyboardInterrupt) else 1)
    )
    assert ns["threading"].Timer.call_args.kwargs["args"] == (expected_code or 1,)
    start, end = [json.loads(line) for line in path.read_text().splitlines()]
    assert start["status"] == "start"
    assert end["status"] == status and end["operation"] == "training"


def test_train_sigterm_flushes_run_outcome_before_hard_exit(tmp_path):
    script = f"""
import os
import runpy
import signal
from types import SimpleNamespace
from relax.utils.agentic_obs.run_hooks import RunRecorder
ns = runpy.run_path({str(Path(__file__).resolve())!r})['_train_namespace']()
recorder = RunRecorder(SimpleNamespace(dump_agentic_obs={str(tmp_path)!r}, agentic_obs_run_id='run_test'))
recorder.start()
recorder.set_phase('training')
ns['_run_recorder'] = recorder
signal.signal(signal.SIGTERM, ns['_graceful_shutdown'])
os.kill(os.getpid(), signal.SIGTERM)
"""
    result = subprocess.run([sys.executable, "-c", script], timeout=10, capture_output=True, text=True)
    assert result.returncode == 128 + signal.SIGTERM, result.stderr
    path = tmp_path / "run_test/stages/run.jsonl"
    start, end = [json.loads(line) for line in path.read_text().splitlines()]
    assert end["status"] == "cancelled" and "SIGTERM" in end["reason"]
    assert start["stage_id"] == end["stage_id"]


@pytest.mark.parametrize("hang_at", ["serve", "controller"])
@pytest.mark.parametrize("exit_code", [0, 1, 143])
def test_train_shutdown_deadline_exits_blocked_process(tmp_path, hang_at, exit_code):
    script = f"""
import runpy
import threading
from types import SimpleNamespace
from relax.utils.agentic_obs.run_hooks import RunRecorder
ns = runpy.run_path({str(Path(__file__).resolve())!r})["_train_namespace"]()
recorder = RunRecorder(SimpleNamespace(dump_agentic_obs={str(tmp_path)!r}, agentic_obs_run_id='run_test'))
recorder.start()
recorder.record_outcome('completed' if {exit_code} == 0 else 'failed')
ns['_run_recorder'] = recorder
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
    assert result.returncode == (exit_code or 1), result.stderr
    path = tmp_path / "run_test/stages/run.jsonl"
    start, end = [json.loads(line) for line in path.read_text().splitlines()]
    assert end["is_terminal"] and end["status"] == ("completed" if exit_code == 0 else "failed")


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
        "os": os,
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
        "os": os,
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
        "os": os,
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
    assert call.kwargs["args"] == (1,)
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


@pytest.mark.parametrize(
    "failed_phase", ["quiesce_serve_routers", "shutdown_serve_router_loops", "shutdown_async_loop"]
)
@pytest.mark.parametrize("exit_code", [0, 143])
def test_train_does_not_destroy_ray_or_report_success_when_threads_cannot_stop(failed_phase, exit_code):
    ns = _train_namespace()
    ns["threading"] = Mock()
    ns["_hard_exit"] = Mock()
    ns[failed_phase].side_effect = TimeoutError("owned thread did not stop")
    ns["_graceful_shutdown"](exit_code=exit_code)
    ns["ray"].shutdown.assert_not_called()
    ns["_hard_exit"].assert_called_once_with(exit_code or 1)


def test_train_stops_polls_before_serve_and_joins_threads_before_ray():
    ns = _train_namespace()
    events = []
    ns["threading"] = Mock()
    ns["quiesce_serve_routers"].side_effect = lambda: events.append("polls_stopped") or []
    ns["serve"].shutdown.side_effect = lambda: events.append("serve_shutdown")
    ns["shutdown_serve_router_loops"].side_effect = lambda _owners: events.append("serve_threads_joined")
    ns["shutdown_async_loop"].side_effect = lambda: events.append("async_thread_joined")
    ns["ray"].shutdown.side_effect = lambda: events.append("ray_shutdown")
    ns["_hard_exit"] = lambda code: events.append(code)
    ns["_graceful_shutdown"](exit_code=0)
    assert events == [
        "polls_stopped",
        "serve_shutdown",
        "serve_threads_joined",
        "async_thread_joined",
        "ray_shutdown",
        0,
    ]


def test_controller_shutdown_error_still_cleans_ray_and_returns_failure():
    ns = _train_namespace()
    ns["threading"] = Mock()
    ns["_hard_exit"] = Mock()
    ns["_ctrl"] = Mock()
    ns["_ctrl"].shutdown.side_effect = RuntimeError("controller cleanup failed")
    ns["_graceful_shutdown"](exit_code=0)
    ns["serve"].shutdown.assert_called_once()
    ns["ray"].shutdown.assert_called_once()
    ns["_hard_exit"].assert_called_once_with(1)


@pytest.mark.parametrize("mode", ["missing_config", "text_sft", "i2i", "i2i_stop_failure"])
def test_controller_partial_startup_cleans_teacher_and_routers(monkeypatch, mode):
    tree = ast.parse((ROOT / "relax/core/controller.py").read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "Controller")
    methods = [
        node
        for node in cls.body
        if isinstance(node, ast.FunctionDef) and node.name in {"shutdown", "_dispose_rollout_manager"}
    ]
    stop_routers = Mock(return_value=1)
    monkeypatch.setitem(
        sys.modules, "relax.distributed.ray.rollout", SimpleNamespace(stop_launched_routers=stop_routers)
    )
    stop_teacher = Mock()
    ns = {
        "logger": Mock(),
        "ROLES": SimpleNamespace(rollout="rollout", sft="sft"),
        "shutdown_managed_opd_teacher": stop_teacher,
        "asyncio": asyncio,
        "run": asyncio.run,
    }
    exec(compile(ast.Module(body=methods, type_ignores=[]), "controller.py", "exec"), ns)
    teacher = object()
    partial_controller = SimpleNamespace(
        serve_dict={},
        _teacher_manager=teacher,
        _shutdown_agentic_rollout_services=Mock(),
        _close_data_system=Mock(),
        _cleanup_s3_model_weights_after_init=Mock(),
    )
    partial_controller._dispose_rollout_manager = ns["_dispose_rollout_manager"].__get__(partial_controller)
    stop_image = AsyncMock(side_effect=RuntimeError("producer unavailable") if mode == "i2i_stop_failure" else None)
    if mode != "missing_config":
        partial_controller.config = SimpleNamespace(
            loss_type="sft", generation_task="i2i" if mode.startswith("i2i") else "t2i"
        )
        partial_controller.serve_dict["sft"] = SimpleNamespace(
            handle=SimpleNamespace(stop=SimpleNamespace(remote=stop_image))
        )
    ns["shutdown"](partial_controller)
    if mode.startswith("i2i"):
        stop_image.assert_awaited_once()
    else:
        stop_image.assert_not_called()
    stop_teacher.assert_called_once_with(teacher)
    stop_routers.assert_called_once_with()
    partial_controller._close_data_system.assert_called_once_with()
    partial_controller._shutdown_agentic_rollout_services.assert_called_once()
    partial_controller._cleanup_s3_model_weights_after_init.assert_called_once_with(force=True)


@pytest.mark.parametrize("health", [False, True])
@pytest.mark.parametrize("outcome", ["success", "failure", "signal", "atexit"])
def test_tracking_finishes_only_after_confirmed_training_completion(health, outcome):
    ns = _train_namespace()
    events = []
    ns["_ctrl"] = SimpleNamespace(shutdown=Mock(), _metrics_service_enabled=True, _health_check_enabled=health)
    ns["_training_completed"] = outcome in {"success", "signal"}
    ns["_hard_exit"] = Mock()
    finish = ns["serve"].get_app_handle.return_value.finish_tracking.remote
    finish.side_effect = lambda: events.append("finish") or Mock(result=Mock(return_value={"status": "success"}))
    ns["serve"].shutdown.side_effect = lambda: events.append("shutdown")
    ns["_graceful_shutdown"](
        sig=signal.SIGTERM if outcome == "signal" else None,
        exit_code={"success": 0, "failure": 1, "signal": 0, "atexit": None}[outcome],
    )
    assert events == (["finish", "shutdown"] if outcome == "success" else ["shutdown"])


@pytest.mark.parametrize("manager_fails", [False, True])
def test_controller_shutdown_cleans_routers_in_both_processes(monkeypatch, manager_fails):
    tree = ast.parse((ROOT / "relax/core/controller.py").read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "Controller")
    methods = [
        node
        for node in cls.body
        if isinstance(node, ast.FunctionDef) and node.name in {"shutdown", "_dispose_rollout_manager"}
    ]
    stop_routers = Mock(return_value=1)
    monkeypatch.setitem(
        sys.modules, "relax.distributed.ray.rollout", SimpleNamespace(stop_launched_routers=stop_routers)
    )
    dispose_ref = object()
    dispose_remote = Mock(return_value=dispose_ref)
    manager = SimpleNamespace(dispose=SimpleNamespace(remote=dispose_remote))
    service = SimpleNamespace(get_rollout_manager=AsyncMock(return_value=manager))
    ray_get = Mock(side_effect=RuntimeError("Manager unavailable") if manager_fails else None)
    ns = {
        "logger": Mock(),
        "ROLES": SimpleNamespace(rollout="rollout"),
        "asyncio": asyncio,
        "run": asyncio.run,
        "ray": SimpleNamespace(get=ray_get),
        "shutdown_managed_opd_teacher": Mock(),
    }
    exec(compile(ast.Module(body=methods, type_ignores=[]), "controller.py", "exec"), ns)
    controller = SimpleNamespace(
        serve_dict={"rollout": service},
        _teacher_manager=None,
        _shutdown_agentic_rollout_services=Mock(),
        _close_data_system=Mock(),
        _cleanup_s3_model_weights_after_init=Mock(),
    )
    controller._dispose_rollout_manager = ns["_dispose_rollout_manager"].__get__(controller)

    ns["shutdown"](controller)

    dispose_remote.assert_called_once_with()
    ray_get.assert_called_once_with(dispose_ref, timeout=35.0)
    controller._close_data_system.assert_called_once_with()
    stop_routers.assert_called_once_with()
