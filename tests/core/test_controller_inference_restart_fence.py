# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import ast
import asyncio
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest
from ray.exceptions import RayActorError

from relax.distributed.ray.inference_manager import InferenceRecoveryRequired


@pytest.fixture
def restart_controller():
    # Execute the production methods without importing the optional SGLang,
    # Torch, or training runtime. Only transport and unrelated teardown are mocks.
    path = Path(__file__).parents[2] / "relax/core/controller.py"
    tree = ast.parse(path.read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "Controller")
    method_names = {"_confirm_inference_cleanup_for_restart", "_global_restart", "_run_global_restart"}
    methods = [node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name in method_names]
    events = []

    def get(ref, *, timeout):
        assert timeout == 180.0
        label, method, outcome = ref
        events.append(("cleanup", label, method))
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    ray = SimpleNamespace(get=MagicMock(side_effect=get), shutdown=MagicMock(), init=MagicMock())
    serve = SimpleNamespace(
        delete=MagicMock(side_effect=lambda role: events.append(("delete", role))),
        shutdown=MagicMock(),
        start=MagicMock(),
    )
    namespace = {
        "Any": Any,
        "asyncio": SimpleNamespace(wait_for=asyncio.wait_for),
        "ray": ray,
        "serve": serve,
        "run": asyncio.run,
        "ROLES": SimpleNamespace(rollout="rollout"),
        "GENRM_ROLE": "genrm",
        "InferenceRecoveryRequired": InferenceRecoveryRequired,
        "logger": MagicMock(),
        "recovery_load_path": lambda config: None,
        "shutdown_managed_opd_teacher": MagicMock(side_effect=lambda _: events.append(("kill_teacher",))),
        "tq": SimpleNamespace(close=MagicMock()),
        "shutdown_async_loop": MagicMock(side_effect=RuntimeError("stop before Ray teardown")),
    }
    exec(compile(ast.Module(body=methods, type_ignores=[]), str(path), "exec"), namespace)
    controller_type = type("ControllerMethods", (), {name: namespace[name] for name in method_names})
    instance = controller_type()
    instance.config = SimpleNamespace(_genrm_instances_resolved={"math": {}, "code": {}})
    instance.runtime_env = None
    instance._genrm_shutdown_complete = set()
    instance._global_restart_count = 0
    instance._max_global_restart = 3
    instance._health_manager = SimpleNamespace(_checker=None, stop=MagicMock())
    instance._cancel_pending_tasks = MagicMock()
    instance._restart_done_event = threading.Event()
    instance._restart_error = None
    instance._shutdown_teacher_gateway = MagicMock(side_effect=lambda: events.append(("delete_gateway",)))
    instance._shutdown_genrm_managers = MagicMock(side_effect=lambda: events.append(("kill_genrm",)))
    instance._shutdown_inference_coordinator = MagicMock(side_effect=lambda: events.append(("kill_coordinator",)))
    instance._shutdown_agentic_rollout_services = MagicMock()
    instance._metrics_service_enabled = False
    instance._autoscaler_config = None
    instance._owned_actor_inference_pg = object()

    def manager(label, *, outcome=None):
        handle = SimpleNamespace()
        for method in ("shutdown", "dispose"):
            setattr(handle, method, SimpleNamespace(remote=MagicMock(return_value=(label, method, outcome))))
        return handle

    handles = {key: manager(key) for key in ("rollout", "teacher", "math", "code")}

    async def get_rollout():
        return handles["rollout"]

    async def get_genrm(key):
        return handles[key]

    instance.serve_dict = {
        "rollout": SimpleNamespace(get_rollout_manager=get_rollout, _stop_heartbeat_thread=MagicMock()),
        "genrm": SimpleNamespace(get_genrm_manager=get_genrm, _stop_heartbeat_thread=MagicMock()),
    }
    instance._teacher_manager = handles["teacher"]
    instance._genrm_shutdown_managers = {key: handles[key] for key in ("math", "code")}
    return SimpleNamespace(instance=instance, handles=handles, manager=manager, events=events, namespace=namespace)


@pytest.mark.parametrize("teacher_list", [False, True])
def test_restart_cleanup_confirms_all_roles_and_keeps_owners(restart_controller, teacher_list):
    state = restart_controller
    controller = state.instance
    if teacher_list:
        controller._teacher_manager = [state.handles["teacher"], state.manager("teacher2")]
    controller._genrm_shutdown_managers.clear()

    controller._confirm_inference_cleanup_for_restart()

    labels = ["rollout", "teacher", *(["teacher2"] if teacher_list else []), "math", "code"]
    assert state.events == [("cleanup", key, "dispose" if key == "rollout" else "shutdown") for key in labels]
    assert controller._genrm_shutdown_managers == {key: state.handles[key] for key in ("math", "code")}
    controller._shutdown_teacher_gateway.assert_not_called()
    controller._shutdown_genrm_managers.assert_not_called()


@pytest.mark.parametrize("role", ["rollout", "teacher", "math"])
@pytest.mark.parametrize("outcome", [False, TimeoutError("cleanup timeout"), RayActorError("terminal actor death")])
def test_restart_cleanup_aggregates_failure_after_attempting_all_roles(restart_controller, role, outcome):
    state = restart_controller
    method = "dispose" if role == "rollout" else "shutdown"
    getattr(state.handles[role], method).remote.return_value = (role, method, outcome)

    with pytest.raises(InferenceRecoveryRequired, match=role):
        state.instance._confirm_inference_cleanup_for_restart()

    assert [event[1] for event in state.events] == ["rollout", "teacher", "math", "code"]
    assert state.instance._genrm_shutdown_managers["math"] is state.handles["math"]


@pytest.mark.parametrize("role", ["rollout", "genrm"])
@pytest.mark.parametrize("outcome", ["missing", "error", "timeout"])
def test_restart_cleanup_cannot_ignore_manager_discovery_failure(restart_controller, role, outcome):
    state = restart_controller
    controller = state.instance
    controller._genrm_shutdown_managers.clear()

    async def discover(*args):
        if outcome == "error":
            raise RuntimeError("discovery failed")
        if outcome == "timeout":
            await asyncio.Future()
        return None

    async def bounded_wait(awaitable, *, timeout):
        assert timeout == 30.0
        return await asyncio.wait_for(awaitable, timeout=0.001)

    state.namespace["asyncio"].wait_for = bounded_wait
    method = "get_rollout_manager" if role == "rollout" else "get_genrm_manager"
    setattr(controller.serve_dict[role], method, discover)

    with pytest.raises(InferenceRecoveryRequired, match="rollout" if role == "rollout" else "GenRM"):
        controller._confirm_inference_cleanup_for_restart()

    assert ("cleanup", "teacher", "shutdown") in state.events
    if role == "rollout":
        assert ("cleanup", "code", "shutdown") in state.events
    else:
        assert ("cleanup", "rollout", "dispose") in state.events


def test_restart_cleanup_aggregates_missing_owners_instead_of_assuming_disabled_roles(restart_controller):
    controller = restart_controller.instance
    controller._teacher_manager = None
    controller.config._managed_opd_teacher_model_ids = ("teacher",)
    controller.serve_dict.pop("genrm")
    controller._genrm_shutdown_managers.clear()

    with pytest.raises(InferenceRecoveryRequired) as caught:
        controller._confirm_inference_cleanup_for_restart()

    assert "teacher" in str(caught.value)
    assert "GenRM[math]" in str(caught.value)
    assert "GenRM[code]" in str(caught.value)


def test_restart_cleanup_skips_only_explicitly_completed_genrm_owners(restart_controller):
    controller = restart_controller.instance
    controller._genrm_shutdown_complete.add("math")
    controller._genrm_shutdown_managers.pop("math")

    controller._confirm_inference_cleanup_for_restart()

    assert [event[1] for event in restart_controller.events] == ["rollout", "teacher", "code"]


def test_restart_cleanup_rejects_undiscoverable_genrm_service(restart_controller):
    controller = restart_controller.instance
    controller.config._genrm_instances_resolved = {}
    controller._genrm_shutdown_managers.clear()

    with pytest.raises(InferenceRecoveryRequired, match="without recorded manager identities"):
        controller._confirm_inference_cleanup_for_restart()


def test_restart_cleanup_continues_past_missing_teacher_in_list(restart_controller):
    state = restart_controller
    state.instance._teacher_manager = [None, state.handles["teacher"]]

    with pytest.raises(InferenceRecoveryRequired, match=r"teacher\[0\]"):
        state.instance._confirm_inference_cleanup_for_restart()

    assert [event[1] for event in state.events] == ["rollout", "teacher", "math", "code"]


def test_global_restart_unconfirmed_backend_preserves_fence_and_signals_main_thread(restart_controller):
    state = restart_controller
    controller = state.instance
    state.handles["math"].shutdown.remote.side_effect = RayActorError("manager died")
    owner_pg = controller._owned_actor_inference_pg
    services = dict(controller.serve_dict)

    controller._global_restart()

    assert isinstance(controller._restart_error, InferenceRecoveryRequired)
    assert "operator" in str(controller._restart_error)
    assert controller._restart_done_event.is_set()
    assert controller._cancel_pending_tasks.call_count == 2
    assert controller._owned_actor_inference_pg is owner_pg
    assert controller.serve_dict == services
    assert controller._genrm_shutdown_managers["math"] is state.handles["math"]
    controller._shutdown_teacher_gateway.assert_not_called()
    controller._shutdown_genrm_managers.assert_not_called()
    controller._shutdown_inference_coordinator.assert_not_called()
    state.namespace["shutdown_managed_opd_teacher"].assert_not_called()
    state.namespace["serve"].delete.assert_not_called()
    state.namespace["serve"].shutdown.assert_not_called()
    state.namespace["ray"].shutdown.assert_not_called()
    state.namespace["ray"].init.assert_not_called()


def test_global_restart_confirms_backend_cleanup_before_deleting_any_owner(restart_controller):
    state = restart_controller

    with pytest.raises(RuntimeError, match="stop before Ray teardown"):
        state.instance._run_global_restart()

    assert state.events[:4] == [
        ("cleanup", "rollout", "dispose"),
        ("cleanup", "teacher", "shutdown"),
        ("cleanup", "math", "shutdown"),
        ("cleanup", "code", "shutdown"),
    ]
    assert state.events[4] == ("delete_gateway",)
    assert ("delete", "genrm") in state.events
