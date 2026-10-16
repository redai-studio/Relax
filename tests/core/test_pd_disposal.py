# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import asyncio
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from tests.core.controller_test_utils import load_controller_with_stubbed_dependencies


@pytest.fixture
def controller_module():
    return load_controller_with_stubbed_dependencies("_test_pd_disposal_controller")


@pytest.mark.parametrize("fails", [False, True])
def test_controller_bounds_owner_lookup_and_disposal(monkeypatch, controller_module, fails):
    controller = object.__new__(controller_module.Controller)
    manager = SimpleNamespace(dispose=SimpleNamespace(remote=Mock(return_value="dispose-ref")))
    service = SimpleNamespace(get_rollout_manager=AsyncMock(return_value=manager))
    controller.serve_dict = {controller_module.ROLES.rollout: service}
    get = Mock(side_effect=TimeoutError("owner unavailable") if fails else None)
    monkeypatch.setattr(controller_module.ray, "get", get)
    monkeypatch.setattr(controller_module, "run", asyncio.run)

    assert controller._dispose_rollout_manager() is (not fails)
    get.assert_called_once_with("dispose-ref", timeout=35.0)
    service.get_rollout_manager.assert_awaited_once()


def test_controller_owner_lookup_timeout_is_bounded(monkeypatch, controller_module):
    controller = object.__new__(controller_module.Controller)

    async def unavailable():
        await asyncio.Event().wait()

    controller.serve_dict = {controller_module.ROLES.rollout: SimpleNamespace(get_rollout_manager=unavailable)}
    monkeypatch.setattr(controller_module, "run", asyncio.run)
    assert not controller._dispose_rollout_manager(lookup_timeout=0.001)


def test_controller_shutdown_disposes_rollout_owner(monkeypatch, controller_module):
    controller = object.__new__(controller_module.Controller)
    events = []
    manager = SimpleNamespace(dispose=SimpleNamespace(remote=Mock(return_value="dispose-ref")))
    service = SimpleNamespace(get_rollout_manager=AsyncMock(return_value=manager))
    controller.serve_dict = {controller_module.ROLES.rollout: service}
    controller._teacher_manager = None
    controller._shutdown_agentic_rollout_services = lambda: events.append("agentic")
    controller._cleanup_s3_model_weights_after_init = Mock()
    monkeypatch.setattr(controller_module, "run", asyncio.run)
    get = Mock(side_effect=lambda *args, **kwargs: events.append("dispose"))
    monkeypatch.setattr(controller_module.ray, "get", get)
    monkeypatch.setattr(controller_module, "shutdown_managed_opd_teacher", lambda _: events.append("teacher"))
    monkeypatch.setitem(
        sys.modules,
        "relax.distributed.ray.rollout",
        SimpleNamespace(stop_launched_routers=lambda: events.append("routers") or 0),
    )

    controller.shutdown()

    manager.dispose.remote.assert_called_once()
    get.assert_called_once_with("dispose-ref", timeout=35.0)
    assert events == ["dispose", "teacher", "agentic", "routers"]


def test_global_restart_disposes_owner_before_deleting_deployments(monkeypatch, controller_module):
    controller = object.__new__(controller_module.Controller)
    controller._global_restart_count = 0
    controller._max_global_restart = 1
    controller.config = SimpleNamespace()
    controller.runtime_env = {}
    controller._health_manager = SimpleNamespace(_checker=None, stop=Mock())
    controller.serve_dict = {controller_module.ROLES.rollout: SimpleNamespace(_stop_heartbeat_thread=Mock())}
    events = []
    controller._cancel_pending_tasks = lambda: events.append("cancel")
    controller._dispose_rollout_manager = lambda: events.append("dispose")
    monkeypatch.setattr(controller_module, "recovery_load_path", lambda config: None)

    class ReachedDeletion(BaseException):
        pass

    def delete(role):
        events.append("delete")
        raise ReachedDeletion

    monkeypatch.setattr(controller_module.serve, "delete", delete)
    with pytest.raises(ReachedDeletion):
        controller._run_global_restart()
    assert events == ["cancel", "dispose", "delete"]
