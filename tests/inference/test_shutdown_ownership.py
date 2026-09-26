# Copyright (c) 2026 Relax Authors. All Rights Reserved.


from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest


@pytest.fixture
def controller(monkeypatch):
    pytest.importorskip("sglang")

    from relax.core import controller as module
    from relax.distributed.ray import rollout

    instance = object.__new__(module.Controller)
    instance.config = SimpleNamespace(_genrm_instances_resolved={"math": {}, "code": {}}, use_agentic_rollout=False)
    instance.serve_dict = {}
    instance._teacher_manager = None
    instance._owned_actor_inference_pg = object()
    managers = {key: MagicMock() for key in instance.config._genrm_instances_resolved}
    for key, manager in managers.items():
        manager.shutdown.remote.return_value = key
    instance._genrm_shutdown_managers = dict(managers)
    instance.stop_health_check = lambda: None
    instance._cleanup_s3_model_weights_after_init = lambda **kwargs: None
    events = []

    def get(ref, *, timeout):
        assert timeout > 0
        events.append(("shutdown", ref))

    monkeypatch.setattr(module.ray, "get", get)
    monkeypatch.setattr(module.ray, "kill", lambda handle: events.append(("kill", handle)))
    monkeypatch.setattr(module, "shutdown_managed_opd_teacher", lambda handle: None)
    monkeypatch.setattr(rollout, "stop_launched_routers", lambda: 0)
    return module, instance, managers, events


def test_genrm_shutdown_confirms_engines_before_killing_managers_and_is_idempotent(controller):
    _module, instance, managers, events = controller

    instance._shutdown_genrm_managers()
    instance._shutdown_genrm_managers()

    assert events == [
        ("shutdown", "math"),
        ("kill", managers["math"]),
        ("shutdown", "code"),
        ("kill", managers["code"]),
    ]
    assert instance._genrm_shutdown_managers == {}


def test_genrm_failed_child_cleanup_does_not_kill_owner_and_can_retry(controller, monkeypatch):
    module, instance, managers, events = controller

    def fail(ref, **kwargs):
        raise TimeoutError("child shutdown unconfirmed")

    monkeypatch.setattr(module.ray, "get", fail)
    instance._shutdown_genrm_managers()
    assert events == []
    assert instance._genrm_shutdown_managers == managers
    monkeypatch.setattr(module.ray, "get", lambda ref, **kwargs: None)
    instance._shutdown_genrm_managers()
    assert events == [("kill", managers["math"]), ("kill", managers["code"])]


def test_genrm_shutdown_discovers_manager_before_deleting_ingress(controller):
    _module, instance, managers, _events = controller
    instance._genrm_shutdown_managers = {}
    calls = []

    async def get_manager(key):
        calls.append(key)
        return managers[key]

    instance.serve_dict["genrm"] = SimpleNamespace(get_genrm_manager=get_manager)

    instance._shutdown_genrm_managers()

    assert calls == ["math", "code"]


def test_controller_shutdown_retains_actor_pg_until_training_teardown(controller, monkeypatch):
    module, instance, managers, events = controller
    pg = instance._owned_actor_inference_pg
    coordinator = object()
    instance.config._inference_coordinator = coordinator
    remove = MagicMock()
    monkeypatch.setattr(module.ray.util, "remove_placement_group", remove)

    instance.shutdown()

    assert instance._owned_actor_inference_pg is pg
    remove.assert_not_called()
    assert instance.config._inference_coordinator is None
    assert events[-1] == ("kill", coordinator)
    assert all(("kill", manager) in events for manager in managers.values())


def test_coordinator_kill_failure_retains_handle_for_retry(controller, monkeypatch):
    module, instance, _managers, _events = controller
    coordinator = object()
    instance.config._inference_coordinator = coordinator
    monkeypatch.setattr(module.ray, "kill", MagicMock(side_effect=[RuntimeError("not confirmed"), None]))

    instance._shutdown_inference_coordinator()
    assert instance.config._inference_coordinator is coordinator
    instance._shutdown_inference_coordinator()
    assert instance.config._inference_coordinator is None


def test_global_restart_cleans_genrm_and_coordinator_before_serve_delete(controller, monkeypatch):
    module, instance, managers, events = controller
    coordinator = object()
    instance.config._inference_coordinator = coordinator
    instance.runtime_env = None
    instance._global_restart_count = 0
    instance._max_global_restart = 3
    instance._health_manager = SimpleNamespace(_checker=None, stop=lambda **kwargs: None)
    instance._cancel_pending_tasks = lambda: None
    instance._metrics_service_enabled = False
    instance._autoscaler_config = None
    instance.serve_dict["genrm"] = SimpleNamespace(_stop_heartbeat_thread=lambda: None)
    monkeypatch.setattr(module, "recovery_load_path", lambda config: None)
    monkeypatch.setattr(module.serve, "delete", lambda role: events.append(("delete", role)))
    monkeypatch.setattr(module.tq, "close", lambda: None)

    def stop():
        raise RuntimeError("test stopped before Ray shutdown")

    monkeypatch.setattr(module, "shutdown_async_loop", stop)
    with pytest.raises(RuntimeError, match="test stopped"):
        instance._run_global_restart()

    ingress_index = events.index(("delete", "genrm"))
    assert events.index(("kill", coordinator)) < ingress_index
    assert all(events.index(("kill", manager)) < ingress_index for manager in managers.values())
