# Copyright (c) 2026 Relax Authors. All Rights Reserved.


import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest


@pytest.fixture
def opd_utils():
    from relax.utils.opd import opd_utils

    return opd_utils


def test_teacher_gateway_uses_recorded_manager_identity_order(opd_utils):
    math_manager, code_manager = object(), object()
    args = SimpleNamespace(
        _managed_opd_teacher_model_ids=("math", "code"),
        opd_teacher_routes=json.dumps({"code": "/code", "math": "/math"}),
    )

    result = opd_utils.managed_opd_teacher_managers_by_model(args, [math_manager, code_manager])

    assert result == {"math": math_manager, "code": code_manager}


@pytest.mark.parametrize("identities", [(), ("math",), ("math", "math"), ("", "code")])
def test_teacher_gateway_rejects_missing_or_ambiguous_identity_metadata(opd_utils, identities):
    args = SimpleNamespace(_managed_opd_teacher_model_ids=identities)

    with pytest.raises(ValueError, match="identit"):
        opd_utils.managed_opd_teacher_managers_by_model(args, [object(), object()])


def test_single_teacher_launcher_records_identity_and_preserves_generate_urls(opd_utils, monkeypatch):
    args = SimpleNamespace(
        use_opd=True,
        opd_type="sglang",
        teacher_hf_checkpoint="/teacher",
        resource={"teacher": [1, 2]},
    )
    manager = object()
    monkeypatch.setattr(
        opd_utils,
        "create_managed_opd_teacher_manager",
        lambda *args, **kwargs: (manager, ["http://teacher.example:15000/generate"]),
    )

    pg, legacy_handle = opd_utils.maybe_start_managed_opd_teacher(args)

    assert pg is None
    assert legacy_handle is manager
    assert args.opd_teacher_url == "http://teacher.example:15000/generate"
    assert args.opd_teacher_urls == [args.opd_teacher_url]
    assert opd_utils.managed_opd_teacher_managers_by_model(args, legacy_handle) == {"__default__": manager}


def test_disabled_teacher_clears_restart_identity_metadata(opd_utils):
    args = SimpleNamespace(_managed_opd_teacher_model_ids=("stale",))

    assert opd_utils.maybe_start_managed_opd_teacher(args) == (None, None)
    assert args._managed_opd_teacher_model_ids == ()
    assert opd_utils.managed_opd_teacher_managers_by_model(args, None) == {}


def test_multi_teacher_launcher_records_actual_created_manager_order(opd_utils, monkeypatch):
    pytest.importorskip("sglang")

    import ray

    from relax.core import service
    from relax.distributed.ray import multi_instance_orchestrator, teacher_manager

    monkeypatch.setattr(
        service, "create_placement_group", lambda **kwargs: ("actor-pg", list(range(8)), list(range(8)))
    )
    monkeypatch.setattr(teacher_manager, "TeacherManager", MagicMock())
    math_manager, code_manager = MagicMock(), MagicMock()
    math_manager.get_urls.remote.return_value = ["http://math.example/generate"]
    code_manager.get_urls.remote.return_value = ["http://code.example/generate"]
    created = {"math": math_manager, "code": code_manager}
    monkeypatch.setattr(multi_instance_orchestrator, "start_multi_instance_managers", lambda **kwargs: created)
    monkeypatch.setattr(ray, "get", lambda value, timeout=None: value)
    args = SimpleNamespace(
        use_opd=True,
        opd_type="sglang",
        colocate=True,
        hybrid=False,
        rollout_num_gpus=4,
        resource={"actor": [1, 8], "rollout": [1, 4], "teacher": [1, 4]},
        opd_teacher_routes=json.dumps({"code": "/code", "math": "/math"}),
    )

    _pg, legacy_handles = opd_utils.maybe_start_managed_opd_teacher(args)

    assert legacy_handles == [math_manager, code_manager]
    assert args._managed_opd_teacher_model_ids == ("math", "code")
    assert opd_utils.managed_opd_teacher_managers_by_model(args, legacy_handles) == created
    assert args.opd_teacher_routes_map == {
        "math": ["http://math.example/generate"],
        "code": ["http://code.example/generate"],
    }


@pytest.fixture
def teacher_controller(monkeypatch):
    from relax.core import controller as controller_module

    instance = object.__new__(controller_module.Controller)
    instance._test_module = controller_module
    teacher = object()
    instance.config = SimpleNamespace(
        _managed_opd_teacher_model_ids=("__default__",),
        _relax_control_plane_node_group="stable",
        enable_affinity=True,
    )
    instance.runtime_env = {"env_vars": {"TASK3_TEST": "1"}}
    instance._teacher_manager = teacher
    instance.serve_dict = {}
    gateway = MagicMock()
    monkeypatch.setattr(controller_module, "InferenceGateway", gateway)
    run = MagicMock()
    delete = MagicMock()
    monkeypatch.setattr(controller_module.serve, "run", run)
    monkeypatch.setattr(controller_module.serve, "delete", delete)
    return instance, gateway, run, delete, teacher


def test_teacher_gateway_deploys_once_on_cpu_without_gpu_placement(teacher_controller):
    instance, gateway, run, _delete, teacher = teacher_controller

    instance._deploy_teacher_gateway()
    instance._deploy_teacher_gateway()

    assert gateway.options.call_args.kwargs["ray_actor_options"] == {
        "num_cpus": 1,
        "num_gpus": 0,
        "runtime_env": instance.runtime_env,
        "resources": {"stable_cpu": 1},
    }
    gateway.options.return_value.bind.assert_called_once_with(role="teacher", managers={"__default__": teacher})
    run.assert_called_once_with(
        gateway.options.return_value.bind.return_value, name="teacher", route_prefix="/teacher"
    )


def test_teacher_gateway_skips_disabled_role_with_minimal_config(teacher_controller):
    instance, gateway, run, _delete, _teacher = teacher_controller
    instance.config = SimpleNamespace()
    instance._teacher_manager = None

    instance._deploy_teacher_gateway()

    gateway.options.assert_not_called()
    run.assert_not_called()


def test_teacher_gateway_does_not_replace_existing_role_application(teacher_controller):
    instance, _gateway, run, _delete, _teacher = teacher_controller
    instance.serve_dict["teacher"] = object()

    with pytest.raises(RuntimeError, match="already"):
        instance._deploy_teacher_gateway()

    run.assert_not_called()


def test_teacher_gateway_deployment_failure_cleans_partial_application(teacher_controller):
    instance, _gateway, run, delete, _teacher = teacher_controller
    run.side_effect = RuntimeError("deployment failed")

    with pytest.raises(RuntimeError, match="deployment failed"):
        instance._deploy_teacher_gateway()

    delete.assert_called_once_with("teacher")
    assert not instance._teacher_gateway_owned


def test_teacher_gateway_shutdown_is_idempotent_and_owner_only(teacher_controller):
    instance, _gateway, _run, delete, _teacher = teacher_controller

    instance._shutdown_teacher_gateway()
    delete.assert_not_called()
    instance._deploy_teacher_gateway()
    instance._shutdown_teacher_gateway()
    instance._shutdown_teacher_gateway()

    delete.assert_called_once_with("teacher")
    assert instance._teacher_gateway is None


def test_teacher_gateway_failed_delete_retains_ownership_for_retry(teacher_controller):
    instance, _gateway, _run, delete, _teacher = teacher_controller
    instance._deploy_teacher_gateway()
    delete.side_effect = [RuntimeError("unavailable"), None]

    instance._shutdown_teacher_gateway()
    assert instance._teacher_gateway_owned
    instance._shutdown_teacher_gateway()

    assert not instance._teacher_gateway_owned
    assert delete.call_count == 2


def test_controller_shutdown_removes_gateway_before_teacher_engines(teacher_controller, monkeypatch):
    pytest.importorskip("sglang")

    instance, _gateway, _run, delete, teacher = teacher_controller
    module = instance._test_module
    events = []
    instance._deploy_teacher_gateway()
    instance.stop_health_check = lambda: None
    instance._shutdown_agentic_rollout_services = lambda: None
    instance._cleanup_s3_model_weights_after_init = lambda **kwargs: None
    delete.side_effect = lambda name: events.append(("delete", name))
    monkeypatch.setattr(module, "shutdown_managed_opd_teacher", lambda handle: events.append(("engines", handle)))
    from relax.distributed.ray import rollout

    monkeypatch.setattr(rollout, "stop_launched_routers", lambda: 0)

    instance.shutdown()

    assert events == [("delete", "teacher"), ("engines", teacher)]


def test_global_restart_removes_teacher_gateway_before_ray_shutdown(teacher_controller, monkeypatch):
    instance, _gateway, _run, delete, teacher = teacher_controller
    module = instance._test_module
    events = []
    instance._deploy_teacher_gateway()
    instance._global_restart_count = 0
    instance._max_global_restart = 3
    instance._health_manager = SimpleNamespace(_checker=None, stop=lambda **kwargs: None)
    instance._cancel_pending_tasks = lambda: None
    instance._shutdown_agentic_rollout_services = lambda: None
    instance._metrics_service_enabled = False
    instance._autoscaler_config = None
    delete.side_effect = lambda name: events.append(("delete", name))
    monkeypatch.setattr(module, "shutdown_managed_opd_teacher", lambda handle: events.append(("engines", handle)))
    monkeypatch.setattr(module, "recovery_load_path", lambda config: None)
    monkeypatch.setattr(module.tq, "close", lambda: None)

    def stop_before_ray_shutdown():
        raise RuntimeError("end teardown assertion")

    monkeypatch.setattr(module, "shutdown_async_loop", stop_before_ray_shutdown)
    with pytest.raises(RuntimeError, match="end teardown assertion"):
        instance._run_global_restart()

    assert events[:2] == [("delete", "teacher"), ("engines", teacher)]
