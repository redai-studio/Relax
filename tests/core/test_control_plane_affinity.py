# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import ast
import sys
from argparse import Namespace
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock

import pytest
import transfer_queue

from relax.distributed.ray import placement_group as placement_group_module
from relax.utils import health_system
from tests.core.controller_test_utils import load_controller_with_stubbed_dependencies


def _elastic_args(tmp_path, **kwargs):
    config_path = tmp_path / "autoscaler.yaml"
    config_path.write_text("enabled: true\n")
    values = {"autoscaler_config": str(config_path), "enable_affinity": True}
    values.update(kwargs)
    return Namespace(**values)


class _FakeActorClass:
    options_seen = None

    @classmethod
    def options(cls, **options):
        cls.options_seen = options
        return cls

    @classmethod
    def remote(cls, *args, **kwargs):
        return MagicMock()


def test_health_status_requests_stable_cpu(monkeypatch, tmp_path):
    fake = type("FakeHealthStatus", (_FakeActorClass,), {})
    monkeypatch.setattr(health_system, "HealthStatus", fake)

    health_system.HealthManager(config=_elastic_args(tmp_path))

    assert fake.options_seen == {"resources": {"stable_cpu": 1}}


def test_dcs_preserves_base_options_and_stable_cpu(monkeypatch):
    try:
        from relax.distributed.checkpoint_service.coordinator import service as dcs_service
    except ImportError as exc:
        pytest.skip(f"DCS optional dependencies are unavailable: {exc}")

    fake = type("FakeDCS", (_FakeActorClass,), {"bind": classmethod(lambda cls, **kwargs: "deployment")})
    monkeypatch.setattr(dcs_service, "DCSCoordinator", fake)
    monkeypatch.setattr(dcs_service.serve, "run", lambda *args, **kwargs: MagicMock())

    dcs_service.create_dcs_deployment(
        ray_actor_options={"num_cpus": 1, "resources": {"stable_cpu": 1}},
    )

    assert fake.options_seen == {"ray_actor_options": {"num_cpus": 1, "resources": {"stable_cpu": 1}}}


def test_rollout_data_source_requests_stable_cpu(monkeypatch, tmp_path):
    if not hasattr(transfer_queue, "StreamingTokenBudgetSampler"):
        pytest.skip("controller requires a TransferQueue build with StreamingTokenBudgetSampler")

    controller_module = load_controller_with_stubbed_dependencies("_test_control_plane_affinity_controller")

    captured = {}

    class FakeRemoteClass:
        def options(self, **options):
            captured["options"] = options
            return self

        def remote(self, config):
            captured["config"] = config
            return "data-source"

    monkeypatch.setattr(controller_module.ray, "remote", lambda **kwargs: lambda cls: FakeRemoteClass())
    args = _elastic_args(tmp_path)

    assert controller_module.create_data_source_actor(args, object) == "data-source"
    assert captured["options"] == {"resources": {"stable_cpu": 1}}
    assert captured["config"] is args


def test_rollout_worker_keeps_node_affinity_and_requests_matching_marker(monkeypatch, tmp_path):
    captured = {}
    node_id = "a" * 56
    rollout_module = ModuleType("relax.distributed.ray.rollout_worker")

    class FakeRolloutWorker(_FakeActorClass):
        @classmethod
        def options(cls, **options):
            captured["options"] = options
            return cls

    rollout_module.RolloutWorker = FakeRolloutWorker
    monkeypatch.setitem(sys.modules, "relax.distributed.ray.rollout_worker", rollout_module)
    monkeypatch.setattr(placement_group_module, "_get_head_node_id", lambda: node_id)
    monkeypatch.setattr(
        placement_group_module.ray,
        "nodes",
        lambda: [{"NodeID": node_id, "Alive": True, "Resources": {"stable_cpu": 8}}],
    )
    args = _elastic_args(
        tmp_path,
        loss_type="sft",
        num_rollout_per_epoch=1,
        check_weight_update_equal=False,
        offload_rollout=False,
    )

    owner = SimpleNamespace(create_rollout_role=SimpleNamespace(remote=lambda *a: {"router_ip": None}))
    monkeypatch.setattr(placement_group_module.ray, "get", lambda value: value)
    placement_group_module.create_rollout_worker(
        args, "pg", runtime_env={"env_vars": {"A": "B"}}, inference_manager_handle=owner
    )

    assert captured["options"]["resources"] == {"stable_cpu": 1}
    assert captured["options"]["num_cpus"] == 1
    assert captured["options"]["runtime_env"] == {"env_vars": {"A": "B"}}
    assert captured["options"]["scheduling_strategy"].node_id == node_id


def test_task_inference_manager_keeps_head_affinity_and_requests_stable_cpu(monkeypatch, tmp_path):
    """GenRM and Teacher engines live in the task InferenceManager, so it
    carries the stable-CPU marker and stays on the head node with the
    routers."""
    from relax.distributed.ray import inference_manager

    captured = {}
    node_id = "b" * 56

    class FakeManagerActor(_FakeActorClass):
        @classmethod
        def options(cls, **options):
            captured["options"] = options
            return cls

    monkeypatch.setattr(inference_manager, "InferenceManagerActor", FakeManagerActor)
    # Layout preflight imports the sglang-backed rollout module; this test only
    # checks the actor options, and CPU CI runs without sglang.
    monkeypatch.setattr(inference_manager, "validate_task_layout", lambda args: None)
    monkeypatch.setattr(placement_group_module, "_get_head_node_id", lambda: node_id)
    monkeypatch.setattr(
        placement_group_module.ray,
        "nodes",
        lambda: [{"NodeID": node_id, "Alive": True, "Resources": {"stable_cpu": 8}}],
    )

    inference_manager.create_inference_manager(_elastic_args(tmp_path), runtime_env={"env_vars": {"A": "B"}})

    assert captured["options"]["resources"] == {"stable_cpu": 1}
    assert captured["options"]["num_gpus"] == 0
    assert captured["options"]["runtime_env"] == {"env_vars": {"A": "B"}}
    assert captured["options"]["scheduling_strategy"].node_id == node_id


def _manager_handle():
    calls = []

    def record(name):
        def remote(*args, **kwargs):
            calls.append((name, args, kwargs))
            return tuple(config.name for config, _, _ in args[1]) if name == "create_role" else None

        return SimpleNamespace(remote=remote)

    return SimpleNamespace(create_role=record("create_role"), call=record("call")), calls


def test_genrm_instances_become_models_of_one_role(monkeypatch, tmp_path):
    spec = {
        "model_path": "/model",
        "num_gpus": 1,
        "num_gpus_per_engine": 1,
        "engine_config": {},
        "sampling_config": {},
    }
    args = _elastic_args(
        tmp_path,
        offload_rollout=True,
        fully_async=True,
        rollout_num_gpus=0,
        rollout_num_gpus_per_engine=1,
        num_gpus_per_node=8,
        _genrm_instances_resolved={"quality": dict(spec), "safety": dict(spec)},
    )
    manager, calls = _manager_handle()
    monkeypatch.setattr(placement_group_module.ray, "get", lambda refs, **kwargs: refs)

    assert placement_group_module.create_genrm_role(args, ("pg", [0, 1], [0, 1]), manager) == ("quality", "safety")

    assert [name for name, _, _ in calls] == ["create_role", "call", "call"]
    _, (role, models), _ = calls[0]
    assert role == "genrm"
    assert [placement["bundle_offset"] for _, _, placement in models] == [0, 1]
    assert len({placement["base_port"] for _, _, placement in models}) == 2
    assert models[0][1].genrm_model_path == "/model"
    assert [call[1] for call in calls[1:]] == [("genrm", "quality", "deactivate"), ("genrm", "safety", "deactivate")]


def test_dcs_proxy_requests_stable_cpu(monkeypatch, tmp_path):
    try:
        from relax.distributed.checkpoint_service.backends import device_direct
    except ImportError as exc:
        pytest.skip(f"DCS optional dependencies are unavailable: {exc}")

    captured = {}

    class FakeRolloutEngine(_FakeActorClass):
        @classmethod
        def options(cls, **options):
            captured["options"] = options
            return cls

    monkeypatch.setattr(device_direct, "RolloutEngine", FakeRolloutEngine)
    backend = object.__new__(device_direct.DeviceDirectBackend)
    backend.args = _elastic_args(tmp_path)
    backend.rollout_engines = {}

    backend._create_rollout_engines({0: {"ip": "127.0.0.1", "port": 8000}})

    assert captured["options"] == {"resources": {"stable_cpu": 1}}


def test_scale_out_paths_do_not_request_stable_markers():
    rollout_path = Path(placement_group_module.__file__).with_name("rollout.py")
    source = rollout_path.read_text()
    tree = ast.parse(source)
    target_names = {"_scale_out_ray_native", "_bring_up_single_replica", "_scale_out_external"}
    functions = {
        node.name: ast.get_source_segment(source, node) or ""
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in target_names
    }

    assert functions.keys() == target_names
    for function_source in functions.values():
        assert "stable_cpu" not in function_source
        assert "stable_gpu" not in function_source
        assert "with_control_plane_affinity" not in function_source
