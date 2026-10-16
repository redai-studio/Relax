# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import ast
import sys
from argparse import Namespace
from pathlib import Path
from types import ModuleType
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


@pytest.mark.parametrize(
    ("autoscaler_enabled", "expected_resource"),
    [(True, "stable_cpu"), (False, None)],
)
def test_transfer_queue_required_node_resource(
    monkeypatch,
    tmp_path,
    autoscaler_enabled,
    expected_resource,
):
    from relax.core import controller as controller_module

    config = _elastic_args(
        tmp_path,
        fully_async=False,
        use_dynamic_batch_size=False,
        balance_data=False,
        polling_mode=False,
        num_data_storage_units=2,
    )
    if not autoscaler_enabled:
        config.autoscaler_config = None
    captured = {}
    monkeypatch.setattr(controller_module, "resolve_sft_algo_key", lambda _: "dapo")
    monkeypatch.setattr(controller_module, "compute_dp_size", lambda _: 1)
    monkeypatch.setattr(controller_module, "IdentityWindowSampler", lambda **_: object())

    def init_transfer_queue(conf):
        captured["conf"] = conf
        return conf

    monkeypatch.setattr(controller_module.tq, "init", init_transfer_queue)
    controller = object.__new__(controller_module.Controller)
    controller.config = config

    controller._initialize_data_system()

    assert captured["conf"].controller.required_node_resource == expected_resource
    assert captured["conf"].backend.SimpleStorage.required_node_resource == expected_resource


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


@pytest.mark.parametrize("loss_type", ["sft", "dpo", "rm"])
def test_rollout_manager_keeps_node_affinity_and_requests_matching_marker(monkeypatch, tmp_path, loss_type):
    captured = {}
    node_id = "a" * 56
    rollout_module = ModuleType("relax.distributed.ray.rollout")

    class FakeRolloutManager(_FakeActorClass):
        @classmethod
        def options(cls, **options):
            captured["options"] = options
            return cls

    rollout_module.RolloutManager = FakeRolloutManager
    monkeypatch.setitem(sys.modules, "relax.distributed.ray.rollout", rollout_module)
    monkeypatch.setattr(placement_group_module, "_get_head_node_id", lambda: node_id)
    monkeypatch.setattr(
        placement_group_module.ray,
        "nodes",
        lambda: [{"NodeID": node_id, "Alive": True, "Resources": {"stable_cpu": 8}}],
    )
    args = _elastic_args(
        tmp_path,
        loss_type=loss_type,
        num_rollout_per_epoch=1,
        check_weight_update_equal=False,
        offload_rollout=False,
    )

    placement_group_module.create_rollout_manager(args, "pg", runtime_env={"env_vars": {"A": "B"}})

    assert captured["options"]["resources"] == {"stable_cpu": 1}
    assert captured["options"]["num_cpus"] == 1
    assert captured["options"]["runtime_env"] == {"env_vars": {"A": "B"}}
    assert captured["options"]["scheduling_strategy"].node_id == node_id


def test_genrm_manager_requests_stable_cpu(monkeypatch, tmp_path):
    captured = {}
    genrm_module = ModuleType("relax.distributed.ray.genrm")

    class FakeGenRMManager(_FakeActorClass):
        @classmethod
        def options(cls, **options):
            captured["options"] = options
            return cls

    genrm_module.GenRMManager = FakeGenRMManager
    monkeypatch.setitem(sys.modules, "relax.distributed.ray.genrm", genrm_module)
    args = _elastic_args(tmp_path, offload_rollout=False)

    placement_group_module.create_genrm_manager(args, "pg", runtime_env={"env_vars": {"A": "B"}})

    assert captured["options"] == {
        "name": "relax_genrm_manager",
        "num_cpus": 1,
        "num_gpus": 0,
        "runtime_env": {"env_vars": {"A": "B"}},
        "resources": {"stable_cpu": 1},
    }


def test_multi_genrm_managers_have_route_specific_names(monkeypatch, tmp_path):
    captured_options = []
    captured_ctor_kwargs = []
    genrm_module = ModuleType("relax.distributed.ray.genrm")

    class FakeGenRMManager(_FakeActorClass):
        @classmethod
        def options(cls, **options):
            captured_options.append(options)
            return cls

        @classmethod
        def remote(cls, *args, **kwargs):
            captured_ctor_kwargs.append(kwargs)
            return MagicMock()

    genrm_module.GenRMManager = FakeGenRMManager
    monkeypatch.setitem(sys.modules, "relax.distributed.ray.genrm", genrm_module)
    spec = {
        "model_path": "/model",
        "num_gpus": 1,
        "num_gpus_per_engine": 1,
        "engine_config": {},
        "sampling_config": {},
    }
    args = _elastic_args(
        tmp_path,
        offload_rollout=False,
        _genrm_instances_resolved={"quality": dict(spec), "safety": dict(spec)},
    )

    placement_group_module.create_genrm_managers(args, "pg")

    assert [options["name"] for options in captured_options] == [
        "relax_genrm_manager_quality",
        "relax_genrm_manager_safety",
    ]
    assert [kwargs["port_window_index"] for kwargs in captured_ctor_kwargs] == [0, 1]


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
