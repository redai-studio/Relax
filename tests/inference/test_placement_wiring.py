# Copyright (c) 2026 Relax Authors. All Rights Reserved.


from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest


@pytest.fixture
def service_module():
    from relax.core import service

    return service


@pytest.fixture
def allocation(service_module, monkeypatch):
    module = service_module
    runtime = SimpleNamespace(
        pg=SimpleNamespace(ready=lambda: "ready"),
        probes=[],
        killed=[],
        removed=[],
        timeouts=[],
        fail_stage=None,
        devices=[("192.0.2.2", 0), ("192.0.2.1", 1)],
    )
    monkeypatch.delenv("RELAX_INITIAL_NODE_GROUP", raising=False)
    monkeypatch.setattr(module, "placement_group", lambda *args, **kwargs: runtime.pg)
    monkeypatch.setattr(module, "remove_placement_group", runtime.removed.append)
    monkeypatch.setattr(module, "PlacementGroupSchedulingStrategy", lambda **kwargs: kwargs)
    monkeypatch.setattr(module, "get_ray_accelerator_kwargs", lambda count: {"num_gpus": count})
    monkeypatch.setattr(module.device_utils, "get_ray_accelerator_name", lambda: "GPU")

    class Probe:
        @staticmethod
        def options(**kwargs):
            return Probe

        @staticmethod
        def remote():
            if runtime.fail_stage == "actor" and runtime.probes:
                raise RuntimeError("probe actor failed")
            probe = MagicMock()
            runtime.probes.append(probe)
            return probe

    def get(ref, *, timeout):
        runtime.timeouts.append(timeout)
        if runtime.fail_stage == "ready" and ref == "ready":
            raise TimeoutError("PG ready timeout")
        if isinstance(ref, list):
            if runtime.fail_stage == "probe":
                raise TimeoutError("probe timeout")
            return runtime.devices
        return None

    monkeypatch.setattr(module, "InfoActor", Probe)
    monkeypatch.setattr(module.ray, "get", get)
    monkeypatch.setattr(module.ray, "kill", runtime.killed.append)
    monkeypatch.setattr(
        module.ray,
        "nodes",
        lambda: [
            {"Alive": True, "NodeID": "node-a", "NodeManagerAddress": "192.0.2.1"},
            {"Alive": True, "NodeID": "node-b", "NodeManagerAddress": "192.0.2.2"},
        ],
    )
    return runtime


def test_pg_tuple_compatibility_preserves_sorted_physical_node_identity(service_module, allocation):
    result = service_module.create_placement_group(2, timeout_s=3.0)

    assert result == (allocation.pg, [1, 0], [1, 0])
    topology = service_module.get_placement_group_topology(result)
    assert topology == (
        {"bundle_index": 1, "node_id": "node-a", "node_ip": "192.0.2.1", "gpu_id": 1},
        {"bundle_index": 0, "node_id": "node-b", "node_ip": "192.0.2.2", "gpu_id": 0},
    )
    topology[0]["gpu_id"] = 99
    assert service_module.get_placement_group_topology(result)[0]["gpu_id"] == 1
    assert all(0 < timeout <= 3.0 for timeout in allocation.timeouts)
    assert allocation.killed == allocation.probes
    assert allocation.removed == []


def test_real_ray_pg_handle_preserves_physical_metadata_when_serialized(service_module):
    import ray.cloudpickle as cloudpickle
    from ray.util.placement_group import PlacementGroup

    pg = PlacementGroup.empty()
    pg._relax_bundle_topology = ({"bundle_index": 2, "node_id": "node-a", "node_ip": "192.0.2.1", "gpu_id": 3},)

    restored = cloudpickle.loads(cloudpickle.dumps((pg, [2], [3])))

    assert service_module.get_placement_group_topology(restored) == pg._relax_bundle_topology


@pytest.mark.parametrize("stage", ["ready", "actor", "probe", "duplicate", "unknown_node"])
def test_pg_allocation_failure_cleans_all_probe_actors_and_owned_pg(service_module, allocation, stage):
    allocation.fail_stage = stage
    if stage == "duplicate":
        allocation.devices = [("192.0.2.1", 0)] * 2
    elif stage == "unknown_node":
        allocation.devices[0] = ("192.0.2.3", 0)

    with pytest.raises((TimeoutError, RuntimeError, ValueError)):
        service_module.create_placement_group(2, timeout_s=1.0)

    assert allocation.killed == allocation.probes
    assert allocation.removed == [allocation.pg]


@pytest.mark.parametrize("timeout", [0, -1, float("inf"), float("nan")])
def test_pg_invalid_deadline_rejects_before_allocation(service_module, monkeypatch, timeout):
    create = MagicMock()
    monkeypatch.setattr(service_module, "placement_group", create)
    with pytest.raises(ValueError, match="timeout"):
        service_module.create_placement_group(1, timeout_s=timeout)
    create.assert_not_called()


@pytest.fixture
def controller_module():
    from relax.core import controller

    return controller


@pytest.mark.parametrize("failure", ["invalid_layout", "insufficient_gpus", "defer_not_enabled"])
def test_preflight_failure_occurs_before_teacher_datasource_or_pg_start(controller_module, monkeypatch, failure):
    module = controller_module
    instance = object.__new__(module.Controller)
    instance.config = SimpleNamespace()

    def plan(config):
        if failure == "invalid_layout":
            raise ValueError("layout conflict")
        return SimpleNamespace(
            mode="defer" if failure == "defer_not_enabled" else "split", total_required_gpus=8, to_dict=lambda: {}
        )

    monkeypatch.setattr(module, "plan_inference_placement", plan)
    monkeypatch.setattr(module, "validate_ppo_config", lambda config: None)
    monkeypatch.setattr(module.ray, "cluster_resources", lambda: {"GPU": 4})
    monkeypatch.setattr(module.device_utils, "get_ray_accelerator_name", lambda: "GPU")
    teacher, datasource, pg = MagicMock(), MagicMock(), MagicMock()
    monkeypatch.setattr(module, "maybe_start_managed_opd_teacher", teacher)
    monkeypatch.setattr(module, "create_data_source_actor", datasource)
    monkeypatch.setattr(module, "create_placement_group", pg)

    with pytest.raises((ValueError, RuntimeError)) as error:
        instance.register_all_serve()
    if failure == "defer_not_enabled":
        assert isinstance(error.value, ValueError)
        assert "--inference-defer-roles" in str(error.value)

    teacher.assert_not_called()
    datasource.assert_not_called()
    pg.assert_not_called()


def test_preflight_loads_existing_yaml_and_stores_serializable_plan(controller_module, monkeypatch, tmp_path):
    module = controller_module
    instance = object.__new__(module.Controller)
    config_file = tmp_path / "rollout.yaml"
    config_file.write_text("sglang:\n  - name: policy\n    engine_groups: []\n")
    instance.config = SimpleNamespace(sglang_config=str(config_file))
    expected = {"mode": "split", "placements": [], "total_required_gpus": 4}

    def plan(config):
        assert config._inference_rollout_config == {"sglang": [{"name": "policy", "engine_groups": []}]}
        return SimpleNamespace(mode="split", total_required_gpus=4, to_dict=lambda: expected, for_role=lambda role: ())

    monkeypatch.setattr(module, "plan_inference_placement", plan)
    monkeypatch.setattr(module.ray, "cluster_resources", lambda: {"GPU": 4})
    monkeypatch.setattr(module.device_utils, "get_ray_accelerator_name", lambda: "GPU")

    instance._preflight_inference_placement()

    assert instance.config._inference_placement_plan == expected


@pytest.mark.parametrize("borrowed", [False, True])
def test_binding_failure_prevents_deploy_and_deletes_only_owned_pg(service_module, monkeypatch, borrowed):
    service = object.__new__(service_module.Service)
    pg = SimpleNamespace(_relax_bundle_topology=({"bundle_index": 0, "node_id": "n", "gpu_id": 0},))
    service.pgs = (pg, [0], [0])
    service.role = "rollout"
    service._deployed = False
    service._is_shared_pgs = borrowed
    service.config = SimpleNamespace(
        _inference_placement_plan={"placements": [{"pool": "actor" if borrowed else "rollout"}]}
    )
    service._deploy = MagicMock()

    def reject(*args, **kwargs):
        raise ValueError("invalid bound topology")

    monkeypatch.setattr(service_module, "validate_bound_placement", reject)
    remove = MagicMock()
    monkeypatch.setattr(service_module, "remove_placement_group", remove)

    with pytest.raises(ValueError, match="bound topology"):
        service.deploy()

    service._deploy.assert_not_called()
    assert remove.call_count == int(not borrowed)
    assert (service.pgs is not None) is borrowed
