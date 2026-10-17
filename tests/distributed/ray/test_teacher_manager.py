# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import importlib
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest


pytest.importorskip("sglang")

from relax.distributed.ray import teacher_manager as teacher_manager_module


def _import_teacher_manager(monkeypatch):
    placement = importlib.import_module("relax.distributed.ray.placement_group")
    factory = MagicMock()
    monkeypatch.setattr(placement, "create_placement_group", factory)
    monkeypatch.setattr(teacher_manager_module, "create_placement_group", factory)
    monkeypatch.setattr(teacher_manager_module, "_allocate_rollout_engine_addr_and_ports_normal", MagicMock())
    monkeypatch.setattr(teacher_manager_module, "find_available_port", MagicMock(return_value=15000))
    return teacher_manager_module


def test_teacher_gpu_index_uses_rollout_offset_for_shared_pg(monkeypatch):
    teacher_manager = _import_teacher_manager(monkeypatch)
    args = SimpleNamespace(rollout_num_gpus=4)

    assert (
        teacher_manager._resolve_teacher_gpu_index(
            args=args,
            replica=0,
            gpus_per_replica=4,
            shared_pg=True,
        )
        == 4
    )


def test_teacher_gpu_index_starts_at_zero_for_dedicated_pg(monkeypatch):
    teacher_manager = _import_teacher_manager(monkeypatch)
    args = SimpleNamespace(rollout_num_gpus=4)

    assert (
        teacher_manager._resolve_teacher_gpu_index(
            args=args,
            replica=0,
            gpus_per_replica=4,
            shared_pg=False,
        )
        == 0
    )


def test_teacher_gpu_index_uses_teacher_relative_bundle_offset(monkeypatch):
    teacher_manager = _import_teacher_manager(monkeypatch)
    args = SimpleNamespace(rollout_num_gpus=8)

    assert (
        teacher_manager._resolve_teacher_gpu_index(
            args=args,
            replica=1,
            gpus_per_replica=2,
            shared_pg=True,
            bundle_offset=4,
        )
        == 14
    )


def test_teacher_env_matches_rollout_genrm_stability_envs(monkeypatch):
    teacher_manager = _import_teacher_manager(monkeypatch)
    monkeypatch.setenv("SGLANG_MM_AVOID_RETOKENIZE", "false")
    args = SimpleNamespace(fp16=True)

    env = teacher_manager._build_teacher_engine_env(args)

    assert env["SGLANG_MM_AVOID_RETOKENIZE"] == "true"
    assert env["SGLANG_JIT_DEEPGEMM_PRECOMPILE"] == "false"
    assert env["SGL_DISABLE_TP_MEMORY_INBALANCE_CHECK"] == "true"
    assert env["SGLANG_DISABLE_TP_MEMORY_INBALANCE_CHECK"] == "true"
    assert env["SGLANG_MEMORY_SAVER_CUDA_GRAPH"] == "true"
    assert env["SGLANG_BATCH_INVARIANT_OPS_ENABLE_MM_FALLBACK_VARIANT"] == "true"
    assert env["SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION"] == "false"
    assert env["SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_IDLE"] == "false"
    assert env["SGLANG_MAMBA_CONV_DTYPE"] == "float16"


def test_teacher_manager_exposes_ray_actor_api(monkeypatch):
    teacher_manager = _import_teacher_manager(monkeypatch)
    from relax.distributed.ray.inference_manager import InferenceManager

    assert hasattr(teacher_manager.TeacherManager, "remote")
    assert hasattr(teacher_manager.TeacherManager, "options")
    assert issubclass(teacher_manager.TeacherManager.__ray_metadata__.modified_class, InferenceManager)


def test_teacher_recovery_reuses_original_endpoint(monkeypatch):
    teacher_manager = _import_teacher_manager(monkeypatch)
    manager_cls = teacher_manager.TeacherManager.__ray_metadata__.modified_class
    manager = object.__new__(manager_cls)
    manager._shared_pg = True
    original = {
        "host": "192.0.2.1",
        "port": 15001,
        "nccl_port": 15002,
        "dist_init_addr": "192.0.2.1:15003",
    }
    manager._engine_addr_and_ports = {0: original}

    result = manager._allocate_engine_addr_and_ports(new_engines=[(0, object())])

    assert result == {0: original}
    assert result[0] is not original
    teacher_manager.find_available_port.assert_not_called()
    teacher_manager._allocate_rollout_engine_addr_and_ports_normal.assert_not_called()


def test_dedicated_teacher_recovery_requires_global_restart(monkeypatch):
    teacher_manager = _import_teacher_manager(monkeypatch)
    manager_cls = teacher_manager.TeacherManager.__ray_metadata__.modified_class
    manager = object.__new__(manager_cls)
    manager._shared_pg = False
    manager.all_engines = [None]

    with pytest.raises(RuntimeError, match="global restart"):
        manager.recover()


@pytest.mark.parametrize("shared", [False, True])
@pytest.mark.parametrize("affinity", [False, True])
def test_teacher_placement_uses_distributed_factory_and_preserves_ownership(monkeypatch, shared, affinity):
    teacher_manager = _import_teacher_manager(monkeypatch)
    placement = importlib.import_module("relax.distributed.ray.placement_group")
    assert teacher_manager.create_placement_group is placement.create_placement_group
    manager_cls = teacher_manager.TeacherManager.__ray_metadata__.modified_class
    manager = object.__new__(manager_cls)
    manager.args = SimpleNamespace(enable_affinity=affinity, rollout_num_gpus=4)
    manager.gpus_per_replica = 2
    manager._shared_pg = shared
    manager._bundle_offset = 2
    manager._shared_pg_tuple = ("shared", [0, 1], [0, 1])
    manager.nodes_per_engine = 1
    manager._local_gpus_per_worker = 2
    manager._dedicated_replica_pgs = {}
    manager._planned_placement = None
    placement.create_placement_group.return_value = ("dedicated", [0, 1], [0, 1])

    pg, owned, gpu_index = manager._resolve_placement(1)

    assert owned is (not shared)
    if shared:
        assert pg is manager._shared_pg_tuple
        assert gpu_index == 8
        placement.create_placement_group.assert_not_called()
    else:
        assert pg is placement.create_placement_group.return_value
        assert gpu_index == 0
        placement.create_placement_group.assert_called_once_with(num_gpus=2, node_group_affinity=affinity)
