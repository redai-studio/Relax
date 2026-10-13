# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import importlib
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock

import pytest


@pytest.fixture(autouse=True)
def _cleanup_teacher_manager_module():
    yield
    sys.modules.pop("relax.distributed.ray.teacher_manager", None)


def _install_teacher_manager_stubs(monkeypatch):
    sglang_engine = ModuleType("relax.backends.sglang.sglang_engine")
    sglang_engine.SGLangEngine = object

    service = ModuleType("relax.core.service")
    service.create_placement_group = MagicMock()

    rollout = ModuleType("relax.distributed.ray.rollout")
    rollout._allocate_rollout_engine_addr_and_ports_normal = MagicMock()

    ray_utils = ModuleType("relax.distributed.ray.utils")
    ray_utils.NOSET_VISIBLE_DEVICES_ENV_VARS_LIST = []

    http_utils = ModuleType("relax.utils.http_utils")
    http_utils.find_available_port = MagicMock(return_value=15000)

    monkeypatch.setitem(sys.modules, "relax.backends.sglang.sglang_engine", sglang_engine)
    monkeypatch.setitem(sys.modules, "relax.core.service", service)
    monkeypatch.setitem(sys.modules, "relax.distributed.ray.rollout", rollout)
    monkeypatch.setitem(sys.modules, "relax.distributed.ray.utils", ray_utils)
    monkeypatch.setitem(sys.modules, "relax.utils.http_utils", http_utils)


def _import_teacher_manager(monkeypatch):
    _install_teacher_manager_stubs(monkeypatch)
    sys.modules.pop("relax.distributed.ray.teacher_manager", None)
    return importlib.import_module("relax.distributed.ray.teacher_manager")


def test_teacher_gpu_index_starts_at_planned_bundle_offset_for_shared_pg(monkeypatch):
    teacher_manager = _import_teacher_manager(monkeypatch)

    # The placement plan already accounts for the rollout region in front.
    assert (
        teacher_manager._resolve_teacher_gpu_index(
            replica=0,
            gpus_per_replica=4,
            shared_pg=True,
            bundle_offset=4,
        )
        == 4
    )


def test_teacher_gpu_index_starts_at_zero_for_dedicated_pg(monkeypatch):
    teacher_manager = _import_teacher_manager(monkeypatch)

    assert (
        teacher_manager._resolve_teacher_gpu_index(
            replica=0,
            gpus_per_replica=4,
            shared_pg=False,
        )
        == 0
    )


def test_teacher_gpu_index_adds_replica_stride_to_absolute_bundle_offset(monkeypatch):
    teacher_manager = _import_teacher_manager(monkeypatch)

    assert (
        teacher_manager._resolve_teacher_gpu_index(
            replica=1,
            gpus_per_replica=2,
            shared_pg=True,
            bundle_offset=12,
        )
        == 14
    )


def test_teacher_env_matches_rollout_genrm_stability_envs(monkeypatch):
    teacher_manager = _import_teacher_manager(monkeypatch)
    # RELAX_OPD_PREEXPANDED_PATCH is passed through from the driver env (default
    # "0"); set it so the test verifies the pass-through, not the default value.
    monkeypatch.setenv("RELAX_OPD_PREEXPANDED_PATCH", "1")
    args = SimpleNamespace(fp16=True)

    env = teacher_manager._build_teacher_engine_env(args)

    assert env["RELAX_OPD_PREEXPANDED_PATCH"] == "1"
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

    assert hasattr(teacher_manager.TeacherManager, "remote")
    assert hasattr(teacher_manager.TeacherManager, "options")


def test_teacher_manager_answers_snapshots_outside_its_default_concurrency_group(monkeypatch):
    """Discovery must not queue behind an offload or an engine rebuild."""
    teacher_manager = _import_teacher_manager(monkeypatch)
    metadata = teacher_manager.TeacherManager.__ray_metadata__

    group = metadata.modified_class.get_inference_snapshot.__ray_concurrency_group__
    assert metadata.concurrency_groups == {group: 1}
    # Everything that changes engine state stays serialized in the default group.
    for method in ("onload", "offload", "recover", "shutdown"):
        assert not hasattr(getattr(metadata.modified_class, method), "__ray_concurrency_group__")


def test_teacher_manager_starts_engines_with_teacher_profile(monkeypatch):
    teacher_manager = _import_teacher_manager(monkeypatch)
    manager_cls = teacher_manager.TeacherManager.__ray_metadata__.modified_class
    manager = object.__new__(manager_cls)
    manager._overrides = {"model_path": "/teacher"}
    manager.gpus_per_replica = 2

    assert manager._build_engine_ctor_kwargs(0) == {
        "sglang_overrides": {"model_path": "/teacher"},
        "num_gpus_per_engine": 2,
        "register_sigterm_handler": False,
        "profile": "teacher",
    }


def test_teacher_engine_asks_for_a_smaller_slice_on_rollout_bundles(monkeypatch):
    """A deferred teacher on rollout's own bundles shares each of them with the
    training actor and a rollout engine, like a GenRM that shares them."""
    teacher_manager = _import_teacher_manager(monkeypatch)
    manager_cls = teacher_manager.TeacherManager.__ray_metadata__.modified_class

    def resources(*, shared_pg, bundle_offset):
        manager = object.__new__(manager_cls)
        manager.args = SimpleNamespace(rollout_num_gpus=8)
        manager._shared_pg = shared_pg
        manager._bundle_offset = bundle_offset
        return manager._ray_resource_kwargs(0)

    assert resources(shared_pg=True, bundle_offset=0) == {"num_cpus": 0.1, "num_gpus": 0.1}
    assert resources(shared_pg=True, bundle_offset=4) == {"num_cpus": 0.1, "num_gpus": 0.1}
    # After rollout's region, or on a placement group of its own: unchanged.
    assert resources(shared_pg=True, bundle_offset=8) == {"num_cpus": 0.2, "num_gpus": 0.2}
    assert resources(shared_pg=False, bundle_offset=0) == {"num_cpus": 0.2, "num_gpus": 0.2}


def test_teacher_manager_checks_a_shared_placement_group_before_start(monkeypatch):
    teacher_manager = _import_teacher_manager(monkeypatch)
    manager_cls = teacher_manager.TeacherManager.__ray_metadata__.modified_class
    manager = object.__new__(manager_cls)
    manager._shared_pg_tuple = ("pg", list(range(8)), list(range(8)))

    manager._shared_pg = True
    assert manager._physical_placement() == ("actor", manager._shared_pg_tuple)
    # A dedicated replica's placement group does not exist yet; it is checked when it is created.
    manager._shared_pg = False
    assert manager._physical_placement() is None


def test_dedicated_teacher_placement_group_is_checked_and_returned_when_unusable(monkeypatch):
    teacher_manager = _import_teacher_manager(monkeypatch)
    manager_cls = teacher_manager.TeacherManager.__ray_metadata__.modified_class
    manager = object.__new__(manager_cls)
    manager.args = SimpleNamespace()
    manager._shared_pg = False
    manager.gpus_per_replica = 2
    removed = []
    # A 2-GPU group that Ray packed onto GPU 3 and GPU 5.
    monkeypatch.setattr(teacher_manager, "create_placement_group", lambda **kwargs: ("pg", [0, 1], [3, 5]))
    monkeypatch.setattr(teacher_manager, "remove_placement_group", removed.append)

    with pytest.raises(ValueError, match="teacher replica 0.*not contiguous"):
        manager._resolve_placement(0)

    assert removed == ["pg"]


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
    sys.modules["relax.utils.http_utils"].find_available_port.assert_not_called()
    sys.modules["relax.distributed.ray.rollout"]._allocate_rollout_engine_addr_and_ports_normal.assert_not_called()


def _patch_engine_actors(monkeypatch):
    """Let the manager "create" engines without a cluster: every engine actor
    is a recording stand-in and ``ray.get`` resolves its calls immediately."""
    import relax.distributed.ray.multi_engine_manager as mem

    class _Engine:
        def __getattr__(self, method):
            return SimpleNamespace(remote=lambda **kwargs: None)

    class _EngineActor:
        @staticmethod
        def options(**options):
            return SimpleNamespace(remote=lambda *args, **kwargs: _Engine())

    monkeypatch.setattr(mem.ray, "remote", lambda cls: _EngineActor)
    monkeypatch.setattr(mem.ray, "get", lambda refs, timeout=None: refs)
    monkeypatch.setattr(mem.ray, "kill", lambda engine: None)


def test_dedicated_teacher_recovery_rebuilds_and_bumps_revision(monkeypatch):
    teacher_manager = _import_teacher_manager(monkeypatch)
    _patch_engine_actors(monkeypatch)
    # Every dedicated replica gets its own placement group, and the rebuilt one
    # lands on another host.
    placement_groups = iter([("pg-first", [0, 1], [0, 1]), ("pg-rebuilt", [0, 1], [0, 1])])
    hosts = iter(["192.0.2.1", "192.0.2.2"])
    monkeypatch.setattr(teacher_manager, "create_placement_group", lambda **kwargs: next(placement_groups))
    monkeypatch.setattr(
        teacher_manager,
        "_allocate_rollout_engine_addr_and_ports_normal",
        lambda **kwargs: ({0: {"host": next(hosts), "port": kwargs["base_port"]}}, None),
    )
    removed = []
    # ``ray.util.placement_group`` as an attribute is the function of that name; go by module.
    monkeypatch.setattr(importlib.import_module("ray.util.placement_group"), "remove_placement_group", removed.append)

    manager_cls = teacher_manager.TeacherManager.__ray_metadata__.modified_class
    manager = object.__new__(manager_cls)
    manager_cls.__init__(manager, SimpleNamespace(teacher_hf_checkpoint="/teacher"), 1, 2)
    before = manager.get_inference_snapshot()

    manager._retire_engines([0])  # the engine died
    rebuilt = manager.recover()
    after = manager.get_inference_snapshot()

    # Rebuilt in place of a global restart, and the move is visible to discovery.
    assert rebuilt == {0}
    assert before["engines"] == [{"index": 0, "base_url": "http://192.0.2.1:15000", "state": "ready"}]
    assert after["engines"] == [{"index": 0, "base_url": "http://192.0.2.2:15000", "state": "ready"}]
    assert after["topology_revision"] > before["topology_revision"]
    # The dead engine's placement group went back to the cluster.
    assert removed == ["pg-first"]
