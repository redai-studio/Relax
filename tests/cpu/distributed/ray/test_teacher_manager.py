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

    # Import the real factory before installing the optional engine stubs.
    placement = importlib.import_module("relax.distributed.ray.placement_group")
    monkeypatch.setattr(placement, "create_placement_group", MagicMock())

    rollout = ModuleType("relax.distributed.ray.rollout")
    rollout._allocate_rollout_engine_addr_and_ports_normal = MagicMock()

    ray_utils = ModuleType("relax.distributed.ray.utils")
    ray_utils.NOSET_VISIBLE_DEVICES_ENV_VARS_LIST = []

    http_utils = ModuleType("relax.utils.http_utils")
    http_utils.find_available_port = MagicMock(return_value=15000)

    monkeypatch.setitem(sys.modules, "relax.backends.sglang.sglang_engine", sglang_engine)
    monkeypatch.setitem(sys.modules, "relax.distributed.ray.rollout", rollout)
    monkeypatch.setitem(sys.modules, "relax.distributed.ray.utils", ray_utils)
    monkeypatch.setitem(sys.modules, "relax.utils.http_utils", http_utils)


def _import_teacher_manager(monkeypatch):
    _install_teacher_manager_stubs(monkeypatch)
    sys.modules.pop("relax.distributed.ray.teacher_manager", None)
    return importlib.import_module("relax.distributed.ray.teacher_manager")


def _make_teacher_manager(monkeypatch, *, num_replicas=1, gpus_per_replica=16, shared_pg=True, bundle_offset=0):
    teacher_manager_module = _import_teacher_manager(monkeypatch)
    original_init = teacher_manager_module.MultiEngineManager.__init__

    def skip_engine_init(self, *args, **kwargs):
        kwargs["skip_init"] = True
        original_init(self, *args, **kwargs)

    monkeypatch.setattr(teacher_manager_module.MultiEngineManager, "__init__", skip_engine_init)
    args = SimpleNamespace(
        num_gpus_per_node=8,
        rollout_num_gpus=80,
        teacher_hf_checkpoint="/tmp/teacher",
        teacher_sglang_dp_size=8,
    )
    pg = ("shared-pg", list(range(160)), [i % 8 for i in range(160)]) if shared_pg else None
    manager_cls = teacher_manager_module.TeacherManager.__ray_metadata__.modified_class
    manager = manager_cls(
        args,
        num_replicas,
        gpus_per_replica,
        pg=pg,
        shared_pg=shared_pg,
        bundle_offset=bundle_offset,
    )
    return teacher_manager_module, manager


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


@pytest.mark.parametrize(("gpus_per_replica", "num_nodes"), [(2, 1), (16, 2)])
def test_teacher_recovery_reuses_original_endpoints(monkeypatch, gpus_per_replica, num_nodes):
    teacher_manager_module, manager = _make_teacher_manager(monkeypatch, gpus_per_replica=gpus_per_replica)
    original = {
        rank: {
            "host": f"192.0.2.{rank + 1}",
            "port": 15001,
            "nccl_port": 15002,
            "dist_init_addr": "192.0.2.1:15003",
        }
        for rank in range(num_nodes)
    }
    manager._engine_addr_and_ports = original

    result = manager._allocate_engine_addr_and_ports(new_engines=[(rank, object()) for rank in range(num_nodes)])

    assert result == original
    assert all(result[rank] is not original[rank] for rank in range(num_nodes))
    teacher_manager_module.find_available_port.assert_not_called()
    teacher_manager_module._allocate_rollout_engine_addr_and_ports_normal.assert_not_called()


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
    manager_cls.__init__(manager, SimpleNamespace(teacher_hf_checkpoint="/teacher", num_gpus_per_node=8), 1, 2)
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


@pytest.mark.parametrize("shared", [False, True])
@pytest.mark.parametrize("affinity", [False, True])
def test_teacher_placement_uses_distributed_factory_and_preserves_ownership(monkeypatch, shared, affinity):
    teacher_manager = _import_teacher_manager(monkeypatch)
    placement = importlib.import_module("relax.distributed.ray.placement_group")
    assert teacher_manager.create_placement_group is placement.create_placement_group
    manager_cls = teacher_manager.TeacherManager.__ray_metadata__.modified_class
    manager = object.__new__(manager_cls)
    manager.args = SimpleNamespace(enable_affinity=affinity, rollout_num_gpus=4, num_gpus_per_node=8)
    manager.nodes_per_engine = 1
    manager.gpus_per_replica = 2
    manager._shared_pg = shared
    manager._bundle_offset = 2
    manager._shared_pg_tuple = ("shared", [0, 1], [0, 1])
    placement.create_placement_group.return_value = ("dedicated", [0, 1], [0, 1])

    pg, owned, gpu_index = manager._resolve_placement(1)

    assert owned is (not shared)
    if shared:
        assert pg is manager._shared_pg_tuple
        assert gpu_index == 4
        placement.create_placement_group.assert_not_called()
    else:
        assert pg is placement.create_placement_group.return_value
        assert gpu_index == 0
        placement.create_placement_group.assert_called_once_with(num_gpus=2, node_group_affinity=affinity)


@pytest.mark.parametrize("bundle_offset", [0, 16, 32, 48, 64])
def test_tp16_teachers_use_two_distinct_node_slots_in_their_own_slice(monkeypatch, bundle_offset):
    _, manager = _make_teacher_manager(monkeypatch, bundle_offset=bundle_offset)

    assert manager.nodes_per_engine == 2
    assert len(manager.all_engines) == 2
    first_pg, first_owns_pg, first_index = manager._resolve_placement(0)
    second_pg, second_owns_pg, second_index = manager._resolve_placement(1)
    assert first_pg is second_pg
    assert not first_owns_pg and not second_owns_pg
    assert (first_index, second_index) == (bundle_offset, 8 + bundle_offset)
    assert first_pg[2][first_index] == second_pg[2][second_index] == 0


def test_dedicated_multinode_replica_creates_one_pg_for_both_slots(monkeypatch):
    teacher_manager_module, manager = _make_teacher_manager(monkeypatch, num_replicas=2, shared_pg=False)
    pg0 = ("pg0", list(range(16)), list(range(8)) * 2)
    pg1 = ("pg1", list(range(16)), list(range(8)) * 2)
    teacher_manager_module.create_placement_group.side_effect = [pg0, pg1]

    placements = []
    for rank in range(4):
        placement = manager._resolve_placement(rank)
        placements.append(placement)
        manager._engine_placements[rank] = placement[:2]

    assert placements == [(pg0, True, 0), (pg0, False, 8), (pg1, True, 0), (pg1, False, 8)]
    assert teacher_manager_module.create_placement_group.call_count == 2
    for call in teacher_manager_module.create_placement_group.call_args_list:
        assert call.kwargs["num_gpus"] == 16


def test_multinode_teacher_allocates_each_replicas_node_addresses_together(monkeypatch):
    def fake_allocate(*, rollout_engines, rank_offset, base_port, **kwargs):
        return (
            {
                rank: {
                    "host": f"192.0.2.{rank + 1}",
                    "port": base_port,
                    "nccl_port": base_port + 1,
                    "dist_init_addr": f"192.0.2.{1 + (rank // 2) * 2}:{base_port + 2}",
                }
                for rank, _ in rollout_engines
            },
            {},
        )

    allocator = MagicMock(side_effect=fake_allocate)
    teacher_manager_module, manager = _make_teacher_manager(monkeypatch, num_replicas=2)
    monkeypatch.setattr(teacher_manager_module, "_allocate_rollout_engine_addr_and_ports_normal", allocator)
    monkeypatch.setattr(teacher_manager_module.ray, "get", lambda ref: ref)
    engines = [(rank, MagicMock()) for rank in range(4)]
    expected = {
        rank: {
            "host": f"192.0.2.{rank + 1}",
            "port": 15000,
            "nccl_port": 15001,
            "dist_init_addr": f"192.0.2.{1 + (rank // 2) * 2}:15002",
        }
        for rank in range(4)
    }
    result = manager._allocate_engine_addr_and_ports(new_engines=engines)

    assert result == expected
    assert allocator.call_count == 2
    for replica, call in enumerate(allocator.call_args_list):
        assert call.kwargs["rollout_engines"] == engines[replica * 2 : replica * 2 + 2]
        assert call.kwargs["num_gpus_per_engine"] == 16
        assert call.kwargs["rank_offset"] == replica * 2


def test_single_node_dedicated_teachers_allocate_their_own_hosts(monkeypatch):
    def fake_allocate(*, rollout_engines, rank_offset, base_port, **kwargs):
        return (
            {
                rank: {
                    "host": f"192.0.2.{rank + 1}",
                    "port": base_port,
                    "nccl_port": base_port + 1,
                    "dist_init_addr": f"192.0.2.{rank_offset + 1}:{base_port + 2}",
                }
                for rank, _ in rollout_engines
            },
            {},
        )

    allocator = MagicMock(side_effect=fake_allocate)
    teacher_manager_module, manager = _make_teacher_manager(
        monkeypatch, num_replicas=2, gpus_per_replica=2, shared_pg=False
    )
    monkeypatch.setattr(teacher_manager_module, "_allocate_rollout_engine_addr_and_ports_normal", allocator)
    monkeypatch.setattr(teacher_manager_module.ray, "get", lambda ref: ref)
    engines = [(rank, MagicMock()) for rank in range(2)]
    expected = {
        rank: {
            "host": f"192.0.2.{rank + 1}",
            "port": 15000,
            "nccl_port": 15001,
            "dist_init_addr": f"192.0.2.{rank + 1}:15002",
        }
        for rank in range(2)
    }

    result = manager._allocate_engine_addr_and_ports(new_engines=engines)

    assert result == expected
    for rank, call in enumerate(allocator.call_args_list):
        assert call.kwargs["rollout_engines"] == [engines[rank]]
        assert call.kwargs["rank_offset"] == rank


def test_multinode_teacher_routes_and_memory_operations_use_only_heads(monkeypatch):
    teacher_manager_module, manager = _make_teacher_manager(monkeypatch, num_replicas=2)
    engines = [MagicMock() for _ in range(4)]
    for rank, engine in enumerate(engines):
        engine.get_url.remote.return_value = f"http://192.0.2.{rank + 1}:15000"
        engine.health_generate.remote.return_value = True
    manager.all_engines = engines
    monkeypatch.setattr(teacher_manager_module.ray, "get", lambda ref, timeout=None: ref)

    assert manager.get_urls() == ["http://192.0.2.1:15000/generate", "http://192.0.2.3:15000/generate"]
    assert manager.health_check()
    manager.offload()
    manager.onload()

    for rank, engine in enumerate(engines):
        if rank % 2 == 0:
            engine.get_url.remote.assert_called_once_with()
            engine.health_generate.remote.assert_called_once_with()
            engine.release_memory_occupation.remote.assert_called_once_with()
            engine.resume_memory_occupation.remote.assert_called_once_with(tags=None)
        else:
            engine.get_url.remote.assert_not_called()
            engine.health_generate.remote.assert_not_called()
            engine.release_memory_occupation.remote.assert_not_called()
            engine.resume_memory_occupation.remote.assert_not_called()


def test_teacher_rejects_partial_multinode_replica(monkeypatch):
    with pytest.raises(AssertionError, match="must be divisible"):
        _make_teacher_manager(monkeypatch, gpus_per_replica=12)
