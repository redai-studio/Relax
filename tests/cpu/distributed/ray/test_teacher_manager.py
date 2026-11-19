# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest


def _import_teacher_manager(monkeypatch):
    from relax.distributed.ray import engine_pool

    return engine_pool


def _make_teacher_manager(monkeypatch, *, num_replicas=1, gpus_per_replica=16, shared_pg=True, bundle_offset=0):
    module = _import_teacher_manager(monkeypatch)
    original_init = module.EnginePool._initialize

    def skip_engine_init(self, *args, **kwargs):
        kwargs["skip_init"] = True
        original_init(self, *args, **kwargs)

    monkeypatch.setattr(module.EnginePool, "_initialize", skip_engine_init)
    monkeypatch.setattr(module, "find_available_port", MagicMock(return_value=15000))
    monkeypatch.setattr(module, "_allocate_rollout_engine_addr_and_ports_normal", MagicMock())
    args = SimpleNamespace(
        num_gpus_per_node=8, rollout_num_gpus=80, teacher_hf_checkpoint="/tmp/teacher", teacher_sglang_dp_size=8
    )
    pg = ("model-pg", list(range(160)), [i % 8 for i in range(160)])
    manager = module.EnginePool(
        "teacher",
        args,
        pg,
        dict(
            num_replicas=num_replicas,
            gpus_per_replica=gpus_per_replica,
            shared_pg=shared_pg,
            bundle_offset=bundle_offset,
        ),
    )
    return module, manager


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


def test_engine_pool_is_local_not_an_actor(monkeypatch):
    module = _import_teacher_manager(monkeypatch)
    assert not hasattr(module.EnginePool, "remote")
    assert not hasattr(module.EnginePool, "options")


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


@pytest.mark.parametrize("shared", [False, True])
def test_teacher_placement_borrows_model_binding(monkeypatch, shared):
    _, pool = _make_teacher_manager(monkeypatch, num_replicas=2, gpus_per_replica=2, shared_pg=shared, bundle_offset=2)
    binding, owned, offset = pool._resolve_placement(1)
    assert binding is pool._shared_pg_tuple
    assert owned is False
    assert offset == (84 if shared else 2)


@pytest.mark.parametrize("bundle_offset", [0, 16, 32, 48, 64])
def test_tp16_teachers_use_two_distinct_node_slots_in_their_own_slice(monkeypatch, bundle_offset):
    _, manager = _make_teacher_manager(monkeypatch, bundle_offset=bundle_offset)

    assert manager.nodes_per_engine == 2
    assert len(manager.all_engines) == 2
    first_pg, first_owns_pg, first_index = manager._resolve_placement(0)
    second_pg, second_owns_pg, second_index = manager._resolve_placement(1)
    assert first_pg is second_pg
    assert not first_owns_pg and not second_owns_pg
    assert (first_index, second_index) == (80 + bundle_offset, 88 + bundle_offset)
    assert first_pg[2][first_index] == second_pg[2][second_index] == 0


def test_dedicated_multinode_replicas_use_distinct_full_node_slices(monkeypatch):
    _, pool = _make_teacher_manager(monkeypatch, num_replicas=2, shared_pg=False)
    placements = [pool._resolve_placement(rank) for rank in range(4)]
    assert [item[2] for item in placements] == [0, 8, 16, 24]
    assert all(item[0] is pool._shared_pg_tuple and item[1] is False for item in placements)
    assert pool.engines == [None, None]  # Followers never advertise HTTP replicas.


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
