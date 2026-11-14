# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import asyncio
import threading
from types import SimpleNamespace

import pytest


pytest.importorskip("ray")
from relax.distributed.checkpoint_service.config import RoleInfo
from relax.distributed.checkpoint_service.coordinator.service import DCSCoordinator, SendWeightMetaRequest
from relax.distributed.checkpoint_service.coordinator.topology import TopologyManager


def test_periodic_reference_has_distinct_members_and_metadata_lifetime():
    cls = DCSCoordinator.func_or_class
    coordinator = object.__new__(cls)
    coordinator.topology_manager = TopologyManager()
    coordinator.weight_updates = {}
    coordinator.weight_meta_buffer = []
    coordinator._lock = threading.Lock()
    for role in ("actor", "actor_fwd", "reference"):
        coordinator.topology_manager.register(
            RoleInfo(
                role_name=role,
                rank=0,
                metadata={
                    "pp_size": 1,
                    "pp_rank": 0,
                    "is_pp_src_rank": True,
                    "master_address": "127.0.0.1",
                    "master_port": 22001,
                    "reference_master_port": 22002,
                    "worker_incarnation": role + "-first",
                },
            )
        )

    async def verify():
        single = await coordinator.get_model_update_group_ranks("actor", 0, "false", 0)
        combined = await coordinator.get_model_update_group_ranks("actor", 0, "true", 1)
        assert single.world_size == 2
        assert combined.world_size == 3
        assert single.topology_id != combined.topology_id
        assert set(single.pp_groups.values()) == {"tcp://127.0.0.1:22001"}
        assert set(combined.pp_groups.values()) == {"tcp://127.0.0.1:22002"}
        again = await coordinator.get_model_update_group_ranks("actor_fwd", 0, "false", 2)
        assert again.topology_id == single.topology_id
        assert again.update_id != single.update_id
        message = SendWeightMetaRequest(
            names=["w"],
            dtypes=["float32"],
            shapes=[[1]],
            group_name=next(iter(combined.pp_groups)),
            update_id=combined.update_id,
        )
        await coordinator.send_weight_meta(message)
        assert await coordinator.recv_weight_meta(0, update_id=single.update_id) == []
        await coordinator.clear_weight_meta(combined.update_id, "actor_fwd_0")
        assert len(await coordinator.recv_weight_meta(0, update_id=combined.update_id)) == 1
        await coordinator.clear_weight_meta(combined.update_id, "reference_0")
        assert combined.update_id not in coordinator.weight_updates
        node = coordinator.topology_manager.get_node("reference", 0)
        node.metadata["worker_incarnation"] = "reference-restarted"
        changed = await coordinator.get_model_update_group_ranks("actor", 0, "true", 3)
        assert changed.topology_id != combined.topology_id

    asyncio.run(verify())


def test_backend_selects_cached_group_by_exact_member_generation(monkeypatch):
    pytest.importorskip("megatron.core")
    from relax.distributed.checkpoint_service.backends import device_direct

    backend = object.__new__(device_direct.DeviceDirectBackend)
    backend.role_info = RoleInfo(role_name="actor", rank=0)
    backend.backend_type = "gloo"
    backend._megatron = SimpleNamespace(
        mpu=SimpleNamespace(
            get_data_parallel_rank=lambda **kw: 0,
            get_tensor_model_parallel_rank=lambda: 0,
            get_pipeline_model_parallel_rank=lambda: 0,
        )
    )
    created, removed = [], []

    def create(**kwargs):
        group = object()
        created.append((group, kwargs))
        return group

    monkeypatch.setattr(device_direct, "init_process_group", create)
    monkeypatch.setattr(device_direct.dist, "destroy_process_group", removed.append)

    def topology(name, reference, step):
        return {
            "topology_id": name,
            "update_id": f"{name}:{step}",
            "includes_reference": reference,
            "global_rank": 0,
            "world_size": 3 if reference else 2,
            "pp_groups": {f"update_actor_pp_0_{name}": "tcp://127.0.0.1:22001"},
        }

    backend.init_process_groups_for_actor_fwd_ref(topology("fwd", False, 0))
    first = backend._model_update_groups_for_actor_fwd_ref
    backend.init_process_groups_for_actor_fwd_ref(topology("both", True, 1))
    assert backend._model_update_groups_for_actor_fwd_ref != first
    backend.init_process_groups_for_actor_fwd_ref(topology("fwd", False, 2))
    assert backend._model_update_groups_for_actor_fwd_ref is first
    backend.init_process_groups_for_actor_fwd_ref(topology("new-both", True, 3))
    assert len(removed) == 1
    assert removed[0] is created[1][0]
    assert backend._forward_update_id == "new-both:3"
