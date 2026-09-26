# Copyright (c) 2026 Relax Authors. All Rights Reserved.


from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest


@pytest.fixture
def legacy_rollout(monkeypatch):
    pytest.importorskip("sglang")

    from relax.distributed.ray import rollout

    engines = [MagicMock() for _ in range(4)]
    for rank, engine in enumerate(engines):
        engine.get_url.remote.return_value = f"http://worker-{rank}.example:15000"
        engine.get_pid_and_node_id.remote.return_value = {"pid": 100 + rank, "node_id": f"node-{rank}"}
    group = rollout.EngineGroup(
        args=SimpleNamespace(num_gpus_per_node=2),
        pg=None,
        all_engines=engines,
        num_gpus_per_engine=4,
        num_new_engines=2,
        rank_offset=6,
        gpu_offset=8,
    )
    server = rollout.RolloutServer(
        engine_groups=[group],
        router_ip="router.example",
        router_port=16000,
        model_name="policy",
    )
    manager_cls = rollout.RolloutManager.__ray_metadata__.modified_class
    manager = object.__new__(manager_cls)
    manager.servers = {"policy": server}
    manager.rollout_engine_lock = object()
    manager.status = "offload"
    monkeypatch.setattr(rollout.ray, "get", lambda refs, timeout=None: refs)
    return manager, group, engines


def test_legacy_rollout_weight_sync_tuple_contains_only_logical_heads(legacy_rollout):
    manager, _group, engines = legacy_rollout

    result = manager.get_rollout_engines_and_lock("policy")

    assert isinstance(result, tuple)
    assert len(result) == 5
    assert result == ([engines[0], engines[2]], manager.rollout_engine_lock, 2, [4, 4], [8, 12])
    assert manager.get_rollout_engines_and_lock() == result
    assert manager.get_rollout_engines_and_lock("unknown") == ([], manager.rollout_engine_lock, 0, [], [])


def test_legacy_rollout_discovery_retains_worker_diagnostics_even_when_offloaded(legacy_rollout):
    manager, _group, _engines = legacy_rollout

    snapshot = manager.get_engines_info("policy")

    assert snapshot == {
        "models": {
            "policy": {
                "router_ip": "router.example",
                "router_port": 16000,
                "engine_groups": [
                    {
                        "group_index": 0,
                        "worker_type": "regular",
                        "num_gpus_per_engine": 4,
                        "num_new_engines": 2,
                        "engines": [
                            {
                                "rank": 6 + rank,
                                "status": "active",
                                "url": f"http://worker-{rank}.example:15000",
                                "pid": 100 + rank,
                                "node_id": f"node-{rank}",
                            }
                            for rank in range(4)
                        ],
                    }
                ],
                "total_engines": 4,
            }
        },
        "total_engines": 4,
    }


def test_legacy_rollout_discovery_keeps_dead_worker_slot(legacy_rollout):
    manager, group, _engines = legacy_rollout
    group.all_engines[1] = None

    snapshot = manager.get_engines_info("policy")

    workers = snapshot["models"]["policy"]["engine_groups"][0]["engines"]
    assert workers[1] == {"rank": 7, "status": "dead"}
    assert len(workers) == snapshot["total_engines"] == 4
