# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""``RolloutManager.get_inference_snapshot``: the rollout role's topology in
the unified discovery schema."""

import pytest


try:
    from relax.distributed.ray import rollout
    from relax.distributed.ray.rollout import EngineGroupLifecycle

    HAS_DEPS = True
except ImportError:
    HAS_DEPS = False

from conftest import create_test_manager, make_engine_group, make_mock_engine, make_rollout_server, mock_ray_get

from relax.engine.inference.discovery import RoleSnapshot


pytestmark = pytest.mark.skipif(not HAS_DEPS, reason="Missing ray/sglang dependencies")


def _manager(monkeypatch, groups, status=None):
    monkeypatch.setattr(rollout.ray, "get", mock_ray_get)
    manager = create_test_manager(servers={"default": make_rollout_server(engine_groups=groups)})
    manager.status = status
    return manager


def _snapshot(manager) -> RoleSnapshot:
    return RoleSnapshot.from_dict(manager.get_inference_snapshot())


def _engines(*urls):
    return [make_mock_engine(url=url) for url in urls]


def test_rollout_snapshot_reports_router_and_logical_replicas(monkeypatch):
    single_node = make_engine_group(engines=_engines("http://n0:15000", "http://n0:15010"), num_gpus_per_engine=2)
    # 16 GPUs per engine on 8-GPU nodes: head and follower slots alternate.
    multi_node = make_engine_group(
        engines=_engines("http://n1:15000", "http://n2:15000", "http://n3:15000", "http://n4:15000"),
        num_gpus_per_engine=16,
        rank_offset=2,
    )
    snapshot = _snapshot(_manager(monkeypatch, [single_node, multi_node]))

    assert (snapshot.role, snapshot.default_model) == ("rollout", "default")
    model = snapshot.model("default")
    assert model.router_url == "http://127.0.0.1:3000"
    assert model.state.value == "ready"
    assert [(engine.engine_id, engine.base_url) for engine in model.engines] == [
        ("default/0", "http://n0:15000"),
        ("default/1", "http://n0:15010"),
        ("default/2", "http://n1:15000"),
        ("default/4", "http://n3:15000"),
    ]
    # Requests go through the router, never straight to a replica.
    assert not any(engine.direct_eligible for engine in model.engines)


def test_rollout_snapshot_excludes_pd_workers_from_replicas(monkeypatch):
    prefill = make_engine_group(engines=_engines("http://p0:15000"), worker_type="prefill")
    decode = make_engine_group(engines=_engines("http://d0:15000"), worker_type="decode", rank_offset=1)
    model = _snapshot(_manager(monkeypatch, [prefill, decode])).model("default")

    assert model.engines == ()
    assert [(worker.engine_id, worker.base_url) for worker in model.diagnostic_workers] == [
        ("default/prefill-0", "http://p0:15000"),
        ("default/decode-1", "http://d0:15000"),
    ]


def test_rollout_snapshot_state_follows_offload_and_dead_slots(monkeypatch):
    group = make_engine_group(engines=[make_mock_engine(url="http://n0:15000"), None], num_gpus_per_engine=2)
    manager = _manager(monkeypatch, [group])
    resident = _snapshot(manager)

    assert [engine.state.value for engine in resident.model("default").engines] == ["ready", "dead"]
    assert resident.model("default").engines[1].base_url is None

    manager.status = "offload"
    offloaded = _snapshot(manager)

    assert [engine.state.value for engine in offloaded.model("default").engines] == ["sleeping", "dead"]
    assert offloaded.model("default").state.value == "sleeping"
    assert offloaded.topology_revision == resident.topology_revision


def test_rollout_snapshot_tracks_scale_in(monkeypatch):
    base = make_engine_group(engines=_engines("http://n0:15000"), num_gpus_per_engine=2)
    scaled = make_engine_group(
        engines=_engines("http://n1:15000"), num_gpus_per_engine=2, rank_offset=1, is_scaled_out=True
    )
    manager = _manager(monkeypatch, [base, scaled])
    before = _snapshot(manager)

    scaled.lifecycle_status = EngineGroupLifecycle.DRAINING
    draining = _snapshot(manager)
    assert [engine.state.value for engine in draining.model("default").engines] == ["ready", "draining"]
    assert draining.topology_revision == before.topology_revision

    scaled.lifecycle_status = EngineGroupLifecycle.REMOVED
    removed = _snapshot(manager)
    assert [engine.engine_id for engine in removed.model("default").engines] == ["default/0"]
    assert removed.topology_revision == before.topology_revision + 1


def test_rollout_snapshot_leaves_legacy_engines_info_shape(monkeypatch):
    group = make_engine_group(engines=_engines("http://n0:15000"), num_gpus_per_engine=2)
    manager = _manager(monkeypatch, [group])

    legacy = manager.get_engines_info()

    assert set(legacy) == {"models", "total_engines"}
    assert set(legacy["models"]["default"]) == {"router_ip", "router_port", "engine_groups", "total_engines"}
    assert legacy["models"]["default"]["engine_groups"][0]["engines"][0]["url"] == "http://n0:15000"
