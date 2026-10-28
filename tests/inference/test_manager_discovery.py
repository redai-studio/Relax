# Copyright (c) 2026 Relax Authors. All Rights Reserved.


from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest


@pytest.fixture
def static_pool(monkeypatch):
    from relax.distributed.ray import inference_manager as module

    manager = module.InferenceManager(
        SimpleNamespace(), num_slots=4, nodes_per_engine=2, engine_actor_cls=object, skip_init=True
    )
    manager._inference_preserves_weights = True
    engines = [MagicMock() for _ in range(4)]
    for rank, engine in enumerate(engines):
        engine.health_process.remote.return_value = True
        engine.health_generate.remote.return_value = True
        engine.get_url.remote.return_value = f"http://head-{rank}.example:15000"
    manager.all_engines = engines
    monkeypatch.setattr(module.ray, "get", lambda refs, timeout=None: refs)
    for key, workers in manager._inference_replicas().items():
        manager._inference_observation.initialized(key, workers, weights_ready=True)
    return manager, engines, module


def _model(snapshot):
    return snapshot["models"]["__default__"]


def test_static_discovery_publishes_only_complete_logical_heads(static_pool):
    manager, engines, _module = static_pool

    snapshot = manager.get_inference_snapshot()

    model = _model(snapshot)
    assert model["weight_source"] == "STATIC"
    assert model["weight_version"] is None
    assert model["state"] == "READY"
    assert [replica["base_url"] for replica in model["engines"]] == [
        "http://head-0.example:15000",
        "http://head-2.example:15000",
    ]
    assert all(replica["direct_eligible"] for replica in model["engines"])
    for engine in engines:
        engine.health_process.remote.assert_called_once()
    engines[1].get_url.remote.assert_not_called()
    engines[3].get_url.remote.assert_not_called()
    assert manager.get_inference_snapshot()["topology_revision"] == snapshot["topology_revision"]


@pytest.mark.parametrize("failure", ["missing", "dead_process", "unknown_process"])
def test_follower_failure_fences_its_entire_replica(static_pool, failure):
    manager, engines, _module = static_pool
    before = manager.get_inference_snapshot()
    if failure == "missing":
        manager.all_engines[1] = None
    else:
        engines[1].health_process.remote.return_value = False if failure == "dead_process" else None

    after = manager.get_inference_snapshot()

    failed, survivor = _model(after)["engines"]
    assert failed["state"] != "READY"
    assert not failed["direct_eligible"]
    assert failed["base_url"] is None
    assert survivor["state"] == "READY"
    assert after["topology_revision"] > before["topology_revision"]


def test_static_partial_restore_requires_remaining_tags_before_readiness(static_pool):
    manager, engines, _module = static_pool
    initial = manager.get_inference_snapshot()

    manager.offload()
    asleep = manager.get_inference_snapshot()
    assert _model(asleep)["state"] == "SLEEPING"
    for engine in engines:
        engine.health_process.remote.reset_mock()
    manager.get_inference_snapshot()
    for engine in engines:
        engine.health_process.remote.assert_not_called()

    manager.onload(tags=["weights"])
    assert not manager.is_onloaded()
    partial = manager.get_inference_snapshot()
    assert _model(partial)["state"] == "ONLOADING"
    manager.onload()
    assert _model(manager.get_inference_snapshot())["state"] == "READY"
    assert partial["topology_revision"] > asleep["topology_revision"] > initial["topology_revision"]


def test_full_restore_without_weight_backup_remains_unroutable(static_pool):
    manager, _engines, _module = static_pool
    manager._inference_preserves_weights = False

    manager.offload()
    manager.onload()

    assert _model(manager.get_inference_snapshot())["state"] == "ONLOADING"


def test_lifecycle_cycle_invalidates_revision_without_intermediate_read(static_pool):
    manager, _engines, _module = static_pool
    before = manager.get_inference_snapshot()

    manager.offload()
    manager.onload()
    after = manager.get_inference_snapshot()

    assert _model(before) == _model(after)
    assert after["topology_revision"] > before["topology_revision"]


def test_failed_offload_publishes_failure_without_fake_release(static_pool, monkeypatch):
    manager, _engines, _module = static_pool

    def fail(*args, **kwargs):
        raise ValueError("release failed")

    monkeypatch.setattr(manager, "_fanout", fail)
    with pytest.raises(ValueError, match="release failed"):
        manager.offload()

    assert not manager.is_onloaded()
    assert _model(manager.get_inference_snapshot())["state"] == "FAILED"


def test_replaced_worker_changes_generation_and_requires_new_readiness(static_pool):
    manager, _engines, _module = static_pool
    before = manager.get_inference_snapshot()
    manager.all_engines[1] = MagicMock()

    after = manager.get_inference_snapshot()

    old_replica = _model(before)["engines"][0]
    new_replica = _model(after)["engines"][0]
    assert new_replica["engine_id"] == old_replica["engine_id"]
    assert new_replica["generation"] > old_replica["generation"]
    assert new_replica["state"] == "STARTING"


def test_endpoint_change_advances_generation(static_pool):
    manager, engines, _module = static_pool
    before = manager.get_inference_snapshot()
    engines[0].get_url.remote.return_value = "http://replacement.example:16000"

    after = manager.get_inference_snapshot()

    assert _model(after)["engines"][0]["generation"] > _model(before)["engines"][0]["generation"]
    assert after["topology_revision"] > before["topology_revision"]


@pytest.fixture
def rollout_pool(monkeypatch):
    pytest.importorskip("sglang")

    import threading

    from relax.distributed.ray import rollout
    from relax.distributed.ray.inference_manager import _InferenceObservation

    manager = object.__new__(rollout.RolloutManager.__ray_metadata__.modified_class)
    manager.args = SimpleNamespace(fully_async=False, hybrid=False)
    manager._engine_lifecycle_lock = threading.RLock()
    manager._inference_observation = _InferenceObservation("rollout")
    manager._training_weight_updating = False
    manager._scale_out_weight_updating = False
    engine = MagicMock()
    engine.health_process.remote.return_value = True
    engine.health_generate.remote.return_value = True
    engine.get_url.remote.return_value = "http://policy.example:15000"
    engine.get_weight_version.remote.return_value = "7"
    group = rollout.EngineGroup(
        args=SimpleNamespace(num_gpus_per_node=1),
        pg=None,
        all_engines=[engine],
        num_gpus_per_engine=1,
        num_new_engines=0,
    )
    manager.servers = {
        "policy": rollout.RolloutServer(
            engine_groups=[group], router_ip="router.example", router_port=16000, model_name="policy"
        )
    }
    monkeypatch.setattr(rollout.ray, "get", lambda refs, timeout=None: refs)
    for key, workers in manager._inference_replicas().items():
        manager._inference_observation.initialized(key, workers, weights_ready=False)
    return manager, group, engine


def test_rollout_publication_acknowledgement_is_required(rollout_pool):
    manager, _group, _engine = rollout_pool
    assert manager.get_inference_snapshot()["models"]["policy"]["state"] == "ONLOADING"

    manager.mark_inference_weights_ready()
    ready = manager.get_inference_snapshot()["models"]["policy"]
    assert ready["state"] == "READY"
    assert ready["weight_version"] == "7"
    assert ready["route_mode"] == "SGLANG_ROUTER"
    assert not ready["engines"][0]["direct_eligible"]

    manager.mark_inference_weights_updating()
    assert manager.get_inference_snapshot()["models"]["policy"]["state"] == "ONLOADING"


def test_rollout_discovery_honors_global_served_model_name(rollout_pool):
    manager, group, _engine = rollout_pool
    manager.args.sglang_served_model_name = "published-policy"
    group.sglang_overrides["model_path"] = "checkpoint-path"
    manager.mark_inference_weights_ready()
    snapshot = manager.get_inference_snapshot()
    assert snapshot["models"]["policy"]["served_model_name"] == "published-policy"
    assert snapshot["routing"]["aliases"]["published-policy"] == "policy"


def test_pd_discovery_never_exposes_stage_workers(rollout_pool):
    manager, group, _engine = rollout_pool
    group.worker_type = "prefill"
    manager.mark_inference_weights_ready()

    model = manager.get_inference_snapshot()["models"]["policy"]

    assert model["engines"] == []
    assert model["state"] != "READY"


def _route_target(snapshot):
    from relax.inference.routing import RouteResolver

    return RouteResolver().resolve(snapshot, model=None, route_key=None, affinity_key=None)


def test_external_rollout_without_local_process_becomes_routable(rollout_pool):
    manager, _group, engine = rollout_pool
    manager.args.rollout_external = True
    engine.health_process.remote.return_value = None
    manager.mark_inference_weights_ready()

    snapshot = manager.get_inference_snapshot()

    assert snapshot["models"]["policy"]["state"] == "READY"
    engine.health_process.remote.assert_not_called()
    engine.health_generate.remote.assert_called()
    assert _route_target(snapshot).base_url == "http://router.example:16000"


def test_external_scale_out_group_skips_process_probe(rollout_pool):
    manager, group, engine = rollout_pool
    group.external_engines = True
    engine.health_process.remote.return_value = None
    manager.mark_inference_weights_ready()

    assert manager.get_inference_snapshot()["models"]["policy"]["state"] == "READY"
    engine.health_process.remote.assert_not_called()


def test_custom_rollout_engine_without_process_probe_becomes_routable(rollout_pool, monkeypatch):
    from relax.distributed.ray import rollout

    class NativeEngine:
        def health_generate(self):
            return True

        def get_url(self):
            return "http://native.example:15000"

    manager, _group, engine = rollout_pool
    monkeypatch.setattr(rollout, "_resolve_rollout_engine_class", lambda args: NativeEngine)
    engine.health_process.remote.side_effect = AttributeError("health_process")
    manager.mark_inference_weights_ready()

    snapshot = manager.get_inference_snapshot()

    assert snapshot["models"]["policy"]["state"] == "READY"
    assert _route_target(snapshot).model_id == "policy"


def test_local_rollout_unknown_process_still_fences_replica(rollout_pool):
    manager, _group, engine = rollout_pool
    engine.health_process.remote.return_value = None
    manager.mark_inference_weights_ready()

    assert manager.get_inference_snapshot()["models"]["policy"]["state"] != "READY"
