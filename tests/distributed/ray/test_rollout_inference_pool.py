# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""RolloutServer observations are the only evidence a model is READY."""

from typing import Any
from unittest.mock import MagicMock

import pytest
from conftest import (
    HAS_DEPS,
    AwaitableValue,
    create_test_manager,
    make_engine_group,
    make_mock_engine,
    make_rollout_server,
)


if HAS_DEPS:
    import ray

    from relax.backends.sglang.sglang_engine import EngineMemoryStateUnknown
    from relax.distributed.ray.inference_manager import WEIGHT_UPDATE_BACKEND
    from relax.engine.inference.config import InferenceModelSpec
    from relax.engine.inference.types import LifecycleState, Role, RouteMode, WeightSource


pytestmark = pytest.mark.skipif(not HAS_DEPS, reason="Missing ray/sglang dependencies")


def _observed_engine(version: str | None = "v1", **overrides: Any) -> Any:
    engine = make_mock_engine(weight_version=version)
    observation = {"healthy": True, "router_registered": True, "weight_version": version, "base_url": "http://w:1"}
    observation.update(overrides)
    engine.get_inference_observation.remote.return_value = AwaitableValue(observation)
    return engine


def _pool(groups: list[Any]) -> Any:
    pool = create_test_manager(servers={"default": make_rollout_server(engine_groups=groups)})
    pool.rollout_engine_lock = MagicMock()
    return pool


def _model(pool: Any) -> Any:
    pool.owner.refresh(Role.ROLLOUT, "default")
    return pool.owner.snapshot(Role.ROLLOUT).models[0]


@pytest.mark.parametrize("version", [None, "", "default"])
def test_rollout_observe_unknown_policy_version_never_ready(patch_ray_get: Any, version: str | None) -> None:
    model = _model(_pool([make_engine_group(engines=[_observed_engine(version)])]))
    assert model.state == LifecycleState.STARTING
    assert not model.admission
    assert model.required_weight_version is None


@pytest.mark.parametrize("missing", ["healthy", "router_registered"])
def test_rollout_observe_missing_evidence_never_ready(patch_ray_get: Any, missing: str) -> None:
    engine = _observed_engine()
    del engine.get_inference_observation.remote.return_value.value[missing]
    assert not _model(_pool([make_engine_group(engines=[engine])])).admission


def test_rollout_observe_mixed_versions_block_admission(patch_ray_get: Any) -> None:
    groups = [make_engine_group(engines=[_observed_engine("v1")]), make_engine_group(engines=[_observed_engine("v2")])]
    assert not _model(_pool(groups)).admission


def test_rollout_weight_update_closes_admission_until_completion(patch_ray_get: Any) -> None:
    pool = _pool([make_engine_group(engines=[_observed_engine("v1")])])
    assert _model(pool).admission
    pool.owner.begin_weight_update(Role.ROLLOUT, WEIGHT_UPDATE_BACKEND)
    assert not _model(pool).admission
    pool.owner.end_weight_update(Role.ROLLOUT, WEIGHT_UPDATE_BACKEND)
    assert pool.owner.snapshot(Role.ROLLOUT).models[0].admission


@pytest.mark.parametrize("decode_version,ready", [("v1", True), ("v2", False)])
def test_rollout_observe_pd_exposes_only_router_service(patch_ray_get: Any, decode_version: str, ready: bool) -> None:
    groups = [
        make_engine_group(engines=[_observed_engine("v1")], worker_type="prefill"),
        make_engine_group(engines=[_observed_engine(decode_version)], worker_type="decode", rank_offset=1),
    ]
    model = _model(_pool(groups))
    assert [replica.engine_id for replica in model.replicas] == ["default/pd-service"]
    assert model.replicas[0].base_url == model.router_url
    assert sorted(kind for kind, _ in model.pd_workers) == ["decode", "prefill"]
    assert model.admission is ready


def test_rollout_observe_direct_model_serves_without_a_router(patch_ray_get: Any) -> None:
    # A direct-routed engine (native diffusion) never joins a Router.
    engine = _observed_engine("3", router_registered=False)
    server = make_rollout_server(engine_groups=[make_engine_group(engines=[engine])])
    server.model_spec = InferenceModelSpec("default", "ckpt", route_mode=RouteMode.DIRECT)
    pool = create_test_manager(servers={"default": server})
    pool.rollout_engine_lock = MagicMock()

    model = _model(pool)

    assert engine.get_inference_observation.remote.call_args.kwargs == {"ensure_router": False}
    assert model.state == LifecycleState.READY and model.admission
    assert model.router_url is None
    (replica,) = model.replicas
    assert replica.state == LifecycleState.READY and replica.direct_eligible
    assert model.required_weight_version == "3"


def test_static_server_offload_is_idempotent_and_onload_reopens_every_engine(patch_ray_get: Any) -> None:
    engine = _observed_engine(None)
    server = make_rollout_server(engine_groups=[make_engine_group(engines=[engine])])
    server.static = True
    server.model_spec = InferenceModelSpec("default", "ckpt", weight_source=WeightSource.STATIC)
    engine.release_memory_occupation.remote.return_value = AwaitableValue(None)
    engine.resume_memory_occupation.remote.return_value = AwaitableValue(None)

    server.onload()
    server.offload()
    server.offload()
    server.onload()
    server.onload()

    assert engine.release_memory_occupation.remote.call_count == 1
    # The engine resumes only memory it released, so every onload may reopen
    # it; one still resident after a failed release is reopened this way.
    assert engine.resume_memory_occupation.remote.call_count == 3
    assert server.observe().state == LifecycleState.READY


def test_server_abort_requests_reaches_every_live_head_engine(patch_ray_get: Any) -> None:
    first, second = _observed_engine(), _observed_engine()
    for engine in (first, second):
        engine.abort_requests.remote.return_value = AwaitableValue(None)
    groups = [make_engine_group(engines=[first, None]), make_engine_group(engines=[second], rank_offset=2)]
    server = make_rollout_server(engine_groups=groups)

    server.abort_requests()

    first.abort_requests.remote.assert_called_once_with()
    second.abort_requests.remote.assert_called_once_with()


def test_static_server_retries_offload_after_a_failed_release(patch_ray_get: Any, monkeypatch: Any) -> None:
    engine = _observed_engine(None)
    server = make_rollout_server(engine_groups=[make_engine_group(engines=[engine])])
    server.static = True
    server.model_spec = InferenceModelSpec("default", "ckpt", weight_source=WeightSource.STATIC)
    server.onloaded = True
    engine.release_memory_occupation.remote.return_value = AwaitableValue(RuntimeError("busy"))
    get = ray.get

    def checked_get(ref: Any, **kwargs: Any) -> Any:
        result = get(ref, **kwargs)
        if isinstance(result, BaseException):
            raise result
        return result

    monkeypatch.setattr(ray, "get", checked_get)

    with pytest.raises(RuntimeError, match="busy"):
        server.offload()
    # The memory is still held, so the retry releases it again.
    engine.release_memory_occupation.remote.return_value = AwaitableValue(None)
    server.offload()

    assert engine.release_memory_occupation.remote.call_count == 2
    assert server.observe().state == LifecycleState.SLEEPING


def test_policy_weights_onload_retires_dead_engine_before_recovery(patch_ray_get: Any, monkeypatch: Any) -> None:
    dead, healthy, rebuilt = [_observed_engine() for _ in range(3)]
    dead.resume_memory_occupation.remote.return_value = AwaitableValue(ConnectionError("server exited"))
    group = make_engine_group(engines=[dead, healthy])
    group.pg = object()
    group.args.offload_rollout = True
    server = make_rollout_server(engine_groups=[group])
    server.model_spec = InferenceModelSpec("default", "ckpt", fault_tolerance_enabled=True)
    server.onloaded = False
    get = ray.get

    def checked_get(ref: Any, **kwargs: Any) -> Any:
        result = get(ref, **kwargs)
        if isinstance(result, BaseException):
            raise result
        return result

    monkeypatch.setattr(ray, "get", checked_get)
    monkeypatch.setattr(ray, "kill", MagicMock())

    def rebuild(cursors: dict) -> tuple:
        assert group.all_engines == [None, healthy]
        group.all_engines[0] = rebuilt
        group.num_new_engines = 1
        return [], cursors

    group.start_engines = MagicMock(side_effect=rebuild)
    server.onload(tags=["weights"])
    assert group.all_engines == [None, healthy]
    group.start_engines.assert_not_called()
    assert not server.onloaded

    server.recover()
    assert server.num_new_engines == 1
    healthy.resume_memory_occupation.remote.assert_called_once_with(tags=["weights"])
    rebuilt.release_memory_occupation.remote.assert_called_once_with()
    rebuilt.resume_memory_occupation.remote.assert_called_once_with(tags=["weights"])


@pytest.mark.parametrize("fault_tolerance,tags", [(False, ["weights"]), (True, None), (True, ["kv_cache"])])
def test_policy_onload_other_paths_still_propagate_failure(
    patch_ray_get: Any, fault_tolerance: bool, tags: list[str] | None
) -> None:
    group = make_engine_group()
    group.onload = MagicMock(side_effect=ConnectionError("server exited"))
    server = make_rollout_server(engine_groups=[group])
    server.model_spec = InferenceModelSpec("default", "ckpt", fault_tolerance_enabled=fault_tolerance)
    with pytest.raises(ConnectionError, match="server exited"):
        server.onload(tags=tags)


def _raising_get(monkeypatch: Any) -> MagicMock:
    """Make ``ray.get`` raise returned exceptions and stub ``ray.kill``."""
    get = ray.get

    def checked_get(ref: Any, **kwargs: Any) -> Any:
        result = get(ref, **kwargs)
        if isinstance(result, BaseException):
            raise result
        return result

    monkeypatch.setattr(ray, "get", checked_get)
    kill = MagicMock()
    monkeypatch.setattr(ray, "kill", kill)
    return kill


def _stopping_group(*results: Any) -> tuple[Any, Any, list[Any]]:
    engines = [make_mock_engine() for _ in results]
    for engine, result in zip(engines, results, strict=True):
        engine.shutdown.remote.return_value = AwaitableValue(result)
    group = make_engine_group(engines=engines)
    group.placement = object()
    return make_rollout_server(engine_groups=[group]), group, engines


def test_server_shutdown_keeps_an_engine_whose_shutdown_timed_out(patch_ray_get: Any, monkeypatch: Any) -> None:
    kill = _raising_get(monkeypatch)
    server, group, (hung, healthy) = _stopping_group(ray.exceptions.GetTimeoutError("timed out"), None)
    planner = MagicMock()

    with pytest.raises(RuntimeError, match="may still hold GPU memory"):
        server.shutdown(planner)

    # Killing the actor would orphan the SGLang processes holding the GPUs.
    assert group.all_engines == [hung, None]
    kill.assert_called_once_with(healthy)
    planner.release.assert_not_called()


def test_server_shutdown_keeps_an_engine_that_could_not_be_killed(patch_ray_get: Any, monkeypatch: Any) -> None:
    kill = _raising_get(monkeypatch)
    kill.side_effect = RuntimeError("GCS unavailable")
    server, group, (engine,) = _stopping_group(None)

    with pytest.raises(RuntimeError, match="may still hold GPU memory"):
        server.shutdown(MagicMock())
    assert group.all_engines == [engine]


def test_server_shutdown_accepts_an_already_dead_engine(patch_ray_get: Any, monkeypatch: Any) -> None:
    _raising_get(monkeypatch)
    server, group, _ = _stopping_group(ray.exceptions.RayActorError())
    planner = MagicMock()
    planner.release.return_value.remove_placement_group = False

    server.shutdown(planner)
    assert group.all_engines == [None]
    planner.release.assert_called_once_with(group.placement)


def test_engine_recovery_still_drops_an_engine_whose_shutdown_timed_out(patch_ray_get: Any, monkeypatch: Any) -> None:
    _raising_get(monkeypatch)
    _, group, _ = _stopping_group(ray.exceptions.GetTimeoutError("timed out"))
    group.shutdown_engines({0})
    assert group.all_engines == [None]


def _static_server(*engines: Any) -> Any:
    group = make_engine_group(engines=list(engines))
    server = make_rollout_server(engine_groups=[group])
    server.static = True
    server.onloaded = True
    server.model_spec = InferenceModelSpec("judge", "ckpt", weight_source=WeightSource.STATIC)
    return server, group


def _timed_out_release(stop: Any) -> tuple[Any, Any, Any, Any]:
    hung, healthy = make_mock_engine(), make_mock_engine()
    hung.release_memory_occupation.remote.return_value = AwaitableValue(ray.exceptions.GetTimeoutError("timed out"))
    hung.shutdown.remote.return_value = AwaitableValue(stop)
    healthy.release_memory_occupation.remote.return_value = AwaitableValue(None)
    server, group = _static_server(hung, healthy)
    return server, group, hung, healthy


def test_static_offload_fails_while_a_timed_out_release_may_hold_gpus(patch_ray_get: Any, monkeypatch: Any) -> None:
    kill = _raising_get(monkeypatch)
    server, group, hung, healthy = _timed_out_release(ray.exceptions.GetTimeoutError("shutdown timed out"))

    with pytest.raises(RuntimeError, match="may still hold GPU memory"):
        server.offload()
    assert group.all_engines == [hung, healthy] and server.onloaded
    kill.assert_not_called()


def test_static_offload_retires_a_timed_out_release_once_shut_down(patch_ray_get: Any, monkeypatch: Any) -> None:
    _raising_get(monkeypatch)
    server, group, _, healthy = _timed_out_release(None)

    server.offload()
    assert group.all_engines == [None, healthy] and not server.onloaded


@pytest.mark.parametrize("static", [True, False])
def test_onload_keeps_a_timed_out_engine_instead_of_rebuilding_on_its_gpus(
    patch_ray_get: Any, monkeypatch: Any, static: bool
) -> None:
    kill = _raising_get(monkeypatch)
    hung, healthy = make_mock_engine(), make_mock_engine()
    hung.resume_memory_occupation.remote.return_value = AwaitableValue(TimeoutError("resume timed out"))
    hung.shutdown.remote.return_value = AwaitableValue(TimeoutError("shutdown timed out"))
    healthy.resume_memory_occupation.remote.return_value = AwaitableValue(None)
    server, group = _static_server(hung, healthy)
    server.static = static
    server.onloaded = False
    server.model_spec = InferenceModelSpec("judge", "ckpt", fault_tolerance_enabled=True)
    server.recover = MagicMock()

    # Serving degraded beats failing the onload; nothing starts on the hung engine's GPUs.
    server.onload(None if static else ["weights"])

    assert group.all_engines == [hung, healthy]
    server.recover.assert_not_called()
    kill.assert_not_called()


def _policy_server(*engines: Any) -> Any:
    group = make_engine_group(engines=list(engines))
    server = make_rollout_server(engine_groups=[group])
    server.onloaded = True
    return server, group


def test_policy_offload_hears_every_engine_and_stays_onloaded_after_a_failed_release(
    patch_ray_get: Any, monkeypatch: Any
) -> None:
    _raising_get(monkeypatch)
    failed, released = make_mock_engine(), make_mock_engine()
    failed.release_memory_occupation.remote.return_value = AwaitableValue(RuntimeError("router still lists it"))
    released.release_memory_occupation.remote.return_value = AwaitableValue(None)
    server, group = _policy_server(failed, released)

    with pytest.raises(RuntimeError, match="router still lists it"):
        server.offload()
    # Both engines answered; the pool is still onloaded, so a retry releases
    # again (the engine that already released skips it).
    released.release_memory_occupation.remote.assert_called_once()
    assert server.onloaded and group.all_engines == [failed, released]

    failed.release_memory_occupation.remote.return_value = AwaitableValue(None)
    server.offload()
    assert not server.onloaded
    assert failed.release_memory_occupation.remote.call_count == 2


def test_policy_offload_releases_weights_left_by_a_weights_only_onload(patch_ray_get: Any, monkeypatch: Any) -> None:
    _raising_get(monkeypatch)
    engine = make_mock_engine()
    engine.release_memory_occupation.remote.return_value = AwaitableValue(None)
    server, _ = _policy_server(engine)
    # A weights-only onload leaves the pool not onloaded, with weights held.
    server.onloaded = False

    server.offload()
    engine.release_memory_occupation.remote.assert_called_once()


@pytest.mark.parametrize("static", [True, False])
@pytest.mark.parametrize("stopped", [True, False])
def test_offload_retires_an_engine_whose_release_was_not_confirmed(
    patch_ray_get: Any, monkeypatch: Any, static: bool, stopped: bool
) -> None:
    kill = _raising_get(monkeypatch)
    unknown, released = make_mock_engine(), make_mock_engine()
    unknown.release_memory_occupation.remote.return_value = AwaitableValue(EngineMemoryStateUnknown("lost reply"))
    unknown.shutdown.remote.return_value = AwaitableValue(None if stopped else TimeoutError("shutdown timed out"))
    released.release_memory_occupation.remote.return_value = AwaitableValue(None)
    server, group = _static_server(unknown, released) if static else _policy_server(unknown, released)

    if stopped:
        # Stopping the engine freed its memory; it is never released again.
        server.offload()
        assert group.all_engines == [None, released] and not server.onloaded
        kill.assert_called_once_with(unknown)
        return
    with pytest.raises(RuntimeError, match="may still hold GPU memory"):
        server.offload()
    assert group.all_engines == [unknown, released] and server.onloaded
    kill.assert_not_called()


def _recorded_group(engine: Any, record_value: Any = None) -> Any:
    from ray.core.generated.gcs_pb2 import ActorTableData

    from relax.distributed.ray import rollout

    engine._get_local_state.return_value = ActorTableData.ALIVE
    group = make_engine_group(engines=[engine])
    value = record_value if record_value is not None else {"url": "http://w:1", "pid": 1, "node_id": "n"}
    group.engine_init_records = {0: rollout._EngineInitRecord(value)}
    return group


def test_rollout_observe_confirmed_slots_publish_a_complete_observation(patch_ray_get: Any) -> None:
    pool = _pool([_recorded_group(_observed_engine())])
    model = _model(pool)
    assert model.admission and model.issues == ()
    assert pool.owner.snapshot(Role.ROLLOUT).to_dict()["observation"] == {"complete": True, "issues": []}


def test_rollout_observe_skips_the_probe_while_init_is_pending(patch_ray_get: Any) -> None:
    from unittest.mock import patch

    from relax.distributed.ray import rollout

    pending = object()
    get = rollout.ray.get

    def ray_get(ref: Any, **kwargs: Any) -> Any:
        if ref is pending:
            raise ray.exceptions.GetTimeoutError("init is still running")
        return get(ref, **kwargs)

    engine = _observed_engine()
    pool = _pool([_recorded_group(engine, pending)])
    engine.get_inference_observation.remote.reset_mock()
    # Scoped inside the test: restoring after patch_ray_get would leak its mock.
    with patch.object(rollout.ray, "get", ray_get):
        model = _model(pool)
    assert model.replicas[0].state == LifecycleState.STARTING
    assert model.issues == (("default/0/0", "init_pending"),)
    engine.get_inference_observation.remote.assert_not_called()
    observation = pool.owner.snapshot(Role.ROLLOUT).to_dict()["observation"]
    assert observation == {"complete": False, "issues": [{"scope": "default/0/0", "reason": "init_pending"}]}


def test_rollout_observe_reports_a_dead_actor_still_in_its_slot(patch_ray_get: Any) -> None:
    from ray.core.generated.gcs_pb2 import ActorTableData

    engine = _observed_engine()
    group = _recorded_group(engine)
    pool = _pool([group])
    engine._get_local_state.return_value = ActorTableData.DEAD
    engine.get_inference_observation.remote.reset_mock()
    model = _model(pool)
    assert model.replicas[0].state == LifecycleState.DEAD
    assert model.issues == (("default/0/0", "actor_dead_unreconciled"),)
    engine.get_inference_observation.remote.assert_not_called()


def test_rollout_topology_change_marks_the_observation_incomplete(patch_ray_get: Any) -> None:
    pool = _pool([_recorded_group(_observed_engine())])
    _model(pool)
    pool._events.changed("default")
    model = pool.owner.snapshot(Role.ROLLOUT).models[0]
    assert model.issues == (("default", "topology_unsettled"),)
    assert _model(pool).issues == ()


def _replacement_engine(actor_id: str) -> Any:
    """An engine joining the Router only when the observation asks it to."""
    engine = make_mock_engine(weight_version="v2")
    engine._actor_id.hex.return_value = actor_id

    def observe(ensure_router: bool = False) -> AwaitableValue:
        return AwaitableValue(
            {
                "healthy": True,
                "router_registered": ensure_router,
                "weight_version": "v2",
                "base_url": f"http://{actor_id}:1",
            }
        )

    engine.get_inference_observation.remote.side_effect = observe
    return engine


def test_recovered_replica_waits_for_every_pp_source_before_publication(patch_ray_get: Any) -> None:
    seed, recovered = _recorded_group(_replacement_engine("seed")), _recorded_group(_replacement_engine("recovered"))
    # A recovered generation, which has already seen version v2 from one bucket.
    recovered.engine_init_records[0].awaiting_weight_sync = True
    pool = _pool([seed, recovered])

    # One PP source sent to {seed, recovered}, another pruned the replacement.
    synced = sorted({"seed", "recovered"} & {"seed"})
    assert pool.confirm_weight_sync(synced) == 0
    model = _model(pool)
    head = recovered.all_engines[0]
    assert head.get_inference_observation.remote.call_args.kwargs == {"ensure_router": False}
    # The seed still serves; the replacement neither joins the Router nor is READY.
    assert [replica.state for replica in model.replicas] == [LifecycleState.READY, LifecycleState.STARTING]
    assert ("default/1/0", "weight_sync_unconfirmed") in model.issues

    assert pool.confirm_weight_sync(["recovered"]) == 1
    model = _model(pool)
    assert head.get_inference_observation.remote.call_args.kwargs == {"ensure_router": True}
    assert model.admission and model.issues == ()
    assert [replica.state for replica in model.replicas] == [LifecycleState.READY, LifecycleState.READY]


def test_weight_sync_confirmation_never_releases_a_later_generation(patch_ray_get: Any) -> None:
    group = _recorded_group(_replacement_engine("generation-2"))
    group.engine_init_records[0].awaiting_weight_sync = True
    pool = _pool([group])
    # A sync that reached the slot's previous generation does not count.
    assert pool.confirm_weight_sync(["generation-1"]) == 0
    assert group.engine_init_records[0].awaiting_weight_sync


@pytest.mark.parametrize(
    ("overrides", "weight_source", "awaits"),
    [
        ({"fully_async": False}, "dcs", False),
        ({"fully_async": True}, "dcs", True),
        # Nothing confirms these syncs, so a marked replacement would never serve.
        ({"fully_async": True, "hybrid": True}, "dcs", False),
        ({"fully_async": True, "lora_rank": 8, "lora_adapter_mode": True}, "dcs", False),
        ({"fully_async": True}, "static", False),
    ],
    ids=["colocate", "fully-async", "hybrid", "lora-adapter", "static-weights"],
)
def test_recovery_marks_dcs_replacements_until_their_sync_is_confirmed(
    patch_ray_get: Any, monkeypatch: Any, overrides: dict[str, Any], weight_source: str, awaits: bool
) -> None:
    from conftest import make_mock_args

    from relax.distributed.ray import rollout

    args = make_mock_args(num_gpus_per_node=1, offload_rollout=False, **overrides)
    survivor = make_mock_engine()
    group = make_engine_group(args=args, engines=[survivor, None])
    group.pg = (object(), [0, 1], [0, 1])
    replacement = make_mock_engine()

    def start(cursors):
        group.all_engines[1] = replacement
        group.engine_init_records[1] = rollout._EngineInitRecord({"url": "http://r:1", "pid": 1, "node_id": "n"})
        group.num_new_engines = 1
        return [], cursors

    monkeypatch.setattr(group, "start_engines", start)
    server = make_rollout_server(engine_groups=[group])
    server.model_spec = InferenceModelSpec("default", "ckpt", weight_source=weight_source)

    server.recover()

    assert group.engine_init_records[1].awaiting_weight_sync is awaits
    assert 0 not in group.engine_init_records or not group.engine_init_records[0].awaiting_weight_sync


def test_rollout_observe_keeps_a_failed_probe_as_incomplete_evidence(patch_ray_get: Any, monkeypatch: Any) -> None:
    from relax.utils.metrics import sglang_exporter

    engine = _observed_engine()
    engine.get_inference_observation.remote.return_value = None
    engine.get_inference_observation.remote.side_effect = ray.exceptions.GetTimeoutError("probe timed out")
    pool = _pool([_recorded_group(engine)])

    model = _model(pool)

    # Unknown, not dead: it serves nothing, but its endpoint stays known.
    (replica,) = model.replicas
    assert replica.state == LifecycleState.STARTING and replica.base_url == "http://w:1"
    assert not model.admission
    assert ("default/0/0", "probe_failed") in model.issues
    discovery = pool.owner.snapshot(Role.ROLLOUT).to_dict()
    assert discovery["observation"]["complete"] is False
    engines, complete = sglang_exporter._discover_engines(discovery)
    assert [found.url for found in engines] == ["http://w:1"] and complete is False


def _multinode_static_server(*engines: Any) -> tuple[Any, Any]:
    """A static pool whose engines span two 8-GPU nodes each (TP=16): slots
    alternate head, follower."""
    from conftest import make_mock_args

    args = make_mock_args(num_gpus_per_node=8)
    group = make_engine_group(args=args, engines=list(engines), num_gpus_per_engine=16)
    server = make_rollout_server(engine_groups=[group])
    server.static = True
    server.onloaded = True
    server.model_spec = InferenceModelSpec("teacher", "ckpt", weight_source=WeightSource.STATIC)
    return server, group


def test_multinode_static_pool_serves_and_releases_through_heads_only(patch_ray_get: Any, monkeypatch: Any) -> None:
    _raising_get(monkeypatch)
    heads = [make_mock_engine(url=f"http://head-{i}:1") for i in range(2)]
    followers = [make_mock_engine(url=None) for _ in range(2)]
    for engine in (*heads, *followers):
        engine.release_memory_occupation.remote.return_value = AwaitableValue(None)
    server, group = _multinode_static_server(heads[0], followers[0], heads[1], followers[1])

    # A follower serves no HTTP and holds no memory of its own to release.
    assert server.get_urls() == ["http://head-0:1", "http://head-1:1"]
    server.offload()
    for head in heads:
        head.release_memory_occupation.remote.assert_called_once_with()
    for follower in followers:
        follower.release_memory_occupation.remote.assert_not_called()
    assert group.all_engines == [heads[0], followers[0], heads[1], followers[1]]


def test_multinode_static_pool_retires_a_dead_replica_as_a_whole(patch_ray_get: Any, monkeypatch: Any) -> None:
    _raising_get(monkeypatch)
    dead_head, follower, head, other_follower = (make_mock_engine() for _ in range(4))
    dead_head.release_memory_occupation.remote.return_value = AwaitableValue(ray.exceptions.RayActorError("node lost"))
    head.release_memory_occupation.remote.return_value = AwaitableValue(None)
    server, group = _multinode_static_server(dead_head, follower, head, other_follower)

    server.offload()

    # The follower of the dead replica goes with it; the other replica stays.
    assert group.all_engines == [None, None, head, other_follower]
    follower.shutdown.remote.assert_called_once_with()
    other_follower.shutdown.remote.assert_not_called()
    assert not server.onloaded


def test_multinode_teacher_model_names_one_replica_per_node_pair() -> None:
    from types import SimpleNamespace

    from relax.engine.inference.config import EngineGroupSpec

    args = SimpleNamespace(
        num_gpus_per_node=8, rollout_num_gpus_per_engine=16, sglang_hf_checkpoint=None, hf_checkpoint="ckpt"
    )
    spec = InferenceModelSpec(
        "teacher", "ckpt", engine_groups=[EngineGroupSpec("regular", 32, 16)], weight_source=WeightSource.STATIC
    )

    replicas = spec.resolved(args).engine_groups[0].topology.replicas

    assert [(replica.replica_id, replica.node_ranks) for replica in replicas] == [
        ("teacher/replica-0", (0, 1)),
        ("teacher/replica-2", (2, 3)),
    ]
    partial = InferenceModelSpec("teacher", "ckpt", engine_groups=[EngineGroupSpec("regular", 24, 12)])
    with pytest.raises(ValueError, match="complete nodes"):
        partial.resolved(args)
