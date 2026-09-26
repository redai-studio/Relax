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

    from relax.engine.inference.config import InferenceModelSpec
    from relax.engine.inference.types import LifecycleState, Role, WeightSource


pytestmark = pytest.mark.skipif(not HAS_DEPS, reason="Missing ray/sglang dependencies")


def _observed_engine(version: str | None = "v1", **overrides: Any) -> Any:
    engine = make_mock_engine(weight_version=version)
    observation = {"healthy": True, "router_registered": True, "weight_version": version, "base_url": "http://w:1"}
    observation.update(overrides)
    engine.get_inference_observation.remote.return_value = AwaitableValue(observation)
    return engine


def _pool(groups: list[Any]) -> Any:
    pool = create_test_manager(servers={"default": make_rollout_server(engine_groups=groups)})
    pool.status = "onload"
    pool.rollout_engine_lock = MagicMock()
    return pool


def _model(pool: Any) -> Any:
    pool.refresh_inference_state()
    return pool.inference_manager.snapshot(Role.ROLLOUT).models[0]


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
    pool.invalidate_inference_state()
    assert not _model(pool).admission
    pool.complete_inference_weight_update()
    assert pool.inference_manager.snapshot(Role.ROLLOUT).models[0].admission


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


def test_static_server_offload_and_onload_are_idempotent(patch_ray_get: Any) -> None:
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
    assert engine.resume_memory_occupation.remote.call_count == 1
    assert server.observe().state == LifecycleState.READY


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
