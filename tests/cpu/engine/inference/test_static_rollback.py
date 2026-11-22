# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from types import SimpleNamespace

import pytest

from relax.distributed.ray import inference_manager as module


@pytest.mark.parametrize("failure", [TimeoutError("cleanup timeout"), RuntimeError("actor unreachable")])
def test_static_rollback_retains_owner_until_confirmed_cleanup(monkeypatch, failure):
    owner = SimpleNamespace(dispose=SimpleNamespace(remote=lambda: object()))
    killed = []
    monkeypatch.setattr(module.ray, "kill", killed.append)

    def fail_get(*args, **kwargs):
        raise failure

    monkeypatch.setattr(module.ray, "get", fail_get)
    try:
        with pytest.raises(type(failure), match=str(failure)):
            module.rollback_static_inference_manager(owner)
        assert module._FAILED_ROLLBACK_OWNERS[id(owner)] is owner
        assert killed == []
        monkeypatch.setattr(module.ray, "get", lambda *args, **kwargs: None)
        module.rollback_static_inference_manager(owner)
        assert killed == [owner]
        assert id(owner) not in module._FAILED_ROLLBACK_OWNERS
    finally:
        module._FAILED_ROLLBACK_OWNERS.pop(id(owner), None)


@pytest.mark.asyncio
async def test_partial_static_start_and_failed_dispose_preserve_backend_and_fence(monkeypatch):
    from relax.distributed.ray.engine_pool import EnginePool
    from relax.engine.inference.types import Role

    implementation = EnginePool
    shutdowns = []
    killed = []

    def partial_init(backend, *args, **kwargs):
        backend.all_engines = []
        raise RuntimeError("engine startup failed")

    def failed_shutdown(backend):
        shutdowns.append(backend)
        raise TimeoutError("subprocess exit unconfirmed")

    monkeypatch.setattr(implementation, "__init__", partial_init)
    monkeypatch.setattr(implementation, "shutdown", failed_shutdown)
    monkeypatch.setattr(module.ray, "kill", killed.append)
    args = SimpleNamespace(
        genrm_model_path="judge-checkpoint",
        genrm_num_gpus_per_engine=1,
        genrm_num_gpus=1,
        genrm_engine_config={},
        genrm_sampling_config={},
        num_gpus_per_node=1,
    )
    owner = module.InferenceManager.__ray_metadata__.modified_class()
    with pytest.raises(TimeoutError, match="exit unconfirmed"):
        await owner.start_static("genrm", args, None, {})
    pool = owner.pool
    assert pool is not None
    backend = pool.backend
    assert shutdowns == [backend]
    with pytest.raises(TimeoutError, match="exit unconfirmed"):
        await owner.dispose()
    assert owner.pool is pool
    assert owner.pool.backend is backend
    assert "shutdown" in owner.ledger.fences
    assert killed == []
    with pytest.raises(RuntimeError, match="closing"):
        await owner.wait_ready(Role.GENRM)
    monkeypatch.setattr(implementation, "shutdown", lambda backend: None)
    await owner.dispose()
    assert owner.pool is None
    assert killed == []


def test_static_pool_retains_backend_when_shutdown_is_unconfirmed(monkeypatch):
    from relax.distributed.ray.static_inference_pool import StaticInferencePool

    killed = []

    def failed_shutdown():
        raise TimeoutError("exit unconfirmed")

    pool = StaticInferencePool.__new__(StaticInferencePool)
    backend = SimpleNamespace(shutdown=failed_shutdown)
    pool.backend = backend
    monkeypatch.setattr(module.ray, "kill", killed.append)
    with pytest.raises(TimeoutError, match="exit unconfirmed"):
        pool.dispose()
    assert pool.backend is backend
    assert killed == []
    pool.backend.shutdown = lambda: None
    pool.dispose()
    assert killed == []


@pytest.mark.asyncio
async def test_router_survivor_keeps_task_owner_pool_and_locks_for_retry(monkeypatch):
    from unittest.mock import Mock

    from relax.distributed.ray import policy_engine_pool
    from relax.distributed.ray.inference_pool import RolloutPool

    process = Mock()
    process.is_alive.return_value = True
    monkeypatch.setattr(policy_engine_pool, "_LAUNCHED_ROUTER_PROCESSES", [process])
    backend = policy_engine_pool.PolicyEnginePool.__new__(policy_engine_pool.PolicyEnginePool)
    backend._stop_eviction_monitor = Mock()
    backend._health_monitors = []
    backend._shutdown_all_engines = Mock()
    backend.rollout_engine_lock = object()
    backend._weight_sync_lock = object()
    pool = RolloutPool.__new__(RolloutPool)
    pool.backend = backend
    owner = module.InferenceManager.__ray_metadata__.modified_class()
    owner.pool = pool
    killed = []
    monkeypatch.setattr(module.ray, "kill", killed.append)
    with pytest.raises(RuntimeError, match="Router process exit is unconfirmed"):
        await owner.dispose()
    assert owner.pool is pool
    assert killed == []
    assert policy_engine_pool._LAUNCHED_ROUTER_PROCESSES == [process]
    process.is_alive.return_value = False
    await owner.dispose()
    assert owner.pool is None
    assert killed == [backend.rollout_engine_lock, backend._weight_sync_lock]
