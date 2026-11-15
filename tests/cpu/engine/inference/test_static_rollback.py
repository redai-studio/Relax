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
    from relax.distributed.ray.genrm import GenRMManager
    from relax.engine.inference.types import Role

    implementation = GenRMManager.__ray_metadata__.modified_class
    lock = object()
    shutdowns = []
    killed = []

    def partial_init(backend, *args, **kwargs):
        backend.all_engines = []
        backend.genrm_engine_lock = lock
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
    assert backend.genrm_engine_lock is lock
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
    assert killed == [lock]


def test_static_pool_keeps_lock_when_backend_shutdown_is_unconfirmed(monkeypatch):
    from relax.distributed.ray.static_inference_pool import StaticInferencePool

    lock = object()
    killed = []

    def failed_shutdown():
        raise TimeoutError("exit unconfirmed")

    pool = StaticInferencePool.__new__(StaticInferencePool)
    pool.backend = SimpleNamespace(shutdown=failed_shutdown, genrm_engine_lock=lock)
    monkeypatch.setattr(module.ray, "kill", killed.append)
    with pytest.raises(TimeoutError, match="exit unconfirmed"):
        pool.dispose()
    assert pool.backend.genrm_engine_lock is lock
    assert killed == []
    pool.backend.shutdown = lambda: None
    pool.dispose()
    assert killed == [lock]
