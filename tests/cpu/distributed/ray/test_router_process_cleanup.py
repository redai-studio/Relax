# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import multiprocessing
import time
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from conftest import HAS_DEPS


pytestmark = pytest.mark.skipif(not HAS_DEPS, reason="Missing ray/sglang dependencies")


@pytest.mark.parametrize("shutdown_fails", [False, True])
def test_rollout_manager_dispose_stops_owned_router_even_if_engine_shutdown_fails(monkeypatch, shutdown_fails):
    from relax.distributed.ray import policy_engine_pool as rollout

    process = multiprocessing.Process(target=time.sleep, args=(30,))
    process.daemon = True
    process.start()
    monkeypatch.setattr(rollout, "_LAUNCHED_ROUTER_PROCESSES", [process])

    def shutdown_engines():
        assert process.is_alive()
        if shutdown_fails:
            raise RuntimeError("engine shutdown failed")

    manager = SimpleNamespace(
        _stop_eviction_monitor=Mock(),
        _health_monitors=[],
        _eval_judge=None,
        _shutdown_all_engines=Mock(side_effect=shutdown_engines),
    )
    try:
        if shutdown_fails:
            with pytest.raises(RuntimeError, match="engine shutdown failed"):
                rollout.PolicyEnginePool.dispose(manager)
        else:
            rollout.PolicyEnginePool.dispose(manager)

        assert not process.is_alive()
        assert rollout._LAUNCHED_ROUTER_PROCESSES == []
        manager._shutdown_all_engines.assert_called_once_with()
    finally:
        if process.is_alive():
            process.terminate()
            process.join(timeout=5)


@pytest.mark.asyncio
async def test_partial_policy_weight_load_is_released_before_next_phase():
    from relax.distributed.ray.policy_engine_pool import PolicyEnginePool

    calls = []
    pool = PolicyEnginePool.__new__(PolicyEnginePool)
    pool.status = "offload"
    pool.health_monitoring_pause = lambda: calls.append("pause")
    pool.servers = {
        "policy": SimpleNamespace(
            onload=lambda tags: calls.append(("load", tags)),
            offload=lambda: calls.append("release"),
        )
    }
    await pool.onload_weights()
    assert pool.status != "offload"
    await pool.offload()
    assert calls[-2:] == ["pause", "release"]
    assert pool.status == "offload"


@pytest.mark.asyncio
async def test_failed_partial_policy_load_does_not_skip_release():
    from relax.distributed.ray.policy_engine_pool import PolicyEnginePool

    calls = []

    def fail_load(tags):
        raise TimeoutError("second model load unconfirmed")

    pool = PolicyEnginePool.__new__(PolicyEnginePool)
    pool.status = "offload"
    pool.health_monitoring_pause = lambda: None
    pool.servers = {
        "first": SimpleNamespace(
            onload=lambda tags: calls.append("loaded"), offload=lambda: calls.append("release-first")
        ),
        "second": SimpleNamespace(onload=fail_load, offload=lambda: calls.append("release-second")),
    }
    with pytest.raises(TimeoutError, match="second model"):
        await pool.onload_weights()
    await pool.offload()
    assert calls == ["loaded", "release-first", "release-second"]
    assert pool.status == "offload"
