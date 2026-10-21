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
    from relax.distributed.ray import rollout

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
        _shutdown_all_engines=Mock(side_effect=shutdown_engines),
    )
    try:
        if shutdown_fails:
            with pytest.raises(RuntimeError, match="engine shutdown failed"):
                rollout.RolloutManager.dispose.__func__(manager)
        else:
            rollout.RolloutManager.dispose.__func__(manager)

        assert not process.is_alive()
        assert rollout._LAUNCHED_ROUTER_PROCESSES == []
        manager._shutdown_all_engines.assert_called_once_with()
    finally:
        if process.is_alive():
            process.terminate()
            process.join(timeout=5)
