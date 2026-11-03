# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import multiprocessing
import time
from unittest.mock import Mock

import pytest
from conftest import HAS_DEPS

from relax.inference.manager import InferenceManager


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

    manager = object.__new__(rollout.RolloutManager.__ray_metadata__.modified_class)
    manager.inference = InferenceManager("rollout")
    manager._stop_eviction_monitor = Mock()
    manager._health_monitors = []
    manager._shutdown_all_engines = Mock(side_effect=shutdown_engines)
    try:
        if shutdown_fails:
            with pytest.raises(RuntimeError, match="engine shutdown failed"):
                manager.dispose()
        else:
            manager.dispose()

        assert not process.is_alive()
        assert rollout._LAUNCHED_ROUTER_PROCESSES == []
        manager._shutdown_all_engines.assert_called_once_with()
        assert manager.inference.state == ("failed" if shutdown_fails else "dead")
    finally:
        if process.is_alive():
            process.terminate()
            process.join(timeout=5)
