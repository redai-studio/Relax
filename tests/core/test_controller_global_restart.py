# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import threading

import pytest


try:
    from relax.core.controller import Controller
except (ImportError, AssertionError) as exc:
    pytest.skip(f"relax.core.controller requires the full runtime environment: {exc}", allow_module_level=True)


def test_restart_cycle_ack_is_published_after_old_state_is_cleared():
    controller = object.__new__(Controller)
    controller._restart_lock = threading.RLock()
    controller._restart_done_event = threading.Event()
    controller._restart_done_event.set()
    controller._restart_mode = "local"
    controller._restart_error = None
    controller._restarting = True

    class StartNextCycleOnAck:
        def set(self):
            assert controller._restarting is False
            assert controller._restart_mode is None
            controller._restarting = True
            controller._restart_mode = "global"

    controller._restart_consumed_event = StartNextCycleOnAck()

    restart_mode, restart_error = controller._consume_restart_cycle()

    assert restart_mode == "local"
    assert restart_error is None
    assert controller._restarting is True
    assert controller._restart_mode == "global"
