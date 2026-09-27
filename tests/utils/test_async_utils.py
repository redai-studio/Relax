# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from concurrent.futures import Future

import pytest

from relax.utils import async_utils


@pytest.mark.parametrize("from_loop_thread", [False, True])
def test_async_utils_shutdown_stops_loop_from_either_thread(monkeypatch, from_loop_thread: bool) -> None:
    inst = async_utils.AsyncLoopThread()
    monkeypatch.setattr(async_utils, "async_loop", inst)
    completed: Future = Future()

    def shutdown() -> None:
        try:
            async_utils.shutdown_async_loop()
        except Exception as exc:
            completed.set_exception(exc)
        else:
            completed.set_result(None)

    try:
        if from_loop_thread:
            inst.loop.call_soon_threadsafe(shutdown)
        else:
            shutdown()
        completed.result(timeout=5)
        inst._thread.join(timeout=5)
        assert not inst._thread.is_alive()
        assert async_utils.async_loop is None
        async_utils.shutdown_async_loop()
    finally:
        if inst._thread.is_alive():
            inst.loop.call_soon_threadsafe(inst.loop.stop)
            inst._thread.join(timeout=5)
        if not inst._thread.is_alive():
            inst.loop.close()
