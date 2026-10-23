# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Exercise final tracking persistence without external services or CUDA."""

import ast
import asyncio
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict
from unittest.mock import Mock


ROOT = Path(__file__).resolve().parents[2]


def _finish_method():
    path = ROOT / "relax/utils/metrics/service.py"
    tree = ast.parse(path.read_text())
    service = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "MetricsService")
    method = next(
        node for node in service.body if isinstance(node, ast.AsyncFunctionDef) and node.name == "finish_tracking"
    )
    namespace = {"Any": Any, "Dict": Dict, "asyncio": asyncio, "logger": Mock(), "wandb": Mock()}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"), namespace)
    return namespace


def test_finish_persists_last_step_once():
    pending = [(9, "perf/actor_train_tok_per_s_per_card", 700.0)]
    persisted = []

    def close_reporter():
        persisted.extend(pending)
        pending.clear()

    adapter = Mock()
    adapter.finish.side_effect = close_reporter
    notifications = Mock()
    service = SimpleNamespace(
        _tracking_finish_task=None,
        _adapters={"clearml": adapter, "apprise": notifications},
        _use_wandb=False,
    )
    ns = _finish_method()
    assert not persisted
    result = asyncio.run(ns["finish_tracking"](service))
    assert result == {"status": "success", "results": {"clearml": "success"}}
    assert persisted == [(9, "perf/actor_train_tok_per_s_per_card", 700.0)]
    assert asyncio.run(ns["finish_tracking"](service)) == result
    adapter.finish.assert_called_once()
    notifications.finish.assert_not_called()


def test_finish_one_backend_failure_preserves_other_backends():
    tensorboard, clearml = Mock(), Mock()
    tensorboard.finish.side_effect = RuntimeError("disk unavailable")
    service = SimpleNamespace(
        _tracking_finish_task=None,
        _adapters={"tensorboard": tensorboard, "clearml": clearml},
        _use_wandb=True,
    )
    ns = _finish_method()
    result = asyncio.run(ns["finish_tracking"](service))
    assert result["status"] == "error"
    assert result["results"]["tensorboard"] == "error: disk unavailable"
    assert result["results"]["clearml"] == result["results"]["wandb"] == "success"
    clearml.finish.assert_called_once()
    ns["wandb"].finish.assert_called_once()
    assert asyncio.run(ns["finish_tracking"](service)) == result
    tensorboard.finish.assert_called_once()


def test_finish_keeps_event_loop_responsive_and_survives_waiter_cancellation():
    started, release = threading.Event(), threading.Event()

    def slow_finish():
        started.set()
        assert release.wait(timeout=5), "test did not release reporter"

    adapter = Mock()
    adapter.finish.side_effect = slow_finish
    service = SimpleNamespace(_tracking_finish_task=None, _adapters={"clearml": adapter}, _use_wandb=False)
    ns = _finish_method()

    async def exercise():
        first = asyncio.create_task(ns["finish_tracking"](service))
        try:
            # Reaching this assertion before release proves the blocking close
            # did not prevent unrelated event-loop work, including health checks.
            assert await asyncio.to_thread(started.wait, 2)
            second = asyncio.create_task(ns["finish_tracking"](service))
            first.cancel()
            await asyncio.gather(first, return_exceptions=True)
            await asyncio.sleep(0)
            assert not second.done()
            release.set()
            result = await asyncio.wait_for(second, timeout=2)
            assert result == {"status": "success", "results": {"clearml": "success"}}
            assert await ns["finish_tracking"](service) == result
        finally:
            release.set()

    asyncio.run(exercise())
    adapter.finish.assert_called_once()
