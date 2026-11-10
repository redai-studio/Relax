# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Tests for ``Rollout.can_do_update_weight_for_async`` handshake-gate safety.

These exercise the real method on a shell instance (``object.__new__``) with the
collaborators it touches (``rollout_manager`` remote calls,
``_async_check_production_for_update_weight``) stubbed, so we test the
try/finally control flow in isolation -- without Ray Serve / GPUs.

Regression target (deadlock): previously, if a remote call inside the
``can_update`` branch raised (e.g. ``RayActorError`` from a dead engine), the
method returned before ``self._weight_update_ready.set()`` ran. The gate stayed
cleared forever, so the next ``end_update_weight`` blocked on
``_weight_update_ready.wait()`` -> the actor hung. The fix wraps the branch in
try/finally so the gate is always released.
"""

import asyncio
import logging

import pytest
from fastapi import HTTPException
from ray.exceptions import RayActorError

from relax.components.rollout import Rollout as RolloutDeployment


# ``Rollout`` is wrapped by ``@serve.deployment`` / ``@serve.ingress`` -- reach
# the underlying class so we can build a plain shell instance.
Rollout = RolloutDeployment.func_or_class


class _RemoteStub:
    """Mimics a Ray actor-handle method: ``.remote(...)`` returns an awaitable
    (or raises synchronously to simulate a dead handle)."""

    def __init__(self, fn):
        self._fn = fn

    def remote(self, *args, **kwargs):
        return self._fn(*args, **kwargs)


async def _ok(*_args, **_kwargs):
    return None


async def _prepared(*_args, **_kwargs):
    return True


def _raise_dead(*_args, **_kwargs):
    raise RayActorError()


class _ManagerStub:
    def __init__(self, set_weight_updating_fn=None, health_monitoring_pause_fn=None):
        self.health_monitoring_pause = _RemoteStub(health_monitoring_pause_fn or (lambda *a, **k: _ok()))
        self.set_weight_updating = _RemoteStub(set_weight_updating_fn or (lambda *a, **k: _prepared()))


def _make_rollout(*, can_update: bool, manager: _ManagerStub) -> "Rollout":
    shell = object.__new__(Rollout)
    # ``_logger`` is a lazy read-only property on Base; prime its backing field.
    shell._logger_instance = logging.getLogger("test.rollout")
    shell.step = 0
    shell.status = "running"
    shell._weight_update_ready = asyncio.Event()
    shell._weight_update_ready.set()
    shell.rollout_manager = manager

    async def _check(_step):
        return can_update

    shell._async_check_production_for_update_weight = _check  # type: ignore[method-assign]
    return shell


def test_can_do_update_weight_normal_path_sets_event_and_returns_1():
    shell = _make_rollout(can_update=True, manager=_ManagerStub())
    result = asyncio.run(shell.can_do_update_weight_for_async())
    assert result == 1
    assert shell.status == "paused"
    assert shell._weight_update_ready.is_set()


def test_can_do_update_weight_cannot_update_returns_0():
    shell = _make_rollout(can_update=False, manager=_ManagerStub())
    result = asyncio.run(shell.can_do_update_weight_for_async())
    assert result == 0
    # No pause requested, gate untouched (still set from construction).
    assert shell.status == "running"
    assert shell._weight_update_ready.is_set()


def test_can_do_update_weight_dead_engine_releases_gate_then_reraises():
    manager = _ManagerStub(set_weight_updating_fn=_raise_dead)
    shell = _make_rollout(can_update=True, manager=manager)

    with pytest.raises(RayActorError):
        asyncio.run(shell.can_do_update_weight_for_async())

    # The failure path must still release the handshake gate.
    assert shell._weight_update_ready.is_set()


def test_can_do_update_weight_rejected_lease_rolls_back_before_pause():
    calls = []

    async def _set_weight_updating(value):
        calls.append(value)
        return not value

    async def _pause():
        calls.append("pause")

    manager = _ManagerStub(
        set_weight_updating_fn=_set_weight_updating,
        health_monitoring_pause_fn=_pause,
    )
    shell = _make_rollout(can_update=True, manager=manager)

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(shell.can_do_update_weight_for_async())

    assert exc_info.value.status_code == 503
    assert calls == [True, False]
    assert shell.status == "running"
    assert shell._weight_update_ready.is_set()


def test_can_do_update_weight_closes_local_admission_while_lease_is_pending():
    async def _scenario():
        lease_started = asyncio.Event()
        finish_lease = asyncio.Event()

        async def _set_weight_updating(value):
            if value:
                lease_started.set()
                await finish_lease.wait()
                return False
            return True

        shell = _make_rollout(
            can_update=True,
            manager=_ManagerStub(set_weight_updating_fn=_set_weight_updating),
        )
        handshake = asyncio.create_task(shell.can_do_update_weight_for_async())
        await asyncio.wait_for(lease_started.wait(), timeout=1)

        assert shell.status == "paused"

        finish_lease.set()
        with pytest.raises(HTTPException):
            await handshake
        assert shell.status == "running"

    asyncio.run(_scenario())


def test_end_update_weight_does_not_block_after_failed_can_do():
    async def _scenario():
        manager = _ManagerStub(set_weight_updating_fn=_raise_dead)
        shell = _make_rollout(can_update=True, manager=manager)

        with pytest.raises(RayActorError):
            await shell.can_do_update_weight_for_async()

        assert shell._weight_update_ready.is_set()

        # Swap in a healthy manager for the resume; the gate being set means
        # end_update_weight's ``await self._weight_update_ready.wait()`` returns
        # immediately instead of hanging forever.
        shell.rollout_manager = _ManagerStub()
        await asyncio.wait_for(shell.end_update_weight(), timeout=1)
        assert shell.status == "running"

    asyncio.run(_scenario())
