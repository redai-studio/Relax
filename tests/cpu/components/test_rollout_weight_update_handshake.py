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
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi import HTTPException
from ray.exceptions import RayActorError, RayTaskError

from relax.components.rollout import Rollout as RolloutDeployment
from relax.engine.rollout.base_types import DatasetStateMismatchError


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


@pytest.mark.parametrize("algorithm", ["art", "grpo"])
@pytest.mark.parametrize("error", [OSError, DatasetStateMismatchError])
def test_resume_data_load_failure_reaches_service_before_production(algorithm, error):
    from relax.distributed.ray.rollout import RolloutManager

    manager = object.__new__(RolloutManager.__ray_metadata__.modified_class)
    manager.args = SimpleNamespace(sampling_config={"algorithm": algorithm})
    failure = error("dataset unavailable")
    if error is DatasetStateMismatchError:
        failure = RayTaskError("load", "dataset unavailable", failure).as_instanceof_cause()
    read = AsyncMock(side_effect=failure)
    manager.data_source = SimpleNamespace(load=SimpleNamespace(remote=read))
    service = _make_rollout(can_update=False, manager=SimpleNamespace(load=_RemoteStub(manager.load)))
    service.config = SimpleNamespace(rollout_global_dataset=True, sampling_config={"algorithm": algorithm})
    service.step, service._run_task = 1, None
    service._async_run = AsyncMock()
    if algorithm == "art" or error is DatasetStateMismatchError:
        with pytest.raises(error, match="dataset unavailable"):
            asyncio.run(service.run())
        assert service._run_task is None
        service._async_run.assert_not_awaited()
    else:
        asyncio.run(service.run())
        service._async_run.assert_awaited_once()
    read.assert_awaited_once_with(0)


@pytest.mark.parametrize("algorithm", ["art", "grpo"])
def test_rollout_does_not_rewrite_actor_owned_art_checkpoint_cursor(algorithm):
    service = _make_rollout(can_update=False, manager=_ManagerStub())
    service.config = SimpleNamespace(
        save="checkpoints",
        save_interval=1,
        num_rollout=3,
        rotate_ckpt=False,
        train_backend="fsdp",
        sampling_config={"algorithm": algorithm},
    )
    save = AsyncMock()
    service.data_source = SimpleNamespace(save=SimpleNamespace(remote=save))
    asyncio.run(service._maybe_save_data(0))
    if algorithm == "art":
        save.assert_not_awaited()
    else:
        save.assert_awaited_once_with(rollout_id=0)


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
    shell._generating_step = None
    shell._deferred_train_step = None
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


@pytest.mark.parametrize("next_ready", [False, True])
@pytest.mark.parametrize("generation", [1, 2], ids=["original", "backfill"])
def test_update_during_the_batch_needed_by_actor(next_ready, generation):
    async def scenario():
        manager = _ManagerStub()
        service = _make_rollout(can_update=True, manager=manager)
        service.step = service._generating_step = generation
        service.config = SimpleNamespace(num_rollout=3)
        service._async_check_partition_production_complete = AsyncMock(return_value=next_ready)
        result = await service.can_do_update_weight_for_async(1)
        assert result == int(next_ready)
        assert service.status == ("paused" if next_ready else "running")
        if not next_ready:
            assert service._deferred_train_step == 1
            service._async_check_partition_production_complete.return_value = True
            assert await service.can_do_update_weight_for_async(1) == 1
        await service.end_update_weight()
        assert service.status == "running"
        assert service._deferred_train_step is None

    asyncio.run(scenario())


def test_update_during_speculative_batch_when_actor_data_ready():
    service = _make_rollout(can_update=False, manager=_ManagerStub())
    service.config = SimpleNamespace(num_rollout=4)
    service.step = service._generating_step = 2
    service._async_check_partition_production_complete = AsyncMock(return_value=True)
    assert asyncio.run(service.can_do_update_weight_for_async(1)) == 1


@pytest.mark.parametrize("needs_backfill", [False, True])
def test_deferred_update_yields_engines_at_boundary_but_permits_backfill(needs_backfill):
    async def scenario():
        service = _make_rollout(can_update=True, manager=_ManagerStub())
        service.step = 0
        service.eval_handler = None
        service._peer_barrier = None
        service._stop_event = asyncio.Event()
        service.config = SimpleNamespace(
            loss_type="policy_loss",
            offload_rollout=False,
            eval_interval=None,
            save=None,
            fully_async=True,
            max_staleness=2,
            num_rollout=2,
        )
        first_started = asyncio.Event()
        release_first = asyncio.Event()
        second_started = asyncio.Event()
        release_second = asyncio.Event()
        service.healthy = SimpleNamespace(update_heartbeat=SimpleNamespace(remote=lambda *_: None))
        reached_boundary = asyncio.Event()

        async def partitions():
            reached_boundary.set()
            return []

        service.data_system_client = SimpleNamespace(async_get_partition_list=partitions)
        service._async_check_partition_production_complete = AsyncMock(return_value=False)

        async def generate(rollout_id):
            if rollout_id == 0:
                first_started.set()
                await release_first.wait()
                return not needs_backfill
            second_started.set()
            await release_second.wait()
            return True

        service.rollout_manager.generate = _RemoteStub(generate)
        loop = asyncio.create_task(service._async_run())
        try:
            await first_started.wait()
            assert await service.can_do_update_weight_for_async(0) == 0
            release_first.set()
            await reached_boundary.wait()
            if needs_backfill:
                await second_started.wait()
            else:
                assert not second_started.is_set()
            service._async_check_partition_production_complete.return_value = True
            assert await service.can_do_update_weight_for_async(0) == 1
            await service.end_update_weight()
            await second_started.wait()
            release_second.set()
            await loop
        finally:
            loop.cancel()
            await asyncio.gather(loop, return_exceptions=True)

    asyncio.run(asyncio.wait_for(scenario(), timeout=6))


def test_ready_older_partition_cannot_interrupt_final_backfill():
    service = _make_rollout(can_update=False, manager=_ManagerStub())
    service.config = SimpleNamespace(num_rollout=3)
    service.step = 3
    service._generating_step = 2
    service._async_check_partition_production_complete = AsyncMock(return_value=True)
    assert asyncio.run(service.can_do_update_weight_for_async(1)) == 0


def test_readiness_reply_from_previous_step_cannot_pause_final_backfill():
    async def scenario():
        service = _make_rollout(can_update=True, manager=_ManagerStub())
        service.config = SimpleNamespace(num_rollout=3)
        service.step = 2
        read_started = asyncio.Event()
        finish_read = asyncio.Event()

        async def read_partition(_partition):
            read_started.set()
            await finish_read.wait()
            return True

        service._async_check_partition_production_complete = read_partition
        poll = asyncio.create_task(service.can_do_update_weight_for_async(1))
        await read_started.wait()
        service.step = service._generating_step = 3
        finish_read.set()
        assert await poll == 0
        assert service.status == "running"

    asyncio.run(asyncio.wait_for(scenario(), timeout=6))
