# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import sys
from types import SimpleNamespace

import pytest

from relax.distributed.ray import rollout_worker as module
from relax.distributed.ray.inference_handle import shared_task_enabled


@pytest.mark.asyncio
@pytest.mark.parametrize("bind_fails", [False, True])
async def test_shared_agentic_worker_owns_pipeline_lifecycle(monkeypatch, bind_fails):
    events = []
    args = SimpleNamespace(
        use_agentic_rollout=True,
        colocate=True,
        use_opd=True,
        hybrid=False,
        rollout_function_path="relax.agentic.rollout.generate_rollout",
        eval_function_path="relax.agentic.rollout.generate_rollout",
        sglang_router_ip="router",
        sglang_router_port=30000,
        tq_config={},
    )
    assert shared_task_enabled(args)

    def bind(host, port):
        assert (host, port) == ("router", 30000)
        events.append("bind")
        if bind_fails:
            raise RuntimeError("bind failed")

    monkeypatch.setitem(
        sys.modules,
        "relax.agentic.rollout",
        SimpleNamespace(
            init_agentic_resident_pipeline=lambda *args: events.append("initialize"),
            bind_agentic_router=bind,
            shutdown_agentic_resident_pipeline=lambda: events.append("shutdown"),
        ),
    )
    monkeypatch.setattr(module, "init_tracking", lambda *a, **k: None)
    monkeypatch.setattr(module, "init_http_client", lambda *a: None)
    monkeypatch.setattr(module, "attach_tq_client", lambda *a, **k: object())
    monkeypatch.setattr(module, "detach_tq_client", lambda: events.append("detach"))

    async def close_workload():
        pass

    monkeypatch.setattr(module, "RolloutWorkload", lambda *a: SimpleNamespace(close=close_workload))
    implementation = module.RolloutWorker.__ray_metadata__.modified_class
    if bind_fails:
        with pytest.raises(RuntimeError, match="bind failed"):
            implementation(args, object(), object())
    else:
        worker = implementation(args, object(), object())
        await worker.close()
    assert events == ["initialize", "bind", "shutdown", "detach"]


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel", [False, True])
async def test_shared_worker_failure_prevents_partition_replay(monkeypatch, cancel):
    import asyncio
    from unittest.mock import AsyncMock

    implementation = module.RolloutWorker.__ray_metadata__.modified_class
    worker = object.__new__(implementation)
    worker.args = SimpleNamespace(
        colocate=True, _inference_phase_coordinator=SimpleNamespace(switch=SimpleNamespace(remote=AsyncMock()))
    )
    worker._closed = False
    worker._generation_running = False
    worker._generation_failed = False
    worker._training_ready = False
    worker._phase_lock = asyncio.Lock()
    worker._active = set()
    entered = asyncio.Event()
    release = asyncio.Event()

    async def generate(rollout_id):
        entered.set()
        await release.wait()
        if not cancel:
            raise RuntimeError("publication unknown")

    workload = SimpleNamespace(generate=AsyncMock(side_effect=generate), eval=AsyncMock(), save=AsyncMock())
    worker.workload = workload
    worker.manager = SimpleNamespace(
        begin_rollout=SimpleNamespace(remote=AsyncMock()), wait_ready=SimpleNamespace(remote=AsyncMock())
    )
    await worker.eval(0)
    workload.eval.assert_awaited_once_with(0)
    first = asyncio.create_task(worker.generate(0))
    await entered.wait()
    with pytest.raises(RuntimeError, match="already running"):
        await worker.generate(0)
    evaluation = asyncio.create_task(worker.eval(0))
    await asyncio.sleep(0)
    assert not evaluation.done()
    if cancel:
        first.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first
    else:
        release.set()
        with pytest.raises(RuntimeError, match="publication unknown"):
            await first
    with pytest.raises(RuntimeError, match="new RolloutWorker"):
        await worker.generate(0)
    with pytest.raises(RuntimeError, match="new RolloutWorker"):
        await worker.generate(1)
    assert workload.generate.call_count == 1
    release.set()
    await asyncio.gather(*tuple(worker._active), return_exceptions=True)
    with pytest.raises(RuntimeError, match="new RolloutWorker"):
        await asyncio.wait_for(evaluation, 1)
    await worker.save(0)
    workload.eval.assert_awaited_once_with(0)
    workload.save.assert_awaited_once_with(0)


@pytest.mark.asyncio
async def test_only_fully_async_first_startup_has_a_total_ready_timeout(monkeypatch):
    import asyncio

    implementation = module.RolloutWorker.__ray_metadata__.modified_class
    worker = object.__new__(implementation)
    worker.args = SimpleNamespace(fully_async=True)
    ready = asyncio.Event()
    ready.set()
    worker.manager = SimpleNamespace(wait_ready=SimpleNamespace(remote=lambda role: ready.wait()))
    original_wait = asyncio.wait_for

    async def short_startup_budget(awaitable, timeout):
        return await original_wait(awaitable, 0.02 if timeout == 120 else timeout)

    monkeypatch.setattr(module.asyncio, "wait_for", short_startup_budget)
    await worker._activate_generation()
    ready.clear()
    normal = asyncio.create_task(worker._activate_generation())
    await asyncio.sleep(0.04)
    assert not normal.done()
    ready.set()
    await original_wait(normal, 1)
    assert worker._startup_ready


@pytest.mark.asyncio
async def test_fully_async_startup_timeout_includes_phase_rpc(monkeypatch):
    import asyncio

    implementation = module.RolloutWorker.__ray_metadata__.modified_class
    worker = object.__new__(implementation)
    never = asyncio.Event()
    worker.args = SimpleNamespace(
        fully_async=True,
        _inference_phase_coordinator=SimpleNamespace(
            switch=SimpleNamespace(remote=lambda phase: never.wait()),
        ),
    )
    worker.manager = SimpleNamespace(wait_ready=SimpleNamespace(remote=lambda role: never.wait()))
    original_wait = asyncio.wait_for

    async def short_startup_budget(awaitable, timeout):
        return await original_wait(awaitable, 0.02 if timeout == 120 else timeout)

    monkeypatch.setattr(module.asyncio, "wait_for", short_startup_budget)
    with pytest.raises(asyncio.TimeoutError):
        await worker._activate_generation()
    assert not getattr(worker, "_startup_ready", False)


@pytest.mark.asyncio
@pytest.mark.parametrize("shared", [True, False])
async def test_eval_can_run_on_independent_gpus_and_waits_for_shared_training(shared):
    import asyncio
    from unittest.mock import AsyncMock

    worker = object.__new__(module.RolloutWorker.__ray_metadata__.modified_class)
    switch = AsyncMock()
    ready = asyncio.Event()
    generating = asyncio.Event()
    release = asyncio.Event()

    async def generate(step):
        generating.set()
        await release.wait()

    worker.args = SimpleNamespace(
        colocate=shared,
        _inference_phase_coordinator=SimpleNamespace(switch=SimpleNamespace(remote=switch)),
    )
    worker._closed = worker._generation_running = worker._generation_failed = worker._training_ready = False
    worker._phase_lock = asyncio.Lock()
    worker._active = set()
    worker.workload = SimpleNamespace(generate=generate, eval=AsyncMock())
    worker.manager = SimpleNamespace(
        wait_ready=SimpleNamespace(remote=lambda role: ready.wait()),
        begin_rollout=SimpleNamespace(remote=AsyncMock()),
    )
    ready.set()
    train = asyncio.create_task(worker.generate(7))
    await asyncio.wait_for(generating.wait(), 1)
    evaluation = asyncio.create_task(worker.eval(6))
    try:
        if shared:
            await asyncio.sleep(0)
            assert not evaluation.done()
        else:
            await asyncio.wait_for(evaluation, 1)
            worker.workload.eval.assert_awaited_once_with(6)
    finally:
        release.set()
        await asyncio.wait_for(asyncio.gather(train, evaluation), 1)

    if shared:
        await worker.offload()
        ready.clear()
        switch.reset_mock()
        evaluation = asyncio.create_task(worker.eval(7))
        await asyncio.sleep(0)
        assert not evaluation.done()
        switch.assert_not_awaited()
        ready.set()
        await asyncio.wait_for(evaluation, 1)
        switch.assert_not_awaited()
    assert worker.manager.begin_rollout.remote.await_args_list[0].args == (7,)
    assert all(call.args == (None,) for call in worker.manager.begin_rollout.remote.await_args_list[1:])
