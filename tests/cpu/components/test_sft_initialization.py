# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""A slow data scan must not block service health or escape shutdown."""

import asyncio
import threading
from unittest.mock import AsyncMock, MagicMock

import pytest

from relax.components.sft import SFT
from tests.cpu.components.test_sft import _make_args, _make_processed


def _service():
    cls = SFT.func_or_class
    service = cls.__new__(cls)
    service._run_task = None
    service._init_task = None
    service._stop_event = asyncio.Event()
    service._batch_producers = []
    service._dataset = None
    service._eval_dataset = None
    service._logger_instance = MagicMock()
    service._async_run = AsyncMock()
    service.step = 0
    return service


@pytest.mark.asyncio
async def test_sft_slow_initialization_allows_health_and_shares_run():
    service = _service()
    entered = threading.Event()
    release = threading.Event()
    calls = []

    def initialize():
        calls.append(1)
        entered.set()
        assert release.wait(timeout=5)

    service._init_data_pipeline = initialize
    first = asyncio.create_task(service.run())
    second = asyncio.create_task(service.run())
    try:
        for _ in range(100):
            if entered.is_set():
                break
            await asyncio.sleep(0.01)
        assert entered.is_set()
        assert not first.done() and not second.done()
        assert calls == [1]
    finally:
        release.set()
    await asyncio.gather(first, second)
    await service._run_task
    service._async_run.assert_awaited_once()


@pytest.mark.asyncio
async def test_sft_stop_during_initialization_cleans_without_starting_production():
    service = _service()
    entered = threading.Event()
    release = threading.Event()
    dataset = MagicMock()

    def initialize():
        entered.set()
        assert release.wait(timeout=5)
        service._dataset = dataset

    service._init_data_pipeline = initialize
    running = asyncio.create_task(service.run())
    for _ in range(100):
        if entered.is_set():
            break
        await asyncio.sleep(0.01)
    assert entered.is_set()
    stopping = asyncio.create_task(service.stop())
    try:
        await asyncio.sleep(0.02)
        assert not stopping.done()
    finally:
        release.set()
    await asyncio.gather(running, stopping)
    dataset.stop.assert_called_once()
    service._async_run.assert_not_awaited()


@pytest.mark.asyncio
async def test_sft_cancelled_run_keeps_initialization_owned():
    service = _service()
    entered = threading.Event()
    release = threading.Event()
    calls = []

    def initialize():
        calls.append(1)
        entered.set()
        assert release.wait(timeout=5)

    service._init_data_pipeline = initialize
    running = asyncio.create_task(service.run())
    for _ in range(100):
        if entered.is_set():
            break
        await asyncio.sleep(0.01)
    assert entered.is_set()
    running.cancel()
    with pytest.raises(asyncio.CancelledError):
        await running
    assert not service._init_task.cancelled()
    release.set()
    await service.run()
    await service._run_task
    assert calls == [1]


@pytest.mark.asyncio
async def test_sft_initialization_failure_is_preserved_for_all_callers():
    service = _service()
    service._init_data_pipeline = MagicMock(side_effect=ValueError("broken dataset"))
    results = await asyncio.gather(service.run(), service.run(), return_exceptions=True)
    assert all(isinstance(result, ValueError) for result in results)
    with pytest.raises(ValueError, match="broken dataset"):
        await service.run()
    service._init_data_pipeline.assert_called_once()
    service._async_run.assert_not_awaited()
    await service.stop()


@pytest.mark.asyncio
async def test_fresh_prepare_prefills_one_batch_and_run_does_not_replay_it():
    service = _service()
    service._init_data_pipeline = MagicMock()
    produced = []

    async def produce():
        produced.append(service.step)
        await asyncio.sleep(0.01)
        service.step += 1

    service._produce_one_step = produce
    await asyncio.gather(
        service.prepare(prefetch_first_batch=True), service.prepare(prefetch_first_batch=True), service.run()
    )
    await service._run_task
    assert produced == [0]
    assert service.step == 1
    service._async_run.assert_awaited_once()


@pytest.mark.asyncio
async def test_stop_joins_prefilled_batch_before_releasing_processor():
    service = _service()
    service._init_data_pipeline = MagicMock()
    entered, release = asyncio.Event(), asyncio.Event()
    pool = MagicMock()
    service._processor_pool = pool

    async def produce():
        entered.set()
        await release.wait()
        service.step += 1

    service._produce_one_step = produce
    preparing = asyncio.create_task(service.prepare(prefetch_first_batch=True))
    await entered.wait()
    stopping = asyncio.create_task(service.stop())
    await asyncio.sleep(0.01)
    pool.shutdown.assert_not_called()
    release.set()
    await asyncio.gather(preparing, stopping)
    pool.shutdown.assert_called_once_with(wait=True)


@pytest.mark.asyncio
async def test_failed_initial_batch_releases_processor_on_stop():
    service = _service()
    service._init_data_pipeline = MagicMock()
    service._produce_one_step = AsyncMock(side_effect=ValueError("invalid first batch"))
    pool = MagicMock()
    service._processor_pool = pool

    with pytest.raises(ValueError, match="invalid first batch"):
        await service.prepare(prefetch_first_batch=True)
    await service.stop()

    pool.shutdown.assert_called_once_with(wait=True)
    service._async_run.assert_not_awaited()


@pytest.mark.asyncio
async def test_cancelled_stop_request_keeps_resource_cleanup_owned():
    service = _service()
    service._init_data_pipeline = MagicMock()
    entered, release = asyncio.Event(), asyncio.Event()
    pool = MagicMock()
    service._processor_pool = pool

    async def produce():
        entered.set()
        await release.wait()

    service._produce_one_step = produce
    preparing = asyncio.create_task(service.prepare(prefetch_first_batch=True))
    await asyncio.wait_for(entered.wait(), timeout=5)
    stopping = asyncio.create_task(service.stop())
    await asyncio.sleep(0)
    stopping.cancel()
    with pytest.raises(asyncio.CancelledError):
        await stopping
    pool.shutdown.assert_not_called()

    release.set()
    await asyncio.wait_for(asyncio.gather(preparing, service.stop()), timeout=5)
    pool.shutdown.assert_called_once_with(wait=True)


@pytest.mark.asyncio
async def test_prepare_then_run_publishes_consecutive_partitions_without_replaying_samples():
    class Dataset:
        def __init__(self):
            self.cursor = 0

        async def get_batch_async(self, count):
            assert count == 1
            sample = _make_processed(self.cursor)
            sample.tokens.fill_(self.cursor + 1)
            self.cursor += 1
            return [sample], False

        def stop(self):
            pass

    service = _service()
    service.config = _make_args(global_batch_size=1, num_rollout=2)
    service._dataset = Dataset()
    service._tokenizer = MagicMock()
    service._train_size = 2
    service._init_data_pipeline = MagicMock()
    service._async_run = type(service)._async_run.__get__(service)
    service.data_system_client = MagicMock()
    service.data_system_client.async_put = AsyncMock()
    service.data_system_client.async_get_partition_list = AsyncMock(return_value=[])

    await service.prepare(prefetch_first_batch=True)
    assert service.step == service._dataset.cursor == 1
    await service.run()
    await asyncio.wait_for(service._run_task, timeout=5)

    calls = service.data_system_client.async_put.call_args_list
    assert [call.kwargs["partition_id"] for call in calls] == ["sft_0", "sft_1"]
    assert [call.kwargs["data"]["tokens"][0][0].item() for call in calls] == [1, 2]
    assert service.step == service._dataset.cursor == 2
    await service.stop()
