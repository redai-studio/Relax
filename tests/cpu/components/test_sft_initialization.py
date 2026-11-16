# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""A slow data scan must not block service health or escape shutdown."""

import asyncio
import threading
from unittest.mock import AsyncMock, MagicMock

import pytest

from relax.components.sft import SFT


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
