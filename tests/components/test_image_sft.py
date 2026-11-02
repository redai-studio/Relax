# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""CPU producer regressions with queue RPCs and the Serve decorator mocked."""

import asyncio
import importlib.util
import sys
from argparse import Namespace
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
import torch

from relax.engine.sft.image import ImageBatchSampler
from relax.engine.sft.runtime import sft_partition_id
from relax.utils.tq import lifecycle as tq_lifecycle


@pytest.fixture
def producer(monkeypatch):
    client = SimpleNamespace(async_get_partition_list=AsyncMock(return_value=[]), async_put=AsyncMock())
    attach = MagicMock(return_value=client)
    monkeypatch.setattr(tq_lifecycle, "attach_tq_client", attach)
    monkeypatch.setattr(tq_lifecycle, "detach_tq_client", MagicMock())
    ray = ModuleType("ray")
    ray.serve = SimpleNamespace(deployment=lambda cls: cls)
    path = Path(__file__).resolve().parents[2] / "relax/components/image_sft.py"
    spec = importlib.util.spec_from_file_location("_image_sft_cpu_test", path)
    module = importlib.util.module_from_spec(spec)
    with monkeypatch.context() as imports:
        imports.setitem(sys.modules, "ray", ray)
        spec.loader.exec_module(module)

    config = Namespace(
        loss_type="sft",
        generation_task="i2i",
        _i2i_sft_attempt_id="current",
        i2i_sft_config={"dataset_size": 19},
        global_batch_size=8,
        seed=42,
        start_rollout_id=0,
        num_rollout=11,
        max_staleness=1,
        sft_tq_timeout_minutes=1e-10,
        tq_config={},
    )
    instance = module.ImageSFT(MagicMock(), None, 0, config, "sft")
    instance._logger_instance = MagicMock()
    attach.assert_called_once_with(config.tq_config, role="sft")
    yield instance
    instance.__del__()


def test_image_sft_teardown_detaches_worker_client_once(producer):
    producer.__del__()
    producer.__del__()
    tq_lifecycle.detach_tq_client.assert_called_once()
    assert producer.data_system_client is None


@pytest.mark.parametrize("max_staleness", [0, 1])
def test_image_sft_resume_ignores_old_partitions_and_uses_restored_cursor(producer, max_staleness):
    stale = ["sft_9", "sft_12", "sft_13", "sft_previous_9", "sft_previous_12", "sft_currentother_12", "train_10"]
    producer.client.async_get_partition_list.return_value = stale
    producer.config.max_staleness = max_staleness
    producer.set_step(10)  # Controller synchronizes the backend checkpoint cursor before run().
    rng = torch.get_rng_state().clone()
    asyncio.run(producer.run())

    call = producer.client.async_put.await_args.kwargs
    assert producer.client.async_put.await_count == 1
    assert call["partition_id"] == sft_partition_id(producer.config, 10)
    assert call["data"]["sample_id"].tolist() == ImageBatchSampler(19, 8, 42).batch(10)
    assert call["custom_meta"] == [{"sample_index": 80 + index} for index in range(8)]
    assert producer.step == 11
    assert torch.equal(rng, torch.get_rng_state())
    assert producer.client.async_get_partition_list.return_value == stale


def test_image_sft_capacity_still_limits_current_attempt(producer):
    producer.set_step(10)
    producer.client.async_get_partition_list.return_value = ["sft_current_8", "sft_current_9"]
    with pytest.raises(TimeoutError, match="consumer did not drain"):
        asyncio.run(producer.run())
    producer.client.async_put.assert_not_awaited()
    assert producer.step == 10
    assert not producer._healthy
    assert producer.healthy.report_error.remote.call_args.kwargs == {"fatal": True}


def test_image_sft_waits_until_current_attempt_is_drained(producer, monkeypatch):
    producer.set_step(10)
    producer.config.sft_tq_timeout_minutes = 1
    producer.client.async_get_partition_list.side_effect = [["sft_current_8", "sft_current_9"], []]
    monkeypatch.setattr(asyncio, "sleep", AsyncMock())
    asyncio.run(producer.run())
    asyncio.sleep.assert_awaited_once_with(0.5)
    producer.client.async_put.assert_awaited_once()


def test_image_sft_stop_cancels_a_capacity_wait_without_writing(producer):
    async def check():
        polled = asyncio.Event()

        async def partitions():
            polled.set()
            return ["sft_current_8", "sft_current_9"]

        producer.config.sft_tq_timeout_minutes = 1
        producer.client.async_get_partition_list.side_effect = partitions
        task = asyncio.create_task(producer.run())
        await asyncio.wait_for(polled.wait(), timeout=1)
        await producer.stop()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(check())
    producer.client.async_put.assert_not_awaited()
    producer.healthy.report_error.remote.assert_not_called()
    assert producer._healthy
