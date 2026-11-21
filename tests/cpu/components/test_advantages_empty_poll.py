"""Waiting for rollout data must not flood Ray, and arrivals must still
progress."""

import asyncio
import threading
from argparse import Namespace
from types import SimpleNamespace
from unittest.mock import Mock

import torch

from relax.components.advantages import Advantages


async def test_empty_queue_polling_is_bounded_and_resumes_on_arrival(monkeypatch):
    cls = Advantages.func_or_class
    obj = object.__new__(cls)
    obj.config = Namespace(
        num_rollout=1,
        fully_async=True,
        use_dynamic_batch_size=False,
        true_on_policy_mode=True,
        kl_coef=0,
        use_kl_loss=False,
        global_batch_size=1,
        num_iters_per_train_update=1,
        use_health_check=False,
    )
    obj.step = 0
    obj._logger_instance = Mock()
    obj.healthy = Mock()
    obj.data_system_client = Mock()
    arrived, drained = threading.Event(), threading.Event()
    polls = []

    def metadata(**kwargs):
        polls.append(kwargs)
        return SimpleNamespace(size=int(arrived.is_set()), samples=[0] if arrived.is_set() else [])

    obj.data_system_client.async_check_consumption_status.side_effect = lambda *_: drained.is_set()
    obj.data_system_client.async_get_meta.side_effect = metadata
    obj.data_system_client.async_get_data.return_value = {"rewards": torch.ones(1)}
    obj.data_system_client.async_put.side_effect = lambda **_: drained.set()
    obj.compute_advantages_and_returns = Mock(return_value={"advantages": torch.ones(1)})
    namespace = cls._run_blocking.__globals__
    monkeypatch.setitem(namespace, "run_", lambda value: value)
    monkeypatch.setitem(namespace, "consume_opd_advantage_data", lambda *_: None)
    task = asyncio.create_task(obj.run())
    try:
        await asyncio.sleep(0.22)
        assert 1 <= len(polls) <= 8
    finally:
        arrived.set()
        await asyncio.wait_for(task, 2)
    assert drained.is_set()
    obj.compute_advantages_and_returns.assert_called_once()
    obj.data_system_client.async_put.assert_called_once()
    obj.healthy.report_error.remote.assert_not_called()
