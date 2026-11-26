# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import time
from argparse import Namespace
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

from relax.components import advantages as advantages_module
from relax.components.rollout import Rollout


def test_advantages_empty_polling_is_bounded_and_ready_data_runs_once(monkeypatch):
    started = time.monotonic()
    polls = []
    completed = [False]
    fetched = []

    async def drained(*args, **kwargs):
        return completed[0]

    async def metadata(**kwargs):
        polls.append(time.monotonic())
        ready = polls[-1] - started >= 0.12
        return SimpleNamespace(size=int(ready), samples=[object()] if ready else [])

    async def get_data(meta):
        fetched.append(meta)
        return {}

    async def put(**kwargs):
        completed[0] = True

    def run_immediate(coroutine):
        try:
            coroutine.send(None)
        except StopIteration as result:
            return result.value
        finally:
            coroutine.close()
        raise AssertionError("external fixture unexpectedly suspended")

    monkeypatch.setattr(advantages_module, "run_", run_immediate)
    worker = SimpleNamespace(
        config=Namespace(
            fully_async=True, use_dynamic_batch_size=True, global_batch_size=1, num_iters_per_train_update=1
        ),
        data_system_client=SimpleNamespace(
            async_check_stream_drained=drained, async_get_meta=metadata, async_get_data=get_data, async_put=put
        ),
        compute_advantages_and_returns=Mock(return_value={}),
        _logger=Mock(),
    )
    advantages_module.Advantages.func_or_class._compute_partition(worker, 0, ["rewards"])
    assert len(polls) <= 8, f"empty stream was polled {len(polls)} times in 120 ms"
    assert len(fetched) == 1
    worker.compute_advantages_and_returns.assert_called_once_with({})


async def test_rollout_recognizes_retired_terminal_stream_without_accepting_unfinished_stream():
    complete = AsyncMock(return_value=False)
    drained = AsyncMock(side_effect=[True, False])
    service = SimpleNamespace(
        config=Namespace(fully_async=True, use_dynamic_batch_size=True),
        data_system_client=SimpleNamespace(
            async_check_production_completed=complete, async_check_stream_drained=drained
        ),
    )
    check = Rollout.func_or_class._async_check_partition_production_complete
    assert await check(service, "train_0"), "retired stream with producer EOS was treated as unfinished"
    assert not await check(service, "train_1")
    assert drained.call_args_list[0].kwargs == {"task_name": "actor_train", "partition_id": "train_0"}
