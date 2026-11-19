# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import asyncio
from types import SimpleNamespace

import numpy as np
import pytest

from relax.engine.rollout.deferred import PublicationState
from relax.utils.tq.publication import publish_confirmed


class Client:
    def __init__(self):
        self.writes = 0
        self.visible = False
        self.completed = False
        self.write_error = None
        self.query_blocked = False
        self.observed = [10, 11]
        self.queried = asyncio.Event()

    async def async_put(self, **kwargs):
        self.writes += 1
        self.visible = True
        if self.write_error:
            raise self.write_error
        return SimpleNamespace(global_indexes=[10, 11])

    async def async_get_production_status(self, keys, partition):
        self.queried.set()
        if self.query_blocked:
            await asyncio.Event().wait()
        return np.array(self.observed), np.full(len(self.observed), self.visible)

    async def async_check_production_completed(self, partition):
        return self.completed


async def publish(client, **kwargs):
    return await publish_confirmed(
        client, {"tokens": object()}, partition_id="train_0", operation_id="attempt-1:batch-0", **kwargs
    )


@pytest.mark.asyncio
async def test_publication_lost_write_reply_never_replays():
    client = Client()
    client.write_error = ConnectionError("reply lost after write")
    with pytest.raises(ConnectionError, match="reply lost"):
        await publish(client)
    assert client.visible
    assert client.writes == 1
    assert not client.queried.is_set()


@pytest.mark.asyncio
async def test_publication_confirms_visibility_and_terminal_marker():
    client = Client()
    task = asyncio.create_task(publish(client, is_last=True))
    await client.queried.wait()
    assert not task.done()
    client.completed = True
    receipt = await asyncio.wait_for(task, 1)
    assert receipt.state is PublicationState.CONFIRMED
    assert receipt.operation_id == "attempt-1:batch-0"
    assert client.writes == 1


@pytest.mark.asyncio
async def test_publication_query_timeout_does_not_replay_or_confirm():
    client = Client()
    client.query_blocked = True
    with pytest.raises(asyncio.TimeoutError):
        await publish(client, timeout_s=0.02)
    assert client.writes == 1


@pytest.mark.asyncio
async def test_publication_missing_terminal_notification_never_confirms():
    client = Client()
    with pytest.raises(asyncio.TimeoutError):
        await publish(client, is_last=True, timeout_s=0.02)
    assert client.visible
    assert client.writes == 1


@pytest.mark.asyncio
async def test_publication_duplicate_observed_row_cannot_hide_missing_identity():
    client = Client()
    client.observed = [10, 10]
    with pytest.raises(asyncio.TimeoutError):
        await publish(client, timeout_s=0.02)
    assert client.writes == 1
