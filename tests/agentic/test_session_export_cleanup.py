# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Finalization owns publication and resource release even if its watcher
exits."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from relax.agentic.pipeline import RuntimeGroupError, SessionExportTransport, TrainingFieldArtifact
from relax.agentic.session.service import (
    AgenticSessionShard,
    ResidentGroup,
    SessionResources,
    _SessionRecord,
    _SessionResultCell,
)
from relax.agentic.session.state import InflightRequest, RequestKind, SessionForest, check_messages


class _Tokenizer:
    def decode(self, token_ids, **kwargs):
        return "".join(chr(token) for token in token_ids)


@pytest.fixture
def shard():
    shard_class = AgenticSessionShard.__ray_metadata__.modified_class
    instance = object.__new__(shard_class)
    instance.args = SimpleNamespace(agentic_session_lifecycle=True, use_rollout_routing_replay=False)
    instance._generation_backend = SimpleNamespace(
        tokenizer=_Tokenizer(),
        abort_request=AsyncMock(),
        close_session=AsyncMock(return_value=True),
        shutdown=AsyncMock(),
    )
    instance._groups = {}
    instance._session_records = {}
    instance._permit_cleanup_tasks = set()
    instance._lifecycle_close_count = 0
    instance._lifecycle_close_failure_count = 0
    instance._state_revision = 0
    instance._state_changed = asyncio.Event()
    instance._train_generation_open = True
    return instance


def _attach_session(shard, session_id, *, with_response=True):
    forest = SessionForest.create_empty(session_id=session_id, train_metadata={"loss": "grpo"})
    observation = forest.append_obs(
        parent_state_hash=forest.root_state_hash,
        rollout_id=0,
        abort_count=0,
        messages_delta=check_messages([{"role": "user", "content": "A"}]),
        train_token_delta=[65],
        rollout_token_delta=[65],
    )
    if with_response:
        forest.append_resp(
            parent_state_hash=observation.state_hash,
            rollout_id=0,
            abort_count=0,
            messages_delta=check_messages([{"role": "assistant", "content": "BC"}]),
            token_delta=[66, 67],
            logprob_delta=[-0.125, -0.25],
            status="completed",
        )
    cell = _SessionResultCell()
    group = ResidentGroup(rollout_mode="train", group_id=session_id, result_cells={session_id: cell}, sessions=[])
    session = _SessionRecord(
        group=group, session_id=session_id, session_sampling_params={}, result_cell=cell, forest=forest
    )
    group.sessions.append(session)
    shard._groups[group.group_id] = group
    shard._session_records[session_id] = session
    exited = asyncio.Event()

    async def process_result():
        await exited.wait()
        return None

    process_wait = asyncio.create_task(process_result())
    process = SimpleNamespace(
        output=SimpleNamespace(reward=1.0, metadata={"test": True}, records=()), terminate_and_join=AsyncMock()
    )
    watcher = asyncio.create_task(shard._watch_agent_process(session, process_wait))
    session.resources = SessionResources(process=process, process_wait=process_wait, watcher_task=watcher)
    session.resp_state_hash_by_request_id["old-request"] = observation.state_hash
    publish_once = cell.publish

    def publish(outcome):
        assert session.forest is None and session.resources is None
        assert not session.live_irs and not session.queued_irs
        assert session.session_id not in shard._session_records
        assert process_wait.done()
        process.terminate_and_join.assert_awaited_once()
        publish_once(outcome)

    cell.publish = Mock(side_effect=publish)
    return session, process, process_wait, watcher, exited


def _assert_released(shard, session, process, process_wait):
    assert session.forest is None
    assert session.resources is None
    assert not session.resp_state_hash_by_request_id
    assert not session.live_irs
    assert not session.queued_irs
    assert session.session_id not in shard._session_records
    assert session not in session.group.sessions
    assert session.finish_task.done()
    assert process_wait.done()
    process.terminate_and_join.assert_awaited_once()
    session.result_cell.publish.assert_called_once()


@pytest.mark.parametrize("outcome", ["success", "controlled_drop", "export_exception"])
async def test_finalization_publishes_once_after_resource_release(shard, outcome, monkeypatch):
    if outcome == "export_exception":
        monkeypatch.setattr(shard, "_build_session_export_transport", Mock(side_effect=ValueError("broken export")))
    for attempt in range(3):
        session, process, process_wait, watcher, exited = _attach_session(
            shard, str(attempt), with_response=outcome != "controlled_drop"
        )
        exited.set()
        await asyncio.wait_for(watcher, timeout=2)
        _assert_released(shard, session, process, process_wait)
        await shard._finish_session(session, None)
        _assert_released(shard, session, process, process_wait)
        if outcome == "success":
            (payload,) = await shard.take_session_results(session.group.group_id, (session.session_id,))
            assert isinstance(payload, SessionExportTransport)
            sample = TrainingFieldArtifact(sample_payload=payload.exports[0]["sample_payload"]).to_sample()
            assert sample.tokens == [65, 66, 67]
            assert sample.loss_mask == [1, 1]
            assert sample.rollout_log_probs == [-0.125, -0.25]
            assert sample.reward == 1.0
            await shard.release_group(session.group.group_id)
        else:
            if outcome == "controlled_drop":
                assert session.result_cell.dropped
            else:
                assert isinstance(session.result_cell.payload, RuntimeGroupError)
                assert "broken export" in str(session.result_cell.payload)
            await shard.drop_group(session.group.group_id)
        assert not shard._groups
    assert shard._lifecycle_close_count == 3
    assert shard._generation_backend.close_session.await_count == 3


async def test_watcher_cancellation_waits_for_finalization_cleanup(shard):
    close_started, allow_close = asyncio.Event(), asyncio.Event()

    async def close(*args, **kwargs):
        close_started.set()
        await allow_close.wait()
        return True

    shard._generation_backend.close_session.side_effect = close
    session, process, process_wait, watcher, exited = _attach_session(shard, "cancelled-watcher")
    try:
        exited.set()
        await asyncio.wait_for(close_started.wait(), timeout=2)
        watcher.cancel()
        allow_close.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(watcher, timeout=2)
        _assert_released(shard, session, process, process_wait)
        assert isinstance(session.result_cell.payload, SessionExportTransport)
        await shard.drop_group(session.group.group_id)
        assert not shard._groups
        shard._generation_backend.close_session.assert_awaited_once()
    finally:
        allow_close.set()
        await asyncio.gather(watcher, return_exceptions=True)


async def test_shutdown_discards_active_requests_and_releases_owned_tasks(shard):
    session, process, process_wait, watcher, _ = _attach_session(shard, "shutdown")
    runner = asyncio.create_task(asyncio.Event().wait())
    waiter = asyncio.get_running_loop().create_future()
    request = InflightRequest(
        request_id="queued-request",
        parent_state_hash=session.forest.root_state_hash,
        rollout_id=0,
        kind=RequestKind.FRESH,
        abort_count=0,
        waiter=waiter,
        wall_started_at=0,
        runner_task=runner,
    )
    session.live_irs.add(request)
    session.queued_irs.append(request)
    await asyncio.wait_for(shard.shutdown(), timeout=2)
    _assert_released(shard, session, process, process_wait)
    assert watcher.done() and runner.done()
    assert waiter.done()
    assert waiter.result()["_http_status"] == 404
    assert waiter.result()["error"]["code"] == "session_discarded"
    assert request.runner_task is None and not request.backend_started
    assert session.result_cell.dropped
    assert not shard._groups and not shard._permit_cleanup_tasks
    assert shard._train_generation_open is False
    shard._generation_backend.close_session.assert_awaited_once()
    shard._generation_backend.shutdown.assert_awaited_once()
    await asyncio.wait_for(shard.shutdown(), timeout=2)
    assert shard._lifecycle_close_count == 1
    process.terminate_and_join.assert_awaited_once()
