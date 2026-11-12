# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Real watcher, IR, progress observer and capacity accounting on CPU.

Only backend HTTP, Ray transport and external process I/O are substitutes.
Events hold permit release beyond the shortened terminal-join budget; no
assertion depends on elapsed wall time. No Ray cluster or GPU is started.
"""

import asyncio
import json
from argparse import Namespace
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from relax.agentic.observability import SessionObservation, ShardTrajectoryRecorder
from relax.agentic.pipeline import RuntimeGroupError, SessionExportTransport, TrainingFieldArtifact
from relax.agentic.pipeline.runtime import RuntimeDomain, RuntimeGroupStream, _SessionResultRef, _SessionShardBinding
from relax.agentic.runner.ipc import AgentExecutionError
from relax.agentic.session import service as service_module
from relax.agentic.session.service import (
    AgenticSessionShard,
    GroupOwnership,
    ResidentGroup,
    SessionResources,
    _SessionRecord,
    _SessionResultCell,
)
from relax.agentic.session.state import InflightRequest, RequestKind
from relax.utils.types import Sample


pytestmark = pytest.mark.asyncio


async def eventually(predicate):
    async with asyncio.timeout(2):
        while not predicate():
            await asyncio.sleep(0)


@asynccontextmanager
async def finalizing_session(monkeypatch, *, eager=True, protected=False):
    for name in ("_FINALIZATION_ABORT_TIMEOUT_S", "_FINALIZATION_BACKEND_GRACE_S", "_FINALIZATION_CANCEL_GRACE_S"):
        monkeypatch.setattr(service_module, name, 0.01)
    args = Namespace(
        agentic_session_capacity_per_node=2 if eager else None,
        agentic_session_lifecycle=True,
        fully_async=not protected,
        partial_rollout=protected,
        use_slime_router=False,
        agentic_admission_scope="all",
    )
    shard = object.__new__(AgenticSessionShard.__ray_metadata__.modified_class)
    shard.args = args
    shard._trajectory_recorder = ShardTrajectoryRecorder(args, shard_ordinal=0)
    shard._groups = {}
    shard._session_records = {}
    shard._resource_release_receipts = {}
    shard._state_revision = 0
    shard._state_changed = asyncio.Event()
    shard._eager_gate_lock = asyncio.Lock()
    shard._eager_shutdown = False
    shard._train_generation_open = True
    shard._admission_client = None
    shard._permit_cleanup_tasks = set()
    shard._lifecycle_close_count = shard._lifecycle_close_failure_count = 0
    generated, agent_exit, permit_release = asyncio.Event(), asyncio.Event(), asyncio.Event()
    terminating, terminate_release = asyncio.Event(), asyncio.Event()
    terminate_release.set()
    waiters = []

    def spawn(coro):
        task = asyncio.create_task(coro)
        waiters.append(task)
        return task

    async def generate(**_kwargs):
        generated.set()
        await asyncio.Event().wait()

    async def acquire(_permit):
        return True

    async def release(_permit):
        await permit_release.wait()

    shard._generation_backend = SimpleNamespace(
        generate=generate, abort_request=AsyncMock(), close_session=AsyncMock(return_value=True)
    )
    shard._sglang_request_limiter = SimpleNamespace(
        acquire_request_permit=SimpleNamespace(remote=lambda permit: asyncio.create_task(acquire(permit))),
        release_request_permit=SimpleNamespace(remote=lambda permit: asyncio.create_task(release(permit))),
    )

    class Handle:
        def __getattr__(self, name):
            return SimpleNamespace(remote=lambda *a, **kw: asyncio.create_task(getattr(shard, name)(*a, **kw)))

    binding = _SessionShardBinding(actor_name="owner", handle=Handle(), node_id="host")
    runtime = RuntimeDomain(args, "train", [binding], concurrency=1)
    ids = ("done", "waiting")
    cells = {sid: _SessionResultCell() for sid in ids}
    # The first member already cleaned up. Exercise the final unfinished member
    # so no sibling's later result can accidentally wake a stuck collector.
    cells["done"].publish(SessionExportTransport(exports=()))
    group = ResidentGroup(
        "train",
        "group",
        cells,
        [],
        resource_released_session_ids={"done"},
        requires_resource_release=eager,
        ownership=GroupOwnership.RUNTIME,
        eager_session_generation=eager,
        leased_rollout_id=0,
        first_ir_seen_session_ids=set(ids),
    )
    session = _SessionRecord(group, "waiting", {}, cells["waiting"], protected_until_finalize=protected)
    group.sessions.append(session)
    shard._groups[group.group_id] = group
    shard._session_records[session.session_id] = session
    await runtime._resident_group_permits.acquire()
    reservation = await runtime._reserve_group_shard(group.group_id, None, len(ids))
    if runtime._session_capacity is not None:
        runtime._session_capacity.bind(reservation.session_capacity, ids)
        reservation.session_capacity.submitted = True
    loop = asyncio.get_running_loop()
    stream = RuntimeGroupStream(
        group.group_id,
        binding,
        loop.create_future(),
        tuple(_SessionResultRef(sid, loop.create_future()) for sid in ids),
        eager_session_generation=eager,
    )
    stream.first_request_barrier.set_result(True)
    binding.group_streams[group.group_id] = stream
    ir = InflightRequest(
        request_id="request",
        parent_state_hash="hash",
        rollout_id=0,
        kind=RequestKind.FRESH,
        abort_count=0,
        waiter=loop.create_future(),
        wall_started_at=0.0,
        sampling_params={"max_new_tokens": 8},
    )
    session.live_irs.add(ir)
    backend = asyncio.create_task(shard._run_ir(session, ir))
    ir.runner_task = backend
    process = SimpleNamespace(resource_released=False)

    async def terminate():
        terminating.set()
        await terminate_release.wait()
        process.resource_released = True

    process.terminate_and_join = AsyncMock(side_effect=terminate)

    async def wait_process():
        await agent_exit.wait()
        return AgentExecutionError("agent exited with an outstanding request")

    process_wait = asyncio.create_task(wait_process())
    watcher = asyncio.create_task(shard._watch_agent_process(session, process_wait))
    session.resources = SessionResources(process, process_wait, watcher)
    shard._notify_state_change(group)
    runtime._start_shard_observers()
    try:
        await asyncio.wait_for(generated.wait(), timeout=2)
        await eventually(lambda: stream.session_results[0].completion.done())
        if eager:
            await eventually(lambda: runtime.session_capacity_snapshot()["confirmed_releases"] == 1)
        else:
            assert runtime.session_capacity_snapshot() is None
        h = SimpleNamespace(**locals())
        yield h
    finally:
        agent_exit.set()
        permit_release.set()
        terminate_release.set()
        await asyncio.gather(backend, watcher, return_exceptions=True)
        for task in waiters:
            if not task.done():
                task.cancel()
        await asyncio.gather(*waiters, return_exceptions=True)
        for _ in range(2):
            await asyncio.gather(runtime.drop_group(stream), return_exceptions=True)
        await runtime._stop_shard_observers()
        await shard._trajectory_recorder.aclose()
        for ref in (stream.first_request_barrier, *(result.completion for result in stream.session_results)):
            if ref.done() and not ref.cancelled():
                ref.exception()


@pytest.mark.parametrize("eager", [False, True])
async def test_late_backend_exit_retries_cleanup_without_early_or_duplicate_release(monkeypatch, eager):
    async with finalizing_session(monkeypatch, eager=eager) as h:
        collector = h.spawn(h.runtime.collect_group(h.stream))
        h.agent_exit.set()
        with pytest.raises(RuntimeGroupError, match="backend_join did not settle"):
            await asyncio.wait_for(asyncio.shield(collector), timeout=2)
        original_error = h.group.finalization_error
        failed_finish = h.session.finish_task
        failed_drop = h.group.drop_task
        assert h.stream.error is original_error
        assert not h.binding.observer_task.done()
        assert not h.cells["waiting"].terminal
        assert h.group.resource_released_session_ids == {"done"}
        if eager:
            assert h.runtime.session_capacity_snapshot()["outstanding"] == 1
        else:
            assert h.runtime.session_capacity_snapshot() is None
        h.process.terminate_and_join.assert_not_awaited()
        h.shard._generation_backend.close_session.assert_not_awaited()

        # The real _run_ir finally block can now finish its permit-release RPC.
        h.permit_release.set()
        await asyncio.wait_for(h.backend, timeout=2)
        h.terminate_release.clear()
        drops = [h.spawn(h.runtime.drop_group(h.stream)) for _ in range(2)]
        await asyncio.wait_for(h.terminating.wait(), timeout=2)
        assert h.session.finish_task is not failed_finish
        assert h.group.drop_task is not failed_drop
        assert not h.cells["waiting"].terminal
        if eager:
            assert h.runtime.session_capacity_snapshot()["outstanding"] == 1
        else:
            assert h.runtime.session_capacity_snapshot() is None
        h.terminate_release.set()
        outcomes = await asyncio.gather(*drops, return_exceptions=True)
        assert all(isinstance(error, RuntimeGroupError) for error in outcomes)
        assert h.group.first_error() is original_error
        assert h.cells["waiting"].payload is original_error
        assert h.session.session_id not in h.shard._session_records
        assert not h.group.sessions and not h.session.live_irs
        assert h.session.resources is None
        h.process.terminate_and_join.assert_awaited_once()
        h.shard._generation_backend.close_session.assert_awaited_once()

        # The first successful cleanup still reports the fatal failure.
        # Only capacity-managed Groups retain receipts for debit retirement;
        # ordinary Groups still retry cleanup but have no physical credit owner.
        assert h.shard._resource_release_receipts == ({h.group.group_id: h.ids} if eager else {})
        receipt = await h.shard.drop_group(h.group.group_id)
        assert receipt == (h.ids if eager else ())
        h.runtime._confirm_session_resources(h.stream, receipt)
        h.runtime._confirm_session_resources(h.stream, receipt)
        if eager:
            assert h.runtime.session_capacity_snapshot()["confirmed_releases"] == 2
        await h.runtime.drop_group(h.stream)
        await h.runtime.drop_group(h.stream)
        if eager:
            assert h.runtime.session_capacity_snapshot()["outstanding"] == 0
        assert not h.runtime._reserved_group_shards and not h.binding.group_streams
        assert not h.shard._resource_release_receipts
        assert not h.binding.observer_task.done()
        h.process.terminate_and_join.assert_awaited_once()


async def test_finalization_error_wakes_protected_pause_without_claiming_release(monkeypatch):
    async with finalizing_session(monkeypatch, eager=False, protected=True) as h:
        await eventually(lambda: h.stream.protected)
        pause = h.spawn(h.runtime.pause_generation())
        await asyncio.sleep(0)
        assert not pause.done()
        h.agent_exit.set()
        with pytest.raises(RuntimeGroupError, match="backend_join did not settle"):
            await asyncio.wait_for(asyncio.shield(pause), timeout=2)
        assert h.stream.protected
        assert not h.stream.interrupted
        assert h.runtime.session_capacity_snapshot() is None
        assert h.group.resource_released_session_ids == {"done"}
        assert not h.cells["waiting"].terminal


async def test_shutdown_replays_cleanup_receipt_after_late_backend_exit(monkeypatch):
    async with finalizing_session(monkeypatch) as h:
        collector = h.spawn(h.runtime.collect_group(h.stream))
        h.agent_exit.set()
        with pytest.raises(RuntimeGroupError, match="backend_join did not settle"):
            await asyncio.wait_for(asyncio.shield(collector), timeout=2)
        original_error = h.group.finalization_error
        h.permit_release.set()
        await asyncio.wait_for(h.backend, timeout=2)

        # Shutdown's first drop commits cleanup but preserves the fatal error;
        # its second pass must consume the receipt without manual confirmation.
        with pytest.raises(RuntimeGroupError) as failure:
            await asyncio.wait_for(h.runtime.shutdown(), timeout=2)
        assert failure.value is original_error
        assert h.runtime.session_capacity_snapshot()["outstanding"] == 0
        assert h.runtime.session_capacity_snapshot()["confirmed_releases"] == 2
        assert not h.runtime._reserved_group_shards and not h.binding.group_streams
        assert not h.runtime._shards and h.binding.observer_task is None
        assert not h.shard._groups and not h.shard._session_records
        assert not h.shard._resource_release_receipts
        h.process.terminate_and_join.assert_awaited_once()
        h.shard._generation_backend.close_session.assert_awaited_once()
        await h.runtime.shutdown()


async def test_cancelled_cleanup_waiter_keeps_one_retry_owner(monkeypatch):
    async with finalizing_session(monkeypatch) as h:
        h.agent_exit.set()
        with pytest.raises(RuntimeGroupError, match="backend_join did not settle"):
            await asyncio.wait_for(asyncio.shield(h.watcher), timeout=2)
        h.permit_release.set()
        await asyncio.wait_for(h.backend, timeout=2)
        h.terminate_release.clear()
        first = h.spawn(h.shard._finish_session(h.session, None))
        await asyncio.wait_for(h.terminating.wait(), timeout=2)
        owner = h.session.finish_task
        first.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first
        assert not owner.done()
        second = h.spawn(h.shard._finish_session(h.session, None))
        await asyncio.sleep(0)
        assert h.session.finish_task is owner
        h.terminate_release.set()
        assert await second is None
        assert await h.shard._finish_session(h.session, None) is None
        assert h.session.finish_task is owner
        assert h.cells["waiting"].payload is h.group.finalization_error
        h.process.terminate_and_join.assert_awaited_once()
        h.shard._generation_backend.close_session.assert_awaited_once()


@pytest.mark.parametrize("eager", [False, True])
async def test_timed_out_export_has_one_delivery_failure_and_one_session_terminal(monkeypatch, tmp_path, eager):
    async with finalizing_session(monkeypatch, eager=eager) as h:
        h.args.dump_agentic_obs = str(tmp_path)
        h.args.agentic_obs_run_id = "finalization"
        h.shard._trajectory_recorder = ShardTrajectoryRecorder(h.args, shard_ordinal=0)
        h.session.observation = SessionObservation(export_ids=[7])
        sample = Sample(response="completed export", sample_export_id=7, metadata={})
        transport = SessionExportTransport(
            exports=({"name": None, "sample_payload": TrainingFieldArtifact.from_sample(sample).sample_payload},)
        )
        with pytest.raises(RuntimeGroupError, match="backend_join did not settle"):
            await h.shard._finish_session(h.session, transport)
        original_error = h.group.finalization_error
        with pytest.raises(RuntimeGroupError, match="backend_join did not settle"):
            await h.shard._finish_session(h.session, None)
        await eventually(lambda: h.stream.error is original_error)
        assert h.group.finalization_error is original_error
        assert h.session.observation.failure[:2] == ("session_cleanup_failed", "infra")
        assert not h.session.observation.terminal and not h.cells["waiting"].terminal
        if eager:
            assert h.runtime.session_capacity_snapshot()["outstanding"] == 1
        else:
            assert h.runtime.session_capacity_snapshot() is None

        h.permit_release.set()
        await asyncio.wait_for(h.backend, 2)
        with pytest.raises(RuntimeGroupError):
            await h.runtime.drop_group(h.stream)
        await h.runtime.drop_group(h.stream)
        if eager:
            assert h.runtime.session_capacity_snapshot()["outstanding"] == 0
        assert h.cells["waiting"].payload is original_error
        h.process.terminate_and_join.assert_awaited_once()
        h.shard._generation_backend.close_session.assert_awaited_once()
        await h.shard._trajectory_recorder.aclose()

    rows = [json.loads(line) for line in (tmp_path / "finalization/stages/shard_00.jsonl").read_text().splitlines()]
    delivery = [row for row in rows if row.get("operation") == "export_delivery"]
    assert [row["status"] for row in delivery] == ["start", "failed"]
    assert {row["sample_export_id"] for row in delivery} == {"7"}
    terminals = [row for row in rows if row.get("scope") == "session" and row.get("is_terminal")]
    assert len(terminals) == 1
    assert terminals[0]["status"] == "failed" and terminals[0]["category"] == "infra"
    assert terminals[0]["reason_code"] == "session_cleanup_failed"
