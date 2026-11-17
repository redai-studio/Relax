# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Exercise real Runtime, Shard, Forest and permit ownership on CPU.

Only external process, generation HTTP and Ray transport are substituted.
"""

import asyncio
from argparse import Namespace
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from relax.agentic.observability import ShardTrajectoryRecorder
from relax.agentic.pipeline import GroupInput
from relax.agentic.pipeline import runtime as runtime_module
from relax.agentic.pipeline.prepare import PrepareDomain
from relax.agentic.pipeline.runtime import BackendGenerateResult, EncodedMessages, RuntimeDomain, _SessionShardBinding
from relax.agentic.runner import AgentExecutionError
from relax.agentic.session import service as service_module
from relax.agentic.session.service import AgenticSessionShard, SessionPhase
from relax.utils.types import Sample


pytestmark = pytest.mark.asyncio


async def eventually(predicate):
    async with asyncio.timeout(3):
        while not predicate():
            await asyncio.sleep(0)


@asynccontextmanager
async def harness(*, mode="train", **overrides):
    args = Namespace(
        fully_async=False,
        partial_rollout=False,
        partial_rollout_max_aborted_count=None,
        use_slime_router=False,
        rollout_global_dataset=False,
        agentic_program_admission=False,
        agentic_admission_scope="all",
        agentic_session_lifecycle=True,
        use_rollout_routing_replay=False,
        n_samples_per_prompt=2,
        rollout_temperature=1.0,
        rollout_top_p=1.0,
        rollout_top_k=-1,
        rollout_max_response_len=8,
        rollout_stop=None,
        rollout_stop_token_ids=None,
        rollout_skip_special_tokens=False,
        sglang_enable_deterministic_inference=False,
        rollout_max_context_len=100,
        eval_max_context_len=100,
        eval_max_response_len=8,
        agentic_reasoning_parser=None,
        agentic_tool_call_parser=None,
    )
    for name, value in overrides.items():
        setattr(args, name, value)
    shard = object.__new__(AgenticSessionShard.__ray_metadata__.modified_class)
    shard.args = args
    shard._groups, shard._session_records = {}, {}
    shard._train_generation_open = False
    shard._generation_gate_lock = asyncio.Lock()
    shard._state_revision, shard._state_changed = 0, asyncio.Event()
    shard._agent_cleanup_tasks, shard._permit_cleanup_tasks = set(), set()
    shard._admission_client = None
    shard._lifecycle_close_count = shard._lifecycle_close_failure_count = 0
    shard._trajectory_recorder = ShardTrajectoryRecorder(args, shard_ordinal=0)
    processes, generated, held_permits = {}, [], set()
    generation_gate = asyncio.Event()
    generation_gate.set()

    class Process:
        def __init__(self, session_id):
            self.exit = asyncio.Event()
            self.cleanup = asyncio.Event()
            self.cleanup.set()
            self.timeout_events = []
            self.output = SimpleNamespace(reward=1.0, metadata={}, records=())
            self.error = None
            self.cleanup_error = None
            processes[session_id] = self

        async def wait(self):
            await self.exit.wait()
            if self.error is not None:
                raise self.error

        async def set_timeout_active(self, active):
            self.timeout_events.append(active)

        async def terminate_and_join(self):
            await self.cleanup.wait()
            if self.cleanup_error is not None:
                raise self.cleanup_error
            self.exit.set()

    shard._agent_launcher = SimpleNamespace(start_agent=lambda spec: Process(spec.session_id))

    async def encode(*_args, **_kwargs):
        return EncodedMessages(train_prompt_ids=[112], backend_prompt_ids=[112])

    async def generate(**kwargs):
        generated.append(kwargs["session_id"])
        await generation_gate.wait()
        return BackendGenerateResult([97], [-0.5], "stop", {"prompt_tokens": 1}, 0.01)

    shard._generation_backend = SimpleNamespace(
        compiler=SimpleNamespace(
            apply_chat_template_kwargs={}, encode_messages=encode, encode_observation_delta=encode
        ),
        tokenizer=SimpleNamespace(decode=lambda ids, **_kw: "".join(chr(i) for i in ids)),
        generate=generate,
        abort_request=AsyncMock(),
        close_session=AsyncMock(return_value=True),
        shutdown=AsyncMock(),
    )

    async def acquire(permit_id):
        held_permits.add(permit_id)
        return True

    async def release(permit_id):
        held_permits.remove(permit_id)

    shard._sglang_request_limiter = SimpleNamespace(
        acquire_request_permit=SimpleNamespace(remote=lambda pid: asyncio.create_task(acquire(pid))),
        release_request_permit=SimpleNamespace(remote=lambda pid: asyncio.create_task(release(pid))),
    )

    class Handle:
        def __getattr__(self, name):
            return SimpleNamespace(remote=lambda *a, **kw: asyncio.create_task(getattr(shard, name)(*a, **kw)))

    binding = _SessionShardBinding("owner", Handle())
    runtime = RuntimeDomain(args, mode, [binding], concurrency=1)

    async def prepare(group_id="g"):
        return await runtime.prepare_group(GroupInput(group_id, [Sample(prompt="p", index=i) for i in range(2)]))

    async def chat(session):
        return await shard.chat(
            session_id=session.session_id,
            messages=[{"role": "user", "content": "p"}],
            tools=[],
            chat_template_kwargs={},
            max_completion_tokens=None,
            stop=None,
            seed=None,
            logprobs=False,
        )

    h = SimpleNamespace(
        runtime=runtime,
        shard=shard,
        processes=processes,
        generated=generated,
        held_permits=held_permits,
        generation_gate=generation_gate,
        prepare=prepare,
        chat=chat,
    )
    try:
        yield h
    finally:
        for process in processes.values():
            process.cleanup.set()
        generation_gate.set()
        await runtime.shutdown()
        while shard._agent_cleanup_tasks:
            await asyncio.gather(*tuple(shard._agent_cleanup_tasks), return_exceptions=True)
        await shard._trajectory_recorder.aclose()


@pytest.mark.parametrize("mode", ["train", "eval"])
async def test_leased_member_generates_before_siblings_first_request(mode):
    async with harness(mode=mode) as h:
        stream = await h.prepare()
        first, late = tuple(h.shard._groups[stream.group_id].sessions)
        assert len(h.processes) == 2
        before_lease = asyncio.create_task(h.chat(first))
        await eventually(lambda: bool(first.live_irs))
        assert not h.generated and not h.processes[first.session_id].timeout_events
        if mode == "train":
            await h.runtime.resume_generation(7)
        await h.runtime.lease_group(stream, 7)
        response = await asyncio.wait_for(before_lease, 3)
        assert response["message"]["content"] == "a"
        assert h.generated == [first.session_id]
        assert not h.processes[late.session_id].timeout_events
        group = first.group
        assert not group.all_first_ir_arrived and not group.interrupted
        collector = asyncio.create_task(h.runtime.collect_group(stream))
        h.processes[first.session_id].exit.set()
        await eventually(lambda: first.result_cell.terminal)
        assert not collector.done()
        response = await h.chat(late)
        assert response["message"]["content"] == "a"
        h.processes[late.session_id].exit.set()
        result = await asyncio.wait_for(collector, 3)
        assert len(result.sessions) == 2 and [s.tokens for s in result.samples] == [[112, 97], [112, 97]]
        assert all(s.metadata["start_rollout_id"] == 7 for s in result.samples)
        assert h.runtime.resident_group_count == 0


async def test_late_first_request_waits_for_resume_and_uses_current_rollout_id():
    async with harness(partial_rollout=True) as h:
        stream = await h.prepare()
        first, late = h.shard._groups[stream.group_id].sessions
        await h.runtime.resume_generation(3)
        await h.runtime.lease_group(stream, 3)
        await h.chat(first)
        await h.runtime.pause_generation()
        late_chat = asyncio.create_task(h.chat(late))
        await eventually(lambda: bool(late.live_irs))
        assert h.generated == [first.session_id]
        assert not h.processes[late.session_id].timeout_events
        await h.runtime.resume_generation(4)
        assert (await late_chat)["message"]["content"] == "a"
        assert (
            late.forest.build_sample(
                leaf_state_hash=next(iter(late.resp_state_hash_by_request_id.values())),
                tokenizer=h.shard._generation_backend.tokenizer,
            ).metadata["start_rollout_id"]
            == 4
        )


@pytest.mark.parametrize("cleanup_failure", [False, True])
async def test_export_and_group_capacity_do_not_wait_for_cleanup(cleanup_failure):
    async with harness() as h:
        stream = await h.prepare()
        sessions = tuple(h.shard._groups[stream.group_id].sessions)
        await h.runtime.resume_generation(0)
        await h.runtime.lease_group(stream, 0)
        for session in sessions:
            await h.chat(session)
            process = h.processes[session.session_id]
            process.cleanup.clear()
            if cleanup_failure:
                process.cleanup_error = OSError("process cleanup unavailable")
            process.exit.set()
        h.shard._generation_backend.close_session = AsyncMock(side_effect=OSError("cleanup unavailable"))
        group = await asyncio.wait_for(h.runtime.collect_group(stream), 3)
        assert len(group.sessions) == 2 and h.runtime.resident_group_count == 0
        replacement = await asyncio.wait_for(h.prepare("replacement"), 3)
        assert replacement.group_id == "replacement"
        assert all(s.phase is SessionPhase.FINALIZING for s in sessions)
        assert not h.held_permits


async def test_failed_member_drops_siblings_and_compensates_inflight_permits():
    async with harness() as h:
        stream = await h.prepare()
        first, sibling = tuple(h.shard._groups[stream.group_id].sessions)
        await h.runtime.resume_generation(0)
        await h.runtime.lease_group(stream, 0)
        h.generation_gate.clear()
        pending = asyncio.create_task(h.chat(sibling))
        await eventually(lambda: bool(h.held_permits))
        h.processes[sibling.session_id].cleanup.clear()
        h.processes[first.session_id].error = AgentExecutionError("agent failed")
        h.processes[first.session_id].exit.set()
        assert await asyncio.wait_for(h.runtime.collect_group(stream), 3) is None
        assert (await asyncio.wait_for(pending, 3))["error"]["code"] == "session_discarded"
        await eventually(lambda: not h.held_permits)
        assert not h.runtime.resident_group_count and not h.shard._session_records


async def test_discard_prepared_group_releases_registration_without_a_lease():
    async with harness() as h:
        prepare = PrepareDomain("train", None, h.runtime)
        prepare.open_step(prelaunch_next_step=False)
        prepare._start_prepare(GroupInput("g", [Sample(prompt="p"), Sample(prompt="p")]))
        await asyncio.gather(*prepare._warming_prepare_tasks)
        await eventually(lambda: prepare.ready_group_count == 1)
        await prepare.discard_step()
        await eventually(lambda: not h.shard._session_records)
        assert not h.generated and not h.runtime.resident_group_count and not h.shard._session_records


@pytest.mark.parametrize("failed", [False, True])
async def test_remote_drop_cannot_hold_group_capacity_or_fail_shutdown(monkeypatch, failed):
    async with harness() as h:
        stream = await h.prepare()
        entered, release = asyncio.Event(), asyncio.Event()

        async def remote_drop(*_args, **_kwargs):
            entered.set()
            if failed:
                raise OSError("remote cleanup unavailable")
            await release.wait()

        await h.runtime._stop_shard_observers()
        stream.shard.handle = SimpleNamespace(
            drop_group=SimpleNamespace(remote=remote_drop), trim_memory=SimpleNamespace(remote=AsyncMock())
        )
        await h.runtime.drop_group(stream)
        await entered.wait()
        assert not h.runtime.resident_group_count
        await asyncio.wait_for(h.runtime._resident_group_permits.acquire(), 1)
        h.runtime._resident_group_permits.release()
        monkeypatch.setattr(runtime_module, "_CLEANUP_GRACE_S", 0.01)
        await asyncio.wait_for(h.runtime.shutdown(), 1)
        assert not release.is_set()
        release.set()
        await h.shard.drop_group(stream.group_id)


async def test_drop_detaches_result_transfer_that_resists_cancellation(monkeypatch):
    async with harness() as h:
        stream = await h.prepare()
        entered, release = asyncio.Event(), asyncio.Event()

        async def transfer():
            entered.set()
            try:
                await release.wait()
            except asyncio.CancelledError:
                await release.wait()

        task = asyncio.create_task(transfer())
        stream.session_results[0].take_task = task
        await entered.wait()
        monkeypatch.setattr(runtime_module, "_CLEANUP_GRACE_S", 0.01)
        try:
            await asyncio.wait_for(h.runtime.drop_group(stream), 1)
            assert not h.runtime.resident_group_count and not task.done()
        finally:
            release.set()
            await task


@pytest.mark.parametrize("mode", ["train", "eval"])
async def test_first_ir_timeout_drops_group_and_releases_capacity(monkeypatch, mode):
    from relax.agentic.observability import SessionObservation

    monkeypatch.setattr(service_module, "_FIRST_IR_TIMEOUT_S", 0.02)
    async with harness(mode=mode) as h:
        stream = await h.prepare()
        sessions = tuple(h.shard._groups[stream.group_id].sessions)
        for session in sessions:
            session.observation = SessionObservation()
            h.processes[session.session_id].cleanup.clear()
        await h.runtime.lease_group(stream, 0)
        assert await asyncio.wait_for(h.runtime.collect_group(stream), 2) is None
        assert any(s.observation.failure[0] == "agent_startup_timeout" for s in sessions if s.observation.failure)
        assert h.runtime.resident_group_count == 0
        assert (await asyncio.wait_for(h.prepare("replacement"), 2)).group_id == "replacement"


async def test_first_ir_disarms_startup_deadline_before_lease(monkeypatch):
    monkeypatch.setattr(service_module, "_FIRST_IR_TIMEOUT_S", 0.1)
    async with harness() as h:
        stream = await h.prepare()
        sessions = tuple(h.shard._groups[stream.group_id].sessions)
        chats = [asyncio.create_task(h.chat(session)) for session in sessions]
        await eventually(lambda: all(session.first_ir_ready.is_set() for session in sessions))
        expired = asyncio.Event()
        asyncio.get_running_loop().call_later(0.15, expired.set)
        await expired.wait()
        assert all(not session.result_cell.terminal for session in sessions)
        assert not h.generated
        await h.runtime.resume_generation(0)
        await h.runtime.lease_group(stream, 0)
        await asyncio.gather(*chats)


async def test_drop_before_first_ir_does_not_publish_again_on_late_event():
    async with harness() as h:
        stream = await h.prepare()
        sessions = tuple(h.shard._groups[stream.group_id].sessions)
        watchers = [session.resources.watcher_task for session in sessions]
        await h.runtime.drop_group(stream)
        await eventually(lambda: not h.shard._session_records)
        for session in sessions:
            session.first_ir_ready.set()
        await asyncio.gather(*watchers, return_exceptions=True)
        assert all(session.result_cell.dropped for session in sessions)
