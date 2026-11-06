# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Exercise real Runtime/Shard gates with CPU process/backend substitutes."""

import asyncio
from argparse import Namespace
from types import SimpleNamespace

import pytest

from relax.agentic.observability import ShardTrajectoryRecorder
from relax.agentic.pipeline import (
    GroupInput,
    RuntimeGroupError,
    SessionExportTransport,
    SessionSpec,
    TrainingFieldArtifact,
)
from relax.agentic.pipeline.runtime import RuntimeDomain, _SessionShardBinding
from relax.agentic.profile import agentic_trace_events
from relax.agentic.session.service import AgenticSessionShard
from relax.utils.types import Sample


async def eventually(predicate):
    async with asyncio.timeout(2):
        while not predicate():
            await asyncio.sleep(0.005)


def runtime_fixture():
    args = Namespace(
        agentic_session_capacity_per_node=2,
        agentic_incremental_session_start=True,
        agentic_fixed_step_eager=True,
        rollout_global_dataset=False,
        n_samples_per_prompt=2,
        rollout_batch_size=2,
        over_sampling_batch_size=2,
        rollout_temperature=1.0,
        rollout_top_p=1.0,
        rollout_top_k=-1,
        rollout_max_response_len=8,
        rollout_stop=None,
        rollout_stop_token_ids=None,
        rollout_skip_special_tokens=False,
        sglang_enable_deterministic_inference=False,
    )
    shard = object.__new__(AgenticSessionShard.__ray_metadata__.modified_class)
    shard.args = args
    shard._trajectory_recorder = ShardTrajectoryRecorder(args, shard_ordinal=0)
    shard._groups = {}
    shard._session_records = {}
    shard._resource_release_receipts = {}
    shard._train_generation_open = False
    shard._standalone_eager_scope = False
    shard._fixed_step_eager_scope = True
    shard._eager_shutdown = False
    shard._rollout_epoch = None
    shard._eager_gate_lock = asyncio.Lock()
    shard._state_revision = 0
    shard._state_changed = asyncio.Event()
    started, generated = [], []

    async def timeout_active(_active):
        pass

    def start(session, _input, **_kwargs):
        started.append(session.session_id)
        session.resources = SimpleNamespace(process=SimpleNamespace(set_timeout_active=timeout_active))

    shard._start_session = start
    shard._dispatch_queued_irs_locked = lambda session: generated.append((session.session_id, session.rollout_id))

    class Handle:
        def __getattr__(self, name):
            return SimpleNamespace(
                remote=lambda *args, **kwargs: asyncio.create_task(getattr(shard, name)(*args, **kwargs))
            )

    binding = _SessionShardBinding(actor_name="owner", handle=Handle(), node_id="host")
    runtime = RuntimeDomain(args, "train", [binding], concurrency=3)
    return runtime, shard, started, generated


async def prepare(runtime, group_id):
    samples = [Sample(index=index, group_index=0, prompt="p") for index in range(2)]
    decision = asyncio.get_running_loop().create_future()
    stream = await runtime.prepare_group(GroupInput(group_id, samples), decision)
    assert decision.result() is True
    await runtime.lease_group(stream, runtime._rollout_epoch)
    return stream


async def test_late_member_keeps_group_seed_and_observation_milestones(tmp_path):
    runtime, shard, _started, _generated = runtime_fixture()
    shard.args.sglang_enable_deterministic_inference = True
    shard.args.rollout_seed = 101
    shard.args.dump_agentic_obs = str(tmp_path)
    shard.args.agentic_obs_run_id = "late_member"
    shard._trajectory_recorder = ShardTrajectoryRecorder(shard.args, shard_ordinal=0)
    shard._train_generation_open = True
    shard._rollout_epoch = 0
    specs = tuple(SessionSpec(session_id=sid, require_resource_release=True) for sid in ("first", "late"))
    await shard.start_group("train", "g", specs, start_session_ids=("first",), fixed_step_eager_epoch=0)
    group = shard._groups["g"]
    first, late = group.sessions
    await shard.lease_group("g", 0)
    await ready(shard, first)
    first_events = agentic_trace_events(first.forest.static_metadata)
    assert first.session_sampling_params["sampling_seed"] == 101
    finish(shard, first)
    group.sessions.remove(first)

    await shard.start_group_sessions("g", ("late",))
    assert late.session_sampling_params["sampling_seed"] == 102
    late_events = agentic_trace_events(late.forest.static_metadata)
    for event in ("group_runtime_leased_at", "group_generation_open_at"):
        assert late_events[event] == first_events[event]
    await ready(shard, late)
    assert "group_first_requests_ready_at" in late_events
    assert "group_first_requests_ready_at" not in first_events
    finish(shard, late)
    group.sessions.remove(late)
    await shard.release_group("g")
    await shard._trajectory_recorder.aclose()
    await runtime.shutdown()


async def ready(shard, session):
    session.group.first_ir_seen_session_ids.add(session.session_id)
    await shard._activate_eager_session(session)
    shard._notify_state_change(session.group)


def finish(shard, session, *, receipt=True):
    session.resources = None
    sample = Sample(index=0, prompt="p", response="a", tokens=[1, 2], response_length=1, reward=1.0)
    session.result_cell.publish(
        SessionExportTransport(
            exports=(
                {
                    "name": None,
                    "sample_payload": TrainingFieldArtifact.from_sample(sample).sample_payload,
                },
            )
        )
    )
    if receipt:
        session.group.resource_released_session_ids.add(session.session_id)
    shard._session_records.pop(session.session_id)
    shard._notify_state_change(session.group)


async def test_incremental_eager_runs_before_group_barrier_and_drains_two_epochs():
    runtime, shard, started, generated = runtime_fixture()
    try:
        for epoch in range(2):
            await runtime.resume_generation(epoch)
            stream_a = await prepare(runtime, f"a{epoch}")
            a = shard._groups[stream_a.group_id]
            for session in a.sessions:
                await ready(shard, session)
            await eventually(lambda: stream_a.first_request_barrier.done())
            finish(shard, a.sessions[0], receipt=False)
            waiting_b = asyncio.create_task(prepare(runtime, f"b{epoch}"))
            await asyncio.sleep(0.02)
            assert not waiting_b.done()  # A terminal result cannot refund an unconfirmed physical slot.
            a.resource_released_session_ids.add(a.sessions[0].session_id)
            shard._notify_state_change(a)
            stream_b = await asyncio.wait_for(waiting_b, 2)
            b = shard._groups[stream_b.group_id]
            assert b.sessions[0].session_id in started
            assert b.sessions[1].session_id not in started
            await ready(shard, b.sessions[0])
            assert (b.sessions[0].session_id, epoch) in generated
            assert not b.all_first_ir_arrived
            assert not a.sessions[1].result_cell.terminal
            with pytest.raises(RuntimeGroupError, match="resident task resources"):
                await runtime.pause_generation()
            collect_b = asyncio.create_task(runtime.collect_group(stream_b))
            assert not collect_b.done()
            finish(shard, a.sessions[1])
            await eventually(lambda: b.sessions[1].session_id in started)
            await ready(shard, b.sessions[1])
            result_a = await asyncio.wait_for(runtime.collect_group(stream_a), 2)
            assert len(result_a.sessions) == 2
            finish(shard, b.sessions[0])
            await asyncio.sleep(0.02)
            assert not collect_b.done()  # Export still requires every original member.
            finish(shard, b.sessions[1])
            result_b = await asyncio.wait_for(collect_b, 2)
            assert len(result_b.sessions) == 2
            assert runtime.session_capacity_snapshot()["outstanding"] == 0
            assert runtime.session_capacity_snapshot()["global_peak"] == 2
            await runtime.pause_generation()
            with pytest.raises(RuntimeGroupError, match="live rollout epoch"):
                await runtime.lease_group(stream_a, epoch)
            with pytest.raises(RuntimeGroupError, match="new drained epoch"):
                await runtime.resume_generation(epoch)
    finally:
        await runtime._stop_shard_observers()


def test_fixed_step_rejects_cross_step_modes():
    runtime, _shard, _started, _generated = runtime_fixture()
    for name in ("partial_rollout", "fully_async", "agentic_prelaunch", "use_dynamic_global_batch_size"):
        args = Namespace(**vars(runtime.args), **{name: True})
        with pytest.raises(ValueError, match="complete synchronous steps"):
            RuntimeDomain(args, "train", runtime._shards, concurrency=3)
