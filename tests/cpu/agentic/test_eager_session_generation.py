# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Exercise real Runtime/Shard gates with CPU process/backend substitutes."""

import asyncio
from argparse import Namespace
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from relax.agentic.observability import ShardTrajectoryRecorder
from relax.agentic.pipeline import (
    GroupInput,
    RuntimeGroupError,
    SessionExportTransport,
    SessionSpec,
    TrainingFieldArtifact,
)
from relax.agentic.pipeline.prepare import PrepareDomain
from relax.agentic.pipeline.runtime import BackendGenerateResult, RuntimeDomain, _SessionShardBinding
from relax.agentic.profile import agentic_trace_events
from relax.agentic.session.service import AgenticSessionShard, GroupOwnership
from relax.agentic.session.state import InflightRequest, RequestKind
from relax.utils.types import Sample


async def eventually(predicate):
    async with asyncio.timeout(2):
        while not predicate():
            await asyncio.sleep(0.005)


def runtime_fixture(*, capacity=2, rollout_mode="train", **overrides):
    args = Namespace(
        agentic_session_capacity_per_node=capacity,
        fully_async=False,
        partial_rollout=False,
        partial_rollout_max_aborted_count=None,
        agentic_prelaunch=False,
        use_dynamic_global_batch_size=False,
        use_slime_router=False,
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
    for name, value in overrides.items():
        setattr(args, name, value)
    shard = object.__new__(AgenticSessionShard.__ray_metadata__.modified_class)
    shard.args = args
    shard._trajectory_recorder = ShardTrajectoryRecorder(args, shard_ordinal=0)
    shard._groups = {}
    shard._session_records = {}
    shard._resource_release_receipts = {}
    shard._train_generation_open = False
    shard._eager_shutdown = False
    shard._rollout_epoch = None
    shard._eager_gate_lock = asyncio.Lock()
    shard._state_revision = 0
    shard._state_changed = asyncio.Event()
    started, generated = [], []
    shard._timeout_events = []

    def start(session, _input, **_kwargs):
        async def timeout_active(active):
            shard._timeout_events.append((session.session_id, active))

        started.append(session.session_id)
        session.resources = SimpleNamespace(
            process=SimpleNamespace(set_timeout_active=timeout_active),
            watcher_task=asyncio.create_task(asyncio.sleep(0)),
        )

    shard._start_session = start
    shard._dispatch_queued_irs_locked = lambda session: generated.append((session.session_id, session.rollout_id))

    class Handle:
        def __getattr__(self, name):
            return SimpleNamespace(
                remote=lambda *args, **kwargs: asyncio.create_task(getattr(shard, name)(*args, **kwargs))
            )

    binding = _SessionShardBinding(actor_name="owner", handle=Handle(), node_id="host")
    runtime = RuntimeDomain(args, rollout_mode, [binding], concurrency=3)
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
    await shard.start_group("train", "g", specs, start_session_ids=("first",))
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


async def test_capacity_alone_runs_partial_groups_before_barrier_and_drains_two_epochs():
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
                runtime._assert_epoch_drained()
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
            with pytest.raises(RuntimeGroupError, match="new paused rollout"):
                await runtime.resume_generation(epoch)
    finally:
        await runtime._stop_shard_observers()


@pytest.mark.parametrize("rollout_mode,capacity", [("train", None), ("eval", 2)])
async def test_ordinary_train_and_eval_keep_complete_first_request_barrier(rollout_mode, capacity):
    runtime, shard, started, generated = runtime_fixture(capacity=capacity, rollout_mode=rollout_mode)
    try:
        if rollout_mode == "train":
            await runtime.resume_generation(0)
        samples = [Sample(index=i, group_index=0, prompt="p") for i in range(2)]
        preparing = asyncio.create_task(runtime.prepare_group(GroupInput("ordinary", samples)))
        await eventually(lambda: "ordinary" in shard._groups)
        group = shard._groups["ordinary"]
        assert runtime.session_capacity_snapshot() is None
        assert not group.eager_session_generation and not group.requires_resource_release
        assert set(started) == set(group.result_cells)
        for index, session in enumerate(group.sessions):
            session.live_irs.add(
                InflightRequest(
                    request_id=f"request-{index}",
                    parent_state_hash="hash",
                    rollout_id=0,
                    kind=RequestKind.FRESH,
                    abort_count=0,
                    waiter=asyncio.get_running_loop().create_future(),
                    wall_started_at=0.0,
                    sampling_params={},
                )
            )
            await ready(shard, session)
            if index == 0:
                await asyncio.sleep(0.02)
                assert not preparing.done()
                assert not generated  # The first member still waits for its whole Group.
        stream = await asyncio.wait_for(preparing, 2)
        assert not stream.eager_session_generation and stream.preparation_task is None
        await runtime.lease_group(stream, 0)
        assert set(generated) == {(sid, 0) for sid in group.result_cells}
        for session in group.sessions:
            session.live_irs.clear()
            finish(shard, session, receipt=False)
        assert len((await runtime.collect_group(stream)).sessions) == 2
        runtime._assert_epoch_drained()
        if rollout_mode == "train":
            await runtime.pause_generation()
    finally:
        await runtime._stop_shard_observers()


@pytest.mark.parametrize("mode", ["partial_rollout", "agentic_prelaunch", "use_dynamic_global_batch_size"])
def test_unmanaged_train_and_eval_do_not_inherit_capacity_mode_restrictions(mode):
    ordinary, *_ = runtime_fixture(capacity=None, **{mode: True})
    evaluation, *_ = runtime_fixture(rollout_mode="eval", **{mode: True})
    assert ordinary.session_capacity_snapshot() is None
    assert evaluation.session_capacity_snapshot() is None


async def test_capacity_train_cannot_bypass_credit_accounting_at_registration():
    _runtime, shard, started, _generated = runtime_fixture()
    specs = (SessionSpec(session_id="unaccounted"),)
    for options in ({}, {"start_session_ids": ("unaccounted",)}):
        with pytest.raises(RuntimeGroupError, match="resource-accounted train scope"):
            await shard.start_group("train", "invalid", specs, **options)
        assert not shard._groups and not shard._session_records and not started


@pytest.mark.parametrize("rejection", ["shutdown", "missing_credits", "foreign_credits"])
async def test_rejected_registration_refunds_only_its_never_started_identities(rejection):
    runtime, shard, started, _generated = runtime_fixture()
    runtime._start_shard_observers = lambda: None
    # Leave one physical credit available, reproducing a partial new Group.
    held = runtime._session_capacity.reserve("held", "host", 1)
    original_start = shard.start_group

    async def reject(*args, **kwargs):
        if rejection == "shutdown":
            shard._eager_shutdown = True
        elif rejection == "missing_credits":
            kwargs["start_session_ids"] = None
        else:
            kwargs["start_session_ids"] = ("not-in-this-group",)
        await original_start(*args, **kwargs)

    shard.start_group = reject
    message = "exact budgeted" if rejection == "foreign_credits" else "resource-accounted train scope"
    with pytest.raises(RuntimeGroupError, match=message):
        await runtime.prepare_group(GroupInput("rejected", [Sample(prompt="p") for _ in range(2)]))
    assert not started and not shard._groups and not shard._session_records
    assert "rejected" not in runtime._reserved_group_shards
    assert not runtime._shards[0].group_streams
    assert await shard.get_resource_release_receipt("rejected") == ()  # Runtime acknowledged the exact proof.
    assert runtime.session_capacity_snapshot()["outstanding"] == 1  # The unrelated reservation was not refunded.
    assert runtime._resident_group_permits._value == runtime.concurrency
    runtime._session_capacity.retire(held)


async def test_rejected_registration_does_not_drop_an_existing_group_or_replace_its_receipt():
    runtime, shard, started, _generated = runtime_fixture()
    runtime._start_shard_observers = lambda: None
    released = install_resource_cleanup(shard)
    specs = tuple(SessionSpec(session_id=sid, require_resource_release=True) for sid in ("old-0", "old-1"))
    await shard.start_group("train", "same-name", specs, start_session_ids=("old-0", "old-1"))
    previous = shard._groups["same-name"]
    shard._eager_shutdown = True
    with pytest.raises(RuntimeError, match="Unconfirmed task resources") as error:
        await runtime.prepare_group(GroupInput("same-name", [Sample(prompt="p") for _ in range(2)]))
    assert "already resident" in str(error.value.__context__)
    assert shard._groups["same-name"] is previous
    assert started == ["old-0", "old-1"]
    assert await shard.get_resource_release_receipt("same-name") == ()
    assert runtime.session_capacity_snapshot()["outstanding"] == 2
    with pytest.raises(RuntimeGroupError, match="shutdown has unconfirmed task resources"):
        await runtime.shutdown()
    assert shard._groups["same-name"] is previous
    assert not previous.closing and previous.drop_task is None and previous.stop_reason is None
    assert not released
    assert runtime.session_capacity_snapshot()["outstanding"] == 2
    assert "same-name" in runtime._reserved_group_shards
    await shard.drop_group("same-name")  # Test-owned previous owner, never dropped by failed registration.


@pytest.mark.parametrize("expected_ids", [("foreign",), ("old-0",), ("old-0", "old-1", "old-1")])
async def test_drop_group_checks_exact_session_owner_before_mutating_resident_group(expected_ids):
    _runtime, shard, _started, _generated = runtime_fixture()
    released = install_resource_cleanup(shard)
    specs = tuple(SessionSpec(session_id=sid, require_resource_release=True) for sid in ("old-0", "old-1"))
    await shard.start_group("train", "same-name", specs, start_session_ids=("old-0", "old-1"))
    group = shard._groups["same-name"]
    with pytest.raises(RuntimeGroupError, match="cleanup identities do not match"):
        await shard.drop_group("same-name", expected_session_ids=expected_ids, reason=("test", "test", "test"))
    assert shard._groups["same-name"] is group
    assert not group.closing and group.drop_task is None and group.stop_reason is None
    assert not released
    # Order is immaterial; exact membership permits the original cleanup path.
    assert await shard.drop_group("same-name", expected_session_ids=("old-1", "old-0")) == ("old-0", "old-1")


async def test_lost_registration_response_still_cleans_matching_session_owner_on_shutdown():
    runtime, shard, started, _generated = runtime_fixture()
    runtime._start_shard_observers = lambda: None
    released = install_resource_cleanup(shard)
    original_start = shard.start_group

    async def registered_but_response_lost(*args, **kwargs):
        await original_start(*args, **kwargs)
        raise ConnectionError("registered response was lost")

    shard.start_group = registered_but_response_lost
    with pytest.raises(RuntimeError, match="Unconfirmed task resources"):
        await runtime.prepare_group(GroupInput("unknown", [Sample(prompt="p") for _ in range(2)]))
    assert len(started) == 2 and not released
    assert "unknown" in shard._groups and "unknown" in runtime._reserved_group_shards
    assert runtime.session_capacity_snapshot()["outstanding"] == 2
    await runtime.shutdown()
    assert set(released) == set(started)
    assert runtime.session_capacity_snapshot()["outstanding"] == 0
    assert not runtime._reserved_group_shards and not shard._groups and not shard._resource_release_receipts


@pytest.mark.parametrize("rollout_mode,capacity", [("train", None), ("eval", 2)])
async def test_unmanaged_runtime_drop_passes_exact_ids_to_the_internal_shard(rollout_mode, capacity):
    runtime, shard, started, _generated = runtime_fixture(capacity=capacity, rollout_mode=rollout_mode)
    runtime._start_shard_observers = lambda: None
    released = install_resource_cleanup(shard)
    preparing = asyncio.create_task(
        runtime.prepare_group(GroupInput("ordinary-drop", [Sample(prompt="p") for _ in range(2)]))
    )
    await eventually(lambda: "ordinary-drop" in shard._groups)
    stream = runtime._shards[0].group_streams["ordinary-drop"]
    stream.first_request_barrier.set_result(True)
    assert await preparing is stream
    assert runtime.session_capacity_snapshot() is None
    await runtime.drop_group(stream)
    assert set(released) == set(started) and len(released) == 2
    assert not shard._groups and not runtime._reserved_group_shards
    await runtime.shutdown()


async def test_rejected_registration_cannot_consume_an_older_owners_receipt():
    runtime, shard, started, _generated = runtime_fixture()
    runtime._start_shard_observers = lambda: None
    shard._eager_shutdown = True
    shard._resource_release_receipts["reused"] = ("previous-owner",)
    with pytest.raises(ValueError, match="outside its reservation"):
        await runtime.prepare_group(GroupInput("reused", [Sample(prompt="p") for _ in range(2)]))
    assert not started
    assert await shard.get_resource_release_receipt("reused") == ("previous-owner",)
    assert runtime.session_capacity_snapshot()["outstanding"] == 2
    with pytest.raises(RuntimeGroupError, match="shutdown has unconfirmed task resources"):
        await runtime.shutdown()
    assert await shard.get_resource_release_receipt("reused") == ("previous-owner",)
    assert runtime.session_capacity_snapshot()["outstanding"] == 2


@pytest.mark.parametrize("unknown", ["rpc_error", "rpc_cancelled"])
async def test_failed_registration_without_a_remote_proof_retains_credits(unknown):
    runtime, shard, started, _generated = runtime_fixture()
    runtime._start_shard_observers = lambda: None

    async def unavailable(*_args, **_kwargs):
        if unknown == "rpc_cancelled":
            raise asyncio.CancelledError
        raise ConnectionError("registration response unavailable")

    shard.start_group = unavailable
    with pytest.raises(RuntimeError, match="Unconfirmed task resources"):
        await runtime.prepare_group(GroupInput("unknown", [Sample(prompt="p") for _ in range(2)]))
    assert not started and not shard._groups
    assert await shard.get_resource_release_receipt("unknown") == ()
    assert runtime.session_capacity_snapshot()["outstanding"] == 2
    assert "unknown" in runtime._reserved_group_shards
    assert "unknown" in runtime._shards[0].group_streams


async def test_registration_cancellation_joins_the_remote_receipt_before_refunding():
    runtime, shard, started, _generated = runtime_fixture()
    runtime._start_shard_observers = lambda: None
    shard._eager_shutdown = True
    entered, release = asyncio.Event(), asyncio.Event()
    original_receipt = shard.get_resource_release_receipt

    async def delayed_receipt(group_id):
        entered.set()
        await release.wait()
        return await original_receipt(group_id)

    shard.get_resource_release_receipt = delayed_receipt
    preparing = asyncio.create_task(
        runtime.prepare_group(GroupInput("cancelled", [Sample(prompt="p") for _ in range(2)]))
    )
    try:
        await asyncio.wait_for(entered.wait(), 2)
        preparing.cancel()
        await asyncio.sleep(0)
        assert not preparing.done()
        assert runtime.session_capacity_snapshot()["outstanding"] == 2
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(preparing, 2)
        assert not started and not runtime._reserved_group_shards
        assert runtime.session_capacity_snapshot()["outstanding"] == 0
    finally:
        release.set()
        await asyncio.gather(preparing, return_exceptions=True)


@pytest.mark.parametrize("cleanup_confirmed", [False, True])
async def test_started_registration_failure_requires_committed_physical_cleanup(cleanup_confirmed):
    runtime, shard, started, _generated = runtime_fixture()
    runtime._start_shard_observers = lambda: None
    original_start = shard._start_session
    released = install_resource_cleanup(shard)

    def fail_after_process_start(*args, **kwargs):
        original_start(*args, **kwargs)
        raise RuntimeGroupError("process initialization failed after launch")

    async def unconfirmed_cleanup(*_args, **_kwargs):
        raise RuntimeGroupError("physical cleanup not confirmed")

    shard._start_session = fail_after_process_start
    if not cleanup_confirmed:
        shard._finish_session = unconfirmed_cleanup
    expected = "process initialization failed after launch" if cleanup_confirmed else "Unconfirmed task resources"
    with pytest.raises(RuntimeError, match=expected):
        await runtime.prepare_group(GroupInput("started", [Sample(prompt="p") for _ in range(2)]))
    assert len(started) == 1
    assert runtime.session_capacity_snapshot()["outstanding"] == (0 if cleanup_confirmed else 2)
    if cleanup_confirmed:
        assert len(released) == 2  # Started member plus the explicitly never-started sibling.
        assert not shard._groups and not runtime._reserved_group_shards
    else:
        assert not released and "started" in shard._groups
        assert await shard.get_resource_release_receipt("started") == ()
        install_resource_cleanup(shard)
        await shard.drop_group("started")


async def test_registration_receipt_timeout_retains_owner_for_shutdown_retry(monkeypatch):
    import relax.agentic.pipeline.runtime as runtime_module

    runtime, shard, started, _generated = runtime_fixture()
    runtime._start_shard_observers = lambda: None
    shard._eager_shutdown = True
    waiter_exited = asyncio.Event()

    async def blocked_receipt(_group_id):
        try:
            await asyncio.Event().wait()
        finally:
            waiter_exited.set()

    monkeypatch.setattr(runtime_module, "_REGISTRATION_RECEIPT_TIMEOUT_S", 0.01)
    monkeypatch.setattr(shard, "get_resource_release_receipt", blocked_receipt)
    with pytest.raises(RuntimeGroupError, match="receipt query timed out for blocked"):
        await asyncio.wait_for(runtime.prepare_group(GroupInput("blocked", [Sample(prompt="p") for _ in range(2)])), 1)
    await asyncio.wait_for(waiter_exited.wait(), 1)
    assert not started
    assert runtime.session_capacity_snapshot()["outstanding"] == 2
    assert "blocked" in runtime._reserved_group_shards
    assert "blocked" in runtime._shards[0].group_streams
    assert len(shard._resource_release_receipts["blocked"]) == 2
    # Normal shutdown retries drop/receipt replay without using the stuck getter.
    await asyncio.wait_for(runtime.shutdown(), 1)
    assert runtime.session_capacity_snapshot()["outstanding"] == 0
    assert not runtime._reserved_group_shards and not shard._resource_release_receipts


async def test_unmanaged_registration_failure_does_not_query_resource_receipts():
    runtime, shard, _started, _generated = runtime_fixture(capacity=None)
    runtime._start_shard_observers = lambda: None
    shard.start_group = AsyncMock(side_effect=RuntimeGroupError("ordinary startup failed"))
    shard.get_resource_release_receipt = AsyncMock(side_effect=AssertionError("unexpected receipt query"))
    with pytest.raises(RuntimeGroupError, match="ordinary startup failed"):
        await runtime.prepare_group(GroupInput("ordinary", [Sample(prompt="p") for _ in range(2)]))
    shard.get_resource_release_receipt.assert_not_called()
    assert runtime.session_capacity_snapshot() is None
    assert not runtime._reserved_group_shards and not runtime._shards[0].group_streams


@pytest.mark.parametrize("mode", ["fully_async", "partial_rollout"])
async def test_retained_eager_grant_during_pause_keeps_identity_and_capacity(mode):
    runtime, shard, started, generated = runtime_fixture(**{mode: True}, over_sampling_batch_size=3)
    try:
        await runtime.resume_generation(0)
        stream_a = await prepare(runtime, "a")
        a = shard._groups["a"]
        for session in a.sessions:
            await ready(shard, session)
        finish(shard, a.sessions[0])
        stream_b = await prepare(runtime, "b")
        b = shard._groups["b"]
        identities = tuple(session.session_id for session in b.sessions)
        assert identities[0] in started and identities[1] not in started
        await ready(shard, b.sessions[0])
        assert not b.interrupted  # An unstarted member is not interrupted credit.
        assert (identities[0], 0) in generated
        await runtime.pause_generation()
        finish(shard, a.sessions[1])
        await eventually(lambda: identities[1] in started)
        assert runtime.session_capacity_snapshot()["outstanding"] == 2
        await ready(shard, b.sessions[1])
        assert not any(sid == identities[1] for sid, _epoch in generated)
        assert b.interrupted  # All original first IRs now exist; both members are tool-idle.
        await runtime.resume_generation(1)
        assert tuple(session.session_id for session in b.sessions) == identities
        assert all(session.rollout_id == 1 for session in b.sessions)
        assert all((sid, 1) in generated for sid in identities)
        assert runtime.session_capacity_snapshot()["outstanding"] == 2
        result_a = await runtime.collect_group(stream_a)
        assert len(result_a.sessions) == 2
        collect_b = asyncio.create_task(runtime.collect_group(stream_b))
        finish(shard, b.sessions[0])
        await asyncio.sleep(0.02)
        assert not collect_b.done()
        finish(shard, b.sessions[1])
        assert len((await collect_b).sessions) == 2
        assert runtime.session_capacity_snapshot()["outstanding"] == 0
        assert runtime.session_capacity_snapshot()["global_peak"] == 2
        await runtime.pause_generation()
    finally:
        await runtime._stop_shard_observers()


async def test_prepare_discard_joins_registration_and_drops_ready_surplus():
    from relax.agentic.pipeline.prepare import PrepareDomain

    runtime, shard, _started, _generated = runtime_fixture(capacity=6)
    register_entered = asyncio.Event()
    register_finish = asyncio.Event()
    original_start = shard.start_group

    async def registered_group(*args, **kwargs):
        await original_start(*args, **kwargs)
        if args[1] == "registering":
            register_entered.set()
            await register_finish.wait()

    shard.start_group = registered_group

    def start(session, _input, **_kwargs):
        async def timer(_active):
            pass

        session.resources = SimpleNamespace(
            process=SimpleNamespace(set_timeout_active=timer),
            watcher_task=asyncio.create_task(asyncio.sleep(0)),
        )

    async def cleanup(session, _outcome, **_kwargs):
        session.group.resource_released_session_ids.add(session.session_id)
        session.result_cell.publish(None)
        session.resources = None
        session.group.pending_session_specs.pop(session.session_id, None)
        session.group.sessions.remove(session)
        shard._session_records.pop(session.session_id, None)
        shard._notify_state_change(session.group)

    shard._start_session = start
    shard._finish_session = cleanup
    preparation = PrepareDomain("train", None, runtime)
    try:
        await runtime.resume_generation(0)
        active = await prepare(runtime, "active")
        for group_id in ("ready", "registering"):
            samples = [Sample(index=i, group_index=0, prompt="p") for i in range(2)]
            preparation._start_prepare(GroupInput(group_id, samples))
        await register_entered.wait()
        await eventually(lambda: preparation.ready_group_count == 1)
        await runtime.pause_generation()
        discarding = asyncio.create_task(preparation.discard_step())
        await asyncio.sleep(0.02)
        assert not discarding.done()  # Registration must settle before its owner is released.
        register_finish.set()
        await asyncio.wait_for(discarding, 2)
        assert tuple(shard._groups) == ("active",)
        assert preparation.counts == (0, 0)
        assert runtime.session_capacity_snapshot()["outstanding"] == 2
        await runtime.drop_group(active)
        runtime._assert_epoch_drained()
        assert runtime.session_capacity_snapshot()["outstanding"] == 0
    finally:
        register_finish.set()
        await runtime._stop_shard_observers()


async def test_eager_shutdown_fences_late_registration_and_resume():
    runtime, shard, _started, _generated = runtime_fixture()
    dropping, release = asyncio.Event(), asyncio.Event()

    async def drop(group_id, **_kwargs):
        dropping.set()
        await release.wait()
        shard._groups.pop(group_id)

    shard._groups["resident"] = SimpleNamespace(group_id="resident", closing=False)
    shard.drop_group = drop
    shard._permit_cleanup_tasks = set()
    shard._generation_backend = SimpleNamespace(shutdown=AsyncMock())
    shutdown = asyncio.create_task(shard._shutdown())
    await asyncio.wait_for(dropping.wait(), 2)
    try:
        assert shard._groups["resident"].closing
        with pytest.raises(RuntimeGroupError, match="resource-accounted train scope"):
            await shard.start_group("train", "late", (), start_session_ids=())
        with pytest.raises(RuntimeGroupError, match="new paused rollout"):
            await shard.resume_generation(1)
        assert not shard._train_generation_open
    finally:
        release.set()
        await asyncio.wait_for(shutdown, 2)
    assert not shard._groups

    # A prepare already waiting for a resident permit must observe permanent
    # closing before reserving physical credits or submitting registration.
    runtime._resident_group_permits = asyncio.Semaphore(0)
    pending = asyncio.create_task(runtime.prepare_group(GroupInput("late", [Sample(prompt="p")])))
    await asyncio.sleep(0)
    assert not pending.done()
    runtime._eager_closing = True
    runtime._resident_group_permits.release()
    with pytest.raises(RuntimeGroupError, match="Runtime is closing"):
        await asyncio.wait_for(pending, 2)
    assert runtime.session_capacity_snapshot()["outstanding"] == 0


async def test_sync_eager_resume_requires_late_grants_and_resident_resources_to_drain():
    runtime, shard, started, generated = runtime_fixture(over_sampling_batch_size=3)
    try:
        await runtime.resume_generation(0)
        stream_a = await prepare(runtime, "a")
        a = shard._groups["a"]
        for session in a.sessions:
            await ready(shard, session)
        finish(shard, a.sessions[0])
        stream_b = await prepare(runtime, "b")
        b = shard._groups["b"]
        first, late = b.sessions
        await ready(shard, first)
        assert late.session_id not in started
        await runtime.pause_generation()
        finish(shard, a.sessions[1])
        await eventually(lambda: late.session_id in started)
        await ready(shard, late)
        assert not any(sid == late.session_id for sid, _epoch in generated)
        with pytest.raises(RuntimeGroupError, match="resident task resources"):
            await runtime.resume_generation(1)
        with pytest.raises(RuntimeGroupError, match="groups to be released"):
            await shard.resume_generation(1)
        assert runtime._rollout_epoch == shard._rollout_epoch == 0
        assert not shard._train_generation_open
        assert len((await runtime.collect_group(stream_a)).sessions) == 2

        async def cleanup(session, _outcome, **_kwargs):
            session.group.resource_released_session_ids.add(session.session_id)
            session.result_cell.publish(None)
            session.resources = None
            session.group.pending_session_specs.pop(session.session_id, None)
            session.group.sessions.remove(session)
            shard._session_records.pop(session.session_id, None)
            shard._notify_state_change(session.group)

        shard._finish_session = cleanup
        await runtime.drop_group(stream_b)
        runtime._assert_epoch_drained()
        await runtime.resume_generation(1)
        assert shard._train_generation_open and shard._rollout_epoch == 1
        assert runtime.session_capacity_snapshot()["outstanding"] == 0
        await runtime.pause_generation()
    finally:
        await runtime._stop_shard_observers()


async def test_late_grant_rpc_is_joined_by_drop_and_cannot_start_retired_members():
    runtime, shard, started, generated = runtime_fixture()
    entered, release = asyncio.Event(), asyncio.Event()
    original_start = shard.start_group_sessions

    async def delayed_start(group_id, ids):
        if group_id == "b" and len(shard._groups[group_id].pending_session_specs) == 1:
            entered.set()
            await release.wait()
        return await original_start(group_id, ids)

    async def cleanup(session, _outcome, **_kwargs):
        session.group.resource_released_session_ids.add(session.session_id)
        if not session.result_cell.terminal:
            session.result_cell.publish(None)
        session.resources = None
        session.group.pending_session_specs.pop(session.session_id, None)
        session.group.sessions.remove(session)
        shard._session_records.pop(session.session_id, None)
        shard._notify_state_change(session.group)

    try:
        await runtime.resume_generation(0)
        stream_a = await prepare(runtime, "a")
        a = shard._groups["a"]
        for session in a.sessions:
            await ready(shard, session)
        finish(shard, a.sessions[0])
        stream_b = await prepare(runtime, "b")
        b = shard._groups["b"]
        first, late = b.sessions
        await ready(shard, first)
        shard.start_group_sessions = delayed_start
        finish(shard, a.sessions[1])
        await asyncio.wait_for(entered.wait(), 2)
        assert late.session_id not in started
        assert runtime.session_capacity_snapshot()["outstanding"] == 2
        shard._finish_session = cleanup
        dropping = asyncio.create_task(runtime.drop_group(stream_b))
        await eventually(lambda: b.closing)
        await asyncio.sleep(0.02)
        assert not dropping.done()  # An in-flight grant still owns its exact debited identities.
        release.set()
        await asyncio.wait_for(dropping, 2)
        assert late.session_id not in started
        assert not any(sid == late.session_id for sid, _epoch in generated)
        assert runtime.session_capacity_snapshot()["outstanding"] == 0
        assert len((await runtime.collect_group(stream_a)).sessions) == 2
        runtime._assert_epoch_drained()
        await runtime.pause_generation()
        await runtime.resume_generation(1)
        assert "b" not in shard._groups
        await runtime.pause_generation()
    finally:
        release.set()
        await runtime._stop_shard_observers()


def install_resource_cleanup(shard):
    """Substitute only the external process's confirmed release boundary."""
    released = []

    async def cleanup(session, _outcome, **_kwargs):
        released.append(session.session_id)
        session.group.resource_released_session_ids.add(session.session_id)
        if not session.result_cell.terminal:
            session.result_cell.publish(None)
        session.resources = None
        session.group.pending_session_specs.pop(session.session_id, None)
        session.group.sessions.remove(session)
        shard._session_records.pop(session.session_id, None)
        shard._notify_state_change(session.group)

    shard._finish_session = cleanup
    return released


async def test_partial_capacity_preserves_live_irs_and_waits_for_protected_session_finalization():
    runtime, shard, _started, _generated = runtime_fixture(partial_rollout=True, partial_rollout_max_aborted_count=1)
    attempts, aborts, responses = {}, [], {}
    runners = set()

    @asynccontextmanager
    async def no_external_permit(*_args):
        yield

    async def generate(**kwargs):
        request_id = kwargs["request_id"]
        attempts[request_id] = kwargs
        response = asyncio.get_running_loop().create_future()
        responses[request_id] = response
        runners.add(asyncio.current_task())
        return await response

    async def abort(request_id):
        aborts.append(request_id)
        responses[request_id].set_result(BackendGenerateResult([42], [-0.2], "abort", {"weight_version": "v0"}, 0.1))

    shard._admission_lease = no_external_permit
    shard._sglang_request_permit = no_external_permit
    shard._generation_backend = SimpleNamespace(generate=generate, abort_request=abort, tokenizer=SimpleNamespace())
    # Keep the real dispatch, backend attempt, pause/resume, and IR lifecycle;
    # only API response serialization and external services are substituted.
    shard._dispatch_queued_irs_locked = type(shard)._dispatch_queued_irs_locked.__get__(shard)
    shard._terminal_response_locked = lambda *, session, ir, finish_type: {"tokens": list(ir.pending_token_delta)}
    try:
        await runtime.resume_generation(0)
        stream = await prepare(runtime, "partial")
        group = shard._groups["partial"]
        session, tool_idle = group.sessions
        irs = [
            InflightRequest(
                request_id=f"request-{index}",
                parent_state_hash="hash",
                rollout_id=0,
                kind=RequestKind.FRESH,
                abort_count=0,
                waiter=asyncio.get_running_loop().create_future(),
                wall_started_at=0.0,
                sampling_params={"max_new_tokens": 8},
                history_rollout_token_prefix=[7],
                pending_token_delta=[41],
                pending_logprob_delta=[-0.8],
            )
            for index in range(2)
        ]
        waiters = [ir.waiter for ir in irs]
        session.live_irs.update(irs)
        session.queued_irs.extend(irs)
        for member in group.sessions:
            await ready(shard, member)
        await eventually(lambda: len(attempts) == 2)
        await runtime.pause_generation()
        assert sorted(aborts) == ["request-0:0", "request-1:0"]
        assert session.live_irs == set(irs)
        assert set(session.queued_irs) == set(irs)
        assert session.protection_pending_until_resume and not session.protected_until_finalize
        assert all(ir.abort_count == 1 and ir.pending_status == "aborted" for ir in irs)
        assert all(ir.pending_token_delta == [41, 42] and ir.pending_logprob_delta == [-0.8, -0.2] for ir in irs)
        assert all(ir.pending_weight_version_delta == ["v0"] for ir in irs)
        assert [ir.waiter for ir in irs] == waiters and not any(waiter.done() for waiter in waiters)
        assert group.interrupted and runtime.session_capacity_snapshot()["outstanding"] == 2
        assert (session.session_id, False) in shard._timeout_events

        await runtime.resume_generation(1)
        await eventually(lambda: len(attempts) == 4)
        assert session.protected_until_finalize and not session.protection_pending_until_resume
        assert all(ir.kind is RequestKind.PROTECTED and ir.rollout_id == 1 for ir in irs)
        for index in range(2):
            resumed = attempts[f"request-{index}:1"]
            assert resumed["input_ids"] == [7, 41, 42]
            assert resumed["sampling_params"]["max_new_tokens"] == 6
        closing = asyncio.create_task(runtime.pause_generation())
        await asyncio.sleep(0.02)
        assert not closing.done() and len(aborts) == 2
        for index in range(2):
            responses[f"request-{index}:1"].set_result(
                BackendGenerateResult([43], [-0.1], "stop", {"weight_version": "v1"}, 0.1)
            )
        assert await asyncio.gather(*waiters) == [{"tokens": [41, 42, 43]}] * 2
        await asyncio.sleep(0.02)
        assert not closing.done()  # Protection covers the Session's remaining tools and cleanup too.
        finish(shard, session)
        group.sessions.remove(session)
        finish(shard, tool_idle)
        await asyncio.wait_for(closing, 2)
        assert len((await runtime.collect_group(stream)).sessions) == 2
        runtime._assert_epoch_drained()
        assert runtime.session_capacity_snapshot()["outstanding"] == 0
    finally:
        for runner in runners:
            if not runner.done():
                runner.cancel()
        await asyncio.gather(*runners, return_exceptions=True)
        await runtime._stop_shard_observers()


async def test_sync_prelaunch_crosses_epochs_without_generating_until_leased_and_finally_discards():
    runtime, shard, started, generated = runtime_fixture(capacity=4, agentic_prelaunch=True)
    released = install_resource_cleanup(shard)
    preparation = PrepareDomain("train", None, runtime)
    preparation.open_step(prelaunch_next_step=True)
    try:
        await runtime.resume_generation(0)
        active = await prepare(runtime, "active")
        for session in shard._groups["active"].sessions:
            await ready(shard, session)
        preparation._start_prepare(GroupInput("prepared", [Sample(prompt="p") for _ in range(2)]))
        await eventually(lambda: preparation.ready_group_count == 1)
        prepared = shard._groups["prepared"]
        identities = tuple(session.session_id for session in prepared.sessions)
        for session in prepared.sessions:
            await ready(shard, session)
        assert prepared.ownership is GroupOwnership.PREPARE
        assert not any(sid in identities for sid, _epoch in generated)
        await runtime.pause_generation()
        preparation.close_step()
        with pytest.raises(RuntimeGroupError, match="Runtime or failed Prepare owner"):
            await runtime.resume_generation(1)  # Prelaunch cannot excuse a Runtime-owned tail.
        for session in shard._groups["active"].sessions:
            finish(shard, session)
        assert len((await runtime.collect_group(active)).sessions) == 2
        runtime._assert_epoch_drained(allow_prepared=True)
        with pytest.raises(RuntimeGroupError, match="resident task resources"):
            runtime._assert_epoch_drained()
        await runtime.resume_generation(1)
        assert prepared.ownership is GroupOwnership.PREPARE
        assert tuple(session.session_id for session in prepared.sessions) == identities
        assert all(started.count(sid) == 1 for sid in identities)
        assert not any(sid in identities for sid, _epoch in generated)
        ((stream, _teacher_id, _response_keys),) = preparation.take_ready(1)
        await runtime.lease_group(stream, 1)
        assert all((sid, 1) in generated for sid in identities)
        for session in prepared.sessions:
            finish(shard, session)
        await runtime.collect_group(stream)

        # Last-step cleanup must also release a prepared Group which never had
        # first requests and was never leased to a model-weight epoch.
        preparation._start_prepare(GroupInput("last-surplus", [Sample(prompt="p") for _ in range(2)]))
        await eventually(lambda: preparation.ready_group_count == 1)
        surplus_ids = tuple(shard._groups["last-surplus"].result_cells)
        await runtime.pause_generation()
        await preparation.discard_step()
        assert set(released) == set(surplus_ids)
        assert all(released.count(sid) == 1 for sid in surplus_ids)
        assert not any(sid in surplus_ids for sid, _epoch in generated)
        assert preparation.counts == (0, 0)
        assert not shard._groups
        runtime._assert_epoch_drained()
        assert runtime.session_capacity_snapshot()["outstanding"] == 0
    finally:
        await runtime._stop_shard_observers()


@pytest.mark.parametrize("prelaunch", [False, True])
async def test_partial_prepare_waiting_for_capacity_has_one_owner_across_close_and_refill(prelaunch):
    runtime, shard, started, generated = runtime_fixture(partial_rollout=True, agentic_prelaunch=prelaunch)
    preparation = PrepareDomain("train", None, runtime)
    registrations = []
    original_start = shard.start_group

    async def register(*args, **kwargs):
        registrations.append(args[1])
        return await original_start(*args, **kwargs)

    shard.start_group = register
    try:
        await runtime.resume_generation(0)
        active = await prepare(runtime, "active")
        active_group = shard._groups["active"]
        for session in active_group.sessions:
            await ready(shard, session)
        preparation.open_step(prelaunch_next_step=prelaunch)
        preparation._start_prepare(GroupInput("waiting", [Sample(prompt="p") for _ in range(2)]))
        await asyncio.sleep(0.02)
        assert "waiting" not in shard._groups
        assert "waiting" not in runtime._reserved_group_shards
        assert runtime.session_capacity_snapshot()["outstanding"] == 2
        await runtime.pause_generation()
        preparation.close_step()
        await runtime.resume_generation(1)
        preparation.open_step(prelaunch_next_step=prelaunch)
        # Trigger requeued same-Group work before the old physical-credit wait
        # can settle. A prelaunch waiter instead keeps its original ownership.
        if not prelaunch:
            preparation.fill(demand=1)
        finish(shard, active_group.sessions[0])
        await eventually(lambda: preparation.ready_group_count == 1)
        ((stream, _teacher_id, _response_keys),) = preparation.take_ready(1)
        await runtime.lease_group(stream, 1)
        waiting_group = shard._groups["waiting"]
        first, late = waiting_group.sessions
        assert first.session_id in started and late.session_id not in started
        await ready(shard, first)
        assert (first.session_id, 1) in generated and not waiting_group.all_first_ir_arrived
        finish(shard, active_group.sessions[1])
        await eventually(lambda: late.session_id in started)
        await ready(shard, late)
        assert len((await runtime.collect_group(active)).sessions) == 2
        for session in waiting_group.sessions:
            finish(shard, session)
        assert len((await runtime.collect_group(stream)).sessions) == 2
        await eventually(lambda: not preparation._warming_prepare_tasks)
        assert registrations == ["active", "waiting"]
        assert len(started) == len(set(started)) == 4
        assert preparation.counts == (0, 0)
        assert runtime.session_capacity_snapshot()["global_peak"] == 2
        assert runtime.session_capacity_snapshot()["outstanding"] == 0
        runtime._assert_epoch_drained()
        await runtime.pause_generation()
    finally:
        await runtime._stop_shard_observers()


@pytest.mark.parametrize("cancel_lease", [False, True])
async def test_prelaunch_does_not_reclassify_submitted_lease_or_pending_cleanup_as_prepared(cancel_lease):
    runtime, shard, _started, _generated = runtime_fixture(agentic_prelaunch=True)
    install_resource_cleanup(shard)
    original_lease, original_cleanup = shard.lease_group, shard._finish_session
    lease_committed, finish_lease = asyncio.Event(), asyncio.Event()
    cleanup_entered, finish_cleanup = asyncio.Event(), asyncio.Event()

    async def uncertain_lease(group_id, rollout_id):
        await original_lease(group_id, rollout_id)
        lease_committed.set()
        await finish_lease.wait()
        raise RuntimeGroupError("lease reply lost")

    async def delayed_cleanup(*args, **kwargs):
        cleanup_entered.set()
        await finish_cleanup.wait()
        return await original_cleanup(*args, **kwargs)

    shard.lease_group = uncertain_lease
    shard._finish_session = delayed_cleanup
    try:
        await runtime.resume_generation(0)
        stream = await runtime.prepare_group(GroupInput("uncertain", [Sample(prompt="p") for _ in range(2)]))
        for session in shard._groups["uncertain"].sessions:
            await ready(shard, session)
        leasing = asyncio.create_task(runtime.lease_group(stream, 0))
        await asyncio.wait_for(lease_committed.wait(), 2)
        assert shard._groups["uncertain"].ownership is GroupOwnership.RUNTIME
        with pytest.raises(RuntimeGroupError, match="Runtime or failed Prepare owner"):
            runtime._assert_epoch_drained(allow_prepared=True)
        await runtime.pause_generation()
        if cancel_lease:
            leasing.cancel()
        else:
            finish_lease.set()
        await asyncio.wait_for(cleanup_entered.wait(), 2)
        assert not leasing.done()
        assert runtime.session_capacity_snapshot()["outstanding"] == 2
        with pytest.raises(RuntimeGroupError, match="Runtime or failed Prepare owner"):
            runtime._assert_epoch_drained(allow_prepared=True)
        with pytest.raises(RuntimeGroupError, match="Runtime or failed Prepare owner"):
            await runtime.resume_generation(1)
        assert runtime._rollout_epoch == shard._rollout_epoch == 0
        finish_cleanup.set()
        with pytest.raises(asyncio.CancelledError if cancel_lease else RuntimeGroupError):
            await asyncio.wait_for(leasing, 2)
        assert runtime.session_capacity_snapshot()["outstanding"] == 0
        runtime._assert_epoch_drained()
        await runtime.resume_generation(1)
        await runtime.pause_generation()
    finally:
        finish_lease.set()
        finish_cleanup.set()
        await runtime._stop_shard_observers()


async def test_prelaunch_controlled_member_exit_retains_sibling_credit_until_next_step_drop():
    runtime, shard, _started, generated = runtime_fixture(agentic_prelaunch=True)
    released = install_resource_cleanup(shard)
    # Exercise real controlled-failure/finish ownership and substitute only
    # the final external-process cleanup/publication boundary.
    shard._finish_session_once = shard._finish_session
    shard._finish_session = type(shard)._finish_session.__get__(shard)
    try:
        await runtime.resume_generation(0)
        stream = await runtime.prepare_group(GroupInput("failed-prepare", [Sample(prompt="p") for _ in range(2)]))
        group = shard._groups["failed-prepare"]
        first, sibling = group.sessions
        await shard._handle_controlled_agent_failure(first)
        await eventually(lambda: stream.preparation_task.done())
        assert stream.preparation_task.result() is False
        assert group.closing and group.eager_controlled_drop and group.first_error() is None
        assert group.ownership is GroupOwnership.PREPARE and not generated
        assert runtime.session_capacity_snapshot()["outstanding"] == 1
        assert released == [first.session_id]
        await runtime.pause_generation()
        runtime._assert_epoch_drained(allow_prepared=True)
        await runtime.resume_generation(1)
        assert runtime.session_capacity_snapshot()["outstanding"] == 1
        assert not generated and sibling.resources is not None
        await runtime.lease_group(stream, 1)
        assert await runtime.collect_group(stream) is None  # Whole Group is replaced, never converted into success.
        assert sorted(released) == sorted((first.session_id, sibling.session_id))
        assert runtime.session_capacity_snapshot()["outstanding"] == 0
        runtime._assert_epoch_drained()
        await runtime.pause_generation()
    finally:
        await runtime._stop_shard_observers()
