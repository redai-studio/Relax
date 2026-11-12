# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Exercise dynamic close with real Pipeline, Group, Transfer and credit state.

Only process/generation/reward completion and the TQ transport are controlled
here. Runtime/Shard admission and Prepare's cross-epoch races are covered by
test_eager_session_generation; these tests target their coordinator boundary.
"""

from __future__ import annotations

import asyncio
import sys
from argparse import Namespace
from collections import defaultdict, deque
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock

import pytest

from relax.agentic.pipeline import GroupExport, SampleExport, SessionExport
from relax.agentic.pipeline.session_capacity import SessionCapacity
from relax.agentic.pipeline.transfer import TransferDomain
from relax.agentic.stage_recorder import StageRecorder
from relax.utils.types import Sample


# Keep the CPU suite independent of the optional serving backend. Restore the
# module table immediately, as in test_agentic_rollout.py.
_module_name = "relax.distributed.ray.rollout"
_existing_module = sys.modules.get(_module_name)
if _existing_module is None:
    _stub = ModuleType(_module_name)
    _stub._log_rollout_data = MagicMock()
    _stub.get_local_rollout_manager = lambda: SimpleNamespace(training_weight_updating=False)
    sys.modules[_module_name] = _stub
try:
    from relax.agentic import rollout as pipeline_module
finally:
    if _existing_module is None:
        sys.modules.pop(_module_name, None)


N = 2


def _export(name):
    return GroupExport(
        group_id=name,
        sessions=tuple(
            SessionExport(
                exports=(
                    SampleExport(
                        name=None,
                        sample=Sample(
                            session_id=f"{name}/{member}",
                            response="complete response",
                            reward=1.0,
                            metadata={"test_group": name},
                        ),
                    ),
                )
            )
            for member in range(N)
        ),
    )


def _names(groups):
    return [group[0].metadata["test_group"] for group in groups]


class _Progress:
    def __init__(self, *, total, **_kwargs):
        self.total, self.n = total, 0

    def set_postfix_str(self, *_args, **_kwargs):
        pass

    def update(self, delta):
        self.n += delta

    def refresh(self):
        pass

    def close(self):
        pass


class _Prepare:
    ready_group_count = 0

    def __init__(self):
        self.demands = []
        self.closed = self.discarded = 0
        self.on_fill = lambda _demand: None

    def collect(self):
        pass

    def take_ready(self, _limit):
        return ()

    def fill(self, demand):
        self.demands.append(demand)
        self.on_fill(demand)

    def close_step(self):
        self.closed += 1

    async def discard_step(self):
        self.discarded += 1

    async def shutdown(self):
        await self.discard_step()


class _Reward:
    group_rm = False
    custom_advantage_func = None
    last_filter_drop_reason = "test whole-group rejection"

    def __init__(self):
        self.scored_session_count = 0
        self.discarded = []
        self.rejected = set()

    async def score_sessions(self, stream):
        await stream.generated.wait()
        await stream.reward_ready.wait()
        self.scored_session_count += N

    def finalize_group(self, group):
        return None if group.group_id in self.rejected else group

    def discard_group_progress(self, stream):
        self.discarded.append(stream.group_id)


class _Runtime:
    def __init__(self):
        self.capacity = SessionCapacity({"host"}, 4)
        self.reservations = {}
        self.dropped = []
        self.paused = asyncio.Event()
        self.before_pause = None
        self.shutdown_called = False

    def own_resources(self, stream):
        reservation = self.capacity.reserve(stream.group_id, "host", N)
        self.capacity.bind(reservation, tuple(f"{stream.group_id}/{i}" for i in range(N)))
        reservation.submitted = True
        self.reservations[stream.group_id] = reservation

    def release_resources(self, stream):
        reservation = self.reservations.pop(stream.group_id, None)
        if reservation is not None:
            self.capacity.confirm(reservation, reservation.session_order)
            self.capacity.retire(reservation)

    async def collect_group(self, stream):
        await stream.generated.wait()
        self.release_resources(stream)
        return None if stream.failed else _export(stream.group_id)

    async def drop_group(self, stream):
        self.dropped.append(stream.group_id)
        self.release_resources(stream)

    async def resume_generation(self, _rollout_id):
        self.paused.clear()

    async def pause_generation(self):
        if self.before_pause is not None:
            await self.before_pause()
        self.paused.set()

    async def trim_memory(self):
        pass

    async def shutdown(self):
        assert not self.reservations
        assert self.capacity.snapshot()["outstanding"] == 0
        self.shutdown_called = True

    def session_capacity_snapshot(self):
        return self.capacity.snapshot()

    def _assert_epoch_drained(self, **_kwargs):
        raise AssertionError("partial dynamic close must retain its active Runtime tail")


class _Harness:
    def __init__(self, monkeypatch, *, dp=2, prelaunch=False):
        self.args = Namespace(
            fully_async=False,
            partial_rollout=True,
            agentic_prelaunch=prelaunch,
            agentic_session_capacity_per_node=4,
            use_dynamic_global_batch_size=True,
            rollout_batch_size=4,
            over_sampling_batch_size=8,
            n_samples_per_prompt=N,
            num_rollout=3,
            resource={"actor": (1, dp)},
        )
        self.runtime, self.prepare, self.reward = _Runtime(), _Prepare(), _Reward()
        self.started = defaultdict(asyncio.Event)
        self.durable = defaultdict(asyncio.Event)
        self.blocked = {}
        self.writes = []
        self.tasks = []
        self.contexts = []
        pipeline = self.pipeline = object.__new__(pipeline_module.AgenticResidentPipeline)
        pipeline.args = self.args
        pipeline.prepare_domain, pipeline.runtime_domain = self.prepare, self.runtime
        pipeline.reward_domain = self.reward
        pipeline.transfer_domain = TransferDomain(args=self.args)
        pipeline._data_system_client = None
        pipeline._opd_manager = None
        pipeline._changed = asyncio.Event()
        pipeline._active_group_tasks = []
        pipeline._group_observations = {}
        pipeline._finalized_groups = deque()
        pipeline._stage_recorder = StageRecorder(self.args)
        pipeline._active_step = None
        monkeypatch.setattr(pipeline_module, "_transfer_batch_to_data_system", self.write)
        monkeypatch.setattr(pipeline_module, "tqdm", _Progress)

    def task(self, coro):
        task = asyncio.create_task(coro)
        self.tasks.append(task)
        return task

    def launch(self, name, *, generated=True, rewarded=True, failed=False, owns_resources=False):
        stream = SimpleNamespace(
            group_id=name,
            generated=asyncio.Event(),
            reward_ready=asyncio.Event(),
            failed=failed,
        )
        if generated:
            stream.generated.set()
        if rewarded:
            stream.reward_ready.set()
        if owns_resources:
            self.runtime.own_resources(stream)
        task = self.task(pipeline_module._run_group(self.runtime, self.reward, stream))
        task.add_done_callback(lambda _task: self.pipeline._changed.set())
        self.pipeline._active_group_tasks.append(task)
        return stream, task

    async def write(self, *, batch_samples, partition_id, is_last, on_put_started, **_kwargs):
        names = tuple(_names(batch_samples))
        key = partition_id, names
        on_put_started()
        self.started[key].set()
        if key in self.blocked:
            await self.blocked[key].wait()
        rows = tuple(sample.session_id for group in batch_samples for sample in group)
        assert len(rows) == len(set(rows)) == N * len(names)
        assert not set(rows).intersection(row for _, _, _, prior in self.writes for row in prior)
        self.writes.append((partition_id, names, is_last, rows))
        self.durable[key].set()

    def context(self, step):
        context = pipeline_module._StepContext(step, final_backfill=False)
        self.contexts.append(context)
        self.pipeline.transfer_domain.open_partition(step, 4, accepts_surplus=True)
        return context

    async def close(self):
        self.tasks.extend(task for context in self.contexts for task in context.tq_put_tasks)
        for task in self.tasks:
            task.cancel()
        await asyncio.gather(*self.tasks, return_exceptions=True)
        await self.pipeline.shutdown()


@pytest.mark.parametrize("prelaunch", [False, True])
@pytest.mark.parametrize("dp", [2, 4])
async def test_dynamic_close_retains_reward_and_fifo_remainder_across_steps(monkeypatch, prelaunch, dp):
    h = _Harness(monkeypatch, dp=dp, prelaunch=prelaunch)
    p = h.pipeline
    try:
        context = h.context(0)
        base = tuple(f"base-{i}" for i in range(4))
        completed_on_pause = tuple(f"pause-{i}" for i in range(dp))
        surplus = ("carry", *completed_on_pause[:-1])
        h.blocked[0, base] = asyncio.Event()
        h.blocked[0, surplus] = asyncio.Event()
        p._finalized_groups.extend(_export(name) for name in (*base, "carry"))
        arriving = [h.launch(name, rewarded=False) for name in completed_on_pause]
        reward_stream, reward_task = h.launch("pending-reward", rewarded=False)
        live_stream, live_task = h.launch("live-process", generated=False, owns_resources=True)

        async def finish_at_pause():
            assert h.durable[0, base].is_set(), "generation paused before base TQ durability"
            for stream, _task in arriving:
                stream.reward_ready.set()
            await asyncio.gather(*(task for _stream, task in arriving))

        h.runtime.before_pause = finish_at_pause
        p._collect_step_progress(context)
        closing = h.task(p._close_rollout_step(context))
        await asyncio.wait_for(h.started[0, base].wait(), 2)
        assert not closing.done() and not h.runtime.paused.is_set()
        h.blocked[0, base].set()
        await asyncio.wait_for(h.started[0, surplus].wait(), 2)
        assert h.runtime.paused.is_set() and not closing.done()
        assert not reward_task.done() and not live_task.done()
        assert h.runtime.capacity.snapshot()["outstanding"] == N
        h.blocked[0, surplus].set()
        output = await asyncio.wait_for(closing, 2)
        assert _names(output.samples) == [*base, *surplus]
        assert len(output.samples) % dp == 0
        assert output.metrics["session_capacity/outstanding"] == N
        assert [g.group_id for g in p._finalized_groups] == [completed_on_pause[-1]]
        assert p._active_group_tasks == [reward_task, live_task]
        assert not context.tq_put_tasks and p.transfer_domain.total_debt == 0
        assert h.prepare.discarded == 0 and not h.runtime.dropped

        # Reward can complete while generation is paused. It joins the next
        # partition after the older unaligned FIFO remainder, before fresh work.
        reward_stream.reward_ready.set()
        await asyncio.wait_for(reward_task, 2)
        next_context = h.context(1)
        p._collect_step_progress(next_context)
        assert _names(next_context.output_groups) == [completed_on_pause[-1], "pending-reward"]
        fresh = [h.launch(f"fresh-{i}")[1] for i in range(2)]
        await asyncio.wait_for(asyncio.gather(*fresh), 2)
        p._collect_step_progress(next_context)
        h.runtime.before_pause = None
        next_output = await asyncio.wait_for(p._close_rollout_step(next_context), 2)
        assert _names(next_output.samples) == [completed_on_pause[-1], "pending-reward", "fresh-0", "fresh-1"]
        assert not next_context.tq_put_tasks and not p._finalized_groups
        assert p._active_group_tasks == [live_task]
        assert h.runtime.capacity.snapshot()["outstanding"] == N
        assert [write[:3] for write in h.writes] == [
            (0, base, False),
            (0, surplus, False),
            (1, (completed_on_pause[-1], "pending-reward", "fresh-0", "fresh-1"), False),
        ]
        # Non-streaming dynamic partitions intentionally have no EOS marker.
        # All accepted groups still cross TQ exactly once, with every member.
        assert sum(len(rows) for _, _, _, rows in h.writes) == N * (8 + dp)
        live_stream.generated.set()
        await asyncio.wait_for(live_task, 2)
        assert h.runtime.capacity.snapshot()["outstanding"] == 0
        assert h.runtime.capacity.snapshot()["confirmed_releases"] == N
    finally:
        await h.close()


@pytest.mark.parametrize("prelaunch", [False, True])
async def test_final_dynamic_step_seals_before_prepare_discard_and_shutdown_releases_tail(monkeypatch, prelaunch):
    h = _Harness(monkeypatch, prelaunch=prelaunch)
    h.args.num_rollout = 1
    try:
        context = h.context(0)
        base = tuple(f"base-{i}" for i in range(4))
        surplus = ("surplus-0", "surplus-1")
        h.blocked[0, surplus] = asyncio.Event()
        h.pipeline._finalized_groups.extend(_export(name) for name in (*base, *surplus, "unaligned"))
        _, reward_task = h.launch("pending-reward", rewarded=False)
        _, live_task = h.launch("live-process", generated=False, owns_resources=True)
        h.pipeline._collect_step_progress(context)
        closing = h.task(h.pipeline._close_rollout_step(context))
        await asyncio.wait_for(h.started[0, surplus].wait(), 2)
        assert h.prepare.discarded == 0 and not closing.done()
        assert not reward_task.done() and not live_task.done()
        h.blocked[0, surplus].set()
        output = await asyncio.wait_for(closing, 2)
        assert _names(output.samples) == [*base, *surplus]
        assert h.prepare.discarded == 1
        assert output.metrics["session_capacity/outstanding"] == N
        assert not reward_task.done() and not live_task.done()
        assert [g.group_id for g in h.pipeline._finalized_groups] == ["unaligned"]

        await asyncio.wait_for(h.pipeline.shutdown(), 2)
        assert reward_task.cancelled() and live_task.cancelled()
        assert set(h.runtime.dropped) == {"pending-reward", "live-process"}
        assert h.runtime.shutdown_called
        assert h.runtime.capacity.snapshot()["outstanding"] == 0
        assert h.runtime.capacity.snapshot()["confirmed_releases"] == N
        assert not h.pipeline._active_group_tasks and not h.pipeline._finalized_groups
        assert [write[:3] for write in h.writes] == [(0, base, False), (0, surplus, False)]
    finally:
        await h.close()


@pytest.mark.parametrize("prelaunch", [False, True])
@pytest.mark.parametrize("failure", ["member_failed", "filtered"])
async def test_dynamic_step_replaces_whole_dropped_group_without_tq_duplicates(monkeypatch, prelaunch, failure):
    h = _Harness(monkeypatch, prelaunch=prelaunch)
    h.args.over_sampling_batch_size = 5
    try:
        if failure == "filtered":
            h.reward.rejected.add("failed")
        initial = [h.launch(f"accepted-{i}")[1] for i in range(3)]
        initial.append(h.launch("failed", failed=failure == "member_failed")[1])
        await asyncio.wait_for(asyncio.gather(*initial), 2)
        replacement_started = False

        def replenish(demand):
            nonlocal replacement_started
            if demand > 0 and not replacement_started:
                replacement_started = True
                h.launch("replacement")

        h.prepare.on_fill = replenish
        context = pipeline_module._StepContext(0, final_backfill=False)
        output = await asyncio.wait_for(h.pipeline._run_rollout_step(context), 2)
        assert replacement_started
        assert h.prepare.demands[0] == 2  # One failed-group debt plus one oversample slot.
        assert _names(output.samples) == ["accepted-0", "accepted-1", "accepted-2", "replacement"]
        assert h.reward.discarded == ["failed"]
        assert h.runtime.dropped == (["failed"] if failure == "member_failed" else [])
        assert [write[:3] for write in h.writes] == [
            (0, ("accepted-0", "accepted-1", "accepted-2", "replacement"), False)
        ]
        assert h.pipeline.transfer_domain.total_debt == 0
        assert not h.pipeline._active_group_tasks and not h.pipeline._finalized_groups
        assert not context.tq_put_tasks and h.runtime.capacity.snapshot()["outstanding"] == 0
    finally:
        await h.close()
