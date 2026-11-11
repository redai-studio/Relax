# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""CPU integration regression; does not start Ray or use GPUs.

Run with the candidate's normal CPU-test environment and candidate on PYTHONPATH:
  python -m pytest -q --asyncio-mode=auto <this-file>

Real: Rollout._async_run, ResidentGroup.interrupted, Pipeline step/close/transfer,
TransferDomain. Fake: external Prepare/Runtime work, sample serialization, TQ RPC,
and the actor's consume/clear boundary. This does not test optimizer numerics or
physical container cleanup. The deadline is only a hang guard; all assertions use
events and ordering, with no elapsed-time or sleep-duration assertions.
"""

import asyncio
import sys
from argparse import Namespace
from collections import Counter, defaultdict, deque
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock

import pytest

from relax.agentic.pipeline import GroupExport, SampleExport, SessionExport
from relax.agentic.pipeline.transfer import TransferDomain
from relax.agentic.session.service import (
    GroupOwnership,
    ResidentGroup,
    _SessionRecord,
    _SessionResultCell,
)
from relax.agentic.stage_recorder import StageRecorder
from relax.utils.logging_utils import get_logger
from relax.utils.types import Sample


# Same narrow, temporary optional-backend import seam as test_agentic_rollout.py.
_optional_module = "relax.distributed.ray.rollout"
_old_module = sys.modules.get(_optional_module)
if _old_module is None:
    _stub = ModuleType(_optional_module)
    _stub._log_rollout_data = MagicMock()
    _stub.get_local_rollout_manager = lambda: SimpleNamespace(training_weight_updating=False)
    sys.modules[_optional_module] = _stub
try:
    from relax.agentic import rollout as pipeline_module
finally:
    if _old_module is None:
        sys.modules.pop(_optional_module, None)

from relax.components import rollout as component_module  # noqa: E402


pytestmark = pytest.mark.asyncio
Pipeline = pipeline_module.AgenticResidentPipeline
StepContext = pipeline_module._StepContext
Rollout = component_module.Rollout.func_or_class
N = 8


def _export(name):
    return GroupExport(
        group_id=name,
        sessions=tuple(
            SessionExport(
                exports=(
                    SampleExport(
                        name=f"{name}/{i}",
                        sample=Sample(
                            index=(ord(name) - ord("A")) * N + i,
                            group_index=ord(name) - ord("A"),
                            session_id=f"{name}/{i}",
                            response="assistant result",
                            reward=1.0,
                            metadata={"test_group": name, "test_member": i},
                        ),
                    ),
                )
            )
            for i in range(N)
        ),
    )


def _tool_group(name):
    cells = {f"{name}/{i}": _SessionResultCell() for i in range(N)}
    group = ResidentGroup(
        rollout_mode="train",
        group_id=name,
        result_cells=cells,
        sessions=[],
        ownership=GroupOwnership.RUNTIME,
        eager_session_generation=True,
        first_ir_seen_session_ids=set(cells),
    )
    group.sessions.extend(
        _SessionRecord(group=group, session_id=key, session_sampling_params={}, result_cell=cell)
        for key, cell in cells.items()
    )
    # All first IRs arrived; every unfinished member is in a tool phase with no
    # live IR. This uses the real interrupted predicate, not synthetic credit.
    assert group.all_first_ir_arrived and group.interrupted
    return group


class _QuietProgress:
    def __init__(self, *, total, **_kwargs):
        self.total, self.n = total, 0

    def set_postfix_str(self, *_args, **_kwargs):
        pass

    def update(self, amount):
        self.n += amount

    def refresh(self):
        pass

    def close(self):
        pass


class _Prepare:
    ready_group_count = 0

    def __init__(self, harness):
        self.h = harness

    def collect(self):
        pass

    def fill(self, gap):
        self.h.filled[self.h.current_step].set()
        if gap <= 0:
            return  # Real Prepare.fill does not admit extra Groups without demand.
        if self.h.failed_first_tool and not self.h.replacement_started:
            self.h.replacement_started = True
            self.h.launch_group("F", self.h.already_ready)

    def take_ready(self, _gap):
        return []

    def close_step(self):
        pass

    async def discard_step(self):
        pass


class _Runtime:
    def __init__(self, harness):
        self.h = harness
        self.gate = asyncio.Event()
        self.tools = {}

    @property
    def interrupted_group_ids(self):
        return tuple(name for name, group in self.tools.items() if group.interrupted)

    async def resume_generation(self, step):
        h = self.h
        h.current_step = step
        h.trace.append(("resume", step))
        self.gate.set()
        if step == 0:
            h.pipeline._finalized_groups.append(_export("A"))
            h.launch_group("B", h.release_first_tool)
            if h.max_staleness == 0:
                h.launch_group("C", h.release_first_tool)  # Oversampled tail.
        elif step == 1 and h.max_staleness == 0:
            h.launch_group("D", h.already_ready)
        elif step == 1:
            h.pipeline._finalized_groups.append(_export("C"))
            h.launch_group("D", h.release_final_tools)
            h.launch_group("E", h.release_surplus)  # Already resident oversampled tail.
        elif step == 2:
            h.release_final_tools.set()
            h.release_surplus.set()
        else:
            raise AssertionError(f"unexpected generation step {step}")
        h.resumed[step].set()

    async def pause_generation(self):
        self.gate.clear()
        self.h.trace.append(("pause", self.h.current_step))
        self.h.paused[self.h.current_step].set()

    async def trim_memory(self):
        pass

    def session_capacity_snapshot(self):
        return None

    def _assert_epoch_drained(self, *, allow_prepared=False):
        assert not allow_prepared
        assert not self.tools
        assert all(task.done() for task in self.h.group_tasks)


class _Sink:
    def __init__(self, harness):
        self.h = harness
        self.rows = defaultdict(list)
        self.groups = defaultdict(list)
        self.active = set()
        self.eos = Counter()
        self.seen_rows = set()
        self.seen_groups = set()
        self.calls = []
        self.changed = asyncio.Event()

    async def write(self, *, args, batch_samples, partition_id, data_system_client, is_last, on_put_started=None):
        assert data_system_client is self and args.n_samples_per_prompt == N
        if on_put_started is not None:
            on_put_started()
        rollout_id = partition_id
        partition = f"train_{rollout_id}"
        assert partition in {"train_0", "train_1"}
        assert not self.eos[partition], "write after EOS"
        names = []
        for samples in batch_samples:
            assert len(samples) == N
            assert {s.metadata["test_member"] for s in samples} == set(range(N))
            assert len({s.metadata["test_group"] for s in samples}) == 1
            name = samples[0].metadata["test_group"]
            assert name not in self.seen_groups, "duplicate group"
            self.seen_groups.add(name)
            names.append(name)
            self.groups[partition].append(name)
            for sample in samples:
                assert sample.session_id not in self.seen_rows, "duplicate member"
                self.seen_rows.add(sample.session_id)
                self.rows[partition].append(sample.session_id)
            self.h.written[name].set()
        self.active.add(partition)
        self.calls.append((partition, tuple(names), is_last))
        self.h.trace.append(("write", partition, tuple(names), is_last))
        assert len(self.rows[partition]) <= 2 * N
        if is_last:
            assert len(self.rows[partition]) == 2 * N, "premature EOS"
            self.eos[partition] += 1
            self.h.trace.append(("eos", rollout_id))
            self.h.eos[rollout_id].set()

    async def async_get_partition_list(self):
        if "train_0" in self.active and self.eos["train_0"] and not self.h.cleared[0].is_set():
            self.h.staleness_wait.set()
        return sorted(self.active)

    async def async_check_production_completed(self, partition):
        return bool(self.eos[partition])

    def clear(self, step):
        partition = f"train_{step}"
        assert self.eos[partition] == 1
        assert len(self.rows[partition]) == 2 * N
        self.active.remove(partition)
        self.h.trace.append(("actor_clear", step))
        self.h.cleared[step].set()
        self.changed.set()


class _Harness:
    def __init__(self, monkeypatch, max_staleness, *, fail_first_tool=False):
        self.max_staleness = max_staleness
        self.fail_first_tool = fail_first_tool
        self.failed_first_tool = False
        self.replacement_started = False
        self.current_step = -1
        self.trace, self.generate_ids, self.open_ids, self.debt_at_return = [], [], [], []
        self.group_tasks, self.contexts, self.tasks, self.actor_steps = [], [], [], []
        self.filled = defaultdict(asyncio.Event)
        self.resumed = defaultdict(asyncio.Event)
        self.paused = defaultdict(asyncio.Event)
        self.written = defaultdict(asyncio.Event)
        self.eos = defaultdict(asyncio.Event)
        self.cleared = defaultdict(asyncio.Event)
        self.allow_clear = defaultdict(asyncio.Event)
        self.release_first_tool = asyncio.Event()
        self.release_final_tools = asyncio.Event()
        self.release_surplus = asyncio.Event()
        self.staleness_wait = asyncio.Event()
        self.rollout_returned = asyncio.Event()
        self.already_ready = asyncio.Event()
        self.already_ready.set()
        self.allow_clear[1].set()
        self.args = Namespace(
            fully_async=True,
            partial_rollout=False,
            max_staleness=max_staleness,
            use_dynamic_global_batch_size=False,
            use_dynamic_batch_size=True,
            colocate=False,
            rollout_batch_size=2,
            over_sampling_batch_size=3,
            global_batch_size=2 * N,
            num_iters_per_train_update=2,
            n_samples_per_prompt=N,
            num_rollout=2,
            eval_interval=None,
            skip_eval_before_train=True,
            offload_rollout=False,
            use_health_check=False,
            save=None,
            save_interval=None,
            agentic_prelaunch=False,
            agentic_session_capacity_per_node=2 * N,
        )
        self.sink = _Sink(self)
        self.manager = SimpleNamespace(training_weight_updating=False)
        monkeypatch.setattr(pipeline_module, "get_local_rollout_manager", lambda: self.manager)
        self.runtime = _Runtime(self)
        self.pipeline = object.__new__(Pipeline)
        self.pipeline.args = self.args
        self.pipeline.prepare_domain = _Prepare(self)
        self.pipeline.runtime_domain = self.runtime
        self.pipeline.reward_domain = SimpleNamespace(scored_session_count=0)
        self.pipeline.transfer_domain = TransferDomain(args=self.args)
        self.pipeline._data_system_client = self.sink
        self.pipeline._opd_manager = None
        self.pipeline._changed = asyncio.Event()
        self.pipeline._active_group_tasks = []
        self.pipeline._group_observations = {}
        self.pipeline._stage_recorder = StageRecorder(self.args)
        self.pipeline._finalized_groups = deque()
        self.pipeline._active_step = None
        real_open = self.pipeline.transfer_domain.open_partition

        def record_open(partition, *args, **kwargs):
            self.open_ids.append(partition)
            return real_open(partition, *args, **kwargs)

        monkeypatch.setattr(self.pipeline.transfer_domain, "open_partition", record_open)
        monkeypatch.setattr(pipeline_module, "_transfer_batch_to_data_system", self.sink.write)
        monkeypatch.setattr(pipeline_module, "tqdm", _QuietProgress)
        self.rollout = object.__new__(Rollout)
        self.rollout._logger_instance = get_logger(__name__)
        self.rollout.config = self.args
        self.rollout.step = 0
        self.rollout.status = "running"
        self.rollout.eval_handler = None
        self.rollout._stop_event = asyncio.Event()
        self.rollout._peer_barrier = None
        self.rollout.data_system_client = self.sink
        self.rollout.rollout_manager = SimpleNamespace(generate=SimpleNamespace(remote=self.generate))
        self.rollout.healthy = SimpleNamespace(
            update_heartbeat=SimpleNamespace(remote=lambda *_args: None),
            report_error=SimpleNamespace(remote=lambda *_args: None),
        )

    def launch_group(self, name, ready):
        assert name not in self.runtime.tools
        self.runtime.tools[name] = _tool_group(name)

        async def finish_after_tool():
            try:
                await ready.wait()
                # A tail cannot finish its next model request while generation
                # is paused. This makes early close a real progress dependency.
                await self.runtime.gate.wait()
                if name == "B" and self.fail_first_tool:
                    self.failed_first_tool = True
                    self.trace.append(("tool_failure", name, self.current_step))
                    return None  # Existing failure/filter boundary discards the complete Group.
                self.trace.append(("tool_finish", name, self.current_step))
                return _export(name)
            finally:
                self.runtime.tools.pop(name)

        task = asyncio.create_task(finish_after_tool(), name=f"test-tool-{name}")
        task.add_done_callback(lambda _task: self.pipeline._changed.set())
        self.group_tasks.append(task)
        self.pipeline._active_group_tasks.append(task)

    async def generate(self, *, rollout_id):
        self.generate_ids.append(rollout_id)
        context = StepContext(rollout_id, final_backfill=rollout_id >= self.args.num_rollout)
        self.contexts.append(context)
        self.pipeline._active_step = context
        try:
            result = await self.pipeline._run_rollout_step(context)
            self.debt_at_return.append((rollout_id, self.pipeline.transfer_domain.total_debt))
            return result
        finally:
            if context.progress is not None:
                context.progress.close()
            self.pipeline._active_step = None

    async def actor(self):
        for step in range(2):
            await self.eos[step].wait()
            assert len(self.sink.rows[f"train_{step}"]) == 2 * N
            await self.allow_clear[step].wait()
            if step == 1:
                # Keep the final partition until Rollout's real final-EOS query;
                # the test need not invent deleted-partition query semantics.
                await self.rollout_returned.wait()
            self.actor_steps.append(step)
            self.sink.clear(step)

    def start(self):
        async def drive():
            try:
                await self.rollout._async_run()
            finally:
                self.rollout_returned.set()

        self.tasks = [asyncio.create_task(drive()), asyncio.create_task(self.actor())]

    async def close(self):
        self.rollout._stop_event.set()
        pending = self.tasks + self.group_tasks + [t for c in self.contexts for t in c.tq_put_tasks]
        for task in pending:
            if not task.done():
                task.cancel()
        await asyncio.gather(*pending, return_exceptions=True)
        await self.pipeline._stage_recorder.aclose()
        # The shell never attached a real TQ client; prevent Base.__del__ from
        # attempting global client detachment when this object is collected.
        self.rollout.data_system_client = None

    def assert_complete(self):
        assert self.open_ids == [0, 1]
        assert self.actor_steps == [0, 1]
        first_tail = "F" if self.fail_first_tool else "B"
        assert dict(self.sink.groups) == {"train_0": ["A", first_tail], "train_1": ["C", "D"]}
        assert self.sink.eos == Counter({"train_0": 1, "train_1": 1})
        assert len(self.sink.seen_rows) == 4 * N
        assert self.sink.seen_groups == {"A", first_tail, "C", "D"}
        assert not self.sink.active
        assert self.pipeline.transfer_domain.total_debt == 0
        assert not self.pipeline._finalized_groups
        assert not self.pipeline._active_group_tasks
        self.runtime._assert_epoch_drained()


async def _first_event(events):
    tasks = {asyncio.create_task(event.wait()): name for name, event in events.items()}
    try:
        done, _pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        return {tasks[task] for task in done}
    finally:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


async def test_zero_staleness_tool_wait_fill_eos_clear_then_second_round(monkeypatch):
    h = _Harness(monkeypatch, max_staleness=0)

    async def scenario():
        h.start()
        await h.written["A"].wait()
        observed = await _first_event({"still_open": h.filled[0], "premature_pause": h.paused[0]})
        assert observed == {"still_open"}
        assert h.runtime.tools["B"].interrupted
        assert h.runtime.gate.is_set() and not h.eos[0].is_set()
        h.release_first_tool.set()
        await h.eos[0].wait()
        await h.paused[0].wait()
        await h.staleness_wait.wait()
        assert h.generate_ids == [0] and not h.cleared[0].is_set()
        h.allow_clear[0].set()
        await asyncio.gather(*h.tasks)
        h.assert_complete()
        assert h.generate_ids == [0, 1]
        assert h.debt_at_return == [(0, 0), (1, 0)]
        assert h.trace.index(("eos", 0)) < h.trace.index(("actor_clear", 0)) < h.trace.index(("resume", 1))
        assert sum(item[:2] == ("tool_finish", "C") for item in h.trace) == 1

    try:
        await asyncio.wait_for(scenario(), timeout=5)
    finally:
        await h.close()


@pytest.mark.parametrize("fail_first_tool", [False, True])
async def test_positive_staleness_final_backfill_fills_fifo_once_without_third_partition(monkeypatch, fail_first_tool):
    h = _Harness(monkeypatch, max_staleness=1, fail_first_tool=fail_first_tool)

    async def scenario():
        h.start()
        await h.written["A"].wait()
        await h.filled[0].wait()
        assert h.runtime.tools["B"].interrupted
        assert not h.paused[0].is_set()  # Tool-idle members alone cannot trigger a step boundary.
        h.release_first_tool.set()
        await h.eos[0].wait()
        await h.written["C"].wait()
        await h.filled[1].wait()
        carried = h.runtime.tools["D"]
        identities = tuple(carried.result_cells)
        assert h.generate_ids == [0, 1]
        assert not h.paused[1].is_set() and h.runtime.gate.is_set()
        assert h.pipeline.transfer_domain.total_debt == 1

        # Actor can update now because train_0 reached EOS. The real manager
        # flag authorizes carrying train_1's unfinished Group; pause the outer
        # component until the simulated update ends, as its HTTP handshake does.
        h.manager.training_weight_updating = True
        h.rollout.status = "paused"
        h.pipeline._changed.set()
        await h.paused[1].wait()
        assert h.runtime.tools["D"] is carried
        assert tuple(carried.result_cells) == identities and not h.runtime.gate.is_set()
        assert not h.eos[1].is_set()
        h.allow_clear[0].set()
        h.manager.training_weight_updating = False
        h.rollout.status = "running"
        await asyncio.gather(*h.tasks)
        h.assert_complete()
        assert h.generate_ids == [0, 1, 2]
        assert h.debt_at_return == [(0, 0), (1, 1), (2, 0)]
        assert h.sink.calls[0] == ("train_0", ("A",), False)
        assert h.trace.index(("eos", 0)) < h.trace.index(("resume", 1))
        assert h.trace.index(("resume", 2)) < h.trace.index(("eos", 1))
        assert ("tool_finish", "E", 2) in h.trace  # Complete surplus is discarded.
        assert "E" not in h.sink.seen_groups
        assert [c.final_backfill for c in h.contexts] == [False, False, True]
        assert ("tool_finish", "D", 2) in h.trace
        if fail_first_tool:
            assert h.replacement_started and "B" not in h.sink.seen_groups
            assert h.trace.index(("tool_failure", "B", 0)) < h.trace.index(("tool_finish", "F", 0))
            assert h.trace.index(("tool_finish", "F", 0)) < h.trace.index(("eos", 0))

    try:
        await asyncio.wait_for(scenario(), timeout=5)
    finally:
        await h.close()
