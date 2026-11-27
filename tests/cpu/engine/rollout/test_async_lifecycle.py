# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Exercise real rollout scheduling; only inference, reward and storage are
external fixtures."""

import asyncio
from argparse import Namespace
from types import MethodType, SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from relax.engine.rollout import sglang_rollout as rollout
from relax.utils.types import Sample


@pytest.fixture
def harness(monkeypatch):
    args = Namespace(
        rollout_global_dataset=True,
        dynamic_sampling_filter_path=None,
        fully_async=True,
        max_staleness=0,
        num_rollout=2,
        rollout_batch_size=2,
        over_sampling_batch_size=2,
        n_samples_per_prompt=1,
        global_batch_size=2,
        num_iters_per_train_update=2,
        partial_rollout=False,
        use_dynamic_global_batch_size=False,
        debug_rollout_only=False,
        sglang_enable_deterministic_inference=False,
        use_rollout_indexer_replay=False,
        lora_rank=0,
        lora_adapter_mode=False,
        group_rm=True,
        use_dynamic_batch_size=False,
    )
    state_class = rollout.GenerateState
    state = SimpleNamespace(
        args=args,
        sampling_params={},
        last_step_current_deficit=0,
        remaining_batch_size=0,
        prefetched_samples_ref=None,
        pendings=set(),
        protected_pendings=set(),
        tokenizer=None,
        reset=Mock(),
        aborted=False,
        abort_event=asyncio.Event(),
        partial_genrm=None,
    )
    state.submit_generate_tasks = MethodType(state_class.submit_generate_tasks, state)
    groups = [[Sample(index=i, group_index=i, response="ok", status=Sample.Status.COMPLETED)] for i in range(2)]
    source = SimpleNamespace(get_samples=SimpleNamespace(remote=Mock(return_value=groups)))
    tracked = []
    writes = []
    write_started = asyncio.Event()
    log = Mock()

    async def generate(args, sample, sampling_params, evaluation=False):
        tracked.append(asyncio.current_task())
        return sample

    async def score(args, group, evaluation=False):
        return [1.0 for _ in group]

    async def put(args, groups, batch_size, rollout_id, client, **kwargs):
        writes.append(asyncio.current_task())
        current_batch.extend(s for group in groups for s in group)
        write_started.set()

    current_batch = []
    monkeypatch.setattr(rollout, "GenerateState", lambda args: state)
    monkeypatch.setattr(rollout, "ray", SimpleNamespace(get=lambda ref: ref))
    monkeypatch.setattr(rollout, "generate_and_rm", generate)
    monkeypatch.setattr(rollout, "batched_async_rm", score)
    monkeypatch.setattr(rollout, "references_enabled", lambda: False)
    monkeypatch.setattr(rollout, "start_sglang_profile", AsyncMock())
    monkeypatch.setattr(rollout, "stop_sglang_profile", AsyncMock())
    monkeypatch.setattr(rollout, "abort", AsyncMock(return_value=([], [])))
    monkeypatch.setattr(rollout, "transfer_batch_to_data_system", put)
    monkeypatch.setattr(rollout, "CURRENT_ROLLOUT_BATCH", current_batch)
    monkeypatch.setattr(rollout, "save_debug_rollout_data", Mock())
    monkeypatch.setattr(rollout, "_log_rollout_data", log)
    monkeypatch.setattr(rollout, "tqdm", Mock())
    # The unrelated process-global Timer retains failed windows between cases.
    monkeypatch.setattr(rollout, "Timer", lambda: SimpleNamespace(start=Mock(), end=lambda name: 10.0))
    return SimpleNamespace(
        args=args,
        state=state,
        groups=groups,
        source=source,
        tracked=tracked,
        writes=writes,
        write_started=write_started,
        log=log,
        current_batch=current_batch,
    )


@pytest.mark.parametrize("evaluation", [False, True])
@pytest.mark.parametrize("cleanup_fails", [False, True])
async def test_group_failure_drains_sibling_requests(harness, monkeypatch, evaluation, cleanup_fails):
    entered, released, closed = asyncio.Event(), asyncio.Event(), asyncio.Event()
    primary = ValueError("inference failed")
    children = []

    async def generate(args, sample, sampling_params, evaluation=False):
        children.append(asyncio.current_task())
        if sample.index == 0:
            await entered.wait()
            raise primary
        try:
            entered.set()
            await released.wait()
            return sample
        finally:
            closed.set()
            if cleanup_fails:
                raise RuntimeError("private cleanup payload")

    monkeypatch.setattr(rollout, "generate_and_rm", generate)
    try:
        with pytest.raises(ValueError) as caught:
            await rollout.generate_and_rm_group(harness.args, [g[0] for g in harness.groups], {}, evaluation)
        assert caught.value is primary
        assert closed.is_set(), "failed group returned while a sibling inference request remained live"
        assert all(task.done() for task in children)
        if cleanup_fails:
            assert primary.__notes__ == ["rollout task cleanup also failed: RuntimeError"]
    finally:
        released.set()
        await asyncio.gather(*children, return_exceptions=True)


@pytest.mark.parametrize("recovery", [False, True])
async def test_transfer_failure_interrupts_blocked_generation(harness, monkeypatch, recovery):
    transfer_entered, release_generation, generation_closed = asyncio.Event(), asyncio.Event(), asyncio.Event()
    primary = OSError("production acknowledgement failed")
    harness.args.inference_recovery = recovery

    async def generate(args, sample, sampling_params, evaluation=False):
        harness.tracked.append(asyncio.current_task())
        if sample.index:
            try:
                await release_generation.wait()
            finally:
                generation_closed.set()
        return sample

    async def put(*args, **kwargs):
        harness.writes.append(asyncio.current_task())
        transfer_entered.set()
        raise primary

    monkeypatch.setattr(rollout, "generate_and_rm", generate)
    monkeypatch.setattr(rollout, "transfer_batch_to_data_system", put)
    runner = asyncio.create_task(rollout.generate_rollout_async(harness.args, 0, harness.source, object()))
    try:
        await asyncio.wait_for(transfer_entered.wait(), 2)
        done, _ = await asyncio.wait([runner], timeout=0.5)
        assert runner in done, "failed storage write did not interrupt the blocked generation wait"
        if recovery:
            from relax.distributed.settlement import UnsafeInferenceError

            assert isinstance(runner.exception(), UnsafeInferenceError)
            assert runner.exception().__cause__ is primary
        else:
            assert runner.exception() is primary
        assert generation_closed.is_set()
        assert all(task.done() for task in harness.tracked + harness.writes)
    finally:
        release_generation.set()
        if not runner.done():
            runner.cancel()
        await asyncio.gather(runner, *harness.tracked, *harness.writes, return_exceptions=True)


@pytest.mark.parametrize("final_flush", [False, True])
async def test_cancellation_settles_dispatched_write_before_returning(harness, monkeypatch, final_flush):
    writing, release_write, release_generation = asyncio.Event(), asyncio.Event(), asyncio.Event()
    write_finished, write_cancelled = asyncio.Event(), asyncio.Event()

    async def generate(args, sample, sampling_params, evaluation=False):
        harness.tracked.append(asyncio.current_task())
        if sample.index and not final_flush:
            await release_generation.wait()
        return sample

    async def put(*args, **kwargs):
        harness.writes.append(asyncio.current_task())
        writing.set()
        try:
            await release_write.wait()
            write_finished.set()
        except asyncio.CancelledError:
            write_cancelled.set()
            raise

    monkeypatch.setattr(rollout, "generate_and_rm", generate)
    monkeypatch.setattr(rollout, "transfer_batch_to_data_system", put)
    runner = asyncio.create_task(rollout.generate_rollout_async(harness.args, 0, harness.source, object()))
    try:
        await asyncio.wait_for(writing.wait(), 2)
        runner.cancel()
        done, _ = await asyncio.wait([runner], timeout=0.05)
        assert not done, "rollout reported cancellation before its dispatched storage write settled"
        runner.cancel()  # Repeated cancellation must not abandon cleanup or cancel the write.
        release_write.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(runner, 2)
        assert write_finished.is_set()
        assert not write_cancelled.is_set()
        assert all(task.done() for task in harness.tracked + harness.writes)
    finally:
        release_write.set()
        release_generation.set()
        await asyncio.gather(runner, *harness.tracked, *harness.writes, return_exceptions=True)


async def test_partial_dynamic_cancellation_settles_extra_completed_write(harness, monkeypatch):
    harness.args.fully_async = False
    harness.args.partial_rollout = True
    harness.args.use_dynamic_global_batch_size = True
    harness.args.resource = {"actor": (1, 1)}
    extra = [Sample(index=2, group_index=2, response="extra", status=Sample.Status.COMPLETED)]
    monkeypatch.setattr(rollout, "abort", AsyncMock(return_value=([], [extra])))
    writing, release_write = asyncio.Event(), asyncio.Event()
    write_finished, write_cancelled = asyncio.Event(), asyncio.Event()
    transferred = []

    async def put(args, groups, batch_size, rollout_id, client, **kwargs):
        harness.writes.append(asyncio.current_task())
        if groups == [extra]:
            writing.set()
            try:
                await release_write.wait()
            except asyncio.CancelledError:
                write_cancelled.set()
                raise
            write_finished.set()
        transferred.extend(sample.index for group in groups for sample in group)
        harness.current_batch.extend(sample for group in groups for sample in group)

    monkeypatch.setattr(rollout, "transfer_batch_to_data_system", put)
    runner = asyncio.create_task(rollout.generate_rollout_async(harness.args, 0, harness.source, object()))
    try:
        await asyncio.wait_for(writing.wait(), 2)
        assert sorted(transferred) == [0, 1]
        runner.cancel()
        done, _ = await asyncio.wait([runner], timeout=0.05)
        assert not write_cancelled.is_set(), "cancellation reached the dispatched extra-completed storage write"
        assert not done, "rollout returned before its extra-completed storage write settled"
        runner.cancel()
        release_write.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(runner, 2)
        assert write_finished.is_set()
        assert not write_cancelled.is_set()
        assert sorted(transferred) == [0, 1, 2]
        assert all(task.done() for task in harness.tracked + harness.writes)
    finally:
        release_write.set()
        if not runner.done():
            runner.cancel()
        await asyncio.gather(runner, *harness.tracked, *harness.writes, return_exceptions=True)


async def test_group_reward_timing_captures_overlap_without_mutating_rewards(harness, monkeypatch):
    generation = [asyncio.Event(), asyncio.Event()]
    scoring = [asyncio.Event(), asyncio.Event()]
    finish = [asyncio.Event(), asyncio.Event()]
    clock = [0.0]
    expected_rewards = [
        {"score": 1, "_perf_genrm_time": 2, "_perf_genrm_calls": "custom metadata"},
        {"score": 2, "_perf_genrm_time": 4, "_perf_genrm_calls": "more custom metadata"},
    ]
    rewards = [reward.copy() for reward in expected_rewards]
    harness.args.reward_key = "_perf_genrm_time"

    async def generate(args, sample, sampling_params, evaluation=False):
        await generation[sample.index].wait()
        return sample

    async def score(args, group, evaluation=False):
        index = group[0].index
        scoring[index].set()
        await finish[index].wait()
        return [rewards[index]]

    monkeypatch.setattr(rollout, "monotonic", lambda: clock[0])
    monkeypatch.setattr(rollout, "generate_and_rm", generate)
    monkeypatch.setattr(rollout, "batched_async_rm", score)
    runner = asyncio.create_task(rollout.generate_rollout_async(harness.args, 0, harness.source, object()))
    try:
        for index, started in [(0, 1.0), (1, 3.0)]:
            clock[0] = started
            generation[index].set()
            await asyncio.wait_for(scoring[index].wait(), 2)
        clock[0] = 4.0
        finish[0].set()
        await asyncio.wait_for(harness.write_started.wait(), 2)
        clock[0] = 8.0
        finish[1].set()
        await asyncio.wait_for(runner, 2)
        metrics = harness.log.call_args.args[3]
        expected = {"groups": 2, "sum": 8, "mean": 4, "max": 5, "wall": 7, "critical": 5}
        for key, value in expected.items():
            assert metrics.get(f"perf/reward_time/{key}") == value, f"missing or incorrect group reward {key} metric"
        assert rewards == expected_rewards
        samples = [sample for group in harness.groups for sample in group]
        assert [sample.reward for sample in samples] == expected_rewards
        assert [sample.get_reward_value(harness.args) for sample in samples] == [2, 4]
    finally:
        for event in [*generation, *finish]:
            event.set()
        if not runner.done():
            runner.cancel()
        await asyncio.wait_for(asyncio.gather(runner, *harness.tracked, *harness.writes, return_exceptions=True), 2)
