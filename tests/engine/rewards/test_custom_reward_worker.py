# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from __future__ import annotations

import asyncio
import os
import threading
from types import SimpleNamespace
from typing import Any, Awaitable, Iterator, TypeVar

import pytest


try:
    import ray

    if not ray.is_initialized():
        try:
            ray.init(num_cpus=8, ignore_reinit_error=True)
        except ValueError as exc:
            if "When connecting to an existing cluster" in str(exc):
                ray.init(ignore_reinit_error=True)
            else:
                raise

    from relax.engine import rewards as rewards_module
    from relax.engine.rewards import RewardExecutionError, RewardExecutor, async_rm, batched_async_rm
    from relax.utils.reload_utils import ReloadableMixin
    from relax.utils.types import Sample
except ModuleNotFoundError as exc:
    if exc.name != "ray":
        raise
    pytest.skip("Ray is not installed", allow_module_level=True)


CUSTOM_REWARD_MODULE = "tests.engine.rewards.custom_reward_fixtures"
SYNC_PID_REWARD = f"{CUSTOM_REWARD_MODULE}.sync_pid_reward"
ASYNC_PID_REWARD = f"{CUSTOM_REWARD_MODULE}.async_pid_reward"
ASYNC_BATCH_PID_REWARD = f"{CUSTOM_REWARD_MODULE}.async_batch_pid_reward"
ASYNC_CALLABLE_PID_REWARD = f"{CUSTOM_REWARD_MODULE}.async_callable_pid_reward"
WRAPPED_ASYNC_PID_REWARD = f"{CUSTOM_REWARD_MODULE}.wrapped_async_pid_reward"
SYNC_CONDITIONAL_REWARD = f"{CUSTOM_REWARD_MODULE}.sync_conditional_reward"
SYNC_CONCURRENCY_REWARD = f"{CUSTOM_REWARD_MODULE}.sync_concurrency_reward"
SYNC_BATCH_CONDITIONAL_REWARD = f"{CUSTOM_REWARD_MODULE}.sync_batch_conditional_reward"
SYNC_BATCH_CONCURRENCY_REWARD = f"{CUSTOM_REWARD_MODULE}.sync_batch_concurrency_reward"
SYNC_LOAD_COUNT_REWARD = f"{CUSTOM_REWARD_MODULE}.sync_load_count_reward"
SYNC_RELOAD_COUNT_REWARD = f"{CUSTOM_REWARD_MODULE}.sync_reload_count_reward"
ASYNC_RELOAD_COUNT_REWARD = f"{CUSTOM_REWARD_MODULE}.async_reload_count_reward"
RAY_TEST_TIMEOUT = 10


T = TypeVar("T")


def _make_args(**overrides: Any) -> SimpleNamespace:
    defaults = {
        "rm_type": "openr1mm",
        "custom_rm_path": None,
        "group_rm": False,
        "reward_max_concurrency": 64,
        "reward_num_workers": 4,
        "rm_url": None,
        "reward_key": None,
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def _make_sample(
    index: int,
    response: str = "response",
    label: str = "label",
    metadata: dict[str, Any] | None = None,
    *,
    group_index: int | None = None,
    session_id: str | None = None,
) -> "Sample":
    return Sample(
        response=response,
        label=label,
        metadata=metadata or {},
        index=index,
        group_index=group_index,
        session_id=session_id,
    )


def _kill_workers(executor: RewardExecutor | None) -> None:
    if executor is None:
        return
    for worker in executor._workers:
        try:
            ray.kill(worker)
        except Exception:
            # A worker may already be dead after an expected failure. Teardown
            # must still clear the remaining actor handles.
            pass
    executor._workers.clear()


async def _wait(awaitable: Awaitable[T], timeout: float = RAY_TEST_TIMEOUT) -> T:
    return await asyncio.wait_for(awaitable, timeout=timeout)


@pytest.fixture(autouse=True)
def _reset_executor() -> Iterator[None]:
    _kill_workers(RewardExecutor._instance)
    RewardExecutor._instance = None
    yield
    _kill_workers(RewardExecutor._instance)
    RewardExecutor._instance = None


@pytest.fixture
def loaded_reward_paths(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    original_load_function = rewards_module.load_function
    loaded_paths = []

    def track_load(path: str):
        loaded_paths.append(path)
        return original_load_function(path)

    monkeypatch.setattr(rewards_module, "load_function", track_load)
    return loaded_paths


class TestCustomRewardDispatch:
    @pytest.mark.asyncio
    async def test_sync_reward_runs_in_worker_without_blocking_event_loop(self, monkeypatch):
        event_loop_thread = threading.get_ident()
        submission_threads = []
        original_to_thread = asyncio.to_thread

        async def tracked_to_thread(function, /, *args, **kwargs):
            def run():
                submission_threads.append(threading.get_ident())
                return function(*args, **kwargs)

            return await original_to_thread(run)

        monkeypatch.setattr(asyncio, "to_thread", tracked_to_thread)
        args = _make_args(custom_rm_path=SYNC_PID_REWARD, reward_num_workers=1)

        result = await _wait(async_rm(args, _make_sample(index=3)))

        assert len(submission_threads) >= 2
        assert all(thread_id != event_loop_thread for thread_id in submission_threads)
        assert result["index"] == 3
        assert result["pid"] != os.getpid()

    @pytest.mark.asyncio
    async def test_batched_sync_custom_reward_keeps_payload_contract(self):
        args = _make_args(custom_rm_path=SYNC_BATCH_CONDITIONAL_REWARD, group_rm=False, fail_reward=False)
        samples = [_make_sample(index=index) for index in range(3)]

        rewards = await _wait(batched_async_rm(args, samples))

        assert [reward["index"] for reward in rewards] == [0, 1, 2]
        assert all(reward["batch_size"] == 3 for reward in rewards)
        assert len({reward["pid"] for reward in rewards}) == 1
        assert rewards[0]["pid"] != os.getpid()

    @pytest.mark.asyncio
    async def test_batched_async_custom_reward_keeps_payload_contract_without_workers(self):
        args = _make_args(custom_rm_path=ASYNC_BATCH_PID_REWARD, group_rm=True)
        samples = [_make_sample(index=index) for index in range(3)]

        rewards = await _wait(batched_async_rm(args, samples))

        assert [reward["index"] for reward in rewards] == [0, 1, 2]
        assert all(reward["batch_size"] == 3 for reward in rewards)
        assert {reward["pid"] for reward in rewards} == {os.getpid()}
        assert RewardExecutor._instance._workers == []

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "custom_rm_path",
        [ASYNC_PID_REWARD, ASYNC_CALLABLE_PID_REWARD, WRAPPED_ASYNC_PID_REWARD],
        ids=["function", "callable", "wrapped"],
    )
    async def test_async_custom_reward_is_awaited_without_workers(self, custom_rm_path):
        args = _make_args(custom_rm_path=custom_rm_path, reward_num_workers=2)

        result = await _wait(async_rm(args, _make_sample(index=4)))

        assert result == {"pid": os.getpid(), "index": 4}
        assert RewardExecutor._instance._workers == []


class TestRewardExecutorConfiguration:
    @pytest.mark.asyncio
    async def test_none_reward_max_concurrency_uses_executor_default(self):
        args = _make_args(
            custom_rm_path=ASYNC_PID_REWARD,
            reward_max_concurrency=None,
        )

        result = await _wait(async_rm(args, _make_sample(index=4)))

        assert result == {"pid": os.getpid(), "index": 4}
        assert RewardExecutor._instance._max_concurrency == 64

    @pytest.mark.asyncio
    async def test_reward_max_concurrency_controls_sync_custom_reward(self, tmp_path):
        counter_path = tmp_path / "counter.txt"
        counter_path.write_text("0,0", encoding="utf-8")
        args = _make_args(
            custom_rm_path=SYNC_CONCURRENCY_REWARD,
            counter_path=str(counter_path),
            reward_delay=0.1,
            reward_max_concurrency=2,
            reward_num_workers=4,
        )
        samples = [_make_sample(index=index) for index in range(8)]

        await _wait(asyncio.gather(*(async_rm(args, sample) for sample in samples)))

        current, maximum = (int(value) for value in counter_path.read_text(encoding="utf-8").split(","))
        assert current == 0
        assert maximum == 2

    @pytest.mark.asyncio
    async def test_reward_max_concurrency_counts_complete_batch_as_one_invocation(self, tmp_path):
        counter_path = tmp_path / "batch-counter.txt"
        counter_path.write_text("0,0", encoding="utf-8")
        args = _make_args(
            custom_rm_path=SYNC_BATCH_CONCURRENCY_REWARD,
            counter_path=str(counter_path),
            reward_delay=0.1,
            reward_max_concurrency=2,
            reward_num_workers=4,
        )
        batches = [[_make_sample(index=batch_index * 3 + offset) for offset in range(3)] for batch_index in range(6)]

        rewards = await _wait(asyncio.gather(*(batched_async_rm(args, samples) for samples in batches)))

        current, maximum = (int(value) for value in counter_path.read_text(encoding="utf-8").split(","))
        assert current == 0
        assert maximum == 2
        assert all(reward == [1.0, 1.0, 1.0] for reward in rewards)


class TestCustomRewardLoading:
    @pytest.mark.asyncio
    async def test_sync_custom_reward_is_loaded_once_per_process(self, loaded_reward_paths):
        args = _make_args(
            custom_rm_path=SYNC_LOAD_COUNT_REWARD,
            reward_delay=0.05,
            reward_num_workers=3,
        )
        samples = [_make_sample(index=index) for index in range(12)]

        results = await _wait(asyncio.gather(*(async_rm(args, sample) for sample in samples)))

        assert loaded_reward_paths == [SYNC_LOAD_COUNT_REWARD]
        assert len(RewardExecutor._instance._workers) == 3
        assert len({result["pid"] for result in results}) == 3
        assert {result["load_count"] for result in results} == {1}

    @pytest.mark.asyncio
    async def test_explicit_reload_refreshes_custom_reward_in_all_worker_processes(self):
        args = _make_args(custom_rm_path=SYNC_RELOAD_COUNT_REWARD, reward_num_workers=3)
        samples = [_make_sample(index=index) for index in range(3)]

        before_reload = await _wait(asyncio.gather(*(async_rm(args, sample) for sample in samples)))
        reload_owner = ReloadableMixin()
        reload_owner.args = args
        reload_result = reload_owner.reload_function_by_name("custom_rm")
        after_reload = await _wait(asyncio.gather(*(async_rm(args, sample) for sample in samples)))

        assert reload_result["success"] is True
        before_reload_counts = {result["pid"]: result["reload_count"] for result in before_reload}
        after_reload_counts = {result["pid"]: result["reload_count"] for result in after_reload}
        assert len(before_reload_counts) == 3
        assert after_reload_counts == {pid: reload_count + 1 for pid, reload_count in before_reload_counts.items()}

    @pytest.mark.asyncio
    async def test_explicit_reload_refreshes_cached_async_custom_reward(self):
        args = _make_args(custom_rm_path=ASYNC_RELOAD_COUNT_REWARD)
        sample = _make_sample(index=0)

        before_reload = await _wait(async_rm(args, sample))
        reload_owner = ReloadableMixin()
        reload_owner.args = args
        reload_result = reload_owner.reload_function_by_name("custom_rm")
        after_reload = await _wait(async_rm(args, sample))

        assert reload_result["success"] is True
        assert before_reload["pid"] == os.getpid()
        assert after_reload["pid"] == os.getpid()
        assert after_reload["reload_count"] == before_reload["reload_count"] + 1
        assert RewardExecutor._instance._workers == []


class TestCustomRewardErrors:
    @pytest.mark.asyncio
    async def test_cancel_during_submission_cancels_eventual_ray_task(self, monkeypatch):
        args = _make_args(
            custom_rm_path=SYNC_PID_REWARD,
            reward_delay=0,
            reward_max_concurrency=1,
            reward_num_workers=1,
        )
        await _wait(async_rm(args, _make_sample(index=-1)))

        submission_started = threading.Event()
        allow_submission = threading.Event()
        original_to_thread = asyncio.to_thread
        original_cancel = ray.cancel
        cancelled_refs = []

        async def delayed_to_thread(function, /, *args, **kwargs):
            def run():
                submission_started.set()
                assert allow_submission.wait(timeout=RAY_TEST_TIMEOUT)
                return function(*args, **kwargs)

            return await original_to_thread(run)

        def tracked_cancel(ref, *, force):
            cancelled_refs.append(ref)
            return original_cancel(ref, force=force)

        monkeypatch.setattr(asyncio, "to_thread", delayed_to_thread)
        monkeypatch.setattr(ray, "cancel", tracked_cancel)

        cancelled_call = asyncio.create_task(async_rm(args, _make_sample(index=0)))
        for _ in range(RAY_TEST_TIMEOUT * 50):
            if submission_started.is_set():
                break
            await asyncio.sleep(0.02)
        else:
            pytest.fail("Ray task submission did not start.")

        cancelled_call.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(cancelled_call, timeout=1)

        assert cancelled_refs == []
        allow_submission.set()
        for _ in range(RAY_TEST_TIMEOUT * 50):
            if cancelled_refs:
                break
            await asyncio.sleep(0.02)
        else:
            pytest.fail("Submitted Ray task was not cancelled.")
        assert len(cancelled_refs) == 1
        args.reward_delay = 0
        assert (await _wait(async_rm(args, _make_sample(index=1))))["index"] == 1

    @pytest.mark.asyncio
    async def test_cancelled_sync_reward_allows_next_call_to_complete(self, tmp_path):
        counter_path = tmp_path / "cancel-counter.txt"
        counter_path.write_text("0,0", encoding="utf-8")
        args = _make_args(
            custom_rm_path=SYNC_CONCURRENCY_REWARD,
            counter_path=str(counter_path),
            reward_delay=0.5,
            reward_max_concurrency=1,
            reward_num_workers=1,
        )
        cancelled_call = asyncio.create_task(async_rm(args, _make_sample(index=0)))
        for _ in range(RAY_TEST_TIMEOUT * 50):
            if cancelled_call.done():
                await cancelled_call
            current, _ = (int(value) for value in counter_path.read_text(encoding="utf-8").split(","))
            if current == 1:
                break
            await asyncio.sleep(0.02)
        else:
            pytest.fail("Synchronous custom reward did not start in the Ray actor.")

        cancelled_call.cancel()
        with pytest.raises(asyncio.CancelledError):
            await cancelled_call

        args.reward_delay = 0
        assert await _wait(async_rm(args, _make_sample(index=1))) == 1.0
        current, _ = (int(value) for value in counter_path.read_text(encoding="utf-8").split(","))
        assert current == 0

    @pytest.mark.asyncio
    async def test_custom_reward_error_identifies_sample_and_next_call_completes(self):
        args = _make_args(custom_rm_path=SYNC_CONDITIONAL_REWARD, fail_reward=True)
        sample = _make_sample(index=17, group_index=2, session_id="session-17")

        with pytest.raises(RewardExecutionError, match="index=17") as error_info:
            await _wait(async_rm(args, sample))

        error_message = str(error_info.value)
        assert "session_id='session-17'" in error_message
        assert "group_index=2" in error_message
        assert error_info.value.__cause__ is error_info.value.cause

        args.fail_reward = False
        assert await _wait(async_rm(args, sample)) == 17

    @pytest.mark.asyncio
    async def test_group_error_is_bounded_and_releases_concurrency_slot(self):
        args = _make_args(
            custom_rm_path=SYNC_BATCH_CONDITIONAL_REWARD,
            group_rm=True,
            fail_reward=True,
            reward_max_concurrency=1,
            reward_num_workers=3,
        )
        long_id = "context-value-" * 100
        samples = [
            _make_sample(
                index=index,
                group_index=7,
                session_id=f"session-{index}-{long_id}",
                metadata={"request_id": f"request-{index}-{long_id}"},
            )
            for index in range(20)
        ]

        with pytest.raises(RewardExecutionError) as error_info:
            await _wait(batched_async_rm(args, samples))

        sample_context = error_info.value.sample_context
        assert len(sample_context) <= 2048
        assert "batch_size=20" in sample_context
        assert "sample[0]" in sample_context
        assert "sample[3]" in sample_context
        assert "sample[4]" not in sample_context
        assert "omitted_samples=16" in sample_context
        assert long_id not in sample_context

        args.fail_reward = False
        rewards = await _wait(batched_async_rm(args, samples))
        assert [reward["index"] for reward in rewards] == list(range(20))
        assert all(reward["batch_size"] == 20 for reward in rewards)
        assert len({reward["pid"] for reward in rewards}) == 1
        assert rewards[0]["pid"] != os.getpid()


class TestCustomRewardAndRouterCoexistence:
    """Joint regressions against the Task 18 format-aware registry."""

    @pytest.mark.asyncio
    async def test_custom_rm_bypasses_router_with_unknown_metadata_rm_type(self):
        from relax.engine.rewards import registry as reward_registry

        reward_registry.reset_router_state()
        args = _make_args(rm_type="math", custom_rm_path=SYNC_PID_REWARD, reward_num_workers=1)
        sample = _make_sample(
            index=9,
            response="ignored",
            label="ignored",
            metadata={"rm_type": "totally_unknown_type"},
        )

        result = await _wait(async_rm(args, sample))

        assert result == {"pid": result["pid"], "index": 9}
        assert result["pid"] != os.getpid()
        assert reward_registry.get_router_stats() == {}

    @pytest.mark.asyncio
    async def test_batched_custom_rm_bypasses_router_with_unknown_metadata_rm_type(self):
        from relax.engine.rewards import registry as reward_registry

        reward_registry.reset_router_state()
        args = _make_args(
            custom_rm_path=SYNC_BATCH_CONDITIONAL_REWARD,
            group_rm=True,
            fail_reward=False,
            reward_num_workers=1,
        )
        samples = [
            _make_sample(index=0, metadata={"rm_type": "totally_unknown_type"}),
            _make_sample(index=1, metadata={"rm_type": "another_unknown_type"}),
        ]

        rewards = await _wait(batched_async_rm(args, samples))

        assert [reward["index"] for reward in rewards] == [0, 1]
        assert all(reward["batch_size"] == 2 for reward in rewards)
        assert rewards[0]["pid"] != os.getpid()
        assert reward_registry.get_router_stats() == {}
