# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import asyncio
import logging
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from fastapi import HTTPException

from relax.components.rollout import Rollout
from relax.distributed.ray.rollout import RolloutManager
from relax.engine.rewards.async_eval import EvalJudgeError


def _manager():
    manager = object.__new__(RolloutManager.__ray_metadata__.modified_class)
    manager.args = SimpleNamespace(inference_recovery=True)
    manager._eval_judge_tasks, manager._eval_judge_errors = set(), []
    manager._eval_judge_init_lock = asyncio.Lock()
    return manager


def _component(eval_rpc, drain_rpc):
    def submit(rollout_id):
        return asyncio.create_task(eval_rpc(rollout_id))

    def drain(refs, **kwargs):
        # Each actor gets its own cancelable waiter for the same Ray ObjectRef.
        return drain_rpc([asyncio.shield(ref) for ref in refs], **kwargs)

    rollout = object.__new__(Rollout.func_or_class.__mro__[1])
    rollout._logger_instance = logging.getLogger(__name__)
    rollout.config = SimpleNamespace(eval_interval=1, eval_prompt_data=["eval"])
    rollout.num_rollout_per_epoch = None
    rollout._eval_draining = False
    rollout._eval_handlers = {}
    rollout._eval_error = None
    rollout.rollout_manager = SimpleNamespace(
        eval=SimpleNamespace(remote=submit), drain_eval_judges=SimpleNamespace(remote=drain)
    )
    return rollout


@pytest.mark.asyncio
async def test_recovery_closes_admission_and_waits_for_earlier_evaluation():
    first_started = asyncio.Event()
    submit_first = asyncio.Event()
    last_submitted = asyncio.Event()
    submitted = []
    drained = []

    async def evaluate(rollout_id):
        if rollout_id == 0:
            first_started.set()
            await submit_first.wait()
        submitted.append(rollout_id)
        if rollout_id == 1:
            last_submitted.set()

    manager = _manager()

    async def drain(refs, **kwargs):
        await manager.drain_eval_judges(refs, **kwargs)
        drained.append(tuple(submitted))

    rollout = _component(evaluate, drain)
    await rollout.evaluate(0)
    await first_started.wait()
    await rollout.evaluate(1)
    await last_submitted.wait()
    draining = asyncio.create_task(rollout.drain_eval_judges(recovering=True))
    try:
        await asyncio.sleep(0)
        assert not (await rollout.is_eval_done())["done"]
        with pytest.raises(HTTPException) as rejected:
            await rollout.evaluate(2)
        assert rejected.value.status_code == 503
        assert not drained
    finally:
        submit_first.set()
        await asyncio.wait_for(draining, timeout=2)
    assert drained == [(1, 0)]


@pytest.mark.asyncio
async def test_recovery_preserves_completed_inline_reporting_failure(monkeypatch):
    from relax.distributed.ray import rollout as rollout_module

    manager = _manager()
    manager.eval_generate_rollout = manager.data_source = manager.data_system_client = None
    manager._begin_generation = manager.health_monitoring_resume = Mock()
    manager._record_eval_result = Mock(side_effect=ValueError("metric write failed"))
    monkeypatch.setattr(rollout_module, "call_eval_with_deferred_rewards", lambda *args: (object(), []))
    last_submitted = asyncio.Event()
    release = asyncio.Event()

    async def evaluate(rollout_id):
        if rollout_id == 0:
            await manager.eval(rollout_id)
        else:
            last_submitted.set()
            await release.wait()

    rollout = _component(evaluate, manager.drain_eval_judges)
    await rollout.evaluate(0)
    # Complete the failing request and overwrite the latest handler before recovery starts.
    async with asyncio.timeout(2):
        while not (await rollout.is_eval_done())["done"]:
            await asyncio.sleep(0)
    await rollout.evaluate(1)
    await last_submitted.wait()
    try:
        with pytest.raises(EvalJudgeError, match="metric write failed"):
            await asyncio.wait_for(rollout.drain_eval_judges(recovering=True), timeout=2)
    finally:
        release.set()
        await asyncio.gather(*rollout._eval_handlers)


@pytest.mark.asyncio
@pytest.mark.parametrize("batched", [False, True])
async def test_inline_reward_failure_is_terminal_only_for_recovery_evaluation(monkeypatch, batched):
    from relax.engine import rewards

    async def fail(*_args, **_kwargs):
        raise ValueError("reward failed")

    monkeypatch.setattr(rewards, "_get_reward_executor", lambda _: SimpleNamespace(execute=fail, execute_batch=fail))
    score = rewards.batched_async_rm if batched else rewards.async_rm
    args = SimpleNamespace(inference_recovery=True, disable_eval_async_judge=True)
    samples = [object()] if batched else object()
    with pytest.raises(EvalJudgeError, match="reward failed"):
        await score(args, samples, evaluation=True)
    with pytest.raises(ValueError, match="reward failed"):
        await score(args, samples, evaluation=False)


@pytest.mark.asyncio
async def test_initial_scoring_failure_does_not_request_rollout_recovery():
    async def score(rollout_id):
        raise EvalJudgeError("initial score failed")

    rollout = _component(score, Mock())
    rollout.config.inference_recovery = True
    rollout.config.skip_eval_before_train = False
    rollout.step = 0
    rollout._recovery_gate = Mock()
    rollout.healthy = Mock()
    with pytest.raises(EvalJudgeError, match="initial score failed"):
        await rollout._async_run()
    rollout._recovery_gate.fail.assert_not_called()
    rollout.healthy.report_error.remote.assert_not_called()


@pytest.mark.asyncio
async def test_pending_judge_does_not_hide_failed_judge_during_drain():
    manager = _manager()
    slow, failed = [asyncio.get_running_loop().create_future() for _ in range(2)]
    manager._eval_judge = SimpleNamespace(score=SimpleNamespace(remote=Mock(side_effect=[slow, failed])))
    for rollout_id in (0, 1):
        await manager.submit_eval_result(rollout_id, object(), [object()])
    drain = asyncio.create_task(manager.drain_eval_judges())
    await asyncio.sleep(0)
    failed.set_exception(ValueError("judge failed"))
    try:
        with pytest.raises(EvalJudgeError, match="rollout 1: judge failed"):
            await asyncio.wait_for(drain, timeout=2)
    finally:
        slow.cancel()
        await asyncio.gather(*manager._eval_judge_tasks, return_exceptions=True)


@pytest.mark.asyncio
async def test_normal_drain_propagates_generation_error_but_recovery_discards_it():
    async def fail(rollout_id):
        raise ConnectionError("generation unavailable")

    rollout = _component(fail, _manager().drain_eval_judges)
    await rollout.evaluate(0)
    await rollout.drain_eval_judges(recovering=True)
    with pytest.raises(ConnectionError, match="generation unavailable"):
        await rollout.drain_eval_judges()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("generation_finished", "judge_failed"),
    [(True, True), (True, False), (False, False)],
    ids=["both-failed", "cached-generation", "in-flight-generation"],
)
async def test_final_drain_prefers_known_score_errors_and_leaves_pending_judges_running(
    generation_finished, judge_failed
):
    manager = _manager()
    score = asyncio.get_running_loop().create_future()
    manager._eval_judge = SimpleNamespace(score=SimpleNamespace(remote=Mock(return_value=score)))
    await manager.submit_eval_result(1, object(), [object()])

    async def evaluate(rollout_id):
        raise ConnectionError("generation unavailable")

    rollout = _component(evaluate, manager.drain_eval_judges)
    await rollout.evaluate(0)
    if generation_finished:
        await asyncio.gather(*rollout._eval_handlers, return_exceptions=True)
    try:
        if judge_failed:
            score.set_exception(ValueError("score failed"))
            await asyncio.gather(*manager._eval_judge_tasks)
            with pytest.raises(EvalJudgeError, match="rollout 1: score failed"):
                await rollout.drain_eval_judges()
        else:
            with pytest.raises(ConnectionError, match="generation unavailable"):
                await asyncio.wait_for(rollout.drain_eval_judges(), timeout=1)
            assert not score.done()
            recovery = asyncio.create_task(rollout.drain_eval_judges(recovering=True))
            await asyncio.sleep(0)
            score.set_exception(ValueError("score failed during recovery"))
            with pytest.raises(EvalJudgeError, match="score failed during recovery"):
                await asyncio.wait_for(recovery, timeout=2)
    finally:
        score.cancel()
        await asyncio.gather(*manager._eval_judge_tasks, return_exceptions=True)


@pytest.mark.asyncio
async def test_final_drain_sees_inline_report_error_before_submission_ref_arrives(monkeypatch):
    from relax.distributed.ray import rollout as rollout_module

    manager = _manager()
    manager.eval_generate_rollout = manager.data_source = manager.data_system_client = None
    manager._begin_generation = manager.health_monitoring_resume = Mock()
    manager._record_eval_result = Mock(side_effect=ValueError("metric write failed"))
    monkeypatch.setattr(rollout_module, "call_eval_with_deferred_rewards", lambda *args: (object(), []))
    reported, deliver = asyncio.Event(), asyncio.Event()

    async def evaluate(rollout_id):
        if rollout_id == 0:
            raise ConnectionError("generation unavailable")
        try:
            await manager.eval(rollout_id)
        finally:
            reported.set()
            await deliver.wait()

    rollout = _component(evaluate, manager.drain_eval_judges)
    await rollout.evaluate(0)
    await rollout.evaluate(1)
    try:
        await asyncio.wait_for(reported.wait(), timeout=2)
        with pytest.raises(EvalJudgeError, match="metric write failed"):
            await asyncio.wait_for(rollout.drain_eval_judges(), timeout=2)
    finally:
        deliver.set()
        await asyncio.gather(*rollout._eval_handlers, return_exceptions=True)


@pytest.mark.asyncio
async def test_final_drain_checks_all_ready_submissions_for_scoring_failure():
    manager = _manager()
    generation, scoring = [asyncio.get_running_loop().create_future() for _ in range(2)]
    generation.set_exception(ConnectionError("generation unavailable"))
    scoring.set_exception(EvalJudgeError("score failed"))
    with pytest.raises(EvalJudgeError, match="score failed"):
        await manager.drain_eval_judges([generation, scoring])


@pytest.mark.asyncio
async def test_new_judge_failure_reaches_drain_while_another_generation_is_pending():
    manager = _manager()
    score = asyncio.get_running_loop().create_future()
    manager._eval_judge = SimpleNamespace(score=SimpleNamespace(remote=Mock(return_value=score)))
    release = asyncio.Event()
    submit = asyncio.Event()
    queued = asyncio.Event()

    async def evaluate(rollout_id):
        if rollout_id == 0:
            await release.wait()
        else:
            await submit.wait()
            await manager.submit_eval_result(rollout_id, object(), [object()])
            queued.set()

    rollout = _component(evaluate, manager.drain_eval_judges)
    await rollout.evaluate(0)
    await rollout.evaluate(1)
    drain = asyncio.create_task(rollout.drain_eval_judges(recovering=True))
    try:
        await asyncio.sleep(0)
        submit.set()
        await asyncio.wait_for(queued.wait(), timeout=2)
        score.set_exception(ValueError("late judge failed"))
        with pytest.raises(EvalJudgeError, match="rollout 1: late judge failed"):
            await asyncio.wait_for(drain, timeout=2)
        assert not (await rollout.is_eval_done())["done"]
    finally:
        release.set()
        await asyncio.gather(*rollout._eval_handlers)
