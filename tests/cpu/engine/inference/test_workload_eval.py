# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from relax.engine.rewards.async_eval import defer_eval_reward
from relax.engine.rollout import workload as module
from relax.engine.rollout.base_types import RolloutFnEvalOutput
from relax.utils.types import Sample


@pytest.mark.asyncio
@pytest.mark.parametrize("failed", [False, True])
async def test_managed_workload_drains_background_scores_before_reporting(monkeypatch, failed):
    sample = Sample(index=0)
    result = RolloutFnEvalOutput({"test": {"samples": [sample]}}, {})
    scored = asyncio.get_running_loop().create_future()
    queued = []
    reported = []
    killed = []

    def evaluate(*args, **kwargs):
        assert kwargs["evaluation"]
        assert defer_eval_reward(sample, {"evaluation": True})
        return result

    def score(args, payload):
        queued.append(payload)
        return scored

    judge = SimpleNamespace(score=SimpleNamespace(remote=score), close=SimpleNamespace(remote=AsyncMock()))
    monkeypatch.setattr(module, "load_function", lambda path: evaluate)
    monkeypatch.setattr(
        module, "EvalJudgeWorker", SimpleNamespace(options=lambda **kwargs: SimpleNamespace(remote=lambda args: judge))
    )
    monkeypatch.setattr(module.ray, "kill", killed.append)
    import relax.distributed.ray.policy_engine_pool as rollout

    monkeypatch.setattr(rollout, "_log_eval_rollout_data", lambda *args: reported.append(args))
    args = SimpleNamespace(
        rollout_function_path="relax.engine.rollout.sglang_rollout.generate_rollout",
        eval_function_path="relax.engine.rollout.sglang_rollout.generate_rollout",
        custom_reward_post_process_path=None,
        custom_convert_samples_to_train_data_path=None,
        ci_test=False,
    )
    workload = module.RolloutWorkload(args, None, None, SimpleNamespace(resume_health_monitoring=AsyncMock()))
    monkeypatch.setattr(workload, "save_debug_rollout_data", lambda *args, **kwargs: None)
    await workload.eval(7)
    assert len(queued) == 1
    assert queued[0][0] is result
    assert queued[0][1][0].samples is sample
    assert reported == []
    draining = asyncio.create_task(workload.drain_eval_judges())
    await asyncio.sleep(0)
    assert not draining.done()
    if failed:
        scored.set_exception(RuntimeError("judge failed"))
        with pytest.raises(RuntimeError, match="judge failed"):
            await asyncio.wait_for(draining, 2)
        assert reported == []
    else:
        sample.reward = 0.5
        scored.set_result(result)
        await asyncio.wait_for(draining, 2)
        assert reported[0][0] == 7
        assert reported[0][2]["test"]["samples"][0].reward == 0.5
        assert reported[0][3]["eval/async_judge"] == 1
    await workload.close()
    assert killed == [judge]


def test_sft_eval_reports_completed_train_step_and_uses_eval_reward_key(monkeypatch):
    from relax.distributed.ray import policy_engine_pool

    published = []
    monkeypatch.setattr(policy_engine_pool.tracking_utils, "log", lambda args, data, **kwargs: published.append(data))
    monkeypatch.setattr(policy_engine_pool.tracking_utils, "flush_metrics", lambda *args: None)
    workload = object.__new__(module.RolloutWorkload)
    workload.save_debug_rollout_data = lambda *args, **kwargs: None
    workload.args = SimpleNamespace(
        loss_type="sft",
        reward_key="train_score",
        eval_reward_key="eval_score",
        wandb_always_use_train_step=True,
        n_samples_per_prompt=2,
        rollout_batch_size=4,
        global_batch_size=2,
        log_reward_category=None,
        advantage_estimator="grpo",
        custom_eval_rollout_log_function_path=None,
        log_passrate=False,
    )
    sample = Sample(response="ready", reward={"eval_score": 1.0})
    data = {"sft_predict": {"rewards": [1.0], "samples": [sample]}}
    workload._record_eval_result(7, RolloutFnEvalOutput(data, {}))
    assert published[0]["eval/step"] == 7
    assert published[0]["eval/sft_predict/zero_std/count_1.0"] == 1
    assert workload.args.reward_key == "train_score"
