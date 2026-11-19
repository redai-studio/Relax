# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from argparse import Namespace

import pytest

from examples.osworld_agentic.app.reward_post_process import (
    compute_legacy_raw_reward,
    post_process_rewards,
    shape_outcome_reward,
)
from relax.utils.types import Sample


@pytest.fixture
def reward_env(monkeypatch):
    monkeypatch.setenv("OSWORLD_MAX_STEPS_REF", "30")
    monkeypatch.setenv("OSWORLD_LENGTH_PENALTY_COEF", "0.05")
    monkeypatch.setenv("OSWORLD_CAP_PENALTY", "0.2")
    monkeypatch.setenv("OSWORLD_PER_TRAJ_NORM", "0")
    monkeypatch.delenv("OSWORLD_REWARD_DUMP", raising=False)


@pytest.mark.parametrize("outcome,turns,expected", [(1, 10, 59 / 60), (0, 30, -1.25), (1, 30, 0.75)])
def test_shaping_matches_handoff(outcome, turns, expected, reward_env):
    assert shape_outcome_reward(outcome, turns) == pytest.approx(expected)


def test_raw_reward_matches_handoff_export_weighting(reward_env):
    samples = [
        Sample(
            group_index=0,
            session_id=session,
            reward=outcome,
            metadata={"outcome": outcome, "agent_turns": turns},
        )
        for session, outcome, turns, count in [("success", 1, 10, 2), ("failure", 0, 30, 3)]
        for _ in range(count)
    ]
    args = Namespace(reward_key=None, advantage_estimator="grpo", rewards_normalization=True)
    shaped, advantages = post_process_rewards(args, samples)
    assert compute_legacy_raw_reward(samples) == pytest.approx((2 * (59 / 60) - 3 * 1.25) / 5)
    assert compute_legacy_raw_reward(samples) == pytest.approx(sum(shaped) / len(shaped))
    assert advantages == pytest.approx([1, 1, -1, -1, -1])


def test_recipe_logger_overrides_success_rate_without_duplicate_reporting(monkeypatch, reward_env):
    from relax.distributed.ray import rollout
    from relax.utils import tracking_utils

    samples = [Sample(session_id="failed", metadata={"outcome": 0, "agent_turns": 30})]
    monkeypatch.setattr(rollout, "save_rollout_result_jsonl", lambda *args: None)
    monkeypatch.setattr(rollout, "compute_metrics_from_samples", lambda *args, **kwargs: {"raw_reward": 0.0})
    monkeypatch.setattr(rollout, "compute_perf_metrics_from_samples", lambda *args: {"rollout_time": 5})

    reported = []
    flushed = []
    monkeypatch.setattr(tracking_utils, "log", lambda args, data, **kwargs: reported.append(data))
    monkeypatch.setattr(tracking_utils, "flush_metrics", lambda args, step: flushed.append(step))
    args = Namespace(
        custom_rollout_log_function_path="examples.osworld_agentic.app.metrics.log_rollout_data",
        wandb_always_use_train_step=False,
        load_debug_rollout_data=None,
    )
    rollout._log_rollout_data(3, args, samples, {"osworld/success_rate/mean": 0.0}, 5)
    assert len(reported) == 1
    assert reported[0]["rollout/raw_reward"] == -1.25
    assert reported[0]["osworld/success_rate/mean"] == 0.0
    assert reported[0]["perf/rollout_time"] == 5
    assert reported[0]["rollout/step"] == 3
    assert flushed == [3]
