# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Regression tests for training-only RLOO rollout diagnostics."""

from types import SimpleNamespace

import pytest

from relax.utils.metrics.metric_utils import compute_rollout_explicit_reward_metrics
from relax.utils.types import Sample


def _args(**overrides):
    values = {
        "advantage_estimator": "rloo",
        "n_samples_per_prompt": 3,
        "reward_key": None,
        "log_passrate": False,
        "custom_reward_post_process_path": None,
        "agentic_custom_advantage_path": None,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _sample(group_index, reward, response_length, loss_mask=None):
    return Sample(
        group_index=group_index,
        reward=reward,
        response_length=response_length,
        loss_mask=loss_mask,
    )


def test_rloo_diagnostics_use_effective_tokens_but_literal_empty_responses():
    samples = [
        _sample(0, 0.0, 4, [1, 1, 1, 1]),
        _sample(0, 1.0, 4, [1, 0, 0, 0]),
        _sample(0, 2.0, 2, [1, 1]),
        _sample(1, 3.0, 2, [1, 1]),
        _sample(1, 3.0, 0, []),
        _sample(1, 3.0, 3, [1, 0, 0]),
    ]

    metrics = compute_rollout_explicit_reward_metrics(_args(), samples)

    assert metrics["rloo/baseline_mean"] == pytest.approx(2.0)
    assert metrics["rloo/adv_abs_mean"] == pytest.approx(0.5)
    assert metrics["rloo/no_signal_frac"] == pytest.approx(0.4)
    assert metrics["rloo/empty_response_frac"] == pytest.approx(1 / 6)
    assert metrics["rloo/zero_adv_group_frac"] == pytest.approx(0.5)
    assert metrics["rloo/dropped_group_frac"] == 0.0


def test_rloo_diagnostics_report_incomplete_groups():
    samples = [
        _sample(0, 0.0, 1),
        _sample(0, 1.0, 1),
        _sample(0, 2.0, 1),
        _sample(7, 0.0, 1),
        _sample(7, 1.0, 1),
    ]

    metrics = compute_rollout_explicit_reward_metrics(_args(), samples)

    assert metrics["rloo/dropped_group_frac"] == pytest.approx(0.5)
    assert metrics["rloo/empty_response_frac"] == 0.0


def test_eval_context_omits_rloo_diagnostics():
    samples = [
        _sample(0, 0.0, 1),
        _sample(0, 1.0, 1),
        _sample(0, 2.0, 1),
    ]

    metrics = compute_rollout_explicit_reward_metrics(
        _args(),
        samples,
        include_rloo_diagnostics=False,
    )

    assert not any(key.startswith("rloo/") for key in metrics)


@pytest.mark.parametrize("custom_path", ["custom_reward_post_process_path", "agentic_custom_advantage_path"])
def test_custom_advantage_paths_omit_standard_rloo_diagnostics(custom_path):
    metrics = compute_rollout_explicit_reward_metrics(
        _args(**{custom_path: "package.module:function"}),
        [_sample(0, 0.0, 1), _sample(0, 1.0, 1), _sample(0, 2.0, 1)],
    )

    assert not any(key.startswith("rloo/") for key in metrics)


def test_non_rloo_estimator_omits_rloo_diagnostics():
    metrics = compute_rollout_explicit_reward_metrics(
        _args(advantage_estimator="grpo"),
        [_sample(0, 1.0, 1)],
    )

    assert not any(key.startswith("rloo/") for key in metrics)


def test_eval_logger_disables_training_rloo_diagnostics(monkeypatch):
    pytest.importorskip("megatron.core")

    import relax.distributed.ray.rollout as rollout_module

    calls = []
    monkeypatch.setattr(rollout_module, "save_eval_summary_jsonl", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        rollout_module,
        "compute_metrics_from_samples",
        lambda _args, _samples, **kwargs: calls.append(kwargs) or {},
    )
    monkeypatch.setattr(rollout_module, "compute_rollout_step", lambda *_args, **_kwargs: 0)
    monkeypatch.setattr(rollout_module.tracking_utils, "log", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(rollout_module.tracking_utils, "flush_metrics", lambda *_args, **_kwargs: None)

    args = SimpleNamespace(
        custom_eval_rollout_log_function_path=None,
        log_passrate=False,
    )
    data = {"gsm8k": {"rewards": [1.0], "samples": [_sample(0, 1.0, 1)]}}

    rollout_module._log_eval_rollout_data(0, args, data)

    assert calls == [{"include_rloo_diagnostics": False}]
