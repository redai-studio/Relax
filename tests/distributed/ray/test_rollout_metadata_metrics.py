# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from types import SimpleNamespace
from typing import Any

import pytest
from conftest import HAS_DEPS


pytestmark = pytest.mark.skipif(not HAS_DEPS, reason="rollout module dependencies are not installed")


def _sample(
    metadata: dict[str, object] | None,
    *,
    index: int = 0,
    reward: Any = None,
    status: Any = None,
) -> Any:
    from relax.utils.types import Sample

    return Sample(
        index=index,
        metadata=metadata,
        response="",
        response_length=0,
        reward=reward,
        status=status or Sample.Status.COMPLETED,
    )


def test_rollout_metadata_metrics_reach_training_and_eval_logs(monkeypatch) -> None:
    import relax.distributed.ray.rollout as rollout_module
    from relax.utils.types import Sample

    for helper_name in (
        "compute_statistics",
        "compute_rollout_reward_metrics",
        "_compute_zero_std_metrics",
        "_compute_spec_metrics",
        "_compute_prefix_cache_metrics",
        "_compute_reward_cat_metrics",
        "compute_mopd_metrics",
        "_compute_min_mean_max_stats",
    ):
        monkeypatch.setattr(rollout_module, helper_name, lambda *args, **kwargs: {})
    monkeypatch.setattr(
        rollout_module,
        "get_sample_multimodal_stats",
        lambda sample: {"image_count": 0, "multimodal_token_count": 0},
    )
    monkeypatch.setattr(rollout_module, "has_repetition", lambda response: False)
    samples = [
        _sample(
            {"rollout_stop_reason": "stop", "rollout_turns": 1},
            index=0,
            reward=1.0,
            status=Sample.Status.COMPLETED,
        ),
        _sample(
            {"stop_reason": "finish_length", "rollout_turns": 2},
            index=1,
            status=Sample.Status.TRUNCATED,
        ),
        _sample(
            {"rollout_stop_reason": "unexpected-value", "rollout_turns": 10},
            index=2,
            reward=0.0,
            status=Sample.Status.ABORTED,
        ),
    ]
    args = SimpleNamespace(
        custom_eval_rollout_log_function_path=None,
        custom_rollout_log_function_path=None,
        fully_async=False,
        load_debug_rollout_data=False,
        log_passrate=False,
        log_reward_category=None,
        partial_rollout=False,
    )

    assert rollout_module.compute_metrics_from_samples(None, []) == {}
    metrics = rollout_module.compute_metrics_from_samples(args, samples)
    assert metrics["stop_reason/stop/count"] == 1.0
    assert metrics["stop_reason/finish_length/ratio"] == pytest.approx(1 / 3)
    assert metrics["stop_reason/unexpected-value/count"] == 1.0
    assert sum(value for key, value in metrics.items() if key.endswith("/count")) == 3.0
    assert sum(value for key, value in metrics.items() if key.endswith("/ratio")) == pytest.approx(1.0)
    assert metrics["num_turn/min"] == 1
    assert metrics["num_turn/mean"] == pytest.approx(13 / 3)
    assert metrics["num_turn/max"] == 10
    assert [metrics[f"num_turn/p{percentile}"] for percentile in (50, 90, 95, 99)] == pytest.approx(
        [2.0, 8.4, 9.2, 9.84]
    )

    logged_metrics = {}
    monkeypatch.setattr(rollout_module, "save_rollout_result_jsonl", lambda *args, **kwargs: None)
    monkeypatch.setattr(rollout_module, "save_eval_summary_jsonl", lambda *args, **kwargs: None)
    monkeypatch.setattr(rollout_module, "compute_perf_metrics_from_samples", lambda *args, **kwargs: {})
    monkeypatch.setattr(rollout_module, "compute_rollout_step", lambda *args, **kwargs: 0)
    monkeypatch.setattr(
        rollout_module.tracking_utils,
        "log",
        lambda args, metrics, **kwargs: logged_metrics.update(metrics),
    )
    flushed = []
    monkeypatch.setattr(
        rollout_module.tracking_utils,
        "flush_metrics",
        lambda args, step: flushed.append(step),
    )

    rollout_module._log_rollout_data(0, args, samples, None, 1.0)
    assert logged_metrics["rollout/stop_reason/stop/count"] == 1.0
    assert logged_metrics["rollout/stop_reason/finish_length/ratio"] == pytest.approx(1 / 3)
    assert logged_metrics["rollout/num_turn/p99"] == pytest.approx(9.84)
    assert logged_metrics["rollout/step"] == 0
    assert flushed == [0]

    logged_metrics.clear()
    rollout_module._log_eval_rollout_data(
        0,
        args,
        {"gsm8k": {"rewards": [1.0, 0.0], "samples": samples[:2]}},
    )
    assert logged_metrics["eval/gsm8k/stop_reason/stop/count"] == 1.0
    assert logged_metrics["eval/gsm8k/stop_reason/finish_length/ratio"] == pytest.approx(0.5)
    assert logged_metrics["eval/gsm8k/num_turn/p99"] == pytest.approx(1.99)
    assert logged_metrics["eval/step"] == 0
    assert flushed == [0, 0]


def test_partial_rollout_staleness_accepts_missing_metadata(monkeypatch) -> None:
    import relax.distributed.ray.rollout as rollout_module

    for helper_name in (
        "compute_statistics",
        "compute_rollout_reward_metrics",
        "_compute_zero_std_metrics",
        "_compute_spec_metrics",
        "_compute_prefix_cache_metrics",
        "_compute_reward_cat_metrics",
        "compute_mopd_metrics",
        "_compute_min_mean_max_stats",
    ):
        monkeypatch.setattr(rollout_module, helper_name, lambda *args, **kwargs: {})
    monkeypatch.setattr(
        rollout_module,
        "get_sample_multimodal_stats",
        lambda sample: {"image_count": 0, "multimodal_token_count": 0},
    )
    monkeypatch.setattr(rollout_module, "has_repetition", lambda response: False)
    args = SimpleNamespace(
        fully_async=False,
        log_reward_category=None,
        partial_rollout=True,
    )
    samples = [_sample(None, index=0), _sample({"start_rollout_id": 5}, index=1)]

    metrics = rollout_module.compute_metrics_from_samples(args, samples, rollout_id=7)

    assert metrics["staleness/avg"] == pytest.approx(1.0)
    assert metrics["staleness/min"] == 0
    assert metrics["staleness/max"] == 2
    assert metrics["global_batch_size"] == 2
