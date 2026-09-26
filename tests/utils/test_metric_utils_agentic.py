# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from types import SimpleNamespace

import pytest

from relax.utils.metrics.metric_utils import compute_rollout_stop_reason_and_turn_metrics
from relax.utils.types import Sample


def _sample(*, stop_reason=None, turns=None):
    metadata = {}
    if stop_reason is not None:
        metadata["rollout_stop_reason"] = stop_reason
    if turns is not None:
        metadata["rollout_turns"] = turns
    return Sample(metadata=metadata)


def test_rollout_stop_reason_and_turn_metrics_empty_input():
    assert compute_rollout_stop_reason_and_turn_metrics([]) == {}


def test_rollout_stop_reason_and_turn_metrics_single_sample():
    metrics = compute_rollout_stop_reason_and_turn_metrics([_sample(stop_reason="completed", turns=5)])

    assert metrics["stop_reason/completed/count"] == 1
    assert metrics["stop_reason/completed/ratio"] == pytest.approx(1.0)

    assert metrics["num_turn/min"] == 5
    assert metrics["num_turn/mean"] == pytest.approx(5.0)
    assert metrics["num_turn/max"] == 5
    assert metrics["num_turn/p50"] == pytest.approx(5.0)
    assert metrics["num_turn/p90"] == pytest.approx(5.0)
    assert metrics["num_turn/p95"] == pytest.approx(5.0)
    assert metrics["num_turn/p99"] == pytest.approx(5.0)


def test_rollout_stop_reason_and_turn_metrics_multiple_samples():
    samples = [
        _sample(stop_reason="completed", turns=2),
        _sample(stop_reason="completed", turns=4),
        _sample(stop_reason="max_turns", turns=10),
        _sample(),
    ]

    metadata_before = [dict(sample.metadata) for sample in samples]

    metrics = compute_rollout_stop_reason_and_turn_metrics(samples)

    assert [sample.metadata for sample in samples] == metadata_before

    assert metrics["stop_reason/completed/count"] == 2
    assert metrics["stop_reason/completed/ratio"] == pytest.approx(0.5)

    assert metrics["stop_reason/max_turns/count"] == 1
    assert metrics["stop_reason/max_turns/ratio"] == pytest.approx(0.25)

    assert metrics["stop_reason/unknown/count"] == 1
    assert metrics["stop_reason/unknown/ratio"] == pytest.approx(0.25)

    ratios = [value for key, value in metrics.items() if key.startswith("stop_reason/") and key.endswith("/ratio")]
    assert sum(ratios) == pytest.approx(1.0)

    # Missing rollout_turns preserves the existing default of 1.
    turns = [2, 4, 10, 1]
    assert metrics["num_turn/min"] == 1
    assert metrics["num_turn/mean"] == pytest.approx(sum(turns) / len(turns))
    assert metrics["num_turn/max"] == 10
    assert metrics["num_turn/p50"] == pytest.approx(3.0)
    assert metrics["num_turn/p90"] == pytest.approx(8.2)
    assert metrics["num_turn/p95"] == pytest.approx(9.1)
    assert metrics["num_turn/p99"] == pytest.approx(9.82)


@pytest.mark.parametrize("stop_reason", [None, "", "   "])
def test_rollout_stop_reason_missing_or_blank_is_unknown(stop_reason):
    sample = _sample(stop_reason=stop_reason, turns=3)

    metrics = compute_rollout_stop_reason_and_turn_metrics([sample])

    assert metrics["stop_reason/unknown/count"] == 1
    assert metrics["stop_reason/unknown/ratio"] == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("metadata", "expected_reason"),
    [
        ({"stop_reason": "env_done", "rollout_turns": 3}, "env_done"),
        ({"stop_reason": "env_error:connection refused", "rollout_turns": 3}, "env_error"),
        ({"rollout_stop_reason": "   ", "stop_reason": "max_turns", "rollout_turns": 3}, "max_turns"),
        (
            {
                "rollout_stop_reason": "finish_abort",
                "stop_reason": "env_done",
                "rollout_turns": 3,
            },
            "finish_abort",
        ),
    ],
)
def test_rollout_stop_reason_fallback_and_precedence(metadata, expected_reason):
    metrics = compute_rollout_stop_reason_and_turn_metrics([Sample(metadata=metadata)])

    assert metrics[f"stop_reason/{expected_reason}/count"] == 1
    assert metrics[f"stop_reason/{expected_reason}/ratio"] == pytest.approx(1.0)


def test_compute_metrics_from_samples_empty_input():
    pytest.importorskip("megatron.core")

    from relax.distributed.ray.rollout import compute_metrics_from_samples

    assert compute_metrics_from_samples(None, []) == {}


def test_compute_metrics_from_samples_reports_agentic_metrics(monkeypatch):
    pytest.importorskip("megatron.core")

    import relax.distributed.ray.rollout as rollout_module

    # Keep this integration test focused on stop-reason and turn metrics rather
    # than requiring the full training configuration used by unrelated metrics.
    monkeypatch.setattr(
        rollout_module,
        "compute_rollout_reward_metrics",
        lambda *_args, **_kwargs: {},
    )
    monkeypatch.setattr(
        rollout_module,
        "_compute_zero_std_metrics",
        lambda *_args, **_kwargs: {},
    )
    monkeypatch.setattr(
        rollout_module,
        "_compute_spec_metrics",
        lambda *_args, **_kwargs: {},
    )
    monkeypatch.setattr(
        rollout_module,
        "_compute_prefix_cache_metrics",
        lambda *_args, **_kwargs: {},
    )
    monkeypatch.setattr(
        rollout_module,
        "_compute_reward_cat_metrics",
        lambda *_args, **_kwargs: {},
    )
    monkeypatch.setattr(
        rollout_module,
        "compute_mopd_metrics",
        lambda *_args, **_kwargs: {},
    )

    samples = [
        Sample(
            response="a",
            response_length=1,
            reward=1.0,
            metadata={
                "stop_reason": "env_done",
                "rollout_turns": 2,
            },
        ),
        Sample(
            response="b",
            response_length=1,
            reward=1.0,
            metadata={
                "rollout_stop_reason": "max_turns",
                "rollout_turns": 6,
            },
        ),
    ]

    args = SimpleNamespace(log_reward_category=None)
    metrics = rollout_module.compute_metrics_from_samples(args, samples)

    assert metrics["stop_reason/env_done/count"] == 1
    assert metrics["stop_reason/env_done/ratio"] == pytest.approx(0.5)
    assert metrics["stop_reason/max_turns/count"] == 1
    assert metrics["stop_reason/max_turns/ratio"] == pytest.approx(0.5)

    assert metrics["num_turn/min"] == 2
    assert metrics["num_turn/mean"] == pytest.approx(4.0)
    assert metrics["num_turn/max"] == 6
    assert metrics["num_turn/p50"] == pytest.approx(4.0)
    assert metrics["num_turn/p90"] == pytest.approx(5.6)
    assert metrics["num_turn/p95"] == pytest.approx(5.8)
    assert metrics["num_turn/p99"] == pytest.approx(5.96)
