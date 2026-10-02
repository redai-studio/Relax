# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import copy
import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pytest

from relax.utils.rollout_stats import compute_rollout_stop_and_turn_metrics


@dataclass
class _Sample:
    metadata: dict[str, Any] | None = field(default_factory=dict)


_PERCENTILE_KEYS = ("num_turn/p50", "num_turn/p90", "num_turn/p95", "num_turn/p99")


def _reason_counts(metrics: dict[str, float | int]) -> dict[str, int]:
    return {
        key.removeprefix("stop_reason/").removesuffix("/count"): int(value)
        for key, value in metrics.items()
        if key.startswith("stop_reason/") and key.endswith("/count")
    }


def _reason_ratios(metrics: dict[str, float | int]) -> dict[str, float]:
    return {
        key.removeprefix("stop_reason/").removesuffix("/ratio"): float(value)
        for key, value in metrics.items()
        if key.startswith("stop_reason/") and key.endswith("/ratio")
    }


def test_rollout_stop_and_turn_metrics_empty_samples() -> None:
    assert compute_rollout_stop_and_turn_metrics([]) == {}


def test_rollout_stop_and_turn_metrics_single_sample() -> None:
    metrics = compute_rollout_stop_and_turn_metrics(
        [_Sample(metadata={"rollout_stop_reason": "completed", "rollout_turns": 7})]
    )

    assert _reason_counts(metrics) == {"completed": 1}
    assert _reason_ratios(metrics) == {"completed": 1.0}
    assert type(metrics["stop_reason/completed/count"]) is float
    assert type(metrics["stop_reason/completed/ratio"]) is float
    assert {key: metrics[key] for key in ("num_turn/min", "num_turn/mean", "num_turn/max")} == {
        "num_turn/min": 7,
        "num_turn/mean": 7.0,
        "num_turn/max": 7,
    }
    assert [metrics[key] for key in _PERCENTILE_KEYS] == [7.0, 7.0, 7.0, 7.0]


def test_missing_metadata_and_empty_reasons_are_unknown() -> None:
    samples = [
        _Sample(),
        _Sample(metadata=None),
        _Sample(metadata={"rollout_stop_reason": None}),
        _Sample(metadata={"rollout_stop_reason": ""}),
        _Sample(metadata={"rollout_stop_reason": "   "}),
        _Sample(metadata={"rollout_stop_reason": {"invalid": "reason"}}),
        _Sample(metadata={"rollout_stop_reason": " completed "}),
        _Sample(metadata={"rollout_stop_reason": None, "stop_reason": " fallback "}),
    ]

    metrics = compute_rollout_stop_and_turn_metrics(samples)

    assert _reason_counts(metrics) == {"unknown": 6, "completed": 1, "fallback": 1}
    assert _reason_ratios(metrics) == pytest.approx({"unknown": 0.75, "completed": 0.125, "fallback": 0.125})
    assert sum(_reason_counts(metrics).values()) == len(samples)
    assert sum(_reason_ratios(metrics).values()) == pytest.approx(1.0)
    assert metrics["num_turn/min"] == 1
    assert metrics["num_turn/mean"] == 1.0
    assert metrics["num_turn/max"] == 1


@pytest.mark.parametrize(
    ("metadata", "expected"),
    [
        ({"rollout_stop_reason": " primary ", "stop_reason": "fallback"}, "primary"),
        ({"rollout_stop_reason": None, "stop_reason": " fallback "}, "fallback"),
        ({"rollout_stop_reason": 7, "stop_reason": "fallback"}, "fallback"),
        ({"rollout_stop_reason": "unknown", "stop_reason": "fallback"}, "unknown"),
        ({"stop_reason": "Env_Done"}, "Env_Done"),
        ({"stop_reason": "env_error: timeout/request-7"}, "env_error: timeout/request-7"),
    ],
)
def test_rollout_stop_reason_precedence_and_content_are_preserved(metadata: dict[str, Any], expected: str) -> None:
    metrics = compute_rollout_stop_and_turn_metrics([_Sample(metadata=metadata)])

    assert _reason_counts(metrics) == {expected: 1}
    assert _reason_ratios(metrics) == {expected: 1.0}


def test_rollout_metrics_match_a_manually_checkable_batch() -> None:
    samples = [
        _Sample(metadata={"rollout_stop_reason": "env_done", "rollout_turns": 1}),
        _Sample(metadata={"stop_reason": "env_done", "rollout_turns": 2}),
        _Sample(metadata={"rollout_stop_reason": "max_turns", "rollout_turns": 4}),
        _Sample(metadata={}),
    ]

    metrics = compute_rollout_stop_and_turn_metrics(samples)

    assert _reason_counts(metrics) == {
        "env_done": 2,
        "max_turns": 1,
        "unknown": 1,
    }
    assert _reason_ratios(metrics) == pytest.approx(
        {
            "env_done": 0.5,
            "max_turns": 0.25,
            "unknown": 0.25,
        }
    )
    assert metrics["num_turn/min"] == 1
    assert metrics["num_turn/mean"] == pytest.approx(2.0)
    assert metrics["num_turn/max"] == 4
    assert [metrics[key] for key in _PERCENTILE_KEYS] == pytest.approx([1.5, 3.4, 3.7, 3.94])


@pytest.mark.parametrize(
    ("turn_count", "expected"),
    [
        (0, 0),
        (np.int64(2), 2),
        (3.0, 3),
        (np.float64(4.0), 4),
        (None, 1),
        (True, 1),
        ("4", 1),
        (2.5, 1),
        (-1, 1),
        (float("nan"), 1),
        (float("inf"), 1),
        ([], 1),
    ],
)
def test_rollout_turn_count_normalization(turn_count: Any, expected: int) -> None:
    metrics = compute_rollout_stop_and_turn_metrics([_Sample(metadata={"rollout_turns": turn_count})])

    assert metrics["num_turn/min"] == expected
    assert metrics["num_turn/mean"] == pytest.approx(expected)
    assert metrics["num_turn/max"] == expected
    assert [metrics[key] for key in _PERCENTILE_KEYS] == pytest.approx([expected] * 4)


def test_rollout_percentiles_handle_repeated_turn_counts() -> None:
    samples = [
        _Sample(metadata={"rollout_turns": 2}),
        _Sample(metadata={"rollout_turns": 2}),
        _Sample(metadata={"rollout_turns": 2}),
        _Sample(metadata={"rollout_turns": 10}),
    ]

    metrics = compute_rollout_stop_and_turn_metrics(samples)

    assert [metrics[key] for key in _PERCENTILE_KEYS] == pytest.approx([2.0, 7.6, 8.8, 9.76])
    assert all(type(metrics[key]) is float for key in _PERCENTILE_KEYS)


def test_rollout_metric_properties_hold_for_order_and_input_immutability() -> None:
    samples = [
        _Sample(metadata={"rollout_stop_reason": "completed", "rollout_turns": 1}),
        _Sample(metadata={"rollout_stop_reason": "length", "rollout_turns": 2}),
        _Sample(metadata={"rollout_stop_reason": "tool_calls", "rollout_turns": 3}),
        _Sample(metadata={"rollout_stop_reason": "finish_abort", "rollout_turns": 8}),
        _Sample(metadata={"rollout_stop_reason": "format_error", "rollout_turns": 21}),
        _Sample(metadata={"rollout_stop_reason": "unexpected", "rollout_turns": 55}),
    ]
    original_samples = copy.deepcopy(samples)
    metadata_ids = [id(sample.metadata) for sample in samples]

    metrics = compute_rollout_stop_and_turn_metrics(samples)
    shuffled_metrics = compute_rollout_stop_and_turn_metrics(list(reversed(samples)))
    repeated_call_metrics = compute_rollout_stop_and_turn_metrics(samples)

    assert shuffled_metrics == metrics
    assert repeated_call_metrics == metrics
    assert sum(_reason_counts(metrics).values()) == len(samples)
    assert sum(_reason_ratios(metrics).values()) == pytest.approx(1.0)
    assert metrics["num_turn/p50"] <= metrics["num_turn/p90"] <= metrics["num_turn/p95"] <= metrics["num_turn/p99"]
    assert metrics["num_turn/min"] == 1
    assert metrics["num_turn/mean"] == pytest.approx(15.0)
    assert metrics["num_turn/max"] == 55
    assert samples == original_samples
    assert [id(sample.metadata) for sample in samples] == metadata_ids
    assert all(type(value) in (int, float) and math.isfinite(value) for value in metrics.values())

    next_batch_metrics = compute_rollout_stop_and_turn_metrics([_Sample(metadata={"rollout_stop_reason": "new"})])
    assert _reason_counts(next_batch_metrics) == {"new": 1}
    assert not any("completed" in key for key in next_batch_metrics)


def test_duplicate_batch_doubles_counts_and_preserves_ratios_and_basic_turn_stats() -> None:
    samples = [
        _Sample(metadata={"rollout_stop_reason": "completed", "rollout_turns": 1}),
        _Sample(metadata={"rollout_stop_reason": "completed", "rollout_turns": 2}),
        _Sample(metadata={"rollout_stop_reason": "length", "rollout_turns": 5}),
    ]

    metrics = compute_rollout_stop_and_turn_metrics(samples)
    duplicated_metrics = compute_rollout_stop_and_turn_metrics(samples + samples)

    assert _reason_counts(duplicated_metrics) == {
        reason: 2 * count for reason, count in _reason_counts(metrics).items()
    }
    assert _reason_ratios(duplicated_metrics) == pytest.approx(_reason_ratios(metrics))
    for key in ("num_turn/min", "num_turn/mean", "num_turn/max"):
        assert duplicated_metrics[key] == metrics[key]


def test_percentiles_use_full_batch_instead_of_averaging_worker_percentiles() -> None:
    worker_one = [_Sample(metadata={"rollout_turns": 1}), _Sample(metadata={"rollout_turns": 2})]
    worker_two = [_Sample(metadata={"rollout_turns": 3}), _Sample(metadata={"rollout_turns": 100})]

    full_batch_p90 = compute_rollout_stop_and_turn_metrics(worker_one + worker_two)["num_turn/p90"]
    averaged_worker_p90 = (
        compute_rollout_stop_and_turn_metrics(worker_one)["num_turn/p90"]
        + compute_rollout_stop_and_turn_metrics(worker_two)["num_turn/p90"]
    ) / 2

    assert full_batch_p90 == pytest.approx(70.9)
    assert averaged_worker_p90 == pytest.approx(46.1)
    assert full_batch_p90 != averaged_worker_p90


def test_rollout_stop_reason_is_preserved_and_missing_turn_count_defaults_to_one() -> None:
    sample = _Sample(
        metadata={
            "rollout_stop_reason": "finish_abort",
            "stop_reason": "ignored-fallback",
        }
    )

    metrics = compute_rollout_stop_and_turn_metrics([sample])

    assert _reason_counts(metrics) == {"finish_abort": 1}
    assert metrics["num_turn/min"] == 1
    assert metrics["num_turn/mean"] == 1.0
    assert metrics["num_turn/max"] == 1


def test_rollout_turns_does_not_fall_back_to_other_turn_fields() -> None:
    sample = _Sample(metadata={"num_turn": 8, "agentic_trace": {"turn_count": 13}})

    metrics = compute_rollout_stop_and_turn_metrics([sample])

    assert metrics["num_turn/min"] == 1
    assert metrics["num_turn/mean"] == 1.0
    assert metrics["num_turn/max"] == 1
