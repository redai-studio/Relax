# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Tests for rollout stop reason and turn count metrics."""

import copy
from types import SimpleNamespace

import numpy as np
import pytest

from relax.utils.metrics.metric_utils import compute_stop_reason_and_num_turn_metrics
from relax.utils.types import Sample


def _samples(*metadatas):
    return [Sample(metadata=dict(metadata)) for metadata in metadatas]


def _stop_reasons(metrics):
    return {key: value for key, value in metrics.items() if key.startswith("stop_reason/")}


@pytest.mark.filterwarnings("error")
def test_stop_reason_metrics_empty_input_returns_empty_dict():
    assert compute_stop_reason_and_num_turn_metrics([]) == {}


@pytest.mark.parametrize(
    ("metadata", "expected"),
    [
        ({}, "unknown"),
        ({"rollout_turns": 2}, "unknown"),
        ({"rollout_stop_reason": None}, "unknown"),
        ({"rollout_stop_reason": ""}, "unknown"),
        ({"rollout_stop_reason": "   "}, "unknown"),
        ({"stop_reason": ":detail"}, "unknown"),
        ({"rollout_stop_reason": " finish_abort "}, "finish_abort"),
        ({"stop_reason": "env_done"}, "env_done"),
        ({"stop_reason": "env_error:timeout"}, "env_error"),
        ({"stop_reason": " env_error : connection refused"}, "env_error"),
        ({"rollout_stop_reason": "finish_abort", "stop_reason": "env_done"}, "finish_abort"),
        ({"rollout_stop_reason": "  ", "stop_reason": "env_done"}, "env_done"),
        ({"rollout_stop_reason": None, "stop_reason": "max_turns"}, "max_turns"),
        # The canonical field is reported as is; only ``stop_reason`` drops details.
        ({"rollout_stop_reason": "canonical:value", "stop_reason": "env_error:timeout"}, "canonical:value"),
    ],
)
def test_stop_reason_metrics_resolves_reason(metadata, expected):
    metrics = compute_stop_reason_and_num_turn_metrics(_samples(metadata))

    assert _stop_reasons(metrics) == {f"stop_reason/{expected}/count": 1, f"stop_reason/{expected}/ratio": 1.0}


def test_stop_reason_metrics_single_sample():
    metrics = compute_stop_reason_and_num_turn_metrics(_samples({"stop_reason": "env_done", "rollout_turns": 3}))

    assert metrics == {
        "stop_reason/env_done/count": 1,
        "stop_reason/env_done/ratio": 1.0,
        "num_turn/mean": 3.0,
        "num_turn/max": 3,
        "num_turn/min": 3,
        "num_turn/p50": 3.0,
        "num_turn/p90": 3.0,
        "num_turn/p95": 3.0,
        "num_turn/p99": 3.0,
    }


def test_stop_reason_metrics_counts_and_ratios_cover_all_samples():
    samples = _samples(
        *[{"rollout_stop_reason": "env_done"}] * 5,
        *[{"stop_reason": "max_turns"}] * 3,
        *[{}] * 2,
    )

    metrics = _stop_reasons(compute_stop_reason_and_num_turn_metrics(samples))

    assert metrics == {
        "stop_reason/env_done/count": 5,
        "stop_reason/env_done/ratio": pytest.approx(0.5),
        "stop_reason/max_turns/count": 3,
        "stop_reason/max_turns/ratio": pytest.approx(0.3),
        "stop_reason/unknown/count": 2,
        "stop_reason/unknown/ratio": pytest.approx(0.2),
    }
    assert sum(v for k, v in metrics.items() if k.endswith("/count")) == len(samples)
    assert sum(v for k, v in metrics.items() if k.endswith("/ratio")) == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("num_turns", "expected"),
    [
        ([1, 2], (1.5, 1.9, 1.95, 1.99)),
        (list(range(10, 0, -1)), (5.5, 9.1, 9.55, 9.91)),
    ],
)
def test_stop_reason_metrics_turn_percentiles(num_turns, expected):
    metrics = compute_stop_reason_and_num_turn_metrics(_samples(*({"rollout_turns": turns} for turns in num_turns)))

    assert tuple(metrics[f"num_turn/p{p}"] for p in (50, 90, 95, 99)) == pytest.approx(expected)


def test_stop_reason_metrics_missing_turns_default_to_one():
    metrics = compute_stop_reason_and_num_turn_metrics(_samples({"rollout_turns": 5}, {}, {}))

    # Turn counts are [5, 1, 1].
    assert (metrics["num_turn/min"], metrics["num_turn/max"]) == (1, 5)
    assert metrics["num_turn/p50"] == pytest.approx(1.0)
    assert metrics["num_turn/p90"] == pytest.approx(4.2)


@pytest.mark.parametrize(
    ("metadatas", "expected"),
    [
        ([{"rollout_turns": t} for t in (1, 2, 7, 10)], {"min": 1, "mean": 5.0, "max": 10}),
        ([{"rollout_turns": np.int64(4)}, {}], {"min": 1, "mean": 2.5, "max": 4}),
    ],
)
def test_stop_reason_metrics_keep_legacy_min_mean_max(metadatas, expected):
    samples = _samples(*metadatas)
    # The expressions previously inlined in ``compute_metrics_from_samples``.
    legacy_turns = [s.metadata.get("rollout_turns", 1) for s in samples]
    legacy = {
        "min": np.min(legacy_turns).item(),
        "mean": np.mean(legacy_turns).item(),
        "max": np.max(legacy_turns).item(),
    }

    metrics = compute_stop_reason_and_num_turn_metrics(samples)

    assert legacy == expected
    for stat, value in legacy.items():
        assert metrics[f"num_turn/{stat}"] == value
        assert type(metrics[f"num_turn/{stat}"]) is type(value)


def test_stop_reason_metrics_does_not_mutate_samples():
    samples = _samples({"rollout_stop_reason": " finish_abort ", "rollout_turns": 2}, {"stop_reason": "env_error:x"})
    before = copy.deepcopy(samples)

    compute_stop_reason_and_num_turn_metrics(samples)

    assert samples == before


@pytest.fixture
def rollout_module():
    return pytest.importorskip("relax.distributed.ray.rollout")


_ARGS = SimpleNamespace(
    log_reward_category=None,
    advantage_estimator="grpo",
    reward_key=None,
    log_passrate=False,
    sglang_speculative_algorithm=None,
    partial_rollout=False,
    fully_async=False,
)


def test_compute_metrics_from_samples_empty_input_returns_empty_dict(rollout_module):
    assert rollout_module.compute_metrics_from_samples(_ARGS, []) == {}


def test_compute_metrics_from_samples_includes_stop_reason_and_turn_metrics(rollout_module):
    samples = [
        Sample(group_index=0, reward=1.0, metadata={"stop_reason": "env_done", "rollout_turns": 1}),
        Sample(group_index=0, reward=0.0, metadata={"rollout_turns": 2}),
    ]

    metrics = rollout_module.compute_metrics_from_samples(_ARGS, samples, include_rloo_diagnostics=False)

    assert metrics["stop_reason/env_done/count"] == 1
    assert metrics["stop_reason/unknown/ratio"] == pytest.approx(0.5)
    assert (metrics["num_turn/min"], metrics["num_turn/mean"], metrics["num_turn/max"]) == (1, 1.5, 2)
    assert tuple(metrics[f"num_turn/p{p}"] for p in (50, 90, 95, 99)) == pytest.approx((1.5, 1.9, 1.95, 1.99))
    assert metrics["raw_reward"] == pytest.approx(0.5)
