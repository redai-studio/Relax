# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Tests for rollout stop-reason distribution and num-turn quantile metrics."""

import copy
import math
from types import SimpleNamespace

import numpy as np
import pytest

from relax.utils.metrics.metric_utils import compute_num_turn_metrics, compute_stop_reason_metrics
from relax.utils.types import Sample


def _sample(metadata=None, **kwargs):
    return Sample(metadata=metadata, **kwargs)


def _task_issue_samples():
    """The four-sample example from the task issue (#326)."""
    return [
        _sample({"rollout_stop_reason": "env_done", "rollout_turns": 1}),
        _sample({"stop_reason": "env_done", "rollout_turns": 2}),
        _sample({"rollout_stop_reason": "max_turns", "rollout_turns": 4}),
        _sample({}),
    ]


class TestNumTurnAggregation:
    @pytest.mark.filterwarnings("error")
    def test_empty_input_returns_empty_dict(self):
        assert compute_num_turn_metrics([]) == {}

    @pytest.mark.parametrize("turns", [5, 0, np.int64(3)])
    def test_single_sample_reports_normalized_turns_everywhere(self, turns):
        metrics = compute_num_turn_metrics([_sample({"rollout_turns": turns})])

        assert set(metrics) == {
            "num_turn/mean",
            "num_turn/max",
            "num_turn/min",
            "num_turn/p50",
            "num_turn/p90",
            "num_turn/p95",
            "num_turn/p99",
        }
        assert all(value == int(turns) for value in metrics.values())
        assert all(isinstance(value, (int, float)) for value in metrics.values())

    def test_task_issue_example_quantiles(self):
        metrics = compute_num_turn_metrics(_task_issue_samples())

        assert metrics["num_turn/min"] == 1
        assert metrics["num_turn/max"] == 4
        assert metrics["num_turn/mean"] == pytest.approx(2.0)
        assert metrics["num_turn/p50"] == pytest.approx(1.5)
        assert metrics["num_turn/p90"] == pytest.approx(3.4)
        assert metrics["num_turn/p95"] == pytest.approx(3.7)
        assert metrics["num_turn/p99"] == pytest.approx(3.94)

    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            (0, 0),
            (1, 1),
            (np.int64(3), 3),
            (2.0, 2),
            (np.float64(4.0), 4),
            (None, 1),
            (True, 1),
            (False, 1),
            (np.bool_(True), 1),
            ("3", 1),
            (-3, 1),
            (-1.0, 1),
            (2.5, 1),
            (float("nan"), 1),
            (float("inf"), 1),
            ([1], 1),
            ({"turns": 1}, 1),
        ],
    )
    def test_turn_normalization_table(self, value, expected):
        metrics = compute_num_turn_metrics([_sample({"rollout_turns": value})])

        assert metrics["num_turn/min"] == expected
        assert metrics["num_turn/max"] == expected
        assert metrics["num_turn/mean"] == expected
        assert metrics["num_turn/p50"] == expected
        assert metrics["num_turn/p99"] == expected

    def test_missing_or_none_metadata_defaults_to_one(self):
        metrics = compute_num_turn_metrics([_sample({}), _sample(None)])

        assert all(value == 1 for value in metrics.values())


class TestStopReasonAggregation:
    @pytest.mark.filterwarnings("error")
    def test_empty_input_returns_empty_dict(self):
        assert compute_stop_reason_metrics([]) == {}

    def test_task_issue_example_distribution(self):
        metrics = compute_stop_reason_metrics(_task_issue_samples())

        assert metrics == {
            "stop_reason/env_done/count": 2.0,
            "stop_reason/env_done/ratio": 0.5,
            "stop_reason/max_turns/count": 1.0,
            "stop_reason/max_turns/ratio": 0.25,
            "stop_reason/unknown/count": 1.0,
            "stop_reason/unknown/ratio": 0.25,
        }

    @pytest.mark.parametrize("reason", ["env_done", "unknown"])
    def test_single_sample_ratio_is_one(self, reason):
        metrics = compute_stop_reason_metrics([_sample({"rollout_stop_reason": reason})])

        assert metrics == {f"stop_reason/{reason}/count": 1.0, f"stop_reason/{reason}/ratio": 1.0}

    def test_legacy_field_takes_precedence_over_compat_field(self):
        sample = _sample({"rollout_stop_reason": "env_done", "stop_reason": "max_turns"})

        metrics = compute_stop_reason_metrics([sample])

        assert set(metrics) == {"stop_reason/env_done/count", "stop_reason/env_done/ratio"}

    @pytest.mark.parametrize("invalid", [None, "", "   ", 42, True])
    def test_invalid_legacy_falls_back_to_compat_field(self, invalid):
        sample = _sample({"rollout_stop_reason": invalid, "stop_reason": "finish_length"})

        metrics = compute_stop_reason_metrics([sample])

        assert set(metrics) == {"stop_reason/finish_length/count", "stop_reason/finish_length/ratio"}

    def test_whitespace_stripped_and_case_preserved(self):
        sample = _sample({"rollout_stop_reason": "  Max_Turns "})

        metrics = compute_stop_reason_metrics([sample])

        assert set(metrics) == {"stop_reason/Max_Turns/count", "stop_reason/Max_Turns/ratio"}

    def test_non_string_compat_value_buckets_unknown(self):
        sample = _sample({"rollout_stop_reason": 42, "stop_reason": 7})

        metrics = compute_stop_reason_metrics([sample])

        assert set(metrics) == {"stop_reason/unknown/count", "stop_reason/unknown/ratio"}

    def test_explicit_unknown_does_not_fall_back(self):
        sample = _sample({"rollout_stop_reason": "unknown", "stop_reason": "env_done"})

        metrics = compute_stop_reason_metrics([sample])

        assert set(metrics) == {"stop_reason/unknown/count", "stop_reason/unknown/ratio"}

    @pytest.mark.parametrize("metadata", [{}, None, {"other": 1}])
    def test_missing_metadata_buckets_unknown(self, metadata):
        metrics = compute_stop_reason_metrics([_sample(metadata)])

        assert metrics == {"stop_reason/unknown/count": 1.0, "stop_reason/unknown/ratio": 1.0}

    def test_denominator_ignores_reward_and_status(self):
        samples = [
            _sample({"rollout_stop_reason": "env_done"}, reward=1.0, status=Sample.Status.COMPLETED),
            _sample({"stop_reason": "max_turns"}, reward=None, status=Sample.Status.FAILED),
            _sample({}, reward=None, status=Sample.Status.TRUNCATED),
        ]

        metrics = compute_stop_reason_metrics(samples)

        assert metrics["stop_reason/env_done/ratio"] == pytest.approx(1 / 3)
        assert metrics["stop_reason/max_turns/ratio"] == pytest.approx(1 / 3)
        assert metrics["stop_reason/unknown/ratio"] == pytest.approx(1 / 3)
        assert sum(value for key, value in metrics.items() if key.endswith("/count")) == 3.0


class TestPurityAndIdentities:
    def test_inputs_are_not_mutated(self):
        samples = _task_issue_samples()
        snapshot = copy.deepcopy([s.metadata for s in samples])
        metadata_ids = [id(s.metadata) for s in samples]

        compute_num_turn_metrics(samples)
        compute_stop_reason_metrics(samples)

        assert [s.metadata for s in samples] == snapshot
        assert [id(s.metadata) for s in samples] == metadata_ids

    def test_repeated_calls_and_reordering_agree(self):
        samples = _task_issue_samples()

        first = compute_stop_reason_metrics(samples)
        second = compute_stop_reason_metrics(samples)
        reordered = compute_stop_reason_metrics(list(reversed(samples)))

        assert first == second == reordered
        assert compute_num_turn_metrics(samples) == compute_num_turn_metrics(list(reversed(samples)))

    def test_no_categories_carry_over_between_batches(self):
        assert set(compute_stop_reason_metrics([_sample({"rollout_stop_reason": "env_done"})])) == {
            "stop_reason/env_done/count",
            "stop_reason/env_done/ratio",
        }

        metrics = compute_stop_reason_metrics([_sample({"rollout_stop_reason": "max_turns"})])

        assert set(metrics) == {"stop_reason/max_turns/count", "stop_reason/max_turns/ratio"}

    def test_aggregation_identities(self):
        samples = _task_issue_samples() + [_sample({"stop_reason": "env_done"}, status=Sample.Status.ABORTED)]

        turn_metrics = compute_num_turn_metrics(samples)
        reason_metrics = compute_stop_reason_metrics(samples)

        counts = [value for key, value in reason_metrics.items() if key.endswith("/count")]
        ratios = [value for key, value in reason_metrics.items() if key.endswith("/ratio")]
        assert sum(counts) == len(samples)
        assert sum(ratios) == pytest.approx(1.0)
        for value in list(turn_metrics.values()) + list(reason_metrics.values()):
            assert isinstance(value, (int, float)) and math.isfinite(value)


def _entry_args(**overrides):
    values = dict(
        log_reward_category=None,
        reward_key=None,
        log_passrate=False,
        advantage_estimator="grpo",
        partial_rollout=False,
        fully_async=False,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def _entry_samples():
    return [
        Sample(
            status=Sample.Status.COMPLETED,
            reward=1.0,
            response_length=4,
            metadata={"rollout_turns": 3, "rollout_stop_reason": "env_done"},
        ),
        Sample(
            status=Sample.Status.TRUNCATED,
            reward=0.0,
            response_length=8,
            metadata={"rollout_turns": 9},
        ),
    ]


class TestEntryIntegration:
    """Entry-point behaviour that needs the full rollout module (CI-only)."""

    @pytest.mark.filterwarnings("error")
    def test_empty_input_returns_empty_dict(self):
        pytest.importorskip("megatron.core")

        import relax.distributed.ray.rollout as rollout_module

        assert rollout_module.compute_metrics_from_samples(_entry_args(), []) == {}

    def test_reports_new_and_existing_metrics(self):
        pytest.importorskip("megatron.core")

        import relax.distributed.ray.rollout as rollout_module

        metrics = rollout_module.compute_metrics_from_samples(_entry_args(), _entry_samples())

        assert metrics["num_turn/min"] == 3
        assert metrics["num_turn/max"] == 9
        assert metrics["num_turn/mean"] == pytest.approx(6.0)
        assert metrics["num_turn/p50"] == pytest.approx(6.0)
        assert metrics["num_turn/p90"] == pytest.approx(8.4)
        assert metrics["num_turn/p95"] == pytest.approx(8.7)
        assert metrics["num_turn/p99"] == pytest.approx(8.94)
        assert metrics["stop_reason/env_done/count"] == 1.0
        assert metrics["stop_reason/env_done/ratio"] == pytest.approx(0.5)
        assert metrics["stop_reason/unknown/count"] == 1.0
        assert metrics["stop_reason/unknown/ratio"] == pytest.approx(0.5)
        assert metrics["raw_reward"] == pytest.approx(0.5)
        assert metrics["truncated_ratio"] == pytest.approx(0.5)
        assert metrics["response_len/mean"] == pytest.approx(6.0)
        assert metrics["prefix_cache_hit_rate"] == 0.0
        assert metrics["image_count/mean"] == 0.0

    def test_partial_rollout_tolerates_none_metadata(self):
        pytest.importorskip("megatron.core")

        import relax.distributed.ray.rollout as rollout_module

        samples = [Sample(metadata=None, index=2), Sample(metadata=None, index=2)]
        metrics = rollout_module.compute_metrics_from_samples(
            _entry_args(partial_rollout=True, fully_async=False),
            samples,
            rollout_id=7,
        )

        assert metrics["staleness/avg"] == 0.0
        assert metrics["staleness/max"] == 0.0
        assert metrics["staleness/min"] == 0.0
        assert metrics["global_batch_size"] == 1
        assert metrics["num_turn/mean"] == 1.0
        assert metrics["stop_reason/unknown/ratio"] == 1.0

    def test_training_and_eval_loggers_publish_prefixed_metrics(self, monkeypatch):
        pytest.importorskip("megatron.core")

        import relax.distributed.ray.rollout as rollout_module

        logged = {}
        flushed = []
        monkeypatch.setattr(rollout_module, "save_rollout_result_jsonl", lambda *_a, **_k: None)
        monkeypatch.setattr(rollout_module, "save_eval_summary_jsonl", lambda *_a, **_k: None)
        monkeypatch.setattr(rollout_module, "compute_rollout_step", lambda *_a, **_k: 0)
        monkeypatch.setattr(
            rollout_module.tracking_utils,
            "log",
            lambda _args, metrics, step_key: logged.update(metrics),
        )
        monkeypatch.setattr(rollout_module.tracking_utils, "flush_metrics", lambda _args, step: flushed.append(step))

        args = SimpleNamespace(
            custom_rollout_log_function_path=None,
            custom_eval_rollout_log_function_path=None,
            load_debug_rollout_data=False,
            rollout_num_gpus=0,
            **vars(_entry_args()),
        )

        rollout_module._log_rollout_data(7, args, _entry_samples(), None, 1.0)
        assert logged["rollout/stop_reason/env_done/ratio"] == pytest.approx(0.5)
        assert logged["rollout/num_turn/p95"] == pytest.approx(8.7)
        assert logged["rollout/num_turn/min"] == 3
        assert logged["rollout/step"] == 0
        assert flushed == [0]

        logged.clear()
        data = {"gsm8k": {"rewards": [1.0], "samples": _entry_samples()}}
        rollout_module._log_eval_rollout_data(0, args, data)
        assert logged["eval/gsm8k"] == pytest.approx(1.0)
        assert logged["eval/gsm8k/stop_reason/unknown/ratio"] == pytest.approx(0.5)
        assert logged["eval/gsm8k/num_turn/p99"] == pytest.approx(8.94)
        assert logged["eval/step"] == 0
        assert flushed == [0]
