# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Unit tests for sequence-classification eval metric aggregation."""

import math

from relax.engine.sft.eval.classification import compute_classification_metrics


def test_single_label_classification_metrics():
    metrics = compute_classification_metrics(
        "single_label_classification",
        {"loss_sum": 3.0, "num_examples": 4.0, "correct": 3.0},
    )

    assert metrics == {
        "eval/loss": 0.75,
        "eval/accuracy": 0.75,
        "eval/num_examples": 4.0,
    }


def test_multi_label_classification_metrics_use_global_sufficient_statistics():
    metrics = compute_classification_metrics(
        "multi_label_classification",
        {
            "loss_sum": 2.0,
            "num_examples": 4.0,
            "tp": 3.0,
            "fp": 1.0,
            "fn": 2.0,
            "exact_match": 2.0,
        },
    )

    assert metrics["eval/loss"] == 0.5
    assert metrics["eval/micro_precision"] == 0.75
    assert metrics["eval/micro_recall"] == 0.6
    assert math.isclose(metrics["eval/micro_f1"], 2 * 0.75 * 0.6 / (0.75 + 0.6))
    assert metrics["eval/subset_accuracy"] == 0.5


def test_classification_metrics_define_zero_denominators():
    metrics = compute_classification_metrics(
        "multi_label_classification",
        {
            "loss_sum": 0.0,
            "num_examples": 0.0,
            "tp": 0.0,
            "fp": 0.0,
            "fn": 0.0,
            "exact_match": 0.0,
        },
    )

    assert all(value == 0.0 for value in metrics.values())
