# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from types import SimpleNamespace

import pytest

from relax.utils.metrics.speculative_metrics import compute_speculative_metrics
from relax.utils.speculative import SpeculativeCounts, SpeculativeGeneration


def record(identity: str, counts: SpeculativeCounts, session: str = "s") -> dict:
    return SpeculativeGeneration(session, identity, "content-" + identity, counts).to_dict()


def sample(*records: dict) -> SimpleNamespace:
    return SimpleNamespace(spec_generations=list(records))


def test_speculative_metrics_weighted_ratios() -> None:
    metrics = compute_speculative_metrics(
        [
            sample(record("a", SpeculativeCounts(1, 2, 1, 2))),
            sample(record("b", SpeculativeCounts(9, 10, 3, 18))),
        ]
    )
    assert metrics["spec/accept_rate"] == pytest.approx(10 / 12)
    assert metrics["spec/tokens_per_verify"] == 5
    assert metrics["spec/accepted_total"] == 10
    assert metrics["spec/proposed_total"] == 12


def test_speculative_metrics_shared_nodes_and_batch_local_identity() -> None:
    a = record("a", SpeculativeCounts(1, 2, 1, 2))
    b = record("b", SpeculativeCounts(9, 10, 2, 11))
    c = record("c", SpeculativeCounts(2, 4, 1, 3))
    samples = [sample(a, b), sample(a, c)]
    metrics = compute_speculative_metrics(samples)
    assert metrics["spec/unique_generation_count"] == 3
    assert metrics["spec/record_occurrence_count"] == 4
    assert metrics["spec/accept_rate"] == 0.75
    assert metrics["spec/tokens_per_verify"] == 4
    assert metrics == compute_speculative_metrics(samples)
    assert metrics == compute_speculative_metrics(list(reversed(samples)))


def test_speculative_metrics_sessions_are_separate() -> None:
    metrics = compute_speculative_metrics(
        [
            sample(record("same", SpeculativeCounts(1, 2), session="one")),
            sample(record("same", SpeculativeCounts(9, 10), session="two")),
        ]
    )
    assert metrics["spec/unique_generation_count"] == 2
    assert metrics["spec/accept_rate"] == pytest.approx(10 / 12)


def test_speculative_metrics_missing_fields_use_paired_cohorts() -> None:
    metrics = compute_speculative_metrics(
        [
            sample(record("accept", SpeculativeCounts(0, 2, None, 100))),
            sample(record("verify", SpeculativeCounts(None, 100, 2, 6))),
            sample(record("missing", SpeculativeCounts())),
        ]
    )
    assert metrics["spec/accept_rate"] == 0
    assert metrics["spec/proposed_total"] == 2
    assert metrics["spec/tokens_per_verify"] == 3
    assert metrics["spec/completion_total"] == 6
    assert metrics["spec/accept_count_coverage"] == pytest.approx(1 / 3)
    assert metrics["spec/verify_count_coverage"] == pytest.approx(1 / 3)


def test_speculative_metrics_empty_and_observed_zero_denominators() -> None:
    empty = compute_speculative_metrics([])
    assert empty["spec/unique_generation_count"] == 0
    assert "spec/accept_rate" not in empty
    assert "spec/accept_count_coverage" not in empty
    zero = compute_speculative_metrics([sample(record("zero", SpeculativeCounts(0, 0, 0, 0)))])
    assert zero["spec/accept_count_coverage"] == 1
    assert "spec/accept_rate" not in zero
    assert "spec/tokens_per_verify" not in zero


def test_speculative_metrics_conflicts_exclude_identity_regardless_of_order() -> None:
    first = record("a", SpeculativeCounts(1, 2))
    conflicting = record("a", SpeculativeCounts(2, 2))
    records = [sample(first), sample(conflicting), sample(first)]
    metrics = compute_speculative_metrics(records)
    assert metrics == compute_speculative_metrics(list(reversed(records)))
    assert metrics["spec/conflicting_generation_count"] == 1
    assert metrics["spec/accept_count_coverage"] == 0
    assert "spec/accept_rate" not in metrics


def test_speculative_metrics_legacy_and_ordinary_are_not_generation_records() -> None:
    legacy = SimpleNamespace(spec_info=SimpleNamespace(spec_accept_token_num=9, spec_draft_token_num=10))
    ordinary = SimpleNamespace(spec_info=SimpleNamespace(counts=SpeculativeCounts(1, 2, 1, 2)))
    agentic = sample(record("a", SpeculativeCounts(9, 10, 2, 11)))
    metrics = compute_speculative_metrics([legacy, ordinary, agentic])
    assert metrics["spec/legacy_sample_count"] == 1
    assert metrics["spec/ordinary_sample_count"] == 1
    assert metrics["spec/unique_generation_count"] == 1
    assert metrics["spec/accept_rate"] == 0.9
    assert metrics["spec/sample/accept_rate"] == 0.5


def test_speculative_metrics_malformed_and_foreign_session_records() -> None:
    metrics = compute_speculative_metrics(
        [
            sample({"version": 99}),
            SimpleNamespace(session_id="other", spec_generations=[record("a", SpeculativeCounts(1, 2))]),
            sample(record("valid", SpeculativeCounts(9, 10))),
        ]
    )
    assert metrics["spec/invalid_record_count"] == 2
    assert "spec/accept_count_coverage" not in metrics
    assert metrics["spec/accept_rate"] == 0.9
