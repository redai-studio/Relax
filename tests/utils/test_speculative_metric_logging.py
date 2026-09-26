# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from argparse import Namespace

import pytest

from relax.utils.metrics.speculative_metrics import compute_speculative_log_metrics
from relax.utils.speculative import SpeculativeCounts, SpeculativeGeneration
from relax.utils.types import Sample


def _sample(identity: str, counts: SpeculativeCounts) -> Sample:
    return Sample(
        session_id="session",
        spec_generations=[SpeculativeGeneration("session", identity, identity, counts).to_dict()],
    )


def test_speculative_logging_preserves_names_but_distinguishes_weighted_values() -> None:
    samples = [_sample("one", SpeculativeCounts(1, 2, 1, 2)), _sample("two", SpeculativeCounts(9, 10, 3, 18))]
    metrics = compute_speculative_log_metrics(samples)
    assert metrics["spec/accept_rate"] == pytest.approx(10 / 12)
    assert metrics["spec_accept_rate"] == pytest.approx(0.7)
    assert metrics["spec/tokens_per_verify"] == 5
    assert metrics["spec_accept_length"] == 4


def test_speculative_logging_unknown_legacy_does_not_report_zero_rate() -> None:
    legacy = Sample.from_dict({"status": "completed"})
    metrics = compute_speculative_log_metrics([legacy], enabled=True)
    assert metrics["spec/legacy_sample_count"] == 1
    assert "spec_accept_rate" not in metrics
    assert "spec/accept_rate" not in metrics
    assert "spec_accept_length" not in metrics


def test_speculative_logging_empty_and_disabled() -> None:
    assert compute_speculative_log_metrics([]) == {}
    metrics = compute_speculative_log_metrics([], enabled=True)
    assert metrics["spec/unique_generation_count"] == 0
    assert "spec/accept_rate" not in metrics


def test_speculative_logging_actual_backend_counts_override_global_flag() -> None:
    ordinary = Sample()
    ordinary.update_from_meta_info(
        Namespace(sglang_speculative_algorithm=None),
        {
            "spec_accepted_drafts": 1,
            "spec_proposed_drafts": 2,
            "spec_verify_ct": 1,
            "completion_tokens": 2,
            "finish_reason": {"type": "stop"},
        },
    )
    metrics = compute_speculative_log_metrics([ordinary])
    assert metrics["spec/sample/accept_rate"] == 0.5
    assert metrics["spec/unique_generation_count"] == 0
    assert "spec/accept_rate" not in metrics


def test_speculative_logging_mixed_sources_never_claim_full_batch_node_coverage() -> None:
    ordinary = Sample()
    ordinary.spec_info.add({"spec_accepted_drafts": 0, "spec_proposed_drafts": 2})
    legacy = Sample.from_dict({"status": "completed", "spec_info": {"spec_draft_token_num": 10}})
    metrics = compute_speculative_log_metrics([_sample("a", SpeculativeCounts(9, 10)), ordinary, legacy])
    assert metrics["spec/agentic_sample_count"] == 1
    assert metrics["spec/ordinary_sample_count"] == 1
    assert metrics["spec/legacy_sample_count"] == 1
    assert metrics["spec/accept_rate"] == 0.9
    assert metrics["spec/sample/accept_rate"] == 0
    assert "spec_accept_rate" not in metrics


def test_speculative_logging_legacy_agentic_is_visible_without_global_flag() -> None:
    sample = Sample.from_dict({"status": "completed", "metadata": {"agentic_trace": {}}})
    metrics = compute_speculative_log_metrics([sample])
    assert metrics["spec/legacy_sample_count"] == 1
    assert "spec/accept_rate" not in metrics


def test_speculative_logging_restores_legacy_ordinary_compatibility_metrics() -> None:
    legacy = Sample.from_dict(
        {
            "status": "completed",
            "spec_info": {
                "spec_accept_token_num": 1,
                "spec_draft_token_num": 2,
                "spec_verify_ct": 2,
                "completion_token_num": 6,
            },
        }
    )

    metrics = compute_speculative_log_metrics([legacy])

    assert metrics["spec/legacy_sample_count"] == 1
    assert metrics["spec_accept_rate"] == 0.5
    assert metrics["spec_accept_length"] == 3
    assert "spec/accept_rate" not in metrics
    assert "spec/sample/accept_rate" not in metrics


def test_speculative_logging_restores_legacy_zero_acceptance() -> None:
    legacy = Sample.from_dict(
        {
            "status": "completed",
            "spec_info": {
                "spec_accept_token_num": 0,
                "spec_draft_token_num": 2,
            },
        }
    )

    metrics = compute_speculative_log_metrics([legacy])

    assert metrics["spec_accept_rate"] == 0
    assert "spec_accept_length" not in metrics
    assert "spec/accept_rate" not in metrics


def test_speculative_logging_legacy_unknown_survives_roundtrip() -> None:
    sample = Sample.from_dict(
        {
            "status": "completed",
            "spec_info": {
                "spec_draft_token_num": 10,
            },
        }
    )
    restored = Sample.from_dict(sample.to_dict())

    metrics = compute_speculative_log_metrics([restored])

    assert "spec_accept_rate" not in metrics


def test_speculative_logging_does_not_restore_legacy_agentic_ratios() -> None:
    legacy = Sample.from_dict(
        {
            "status": "completed",
            "metadata": {
                "agentic_trace": {},
            },
            "spec_info": {
                "spec_accept_token_num": 9,
                "spec_draft_token_num": 10,
                "spec_verify_ct": 2,
                "completion_token_num": 12,
            },
        }
    )

    metrics = compute_speculative_log_metrics([legacy])

    assert metrics["spec/legacy_sample_count"] == 1
    assert "spec_accept_rate" not in metrics
    assert "spec_accept_length" not in metrics
    assert "spec/accept_rate" not in metrics


@pytest.mark.parametrize(
    ("spec_info", "metric_key"),
    [
        (
            {
                "spec_draft_token_num": 10,
            },
            "spec_accept_rate",
        ),
        (
            {
                "spec_accept_token_num": None,
                "spec_draft_token_num": 10,
            },
            "spec_accept_rate",
        ),
        (
            {
                "spec_verify_ct": 2,
            },
            "spec_accept_length",
        ),
        (
            {
                "spec_verify_ct": 2,
                "completion_token_num": None,
            },
            "spec_accept_length",
        ),
    ],
)
def test_speculative_logging_legacy_missing_numerator_stays_unknown(
    spec_info: dict,
    metric_key: str,
) -> None:
    sample = Sample.from_dict(
        {
            "status": "completed",
            "spec_info": spec_info,
        }
    )

    metrics = compute_speculative_log_metrics([sample])

    assert metrics["spec/legacy_sample_count"] == 1
    assert metric_key not in metrics
