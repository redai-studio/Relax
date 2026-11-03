# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import json

import pytest

from relax.utils.metrics.speculative_metrics import compute_speculative_log_metrics, compute_speculative_metrics
from relax.utils.speculative import SpeculativeCounts, SpeculativeGeneration
from relax.utils.types import Sample


def _record(identity: str, counts: SpeculativeCounts, session: str = "session") -> dict:
    return SpeculativeGeneration(session, identity, f"state-{identity}", counts).to_dict()


def _sample(*records: dict, session_id: str | None = None) -> Sample:
    return Sample(spec_generations=list(records), session_id=session_id)


def test_weighted_aggregation_and_shared_generation_deduplication() -> None:
    samples = [
        _sample(_record("a", SpeculativeCounts(1, 2, 1, 2)), _record("b", SpeculativeCounts(9, 10, 2, 11))),
        _sample(_record("a", SpeculativeCounts(1, 2, 1, 2)), _record("c", SpeculativeCounts(2, 4, 1, 3))),
    ]
    metrics = compute_speculative_metrics(samples)
    assert metrics["spec/unique_generation_count"] == 3
    assert metrics["spec/record_occurrence_count"] == 4
    assert metrics["spec/accept_rate"] == pytest.approx(12 / 16)
    assert metrics["spec/tokens_per_verify"] == pytest.approx(16 / 4)


def test_independent_sessions_with_same_generation_id_are_distinct() -> None:
    metrics = compute_speculative_metrics(
        [
            _sample(_record("same", SpeculativeCounts(1, 2), session="one")),
            _sample(_record("same", SpeculativeCounts(9, 10), session="two")),
        ]
    )
    assert metrics["spec/unique_generation_count"] == 2
    assert metrics["spec/accept_rate"] == pytest.approx(10 / 12)


def test_missing_fields_and_zero_denominators_report_coverage_without_fake_ratio() -> None:
    metrics = compute_speculative_metrics(
        [
            _sample(_record("zero", SpeculativeCounts(0, 0, 0, 0))),
            _sample(_record("partial", SpeculativeCounts(0, 2, None, 3))),
            _sample(_record("missing", SpeculativeCounts())),
        ]
    )
    assert metrics["spec/accept_count_coverage"] == pytest.approx(2 / 3)
    assert metrics["spec/verify_count_coverage"] == pytest.approx(1 / 3)
    assert metrics["spec/accept_rate"] == 0
    assert "spec/tokens_per_verify" not in metrics


def test_partial_counters_keep_totals_but_ratios_use_complete_pairs() -> None:
    samples = [
        _sample(_record("numerators-only", SpeculativeCounts(1, None, None, 3))),
        _sample(_record("denominators-only", SpeculativeCounts(None, 2, 1, None))),
    ]
    metrics = compute_speculative_metrics(samples)
    assert metrics["spec/accepted_total"] == 1
    assert metrics["spec/proposed_total"] == 2
    assert metrics["spec/completion_total"] == 3
    assert metrics["spec/verify_total"] == 1
    assert metrics["spec/accept_covered_count"] == 0
    assert metrics["spec/verify_covered_count"] == 0
    assert "spec/accept_rate" not in metrics
    assert "spec/tokens_per_verify" not in metrics

    samples.append(_sample(_record("complete", SpeculativeCounts(1, 2, 1, 2))))
    metrics = compute_speculative_metrics(samples)
    assert metrics["spec/accepted_total"] == 2
    assert metrics["spec/proposed_total"] == 4
    assert metrics["spec/completion_total"] == 5
    assert metrics["spec/verify_total"] == 2
    assert metrics["spec/accept_count_coverage"] == pytest.approx(1 / 3)
    assert metrics["spec/verify_count_coverage"] == pytest.approx(1 / 3)
    assert metrics["spec/accept_rate"] == 0.5
    assert metrics["spec/tokens_per_verify"] == 2


def test_empty_invalid_and_legacy_inputs_are_safe() -> None:
    empty = compute_speculative_metrics([])
    assert empty["spec/unique_generation_count"] == 0
    assert "spec/accept_rate" not in empty
    assert compute_speculative_log_metrics([]) == {}
    legacy = Sample(
        metadata={"agentic_trace": {"turn_count": 1}},
        spec_info=Sample.SpecInfo(spec_accept_token_num=9, spec_draft_token_num=10),
    )
    invalid = _sample({"version": 99})
    metrics = compute_speculative_metrics([legacy, invalid])
    assert metrics["spec/legacy_sample_count"] == 1
    assert metrics["spec/invalid_record_count"] == 1


def test_legacy_names_remain_arithmetic_sample_averages_for_new_samples() -> None:
    samples = [
        Sample(spec_info=Sample.SpecInfo(counts=SpeculativeCounts(1, 2, 1, 2))),
        Sample(spec_info=Sample.SpecInfo(counts=SpeculativeCounts(9, 10, 2, 11))),
    ]
    metrics = compute_speculative_log_metrics(samples)
    assert metrics["spec/sample/accept_rate"] == pytest.approx(10 / 12)
    assert metrics["spec_accept_rate"] == pytest.approx((1 / 2 + 9 / 10) / 2)
    assert metrics["spec_accept_length"] == pytest.approx((2 / 1 + 11 / 2) / 2)

    samples[1] = Sample.from_dict(
        {
            "status": "completed",
            "spec_info": {
                "spec_accept_token_num": 9,
                "spec_draft_token_num": 10,
                "spec_verify_ct": 2,
                "completion_token_num": 11,
            },
        }
    )
    mixed = compute_speculative_log_metrics(samples)
    assert mixed["spec_accept_rate"] == metrics["spec_accept_rate"]
    assert mixed["spec_accept_length"] == metrics["spec_accept_length"]
    assert mixed["spec/ordinary_sample_count"] == mixed["spec/legacy_sample_count"] == 1


@pytest.mark.parametrize(
    ("patch", "deleted", "expected_aliases"),
    [
        ({}, None, {"spec_accept_rate": 0.9, "spec_accept_length": 5.5}),
        ({}, "spec_accept_token_num", {"spec_accept_length": 5.5}),
        ({}, "spec_draft_token_num", {"spec_accept_length": 5.5}),
        ({}, "spec_verify_ct", {"spec_accept_rate": 0.9}),
        ({}, "completion_token_num", {"spec_accept_rate": 0.9}),
        (
            {"spec_accept_token_num": 0, "completion_token_num": 0},
            None,
            {"spec_accept_rate": 0.0, "spec_accept_length": 0.0},
        ),
        ({"spec_draft_token_num": 0, "spec_verify_ct": 0}, None, {}),
        ({"spec_accept_token_num": None, "completion_token_num": None}, None, {}),
    ],
    ids=[
        "complete",
        "missing-accepted",
        "missing-proposed",
        "missing-verify",
        "missing-completion",
        "explicit-zero",
        "zero-denominators",
        "null-numerators",
    ],
)
def test_old_serialized_counts_do_not_fabricate_ratios(
    patch: dict[str, int | None], deleted: str | None, expected_aliases: dict[str, float]
) -> None:
    payload = {"spec_accept_token_num": 9, "spec_draft_token_num": 10, "spec_verify_ct": 2, "completion_token_num": 11}
    payload.update(patch)
    if deleted is not None:
        del payload[deleted]
    sample = Sample.from_dict({"status": "completed", "spec_info": payload})
    for _ in range(3):
        metrics = compute_speculative_log_metrics([sample])
        assert metrics["spec/legacy_sample_count"] == 1
        assert "spec/accept_rate" not in metrics
        assert "spec/tokens_per_verify" not in metrics
        assert {key: value for key, value in metrics.items() if key.startswith("spec_")} == expected_aliases
        sample = Sample.from_dict(json.loads(json.dumps(sample.to_dict())))


def test_positive_legacy_numerator_is_visible_even_with_zero_denominator() -> None:
    sample = Sample(
        spec_info=Sample.SpecInfo(
            legacy_counts=True,
            counts=None,
            spec_accept_token_num=1,
            spec_draft_token_num=0,
            spec_verify_ct=0,
            completion_token_num=0,
        ),
    )
    metrics = compute_speculative_log_metrics([sample])
    assert metrics["spec/legacy_sample_count"] == 1
    assert "spec/accept_rate" not in metrics


def test_completion_only_metadata_does_not_enable_speculative_logging() -> None:
    sample = Sample(
        spec_info=Sample.SpecInfo(
            counts=SpeculativeCounts(None, None, None, 5),
            legacy_counts=False,
            spec_accept_token_num=0,
            spec_draft_token_num=0,
            spec_verify_ct=0,
            completion_token_num=5,
        ),
    )
    assert compute_speculative_log_metrics([sample], enabled=False) == {}


def test_enabled_speculative_logging_can_report_unknown_counters() -> None:
    sample = Sample(
        metadata={"agentic_trace": {"turn_count": 1}},
        session_id="session",
        spec_generations=[_record("request", SpeculativeCounts(None, None, None, 5))],
    )
    metrics = compute_speculative_log_metrics([sample], enabled=True)
    assert metrics["spec/unique_generation_count"] == 1
    assert "spec/accept_rate" not in metrics


def test_conflicting_generation_is_excluded_from_ratios_and_aliases() -> None:
    samples = [
        _sample(_record("same", SpeculativeCounts(1, 2, 1, 2))),
        _sample(_record("same", SpeculativeCounts(9, 10, 1, 2))),
    ]
    metrics = compute_speculative_log_metrics(samples, enabled=True)

    assert metrics["spec/conflicting_generation_count"] == 1
    assert "spec/accept_rate" not in metrics
    assert "spec_accept_rate" not in metrics
