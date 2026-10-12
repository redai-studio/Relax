# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import json

import pytest

from relax.utils.speculative import SPEC_TOKEN_COUNT_KEYS, SpeculativeCounts, SpeculativeGeneration
from relax.utils.types import Sample, get_spec_token_counts


@pytest.mark.parametrize("keys", SPEC_TOKEN_COUNT_KEYS)
def test_speculative_counts_aliases(keys: tuple[str, str]) -> None:
    meta = {keys[0]: 0, keys[1]: 2, "spec_verify_ct": 1, "completion_tokens": 1}
    assert SpeculativeCounts.from_meta_info(meta) == SpeculativeCounts(0, 2, 1, 1)
    assert get_spec_token_counts(meta) == (0, 2)


def test_speculative_counts_missing_is_not_zero() -> None:
    assert SpeculativeCounts.from_meta_info({}) == SpeculativeCounts()
    assert SpeculativeCounts.from_meta_info({"spec_accepted_drafts": 0}).accepted == 0
    assert SpeculativeCounts.from_meta_info({"spec_accepted_drafts": 0}).proposed is None
    counts = SpeculativeCounts.from_meta_info({"spec_verify_ct": None, "completion_tokens": 0})
    assert counts.verify is None
    assert counts.completion == 0


def test_speculative_counts_attempts_preserve_incompleteness() -> None:
    complete = SpeculativeCounts(1, 2, 1, 2)
    partial = SpeculativeCounts(None, None, 2, 3)
    assert complete.plus(complete) == SpeculativeCounts(2, 4, 2, 4)
    assert complete.plus(partial) == SpeculativeCounts(None, None, 3, 5)
    assert partial.plus(complete) == complete.plus(partial)


@pytest.mark.parametrize("value", [None, -1, True, 1.5, "invalid"])
def test_speculative_counts_invalid_is_unknown(value: object) -> None:
    assert SpeculativeCounts.from_meta_info({"spec_verify_ct": value}).verify is None


def test_speculative_counts_sample_serialization_and_legacy() -> None:
    sample = Sample()
    sample.spec_info.add({"spec_accepted_drafts": 0, "spec_proposed_drafts": 2})
    restored = Sample.from_dict(json.loads(json.dumps(sample.to_dict())))
    assert restored.spec_info.counts == SpeculativeCounts(0, 2, None, None)
    assert restored.spec_info.spec_accept_token_num == 0
    legacy = Sample.from_dict({"status": "completed", "spec_info": {"spec_draft_token_num": 10}})
    assert legacy.spec_info.spec_draft_token_num == 10
    assert legacy.spec_info.counts is None
    assert Sample.from_dict({"status": "completed"}).spec_info.counts is None
    assert SpeculativeCounts.from_dict({"version": 99, "accepted": 2}).accepted is None


def test_speculative_counts_legacy_resume_cannot_recover_availability() -> None:
    legacy = Sample.from_dict({"status": "completed", "spec_info": {"spec_draft_token_num": 0}})
    legacy.spec_info.add({"spec_accepted_drafts": 9, "spec_proposed_drafts": 10})
    assert legacy.spec_info.legacy_counts_availability is None
    assert legacy.spec_info.counts is None
    assert legacy.spec_info.legacy_counts
    restored = Sample.from_dict(legacy.to_dict())
    restored.spec_info.add({"spec_accepted_drafts": 1, "spec_proposed_drafts": 2})
    assert restored.spec_info.counts is None


def test_speculative_counts_add_normalizes_strings_and_invalid_values() -> None:
    info = Sample.SpecInfo()
    info.add(
        {
            "spec_accepted_drafts": "bad",
            "spec_proposed_drafts": "10",
            "spec_verify_ct": "2",
            "completion_tokens": "5",
        }
    )
    assert info.counts == SpeculativeCounts(None, 10, 2, 5)
    assert info.spec_verify_ct == 2
    assert info.completion_token_num == 5


def test_speculative_counts_alias_pairs_are_not_combined() -> None:
    counts = SpeculativeCounts.from_meta_info({"spec_num_correct_drafts": 1, "spec_proposed_drafts": 10})
    assert counts == SpeculativeCounts(1, None, None, None)
    counts = SpeculativeCounts.from_meta_info(
        {"spec_num_correct_drafts": 1, "spec_accepted_drafts": 9, "spec_proposed_drafts": 10}
    )
    assert counts == SpeculativeCounts(9, 10, None, None)


def test_speculative_counts_legacy_null_counters_remain_unavailable() -> None:
    sample = Sample.from_dict(
        {"status": "completed", "spec_info": {"spec_accept_token_num": None, "spec_draft_token_num": "10"}}
    )
    assert sample.spec_info.spec_draft_token_num == 10
    assert sample.spec_info.counts is None
    sample.spec_info.add({"spec_accepted_drafts": 1, "spec_proposed_drafts": 2})
    assert sample.spec_info.counts is None


def test_speculative_counts_sample_generation_serialization_is_isolated() -> None:
    sample = Sample(
        session_id="session",
        spec_generations=[
            SpeculativeGeneration(
                "session",
                "request",
                "state",
                SpeculativeCounts(1, 2, 1, 2),
            ).to_dict()
        ],
    )

    payload = sample.to_dict()

    assert "spec_generation" not in payload
    assert payload["spec_generations"] == sample.spec_generations
    assert payload["spec_generations"] is not sample.spec_generations

    payload["spec_generations"][0]["counts"]["accepted"] = 999

    assert sample.spec_generations[0]["counts"]["accepted"] == 1

    restored = Sample.from_dict(sample.to_dict())

    assert restored.spec_generations == sample.spec_generations
    assert not hasattr(restored, "spec_generation")


def test_speculative_counts_legacy_field_availability_survives_roundtrip() -> None:
    sample = Sample.from_dict(
        {
            "status": "completed",
            "spec_info": {
                "spec_draft_token_num": 10,
                "spec_verify_ct": 2,
                "completion_token_num": None,
            },
        }
    )

    assert sample.spec_info.legacy_counts_availability == SpeculativeCounts(
        None,
        10,
        2,
        None,
    )

    restored = Sample.from_dict(sample.to_dict())

    assert restored.spec_info.legacy_counts_availability == SpeculativeCounts(
        None,
        10,
        2,
        None,
    )


def test_speculative_counts_migrated_legacy_without_availability_stays_unknown() -> None:
    sample = Sample.from_dict(
        {
            "status": "completed",
            "spec_info": {
                "spec_accept_token_num": 0,
                "spec_draft_token_num": 10,
                "spec_verify_ct": 0,
                "completion_token_num": 0,
                "counts": None,
                "legacy_counts": True,
            },
        }
    )

    assert sample.spec_info.legacy_counts
    assert sample.spec_info.legacy_counts_availability is None
