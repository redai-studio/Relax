# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import json
from types import SimpleNamespace

from relax.utils.speculative import SPEC_TOKEN_COUNT_KEYS, SpeculativeCounts
from relax.utils.types import Sample, get_spec_token_counts


def test_aliases_and_explicit_zero_are_preserved() -> None:
    for accepted_key, proposed_key in SPEC_TOKEN_COUNT_KEYS:
        meta = {accepted_key: 0, proposed_key: 2, "spec_verify_ct": 1, "completion_tokens": 1}
        assert SpeculativeCounts.from_meta_info(meta) == SpeculativeCounts(0, 2, 1, 1)
        assert get_spec_token_counts(meta) == (0, 2)


def test_complete_alias_pair_wins_over_partial_earlier_alias() -> None:
    counts = SpeculativeCounts.from_meta_info(
        {
            "spec_num_correct_drafts": 1,
            "spec_accepted_drafts": 3,
            "spec_proposed_drafts": 4,
        }
    )
    assert counts == SpeculativeCounts(3, 4)


def test_missing_and_invalid_values_are_unknown() -> None:
    assert SpeculativeCounts.from_meta_info({}) == SpeculativeCounts()
    counts = SpeculativeCounts.from_meta_info({"spec_accepted_drafts": 0, "spec_verify_ct": None})
    assert counts.accepted == 0
    assert counts.proposed is None
    assert counts.verify is None
    assert SpeculativeCounts.from_meta_info({"spec_verify_ct": -1}).verify is None


def test_partial_attempts_do_not_turn_missing_fields_into_zero() -> None:
    complete = SpeculativeCounts(1, 2, 1, 2)
    partial = SpeculativeCounts(None, None, 2, 3)
    assert complete.plus(complete) == SpeculativeCounts(2, 4, 2, 4)
    assert complete.plus(partial) == SpeculativeCounts(None, None, 3, 5)


def test_sample_roundtrip_and_legacy_payload_are_compatible() -> None:
    sample = Sample()
    sample.spec_info.add({"spec_accepted_drafts": 0, "spec_proposed_drafts": 2})
    restored = Sample.from_dict(json.loads(json.dumps(sample.to_dict())))
    assert restored.spec_info.counts == SpeculativeCounts(0, 2, None, None)
    assert restored.spec_info.spec_accept_token_num == 0
    assert restored.spec_info.legacy_field_counts is None
    restored.spec_info.add({"spec_accepted_drafts": 1, "spec_proposed_drafts": 2})
    assert restored.spec_info.counts == SpeculativeCounts(1, 4, None, None)
    assert restored.spec_info.legacy_field_counts is None

    pending = Sample.from_dict(json.loads(json.dumps(Sample().to_dict())))
    pending.spec_info.add(
        {"spec_accept_token_num": 1, "spec_draft_token_num": 2, "spec_verify_ct": 1, "completion_tokens": 2}
    )
    assert pending.spec_info.legacy_counts is False
    assert pending.spec_info.counts == SpeculativeCounts(1, 2, 1, 2)
    assert pending.spec_info.legacy_field_counts is None

    legacy = Sample.from_dict({"status": "completed", "spec_info": {"spec_draft_token_num": 10}})
    assert legacy.spec_info.legacy_counts is True
    assert legacy.spec_info.counts is None
    assert legacy.spec_info.legacy_field_counts == SpeculativeCounts(None, 10, None, None)
    legacy.spec_info.add({"spec_accept_token_num": 1, "spec_draft_token_num": 2})
    restored = Sample.from_dict(json.loads(json.dumps(legacy.to_dict())))
    assert restored.spec_info.legacy_counts is True
    assert restored.spec_info.counts is None
    assert restored.spec_info.legacy_field_counts == SpeculativeCounts(None, 12, None, None)

    missing_snapshot = Sample.from_dict(
        {"status": "completed", "spec_info": {"legacy_counts": False, "spec_draft_token_num": 10}}
    )
    assert missing_snapshot.spec_info.legacy_counts is False
    assert missing_snapshot.spec_info.counts is None
    assert missing_snapshot.spec_info.legacy_field_counts == SpeculativeCounts(None, 10, None, None)

    legacy_runtime = Sample.SpecInfo(spec_accept_token_num=9, spec_draft_token_num=10, legacy_counts=True)
    legacy_runtime.add({"spec_accept_token_num": 1, "spec_draft_token_num": 2})
    assert legacy_runtime.counts is None
    assert legacy_runtime.spec_accept_rate == 10 / 12


def test_backend_counts_are_recorded_without_global_algorithm_flag() -> None:
    sample = Sample()
    sample.update_from_meta_info(
        SimpleNamespace(sglang_speculative_algorithm=None),
        {
            "spec_accepted_drafts": 1,
            "spec_proposed_drafts": 2,
            "spec_verify_ct": 1,
            "completion_tokens": 2,
            "finish_reason": {"type": "stop"},
        },
    )
    assert sample.spec_info.counts == SpeculativeCounts(1, 2, 1, 2)
