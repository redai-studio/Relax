# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import copy
import json
from types import SimpleNamespace

import pytest

from relax.utils.metrics.speculative import compute_spec_metrics
from relax.utils.types import Sample, get_spec_counts


ARGS = SimpleNamespace(sglang_speculative_algorithm="EAGLE")


def make_sample(accepted=1, proposed=2, verify=1, completion=2):
    sample = Sample()
    sample.spec_info.add(
        {
            "spec_num_correct_drafts": accepted,
            "spec_num_proposed_drafts": proposed,
            "spec_verify_ct": verify,
            "completion_tokens": completion,
        }
    )
    return sample


@pytest.mark.parametrize(
    "keys",
    [
        ("spec_num_correct_drafts", "spec_num_proposed_drafts"),
        ("spec_accepted_drafts", "spec_proposed_drafts"),
        ("spec_accept_token_num", "spec_draft_token_num"),
    ],
)
def test_counter_aliases_preserve_zero(keys):
    counts = get_spec_counts({keys[0]: 0, keys[1]: 2})
    assert counts["spec_accept_token_num"] == 0
    assert counts["spec_draft_token_num"] == 2
    assert counts["spec_verify_ct"] is None


def test_missing_counter_does_not_hide_available_length():
    sample = make_sample(accepted=None)
    result = compute_spec_metrics(ARGS, [sample])
    assert "spec_accept_rate" not in result
    assert result["spec/acceptance_observed_count"] == 0
    assert result["spec_accept_length"] == 2


def test_weighted_ratios_and_round_trip():
    samples = [make_sample(), make_sample(9, 10, 2, 10)]
    samples = [Sample.from_dict(json.loads(json.dumps(s.to_dict()))) for s in samples]
    result = compute_spec_metrics(ARGS, samples)
    assert result["spec_accept_rate"] == pytest.approx(10 / 12)
    assert result["spec_accept_length"] == 4
    assert result["spec/sample_block_count"] == 2


def test_partial_attempts_stay_partial_after_more_updates_and_serialization():
    info = make_sample().spec_info
    info.add({"spec_num_proposed_drafts": 4, "spec_verify_ct": 1, "completion_tokens": 3})
    info = Sample.SpecInfo.from_dict(json.loads(json.dumps(info.to_dict())))
    info.add(
        {"spec_num_correct_drafts": 2, "spec_num_proposed_drafts": 4, "spec_verify_ct": 2, "completion_tokens": 5}
    )
    result = compute_spec_metrics(ARGS, [Sample(spec_info=info)])
    assert info.spec_accept_token_num == 3
    assert info.observed_counts()["spec_accept_token_num"] is None
    assert "spec_accept_rate" not in result
    assert result["spec_accept_length"] == 10 / 4


def test_merge_retains_missing_observations():
    target = Sample.SpecInfo()
    target.merge(make_sample().spec_info)
    assert target.observed_counts() == make_sample().spec_info.observed_counts()
    target.merge(make_sample(accepted=None).spec_info)
    assert target.observed_counts()["spec_accept_token_num"] is None


def test_resuming_zero_filled_legacy_data_does_not_invent_coverage():
    info = Sample.SpecInfo.from_dict({})
    info.add(
        {"spec_num_correct_drafts": 1, "spec_num_proposed_drafts": 2, "spec_verify_ct": 1, "completion_tokens": 2}
    )
    result = compute_spec_metrics(ARGS, [Sample(spec_info=info)])
    assert "spec_accept_rate" not in result
    assert result["spec/acceptance_observed_count"] == 0


@pytest.mark.parametrize(
    "samples,coverage", [([], 0), ([make_sample(0, 0, 0, 0)], 1), ([make_sample(None, None, None, None)], 0)]
)
def test_undefined_ratios_are_omitted(samples, coverage):
    result = compute_spec_metrics(ARGS, samples)
    assert "spec_accept_rate" not in result and "spec_accept_length" not in result
    assert result["spec/acceptance_observed_count"] == coverage
    assert result["spec/length_observed_count"] == coverage


def test_observed_zero_is_a_real_zero_rate():
    result = compute_spec_metrics(ARGS, [make_sample(0, 2)])
    assert result["spec_accept_rate"] == 0
    assert result["spec/acceptance_observed_count"] == 1


def test_ratios_use_separate_paired_populations():
    result = compute_spec_metrics(ARGS, [make_sample(1, 2, None, 100), make_sample(None, 10, 2, 6)])
    assert result["spec_accept_rate"] == 0.5
    assert result["spec_accept_length"] == 3
    assert result["spec/completion_tokens"] == 6
    assert result["spec/proposed_tokens"] == 2


def test_legacy_samples_are_explicitly_separate():
    old = Sample.from_dict(
        {
            "status": "completed",
            "spec_info": {
                "spec_accept_token_num": 9,
                "spec_draft_token_num": 10,
                "spec_verify_ct": 2,
                "completion_token_num": 10,
            },
        }
    )
    result = compute_spec_metrics(ARGS, [make_sample(), old])
    assert old.spec_generations is None
    assert result["spec_accept_rate"] == 0.5
    assert "spec_legacy_accept_rate" not in result
    assert result["spec/legacy_sample_count"] == 1
    for payload in ({}, {"spec_draft_token_num": 2}, {"spec_accept_token_num": 0, "spec_draft_token_num": 2}):
        old = Sample.from_dict({"status": "completed", "spec_info": payload})
        result = compute_spec_metrics(ARGS, [old])
        assert "spec_accept_rate" not in result and "spec_legacy_accept_rate" not in result


def test_mixed_generators_and_duplicate_records():
    record = {"session_id": "s", "request_id": "r", **make_sample().spec_info.observed_counts()}
    agentic = Sample(spec_generations=[record], spec_info=make_sample().spec_info)
    plain = make_sample(9, 10, 2, 10)
    result = compute_spec_metrics(ARGS, [agentic, copy.deepcopy(agentic), plain])
    assert result["spec_accept_rate"] == pytest.approx(10 / 12)
    assert result["spec/generation_count"] == result["spec/sample_block_count"] == 1
    assert compute_spec_metrics(ARGS, [plain, agentic]) == result
    assert compute_spec_metrics(ARGS, [agentic, plain]) == result


def test_empty_record_list_does_not_fall_back_to_sample_counts():
    sample = Sample(spec_generations=[])
    result = compute_spec_metrics(ARGS, [sample])
    assert "spec_accept_rate" not in result
    assert result["spec/sample_block_count"] == 0


def test_disabled_speculation_keeps_existing_logging_behavior():
    assert compute_spec_metrics(SimpleNamespace(sglang_speculative_algorithm=None), [make_sample()]) == {}
