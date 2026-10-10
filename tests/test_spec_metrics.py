# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from types import SimpleNamespace

import pytest

from relax.agentic.session.service import AgenticSessionShard
from relax.agentic.session.state import SessionForest
from relax.utils.metrics.speculative import compute_spec_metrics
from relax.utils.types import Sample


def _args():
    return SimpleNamespace(sglang_speculative_algorithm="EAGLE")


def _generation(
    request_id: str,
    *,
    accepted: int | None = None,
    proposed: int | None = None,
    verify: int | None = None,
    completion: int | None = None,
    resp_state_hash: str = "state",
) -> dict:
    generation = {
        "request_id": request_id,
        "resp_state_hash": resp_state_hash,
    }
    if accepted is not None:
        generation["spec_accept_token_num"] = accepted
    if proposed is not None:
        generation["spec_draft_token_num"] = proposed
    if verify is not None:
        generation["spec_verify_ct"] = verify
    if completion is not None:
        generation["completion_token_num"] = completion
    return generation


def _agentic_sample(session_id: str, generations: list[dict]) -> Sample:
    return Sample(
        session_id=session_id,
        metadata={
            "agentic_trace": {
                "session_id": session_id,
                "spec_generations": generations,
            }
        },
    )


def test_spec_metrics_weight_raw_counters_before_division() -> None:
    samples = [
        _agentic_sample(
            "session",
            [_generation("r1", accepted=1, proposed=2, verify=2, completion=3)],
        ),
        _agentic_sample(
            "session",
            [_generation("r2", accepted=9, proposed=10, verify=4, completion=9)],
        ),
    ]

    metrics = compute_spec_metrics(_args(), samples)

    assert metrics["spec_accept_rate"] == pytest.approx(10 / 12)
    assert metrics["spec_accept_length"] == pytest.approx(12 / 6)
    assert metrics["spec_accept_rate_coverage"] == 1.0
    assert metrics["spec_accept_length_coverage"] == 1.0


def test_spec_metrics_deduplicate_shared_generation_within_session() -> None:
    shared = _generation("A", accepted=1, proposed=2, verify=1, completion=2)
    sample_ab = _agentic_sample(
        "session",
        [
            shared,
            _generation("B", accepted=2, proposed=4, verify=2, completion=3),
        ],
    )
    sample_ac = _agentic_sample(
        "session",
        [
            shared,
            _generation("C", accepted=9, proposed=10, verify=3, completion=6),
        ],
    )

    metrics = compute_spec_metrics(_args(), [sample_ab, sample_ac])

    assert metrics["spec_accept_rate"] == pytest.approx(12 / 16)
    assert metrics["spec_accept_length"] == pytest.approx(11 / 6)
    assert metrics["spec_accept_rate_coverage"] == 1.0
    assert metrics["spec_accept_length_coverage"] == 1.0


def test_spec_metrics_scope_request_id_by_session() -> None:
    first = _agentic_sample(
        "session-1",
        [_generation("same-request", accepted=1, proposed=2)],
    )
    second = _agentic_sample(
        "session-2",
        [_generation("same-request", accepted=9, proposed=10)],
    )

    metrics = compute_spec_metrics(_args(), [first, second])

    assert metrics["spec_accept_rate"] == pytest.approx(10 / 12)
    assert metrics["spec_accept_rate_coverage"] == 1.0


def test_spec_metrics_report_missing_and_explicit_zero_coverage() -> None:
    sample = _agentic_sample(
        "session",
        [
            _generation("explicit-zero", accepted=0, proposed=10),
            _generation("missing"),
            _generation("zero-denominator", accepted=0, proposed=0),
        ],
    )

    metrics = compute_spec_metrics(_args(), [sample])

    assert metrics["spec_accept_rate"] == 0.0
    assert metrics["spec_accept_rate_coverage"] == pytest.approx(2 / 3)
    assert "spec_accept_length" not in metrics
    assert metrics["spec_accept_length_coverage"] == 0.0


def test_spec_metrics_do_not_fabricate_rate_for_zero_denominator() -> None:
    sample = _agentic_sample(
        "session",
        [_generation("zero-denominator", accepted=0, proposed=0, verify=0, completion=0)],
    )

    metrics = compute_spec_metrics(_args(), [sample])

    assert "spec_accept_rate" not in metrics
    assert "spec_accept_length" not in metrics
    assert metrics["spec_accept_rate_coverage"] == 1.0
    assert metrics["spec_accept_length_coverage"] == 1.0


def test_spec_metrics_empty_input_has_only_zero_coverage() -> None:
    metrics = compute_spec_metrics(_args(), [])

    assert metrics == {
        "spec_accept_rate_coverage": 0.0,
        "spec_accept_length_coverage": 0.0,
    }


def test_spec_metrics_legacy_samples_use_weighted_fallback() -> None:
    first = Sample()
    first.spec_info = Sample.SpecInfo(
        spec_accept_token_num=1,
        spec_draft_token_num=2,
        spec_verify_ct=2,
        completion_token_num=3,
    )
    second = Sample()
    second.spec_info = Sample.SpecInfo(
        spec_accept_token_num=9,
        spec_draft_token_num=10,
        spec_verify_ct=4,
        completion_token_num=9,
    )

    metrics = compute_spec_metrics(_args(), [first, second])

    assert metrics["spec_accept_rate"] == pytest.approx(10 / 12)
    assert metrics["spec_accept_length"] == pytest.approx(12 / 6)
    assert metrics["spec_accept_rate_coverage"] == 1.0
    assert metrics["spec_accept_length_coverage"] == 1.0


def test_spec_metrics_legacy_zero_defaults_are_treated_as_uncovered() -> None:
    serialized = Sample().to_dict()
    serialized["spec_info"] = {
        "spec_accept_token_num": 0,
        "spec_draft_token_num": 0,
        "spec_verify_ct": 0,
        "completion_token_num": 0,
    }
    legacy = Sample.from_dict(serialized)

    metrics = compute_spec_metrics(_args(), [legacy])

    assert "spec_accept_rate" not in metrics
    assert "spec_accept_length" not in metrics
    assert metrics["spec_accept_rate_coverage"] == 0.0
    assert metrics["spec_accept_length_coverage"] == 0.0


def test_spec_metrics_disabled_returns_no_metrics() -> None:
    args = SimpleNamespace(sglang_speculative_algorithm=None)

    assert compute_spec_metrics(args, []) == {}


class _CharTokenizer:
    def decode(self, token_ids, skip_special_tokens=False):
        return "".join(chr(token_id) for token_id in token_ids)


def _commit_agentic_generation(
    forest: SessionForest,
    *,
    parent_state_hash: str,
    request_id: str,
    text: str,
    meta_info: dict,
):
    request = SimpleNamespace(
        pending_weight_version_delta=[],
        pending_spec_delta={},
        pending_prefix_cache_delta={
            "cached_tokens": 0,
            "total_prompt_tokens": 0,
        },
    )
    AgenticSessionShard._accumulate_request_meta(request, meta_info=meta_info)

    node = forest.append_resp(
        parent_state_hash=parent_state_hash,
        rollout_id=1,
        abort_count=0,
        messages_delta=[
            {
                "role": "assistant",
                "content": [{"type": "text", "text": text}],
            }
        ],
        token_delta=[ord(char) for char in text],
        logprob_delta=[0.0] * len(text),
        status="completed",
        export_metadata_patch={
            "request_id": request_id,
            "base_state_hash": parent_state_hash,
        },
    )
    forest.commit_generation(
        request_id=request_id,
        response_state_hash=node.state_hash,
        spec_delta=request.pending_spec_delta,
    )
    return node


def test_agentic_spec_metrics_pipeline_deduplicates_shared_generation() -> None:
    forest = SessionForest.create_empty(session_id="session-integration")
    assert forest.root_state_hash is not None

    generation_a = _commit_agentic_generation(
        forest,
        parent_state_hash=forest.root_state_hash,
        request_id="A",
        text="A",
        meta_info={
            "spec_num_correct_drafts": 1,
            "spec_num_proposed_drafts": 2,
            "spec_verify_ct": 1,
            "completion_tokens": 2,
        },
    )
    generation_b = _commit_agentic_generation(
        forest,
        parent_state_hash=generation_a.state_hash,
        request_id="B",
        text="B",
        meta_info={
            "spec_num_correct_drafts": 2,
            "spec_num_proposed_drafts": 4,
            "spec_verify_ct": 2,
            "completion_tokens": 3,
        },
    )
    generation_c = _commit_agentic_generation(
        forest,
        parent_state_hash=generation_a.state_hash,
        request_id="C",
        text="C",
        meta_info={
            "spec_num_correct_drafts": 9,
            "spec_num_proposed_drafts": 10,
            "spec_verify_ct": 3,
            "completion_tokens": 6,
        },
    )

    tokenizer = _CharTokenizer()
    sample_ab = forest.build_sample(
        leaf_state_hash=generation_b.state_hash,
        tokenizer=tokenizer,
    )
    sample_ac = forest.build_sample(
        leaf_state_hash=generation_c.state_hash,
        tokenizer=tokenizer,
    )

    assert [generation["request_id"] for generation in sample_ab.metadata["agentic_trace"]["spec_generations"]] == [
        "A",
        "B",
    ]
    assert [generation["request_id"] for generation in sample_ac.metadata["agentic_trace"]["spec_generations"]] == [
        "A",
        "C",
    ]

    metrics = compute_spec_metrics(_args(), [sample_ab, sample_ac])

    assert metrics["spec_accept_rate"] == pytest.approx(12 / 16)
    assert metrics["spec_accept_length"] == pytest.approx(11 / 6)
    assert metrics["spec_accept_rate_coverage"] == 1.0
    assert metrics["spec_accept_length_coverage"] == 1.0


def test_spec_metrics_count_independent_requests_with_same_state() -> None:
    sample = _agentic_sample(
        "session",
        [
            _generation(
                "request-1",
                accepted=1,
                proposed=2,
                resp_state_hash="same-state",
            ),
            _generation(
                "request-2",
                accepted=9,
                proposed=10,
                resp_state_hash="same-state",
            ),
        ],
    )

    metrics = compute_spec_metrics(_args(), [sample])

    assert metrics["spec_accept_rate"] == pytest.approx(10 / 12)
    assert metrics["spec_accept_rate_coverage"] == 1.0


def test_agentic_spec_metrics_exclude_unexported_committed_branch() -> None:
    forest = SessionForest.create_empty(session_id="session-export-scope")
    assert forest.root_state_hash is not None

    generation_a = _commit_agentic_generation(
        forest,
        parent_state_hash=forest.root_state_hash,
        request_id="A",
        text="A",
        meta_info={
            "spec_num_correct_drafts": 1,
            "spec_num_proposed_drafts": 2,
            "spec_verify_ct": 1,
            "completion_tokens": 2,
        },
    )
    generation_b = _commit_agentic_generation(
        forest,
        parent_state_hash=generation_a.state_hash,
        request_id="B",
        text="B",
        meta_info={
            "spec_num_correct_drafts": 2,
            "spec_num_proposed_drafts": 4,
            "spec_verify_ct": 2,
            "completion_tokens": 3,
        },
    )
    _commit_agentic_generation(
        forest,
        parent_state_hash=generation_a.state_hash,
        request_id="C",
        text="C",
        meta_info={
            "spec_num_correct_drafts": 9,
            "spec_num_proposed_drafts": 10,
            "spec_verify_ct": 3,
            "completion_tokens": 6,
        },
    )

    sample_ab = forest.build_sample(
        leaf_state_hash=generation_b.state_hash,
        tokenizer=_CharTokenizer(),
    )

    assert [generation["request_id"] for generation in sample_ab.metadata["agentic_trace"]["spec_generations"]] == [
        "A",
        "B",
    ]

    metrics = compute_spec_metrics(_args(), [sample_ab])

    assert metrics["spec_accept_rate"] == pytest.approx(3 / 6)
    assert metrics["spec_accept_length"] == pytest.approx(5 / 3)
    assert metrics["spec_accept_rate_coverage"] == 1.0
    assert metrics["spec_accept_length_coverage"] == 1.0
