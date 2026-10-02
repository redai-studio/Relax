# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""CPU coverage for backend metadata, committed exports and batch metrics."""

import json
from types import SimpleNamespace

from relax.agentic.pipeline import TrainingFieldArtifact
from relax.agentic.session.service import AgenticSessionShard
from relax.agentic.session.state import SessionForest
from relax.utils.metrics.speculative_metrics import compute_speculative_metrics
from relax.utils.speculative import SpeculativeCounts


class _Tokenizer:
    def decode(self, tokens, **kwargs):
        del kwargs
        return "".join(chr(token) for token in tokens)


def _forest(session_id: str = "session"):
    forest = SessionForest.create_empty(session_id=session_id)
    prompt = forest.append_obs(
        parent_state_hash=forest.root_state_hash,
        rollout_id=0,
        abort_count=0,
        messages_delta=[{"role": "user", "content": "prompt"}],
        train_token_delta=[112],
        rollout_token_delta=[112],
    )
    return forest, prompt


def _response(forest, parent, request_id, text, counts):
    request = SimpleNamespace(
        pending_weight_version_delta=[],
        pending_spec_counts=None,
        pending_spec_delta={
            "spec_accept_token_num": 0,
            "spec_draft_token_num": 0,
            "spec_verify_ct": 0,
            "completion_token_num": 0,
        },
        pending_prefix_cache_delta={"cached_tokens": 0, "total_prompt_tokens": 0},
    )
    AgenticSessionShard._accumulate_request_meta(
        request,
        meta_info={
            "spec_num_correct_drafts": counts.accepted,
            "spec_num_proposed_drafts": counts.proposed,
            "spec_verify_ct": counts.verify,
            "completion_tokens": counts.completion,
        },
    )
    return forest.append_resp(
        parent_state_hash=parent.state_hash,
        rollout_id=0,
        abort_count=0,
        messages_delta=[{"role": "assistant", "content": text}],
        token_delta=[ord(char) for char in text],
        logprob_delta=[-0.1] * len(text),
        spec_delta=request.pending_spec_delta,
        spec_counts=request.pending_spec_counts,
        export_metadata_patch={"request_id": request_id},
    )


def test_export_tracks_committed_branch_nodes_and_excludes_discarded_branch() -> None:
    forest, prompt = _forest()
    a = _response(forest, prompt, "a", "a", SpeculativeCounts(1, 2, 1, 2))
    b = _response(forest, a, "b", "b", SpeculativeCounts(9, 10, 2, 11))
    c = _response(forest, a, "c", "c", SpeculativeCounts(2, 4, 1, 3))
    _response(forest, prompt, "discarded", "x", SpeculativeCounts(100, 100, 1, 100))

    samples = [forest.build_sample(leaf_state_hash=node.state_hash, tokenizer=_Tokenizer()) for node in (b, c)]
    assert [[item["generation_id"] for item in sample.spec_generations] for sample in samples] == [
        ["a", "b"],
        ["a", "c"],
    ]
    restored_samples = []
    for sample in samples:
        restored = TrainingFieldArtifact.from_sample(sample).to_sample()
        restored = type(restored).from_dict(json.loads(json.dumps(restored.to_dict())))
        assert restored.spec_generations == sample.spec_generations
        restored_samples.append(restored)
    metrics = compute_speculative_metrics(restored_samples)
    assert metrics["spec/record_occurrence_count"] == 4
    assert metrics["spec/accept_rate"] == 12 / 16
    assert metrics["spec/tokens_per_verify"] == 16 / 4
    assert samples[0].spec_info.spec_accept_token_num == 10


def test_equal_content_requests_and_sessions_keep_distinct_identity() -> None:
    first, prompt = _forest("one")
    node = _response(first, prompt, "same", "same", SpeculativeCounts(1, 2))
    _response(first, prompt, "other", "same", SpeculativeCounts(9, 10))
    sample = first.build_sample(leaf_state_hash=node.state_hash, tokenizer=_Tokenizer())
    assert {item["generation_id"] for item in sample.spec_generations} == {"same", "other"}
    same_session_metrics = compute_speculative_metrics([sample])
    assert same_session_metrics["spec/unique_generation_count"] == 2
    assert same_session_metrics["spec/accept_rate"] == 10 / 12

    second, second_prompt = _forest("two")
    second_node = _response(second, second_prompt, "same", "same", SpeculativeCounts(9, 10))
    metrics = compute_speculative_metrics(
        [sample, second.build_sample(leaf_state_hash=second_node.state_hash, tokenizer=_Tokenizer())]
    )
    assert metrics["spec/unique_generation_count"] == 3
    assert metrics["spec/accept_rate"] == 19 / 22


def test_untracked_content_addressed_node_does_not_become_complete_later() -> None:
    forest, prompt = _forest()
    legacy = _response(forest, prompt, "", "same", SpeculativeCounts(1, 2))
    assert forest.build_sample(leaf_state_hash=legacy.state_hash, tokenizer=_Tokenizer()).spec_generations is None
    _response(forest, prompt, "known", "same", SpeculativeCounts(9, 10))
    sample = forest.build_sample(leaf_state_hash=legacy.state_hash, tokenizer=_Tokenizer())
    assert sample.spec_generations is None


def test_backend_metadata_uses_normalized_counters_for_legacy_totals() -> None:
    for invalid in (-1, "invalid"):
        request = SimpleNamespace(
            pending_weight_version_delta=[],
            pending_spec_counts=None,
            pending_spec_delta=dict.fromkeys(
                ("spec_accept_token_num", "spec_draft_token_num", "spec_verify_ct", "completion_token_num"), 0
            ),
            pending_prefix_cache_delta={"cached_tokens": 0, "total_prompt_tokens": 0},
        )
        AgenticSessionShard._accumulate_request_meta(
            request,
            meta_info={
                "spec_accept_token_num": 1,
                "spec_draft_token_num": 2,
                "spec_verify_ct": invalid,
                "completion_tokens": invalid,
            },
        )
        assert request.pending_spec_counts == SpeculativeCounts(1, 2, None, None)
        assert request.pending_spec_delta == {
            "spec_accept_token_num": 1,
            "spec_draft_token_num": 2,
            "spec_verify_ct": 0,
            "completion_token_num": 0,
        }
