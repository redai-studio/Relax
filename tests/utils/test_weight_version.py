# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Weight-version span tracking and policy-provenance audit (task No.15)."""

from __future__ import annotations

from argparse import Namespace
from types import SimpleNamespace

import pytest

from relax.agentic.session.state import SessionForest, check_messages
from relax.utils.metrics.rollout_metrics import compute_metrics_from_samples
from relax.utils.types import Sample
from relax.utils.weight_version import (
    WeightVersionSpan,
    WeightVersionsPerCall,
    append_spans_from_meta_info,
    build_weight_version_diagnostic,
    coalesce_spans,
    compute_policy_provenance_metrics,
)


class _FakeTokenizer:
    def decode(self, token_ids, skip_special_tokens=True):
        return "".join(chr(t) if isinstance(t, int) and 32 <= t < 127 else "?" for t in token_ids)

    def encode(self, text, add_special_tokens=False):
        return [ord(c) for c in text]


def _call(*spans: tuple[str, int, int]) -> WeightVersionsPerCall:
    return WeightVersionsPerCall(spans=[WeightVersionSpan(*span) for span in spans])


def test_from_meta_info_scalar_and_per_token_spans():
    scalar = WeightVersionsPerCall.from_meta_info(
        {
            "weight_version": "7",
            "output_token_logprobs": [(-0.1, 1), (-0.2, 2), (-0.3, 3)],
        },
        output_end=10,
    )
    assert scalar.spans == [WeightVersionSpan("7", 7, 10)]

    multi = WeightVersionsPerCall.from_meta_info(
        {
            "weight_versions": [{"version": "4", "start": 0, "end": 2}, {"version": "5", "start": 2, "end": 3}],
            "output_token_logprobs": [(-0.1, 1), (-0.2, 2), (-0.3, 3)],
        },
        output_end=8,
    )
    assert multi.spans == [WeightVersionSpan("4", 5, 7), WeightVersionSpan("5", 7, 8)]


def test_append_spans_preserves_prefix_across_partial_resume_and_skips_empty_abort():
    spans: list[WeightVersionSpan] = []
    append_spans_from_meta_info(
        spans,
        meta_info={"weight_version": "3"},
        pending_len_before=0,
        num_new_tokens=2,
    )
    # Abort with no new tokens must not clear or invent spans.
    append_spans_from_meta_info(
        spans,
        meta_info={"weight_version": "99"},
        pending_len_before=2,
        num_new_tokens=0,
    )
    append_spans_from_meta_info(
        spans,
        meta_info={
            "weight_versions": [{"version": "5", "start": 0, "end": 1}, {"version": "6", "start": 1, "end": 3}],
        },
        pending_len_before=2,
        num_new_tokens=3,
    )
    assert coalesce_spans(spans) == [
        WeightVersionSpan("3", 0, 2),
        WeightVersionSpan("5", 2, 3),
        WeightVersionSpan("6", 3, 5),
    ]


def test_unknown_versions_are_not_treated_as_zero_or_latest():
    sample = Sample(
        index=0,
        group_index=0,
        weight_versions=[_call(("default", 0, 2), ("4", 2, 5))],
    )
    stats_metrics = compute_policy_provenance_metrics([sample], current_version=10)
    assert stats_metrics["avg_staleness"] == 6
    assert stats_metrics["avg_post_generation_staleness"] == 6
    assert stats_metrics["avg_generation_version_span"] == 0
    assert stats_metrics["token_weighted_staleness"] == 6
    assert stats_metrics["unknown_version_span_count"] == 1
    assert stats_metrics["weight_version_missing_rate"] == 0.0


def test_missing_version_coverage_without_inventing_lag():
    stamped = Sample(index=0, group_index=1, weight_versions=[_call(("7", 1, 4))])
    missing = Sample(index=1, group_index=1, weight_versions=[WeightVersionsPerCall(spans=[])])
    metrics = compute_policy_provenance_metrics([stamped, missing], current_version=10)
    assert metrics["weight_version_sample_coverage"] == 0.5
    assert metrics["weight_version_missing_rate"] == 0.5
    assert metrics["token_weighted_staleness"] == 3
    assert metrics["avg_generation_version_span"] == 0


def test_selected_policy_provenance_separates_generation_span_and_wait_lag():
    # Tool wait between turns: turn0 @ v2, turn1 @ v4; consume at v10.
    sample = Sample(
        index=0,
        group_index=0,
        session_id="sess-tool-wait",
        weight_versions=[_call(("2", 3, 6)), _call(("4", 8, 10))],
        metadata={
            "agentic_trace": {
                "session_id": "sess-tool-wait",
                "leaf_state_hash": "leaf",
                "turn_count": 2,
                "turns": [
                    {"turn_idx": 0, "request_id": "r0", "resp_state_hash": "h0", "rollout_id": 1, "abort_count": 0},
                    {"turn_idx": 1, "request_id": "r1", "resp_state_hash": "h1", "rollout_id": 3, "abort_count": 0},
                ],
            }
        },
    )
    metrics = compute_policy_provenance_metrics([sample], current_version=10)
    assert metrics["avg_staleness"] == 8  # 10 - 2
    assert metrics["avg_post_generation_staleness"] == 6  # 10 - 4
    assert metrics["avg_generation_version_span"] == 2  # 4 - 2
    # tokens: 3@v2 + 2@v4 → weighted lag = (3*8 + 2*6) / 5 = 7.2
    assert metrics["token_weighted_staleness"] == pytest.approx(7.2)

    report = build_weight_version_diagnostic(sample, current_version=10)
    assert report["session_id"] == "sess-tool-wait"
    assert report["generation_version_span"] == 2
    assert report["post_generation_staleness"] == 6
    assert [seg["version"] for seg in report["segments"]] == ["2", "4"]
    assert report["segments"][0]["request_id"] == "r0"
    assert report["segments"][1]["consume_lag"] == 6


def test_cancelled_prefix_not_committed_does_not_enter_sample_stats():
    forest = SessionForest.create_empty(session_id="sess-cancel", group_index=0, index=1)
    obs = forest.append_obs(
        parent_state_hash=forest.root_state_hash,
        rollout_id=0,
        abort_count=0,
        messages_delta=check_messages([{"role": "user", "content": "hi"}]),
        train_token_delta=[104, 105],
        rollout_token_delta=[104, 105],
    )
    # Only the successfully committed resp is exported; a cancelled IR never reaches append_resp.
    leaf = forest.append_resp(
        parent_state_hash=obs.state_hash,
        rollout_id=1,
        abort_count=1,
        messages_delta=[{"role": "assistant", "content": "ok"}],
        token_delta=[111, 107],
        logprob_delta=[-0.1, -0.2],
        weight_version_delta=[WeightVersionSpan("8", 0, 2)],
        status="completed",
        export_metadata_patch={"request_id": "final"},
    )
    sample = forest.build_sample(leaf_state_hash=leaf.state_hash, tokenizer=_FakeTokenizer())
    assert len(sample.weight_versions) == 1
    assert sample.weight_versions[0].spans == [WeightVersionSpan("8", 2, 4)]
    assert sample.tokens == [104, 105, 111, 107]
    assert sample.metadata["agentic_trace"]["turns"][0]["weight_version_spans"][0]["abs_start"] == 2


def test_build_sample_anchors_multi_turn_spans_across_tool_observation():
    forest = SessionForest.create_empty(session_id="sess-mt", group_index=0, index=2)
    obs = forest.append_obs(
        parent_state_hash=forest.root_state_hash,
        rollout_id=0,
        abort_count=0,
        messages_delta=check_messages([{"role": "user", "content": "p"}]),
        train_token_delta=[10, 11],
        rollout_token_delta=[10, 11],
    )
    first = forest.append_resp(
        parent_state_hash=obs.state_hash,
        rollout_id=1,
        abort_count=0,
        messages_delta=[{"role": "assistant", "content": "a"}],
        token_delta=[97, 98],
        logprob_delta=[-0.1, -0.2],
        weight_version_delta=[WeightVersionSpan("2", 0, 2)],
        status="completed",
        export_metadata_patch={"request_id": "t0"},
    )
    tool = forest.append_obs(
        parent_state_hash=first.state_hash,
        rollout_id=1,
        abort_count=0,
        messages_delta=[{"role": "tool", "tool_call_id": "c", "content": "tool"}],
        train_token_delta=[20, 21, 22],
        rollout_token_delta=[20, 21, 22],
    )
    # Two published weight updates landed during the tool wait: next turn stamps v4/v5.
    leaf = forest.append_resp(
        parent_state_hash=tool.state_hash,
        rollout_id=3,
        abort_count=0,
        messages_delta=[{"role": "assistant", "content": "b"}],
        token_delta=[99, 100, 101],
        logprob_delta=[-0.3, -0.4, -0.5],
        weight_version_delta=[
            WeightVersionSpan("4", 0, 1),
            WeightVersionSpan("5", 1, 3),
        ],
        status="completed",
        export_metadata_patch={"request_id": "t1"},
    )
    sample = forest.build_sample(leaf_state_hash=leaf.state_hash, tokenizer=_FakeTokenizer())
    assert [[(s.version, s.abs_start, s.abs_end) for s in call.spans] for call in sample.weight_versions] == [
        [("2", 2, 4)],
        [("4", 7, 8), ("5", 8, 10)],
    ]
    report = build_weight_version_diagnostic(sample, current_version=7)
    assert report["generation_version_span"] == 3  # 5 - 2
    assert report["post_generation_staleness"] == 2  # 7 - 5
    assert report["staleness"] == 5  # 7 - 2
    assert report["segments"][0]["turn_idx"] == 0
    assert report["segments"][-1]["version"] == "5"


def test_sample_update_from_meta_info_and_round_trip():
    sample = Sample(tokens=[1, 2, 3, 4, 5], response_length=3)
    sample.update_from_meta_info(
        SimpleNamespace(sglang_speculative_algorithm=None),
        {
            "finish_reason": {"type": "stop"},
            "weight_versions": [{"version": "9", "start": 0, "end": 3}],
            "output_token_logprobs": [(-0.1, 3), (-0.2, 4), (-0.3, 5)],
            "cached_tokens": 0,
            "prompt_tokens": 2,
        },
    )
    assert sample.weight_versions == [_call(("9", 2, 5))]
    restored = Sample.from_dict(sample.to_dict())
    assert restored.weight_versions == sample.weight_versions


def test_rollout_metrics_emit_policy_provenance_prefix():
    samples = [
        Sample(index=0, group_index=0, response_length=2, reward=1.0, weight_versions=[_call(("2", 0, 2), ("4", 2, 3))]),
        Sample(index=1, group_index=0, response_length=1, reward=1.0, weight_versions=[]),
    ]
    metrics = compute_metrics_from_samples(
        Namespace(
            advantage_estimator="grpo",
            log_reward_category=None,
            reward_key=None,
            log_passrate=False,
            n_samples_per_prompt=2,
            partial_rollout=False,
            fully_async=True,
            sglang_speculative_algorithm=None,
            opd_teacher_key=None,
            current_weight_version=10,
        ),
        samples,
        rollout_id=10,
    )
    assert metrics["policy_provenance/avg_staleness"] == 8
    assert metrics["policy_provenance/avg_generation_version_span"] == 2
    assert metrics["policy_provenance/weight_version_sample_coverage"] == 0.5


def test_partial_resume_coalesces_adjacent_same_version_spans():
    spans = [WeightVersionSpan("3", 0, 2)]
    append_spans_from_meta_info(
        spans,
        meta_info={"weight_version": "3"},
        pending_len_before=2,
        num_new_tokens=2,
    )
    assert coalesce_spans(spans) == [WeightVersionSpan("3", 0, 4)]
