# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Unit tests for weight version span plumbing through session state."""

from __future__ import annotations

import asyncio

from relax.agentic.session.state import InflightRequest, RequestKind, SessionForest, check_messages
from relax.utils.types import WeightVersionSpan


def _chars(text: str) -> list[int]:
    return [ord(ch) for ch in text]


def test_ir_defaults_and_attempt_counter():
    ir = InflightRequest(
        request_id="r1",
        parent_state_hash="hash",
        rollout_id=0,
        kind=RequestKind.FRESH,
        abort_count=0,
        waiter=asyncio.get_event_loop_policy().new_event_loop().create_future(),
        wall_started_at=0.0,
    )
    assert ir.pending_weight_version_spans == []
    assert ir.weight_version_attempt_count == 0


def _forest() -> SessionForest:
    return SessionForest.create_empty(
        session_id="s1",
        group_index=2,
        index=5,
        label="gold",
        train_metadata={"loss": "grpo"},
        metadata={},
    )


def test_append_resp_commits_spans_to_resp_node():
    forest = _forest()
    obs = forest.append_obs(
        parent_state_hash=forest.root_state_hash,
        rollout_id=7,
        abort_count=0,
        messages_delta=check_messages([{"role": "user", "content": [{"type": "text", "text": "hi"}]}]),
        train_token_delta=_chars("hi"),
        rollout_token_delta=_chars("hi"),
    )
    spans = [
        WeightVersionSpan(version="1", token_start=0, token_end=1, attempt_seq=0),
        WeightVersionSpan(version=None, token_start=1, token_end=2, attempt_seq=1),
    ]
    resp = forest.append_resp(
        parent_state_hash=obs.state_hash,
        rollout_id=7,
        abort_count=0,
        messages_delta=[{"role": "assistant", "content": [{"type": "text", "text": "ab"}]}],
        token_delta=_chars("ab"),
        logprob_delta=[-0.1, -0.2],
        weight_version_delta=["1"],
        weight_version_spans=spans,
    )
    assert forest.nodes_by_hash[resp.state_hash].weight_version_spans == spans
    # obs node 不带版本（loss_mask=0，不参与统计）。
    assert forest.nodes_by_hash[obs.state_hash].weight_version_spans == []


def test_append_resp_spans_default_empty():
    forest = _forest()
    obs = forest.append_obs(
        parent_state_hash=forest.root_state_hash,
        rollout_id=7,
        abort_count=0,
        messages_delta=check_messages([{"role": "user", "content": [{"type": "text", "text": "hi"}]}]),
        train_token_delta=_chars("hi"),
        rollout_token_delta=_chars("hi"),
    )
    resp = forest.append_resp(
        parent_state_hash=obs.state_hash,
        rollout_id=7,
        abort_count=0,
        messages_delta=[{"role": "assistant", "content": [{"type": "text", "text": "ab"}]}],
        token_delta=_chars("ab"),
        logprob_delta=[-0.1, -0.2],
    )
    assert forest.nodes_by_hash[resp.state_hash].weight_version_spans == []


def _two_resp_forest():
    """root -> obs -> resp(v1, 2 token) -> obs -> resp(v3, 2 token)"""
    forest = _forest()
    obs1 = forest.append_obs(
        parent_state_hash=forest.root_state_hash,
        rollout_id=7,
        abort_count=0,
        messages_delta=check_messages([{"role": "user", "content": [{"type": "text", "text": "hi"}]}]),
        train_token_delta=_chars("hi"),
        rollout_token_delta=_chars("hi"),
    )
    resp1 = forest.append_resp(
        parent_state_hash=obs1.state_hash,
        rollout_id=7,
        abort_count=0,
        messages_delta=[{"role": "assistant", "content": [{"type": "text", "text": "ab"}]}],
        token_delta=_chars("ab"),
        logprob_delta=[-0.1, -0.2],
        weight_version_delta=["1"],
        weight_version_spans=[WeightVersionSpan("1", 0, 2, 0)],
    )
    obs2 = forest.append_obs(
        parent_state_hash=resp1.state_hash,
        rollout_id=8,
        abort_count=0,
        messages_delta=check_messages([{"role": "user", "content": [{"type": "text", "text": "tool"}]}]),
        train_token_delta=_chars("tool"),
        rollout_token_delta=_chars("tool"),
    )
    resp2 = forest.append_resp(
        parent_state_hash=obs2.state_hash,
        rollout_id=8,
        abort_count=0,
        messages_delta=[{"role": "assistant", "content": [{"type": "text", "text": "cd"}]}],
        token_delta=_chars("cd"),
        logprob_delta=[-0.3, -0.4],
        weight_version_delta=["3"],
        weight_version_spans=[WeightVersionSpan("3", 0, 2, 1)],
    )
    return forest, resp2.state_hash


def test_build_sample_flattens_spans_to_response_coordinates():
    forest, leaf = _two_resp_forest()
    sample = forest.build_sample(leaf_state_hash=leaf, tokenizer=_FakeTokenizer())
    # 展平后坐标系变为相对整条 response 的连续下标：第二个 resp node 的 span
    # 偏移 response_start=2。
    assert sample.weight_version_spans == [
        WeightVersionSpan("1", 0, 2, 0),
        WeightVersionSpan("3", 2, 4, 1),
    ]
    # 旧字段并存。
    assert sample.weight_versions == ["1", "3"]


class _FakeTokenizer:
    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        return [ord(ch) for ch in str(text)]

    def decode(self, token_ids, skip_special_tokens=False):
        del skip_special_tokens
        return "".join(chr(token_id) for token_id in token_ids)
