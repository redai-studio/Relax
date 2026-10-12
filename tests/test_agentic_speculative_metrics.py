# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""CPU coverage of backend metadata, committed Forest exports and metrics."""

import asyncio
import json
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from relax.agentic.pipeline import TrainingFieldArtifact
from relax.agentic.session.service import AgenticSessionShard
from relax.agentic.session.state import InflightRequest, RequestKind, SessionForest, check_messages
from relax.utils.metrics.speculative import compute_spec_metrics
from relax.utils.types import Sample


class Tokenizer:
    def decode(self, tokens, **kwargs):
        return "".join(map(chr, tokens))


def setup_session(session_id="session-1"):
    cls = AgenticSessionShard.__ray_metadata__.modified_class
    shard = object.__new__(cls)
    shard.args = SimpleNamespace(
        agentic_reasoning_parser=None,
        agentic_tool_call_parser=None,
        use_rollout_routing_replay=False,
        sglang_speculative_algorithm="EAGLE",
    )
    shard._generation_backend = SimpleNamespace(
        tokenizer=Tokenizer(), compiler=SimpleNamespace(apply_chat_template_kwargs={})
    )
    forest = SessionForest.create_empty(session_id=session_id)
    root = forest.append_obs(
        parent_state_hash=forest.root_state_hash,
        rollout_id=0,
        abort_count=0,
        messages_delta=check_messages([{"role": "user", "content": "q"}]),
        train_token_delta=[ord("q")],
        rollout_token_delta=[ord("q")],
    )
    session = SimpleNamespace(
        forest=forest,
        session_id=session_id,
        resp_state_hash_by_request_id={},
        group=SimpleNamespace(rollout_mode="train"),
    )
    return shard, session, root


def request(parent, request_id):
    return InflightRequest(
        request_id=request_id,
        parent_state_hash=parent.state_hash,
        rollout_id=0,
        kind=RequestKind.FRESH,
        abort_count=0,
        waiter=asyncio.get_running_loop().create_future(),
        wall_started_at=time.monotonic(),
    )


def apply(shard, ir, text, counts):
    a, p, v, c = counts
    text = (text * c)[:c]
    shard._apply_generate_result(
        ir,
        SimpleNamespace(
            new_tokens=list(map(ord, text)),
            new_log_probs=[-0.1] * len(text),
            elapsed=0.01,
            meta_info={
                "spec_num_correct_drafts": a,
                "spec_num_proposed_drafts": p,
                "spec_verify_ct": v,
                "completion_tokens": c,
            },
        ),
    )


def commit(shard, session, ir, finish_type="stop"):
    ir.pending_status = "truncated" if finish_type == "length" else "completed"
    shard._terminal_response_locked(session=session, ir=ir, finish_type=finish_type)
    return session.forest.nodes_by_hash[session.resp_state_hash_by_request_id[ir.request_id]]


def export(shard, session, leaves):
    records = tuple(
        {
            "messages": session.forest.full_messages(leaf.state_hash),
            "name": str(i),
            "metadata": {},
            "reward": 1,
        }
        for i, leaf in enumerate(leaves)
    )
    transport = shard._build_session_export_transport(
        session=session, reward=None, metadata={}, output_records=records, finalize_events={}
    )
    samples = [TrainingFieldArtifact(sample_payload=row["sample_payload"]).to_sample() for row in transport.exports]
    return [Sample.from_dict(json.loads(json.dumps(sample.to_dict()))) for sample in samples]


async def test_shared_prefix_export_matches_report():
    shard, session, root = setup_session()
    ir_a = request(root, "A")
    apply(shard, ir_a, "a", (1, 2, 1, 2))
    a = commit(shard, session, ir_a)
    leaves = []
    for name, counts in [("B", (9, 10, 2, 10)), ("C", (2, 4, 1, 3)), ("D", (100, 100, 1, 100))]:
        ir = request(a, name)
        apply(shard, ir, name.lower(), counts)
        leaves.append(commit(shard, session, ir))

    samples = export(shard, session, leaves[:2])
    result = compute_spec_metrics(shard.args, samples)
    expected = json.loads((Path(__file__).parent / "fixtures/speculative_metrics.json").read_text())
    assert result == expected
    assert compute_spec_metrics(shard.args, samples + samples) == expected
    assert compute_spec_metrics(shard.args, list(reversed(samples))) == expected
    assert {r["request_id"] for s in samples for r in s.spec_generations} == {"A", "B", "C"}


async def test_identical_text_independent_requests():
    shard, session, root = setup_session()
    first = request(root, "first")
    apply(shard, first, "same", (1, 2, 10, 12))
    a = commit(shard, session, first)
    second = request(root, "second")
    apply(shard, second, "same", (9, 10, 2, 12))
    b = commit(shard, session, second)
    assert a is b
    samples = export(shard, session, [a])
    result = compute_spec_metrics(shard.args, samples)
    assert result["spec/generation_count"] == 2
    assert result["spec_accept_rate"] == pytest.approx(10 / 12)
    assert samples[0].spec_info.spec_accept_token_num == 10
    assert samples[0].spec_info.spec_draft_token_num == 12
    parent, unmatched = shard._match_parent_state(
        forest=session.forest, messages=session.forest.full_messages(a.state_hash), tools=[], chat_template_kwargs={}
    )
    assert parent is a and unmatched == []


async def test_sessions_do_not_share_generation_identity():
    samples = []
    for sid, counts in [("one", (1, 2, 10, 12)), ("two", (9, 10, 2, 12))]:
        shard, session, root = setup_session(sid)
        ir = request(root, f"req_{sid}_0")
        apply(shard, ir, "same-text", counts)
        samples.extend(export(shard, session, [commit(shard, session, ir)]))
    result = compute_spec_metrics(shard.args, samples)
    assert result["spec/generation_count"] == 2
    assert result["spec_accept_rate"] == pytest.approx(10 / 12)


@pytest.mark.parametrize("accepted", [9, None])
async def test_resumed_ir_is_committed_once_with_complete_attempt_coverage(accepted):
    shard, session, root = setup_session()
    ir = request(root, "resumed")
    apply(shard, ir, "first", (1, 2, 1, 2))
    ir.abort_count += 1
    ir.kind = RequestKind.RESUMED
    apply(shard, ir, "second", (accepted, 10, 2, 10))
    samples = export(shard, session, [commit(shard, session, ir, "length")])
    result = compute_spec_metrics(shard.args, samples)
    assert samples[0].status == Sample.Status.TRUNCATED
    assert result["spec/generation_count"] == 1
    assert result["spec_accept_length"] == 4
    if accepted is None:
        assert "spec_accept_rate" not in result
        assert result["spec/acceptance_observed_count"] == 0
    else:
        assert result["spec_accept_rate"] == pytest.approx(10 / 12)
