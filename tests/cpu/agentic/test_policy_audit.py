# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import asyncio
from argparse import Namespace
from unittest.mock import Mock

from relax.agentic.policy_audit import audit_consumed_rows, encode_policy_audit, policy_audit_enabled
from relax.agentic.session.state import InflightRequest, RequestKind, SessionForest
from relax.utils.agentic_obs.records import decode_group_ids, encode_group_ids, group_ids_to_tensor


class _Tokenizer:
    def decode(self, token_ids, **kwargs):
        return "".join(map(chr, token_ids))


def _forest() -> SessionForest:
    return SessionForest.create_empty(
        session_id="policy-audit-session", group_index=0, index=0, label="", train_metadata=None, metadata={}
    )


def _observe(forest, parent, role):
    return forest.append_obs(
        parent_state_hash=parent,
        rollout_id=0,
        abort_count=0,
        messages_delta=[{"role": role, "content": "observation"}],
        train_token_delta=[65],
        rollout_token_delta=[65],
    )


def _respond(parent, request_id, fragments, *, abort_count=0):
    request = InflightRequest(
        request_id=request_id,
        parent_state_hash=parent,
        rollout_id=0,
        kind=RequestKind.FRESH,
        abort_count=abort_count,
        waiter=Mock(spec=asyncio.Future),
        wall_started_at=0,
    )
    for tokens, version in fragments:
        request.append_generated_tokens(tokens, version)
    return request


def _commit(forest, request):
    return forest.append_resp(
        parent_state_hash=request.parent_state_hash,
        rollout_id=request.rollout_id,
        abort_count=request.abort_count,
        messages_delta=[{"role": "assistant", "content": request.request_id}],
        token_delta=request.pending_token_delta,
        logprob_delta=[-0.1] * len(request.pending_token_delta),
        weight_version_spans=request.pending_weight_version_spans,
        export_metadata_patch={"request_id": request.request_id},
    )


def test_tool_wait_resume_and_train_consumption():
    forest = _forest()
    prompt = _observe(forest, forest.root_state_hash, "user")
    first = _commit(forest, _respond(prompt.state_hash, "first", [([66, 67], "10")]))
    tool = _observe(forest, first.state_hash, "tool")

    # An abandoned request and a committed sibling are both outside the chosen export.
    _respond(tool.state_hash, "cancelled", [([88], "11")])
    _commit(forest, _respond(tool.state_hash, "sibling", [([89], "11")]))
    resumed = _respond(tool.state_hash, "retry", [([68], "13"), ([], None), ([69, 70], "15")], abort_count=1)
    leaf = _commit(forest, resumed)
    sample = forest.build_sample(leaf_state_hash=leaf.state_hash, tokenizer=_Tokenizer())
    assert sample.loss_mask == [1, 1, 0, 1, 1, 1]
    # Exercise the same padded UTF-8 tensor encoding used across the queue.
    rows = decode_group_ids(group_ids_to_tensor(encode_group_ids([encode_policy_audit(sample, sample.loss_mask)])))
    reports, metrics, weights = audit_consumed_rows(rows, {"10": 1, "11": 2, "13": 3, "15": 4, "17": 5})

    report = reports[0]
    assert report["session_id"] == "policy-audit-session"
    assert report["leaf_state_hash"] == leaf.state_hash
    assert [
        (f["turn_idx"], f["request_id"], f["start"], f["end"], f["weight_version"], f["age_at_consumption"])
        for f in report["fragments"]
    ] == [(0, "first", 0, 2, "10", 4), (1, "retry", 3, 4, "13", 2), (1, "retry", 4, 6, "15", 1)]
    assert metrics == {
        "generation_version_span": 3,
        "post_generation_version_lag": 1,
        "token_weighted_version_lag": 12 / 5,
        "missing_version_fraction": 0,
    }
    assert weights["token_weighted_version_lag"] == 5


def test_missing_versions_and_masked_tokens():
    forest = _forest()
    prompt = _observe(forest, forest.root_state_hash, "user")
    leaf = _commit(
        forest,
        _respond(prompt.state_hash, "partial", [([66, 67], "2"), ([68], "default"), ([69], None)]),
    )
    sample = forest.build_sample(leaf_state_hash=leaf.state_hash, tokenizer=_Tokenizer())
    row = encode_policy_audit(sample, [1, 0, 1, 1])
    sample.metadata.clear()  # Legacy data has no precise fragment boundaries.
    legacy = encode_policy_audit(sample, [1, 0, 0, 0])
    masked = encode_policy_audit(sample, [0, 0, 0, 0])
    reports, metrics, weights = audit_consumed_rows([row, legacy, masked], {"2": 1, "4": 2})

    assert [f["age_at_consumption"] for f in reports[0]["fragments"]] == [1, None, None]
    assert reports[0]["generation_version_span"] is None
    assert reports[0]["post_generation_version_lag"] is None
    assert reports[0]["missing_version_fraction"] == 2 / 3
    assert reports[1]["missing_version_fraction"] == 1
    assert reports[2]["missing_version_fraction"] is None
    assert metrics["missing_version_fraction"] == 3 / 4
    assert metrics["token_weighted_version_lag"] == 1
    assert weights["generation_version_span"] == weights["post_generation_version_lag"] == 0


def test_policy_audit_scope():
    args = Namespace(use_agentic_rollout=True, fully_async=True)
    assert policy_audit_enabled(args)
    args.custom_convert_samples_to_train_data_path = "custom.converter"
    assert not policy_audit_enabled(args)
