# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import copy

import pytest

from relax.agentic.session.state import SessionForest


class _Tokenizer:
    def decode(self, tokens, **kwargs):
        return " ".join(map(str, tokens))


def _forest():
    forest = SessionForest.create_empty(
        session_id="session", group_index=2, index=3, label=None, train_metadata=None, metadata={}
    )
    root = forest.append_obs(
        parent_state_hash=forest.root_state_hash,
        rollout_id=0,
        abort_count=0,
        messages_delta=[{"role": "user", "content": "prompt"}],
        train_token_delta=[1],
        rollout_token_delta=[1],
    )
    return forest, root


def _response(forest, parent, text, tokens, spans=None):
    return forest.append_resp(
        parent_state_hash=parent.state_hash,
        rollout_id=0,
        abort_count=0,
        messages_delta=[{"role": "assistant", "content": text}],
        token_delta=tokens,
        logprob_delta=[-0.5] * len(tokens),
        weight_version_spans=spans,
        export_metadata_patch={"request_id": text},
    )


def _sample(forest, leaf):
    return forest.build_sample(leaf_state_hash=leaf.state_hash, tokenizer=_Tokenizer())


def test_multiturn_spans_follow_continuation_coordinates_and_selected_lineage():
    forest, root = _forest()
    first = _response(forest, root, "first", [10] * 4, [{"start": 0, "end": 4, "weight_version": "10"}])
    tool = forest.append_obs(
        parent_state_hash=first.state_hash,
        rollout_id=0,
        abort_count=0,
        messages_delta=[{"role": "user", "content": "tool result"}],
        train_token_delta=[20] * 3,
        rollout_token_delta=[20] * 3,
    )
    second = _response(forest, tool, "second", [30] * 6, [{"start": 0, "end": 6, "weight_version": "12"}])
    sibling = _response(forest, tool, "unselected", [40], [{"start": 0, "end": 1, "weight_version": "99"}])
    sample = _sample(forest, second)
    spans = sample.metadata["agentic_trace"]["weight_version_spans"]
    assert [(s["start"], s["end"], s["weight_version"]) for s in spans] == [(0, 4, "10"), (7, 13, "12")]
    assert [s["turn_idx"] for s in spans] == [0, 1]
    assert [s["request_id"] for s in spans] == ["first", "second"]
    assert [s["resp_state_hash"] for s in spans] == [first.state_hash, second.state_hash]
    assert sample.loss_mask == [1] * 4 + [0] * 3 + [1] * 6
    assert sample.index == 3 and sample.group_index == 2 and sample.session_id == "session"
    assert _sample(forest, sibling).metadata["agentic_trace"]["weight_version_spans"][-1]["weight_version"] == "99"


def test_partial_fragments_and_missing_versions_are_preserved_without_aliasing():
    forest, root = _forest()
    pending = [
        {"start": 0, "end": 2, "weight_version": "10", "publication": {"sequence": 1}},
        {"start": 2, "end": 3, "weight_version": None},
        {"start": 3, "end": 5, "weight_version": "12"},
    ]
    leaf = _response(forest, root, "resumed", [2, 3, 4, 5, 6], pending)
    expected = copy.deepcopy(pending)
    pending[0]["weight_version"] = "cancelled-retry"
    pending.append({"start": 5, "end": 7, "weight_version": "uncommitted"})
    sample = _sample(forest, leaf)
    exported = sample.metadata["agentic_trace"]["weight_version_spans"]
    assert [{k: span[k] for k in expected[i]} for i, span in enumerate(exported)] == expected
    assert [span["fragment_index"] for span in exported] == [0, 1, 2]
    exported[0]["weight_version"] = "changed-by-consumer"
    exported[0]["publication"]["sequence"] = 99
    assert leaf.weight_version_spans == expected


@pytest.mark.parametrize("tokens", [[], [2, 3]])
def test_legacy_node_does_not_infer_token_provenance_from_version_list(tokens):
    forest, root = _forest()
    leaf = _response(forest, root, "legacy", tokens)
    leaf.weight_version_delta = ["10", "12"]
    sample = _sample(forest, leaf)
    spans = sample.metadata["agentic_trace"]["weight_version_spans"]
    assert sample.weight_versions == ["10", "12"]
    assert len(spans) == bool(tokens)
    if tokens:
        assert (spans[0]["start"], spans[0]["end"], spans[0]["weight_version"]) == (0, 2, None)


def test_provenance_does_not_change_existing_sample_fields():
    forest, root = _forest()
    leaf = _response(forest, root, "answer", [2, 3])
    before = _sample(forest, leaf)
    leaf.weight_version_spans = [{"start": 0, "end": 2, "weight_version": "10"}]
    after = _sample(forest, leaf)
    before.metadata["agentic_trace"].pop("weight_version_spans")
    after.metadata["agentic_trace"].pop("weight_version_spans")
    assert vars(before) == vars(after)


def test_empty_response_and_unselected_branch_do_not_replace_completion_snapshot():
    forest, root = _forest()
    snapshot = {"epoch": "policy", "sequence": 2}
    first = forest.append_resp(
        parent_state_hash=root.state_hash,
        rollout_id=0,
        abort_count=0,
        messages_delta=[{"role": "assistant", "content": "answer"}],
        token_delta=[2],
        logprob_delta=[-0.5],
        generation_end_publication=snapshot,
    )
    snapshot["sequence"] = 99
    empty = _response(forest, first, "empty", [])
    empty.generation_end_publication = {"epoch": "policy", "sequence": 4}
    sibling = _response(forest, first, "unselected", [3])
    sibling.generation_end_publication = {"epoch": "policy", "sequence": 5}
    sample = _sample(forest, empty)
    assert sample.metadata["agentic_trace"]["generation_end_publication"]["sequence"] == 2
    sample.metadata["agentic_trace"]["generation_end_publication"]["sequence"] = 100
    assert first.generation_end_publication["sequence"] == 2
