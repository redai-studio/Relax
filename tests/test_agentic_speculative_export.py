# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import json
from typing import Any

import pytest

from relax.agentic.pipeline import TrainingFieldArtifact
from relax.agentic.session.state import MsgNode, SessionForest
from relax.utils.speculative import SpeculativeCounts
from relax.utils.types import Sample


class Tokenizer:
    def decode(self, tokens: list[int], **kwargs: Any) -> str:
        return "".join(chr(token) for token in tokens)


def forest_with_prompt(session_id: str = "session") -> tuple[SessionForest, MsgNode]:
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


def commit_response(
    forest: SessionForest,
    parent: MsgNode,
    request_id: str,
    text: str = "a",
    counts: SpeculativeCounts | None = None,
) -> MsgNode:
    return forest.append_resp(
        parent_state_hash=parent.state_hash,
        rollout_id=0,
        abort_count=0,
        messages_delta=[{"role": "assistant", "content": text}],
        token_delta=[ord(char) for char in text],
        logprob_delta=[-0.1] * len(text),
        spec_counts=counts,
        export_metadata_patch={"request_id": request_id},
    )


def test_speculative_identity_separates_equal_content_and_is_idempotent() -> None:
    forest, prompt = forest_with_prompt()
    first = commit_response(forest, prompt, "one", counts=SpeculativeCounts(1, 2, 1, 2))
    second = commit_response(forest, prompt, "two", counts=SpeculativeCounts(9, 10, 2, 11))
    assert first is second  # Preserve existing content-addressed matching.
    commit_response(forest, prompt, "two", counts=SpeculativeCounts(9, 10, 2, 11))
    assert forest.generation_ids_by_state[first.state_hash] == ["one", "two"]
    assert len(forest.committed_generations) == 2
    assert forest.committed_generations["one"].counts.accepted == 1
    assert forest.committed_generations["two"].counts.accepted == 9


def test_speculative_identity_conflict_does_not_register_a_state() -> None:
    forest, prompt = forest_with_prompt()
    commit_response(forest, prompt, "one", counts=SpeculativeCounts(1, 2))
    leaves = forest.export_leaf_hashes()
    with pytest.raises(ValueError, match="Conflicting committed generation"):
        commit_response(forest, prompt, "one", text="different", counts=SpeculativeCounts(1, 2))
    with pytest.raises(ValueError, match="Conflicting committed generation"):
        commit_response(forest, prompt, "one", counts=SpeculativeCounts(2, 2))
    assert forest.export_leaf_hashes() == leaves


def test_speculative_identity_sessions_and_uncommitted_requests() -> None:
    first, prompt = forest_with_prompt("first")
    second, other_prompt = forest_with_prompt("second")
    assert first.committed_generations == {}
    commit_response(first, prompt, "same-id")
    commit_response(second, other_prompt, "same-id")
    assert first.committed_generations["same-id"].session_id == "first"
    assert second.committed_generations["same-id"].session_id == "second"


def test_speculative_export_shared_lineage_excludes_other_branch() -> None:
    forest, prompt = forest_with_prompt()
    a = commit_response(forest, prompt, "a", counts=SpeculativeCounts(1, 2, 1, 2))
    b = commit_response(forest, a, "b", text="b", counts=SpeculativeCounts(9, 10, 2, 11))
    c = commit_response(forest, a, "c", text="c", counts=SpeculativeCounts(2, 4, 1, 3))
    commit_response(forest, prompt, "discarded", text="d", counts=SpeculativeCounts(100, 100, 1, 100))
    samples = [forest.build_sample(leaf_state_hash=node.state_hash, tokenizer=Tokenizer()) for node in (b, c)]
    assert [[item["generation_id"] for item in sample.spec_generations] for sample in samples] == [
        ["a", "b"],
        ["a", "c"],
    ]
    assert samples[0].spec_generations[0] == samples[1].spec_generations[0]
    assert samples[0].spec_generations[0]["counts"]["accepted"] == 1
    artifact = TrainingFieldArtifact.from_sample(samples[0])
    restored = artifact.to_sample()
    assert restored.spec_generations == samples[0].spec_generations
    json_restored = Sample.from_dict(json.loads(json.dumps(restored.to_dict())))
    assert json_restored.spec_generations == restored.spec_generations
    restored.spec_generations[0]["counts"]["accepted"] = 999
    assert artifact.to_sample().spec_generations[0]["counts"]["accepted"] == 1
    assert forest.committed_generations["a"].counts.accepted == 1


def test_speculative_export_equal_content_covers_all_committed_instances() -> None:
    forest, prompt = forest_with_prompt()
    node = commit_response(forest, prompt, "first", counts=SpeculativeCounts(1, 2))
    before = forest.build_sample(leaf_state_hash=node.state_hash, tokenizer=Tokenizer())
    commit_response(forest, prompt, "second", counts=SpeculativeCounts(9, 10))
    after = forest.build_sample(leaf_state_hash=node.state_hash, tokenizer=Tokenizer())
    assert len(before.spec_generations) == 1  # Export is a snapshot, not a live list.
    assert [record["generation_id"] for record in after.spec_generations] == ["first", "second"]


def test_speculative_export_legacy_identity_is_not_fabricated() -> None:
    forest, prompt = forest_with_prompt()
    node = commit_response(forest, prompt, "", counts=SpeculativeCounts(1, 2))
    sample = forest.build_sample(leaf_state_hash=node.state_hash, tokenizer=Tokenizer())
    assert sample.spec_generations is None
    assert forest.committed_generations == {}


@pytest.mark.parametrize("identities", [("", "known"), ("known", "")])
def test_speculative_export_mixed_identity_state_stays_legacy(identities: tuple[str, str]) -> None:
    forest, prompt = forest_with_prompt()
    for identity in identities:
        node = commit_response(forest, prompt, identity, counts=SpeculativeCounts(1, 2))
    sample = forest.build_sample(leaf_state_hash=node.state_hash, tokenizer=Tokenizer())
    assert sample.spec_generations is None
