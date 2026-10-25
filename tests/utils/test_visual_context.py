# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Shared VCS contracts, including actual Forest token and logprob
ownership."""

import copy
from types import SimpleNamespace

import pytest

from relax.agentic.session.service import AgenticSessionShard
from relax.agentic.session.state import (
    SessionForest,
    _messages_tools_template_state_hash,
    check_messages,
)
from relax.utils.visual_context import VisualContext, image_count


def observation(index, role="tool"):
    return {
        "role": role,
        **({"tool_call_id": f"call-{index}"} if role == "tool" else {}),
        "content": [
            {"type": "text", "text": f"observation-{index}"},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{index}"}},
        ],
    }


def commit(vcs, request, answer="answer"):
    record = {
        "messages": [*copy.deepcopy(request["messages"]), {"role": "assistant", "content": answer}],
        "tools": copy.deepcopy(request.get("tools") or []),
        "chat_template_kwargs": copy.deepcopy(request.get("chat_template_kwargs") or {}),
    }
    vcs.commit(record, {"prompt_tokens": 100, "completion_tokens": 2})
    return record


def test_full_history_tool_observations_keep_task_and_unique_forest_tokens():
    vcs = VisualContext()
    forest = SessionForest.create_empty(session_id="test", group_index=7, index=42)
    shard = AgenticSessionShard.__ray_metadata__.modified_class
    source = [{"role": "system", "content": "rules"}, {"role": "user", "content": "original task"}]
    sizes = []
    for turn in range(10):
        vcs.begin_turn(turn)
        source.append(observation(turn))
        before = copy.deepcopy(source)
        request = vcs.prepare({"messages": source})
        assert source == before  # no mutation, including on prune boundaries
        assert {"role": "user", "content": "original task"} in request["messages"]
        assert request["messages"][-1] == source[-1]
        sizes.append(image_count(request["messages"]))
        messages = check_messages(request["messages"])
        parent, delta = shard._match_parent_state(forest=forest, messages=messages, tools=[], chat_template_kwargs={})
        obs = forest.append_obs(
            parent_state_hash=parent.state_hash,
            rollout_id=0,
            abort_count=0,
            messages_delta=delta,
            train_token_delta=[1] * len(delta),
            rollout_token_delta=[1] * len(delta),
            tools=[],
            chat_template_kwargs={},
        )
        answer = {
            "role": "assistant",
            "content": f"answer-{turn}",
            "tool_calls": [
                {
                    "id": f"call-{turn + 1}",
                    "type": "function",
                    "function": {"name": "click", "arguments": "{}"},
                }
            ],
        }
        forest.append_resp(
            parent_state_hash=obs.state_hash,
            rollout_id=0,
            abort_count=0,
            messages_delta=[answer],
            token_delta=[1000 + turn],
            logprob_delta=[-0.1 * (turn + 1)],
        )
        record = {"messages": [*copy.deepcopy(request["messages"]), answer], "tools": [], "chat_template_kwargs": {}}
        vcs.commit(record, {"prompt_tokens": 100, "completion_tokens": 1})
        source.append(copy.deepcopy(answer))
    vcs.finish("done")
    vcs.finish("done")  # idempotent
    assert sizes == [1, 2, 3, 4, 3, 4, 3, 4, 3, 4]
    trained = []
    for record in vcs.records:
        state_hash = _messages_tools_template_state_hash(check_messages(record["messages"]), [], {})
        sample = forest.build_sample(
            leaf_state_hash=state_hash,
            tokenizer=SimpleNamespace(decode=lambda ids, **kwargs: str(ids)),
        )
        assert sample.index == 42
        trained.extend(
            (t, p)
            for t, p, m in zip(
                sample.tokens[-sample.response_length :],
                sample.rollout_log_probs,
                sample.loss_mask,
                strict=True,
            )
            if m
        )
    assert trained == [(1000 + i, -0.1 * (i + 1)) for i in range(10)]
    assert [r["vcs_segment"]["new_answers"] for r in vcs.records] == [4, 2, 2, 2]
    assert vcs.metrics()["agentic/vcs_pruned_images"] == 6


def test_retries_corrections_tools_and_template_changes_preserve_inputs():
    vcs = VisualContext()
    payload = {"messages": [{"role": "system", "content": "rules"}, observation(0, "user")]}
    first = vcs.prepare(payload)
    commit(vcs, first, "bad answer")
    retry = vcs.prepare(payload)
    assert retry["messages"][1] != first["messages"][1]
    assert "bad answer" not in str(retry)
    commit(vcs, retry)
    correction = copy.deepcopy(payload)
    correction["messages"].append({"role": "user", "content": "please correct your action"})
    corrected = vcs.prepare(correction)
    assert corrected["messages"][-1]["content"] == "please correct your action"
    commit(vcs, corrected)
    changed = {
        **correction,
        "tools": [{"type": "function", "function": {"name": "click"}}],
        "chat_template_kwargs": {"enable_thinking": False},
    }
    request = vcs.prepare(changed)
    assert request["tools"] == changed["tools"]
    assert request["chat_template_kwargs"] == changed["chat_template_kwargs"]
    commit(vcs, request)
    vcs.finish("done")
    assert sum(r["vcs_segment"]["new_answers"] for r in vcs.records) == 4
    assert "tools_or_system_or_template_changed" in vcs.records[-2]["vcs_segment"]["reason"]


def test_env_switch_and_parameters(monkeypatch):
    for name in ("RELAX_VCS", "RELAX_VCS_KEEP_IMAGES", "RELAX_VCS_PRUNE_IMAGES"):
        monkeypatch.delenv(name, raising=False)
    assert VisualContext.from_env() is None
    assert VisualContext.from_env(enabled=True) is not None
    monkeypatch.setenv("RELAX_VCS", "0")
    assert VisualContext.from_env(enabled=True) is None
    monkeypatch.setenv("RELAX_VCS", "1")
    monkeypatch.setenv("RELAX_VCS_KEEP_IMAGES", "4")
    monkeypatch.setenv("RELAX_VCS_PRUNE_IMAGES", "3")
    vcs = VisualContext.from_env()
    assert (vcs.keep_images, vcs.prune_images) == (4, 3)
    monkeypatch.setenv("RELAX_VCS_PRUNE_IMAGES", "0")
    with pytest.raises(ValueError, match="positive"):
        VisualContext.from_env()
