# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""CPU regressions against a caller-supplied, local Qwen HF tokenizer."""

import os
from copy import deepcopy

import pytest
from transformers import AutoTokenizer

from relax.engine.sft.dataset.chat_template import render_to_text, render_with_loss_mask
from relax.engine.sft.dataset.streaming import _build_canonical_sample_from_row


@pytest.fixture(scope="module")
def tokenizer():
    path = os.environ.get("SFT_TEST_TOKENIZER_PATH")
    if not path:
        pytest.skip("Set SFT_TEST_TOKENIZER_PATH to a local Qwen3.5 HF tokenizer for CPU integration tests")
    return AutoTokenizer.from_pretrained(path, local_files_only=True, trust_remote_code=True)


def _tool_calls():
    return [{"type": "function", "function": {"name": "lookup", "arguments": {"query": "test"}}}]


def _canonical(row):
    return _build_canonical_sample_from_row(
        row,
        row_index=0,
        prompt_key="messages",
        label_key=None,
        multimodal_keys=None,
        metadata_key="metadata",
        tool_key="tools",
        system_prompt=None,
        require_response=False,
        source_name="tokenizer-regression",
    )


def _render(tokenizer, row, *, preserve_thinking=True, ignore_empty_think=True, last_turn_only=False):
    original = deepcopy(row)
    sample = _canonical(row)
    kwargs = {"enable_thinking": False, "add_non_thinking_prefix": True}
    if preserve_thinking is not None:
        kwargs["preserve_thinking"] = preserve_thinking
    ids, mask = render_with_loss_mask(
        sample,
        tokenizer=tokenizer,
        apply_chat_template_kwargs=kwargs,
        ignore_empty_think=ignore_empty_think,
        last_turn_only=last_turn_only,
    )
    text = tokenizer.decode(ids)
    learned = tokenizer.decode(ids[mask.bool()])
    assert row == original
    assert text == render_to_text(
        sample, tokenizer=tokenizer, apply_chat_template_kwargs=kwargs, last_turn_only=last_turn_only
    )
    assert len(ids) == len(mask)
    assert set(mask.tolist()) <= {0, 1}
    return text, learned


@pytest.mark.parametrize("content", ["Search now.", "<think>\n\n</think>\n\nSearch now.", "", None])
@pytest.mark.parametrize("n_calls", [1, 2])
def test_tool_turn_keeps_empty_thinking_context_but_excludes_its_loss(tokenizer, content, n_calls):
    row = {
        "messages": [
            {"role": "user", "content": "Question"},
            {"role": "assistant", "content": content, "tool_calls": _tool_calls() * n_calls},
        ]
    }
    text, learned = _render(tokenizer, row)
    assert text.count("<think>") == text.count("</think>") == 1
    assert "<think>" not in learned and "</think>" not in learned
    assert text.count("<tool_call>") == learned.count("<tool_call>") == n_calls
    assert learned.count("</tool_call>") == n_calls
    assert "<|im_end|>" in learned
    if content:
        assert "Search now.\n\n<tool_call>" in text
        assert "Search now.\n\n<tool_call>" in learned


@pytest.mark.parametrize("content", ["Search now.", ""])
@pytest.mark.parametrize("ignore_empty_think", [True, False])
def test_independent_reasoning_and_closing_boundary_are_supervised(tokenizer, content, ignore_empty_think):
    row = {
        "messages": [
            {"role": "user", "content": "Question"},
            {
                "role": "assistant",
                "content": content,
                "reasoning_content": "INDEPENDENT_REASONING",
                "tool_calls": _tool_calls(),
            },
        ]
    }
    text, learned = _render(tokenizer, row, ignore_empty_think=ignore_empty_think)
    assert text.count("<think>") == text.count("</think>") == 1
    assert "INDEPENDENT_REASONING" in learned and "</think>" in learned
    assert "<tool_call>" in learned and "<|im_end|>" in learned


def test_step_masked_tool_turn_remains_context_only(tokenizer):
    row = {
        "messages": [
            {"role": "user", "content": "Question"},
            {
                "role": "assistant",
                "content": "EXCLUDED_CONTENT",
                "reasoning_content": "EXCLUDED_REASONING",
                "tool_calls": _tool_calls(),
                "step_loss_mask": 0,
            },
            {"role": "tool", "content": "EXCLUDED_TOOL_RESULT"},
            {"role": "assistant", "content": "LEARNED_ANSWER", "reasoning_content": "LEARNED_REASONING"},
        ]
    }
    text, learned = _render(tokenizer, row)
    assert "EXCLUDED_CONTENT" in text and "EXCLUDED_REASONING" in text and "<tool_call>" in text
    assert "EXCLUDED_CONTENT" not in learned and "EXCLUDED_REASONING" not in learned
    assert "EXCLUDED_TOOL_RESULT" not in learned and "<tool_call>" not in learned
    assert "LEARNED_ANSWER" in learned and "LEARNED_REASONING" in learned
    assert learned.count("<|im_end|>") == 1


@pytest.mark.parametrize("preserve", [None, True, False])
def test_historical_independent_reasoning_respects_explicit_policy(tokenizer, preserve):
    row = {
        "messages": [
            {"role": "user", "content": "First question"},
            {
                "role": "assistant",
                "content": "FIRST_ANSWER",
                "reasoning_content": "HISTORICAL_REASONING",
                "tool_calls": _tool_calls(),
            },
            {"role": "tool", "content": "Result"},
            {"role": "user", "content": "Second question"},
            {"role": "assistant", "content": "SECOND_ANSWER"},
        ]
    }
    text, learned = _render(tokenizer, row, preserve_thinking=preserve)
    assert ("HISTORICAL_REASONING" in text) is (preserve is not False)
    assert ("HISTORICAL_REASONING" in learned) is (preserve is not False)
    assert "FIRST_ANSWER" in learned and "SECOND_ANSWER" in learned


def test_last_turn_mask_excludes_historical_independent_reasoning(tokenizer):
    row = {
        "messages": [
            {"role": "user", "content": "First"},
            {"role": "assistant", "content": "FIRST_ANSWER", "reasoning_content": "HISTORY"},
            {"role": "user", "content": "Second"},
            {"role": "assistant", "content": "SECOND_ANSWER", "reasoning_content": "CURRENT"},
        ]
    }
    text, learned = _render(tokenizer, row, last_turn_only=True)
    assert "HISTORY" in text and "HISTORY" not in learned
    assert "FIRST_ANSWER" not in learned
    assert "CURRENT" in learned and "SECOND_ANSWER" in learned
    assert learned.count("</think>") == learned.count("<|im_end|>") == 1
