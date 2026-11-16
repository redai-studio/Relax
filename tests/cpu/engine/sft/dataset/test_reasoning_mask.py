# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Reasoning-loss contracts through real tokenization and the native
dataset."""

import json
from copy import deepcopy

import pytest
import torch
from tokenizers import Tokenizer, decoders, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast

from relax.engine.sft.dataset.chat_template import render_with_loss_mask
from relax.engine.sft.dataset.streaming import SFTStreamingDataset, _build_canonical_sample_from_row
from tests.cpu.engine.sft.dataset.test_chat_template import (
    _ds_sample,
    _FakeDeepSeekTokenizer,
    _FakeFastTokenizerNoGenerationMarker,
    _FakeGemma4ToolTokenizer,
)


def _tokenizer(native=False):
    alphabet = sorted(pre_tokenizers.ByteLevel.alphabet())
    backend = Tokenizer(models.BPE({char: i for i, char in enumerate(alphabet)}, []))
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    backend.decoder = decoders.ByteLevel()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend)
    body = (
        "{% if message.reasoning_content is defined %}"
        "{{ '<think>\\n' + message.reasoning_content + '\\n</think>' }}{% endif %}"
        "{{ message.content or '' }}"
        "{% for call in message.get('tool_calls', []) %}"
        "{{ '<tool_call>' + call.function.name + '</tool_call>' }}{% endfor %}"
        "{{ '<|im_end|>\\n' }}"
    )
    tokenizer.chat_template = (
        "{% for message in messages %}{{ '<|im_start|>' + message.role + '\\n' }}"
        + (
            "{% if message.role == 'assistant' %}{% generation %}"
            + body
            + "{% endgeneration %}{% else %}"
            + body
            + "{% endif %}"
            if native
            else body
        )
        + "{% endfor %}"
    )
    return tokenizer


def _sample(messages):
    return _build_canonical_sample_from_row(
        {"messages": messages},
        row_index=0,
        prompt_key="messages",
        label_key=None,
        multimodal_keys=None,
        metadata_key="metadata",
        tool_key="tools",
        system_prompt=None,
        require_response=True,
        source_name="test",
    )


@pytest.mark.parametrize(
    "native,reasoning",
    [(False, "SECRET"), (True, "SECRET"), (False, "literal </think> inside reasoning")],
)
def test_mask_keeps_input_and_learns_answer_tools_and_eos(native, reasoning):
    tokenizer = _tokenizer(native)
    messages = [
        {"role": "user", "content": "Question"},
        {
            "role": "assistant",
            "content": "ANSWER with literal </think>",
            "reasoning_content": reasoning,
            "tool_calls": [{"type": "function", "function": {"name": "lookup", "arguments": {}}}],
        },
    ]
    original = deepcopy(messages)
    sample = _sample(messages)
    before_ids, before_mask = render_with_loss_mask(sample, tokenizer=tokenizer)
    ids, mask = render_with_loss_mask(sample, tokenizer=tokenizer, mask_reasoning=True)
    assert torch.equal(ids, before_ids)
    assert torch.all(mask <= before_mask)
    learned = tokenizer.decode(ids[mask.bool()])
    assert learned == "ANSWER with literal </think><tool_call>lookup</tool_call><|im_end|>\n"
    assert messages == original


@pytest.mark.parametrize("native", [False, True])
@pytest.mark.parametrize("last_only", [False, True])
def test_mask_composes_with_step_exclusion_and_last_round(native, last_only):
    tokenizer = _tokenizer(native)
    sample = _sample(
        [
            {"role": "user", "content": "FIRST"},
            {"role": "assistant", "content": "HISTORY", "reasoning_content": "OLD_REASON"},
            {"role": "user", "content": "SECOND"},
            {"role": "assistant", "content": "EXCLUDED", "reasoning_content": "SKIP", "step_loss_mask": 0},
            {"role": "assistant", "content": "ANSWER", "reasoning_content": "CURRENT"},
        ]
    )
    ids, mask = render_with_loss_mask(sample, tokenizer=tokenizer, mask_reasoning=True, last_turn_only=last_only)
    learned = tokenizer.decode(ids[mask.bool()])
    assert learned == ("" if last_only else "HISTORY<|im_end|>\n") + "ANSWER<|im_end|>\n"
    assert "OLD_REASON" in tokenizer.decode(ids) and "EXCLUDED" in tokenizer.decode(ids)


@pytest.mark.parametrize("ignore_empty", [False, True])
def test_empty_thinking_option_still_controls_trailing_whitespace(ignore_empty):
    tokenizer = _tokenizer(native=True)
    sample = _sample([{"role": "assistant", "content": "\n\nANSWER", "reasoning_content": ""}])
    ids, mask = render_with_loss_mask(
        sample, tokenizer=tokenizer, mask_reasoning=True, ignore_empty_think=ignore_empty
    )
    assert tokenizer.decode(ids[mask.bool()]) == ("" if ignore_empty else "\n\n") + "ANSWER<|im_end|>\n"


@pytest.mark.parametrize("content", ["<think>unfinished", "<think>secret</think>answer </think>"])
def test_ambiguous_or_incomplete_embedded_thinking_fails(content):
    with pytest.raises(ValueError, match="thinking block"):
        render_with_loss_mask(
            _sample([{"role": "assistant", "content": content}]), tokenizer=_tokenizer(), mask_reasoning=True
        )


def test_embedded_thinking_and_answer_literal_tags():
    tokenizer = _tokenizer()
    for content, learned in [
        ("<think>SECRET</think>ANSWER", "ANSWER"),
        ("<think></think>ANSWER </think>", "ANSWER </think>"),
        ("ANSWER mentions <think>literal</think>", "ANSWER mentions <think>literal</think>"),
    ]:
        ids, mask = render_with_loss_mask(
            _sample([{"role": "assistant", "content": content}]),
            tokenizer=tokenizer,
            mask_reasoning=True,
        )
        assert tokenizer.decode(ids[mask.bool()]) == learned + "<|im_end|>\n"


def test_token_crossing_reasoning_boundary_is_excluded():
    class PairTokenizer(_FakeFastTokenizerNoGenerationMarker):
        @staticmethod
        def _tokenize(text):
            offsets = [(i, min(i + 2, len(text))) for i in range(0, len(text), 2)]
            return list(range(len(offsets))), offsets

    tokenizer = PairTokenizer()
    sample = _sample([{"role": "assistant", "content": "<think>SECRET</think>ANSWER"}])
    text = tokenizer.apply_chat_template(
        [{"role": "assistant", "content": sample.messages[0].content}], tokenize=False
    )
    boundary = text.index("ANSWER")
    assert boundary % 2 == 1
    before_ids, before_mask = render_with_loss_mask(sample, tokenizer=tokenizer)
    ids, mask = render_with_loss_mask(sample, tokenizer=tokenizer, mask_reasoning=True)
    assert torch.equal(ids, before_ids)
    assert before_mask[boundary // 2] == 1 and mask[boundary // 2] == 0
    assert mask[boundary // 2 + 1] == 1


def test_native_dataset_reads_jsonl_and_applies_reasoning_policy(tmp_path):
    path = tmp_path / "data.jsonl"
    path.write_text(
        json.dumps(
            {
                "messages": [
                    {"role": "user", "content": "QUESTION"},
                    {"role": "assistant", "content": "ANSWER", "reasoning_content": "SECRET"},
                ]
            }
        )
        + "\n"
    )
    tokenizer = _tokenizer()
    dataset = SFTStreamingDataset(
        str(path), tokenizer=tokenizer, prompt_key="messages", prefetch_max_cached=0, loss_mask_reasoning=True
    )
    try:
        samples, _ = dataset.get_batch(1)
        assert "SECRET" in tokenizer.decode(samples[0].tokens)
        assert tokenizer.decode(samples[0].tokens[samples[0].loss_mask.bool()]) == "ANSWER<|im_end|>\n"
    finally:
        dataset.stop()


def test_deepseek_reasoning_is_masked_without_changing_input():
    sample = _ds_sample("<think>\nSECRET\n</think>\nANSWER")
    tokenizer = _FakeDeepSeekTokenizer()
    before, _ = render_with_loss_mask(sample, tokenizer=tokenizer)
    ids, mask = render_with_loss_mask(sample, tokenizer=tokenizer, mask_reasoning=True)
    assert torch.equal(ids, before)
    assert "".join(chr(i) for i in ids[mask.bool()].tolist()) == "\nANSWER<｜end▁of▁sentence｜>"


def test_gemma_thinking_override_still_masks_reasoning(monkeypatch):
    monkeypatch.setenv("GEMMA4_SFT_THINKING", "1")
    tokenizer = _FakeGemma4ToolTokenizer()
    sample = _sample(
        [
            {"role": "user", "content": "Q"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "lookup", "arguments": {}}}],
            },
            {"role": "tool", "content": "RESULT", "tool_call_id": "call_1", "name": "lookup"},
            {"role": "assistant", "content": "ANSWER"},
        ]
    )
    before, before_mask = render_with_loss_mask(sample, tokenizer=tokenizer)
    assert "private" in "".join(chr(i) for i in before[before_mask.bool()].tolist())
    ids, mask = render_with_loss_mask(sample, tokenizer=tokenizer, mask_reasoning=True)
    assert torch.equal(ids, before)
    learned = "".join(chr(i) for i in ids[mask.bool()].tolist())
    assert "<|channel>" not in learned and "<channel|>" not in learned
    assert "private" not in learned and "RESULT" not in learned
    assert "ANSWER<turn|>" in learned and "call:lookup" in learned


def test_unknown_template_does_not_silently_ignore_mask():
    tokenizer = _tokenizer(native=True)
    tokenizer.chat_template = tokenizer.chat_template.replace("<|im_start|>", "<header>")
    with pytest.raises(ValueError, match="Reasoning masking requires"):
        render_with_loss_mask(
            _sample([{"role": "assistant", "content": "ANSWER", "reasoning_content": "SECRET"}]),
            tokenizer=tokenizer,
            mask_reasoning=True,
        )
