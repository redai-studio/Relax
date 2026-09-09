# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Qwen history-thinking template patch tests."""

import hashlib
import re

import pytest
import torch

from relax.engine.sft.dataset.chat_template import render_to_text, render_with_loss_mask
from relax.engine.sft.dataset.qwen_chat_template_patch import (
    _QWEN38_PRESERVE_HISTORY_GATE,
    _QWEN_ASSISTANT_TOOL_CALL,
    _QWEN_ASSISTANT_WITH_THINK,
    _QWEN_HISTORY_GATE,
    _QWEN_PRESERVE_HISTORY_GATE,
    _QWEN_PRESERVE_REASONING_GATE,
    _QWEN_TOOL_CALL_SEPARATOR_PATCH,
    _QWEN_TOOL_CALL_WITH_SEPARATOR,
    _QWEN_UNCONDITIONAL_REASONING_GATE,
    try_patch_qwen_chat_template,
)
from relax.engine.sft.dataset.sample import CanonicalMessage, CanonicalSample


_QWEN35_TEMPLATE = "\n".join(("template-start", _QWEN_HISTORY_GATE, "template-end"))
_QWEN36_TEMPLATE = _QWEN35_TEMPLATE.replace(_QWEN_HISTORY_GATE, _QWEN_PRESERVE_HISTORY_GATE)
# Qwen3.8 ships a preserve-by-default gate that also references reasoning_content.
_QWEN38_TEMPLATE = "\n".join(("template-start reasoning_content", _QWEN38_PRESERVE_HISTORY_GATE, "template-end"))
_QWEN35_RENDER_TEMPLATE = "\n".join(
    (
        "template-start",
        "{%- set ns = namespace(multi_step_tool=true, last_query_index=messages|length - 1) %}",
        "{%- for message in messages %}",
        "    {%- set content = message.content|trim %}",
        '    {%- if message.role == "assistant" %}',
        "        {%- set reasoning_content = '' %}",
        "        {%- if message.reasoning_content is string %}",
        "            {%- set reasoning_content = message.reasoning_content %}",
        "        {%- else %}",
        "            {%- if '</think>' in content %}",
        "                {%- set reasoning_content = content.split('</think>')[0].rstrip('\\n').split('<think>')[-1].lstrip('\\n') %}",
        "                {%- set content = content.split('</think>')[-1].lstrip('\\n') %}",
        "            {%- endif %}",
        "        {%- endif %}",
        "        {%- set reasoning_content = reasoning_content|trim %}",
        f"        {_QWEN_HISTORY_GATE}",
        "            {{- '<|im_start|>' + message.role + '\\n<think>\\n' + reasoning_content + '\\n</think>\\n\\n' + content }}",
        "        {%- else %}",
        "            {{- '<|im_start|>' + message.role + '\\n' + content }}",
        "        {%- endif %}",
        "        {%- if message.tool_calls and message.tool_calls is iterable and message.tool_calls is not mapping %}",
        "            {{- '<tool_call>dummy</tool_call>' }}",
        "        {%- endif %}",
        "        {{- '<|im_end|>\\n' }}",
        "    {%- endif %}",
        "{%- endfor %}",
        "template-end",
    )
)
_QWEN3_COMPACT_RENDER_TEMPLATE = "\n".join(
    (
        "template-start",
        "{%- set ns = namespace(multi_step_tool=true, last_query_index=messages|length - 1) %}",
        "{%- for message in messages %}",
        "    {%- set content = message.content|trim %}",
        '    {%- if message.role == "assistant" %}',
        "        {%- set reasoning_content = '' %}",
        "        {%- if message.reasoning_content is string %}",
        "            {%- set reasoning_content = message.reasoning_content %}",
        "        {%- else %}",
        "            {%- if '</think>' in content %}",
        "                {%- set reasoning_content = content.split('</think>')[0].rstrip('\\n').split('<think>')[-1].lstrip('\\n') %}",
        "                {%- set content = content.split('</think>')[-1].lstrip('\\n') %}",
        "            {%- endif %}",
        "        {%- endif %}",
        f"        {_QWEN_HISTORY_GATE}",
        "            {%- if loop.last or (not loop.last and reasoning_content) %}",
        "                {{- '<|im_start|>' + message.role + '\\n<think>\\n' + reasoning_content.strip('\\n') + '\\n</think>\\n\\n' + content.lstrip('\\n') }}",
        "            {%- else %}",
        "                {{- '<|im_start|>' + message.role + '\\n' + content }}",
        "            {%- endif %}",
        "        {%- else %}",
        "            {{- '<|im_start|>' + message.role + '\\n' + content }}",
        "        {%- endif %}",
        "        {%- if message.tool_calls %}",
        "            {{- '<tool_call>dummy</tool_call>' }}",
        "        {%- endif %}",
        "        {{- '<|im_end|>\\n' }}",
        "    {%- endif %}",
        "{%- endfor %}",
        "template-end",
    )
)


def _make_sample(*, historical_learn: bool = True) -> CanonicalSample:
    return CanonicalSample(
        messages=[
            CanonicalMessage(role="user", content="plan a trip", learn=False),
            CanonicalMessage(
                role="assistant",
                content="<think>\nNEED_SKILL\n</think>\n\n",
                learn=historical_learn,
                tool_calls=[
                    {
                        "type": "function",
                        "function": {
                            "name": "activate_skill",
                            "arguments": {"skill_name": "daily-overview"},
                        },
                    }
                ],
            ),
            CanonicalMessage(role="tool", content="skill loaded", learn=False),
            CanonicalMessage(role="user", content="# daily-overview skill", learn=False),
            CanonicalMessage(
                role="assistant",
                content="<think>\nFINAL_REASON\n</think>\n\nfinal answer",
                learn=True,
            ),
        ],
        metadata={"source_dataset": "x", "row_index": 0},
    )


class _FakeQwenHistoryTokenizer:
    """Execute only the Qwen history behavior relevant to this regression."""

    chat_template = _QWEN35_TEMPLATE

    def __init__(self, chat_template: str | None = None):
        if chat_template is not None:
            self.chat_template = chat_template
        self.last_template = self.chat_template
        self.used_assistant_mask = False

    @staticmethod
    def _tokenize(text):
        return [ord(char) for char in text], [(index, index + 1) for index in range(len(text))]

    @staticmethod
    def _render_tool_calls(tool_calls):
        text = ""
        for tool_call in tool_calls or []:
            function = tool_call.get("function", tool_call)
            text += f"\n<tool_call>\n<function={function['name']}>"
            for name, value in (function.get("arguments") or {}).items():
                text += f"\n<parameter={name}>\n{value}\n</parameter>"
            text += "\n</function>\n</tool_call>"
        return text

    @staticmethod
    def _assistant_mask(messages, rendered):
        mask = [0] * len(rendered)
        cursor = 0
        for message in messages:
            role = message["role"]
            if role == "tool":
                open_pos = rendered.find("\n<tool_response>\n", cursor)
                close_pos = rendered.find("\n</tool_response>", open_pos + 1)
                cursor = close_pos + len("\n</tool_response>")
                continue
            header = f"<|im_start|>{role}\n"
            header_pos = rendered.find(header, cursor)
            content_start = header_pos + len(header)
            end_pos = rendered.find("<|im_end|>", content_start)
            span_end = end_pos + len("<|im_end|>")
            if span_end < len(rendered) and rendered[span_end] == "\n":
                span_end += 1
            cursor = span_end
            if role != "assistant":
                continue
            mask_start = content_start
            if rendered[content_start : content_start + len("<think>\n")] == "<think>\n":
                mask_start += len("<think>\n")
            for pos in range(mask_start, span_end):
                mask[pos] = 1
        return mask

    def apply_chat_template(
        self,
        messages,
        *,
        tools=None,
        tokenize=True,
        return_tensors=None,
        return_dict=False,
        return_assistant_tokens_mask=False,
        **kwargs,
    ):  # noqa: ARG002
        self.last_template = kwargs.get("chat_template", self.chat_template)
        has_preserve_gate = (
            _QWEN_PRESERVE_HISTORY_GATE in self.last_template or "relax_has_visible_thinking" in self.last_template
        )
        preserve = kwargs.get("preserve_thinking") is True and has_preserve_gate
        last_user_index = max(
            (index for index, message in enumerate(messages) if message["role"] == "user"),
            default=-1,
        )

        rendered = ""
        previous_role = None
        for index, message in enumerate(messages):
            role = message["role"]
            content = message.get("content") or ""
            if role == "tool":
                if previous_role != "tool":
                    rendered += "<|im_start|>user"
                rendered += f"\n<tool_response>\n{content}\n</tool_response>"
                next_role = messages[index + 1]["role"] if index + 1 < len(messages) else None
                if next_role != "tool":
                    rendered += "<|im_end|>\n"
            else:
                if role == "assistant" and "</think>" in content:
                    thinking, _, answer = content.partition("</think>")
                    reasoning = thinking.rsplit("<think>", 1)[-1].strip("\n")
                    content = answer.lstrip("\n")
                    if preserve or index > last_user_index:
                        content = f"<think>\n{reasoning}\n</think>\n\n{content}"
                rendered += f"<|im_start|>{role}\n{content}"
                if role == "assistant":
                    rendered += self._render_tool_calls(message.get("tool_calls"))
                rendered += "<|im_end|>\n"
            previous_role = role

        if not tokenize:
            return rendered
        ids, _ = self._tokenize(rendered)
        if return_assistant_tokens_mask:
            self.used_assistant_mask = True
            result_ids = torch.tensor([ids], dtype=torch.long) if return_tensors == "pt" else [ids]
            return {
                "input_ids": result_ids,
                "assistant_masks": [self._assistant_mask(messages, rendered)],
            }
        if return_dict:
            result_ids = torch.tensor([ids], dtype=torch.long) if return_tensors == "pt" else [ids]
            return {"input_ids": result_ids}
        return ids

    def __call__(self, text, *, add_special_tokens=False, return_offsets_mapping=False, **kwargs):  # noqa: ARG002
        ids, offsets = self._tokenize(text)
        result = {"input_ids": ids}
        if return_offsets_mapping:
            result["offset_mapping"] = offsets
        return result


def _learned_text(input_ids: torch.Tensor, loss_mask: torch.Tensor) -> str:
    return "".join(chr(int(char)) for char, mask in zip(input_ids.tolist(), loss_mask.tolist()) if mask == 1)


def test_qwen_patch_unknown_template_is_not_applicable():
    assert try_patch_qwen_chat_template(_make_sample(), "plain", {}) is None


def test_qwen35_patch_backports_gate_and_auto_preserves_history():
    result = try_patch_qwen_chat_template(_make_sample(), _QWEN35_TEMPLATE, {})
    assert result is not None
    assert result.changed
    assert result.template == _QWEN36_TEMPLATE
    assert "chat_template" not in result.kwargs
    assert result.kwargs["preserve_thinking"] is True


def test_qwen36_native_gate_is_idempotent():
    result = try_patch_qwen_chat_template(_make_sample(), _QWEN36_TEMPLATE, {})
    assert result is not None
    assert not result.changed
    assert result.template == _QWEN36_TEMPLATE
    assert "chat_template" not in result.kwargs
    assert result.kwargs["preserve_thinking"] is True


def test_qwen38_preserve_gate_is_recognized_without_rewrite():
    result = try_patch_qwen_chat_template(_make_sample(), _QWEN38_TEMPLATE, {})
    assert result is not None
    assert not result.changed
    assert result.template == _QWEN38_TEMPLATE
    assert "chat_template" not in result.kwargs
    assert result.kwargs["preserve_thinking"] is True


def test_qwen38_preserve_gate_explicit_false_disables_auto_preserve():
    result = try_patch_qwen_chat_template(
        _make_sample(),
        _QWEN38_TEMPLATE,
        {"preserve_thinking": False},
    )
    assert result is not None
    assert not result.changed
    assert result.kwargs["preserve_thinking"] is False


def test_qwen35_unconditional_reasoning_gate_gets_history_policy(monkeypatch):
    template = "\n".join(("ns.last_query_index", "reasoning_content", _QWEN_UNCONDITIONAL_REASONING_GATE))
    monkeypatch.setattr(
        "relax.engine.sft.dataset.qwen_chat_template_patch._QWEN_UNCONDITIONAL_REASONING_TEMPLATE_SHA256",
        hashlib.sha256(template.encode()).hexdigest(),
    )
    result = try_patch_qwen_chat_template(_make_sample(), template, {"preserve_thinking": False})

    assert result is not None
    assert result.changed
    assert _QWEN_UNCONDITIONAL_REASONING_GATE not in result.template
    assert _QWEN_PRESERVE_REASONING_GATE in result.template
    assert result.kwargs["preserve_thinking"] is False


def test_qwen35_native_reasoning_gate_is_idempotent(monkeypatch):
    template = "\n".join(("ns.last_query_index", "reasoning_content", _QWEN_PRESERVE_REASONING_GATE))
    monkeypatch.setattr(
        "relax.engine.sft.dataset.qwen_chat_template_patch._QWEN_PRESERVE_REASONING_TEMPLATE_SHA256",
        hashlib.sha256(template.encode()).hexdigest(),
    )
    result = try_patch_qwen_chat_template(_make_sample(), template, {"preserve_thinking": False})

    assert result is not None
    assert not result.changed
    assert result.template == template


def test_qwen35_patch_matches_ms_swift_assistant_tool_call_merge():
    template = "\n".join(
        (
            "reasoning_content",
            _QWEN_HISTORY_GATE,
            _QWEN_ASSISTANT_WITH_THINK,
            _QWEN_TOOL_CALL_WITH_SEPARATOR,
        )
    )
    result = try_patch_qwen_chat_template(_make_sample(), template, {"preserve_thinking": False})

    assert result is not None
    assert result.changed
    assert _QWEN_ASSISTANT_TOOL_CALL in result.template
    assert result.template.count(_QWEN_ASSISTANT_WITH_THINK) == 1
    assert result.template.count(_QWEN_TOOL_CALL_WITH_SEPARATOR) == 1
    assert _QWEN_TOOL_CALL_SEPARATOR_PATCH in result.template


def test_qwen_patch_rejects_ambiguous_gate():
    with pytest.raises(RuntimeError, match="old=2"):
        try_patch_qwen_chat_template(_make_sample(), _QWEN35_TEMPLATE + _QWEN_HISTORY_GATE, {})


def test_qwen_patch_fails_fast_on_recognized_template_drift():
    drifted = "reasoning_content\n{%- if loop.index0 >= ns.last_query_index %}"
    with pytest.raises(RuntimeError, match="old=0.*native=0"):
        try_patch_qwen_chat_template(_make_sample(), drifted, {})


def test_qwen_patch_explicit_false_disables_auto_preserve():
    result = try_patch_qwen_chat_template(
        _make_sample(),
        _QWEN35_TEMPLATE,
        {"preserve_thinking": False},
    )
    assert result is not None
    assert result.kwargs["preserve_thinking"] is False


def test_qwen_patch_explicit_null_uses_auto_preserve():
    result = try_patch_qwen_chat_template(
        _make_sample(),
        _QWEN35_TEMPLATE,
        {"preserve_thinking": None},
    )
    assert result is not None
    assert result.kwargs["preserve_thinking"] is True


def test_qwen_patch_rejects_non_boolean_preserve_thinking():
    with pytest.raises(ValueError, match="must be true, false, or null"):
        try_patch_qwen_chat_template(
            _make_sample(),
            _QWEN35_TEMPLATE,
            {"preserve_thinking": "true"},
        )


def test_qwen_patch_allows_compression_of_unlearned_history():
    result = try_patch_qwen_chat_template(
        _make_sample(historical_learn=False),
        _QWEN35_TEMPLATE,
        {"preserve_thinking": False},
    )
    assert result is not None
    assert result.kwargs["preserve_thinking"] is False


def test_qwen_patch_adds_generation_markers_for_all_learned_assistants():
    result = try_patch_qwen_chat_template(_make_sample(), _QWEN35_RENDER_TEMPLATE, {})
    assert result is not None
    assert result.changed
    assert "{%- generation %}" in result.template
    assert "{%- endgeneration %}" in result.template


def test_qwen_patch_adds_generation_markers_for_compact_template():
    result = try_patch_qwen_chat_template(_make_sample(), _QWEN3_COMPACT_RENDER_TEMPLATE, {})
    assert result is not None
    assert result.changed
    assert "{%- generation %}" in result.template
    assert "{%- endgeneration %}" in result.template
    assert "reasoning_content.strip('\\n')" in result.template
    assert "content.lstrip('\\n')" in result.template


def test_qwen_patch_keeps_fallback_when_assistant_learn_flags_need_custom_mask():
    result = try_patch_qwen_chat_template(
        _make_sample(historical_learn=False),
        _QWEN35_RENDER_TEMPLATE,
        {"preserve_thinking": False},
    )
    assert result is not None
    assert result.changed
    assert "{%- generation %}" not in result.template


def test_qwen_compact_patch_keeps_fallback_when_assistant_learn_flags_need_custom_mask():
    result = try_patch_qwen_chat_template(
        _make_sample(historical_learn=False),
        _QWEN3_COMPACT_RENDER_TEMPLATE,
        {"preserve_thinking": False},
    )
    assert result is not None
    assert result.changed
    assert "{%- generation %}" not in result.template


def test_qwen_patch_explicit_true_preserves_unlearned_history():
    result = try_patch_qwen_chat_template(
        _make_sample(historical_learn=False),
        _QWEN35_TEMPLATE,
        {"preserve_thinking": True},
    )
    assert result is not None
    assert result.kwargs["preserve_thinking"] is True


def test_qwen_patch_excludes_wrapped_tool_response_from_last_user_boundary():
    sample = CanonicalSample(
        messages=[
            CanonicalMessage(role="user", content="query", learn=False),
            CanonicalMessage(role="assistant", content="<think>reason</think>", learn=True),
            CanonicalMessage(
                role="user",
                content="<tool_response>result</tool_response>",
                learn=False,
            ),
            CanonicalMessage(role="assistant", content="answer", learn=True),
        ],
        metadata={"source_dataset": "x", "row_index": 0},
    )
    result = try_patch_qwen_chat_template(
        sample,
        _QWEN35_TEMPLATE,
        {"preserve_thinking": False},
    )
    assert result is not None
    assert result.kwargs["preserve_thinking"] is False


def test_qwen35_render_with_loss_mask_preserves_think_before_tool_call():
    tokenizer = _FakeQwenHistoryTokenizer(_QWEN35_RENDER_TEMPLATE)
    input_ids, loss_mask = render_with_loss_mask(_make_sample(), tokenizer=tokenizer)
    learned = _learned_text(input_ids, loss_mask)

    assert tokenizer.used_assistant_mask
    assert tokenizer.last_template != tokenizer.chat_template
    assert "{%- generation %}" in tokenizer.last_template
    assert re.search(r"NEED_SKILL\n</think>\n+<tool_call>", learned)
    assert "activate_skill" in learned
    assert "skill loaded" not in learned
    assert "# daily-overview skill" not in learned


def test_qwen35_render_with_loss_mask_explicit_false_compresses_history():
    tokenizer = _FakeQwenHistoryTokenizer(_QWEN35_RENDER_TEMPLATE)
    input_ids, loss_mask = render_with_loss_mask(
        _make_sample(),
        tokenizer=tokenizer,
        apply_chat_template_kwargs={"preserve_thinking": False},
    )
    learned = _learned_text(input_ids, loss_mask)

    assert tokenizer.used_assistant_mask
    assert "{%- generation %}" in tokenizer.last_template
    assert "NEED_SKILL" not in learned
    assert "activate_skill" in learned


def test_qwen35_render_to_text_uses_same_patch_dispatcher():
    tokenizer = _FakeQwenHistoryTokenizer(_QWEN35_RENDER_TEMPLATE)
    text = render_to_text(_make_sample(), tokenizer=tokenizer)
    first_assistant = text.split("<|im_start|>assistant\n", 1)[1].split("<|im_end|>", 1)[0]
    assert "<think>\nNEED_SKILL\n</think>" in first_assistant
    assert tokenizer.last_template != tokenizer.chat_template
