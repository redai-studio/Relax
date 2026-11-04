# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Unit tests for chat_template.render_with_loss_mask."""

from unittest.mock import MagicMock

import pytest
import torch

from relax.engine.sft.dataset.chat_template import (
    HAS_GENERATION_MARKER,
    _to_chat_messages,
    render_to_text,
    render_with_loss_mask,
)
from relax.engine.sft.dataset.deepseek_chat_template_patch import (
    _DS_ASSISTANT_NON_TRANSITION_SCAFFOLD,
    _DS_ASSISTANT_SCAFFOLD,
    _DS_HISTORY_GATE,
)
from relax.engine.sft.dataset.gemma4_chat_template_patch import (
    GEMMA4_EMPTY_THOUGHT,
    GEMMA4_MS_SWIFT_COMPAT_KWARG,
    GEMMA4_SFT_MS_SWIFT_COMPAT_SENTINEL,
    GEMMA4_SFT_TOOL_TURN_BOUNDARY_SENTINEL,
    try_patch_gemma4_thinking,
)
from relax.engine.sft.dataset.sample import (
    CanonicalMessage,
    CanonicalSample,
)


def _make_sample():
    return CanonicalSample(
        messages=[
            CanonicalMessage(role="user", content="Q", learn=False),
            CanonicalMessage(role="assistant", content="A", learn=True),
        ],
        metadata={"source_dataset": "x", "row_index": 0},
    )


def _mock_tokenizer_with_generation_marker(template_str: str = "{% generation %}assistant{% endgeneration %}"):
    """Mock that simulates a chat template containing {% generation %}."""
    tok = MagicMock()
    tok.chat_template = template_str

    # apply_chat_template returns dict with input_ids + assistant_masks
    def _apply(
        messages,
        *,
        tools=None,
        tokenize=True,
        return_tensors=None,
        return_dict=False,
        return_assistant_tokens_mask=False,
        **kwargs,
    ):
        ids = torch.tensor([[1, 2, 3, 4, 5]])
        if return_assistant_tokens_mask:
            return {"input_ids": ids, "assistant_masks": [[0, 0, 0, 1, 1]]}
        return ids

    tok.apply_chat_template.side_effect = _apply
    return tok


class _FakeFastTokenizerNoGenerationMarker:
    """Fake fast tokenizer that wraps messages in Qwen-style ChatML and
    tokenizes char-by-char with offset_mapping support — the minimum surface
    needed by the offset-mapping fallback in chat_template.py."""

    chat_template = "{{ messages[0]['content'] }}"  # no {% generation %}

    @staticmethod
    def _render(messages):
        out = ""
        for m in messages:
            out += f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n"
        return out

    @staticmethod
    def _tokenize(text):
        ids = [ord(c) for c in text]
        offsets = [(i, i + 1) for i in range(len(text))]
        return ids, offsets

    def apply_chat_template(self, messages, *, tools=None, tokenize=True, **kwargs):  # noqa: ARG002
        text = self._render(messages)
        if not tokenize:
            return text
        ids, _ = self._tokenize(text)
        return ids

    def __call__(self, text, *, add_special_tokens=False, return_offsets_mapping=False, **kwargs):  # noqa: ARG002
        ids, offsets = self._tokenize(text)
        result = {"input_ids": ids}
        if return_offsets_mapping:
            result["offset_mapping"] = offsets
        return result


def _mock_tokenizer_without_generation_marker():
    return _FakeFastTokenizerNoGenerationMarker()


def test_has_generation_marker_detects_correctly():
    assert HAS_GENERATION_MARKER("foo {% generation %} bar")
    assert HAS_GENERATION_MARKER("{%generation%}")  # no spaces
    assert not HAS_GENERATION_MARKER("plain template")
    assert not HAS_GENERATION_MARKER("")


def test_render_uses_assistant_mask_when_template_supports_it():
    tok = _mock_tokenizer_with_generation_marker()
    sample = _make_sample()
    input_ids, loss_mask = render_with_loss_mask(sample, tokenizer=tok)
    assert isinstance(input_ids, torch.Tensor)
    assert isinstance(loss_mask, torch.Tensor)
    assert input_ids.shape == loss_mask.shape
    assert loss_mask.tolist() == [0, 0, 0, 1, 1]
    # Verify return_assistant_tokens_mask was requested
    call_kwargs = tok.apply_chat_template.call_args.kwargs
    assert call_kwargs.get("return_assistant_tokens_mask") is True
    assert call_kwargs["chat_template"].startswith(tok.chat_template)
    assert "relax_thread=" in call_kwargs["chat_template"]


@pytest.mark.parametrize("ignore_empty_think", [False, True])
def test_non_chatml_generation_template_keeps_native_mask(ignore_empty_think):
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import PreTrainedTokenizerFast

    backend = Tokenizer(WordLevel({"<unk>": 0, "QUESTION": 1, "ANSWER": 2}, unk_token="<unk>"))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="<unk>")
    tokenizer.chat_template = (
        "{% for message in messages %}"
        "{{ '<|start_header_id|>' + message.role + '<|end_header_id|>' }}"
        "{% if message.role == 'assistant' %}{% generation %}"
        "{{ message.content + '<|eot_id|>' }}{% endgeneration %}"
        "{% else %}{{ message.content + '<|eot_id|>' }}{% endif %}{% endfor %}"
    )
    sample = CanonicalSample(
        messages=[
            CanonicalMessage(role="user", content="QUESTION", learn=False),
            CanonicalMessage(role="assistant", content="ANSWER", learn=True),
        ],
        metadata={"source_dataset": "test", "row_index": 0},
    )
    expected = tokenizer.apply_chat_template(
        _to_chat_messages(sample),
        return_dict=True,
        return_tensors="pt",
        return_assistant_tokens_mask=True,
    )
    ids, mask = render_with_loss_mask(sample, tokenizer=tokenizer, ignore_empty_think=ignore_empty_think)
    assert ids.tolist() == expected["input_ids"].squeeze(0).tolist()
    assert mask.tolist() == torch.as_tensor(expected["assistant_masks"]).squeeze().tolist()
    assert "ANSWER" in tokenizer.decode(ids[mask.bool()])


def test_render_ms_swift_compat_omits_tool_correlation_fields():
    tok = _mock_tokenizer_with_generation_marker()
    sample = CanonicalSample(
        messages=[
            CanonicalMessage(role="user", content="Q", learn=False),
            CanonicalMessage(
                role="tool",
                content="result",
                learn=False,
                tool_call_id="call_1",
                name="lookup",
            ),
        ],
        metadata={"source_dataset": "x", "row_index": 0},
    )

    render_with_loss_mask(
        sample,
        tokenizer=tok,
        apply_chat_template_kwargs={GEMMA4_MS_SWIFT_COMPAT_KWARG: True},
    )

    messages = tok.apply_chat_template.call_args.args[0]
    assert messages[1] == {"role": "tool", "content": "result"}
    assert GEMMA4_MS_SWIFT_COMPAT_KWARG not in tok.apply_chat_template.call_args.kwargs


def test_gemma4_ms_swift_patch_injects_only_model_empty_thought(monkeypatch):
    monkeypatch.setenv("GEMMA4_SFT_THINKING", "1")
    anchor = (
        "    {%- if thinking_text and thinking_gate -%}\n"
        "        {{- '<|channel>thought\\n' + thinking_text + '\\n<channel|>' -}}\n"
        "    {%- endif -%}\n"
    )
    template = "<|turn><channel|>{{ strip_thinking('x') }}\n" + anchor

    result = try_patch_gemma4_thinking(_make_sample(), template, {})

    assert result is not None and result.changed
    assert "role == 'model' and not continue_same_model_turn" in result.template
    assert GEMMA4_EMPTY_THOUGHT.replace("\n", "\\n") in result.template
    assert result.kwargs == {}


def test_gemma4_ms_swift_compat_is_independent(monkeypatch):
    monkeypatch.setenv("GEMMA4_SFT_MS_SWIFT_COMPAT", "1")
    template = "<|turn><channel|>{{ strip_thinking('x') }}"

    result = try_patch_gemma4_thinking(_make_sample(), template, {})

    assert result is not None and result.changed
    assert GEMMA4_SFT_MS_SWIFT_COMPAT_SENTINEL in result.template
    assert result.template.endswith(template)
    assert result.kwargs == {GEMMA4_MS_SWIFT_COMPAT_KWARG: True}


def test_gemma4_ms_swift_patch_closes_tool_call_before_response(monkeypatch):
    monkeypatch.setenv("GEMMA4_SFT_TOOL_TURN_BOUNDARY", "1")
    thinking_anchor = (
        "    {%- if thinking_text and thinking_gate -%}\n"
        "        {{- '<|channel>thought\\n' + thinking_text + '\\n<channel|>' -}}\n"
        "    {%- endif -%}\n"
    )
    template = (
        "<|turn><channel|>{{ strip_thinking('x') }}\n"
        + thinking_anchor
        + "{%- macro format_tool_response_block(tool_name, response) -%}\n"
        + "{{- '<|tool_response>' + response + '<tool_response|>' -}}\n"
        + "{%- endmacro -%}\n"
        + "            {%- set ns_tr_out = namespace(flag=false) -%}\n"
        + "                    {{- format_tool_response_block('lookup', 'ok') -}}\n"
        + "        {%- if ns.prev_message_type == 'tool_call' and not ns_tr_out.flag -%}\n"
        + "            {{- '<|tool_response>' -}}\n"
    )

    result = try_patch_gemma4_thinking(_make_sample(), template, {})

    assert result is not None and result.changed
    assert "ns_tool_turn = namespace(opened=false)" in result.template
    assert "message.get('tool_calls') and not ns_tool_turn.opened" in result.template
    assert "{{- '<turn|>\\n<|turn>model\\n' -}}" in result.template
    assert "{{- '<|tool_response>' -}}" not in result.template
    assert "{{- '<turn|>\\n' -}}" in result.template
    assert "role == 'model' and not continue_same_model_turn" not in result.template


def test_gemma4_tool_boundary_rejects_unresolved_non_terminal_call(monkeypatch):
    monkeypatch.setenv("GEMMA4_SFT_TOOL_TURN_BOUNDARY", "1")
    sample = CanonicalSample(
        messages=[
            CanonicalMessage(role="user", content="Q", learn=False),
            CanonicalMessage(
                role="assistant",
                content="",
                learn=True,
                tool_calls=[{"type": "function", "function": {"name": "lookup", "arguments": {}}}],
            ),
            CanonicalMessage(role="assistant", content="unexpected", learn=True),
        ],
        metadata={"source_dataset": "x", "row_index": 0},
    )
    template = "<|turn><channel|>{{ strip_thinking('x') }}"

    with pytest.raises(ValueError, match="followed immediately by a tool response"):
        try_patch_gemma4_thinking(sample, template, {})


def test_gemma4_tool_boundary_allows_terminal_unresolved_call(monkeypatch):
    monkeypatch.setenv("GEMMA4_SFT_TOOL_TURN_BOUNDARY", "1")
    sample = CanonicalSample(
        messages=[
            CanonicalMessage(role="user", content="Q", learn=False),
            CanonicalMessage(
                role="assistant",
                content="",
                learn=True,
                tool_calls=[{"type": "function", "function": {"name": "lookup", "arguments": {}}}],
            ),
        ],
        metadata={"source_dataset": "x", "row_index": 0},
    )

    result = try_patch_gemma4_thinking(sample, "<|turn><channel|>{{ strip_thinking('x') }}", {})

    assert result is not None and not result.changed


def test_render_falls_back_when_no_generation_marker(capsys):
    tok = _mock_tokenizer_without_generation_marker()
    sample = _make_sample()
    input_ids, loss_mask = render_with_loss_mask(sample, tokenizer=tok)
    # Per-message fallback: user contributes 1 token (len('Q')), assistant 1 (len('A'))
    assert input_ids.shape == loss_mask.shape
    # Only assistant turn participates
    assert loss_mask.sum().item() >= 1
    # The first message (user) contribution must be 0
    assert loss_mask[0].item() == 0


def test_render_passes_tools_through_to_tokenizer():
    tok = _mock_tokenizer_with_generation_marker()
    sample = CanonicalSample(
        messages=[
            CanonicalMessage(role="user", content="x", learn=False),
            CanonicalMessage(role="assistant", content="y", learn=True),
        ],
        metadata={"source_dataset": "x", "row_index": 0},
        tools=[{"type": "function", "function": {"name": "add"}}],
    )
    render_with_loss_mask(sample, tokenizer=tok)
    call_kwargs = tok.apply_chat_template.call_args.kwargs
    assert call_kwargs["tools"] == sample.tools


def test_render_returns_int_tensors_no_padding():
    tok = _mock_tokenizer_with_generation_marker()
    sample = _make_sample()
    input_ids, loss_mask = render_with_loss_mask(sample, tokenizer=tok)
    assert input_ids.dtype in (torch.long, torch.int32, torch.int64)
    assert loss_mask.dtype in (torch.long, torch.int32, torch.int64, torch.bool)
    # 1D after squeeze
    assert input_ids.dim() == 1
    assert loss_mask.dim() == 1


def test_fallback_loss_mask_only_on_learn_messages():
    tok = _mock_tokenizer_without_generation_marker()
    sample = CanonicalSample(
        messages=[
            CanonicalMessage(role="system", content="sys", learn=False),
            CanonicalMessage(role="user", content="ab", learn=False),
            CanonicalMessage(role="assistant", content="cde", learn=True),
        ],
        metadata={"source_dataset": "x", "row_index": 0},
    )
    input_ids, loss_mask = render_with_loss_mask(sample, tokenizer=tok)
    # New fallback marks the entire assistant content span up to (and including)
    # "<|im_end|>\n", so 3 chars of "cde" + 10 chars of "<|im_end|>" + 1 newline = 14.
    learn_span = len("cde") + len("<|im_end|>") + 1
    assert loss_mask.sum().item() == learn_span
    # The trailing learn_span positions should all be 1 (mock tokenizes char-by-char,
    # so the assistant span lands at the tail of the rendered ChatML).
    assert loss_mask[-learn_span:].tolist() == [1] * learn_span
    # All earlier positions (system + user turns) should be 0.
    assert loss_mask[:-learn_span].sum().item() == 0
    assert input_ids.shape == loss_mask.shape


class _FakeQwenStyleTokenizer:
    """Fake fast tokenizer that mimics Qwen3.5 chat-template behavior for tool
    data:

    * ``tool`` messages are wrapped in ``<|im_start|>user\\n<tool_response>...</tool_response>``
      and consecutive tools share one ``user`` wrapper (closed by ``<|im_end|>\\n`` after the
      last one).
    * ``assistant.tool_calls`` are rendered inline as
      ``<tool_call>\\n<function=NAME>...</function>\\n</tool_call>`` between the assistant
      content and ``<|im_end|>``.
    * Chat template lacks ``{% generation %}`` so the fallback path is exercised.
    * Tokenization is char-by-char with offset_mapping support.
    """

    chat_template = "{{ messages[0]['content'] }}"  # no {% generation %}

    @staticmethod
    def _render(messages):
        out = ""
        prev_role = None
        for i, m in enumerate(messages):
            role = m["role"]
            content = m.get("content", "")
            if role == "tool":
                if prev_role != "tool":
                    out += "<|im_start|>user"
                out += "\n<tool_response>\n" + content + "\n</tool_response>"
                next_role = messages[i + 1]["role"] if i + 1 < len(messages) else None
                if next_role != "tool":
                    out += "<|im_end|>\n"
            else:
                out += f"<|im_start|>{role}\n{content}"
                if role == "assistant":
                    for tc in m.get("tool_calls") or []:
                        fn = tc.get("function", tc)
                        name = fn["name"]
                        args = fn.get("arguments") or {}
                        out += f"\n<tool_call>\n<function={name}>"
                        for k, v in args.items():
                            out += f"\n<parameter={k}>\n{v}\n</parameter>"
                        out += "\n</function>\n</tool_call>"
                out += "<|im_end|>\n"
            prev_role = role
        return out

    @staticmethod
    def _tokenize(text):
        return [ord(c) for c in text], [(i, i + 1) for i in range(len(text))]

    def apply_chat_template(self, messages, *, tools=None, tokenize=True, **kwargs):  # noqa: ARG002
        text = self._render(messages)
        if not tokenize:
            return text
        ids, _ = self._tokenize(text)
        return ids

    def __call__(self, text, *, add_special_tokens=False, return_offsets_mapping=False, **kwargs):  # noqa: ARG002
        ids, offsets = self._tokenize(text)
        result = {"input_ids": ids}
        if return_offsets_mapping:
            result["offset_mapping"] = offsets
        return result


def _learned_text(input_ids: torch.Tensor, loss_mask: torch.Tensor) -> str:
    return "".join(chr(int(c)) for c, m in zip(input_ids.tolist(), loss_mask.tolist()) if m == 1)


def test_to_chat_messages_includes_tool_calls():
    """Plan C: CanonicalMessage.tool_calls propagates into the chat-template
    dict."""
    tool_call = {"type": "function", "function": {"name": "f", "arguments": {"x": 1}}}
    sample = CanonicalSample(
        messages=[
            CanonicalMessage(role="user", content="q", learn=False),
            CanonicalMessage(role="assistant", content="", learn=True, tool_calls=[tool_call]),
            CanonicalMessage(role="tool", content="r", learn=False),
        ],
        metadata={"source_dataset": "x", "row_index": 0},
    )
    msgs = _to_chat_messages(sample)
    assert msgs[0] == {"role": "user", "content": "q"}
    assert msgs[1] == {"role": "assistant", "content": "", "tool_calls": [tool_call]}
    # tool_calls absent on messages that don't carry it
    assert "tool_calls" not in msgs[2]


def test_to_chat_messages_preserves_tool_response_correlation_fields():
    sample = CanonicalSample(
        messages=[
            CanonicalMessage(role="user", content="q", learn=False),
            CanonicalMessage(
                role="tool",
                content="result",
                learn=False,
                tool_call_id="call_1",
                name="lookup",
            ),
        ],
        metadata={"source_dataset": "x", "row_index": 0},
    )

    assert _to_chat_messages(sample)[1] == {
        "role": "tool",
        "content": "result",
        "tool_call_id": "call_1",
        "name": "lookup",
    }


def test_fallback_handles_tool_role_wrapped_in_user_block():
    """Plan A: role=='tool' is rendered inside a user wrapper by Qwen3.5;
    fallback must scan ``<tool_response>...</tool_response>`` rather than a
    ``<|im_start|>tool`` header and not raise."""
    tok = _FakeQwenStyleTokenizer()
    sample = CanonicalSample(
        messages=[
            CanonicalMessage(role="user", content="q", learn=False),
            CanonicalMessage(role="assistant", content="A", learn=True),
            CanonicalMessage(role="tool", content="RESPONSE_X", learn=False),
            CanonicalMessage(role="assistant", content="B", learn=True),
        ],
        metadata={"source_dataset": "x", "row_index": 0},
    )
    input_ids, loss_mask = render_with_loss_mask(sample, tokenizer=tok)
    learned = _learned_text(input_ids, loss_mask)
    assert "A" in learned and "B" in learned
    assert "RESPONSE_X" not in learned


def test_fallback_handles_consecutive_tool_messages_in_one_wrapper():
    """Two adjacent tool messages share a single ``<|im_start|>user`` wrapper —
    fallback must locate each ``<tool_response>`` independently and not get
    confused by the missing per-tool header."""
    tok = _FakeQwenStyleTokenizer()
    sample = CanonicalSample(
        messages=[
            CanonicalMessage(role="assistant", content="A", learn=True),
            CanonicalMessage(role="tool", content="R1", learn=False),
            CanonicalMessage(role="tool", content="R2", learn=False),
            CanonicalMessage(role="user", content="U", learn=False),
            CanonicalMessage(role="assistant", content="Z", learn=True),
        ],
        metadata={"source_dataset": "x", "row_index": 0},
    )
    input_ids, loss_mask = render_with_loss_mask(sample, tokenizer=tok)
    learned = _learned_text(input_ids, loss_mask)
    assert "A" in learned and "Z" in learned
    assert "R1" not in learned and "R2" not in learned
    assert "U" not in learned


def test_fallback_assistant_tool_calls_are_inlined_into_loss():
    """Plan C end-to-end: assistant.tool_calls are rendered inline and the
    resulting ``<tool_call>...</tool_call>`` XML lands inside the assistant
    loss region."""
    tok = _FakeQwenStyleTokenizer()
    sample = CanonicalSample(
        messages=[
            CanonicalMessage(role="user", content="q", learn=False),
            CanonicalMessage(
                role="assistant",
                content="",
                learn=True,
                tool_calls=[{"type": "function", "function": {"name": "MY_FN", "arguments": {"k": "v"}}}],
            ),
            CanonicalMessage(role="tool", content="ok", learn=False),
        ],
        metadata={"source_dataset": "x", "row_index": 0},
    )
    input_ids, loss_mask = render_with_loss_mask(sample, tokenizer=tok)
    learned = _learned_text(input_ids, loss_mask)
    assert "<tool_call>" in learned
    assert "MY_FN" in learned
    assert "</tool_call>" in learned
    # Tool response stays out of the loss
    assert "ok" not in learned


class _FakeGemma4ToolTokenizer:
    chat_template = "{{ '<|turn>' }}"  # Gemma4 delimiters, no generation marker.

    @staticmethod
    def _render(messages):
        call = messages[1]["tool_calls"][0]
        tool = messages[2]
        assert tool["tool_call_id"] == call["id"]
        return (
            f"<|turn>user\n{messages[0]['content']}<turn|>\n"
            "<|turn>model\n"
            "<|channel>thought\nprivate<channel|>"
            f"<|tool_call>call:{call['function']['name']}<tool_call|>"
            f"<|tool_response>response:{tool['name']}:{tool['content']}<tool_response|>"
            f"{messages[3]['content']}<turn|>\n"
        )

    def apply_chat_template(self, messages, *, tools=None, tokenize=True, **kwargs):  # noqa: ARG002
        text = self._render(messages)
        return [ord(char) for char in text] if tokenize else text

    def __call__(self, text, *, add_special_tokens=False, return_offsets_mapping=False, **kwargs):  # noqa: ARG002
        result = {"input_ids": [ord(char) for char in text]}
        if return_offsets_mapping:
            result["offset_mapping"] = [(index, index + 1) for index in range(len(text))]
        return result


def test_gemma4_fallback_masks_embedded_tool_response_only():
    tokenizer = _FakeGemma4ToolTokenizer()
    tool_call = {
        "id": "call_1",
        "type": "function",
        "function": {"name": "lookup", "arguments": {"city": "Shanghai"}},
    }
    sample = CanonicalSample(
        messages=[
            CanonicalMessage(role="user", content="weather", learn=False),
            CanonicalMessage(role="assistant", content="", learn=True, tool_calls=[tool_call]),
            CanonicalMessage(role="tool", content="sunny", learn=False, tool_call_id="call_1", name="lookup"),
            CanonicalMessage(role="assistant", content="FINAL", learn=True),
        ],
        metadata={"source_dataset": "x", "row_index": 0},
    )

    input_ids, loss_mask = render_with_loss_mask(sample, tokenizer=tokenizer)
    learned = _learned_text(input_ids, loss_mask)

    assert "<|tool_call>call:lookup" in learned
    assert "FINAL<turn|>" in learned
    assert "private" not in learned
    assert "<|tool_response>" not in learned
    assert "sunny" not in learned


def test_gemma4_tool_fallback_final_only_excludes_call_in_same_model_turn():
    sample = CanonicalSample(
        messages=[
            CanonicalMessage(role="user", content="weather", learn=False),
            CanonicalMessage(
                role="assistant",
                content="",
                learn=False,
                tool_calls=[
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "lookup", "arguments": {}},
                    }
                ],
            ),
            CanonicalMessage(role="tool", content="FINAL", learn=False, name="lookup", tool_call_id="call_1"),
            CanonicalMessage(role="assistant", content="FINAL", learn=True),
        ],
        metadata={"source_dataset": "test", "row_index": 0},
    )
    ids, mask = render_with_loss_mask(sample, tokenizer=_FakeGemma4ToolTokenizer())
    assert _learned_text(ids, mask) == "FINAL<turn|>\n"


class _FakeGemma4SplitToolTokenizer(_FakeGemma4ToolTokenizer):
    chat_template = GEMMA4_SFT_TOOL_TURN_BOUNDARY_SENTINEL + _FakeGemma4ToolTokenizer.chat_template

    @staticmethod
    def _render(messages):
        call = messages[3]["tool_calls"][0]
        tool = messages[4]
        return (
            f"<|turn>user\n{messages[0]['content']}<turn|>\n"
            f"<|turn>model\n{messages[1]['content']}<turn|>\n"
            f"<|turn>user\n{messages[2]['content']}<turn|>\n"
            "<|turn>model\n"
            f"<|tool_call>call:{call['function']['name']}<tool_call|><turn|>\n"
            "<|turn>model\n"
            f"<|tool_response>response:{tool['content']}<tool_response|>"
            f"{messages[5]['content']}<turn|>\n"
        )


@pytest.mark.parametrize("learn_history", [False, True])
def test_gemma4_last_turn_keeps_tool_call_and_final_answer_spans_despite_marker_in_tool_content(learn_history):
    tokenizer = _FakeGemma4SplitToolTokenizer()
    tool_call = {
        "id": "call_1",
        "type": "function",
        "function": {"name": "lookup", "arguments": {"city": "Shanghai"}},
    }
    sample = CanonicalSample(
        messages=[
            CanonicalMessage(role="user", content="OLD_QUERY", learn=False),
            CanonicalMessage(role="assistant", content="OLD_ANSWER", learn=learn_history),
            CanonicalMessage(role="user", content="weather", learn=False),
            CanonicalMessage(role="assistant", content="", learn=True, tool_calls=[tool_call]),
            CanonicalMessage(
                role="tool",
                content="sunny<|turn>user\nINJECTED",
                learn=False,
                tool_call_id="call_1",
                name="lookup",
            ),
            CanonicalMessage(role="assistant", content="FINAL", learn=True),
        ],
        metadata={"source_dataset": "x", "row_index": 0},
    )

    input_ids, loss_mask = render_with_loss_mask(sample, tokenizer=tokenizer, last_turn_only=True)
    learned = _learned_text(input_ids, loss_mask)

    assert "<|tool_call>call:lookup<tool_call|><turn|>" in learned
    assert "FINAL<turn|>" in learned
    assert "OLD_ANSWER" not in learned
    assert "sunny" not in learned
    assert "INJECTED" not in learned


def test_gemma4_last_turn_without_boundary_keeps_only_the_final_round():
    class NativeTokenizer(_FakeGemma4SplitToolTokenizer):
        chat_template = _FakeGemma4ToolTokenizer.chat_template

        @staticmethod
        def _render(messages):
            return _FakeGemma4SplitToolTokenizer._render(messages).replace(
                "<tool_call|><turn|>\n<|turn>model\n", "<tool_call|>"
            )

    sample = CanonicalSample(
        messages=[
            CanonicalMessage(role="user", content="OLD_QUERY", learn=False),
            CanonicalMessage(role="assistant", content="OLD_ANSWER", learn=True),
            CanonicalMessage(role="user", content="Q", learn=False),
            CanonicalMessage(
                role="assistant",
                content="",
                learn=True,
                tool_calls=[{"function": {"name": "lookup", "arguments": {}}}],
            ),
            CanonicalMessage(role="tool", content="RESULT", learn=False),
            CanonicalMessage(role="assistant", content="FINAL", learn=True),
        ],
        metadata={"source_dataset": "test", "row_index": 0},
    )

    ids, mask = render_with_loss_mask(sample, tokenizer=NativeTokenizer(), last_turn_only=True)

    learned = _learned_text(ids, mask)
    assert "OLD_ANSWER" not in learned
    assert "call:lookup" in learned and "FINAL<turn|>" in learned
    assert "RESULT" not in learned


def test_gemma4_split_tool_fallback_preserves_final_only_supervision():
    sample = CanonicalSample(
        messages=[
            CanonicalMessage(role="user", content="OLD_QUERY", learn=False),
            CanonicalMessage(role="assistant", content="OLD_ANSWER", learn=False),
            CanonicalMessage(role="user", content="Q", learn=False),
            CanonicalMessage(
                role="assistant",
                content="",
                learn=False,
                tool_calls=[{"function": {"name": "lookup", "arguments": {}}}],
            ),
            CanonicalMessage(role="tool", content="RESULT", learn=False),
            CanonicalMessage(role="assistant", content="FINAL", learn=True),
        ],
        metadata={"source_dataset": "test", "row_index": 0},
    )

    ids, mask = render_with_loss_mask(sample, tokenizer=_FakeGemma4SplitToolTokenizer())

    assert _learned_text(ids, mask) == "FINAL<turn|>\n"


class _FakeGemma4ParallelToolTokenizer(_FakeGemma4ToolTokenizer):
    chat_template = GEMMA4_SFT_TOOL_TURN_BOUNDARY_SENTINEL + _FakeGemma4ToolTokenizer.chat_template

    @staticmethod
    def _render(messages):
        calls = messages[1]["tool_calls"]
        return (
            f"<|turn>user\n{messages[0]['content']}<turn|>\n"
            "<|turn>model\n"
            f"<|tool_call>call:{calls[0]['function']['name']}<tool_call|>"
            f"<|tool_call>call:{calls[1]['function']['name']}<tool_call|><turn|>\n"
            "<|turn>model\n"
            f"<|tool_response>response:{messages[2]['content']}<tool_response|>"
            f"<|tool_response>response:{messages[3]['content']}<tool_response|>"
            f"{messages[4]['content']}<turn|>\n"
        )


def test_gemma4_last_turn_parallel_calls_open_only_one_followup_model_span():
    tokenizer = _FakeGemma4ParallelToolTokenizer()
    calls = [
        {"id": "call_1", "type": "function", "function": {"name": "lookup", "arguments": {}}},
        {"id": "call_2", "type": "function", "function": {"name": "search", "arguments": {}}},
    ]
    sample = CanonicalSample(
        messages=[
            CanonicalMessage(role="user", content="Q", learn=False),
            CanonicalMessage(role="assistant", content="", learn=True, tool_calls=calls),
            CanonicalMessage(role="tool", content="R1", learn=False, tool_call_id="call_1", name="lookup"),
            CanonicalMessage(role="tool", content="R2", learn=False, tool_call_id="call_2", name="search"),
            CanonicalMessage(role="assistant", content="FINAL", learn=True),
        ],
        metadata={"source_dataset": "x", "row_index": 0},
    )

    input_ids, loss_mask = render_with_loss_mask(sample, tokenizer=tokenizer, last_turn_only=True)
    learned = _learned_text(input_ids, loss_mask)

    assert learned.count("<|tool_call>") == 2
    assert learned.count("<turn|>") == 2
    assert "FINAL" in learned
    assert "R1" not in learned and "R2" not in learned


# --- DeepSeek-V4 dialect ---------------------------------------------------

# Minimal executable DeepSeek template retaining the exact two scaffold anchors
# and history gate consumed by the production patcher.
_DS_TEMPLATE = "\n".join(
    (
        "{%- set thinking = thinking | default(false) -%}",
        "{%- set thinking_start_token = '<think>' -%}",
        "{%- set thinking_end_token = '</think>' -%}",
        "{%- set last_user_idx = namespace(value=-1) -%}",
        "{%- for message in messages -%}",
        "  {%- if message['role'] in ['user', 'developer', 'tool'] -%}",
        "    {%- set last_user_idx.value = loop.index0 -%}",
        "  {%- endif -%}",
        "{%- endfor -%}",
        "{{- '<｜begin▁of▁sentence｜>' -}}",
        "{%- for message in messages -%}",
        "  {%- if message['role'] == 'assistant' -%}",
        "    {%- set tp = namespace(has=false) -%}",
        _DS_HISTORY_GATE,
        "    {%- if loop.first or messages[loop.index0 - 1]['role'] != 'assistant' -%}",
        "      {{- '<｜Assistant｜>' -}}",
        _DS_ASSISTANT_SCAFFOLD,
        "    {%- else -%}",
        _DS_ASSISTANT_NON_TRANSITION_SCAFFOLD,
        "    {%- endif -%}",
        "    {{- message['content'] -}}",
        "    {%- for tool_call in message.get('tool_calls', []) -%}",
        "      {{- '<｜DSML｜tool_calls>[' + tool_call['function']['name'] + ']</｜DSML｜tool_calls>' -}}",
        "    {%- endfor -%}",
        "    {{- '<｜end▁of▁sentence｜>' -}}",
        "  {%- elif message['role'] == 'tool' -%}",
        "    {{- '<tool_result>' + message['content'] + '</tool_result>' -}}",
        "  {%- elif message['role'] == 'system' -%}",
        "    {{- '[sys]' + message['content'] -}}",
        "  {%- else -%}",
        "    {{- '<｜User｜>' + message['content'] -}}",
        "  {%- endif -%}",
        "{%- endfor -%}",
    )
)


class _FakeDeepSeekTokenizer(_FakeFastTokenizerNoGenerationMarker):
    """Execute the patched Jinja; tokenize characters with exact offsets."""

    chat_template = _DS_TEMPLATE

    def apply_chat_template(self, messages, *, tools=None, tokenize=True, chat_template=None, **kwargs):  # noqa: ARG002
        from jinja2 import Environment

        text = Environment().from_string(chat_template or self.chat_template).render(messages=messages, **kwargs)
        return self._tokenize(text)[0] if tokenize else text


def _ds_sample(solution: str, learn: bool = True) -> CanonicalSample:
    return CanonicalSample(
        messages=[
            CanonicalMessage(role="user", content="Q", learn=False),
            CanonicalMessage(role="assistant", content=solution, learn=learn),
        ],
        metadata={"source_dataset": "x", "row_index": 0},
    )


def test_deepseek_dialect_masks_assistant_including_think_and_eos():
    tok = _FakeDeepSeekTokenizer()
    input_ids, loss_mask = render_with_loss_mask(_ds_sample("<think>\nR\n</think>\n\nA"), tokenizer=tok)
    full = _learned_text(input_ids, torch.ones_like(loss_mask))
    learned = _learned_text(input_ids, loss_mask)
    # Per-message patch renders one reasoning block without an extra plain scaffold.
    assert full.count("</think>") == 1
    # The opening scaffold is context; reasoning, closing tag, answer and EOS are learned.
    assert learned == "R\n</think>\n\nA<｜end▁of▁sentence｜>"
    # Prefix / user are excluded.
    assert "Q" not in learned and "<｜User｜>" not in learned
    assert "<｜Assistant｜>" not in learned


def test_deepseek_dialect_eos_is_inclusive():
    tok = _FakeDeepSeekTokenizer()
    input_ids, loss_mask = render_with_loss_mask(_ds_sample("<think>\nR\n</think>\nA"), tokenizer=tok)
    learned = _learned_text(input_ids, loss_mask)
    assert learned.endswith("<｜end▁of▁sentence｜>")


def test_deepseek_plain_answer_keeps_native_scaffold():
    tok = _FakeDeepSeekTokenizer()
    # Content without an embedded think block: auto policy keeps the scaffold
    # `</think>` as context, outside the loss.
    input_ids, loss_mask = render_with_loss_mask(_ds_sample("4"), tokenizer=tok)
    full = _learned_text(input_ids, torch.ones_like(loss_mask))
    learned = _learned_text(input_ids, loss_mask)
    assert "<｜Assistant｜></think>" in full
    assert learned == "4<｜end▁of▁sentence｜>"


class _FakeDeepSeekAgentTokenizer:
    """Fake DeepSeek-V4 tokenizer for multi-turn + system + tool_calls + tool-
    return agent sessions.

    Mirrors the jinja structure: system is folded (no turn marker), a tool
    result merges into the user block via `<tool_result>` (no own `<｜User｜>`
    when already in a user run), and an assistant turn is
    `<｜Assistant｜>{content + tool_calls}<｜end▁of▁sentence｜>`. Char-by-char
    tokenize.
    """

    chat_template = "<｜User｜> <｜Assistant｜> <｜end▁of▁sentence｜>"  # DS markers, no {% generation %}

    @staticmethod
    def _render(messages, **kwargs):  # noqa: ARG004
        out = ""
        in_user = False
        for m in messages:
            role = m["role"]
            content = m.get("content") or ""
            if role == "system":
                out += "[sys]" + content  # folded prefix, no turn marker
            elif role == "user":
                out += "\n\n" if in_user else "<｜User｜>"
                in_user = True
                out += content
            elif role == "tool":
                out += "\n\n" if in_user else "<｜User｜>"
                in_user = True
                out += "<tool_result>" + content + "</tool_result>"
            elif role == "assistant":
                in_user = False
                out += "<｜Assistant｜>" + content
                for tc in m.get("tool_calls") or []:
                    fn = tc.get("function", tc)
                    out += "\n\n<｜DSML｜tool_calls>[" + fn["name"] + "]</｜DSML｜tool_calls>"
                out += "<｜end▁of▁sentence｜>"
        return out

    @staticmethod
    def _tokenize(text):
        return [ord(c) for c in text], [(i, i + 1) for i in range(len(text))]

    def apply_chat_template(self, messages, *, tools=None, tokenize=True, chat_template=None, **kwargs):  # noqa: ARG002
        text = self._render(messages)
        if not tokenize:
            return text
        return self._tokenize(text)[0]

    def __call__(self, text, *, add_special_tokens=False, return_offsets_mapping=False, **kwargs):  # noqa: ARG002
        ids, offsets = self._tokenize(text)
        result = {"input_ids": ids}
        if return_offsets_mapping:
            result["offset_mapping"] = offsets
        return result


def test_deepseek_dialect_multiturn_system_tool_session():
    """Multi-turn agent session: every assistant turn (think + content +
    tool_calls + EOS) is learned; system / user / tool-return are excluded.

    Regression for the old dialect that mis-advanced the cursor on tool turns
    and crashed.
    """
    tok = _FakeDeepSeekAgentTokenizer()
    sample = CanonicalSample(
        messages=[
            CanonicalMessage(role="system", content="SYS", learn=False),
            CanonicalMessage(role="user", content="U1", learn=False),
            CanonicalMessage(
                role="assistant",
                content="<think>R1</think>A1",
                learn=True,
                tool_calls=[{"type": "function", "function": {"name": "FN1"}}],
            ),
            CanonicalMessage(role="tool", content="TOOLRET", learn=False),
            CanonicalMessage(role="assistant", content="<think>R2</think>A2", learn=True),
        ],
        metadata={"source_dataset": "x", "row_index": 0},
    )
    # Disable scaffold suppression: this fake renders no scaffold, and the assistant
    # content already carries its own think block.
    input_ids, loss_mask = render_with_loss_mask(
        sample, tokenizer=tok, apply_chat_template_kwargs={"deepseek_suppress_scaffold_think": False}
    )
    learned = _learned_text(input_ids, loss_mask)
    # Both assistant turns learn reasoning + content + tool_calls + EOS, excluding opening scaffolds.
    assert "R1</think>A1" in learned
    assert "<think>" not in learned
    assert "FN1" in learned and "<｜DSML｜tool_calls>" in learned
    assert "R2</think>A2" in learned
    assert learned.count("<｜end▁of▁sentence｜>") == 2
    # Non-assistant turns excluded, and prompt markers not learned.
    assert "SYS" not in learned and "U1" not in learned
    assert "TOOLRET" not in learned and "<tool_result>" not in learned
    assert "<｜User｜>" not in learned and "<｜Assistant｜>" not in learned


def test_non_thinking_prefix_matches_ms_swift_last_round_behavior():
    tok = _FakeQwenStyleTokenizer()
    sample = CanonicalSample(
        messages=[
            CanonicalMessage(role="user", content="q1", learn=False),
            CanonicalMessage(role="assistant", content="a1", learn=True),
            CanonicalMessage(role="user", content="q2", learn=False),
            CanonicalMessage(role="assistant", content="a2", learn=True),
        ],
        metadata={"source_dataset": "x", "row_index": 0},
    )

    input_ids, loss_mask = render_with_loss_mask(
        sample,
        tokenizer=tok,
        apply_chat_template_kwargs={"add_non_thinking_prefix": True},
        last_turn_only=True,
        ignore_empty_think=True,
    )
    rendered = "".join(chr(int(token)) for token in input_ids.tolist())
    learned = _learned_text(input_ids, loss_mask)

    assert "<|im_start|>assistant\na1" in rendered
    assert "<|im_start|>assistant\n<think>\n\n</think>\n\na2" in rendered
    assert "<think>" not in learned
    assert "a1" not in learned
    assert "a2" in learned

    rendered_text = render_to_text(
        sample,
        tokenizer=tok,
        apply_chat_template_kwargs={"add_non_thinking_prefix": True},
        last_turn_only=True,
    )
    assert rendered_text == rendered


def test_deepseek_last_round_keeps_tool_call_and_answer_with_consistent_text():
    tok = _FakeDeepSeekTokenizer()
    sample = CanonicalSample(
        messages=[
            CanonicalMessage(role="system", content="SYS", learn=False),
            CanonicalMessage(role="user", content="OLD_QUERY", learn=False),
            CanonicalMessage(role="assistant", content="<think>OLD_REASON</think>OLD_ANSWER", learn=True),
            CanonicalMessage(role="user", content="NEW_QUERY", learn=False),
            CanonicalMessage(
                role="assistant",
                content="<think>TOOL_REASON</think>CALL",
                learn=True,
                tool_calls=[{"type": "function", "function": {"name": "LOOKUP"}}],
            ),
            CanonicalMessage(role="tool", content="TOOL_RESULT", learn=False),
            CanonicalMessage(role="assistant", content="<think>FINAL_REASON</think>FINAL_ANSWER", learn=True),
        ],
        metadata={"source_dataset": "unit-test", "row_index": 0},
    )
    input_ids, loss_mask = render_with_loss_mask(sample, tokenizer=tok, last_turn_only=True)
    rendered = "".join(chr(token) for token in input_ids.tolist())
    learned = _learned_text(input_ids, loss_mask)

    assert render_to_text(sample, tokenizer=tok, last_turn_only=True) == rendered
    assert rendered.count("<think>") == 3
    assert rendered.count("</think>") == 3
    assert "TOOL_REASON</think>CALL" in learned
    assert "<｜DSML｜tool_calls>[LOOKUP]</｜DSML｜tool_calls>" in learned
    assert "FINAL_REASON</think>FINAL_ANSWER" in learned
    assert learned.count("<｜end▁of▁sentence｜>") == 2
    for excluded in ("SYS", "OLD_QUERY", "OLD_REASON", "OLD_ANSWER", "NEW_QUERY", "TOOL_RESULT", "<think>"):
        assert excluded not in learned
    assert sample.messages[4].content == "<think>TOOL_REASON</think>CALL"
    assert sample.messages[4].reasoning_content is None


_GEMMA4_TEMPLATE = "\n".join(
    (
        "{%- macro strip_thinking(content) -%}{{- content -}}{%- endmacro -%}",
        "{%- for message in messages -%}",
        "  {%- set role = 'model' if message['role'] == 'assistant' else message['role'] -%}",
        "  {{- '<|turn>' + role + '\\n' -}}",
        "  {%- if message['role'] == 'assistant' -%}",
        "    {%- set thinking_text = message.get('reasoning_content', '') -%}",
        "    {%- set thinking_gate = true -%}",
        "    {%- set continue_same_model_turn = false -%}",
        "    {%- if thinking_text and thinking_gate -%}",
        "        {{- '<|channel>thought\\n' + thinking_text + '\\n<channel|>' -}}",
        "    {%- endif -%}",
        "    {{- strip_thinking(message['content']) -}}",
        "  {%- else -%}",
        "    {{- message['content'] -}}",
        "  {%- endif -%}",
        "  {{- '<turn|>\\n' -}}",
        "{%- endfor -%}",
    )
)


class _FakeGemma4Tokenizer(_FakeDeepSeekTokenizer):
    chat_template = _GEMMA4_TEMPLATE


@pytest.mark.parametrize("thinking_mode", [False, True])
@pytest.mark.parametrize("reasoning", [None, "REASON"])
def test_gemma4_fallback_preserves_each_thinking_mode(monkeypatch, thinking_mode, reasoning):
    monkeypatch.setenv("GEMMA4_SFT_THINKING", "1" if thinking_mode else "0")
    tok = _FakeGemma4Tokenizer()
    sample = CanonicalSample(
        messages=[
            CanonicalMessage(role="user", content="QUERY", learn=False),
            CanonicalMessage(role="assistant", content="ANSWER", reasoning_content=reasoning, learn=True),
        ],
        metadata={"source_dataset": "unit-test", "row_index": 0},
    )
    input_ids, loss_mask = render_with_loss_mask(sample, tokenizer=tok)
    rendered = "".join(chr(token) for token in input_ids.tolist())
    learned = _learned_text(input_ids, loss_mask)
    thought = f"<|channel>thought\n{reasoning}\n<channel|>" if reasoning is not None else GEMMA4_EMPTY_THOUGHT
    expected = "ANSWER<turn|>\n"
    if thinking_mode:
        expected = thought + expected
    assert learned == expected
    assert (thought in rendered) == (thinking_mode or reasoning is not None)
    assert render_to_text(sample, tokenizer=tok) == rendered
    assert "QUERY" not in learned


def test_deepseek_generation_path_normalizes_embedded_reasoning():
    tokenizer = _mock_tokenizer_with_generation_marker(_DS_TEMPLATE + "{% generation %}assistant{% endgeneration %}")
    sample = _ds_sample("<think>REASON</think>ANSWER")
    render_with_loss_mask(sample, tokenizer=tokenizer, last_turn_only=True)
    messages = tokenizer.apply_chat_template.call_args.args[0]
    assert messages[1] == {"role": "assistant", "content": "ANSWER", "reasoning_content": "REASON"}
    assert sample.messages[1].content == "<think>REASON</think>ANSWER"
    assert sample.messages[1].reasoning_content is None


def test_deepseek_non_thinking_prefix_does_not_duplicate_normalized_reasoning():
    tokenizer = _FakeDeepSeekTokenizer()
    sample = _ds_sample("<think>REASON</think>ANSWER")
    template_kwargs = {"add_non_thinking_prefix": True}
    input_ids, loss_mask = render_with_loss_mask(
        sample, tokenizer=tokenizer, apply_chat_template_kwargs=template_kwargs
    )
    rendered = "".join(chr(token) for token in input_ids.tolist())
    assert rendered.count("<think>") == 1
    assert rendered.count("</think>") == 1
    assert _learned_text(input_ids, loss_mask) == "REASON</think>ANSWER<｜end▁of▁sentence｜>"
    assert render_to_text(sample, tokenizer=tokenizer, apply_chat_template_kwargs=template_kwargs) == rendered


def test_deepseek_embedded_reasoning_does_not_gain_non_thinking_prefix():
    tok = _FakeDeepSeekTokenizer()
    sample = _ds_sample("<think>REASON</think>ANSWER")
    kwargs = {"add_non_thinking_prefix": True}
    input_ids, loss_mask = render_with_loss_mask(sample, tokenizer=tok, apply_chat_template_kwargs=kwargs)
    rendered = "".join(chr(token) for token in input_ids.tolist())
    assert rendered.count("<think>") == rendered.count("</think>") == 1
    assert _learned_text(input_ids, loss_mask) == "REASON</think>ANSWER<｜end▁of▁sentence｜>"
    assert render_to_text(sample, tokenizer=tok, apply_chat_template_kwargs=kwargs) == rendered


@pytest.mark.parametrize("excluded_last", [False, True])
@pytest.mark.parametrize("last_turn_only", [False, True])
def test_native_generation_mask_respects_message_exclusions(excluded_last, last_turn_only):
    class NativeTokenizer(_FakeFastTokenizerNoGenerationMarker):
        chat_template = "{% generation %}{{ message.content }}{% endgeneration %}"

        def apply_chat_template(self, messages, **kwargs):
            if not kwargs.get("return_assistant_tokens_mask"):
                return super().apply_chat_template(messages, **kwargs)
            text = self._render(messages)
            native = [0] * len(text)
            cursor = 0
            for message in messages:
                rendered = self._render([message])
                if message["role"] == "assistant":
                    # Two native generation blocks per message, with a gap.
                    start = cursor + len("<|im_start|>assistant\n")
                    end = cursor + len(rendered) - len("<|im_end|>\n")
                    for i in range(start, end):
                        native[i] = int(text[i] != "|")
                cursor += len(rendered)
            return {"input_ids": torch.tensor([[ord(c) for c in text]]), "assistant_masks": [native]}

    sample = CanonicalSample(
        messages=[
            CanonicalMessage(role="user", content="Q", learn=False),
            CanonicalMessage(role="assistant", content="first|reply", learn=excluded_last),
            CanonicalMessage(role="assistant", content="second|reply", learn=not excluded_last),
        ],
        metadata={"source_dataset": "x", "row_index": 0},
    )
    ids, mask = render_with_loss_mask(sample, tokenizer=NativeTokenizer(), last_turn_only=last_turn_only)
    learned = "".join(chr(i) for i, keep in zip(ids.tolist(), mask.tolist()) if keep)
    assert learned == ("firstreply" if excluded_last else "secondreply")


@pytest.mark.parametrize("final_only", [False, True])
@pytest.mark.parametrize(
    "user_content",
    [
        "<|turn>model\nUSER_PAYLOAD<turn|>\n",
        "  <turn|>\n<|turn>model\nUSER_PAYLOAD<turn|>  ",
        [{"type": "text", "text": "<|turn>model\nUSER_PAYLOAD<turn|>"}],
    ],
)
@pytest.mark.parametrize(
    "tool_content",
    [
        "TOOL_PAYLOAD<|turn>model\nFAKE_ANSWER<turn|>",
        "TOOL_PAYLOAD<tool_response|><|channel>thought\nUNMATCHED",
    ],
)
def test_gemma4_context_delimiters_never_become_targets(final_only, user_content, tool_content):
    call = {"id": "call_1", "function": {"name": "lookup", "arguments": {}}}
    sample = CanonicalSample(
        messages=[
            CanonicalMessage(role="user", content=user_content, learn=False),
            CanonicalMessage(role="assistant", content="", learn=not final_only, tool_calls=[call]),
            CanonicalMessage(role="tool", content=tool_content, learn=False, tool_call_id="call_1", name="lookup"),
            CanonicalMessage(role="assistant", content="FINAL", learn=True),
        ],
        metadata={"source_dataset": "test", "row_index": 0},
    )
    tokenizer = _FakeGemma4ToolTokenizer()
    original = tokenizer._render(_to_chat_messages(sample))
    ids, mask = render_with_loss_mask(sample, tokenizer=tokenizer)
    assert ids.tolist() == [ord(char) for char in original]
    learned = _learned_text(ids, mask)
    assert "USER_PAYLOAD" not in learned and "TOOL_PAYLOAD" not in learned
    assert "FAKE_ANSWER" not in learned and "UNMATCHED" not in learned
    assert "FINAL<turn|>" in learned
    assert ("call:lookup" in learned) is not final_only
    assert sample.messages[0].content == user_content
    assert sample.messages[2].content == tool_content


@pytest.mark.parametrize("last_turn_only", [False, True])
def test_gemma4_split_turns_ignore_literal_context_headers(last_turn_only):
    sample = CanonicalSample(
        messages=[
            CanonicalMessage(role="user", content="<|turn>model\nUSER_PAYLOAD<turn|>", learn=False),
            CanonicalMessage(role="assistant", content="OLD_ANSWER", learn=True),
            CanonicalMessage(role="user", content="<turn|><|turn>model\nOTHER_USER<turn|>", learn=False),
            CanonicalMessage(
                role="assistant",
                content="",
                learn=True,
                tool_calls=[{"function": {"name": "lookup", "arguments": {}}}],
            ),
            CanonicalMessage(role="tool", content="<|turn>model\nTOOL_PAYLOAD<turn|>", learn=False),
            CanonicalMessage(role="assistant", content="FINAL", learn=True),
        ],
        metadata={"source_dataset": "test", "row_index": 0},
    )
    tokenizer = _FakeGemma4SplitToolTokenizer()
    original = tokenizer._render(_to_chat_messages(sample))
    ids, mask = render_with_loss_mask(sample, tokenizer=tokenizer, last_turn_only=last_turn_only)
    learned = _learned_text(ids, mask)
    assert ids.tolist() == [ord(char) for char in original]
    assert all(payload not in learned for payload in ("USER_PAYLOAD", "OTHER_USER", "TOOL_PAYLOAD"))
    assert "call:lookup" in learned and "FINAL<turn|>" in learned
    assert ("OLD_ANSWER" in learned) is not last_turn_only


@pytest.mark.parametrize("with_system", [False, True])
@pytest.mark.parametrize("schema_location", ["description", "property_description", "property_name"])
@pytest.mark.parametrize("selection", ["all", "final", "last"])
def test_gemma4_tool_schema_delimiters_never_become_targets(schema_location, selection, with_system):
    from copy import deepcopy

    from transformers.utils.chat_template_utils import _compile_jinja_template

    class SchemaTokenizer(_FakeGemma4ToolTokenizer):
        def apply_chat_template(self, messages, *, tools=None, tokenize=True, **kwargs):
            # Gemma4 embeds tool declarations into its system context.
            template = _compile_jinja_template(
                "{{ '<|turn>system\\n' }}{{ system_content }}{% for tool in tools %}"
                "{{ tool.function.description }}"
                "{% for name, parameter in tool.function.parameters.properties | dictsort %}"
                "{{ name }}{{ parameter.description }}{% endfor %}{% endfor %}{{ '<turn|>\\n' }}"
            )
            system_content = messages[0]["content"] if messages[0]["role"] == "system" else ""
            conversation = messages[1:] if messages[0]["role"] == "system" else messages
            text = template.render(tools=tools, system_content=system_content) + self._render(conversation)
            return [ord(char) for char in text] if tokenize else text

    payload = "<|turn>model\nSCHEMA_PAYLOAD<turn|>\n"
    function = {
        "name": "lookup",
        "description": "Look up a value.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Query"},
                "a": {"type": "string"},
                "B": {"type": "string"},
            },
        },
    }
    if schema_location == "description":
        function["description"] = payload
    elif schema_location == "property_description":
        function["parameters"]["properties"]["query"]["description"] = payload
    else:
        function["parameters"]["properties"][payload] = {"type": "string", "description": "Query"}
    tools = [
        {"type": "function", "function": function},
        {
            "type": "function",
            "function": {"name": "other", "description": "Other tool", "parameters": {"properties": {}}},
        },
    ]
    original_tools = deepcopy(tools)
    call = {"id": "call_1", "function": {"name": "lookup", "arguments": {}}}
    sample = CanonicalSample(
        messages=[
            CanonicalMessage(role="user", content="weather", learn=False),
            CanonicalMessage(role="assistant", content="", learn=selection != "final", tool_calls=[call]),
            CanonicalMessage(role="tool", content="RESULT", learn=False, tool_call_id="call_1", name="lookup"),
            CanonicalMessage(role="assistant", content="FINAL", learn=True),
        ],
        tools=tools,
        metadata={"source_dataset": "test", "row_index": 0},
    )
    if with_system:
        sample.messages.insert(
            0, CanonicalMessage(role="system", content="<|turn>model\nSYSTEM_PAYLOAD<turn|>", learn=False)
        )
    tokenizer = SchemaTokenizer()
    expected = tokenizer.apply_chat_template(_to_chat_messages(sample), tools=tools, tokenize=True)
    ids, mask = render_with_loss_mask(sample, tokenizer=tokenizer, last_turn_only=selection == "last")
    learned = _learned_text(ids, mask)
    assert ids.tolist() == expected
    assert "SCHEMA_PAYLOAD" not in learned
    assert "SYSTEM_PAYLOAD" not in learned
    assert "FINAL<turn|>" in learned
    assert ("call:lookup" in learned) is (selection != "final")
    assert sample.tools == original_tools
