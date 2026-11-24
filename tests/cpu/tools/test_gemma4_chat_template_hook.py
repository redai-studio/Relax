# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from types import SimpleNamespace

import pytest

from scripts.tools import gemma4_chat_template_hook as hook


def _gemma4_template():
    return (
        "<|turn><channel|>{{ strip_thinking('x') }}\n"
        "    {%- if thinking_text and thinking_gate -%}\n"
        "        {{- '<|channel>thought\\n' + thinking_text + '\\n<channel|>' -}}\n"
        "    {%- endif -%}\n"
        "{%- macro format_tool_response_block(tool_name, response) -%}\n"
        "{{- '<|tool_response>' + response + '<tool_response|>' -}}\n"
        "{%- endmacro -%}\n"
        "            {%- set ns_tr_out = namespace(flag=false) -%}\n"
        "                    {{- format_tool_response_block('lookup', 'ok') -}}\n"
        "        {%- if ns.prev_message_type == 'tool_call' and not ns_tr_out.flag -%}\n"
        "            {{- '<|tool_response>' -}}\n"
    )


def test_patches_exported_gemma4_chat_template_once(monkeypatch, tmp_path):
    monkeypatch.setenv("GEMMA4_SFT_THINKING", "1")
    monkeypatch.setenv("GEMMA4_SFT_TOOL_TURN_BOUNDARY", "1")
    path = tmp_path / "chat_template.jinja"
    path.write_text(_gemma4_template())
    args = SimpleNamespace()

    hook.patch_gemma4_chat_template(args, str(tmp_path), 1, dtype="bf16", is_lora=False)
    first = path.read_text()
    hook.patch_gemma4_chat_template(args, str(tmp_path), 2, dtype="bf16", is_lora=False)

    assert path.read_text() == first
    assert first.count("relax_gemma4_sft_thinking_v1") == 1
    assert first.count("relax_gemma4_sft_tool_turn_boundary_v1") == 1
    assert "{{- '<turn|>\\n<|turn>model\\n' -}}" in first
    assert "{{- '<|tool_response>' -}}" not in first


def test_does_not_patch_chat_template_when_gemma4_sft_mode_is_disabled(monkeypatch, tmp_path):
    monkeypatch.delenv("GEMMA4_SFT_THINKING", raising=False)
    monkeypatch.delenv("GEMMA4_SFT_TOOL_TURN_BOUNDARY", raising=False)
    monkeypatch.delenv("GEMMA4_SFT_MS_SWIFT_COMPAT", raising=False)
    path = tmp_path / "chat_template.jinja"
    original = _gemma4_template()
    path.write_text(original)

    hook.patch_gemma4_chat_template(SimpleNamespace(), str(tmp_path), 1, dtype="bf16", is_lora=False)

    assert path.read_text() == original


@pytest.mark.parametrize("compat", [False, True])
@pytest.mark.parametrize("thinking,boundary", [(False, False), (False, True), (True, False), (True, True)])
def test_exported_tokenizer_preserves_training_render(compat, thinking, boundary, monkeypatch, tmp_path):
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from transformers import PreTrainedTokenizerFast

    from relax.engine.sft.dataset.chat_template import render_to_text
    from relax.engine.sft.dataset.sample import CanonicalMessage, CanonicalSample

    for flag, enabled in (
        ("GEMMA4_SFT_MS_SWIFT_COMPAT", compat),
        ("GEMMA4_SFT_THINKING", thinking),
        ("GEMMA4_SFT_TOOL_TURN_BOUNDARY", boundary),
    ):
        monkeypatch.setenv(flag, str(int(enabled)))
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(WordLevel({"<unk>": 0}, unk_token="<unk>")), unk_token="<unk>"
    )
    # Executable Jinja preserving the production thought/tool-response patch anchors.
    tokenizer.chat_template = (
        "{%- macro strip_thinking(text) -%}{{ text }}{%- endmacro -%}\n"
        "{%- set role = 'model' -%}{%- set continue_same_model_turn = false -%}\n"
        "{%- set thinking_text = '' -%}{%- set thinking_gate = false -%}\n"
        "{%- set message = messages[1] -%}{%- set ns = namespace(prev_message_type='resolved') -%}\n"
        + _gemma4_template()
        .replace(
            "format_tool_response_block('lookup', 'ok')",
            "format_tool_response_block(messages[2].get('name', 'unknown'), messages[2]['content'])",
        )
        .replace("'<|tool_response>' + response", "'<|tool_response>response:' + tool_name + ':' + response")
        + "{%- endif -%}"
    )
    calls = [{"id": "call_1", "function": {"name": "lookup", "arguments": {}}}]
    sample = CanonicalSample(
        messages=[
            CanonicalMessage(role="user", content="Q", learn=False),
            CanonicalMessage(role="assistant", content="", learn=True, tool_calls=calls),
            CanonicalMessage(role="tool", content="RESULT", learn=False, name="lookup", tool_call_id="call_1"),
        ],
        metadata={"source_dataset": "test", "row_index": 0},
    )
    training = render_to_text(sample, tokenizer=tokenizer)
    tokenizer.save_pretrained(tmp_path)
    hook.patch_gemma4_chat_template(SimpleNamespace(), str(tmp_path), 1, dtype="bf16", is_lora=False)
    first_template = (tmp_path / "chat_template.jinja").read_text()
    hook.patch_gemma4_chat_template(SimpleNamespace(), str(tmp_path), 2, dtype="bf16", is_lora=False)
    assert (tmp_path / "chat_template.jinja").read_text() == first_template
    # Serving has no Relax flags or Python preprocessing after reload.
    for flag in ("GEMMA4_SFT_MS_SWIFT_COMPAT", "GEMMA4_SFT_THINKING", "GEMMA4_SFT_TOOL_TURN_BOUNDARY"):
        monkeypatch.delenv(flag)
    exported = PreTrainedTokenizerFast.from_pretrained(tmp_path)
    messages = [
        {"role": "user", "content": "Q"},
        {"role": "assistant", "content": "", "tool_calls": calls},
        {"role": "tool", "content": "RESULT", "name": "lookup", "tool_call_id": "call_1"},
    ]
    served = exported.apply_chat_template(messages, tokenize=False)
    assert training == served
    assert f"response:{'unknown' if compat else 'lookup'}:RESULT" in served
    assert "name" in messages[2] and "tool_call_id" in messages[2]
