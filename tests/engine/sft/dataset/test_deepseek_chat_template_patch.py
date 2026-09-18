# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Unit tests for the DeepSeek-V4 SFT chat-template scaffold patcher."""

import pytest

from relax.engine.sft.dataset.chat_template_patch import apply_chat_template_patchers
from relax.engine.sft.dataset.deepseek_chat_template_patch import (
    _DS_ASSISTANT_NON_TRANSITION_SCAFFOLD,
    _DS_ASSISTANT_SCAFFOLD,
    _EMBEDDED_THINK_INDICES_KEY,
    _PER_MESSAGE_GUARD,
    _patch_deepseek_scaffold,
    try_patch_deepseek_chat_template,
)
from relax.engine.sft.dataset.qwen_chat_template_patch import try_patch_qwen_chat_template
from relax.engine.sft.dataset.sample import CanonicalMessage, CanonicalSample


# Both official scaffold anchors must be present for the guarded rewrite.
_DS_TEMPLATE = "\n".join(
    (
        "<｜User｜>{{ x }}<｜Assistant｜>",
        _DS_ASSISTANT_SCAFFOLD,
        _DS_ASSISTANT_NON_TRANSITION_SCAFFOLD,
        "{{ content }}",
    )
)


def _sample(solution: str, learn: bool = True) -> CanonicalSample:
    return CanonicalSample(
        messages=[
            CanonicalMessage(role="user", content="Q", learn=False),
            CanonicalMessage(role="assistant", content=solution, learn=learn),
        ],
        metadata={"source_dataset": "x", "row_index": 0},
    )


def test_returns_none_for_non_deepseek_template():
    assert try_patch_deepseek_chat_template(_sample("<think>a</think>b"), "<|im_start|>user\n", {}) is None
    assert try_patch_deepseek_chat_template(_sample("x"), None, {}) is None


def test_suppresses_scaffold_when_content_has_think():
    res = try_patch_deepseek_chat_template(_sample("<think>r</think>ans"), _DS_TEMPLATE, {})
    assert res is not None and res.changed
    assert res.template.count(_PER_MESSAGE_GUARD) == 2
    assert res.kwargs[_EMBEDDED_THINK_INDICES_KEY] == (1,)
    assert "<｜Assistant｜>" in res.template  # marker preserved


def test_keeps_scaffold_for_plain_answer():
    res = try_patch_deepseek_chat_template(_sample("just an answer"), _DS_TEMPLATE, {})
    assert res is not None and not res.changed
    assert res.template == _DS_TEMPLATE


def test_explicit_override_true_suppresses_even_without_think():
    res = try_patch_deepseek_chat_template(_sample("plain"), _DS_TEMPLATE, {"deepseek_suppress_scaffold_think": True})
    assert res.changed and res.template.count(_PER_MESSAGE_GUARD) == 2
    assert res.kwargs[_EMBEDDED_THINK_INDICES_KEY] == (1,)
    assert "deepseek_suppress_scaffold_think" not in res.kwargs  # popped before render


def test_explicit_override_false_keeps_scaffold_even_with_think():
    res = try_patch_deepseek_chat_template(
        _sample("<think>r</think>a"), _DS_TEMPLATE, {"deepseek_suppress_scaffold_think": False}
    )
    assert not res.changed
    assert "deepseek_suppress_scaffold_think" not in res.kwargs


def test_invalid_override_type_raises():
    with pytest.raises(ValueError):
        try_patch_deepseek_chat_template(_sample("x"), _DS_TEMPLATE, {"deepseek_suppress_scaffold_think": "yes"})


def test_count_guard_raises_when_scaffold_missing():
    with pytest.raises(RuntimeError):
        _patch_deepseek_scaffold("<｜User｜><｜Assistant｜> no scaffold block here")


def test_scaffold_guards_preserve_native_branches_and_are_idempotent():
    patched = _patch_deepseek_scaffold(_DS_TEMPLATE)
    assert patched.count(_PER_MESSAGE_GUARD) == 2
    assert _DS_ASSISTANT_SCAFFOLD in patched
    assert _DS_ASSISTANT_NON_TRANSITION_SCAFFOLD in patched
    assert patched.count("<｜Assistant｜>") == 1
    assert _patch_deepseek_scaffold(patched) == patched


def test_no_collision_with_qwen_patcher():
    # A DeepSeek template must be claimed only by the DeepSeek patcher.
    res = apply_chat_template_patchers(
        _sample("<think>r</think>a"),
        _DS_TEMPLATE,
        {},
        patchers=(try_patch_qwen_chat_template, try_patch_deepseek_chat_template),
    )
    assert res.patch_name == "deepseek_v4_scaffold_think"


def test_embedded_think_indices_preserve_plain_historical_assistant():
    sample = CanonicalSample(
        messages=[
            CanonicalMessage(role="user", content="Q1", learn=False),
            CanonicalMessage(role="assistant", content="plain answer", learn=True),
            CanonicalMessage(role="user", content="Q2", learn=False),
            CanonicalMessage(role="assistant", content="<think>r</think>answer", learn=True),
        ],
        metadata={"source_dataset": "unit-test", "row_index": 0},
    )
    result = try_patch_deepseek_chat_template(sample, _DS_TEMPLATE, {})
    assert result.kwargs[_EMBEDDED_THINK_INDICES_KEY] == (3,)
    assert sample.messages[3].content == "<think>r</think>answer"
