# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Unit tests for the model-specific chat-template patch dispatcher."""

import pytest

from relax.engine.sft.dataset.chat_template_patch import TemplatePatchResult, apply_chat_template_patchers
from relax.engine.sft.dataset.sample import CanonicalMessage, CanonicalSample


def _sample() -> CanonicalSample:
    return CanonicalSample(
        messages=[CanonicalMessage(role="user", content="hello", learn=False)],
        metadata={"source_dataset": "x", "row_index": 0},
    )


def test_chat_template_patchers_unknown_template_is_noop_without_mutation():
    kwargs = {"enable_thinking": True}

    def no_match(sample, template, patch_kwargs):  # noqa: ARG001
        patch_kwargs["mutated"] = True
        return None

    result = apply_chat_template_patchers(_sample(), "plain", kwargs, patchers=(no_match,))
    assert result == TemplatePatchResult(template="plain", kwargs=kwargs)
    assert kwargs == {"enable_thinking": True}


def test_chat_template_patchers_returns_unique_match():
    def match(sample, template, patch_kwargs):  # noqa: ARG001
        return TemplatePatchResult(
            template=f"{template}-patched",
            kwargs={**patch_kwargs, "flag": True},
            patch_name="test",
            changed=True,
        )

    result = apply_chat_template_patchers(_sample(), "plain", {}, patchers=(match,))
    assert result.template == "plain-patched"
    assert result.kwargs == {"flag": True, "chat_template": "plain-patched"}
    assert result.patch_name == "test"
    assert result.changed


def test_chat_template_patchers_rejects_multiple_matches():
    def match_one(sample, template, patch_kwargs):  # noqa: ARG001
        return TemplatePatchResult(template=template, kwargs=dict(patch_kwargs), patch_name="one")

    def match_two(sample, template, patch_kwargs):  # noqa: ARG001
        return TemplatePatchResult(template=template, kwargs=dict(patch_kwargs), patch_name="two")

    with pytest.raises(RuntimeError, match="one.*two"):
        apply_chat_template_patchers(_sample(), "plain", {}, patchers=(match_one, match_two))


def test_chat_template_patchers_rejects_inconsistent_unchanged_result():
    def inconsistent(sample, template, patch_kwargs):  # noqa: ARG001
        return TemplatePatchResult(
            template=template,
            kwargs={**patch_kwargs, "chat_template": "other"},
            patch_name="bad",
        )

    with pytest.raises(RuntimeError, match="inconsistent template and kwargs"):
        apply_chat_template_patchers(_sample(), "plain", {}, patchers=(inconsistent,))
