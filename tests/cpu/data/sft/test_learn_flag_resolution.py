# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Per-message loss-flag resolution, including the ms-swift ``loss``
convention."""

import pytest

from relax.engine.sft.dataset.streaming import _canonicalize_messages, _resolve_learn_flag


@pytest.mark.parametrize("role", ["assistant", "function_call"])
def test_learn_flag_defaults_to_true_for_learnable_roles(role):
    assert _resolve_learn_flag({"role": role, "content": "x"}, role) is True


@pytest.mark.parametrize("role", ["user", "system", "tool"])
def test_learn_flag_defaults_to_false_for_non_learnable_roles(role):
    assert _resolve_learn_flag({"role": role, "content": "x"}, role) is False


@pytest.mark.parametrize("value,expected", [(True, True), (False, False)])
def test_learn_flag_honors_explicit_learn(value, expected):
    assert _resolve_learn_flag({"role": "assistant", "learn": value}, "assistant") is expected


@pytest.mark.parametrize("value,expected", [(True, True), (False, False)])
def test_learn_flag_honors_ms_swift_loss_field(value, expected):
    """``loss`` is ms-swift's spelling of the same flag.

    Ignoring it silently trains on turns the dataset explicitly excluded.
    """
    assert _resolve_learn_flag({"role": "assistant", "loss": value}, "assistant") is expected


def test_learn_flag_prefers_learn_over_loss_when_both_present():
    """Relax's native flag wins so existing datasets keep their behavior."""
    assert _resolve_learn_flag({"role": "assistant", "learn": True, "loss": False}, "assistant") is True
    assert _resolve_learn_flag({"role": "assistant", "learn": False, "loss": True}, "assistant") is False


def test_learn_flag_loss_false_overrides_role_default():
    """The regression this guards: a `loss: false` assistant turn must not be
    trained."""
    assert _resolve_learn_flag({"role": "assistant", "content": "x", "loss": False}, "assistant") is False


@pytest.mark.parametrize("flags,expected", [({"loss": False}, False), ({"learn": True, "loss": False}, True)])
def test_reasoning_content_survives_loss_flag_resolution(flags, expected):
    messages = _canonicalize_messages(
        [{"role": "assistant", "content": "answer", "reasoning_content": "reason", **flags}],
        require_response=False,
    )
    assert messages[0].reasoning_content == "reason"
    assert messages[0].content == "answer"
    assert messages[0].learn is expected
