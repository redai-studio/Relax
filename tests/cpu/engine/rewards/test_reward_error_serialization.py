# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import pickle

import pytest


@pytest.mark.parametrize("serializer", ["pickle", "cloudpickle"])
def test_reward_error_serialization_preserves_context_cause_and_notes(serializer):
    ray = pytest.importorskip("ray")
    from relax.engine.rewards import RewardExecutionError

    codec = pickle if serializer == "pickle" else ray.cloudpickle
    cause = ValueError("invalid reward result")
    error = RewardExecutionError(
        custom_rm_path="custom.reward.score",
        sample_context="session_id='session-7', group_index=3",
        cause=cause,
    )
    if hasattr(error, "add_note"):
        error.add_note("reward batch context")

    restored = codec.loads(codec.dumps(error))

    assert type(restored) is RewardExecutionError
    assert str(restored) == str(error)
    assert restored.custom_rm_path == error.custom_rm_path
    assert restored.sample_context == error.sample_context
    assert isinstance(restored.cause, ValueError)
    assert str(restored.cause) == str(cause)
    assert restored.__cause__ is restored.cause
    if hasattr(error, "__notes__"):
        assert restored.__notes__ == error.__notes__
