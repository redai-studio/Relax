# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from __future__ import annotations

import json

import pytest
from relax_nemo_gym_example.app.protocol import TrialResult, TrialStatus
from relax_nemo_gym_example.app.result import to_relax_output, write_relax_output

from relax.agentic.runner import SessionOutput


def test_scalar_reward_wrapper_maps_to_relax_reward_and_metadata():
    result = TrialResult(
        request_id="request-1",
        status=TrialStatus.COMPLETED,
        reward={"scalar": 0.75, "components": {"verifier": 1.0, "format": 0.5}},
        metrics={"agent_steps": 3},
        artifact_ref="artifact://request-1",
    )

    payload = to_relax_output(result)

    assert payload["reward"] == 0.75
    assert payload["metadata"]["nemo_gym"]["status"] == "completed"
    assert payload["metadata"]["nemo_gym"]["reward_components"] == {"verifier": 1.0, "format": 0.5}
    parsed = SessionOutput.from_payload(payload)
    assert parsed.reward == 0.75
    assert parsed.metadata == payload["metadata"]


@pytest.mark.parametrize("reward", [0.0, 1.0, {"verifier": 1.0}])
def test_relax_reward_shapes_are_preserved(reward):
    result = TrialResult(request_id="request-1", status=TrialStatus.TRUNCATED, reward=reward)

    payload = to_relax_output(result)

    assert payload["reward"] == reward
    assert SessionOutput.from_payload(payload).reward == reward


def test_non_success_result_cannot_be_materialized():
    result = TrialResult(request_id="request-1", status=TrialStatus.FAILED)

    with pytest.raises(ValueError, match="non-success"):
        to_relax_output(result)


@pytest.mark.parametrize("status", [TrialStatus.COMPLETED, TrialStatus.TRUNCATED])
def test_terminal_result_without_reward_cannot_be_materialized(status):
    result = TrialResult(request_id="request-1", status=status, reward=None)

    with pytest.raises(ValueError, match="NeMo Gym trial has no reward:.*error_code=missing_reward"):
        to_relax_output(result)


def test_output_drops_gateway_error_message_that_could_contain_secrets():
    result = TrialResult(
        request_id="request-1",
        status=TrialStatus.TRUNCATED,
        reward=0.0,
        error={"code": "deadline_exceeded", "type": "timeout", "message": "token=session-secret"},
    )

    payload = to_relax_output(result)

    assert payload["metadata"]["nemo_gym"]["error"] == {
        "code": "deadline_exceeded",
        "type": "timeout",
    }
    assert "session-secret" not in json.dumps(payload)


def test_output_writer_atomically_replaces_existing_file(tmp_path):
    output_path = tmp_path / "session_output.json"
    output_path.write_text('{"old":true}', encoding="utf-8")
    payload = {"metadata": {"unicode": "你好", "nested": {"value": 1}}, "reward": 1.0}

    write_relax_output(output_path, payload)

    assert json.loads(output_path.read_text(encoding="utf-8")) == payload
    assert not (tmp_path / ".session_output.json.tmp").exists()


def test_output_writer_does_not_replace_target_with_invalid_json(tmp_path):
    output_path = tmp_path / "session_output.json"
    output_path.write_text('{"old":true}', encoding="utf-8")

    with pytest.raises(TypeError):
        write_relax_output(output_path, {"metadata": {"bad": object()}, "reward": None})

    assert json.loads(output_path.read_text(encoding="utf-8")) == {"old": True}
    assert not (tmp_path / ".session_output.json.tmp").exists()
