# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from __future__ import annotations

import copy

import pytest
from relax_nemo_gym_example.app.protocol import (
    PROTOCOL_VERSION,
    InterruptPolicy,
    ModelEndpoint,
    ProtocolValidationError,
    TrialRequest,
    TrialResult,
    TrialStatus,
    stable_request_id,
)


def _request(**overrides):
    values = {
        "request_id": "request-1",
        "session_id": "session-secret",
        "group_id": "group-1",
        "rollout_mode": "train",
        "environment": "multi_step",
        "config": "multi_step-v1",
        "task": {"messages": [{"role": "user", "content": "solve"}], "metadata": {}},
        "model_endpoint": ModelEndpoint(
            base_url="http://relax.example/agentic_api/",
            api_key="session-secret",
            model="model",
        ),
    }
    values.update(overrides)
    return TrialRequest(**values)


def test_request_payload_is_versioned_and_does_not_mutate_input():
    task = {
        "messages": [
            {"role": "system", "content": "be concise"},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "inspect"},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA=="}},
                ],
            },
            {
                "role": "assistant",
                "content": None,
                "reasoning_content": "need a tool",
                "tool_calls": [
                    {"id": "call-1", "type": "function", "function": {"name": "search", "arguments": "{}"}}
                ],
            },
            {"role": "tool", "tool_call_id": "call-1", "content": "done"},
        ],
        "metadata": {"nested": {"value": 1}},
    }
    original = copy.deepcopy(task)

    payload = _request(task=task).to_payload()

    assert payload["protocol_version"] == PROTOCOL_VERSION
    assert payload["environment"]["task"] == original
    assert task == original
    payload["environment"]["task"]["metadata"]["nested"]["value"] = 2
    assert task == original
    assert payload["model_endpoint"]["base_url"] == "http://relax.example/agentic_api"


def test_request_repr_redacts_api_key():
    request = _request()

    assert "session-secret" not in repr(request.model_endpoint)
    assert "session-secret" not in repr(request)


def test_stable_request_id_is_deterministic_and_hides_session_id():
    first = stable_request_id("session-secret")

    assert first == stable_request_id("session-secret")
    assert first != stable_request_id("other-session")
    assert first != stable_request_id("session-secret", 2)
    assert "session-secret" not in first


def test_stable_request_id_scopes_same_session_to_managed_invocation():
    first = stable_request_id("session-secret", invocation_id="/tmp/relax-agentic-command-one")

    assert first == stable_request_id("session-secret", invocation_id="/tmp/relax-agentic-command-one")
    assert first != stable_request_id("session-secret", invocation_id="/tmp/relax-agentic-command-two")


def test_request_round_trip_preserves_attempt_scoped_contract():
    request = _request(attempt=2)

    restored = TrialRequest.from_payload(request.to_payload())

    assert restored.to_payload() == request.to_payload()
    assert restored.model_endpoint.api_key == "session-secret"


def test_request_round_trip_preserves_custom_upstream_auth_without_repr_leaks():
    endpoint = ModelEndpoint(
        base_url="https://model.example/v1",
        api_key="custom-secret",
        model="model",
        api_key_header="api-key",
        api_key_prefix="",
        headers={"x-user": "user@example.com", "x-app-id": "app"},
    )
    request = _request(model_endpoint=endpoint)

    restored = TrialRequest.from_payload(request.to_payload())

    assert restored.model_endpoint == endpoint
    assert "custom-secret" not in repr(restored)
    assert "user@example.com" not in repr(restored)


@pytest.mark.parametrize("header", ["Host", "content-length", "bad header"])
def test_request_rejects_unsafe_upstream_headers(header):
    with pytest.raises(ProtocolValidationError):
        ModelEndpoint(
            base_url="https://model.example/v1",
            api_key="secret",
            model="model",
            headers={header: "value"},
        )


def test_request_rejects_non_json_task():
    with pytest.raises(ProtocolValidationError, match="finite JSON"):
        _request(task={"messages": [], "bad": float("nan")})


def test_result_parses_terminal_reward():
    result = TrialResult.from_payload(
        {
            "protocol_version": PROTOCOL_VERSION,
            "request_id": "request-1",
            "status": "completed",
            "reward": {"scalar": 1.0, "components": {"verifier": 1.0}},
            "metrics": {"model_calls": 2},
            "artifact_ref": "artifact://one",
            "error": None,
        },
        expected_request_id="request-1",
    )

    assert result.status is TrialStatus.COMPLETED
    assert result.status.is_terminal
    assert result.reward == {"scalar": 1.0, "components": {"verifier": 1.0}}
    assert result.metrics == {"model_calls": 2}


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("protocol_version", "other/v1", "Unsupported protocol_version"),
        ("request_id", "wrong", "does not match"),
        ("status", "unknown", "Unknown trial status"),
    ],
)
def test_result_rejects_contract_mismatch(field, value, message):
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "request_id": "request-1",
        "status": "running",
    }
    payload[field] = value

    with pytest.raises(ProtocolValidationError, match=message):
        TrialResult.from_payload(payload, expected_request_id="request-1")


def test_interrupt_policy_values_are_wire_stable():
    assert [policy.value for policy in InterruptPolicy] == ["protected", "interruptible", "resumable"]
