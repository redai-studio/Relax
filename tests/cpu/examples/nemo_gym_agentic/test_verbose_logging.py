# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from relax_nemo_gym_example.service.verbose_logging import _redact_payload


def test_verbose_logging_redacts_credentials_without_removing_request_content():
    payload = {
        "model_endpoint": {
            "api_key": "secret-key",
            "headers": {"Authorization": "Bearer secret", "x-request-id": "request-one"},
        },
        "max_tokens": 4096,
        "messages": [{"role": "user", "content": "fix the bug"}],
    }

    assert _redact_payload(payload) == {
        "model_endpoint": {
            "api_key": "***REDACTED***",
            "headers": {"Authorization": "***REDACTED***", "x-request-id": "request-one"},
        },
        "max_tokens": 4096,
        "messages": [{"role": "user", "content": "fix the bug"}],
    }
