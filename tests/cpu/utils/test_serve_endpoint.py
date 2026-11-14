# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Driver and workers must address their own isolated Serve endpoint."""

from types import SimpleNamespace

import pytest

from relax.utils.http_utils import get_serve_http_port
from relax.utils.utils import get_serve_url, post_process_env


@pytest.mark.parametrize("configured", [None, "18200"])
def test_serve_endpoint_is_consistent_in_driver_and_worker_configuration(monkeypatch, configured):
    monkeypatch.setenv("RELAX_SERVE_HTTP_PORT", "18100")
    monkeypatch.setattr("relax.utils.utils._resolve_to_ip", lambda _: "127.0.0.1")
    monkeypatch.setattr(
        "relax.utils.utils.ray.nodes",
        lambda: [{"Alive": True, "Resources": {"node:__internal_head__": 1}, "NodeManagerAddress": "127.0.0.2"}],
    )
    args = SimpleNamespace(fully_async=False, rollout_batch_size=1, n_samples_per_prompt=1, partial_rollout=False)
    env = {"env_vars": {"RELAX_SERVE_HTTP_PORT": configured} if configured else {}}
    result = post_process_env(args, env)
    expected = configured or "18100"
    assert result["env_vars"]["RELAX_SERVE_HTTP_PORT"] == expected
    assert get_serve_url("rollout") == f"http://127.0.0.2:{expected}/rollout"
    assert get_serve_http_port() == int(expected)


@pytest.mark.parametrize("value", [0, -1, 65536, True, 8.5, "invalid", "80.0", "８０"])
def test_invalid_serve_port_fails_before_starting_workers(value):
    with pytest.raises(ValueError, match="RELAX_SERVE_HTTP_PORT"):
        get_serve_http_port(value)


def test_default_serve_port(monkeypatch):
    monkeypatch.delenv("RELAX_SERVE_HTTP_PORT", raising=False)
    assert get_serve_http_port() == 8000
