# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from __future__ import annotations

import json
from pathlib import Path

import pytest
from relax_nemo_gym_example.service.config import (
    GatewayConfigError,
    GatewaySettings,
    validate_callback_url,
    validate_nemo_gym_graph,
)


def _environ():
    return {
        "NEMO_GYM_GATEWAY_ENVIRONMENTS_JSON": json.dumps(
            {
                "multi-step-v1": {
                    "environment": "multi_step",
                    "agent_name": "example_multi_step_simple_agent",
                    "agent_url": "http://127.0.0.1:10001",
                    "interrupt_policy": "protected",
                    "max_concurrency": 2,
                    "queue_capacity": 4,
                }
            }
        ),
        "NEMO_GYM_CALLBACK_ALLOWED_HOSTS": "relax-head,relax-worker.example",
        "NEMO_GYM_COMMIT": "abc123",
    }


def test_settings_register_only_named_server_side_configs():
    settings = GatewaySettings.from_env(_environ())

    spec = settings.environments[("multi_step", "multi-step-v1")]
    assert spec.agent_url == "http://127.0.0.1:10001"
    assert spec.agent_name == "example_multi_step_simple_agent"
    assert spec.max_concurrency == 2
    assert settings.gym_commit == "abc123"


def test_settings_load_explicit_callback_proxy_and_timeout():
    environ = _environ()
    environ["NEMO_GYM_CALLBACK_PROXY"] = "http://proxy.example:3128"
    environ["NEMO_GYM_CALLBACK_TIMEOUT_S"] = "45"
    environ["NEMO_GYM_ARTIFACT_ROOT"] = "/shared/r2e-artifacts"

    settings = GatewaySettings.from_env(environ)

    assert settings.callback_proxy == "http://proxy.example:3128"
    assert settings.callback_timeout_s == 45.0
    assert settings.artifact_root == Path("/shared/r2e-artifacts")
    assert "proxy.example" not in repr(settings)


@pytest.mark.parametrize(
    "proxy",
    [
        "http://user:secret@proxy.example:3128",
        "http://proxy.example:3128/path",
        "http://proxy.example:3128?target=model",
    ],
)
def test_settings_reject_unsafe_callback_proxy_urls(proxy):
    environ = _environ()
    environ["NEMO_GYM_CALLBACK_PROXY"] = proxy

    with pytest.raises(GatewayConfigError, match="NEMO_GYM_CALLBACK_PROXY"):
        GatewaySettings.from_env(environ)


def test_settings_reject_wildcard_callback_hosts():
    environ = _environ()
    environ["NEMO_GYM_CALLBACK_ALLOWED_HOSTS"] = "*"

    with pytest.raises(GatewayConfigError, match="Wildcard"):
        GatewaySettings.from_env(environ)


def test_callback_url_accepts_ip_in_allowlisted_network():
    environ = _environ()
    environ["NEMO_GYM_CALLBACK_ALLOWED_HOSTS"] = ""
    environ["NEMO_GYM_CALLBACK_ALLOWED_NETWORKS"] = "192.0.2.0/24,198.51.100.0/24"
    settings = GatewaySettings.from_env(environ)

    validate_callback_url(
        "http://192.0.2.148:8000/agentic_api",
        settings.callback_allowed_hosts,
        allowed_networks=settings.callback_allowed_networks,
    )
    with pytest.raises(GatewayConfigError, match="allowlist"):
        validate_callback_url(
            "http://203.0.113.1:8000/agentic_api",
            settings.callback_allowed_hosts,
            allowed_networks=settings.callback_allowed_networks,
        )


def test_callback_url_keeps_exact_ip_host_compatibility():
    validate_callback_url("http://192.0.2.148:8000/agentic_api", frozenset({"192.0.2.148"}))


def test_callback_url_accepts_ipv6_in_allowlisted_network():
    environ = _environ()
    environ["NEMO_GYM_CALLBACK_ALLOWED_HOSTS"] = ""
    environ["NEMO_GYM_CALLBACK_ALLOWED_NETWORKS"] = "fd00::/8"
    settings = GatewaySettings.from_env(environ)

    validate_callback_url(
        "http://[fd12::34]:8000/agentic_api",
        settings.callback_allowed_hosts,
        allowed_networks=settings.callback_allowed_networks,
    )


def test_settings_reject_invalid_callback_network():
    environ = _environ()
    environ["NEMO_GYM_CALLBACK_ALLOWED_NETWORKS"] = "192.0.2.3/24"

    with pytest.raises(GatewayConfigError, match="valid CIDR"):
        GatewaySettings.from_env(environ)


@pytest.mark.parametrize("network", ["0.0.0.0/0", "::/0"])
def test_settings_reject_default_route_callback_network(network):
    environ = _environ()
    environ["NEMO_GYM_CALLBACK_ALLOWED_NETWORKS"] = network

    with pytest.raises(GatewayConfigError, match="default-route"):
        GatewaySettings.from_env(environ)


def test_callback_url_requires_exact_allowlisted_host():
    allowed = frozenset({"relax-head"})

    validate_callback_url("http://relax-head:8000/agentic_api", allowed)
    with pytest.raises(GatewayConfigError, match="allowlist"):
        validate_callback_url("http://metadata.internal/latest", allowed)


def test_callback_url_requires_tls_when_proxy_is_enabled():
    allowed = frozenset({"model.example"})

    validate_callback_url("https://model.example/v1", allowed, require_tls=True)
    with pytest.raises(GatewayConfigError, match="must use https"):
        validate_callback_url("http://model.example/v1", allowed, require_tls=True)


def test_gym_graph_validation_requires_prefix_and_request_scoped_gateway_model():
    settings = GatewaySettings.from_env(_environ())
    graph = {
        "observability_enabled": True,
        "ray_head_node_address": "gym-ray:6379",
        "skip_venv_if_present": True,
        "example_multi_step_simple_agent": {
            "responses_api_agents": {
                "simple_agent": {
                    "host": "127.0.0.1",
                    "port": 10001,
                    "model_server": {"type": "responses_api_models", "name": "policy_model"},
                }
            }
        },
    }

    validate_nemo_gym_graph(graph, gateway_name="policy_model", settings=settings)

    graph["observability_enabled"] = False
    with pytest.raises(GatewayConfigError, match="observability_enabled"):
        validate_nemo_gym_graph(graph, gateway_name="policy_model", settings=settings)


def test_gym_graph_validation_requires_prebuilt_server_environments():
    settings = GatewaySettings.from_env(_environ())
    graph = {
        "observability_enabled": True,
        "ray_head_node_address": "gym-ray:6379",
        "skip_venv_if_present": False,
    }

    with pytest.raises(GatewayConfigError, match="skip_venv_if_present"):
        validate_nemo_gym_graph(graph, gateway_name="policy_model", settings=settings)


def test_gym_graph_validation_allows_wildcard_bind_host():
    settings = GatewaySettings.from_env(_environ())
    graph = {
        "observability_enabled": True,
        "ray_head_node_address": "gym-ray:6379",
        "skip_venv_if_present": True,
        "example_multi_step_simple_agent": {
            "responses_api_agents": {
                "simple_agent": {
                    "host": "0.0.0.0",
                    "port": 10001,
                    "model_server": {"type": "responses_api_models", "name": "policy_model"},
                }
            }
        },
    }

    validate_nemo_gym_graph(graph, gateway_name="policy_model", settings=settings)
