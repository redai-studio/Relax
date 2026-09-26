# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Deployment-time configuration for the reference NeMo Gym Gateway."""

from __future__ import annotations

import hashlib
import ipaddress
import json
import math
import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from ..app.protocol import InterruptPolicy


class GatewayConfigError(ValueError):
    """Raised when deployment-time Gateway configuration is invalid."""


@dataclass(frozen=True)
class EnvironmentSpec:
    environment: str
    config: str
    agent_name: str
    agent_url: str
    interrupt_policy: InterruptPolicy
    max_concurrency: int = 1
    queue_capacity: int = 8
    max_deadline_s: float = 3600.0
    abort_url: str | None = None
    force_cleanup_url: str | None = None
    cleanup_probe_url: str | None = None
    readiness_urls: tuple[str, ...] = ()

    @classmethod
    def from_payload(cls, config: str, payload: Any) -> "EnvironmentSpec":
        if not isinstance(payload, dict):
            raise GatewayConfigError(f"Environment config {config!r} must be a JSON object")
        try:
            interrupt_policy = InterruptPolicy(payload.get("interrupt_policy", "protected"))
        except (TypeError, ValueError) as exc:
            raise GatewayConfigError(f"Environment config {config!r} has an invalid interrupt_policy") from exc
        agent_url = _absolute_http_url(payload.get("agent_url"), field_name=f"{config}.agent_url")
        abort_url = payload.get("abort_url")
        if abort_url is not None:
            abort_url = _absolute_http_url(abort_url, field_name=f"{config}.abort_url")
        force_cleanup_url = payload.get("force_cleanup_url")
        if force_cleanup_url is not None:
            force_cleanup_url = _absolute_http_url(
                force_cleanup_url,
                field_name=f"{config}.force_cleanup_url",
            )
        cleanup_probe_url = payload.get("cleanup_probe_url")
        if cleanup_probe_url is not None:
            cleanup_probe_url = _absolute_http_url(
                cleanup_probe_url,
                field_name=f"{config}.cleanup_probe_url",
            )
        readiness_urls = payload.get("readiness_urls", [agent_url])
        if not isinstance(readiness_urls, list) or not readiness_urls:
            raise GatewayConfigError(f"{config}.readiness_urls must be a non-empty JSON array")
        return cls(
            environment=_non_empty_string(payload.get("environment"), field_name=f"{config}.environment"),
            config=_non_empty_string(config, field_name="config"),
            agent_name=_non_empty_string(payload.get("agent_name"), field_name=f"{config}.agent_name"),
            agent_url=agent_url.rstrip("/"),
            interrupt_policy=interrupt_policy,
            max_concurrency=_positive_int(payload.get("max_concurrency", 1), field_name=f"{config}.max_concurrency"),
            queue_capacity=_non_negative_int(
                payload.get("queue_capacity", 8),
                field_name=f"{config}.queue_capacity",
            ),
            max_deadline_s=_positive_number(
                payload.get("max_deadline_s", 3600.0),
                field_name=f"{config}.max_deadline_s",
            ),
            abort_url=abort_url,
            force_cleanup_url=force_cleanup_url,
            cleanup_probe_url=cleanup_probe_url,
            readiness_urls=tuple(
                _absolute_http_url(value, field_name=f"{config}.readiness_urls").rstrip("/")
                for value in readiness_urls
            ),
        )

    def to_fingerprint_payload(self) -> dict[str, Any]:
        return {
            "environment": self.environment,
            "config": self.config,
            "agent_name": self.agent_name,
            "agent_url": self.agent_url,
            "interrupt_policy": self.interrupt_policy.value,
            "max_concurrency": self.max_concurrency,
            "queue_capacity": self.queue_capacity,
            "max_deadline_s": self.max_deadline_s,
            "abort_url": self.abort_url,
            "force_cleanup_url": self.force_cleanup_url,
            "cleanup_probe_url": self.cleanup_probe_url,
            "readiness_urls": list(self.readiness_urls),
        }


@dataclass(frozen=True)
class GatewaySettings:
    environments: dict[tuple[str, str], EnvironmentSpec]
    callback_allowed_hosts: frozenset[str]
    gym_commit: str
    config_fingerprint: str
    callback_allowed_networks: tuple[ipaddress.IPv4Network | ipaddress.IPv6Network, ...] = ()
    lease_scan_interval_s: float = 0.25
    cleanup_grace_s: float = 30.0
    callback_proxy: str | None = field(default=None, repr=False)
    callback_timeout_s: float = 600.0
    artifact_root: Path | None = None

    @classmethod
    def from_env(cls, environ: dict[str, str] | None = None) -> "GatewaySettings":
        values = dict(os.environ) if environ is None else environ
        raw_environments = values.get("NEMO_GYM_GATEWAY_ENVIRONMENTS_JSON")
        if not raw_environments:
            raise GatewayConfigError("NEMO_GYM_GATEWAY_ENVIRONMENTS_JSON must be set")
        try:
            environment_payloads = json.loads(raw_environments)
        except json.JSONDecodeError as exc:
            raise GatewayConfigError("NEMO_GYM_GATEWAY_ENVIRONMENTS_JSON must contain valid JSON") from exc
        if not isinstance(environment_payloads, dict) or not environment_payloads:
            raise GatewayConfigError("NEMO_GYM_GATEWAY_ENVIRONMENTS_JSON must contain a non-empty JSON object")

        environments: dict[tuple[str, str], EnvironmentSpec] = {}
        for config_name, payload in environment_payloads.items():
            spec = EnvironmentSpec.from_payload(config_name, payload)
            key = (spec.environment, spec.config)
            if key in environments:
                raise GatewayConfigError(f"Duplicate environment registration: {key!r}")
            environments[key] = spec

        raw_hosts = values.get("NEMO_GYM_CALLBACK_ALLOWED_HOSTS", "")
        allowed_hosts = frozenset(host.strip().lower() for host in raw_hosts.split(",") if host.strip())
        if "*" in allowed_hosts:
            raise GatewayConfigError("Wildcard callback hosts are not supported")
        raw_networks = values.get("NEMO_GYM_CALLBACK_ALLOWED_NETWORKS", "")
        try:
            allowed_networks = tuple(
                ipaddress.ip_network(value.strip(), strict=True) for value in raw_networks.split(",") if value.strip()
            )
        except ValueError as exc:
            raise GatewayConfigError("NEMO_GYM_CALLBACK_ALLOWED_NETWORKS must contain valid CIDR networks") from exc
        if any(network.prefixlen == 0 for network in allowed_networks):
            raise GatewayConfigError("NEMO_GYM_CALLBACK_ALLOWED_NETWORKS must not contain a default-route network")
        if not allowed_hosts and not allowed_networks:
            raise GatewayConfigError("At least one exact callback hostname or CIDR network must be configured")

        raw_callback_proxy = values.get("NEMO_GYM_CALLBACK_PROXY", "").strip()
        callback_proxy = (
            _absolute_proxy_url(raw_callback_proxy, field_name="NEMO_GYM_CALLBACK_PROXY")
            if raw_callback_proxy
            else None
        )
        raw_artifact_root = values.get("NEMO_GYM_ARTIFACT_ROOT", "").strip()
        artifact_root = (
            _absolute_path(raw_artifact_root, field_name="NEMO_GYM_ARTIFACT_ROOT") if raw_artifact_root else None
        )

        fingerprint_payload = [
            spec.to_fingerprint_payload()
            for spec in sorted(environments.values(), key=lambda item: (item.environment, item.config))
        ]
        fingerprint = hashlib.sha256(
            json.dumps(fingerprint_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        return cls(
            environments=environments,
            callback_allowed_hosts=allowed_hosts,
            gym_commit=values.get("NEMO_GYM_COMMIT", "unknown"),
            config_fingerprint=fingerprint,
            callback_allowed_networks=allowed_networks,
            lease_scan_interval_s=_positive_number(
                values.get("NEMO_GYM_LEASE_SCAN_INTERVAL_S", 0.25),
                field_name="NEMO_GYM_LEASE_SCAN_INTERVAL_S",
            ),
            cleanup_grace_s=_positive_number(
                values.get("NEMO_GYM_CLEANUP_GRACE_S", 30.0),
                field_name="NEMO_GYM_CLEANUP_GRACE_S",
            ),
            callback_proxy=callback_proxy,
            callback_timeout_s=_positive_number(
                values.get("NEMO_GYM_CALLBACK_TIMEOUT_S", 600.0),
                field_name="NEMO_GYM_CALLBACK_TIMEOUT_S",
            ),
            artifact_root=artifact_root,
        )


def validate_callback_url(
    base_url: str,
    allowed_hosts: frozenset[str],
    *,
    allowed_networks: tuple[ipaddress.IPv4Network | ipaddress.IPv6Network, ...] = (),
    require_tls: bool = False,
) -> None:
    parsed = urlparse(base_url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise GatewayConfigError("model_endpoint.base_url must be an absolute http(s) URL")
    if require_tls and parsed.scheme != "https":
        raise GatewayConfigError("model_endpoint.base_url must use https when the callback proxy is enabled")
    if parsed.username is not None or parsed.password is not None:
        raise GatewayConfigError("model_endpoint.base_url must not contain user info")
    if parsed.fragment:
        raise GatewayConfigError("model_endpoint.base_url must not contain a fragment")
    callback_host = parsed.hostname.lower()
    host_allowed = callback_host in allowed_hosts
    try:
        callback_ip = ipaddress.ip_address(callback_host)
    except ValueError:
        callback_ip = None
    network_allowed = callback_ip is not None and any(callback_ip in network for network in allowed_networks)
    if not host_allowed and not network_allowed:
        raise GatewayConfigError("model_endpoint.base_url host is not in the callback allowlist")


def validate_nemo_gym_graph(
    global_config: Mapping,
    *,
    gateway_name: str,
    settings: GatewaySettings,
) -> None:
    if global_config.get("observability_enabled") is not True:
        raise GatewayConfigError("observability_enabled=true is required for request-scoped rollout paths")
    if not global_config.get("ray_head_node_address"):
        raise GatewayConfigError("The Gym graph must have an explicit private ray_head_node_address")
    if global_config.get("skip_venv_if_present") is not True:
        raise GatewayConfigError("skip_venv_if_present=true is required for the image's prebuilt server environments")

    for spec in settings.environments.values():
        agent = global_config.get(spec.agent_name)
        if not isinstance(agent, Mapping):
            raise GatewayConfigError(f"Registered Gym agent {spec.agent_name!r} is missing")
        agent_types = agent.get("responses_api_agents")
        if not isinstance(agent_types, Mapping) or len(agent_types) != 1:
            raise GatewayConfigError(f"Registered Gym agent {spec.agent_name!r} has an invalid config")
        agent_config = next(iter(agent_types.values()))
        if not isinstance(agent_config, Mapping):
            raise GatewayConfigError(f"Registered Gym agent {spec.agent_name!r} has an invalid inner config")
        model_ref = agent_config.get("model_server")
        if not isinstance(model_ref, Mapping) or model_ref.get("name") != gateway_name:
            raise GatewayConfigError(f"Registered Gym agent {spec.agent_name!r} does not reference the Gateway model")

        parsed_agent_url = urlparse(spec.agent_url)
        graph_host = agent_config.get("host")
        host_matches = graph_host == parsed_agent_url.hostname or graph_host in {"0.0.0.0", "::"}
        if not host_matches or agent_config.get("port") != parsed_agent_url.port:
            raise GatewayConfigError(f"Registered Gym agent {spec.agent_name!r} URL does not match the Gym graph")


def _non_empty_string(value: Any, *, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise GatewayConfigError(f"{field_name} must be a non-empty string")
    return value


def _absolute_http_url(value: Any, *, field_name: str) -> str:
    normalized = _non_empty_string(value, field_name=field_name)
    parsed = urlparse(normalized)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise GatewayConfigError(f"{field_name} must be an absolute http(s) URL")
    return normalized


def _absolute_proxy_url(value: Any, *, field_name: str) -> str:
    normalized = _absolute_http_url(value, field_name=field_name)
    parsed = urlparse(normalized)
    if parsed.username is not None or parsed.password is not None:
        raise GatewayConfigError(f"{field_name} must not contain user info")
    if parsed.path not in {"", "/"} or parsed.query or parsed.fragment:
        raise GatewayConfigError(f"{field_name} must not contain a path, query, or fragment")
    return normalized


def _absolute_path(value: Any, *, field_name: str) -> Path:
    normalized = _non_empty_string(value, field_name=field_name)
    path = Path(normalized)
    if not path.is_absolute():
        raise GatewayConfigError(f"{field_name} must be an absolute path")
    return path


def _positive_int(value: Any, *, field_name: str) -> int:
    if isinstance(value, bool):
        raise GatewayConfigError(f"{field_name} must be a positive integer")
    try:
        normalized = int(value)
    except (TypeError, ValueError) as exc:
        raise GatewayConfigError(f"{field_name} must be a positive integer") from exc
    if normalized < 1:
        raise GatewayConfigError(f"{field_name} must be a positive integer")
    return normalized


def _non_negative_int(value: Any, *, field_name: str) -> int:
    if isinstance(value, bool):
        raise GatewayConfigError(f"{field_name} must be a non-negative integer")
    try:
        normalized = int(value)
    except (TypeError, ValueError) as exc:
        raise GatewayConfigError(f"{field_name} must be a non-negative integer") from exc
    if normalized < 0:
        raise GatewayConfigError(f"{field_name} must be a non-negative integer")
    return normalized


def _positive_number(value: Any, *, field_name: str) -> float:
    if isinstance(value, bool):
        raise GatewayConfigError(f"{field_name} must be a positive number")
    try:
        normalized = float(value)
    except (TypeError, ValueError) as exc:
        raise GatewayConfigError(f"{field_name} must be a positive number") from exc
    if not math.isfinite(normalized) or normalized <= 0:
        raise GatewayConfigError(f"{field_name} must be a positive finite number")
    return normalized
