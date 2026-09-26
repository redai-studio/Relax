# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Versioned wire types for the Relax to NeMo Gym gateway contract."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any
from urllib.parse import urlparse


PROTOCOL_VERSION = "relax-nemo-gym/v1"
_HEADER_NAME_PATTERN = re.compile(r"^[!#$%&'*+\-.^_`|~0-9A-Za-z]+$")
_FORBIDDEN_UPSTREAM_HEADERS = frozenset(
    {
        "connection",
        "content-length",
        "host",
        "proxy-authenticate",
        "proxy-authorization",
        "transfer-encoding",
    }
)


class ProtocolValidationError(ValueError):
    """Raised when a gateway payload violates the negotiated protocol."""


class InterruptPolicy(str, Enum):
    PROTECTED = "protected"
    INTERRUPTIBLE = "interruptible"
    RESUMABLE = "resumable"


class TrialStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    TRUNCATED = "truncated"
    ABORTED = "aborted"
    FAILED = "failed"

    @property
    def is_terminal(self) -> bool:
        return self in {
            TrialStatus.COMPLETED,
            TrialStatus.TRUNCATED,
            TrialStatus.ABORTED,
            TrialStatus.FAILED,
        }


RewardValue = float | dict[str, Any] | None


def stable_request_id(session_id: str, attempt: int = 1, *, invocation_id: str | None = None) -> str:
    """Derive a stable invocation-scoped id without exposing bearer values."""

    normalized = _non_empty_string(session_id, field_name="session_id")
    if isinstance(attempt, bool) or not isinstance(attempt, int) or attempt < 1:
        raise ProtocolValidationError("attempt must be an integer greater than or equal to 1")
    if invocation_id is None:
        digest_input = f"{normalized}\0{attempt}"
    else:
        normalized_invocation = _non_empty_string(invocation_id, field_name="invocation_id")
        digest_input = f"{normalized_invocation}\0{normalized}\0{attempt}"
    digest = hashlib.sha256(digest_input.encode("utf-8")).hexdigest()
    return f"relax-{digest}"


@dataclass(frozen=True)
class ModelEndpoint:
    base_url: str
    api_key: str = field(repr=False)
    model: str
    api_key_header: str = "Authorization"
    api_key_prefix: str = field(default="Bearer ", repr=False)
    headers: dict[str, str] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        _http_url(self.base_url, field_name="model_endpoint.base_url")
        _non_empty_string(self.api_key, field_name="model_endpoint.api_key")
        _non_empty_string(self.model, field_name="model_endpoint.model")
        _header_name(self.api_key_header, field_name="model_endpoint.api_key_header")
        _header_value(self.api_key_prefix, field_name="model_endpoint.api_key_prefix")
        _upstream_headers(self.headers, api_key_header=self.api_key_header)

    def to_payload(self) -> dict[str, str]:
        payload: dict[str, Any] = {
            "base_url": self.base_url.rstrip("/"),
            "api_key": self.api_key,
            "model": self.model,
        }
        if self.api_key_header != "Authorization":
            payload["api_key_header"] = self.api_key_header
        if self.api_key_prefix != "Bearer ":
            payload["api_key_prefix"] = self.api_key_prefix
        if self.headers:
            payload["headers"] = copy.deepcopy(self.headers)
        return payload

    @classmethod
    def from_payload(cls, payload: Any) -> "ModelEndpoint":
        value = _mapping(payload, field_name="model_endpoint")
        return cls(
            base_url=value.get("base_url"),
            api_key=value.get("api_key"),
            model=value.get("model"),
            api_key_header=value.get("api_key_header", "Authorization"),
            api_key_prefix=value.get("api_key_prefix", "Bearer "),
            headers=copy.deepcopy(value.get("headers", {})),
        )


@dataclass(frozen=True)
class TrialRequest:
    request_id: str
    session_id: str = field(repr=False)
    group_id: str
    rollout_mode: str
    environment: str
    config: str
    task: dict[str, Any]
    model_endpoint: ModelEndpoint
    generation: dict[str, Any] = field(default_factory=dict)
    interrupt_policy: InterruptPolicy = InterruptPolicy.PROTECTED
    attempt: int = 1
    deadline_s: float = 1800.0
    lease_s: float = 60.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _non_empty_string(self.request_id, field_name="request_id")
        _non_empty_string(self.session_id, field_name="session.session_id")
        _non_empty_string(self.group_id, field_name="session.group_id")
        _non_empty_string(self.rollout_mode, field_name="session.rollout_mode")
        _non_empty_string(self.environment, field_name="environment.name")
        _non_empty_string(self.config, field_name="environment.config")
        if not isinstance(self.task, dict):
            raise ProtocolValidationError("environment.task must be a JSON object")
        if not isinstance(self.generation, dict):
            raise ProtocolValidationError("generation must be a JSON object")
        if not isinstance(self.metadata, dict):
            raise ProtocolValidationError("metadata must be a JSON object")
        _ensure_json_value(self.task, field_name="environment.task")
        _ensure_json_value(self.generation, field_name="generation")
        _ensure_json_value(self.metadata, field_name="metadata")
        if isinstance(self.attempt, bool) or not isinstance(self.attempt, int) or self.attempt < 1:
            raise ProtocolValidationError("session.attempt must be an integer greater than or equal to 1")
        _positive_finite(self.deadline_s, field_name="deadline_s")
        _positive_finite(self.lease_s, field_name="lease_s")

    def to_payload(self) -> dict[str, Any]:
        return {
            "protocol_version": PROTOCOL_VERSION,
            "request_id": self.request_id,
            "session": {
                "session_id": self.session_id,
                "group_id": self.group_id,
                "rollout_mode": self.rollout_mode,
                "attempt": self.attempt,
            },
            "environment": {
                "name": self.environment,
                "config": self.config,
                "task": copy.deepcopy(self.task),
            },
            "model_endpoint": self.model_endpoint.to_payload(),
            "generation": copy.deepcopy(self.generation),
            "interrupt_policy": self.interrupt_policy.value,
            "deadline_s": self.deadline_s,
            "lease_s": self.lease_s,
            "metadata": copy.deepcopy(self.metadata),
        }

    @classmethod
    def from_payload(cls, payload: Any) -> "TrialRequest":
        value = _mapping(payload, field_name="request")
        if value.get("protocol_version") != PROTOCOL_VERSION:
            raise ProtocolValidationError(
                f"Unsupported protocol_version: {value.get('protocol_version')!r}; expected {PROTOCOL_VERSION!r}"
            )
        session = _mapping(value.get("session"), field_name="session")
        environment = _mapping(value.get("environment"), field_name="environment")
        try:
            interrupt_policy = InterruptPolicy(value.get("interrupt_policy"))
        except (TypeError, ValueError) as exc:
            raise ProtocolValidationError(f"Unknown interrupt_policy: {value.get('interrupt_policy')!r}") from exc
        return cls(
            request_id=value.get("request_id"),
            session_id=session.get("session_id"),
            group_id=session.get("group_id"),
            rollout_mode=session.get("rollout_mode"),
            environment=environment.get("name"),
            config=environment.get("config"),
            task=copy.deepcopy(environment.get("task")),
            model_endpoint=ModelEndpoint.from_payload(value.get("model_endpoint")),
            generation=copy.deepcopy(value.get("generation", {})),
            interrupt_policy=interrupt_policy,
            attempt=session.get("attempt", 1),
            deadline_s=value.get("deadline_s", 1800.0),
            lease_s=value.get("lease_s", 60.0),
            metadata=copy.deepcopy(value.get("metadata", {})),
        )


@dataclass(frozen=True)
class TrialResult:
    request_id: str
    status: TrialStatus
    reward: RewardValue = None
    metrics: dict[str, Any] = field(default_factory=dict)
    artifact_ref: str | None = None
    error: dict[str, Any] | None = None

    @classmethod
    def from_payload(cls, payload: Any, *, expected_request_id: str) -> "TrialResult":
        if not isinstance(payload, dict):
            raise ProtocolValidationError("Gateway response must be a JSON object")
        if payload.get("protocol_version") != PROTOCOL_VERSION:
            raise ProtocolValidationError(
                f"Unsupported protocol_version: {payload.get('protocol_version')!r}; expected {PROTOCOL_VERSION!r}"
            )
        request_id = _non_empty_string(payload.get("request_id"), field_name="request_id")
        if request_id != expected_request_id:
            raise ProtocolValidationError("Gateway response request_id does not match the submitted request")
        try:
            status = TrialStatus(payload.get("status"))
        except (TypeError, ValueError) as exc:
            raise ProtocolValidationError(f"Unknown trial status: {payload.get('status')!r}") from exc
        reward = _reward_value(payload.get("reward"))
        metrics = payload.get("metrics", {})
        if not isinstance(metrics, dict):
            raise ProtocolValidationError("metrics must be a JSON object")
        _ensure_json_value(metrics, field_name="metrics")
        artifact_ref = payload.get("artifact_ref")
        if artifact_ref is not None and (not isinstance(artifact_ref, str) or not artifact_ref):
            raise ProtocolValidationError("artifact_ref must be null or a non-empty string")
        error = payload.get("error")
        if error is not None and not isinstance(error, dict):
            raise ProtocolValidationError("error must be null or a JSON object")
        if error is not None:
            _ensure_json_value(error, field_name="error")
        return cls(
            request_id=request_id,
            status=status,
            reward=copy.deepcopy(reward),
            metrics=copy.deepcopy(metrics),
            artifact_ref=artifact_ref,
            error=copy.deepcopy(error),
        )

    @property
    def error_code(self) -> str | None:
        if not isinstance(self.error, dict):
            return None
        code = self.error.get("code")
        return code if isinstance(code, str) and code else None


def _non_empty_string(value: Any, *, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ProtocolValidationError(f"{field_name} must be a non-empty string")
    return value


def _mapping(value: Any, *, field_name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ProtocolValidationError(f"{field_name} must be a JSON object")
    return value


def _http_url(value: Any, *, field_name: str) -> str:
    normalized = _non_empty_string(value, field_name=field_name)
    parsed = urlparse(normalized)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ProtocolValidationError(f"{field_name} must be an absolute http(s) URL")
    return normalized


def _header_name(value: Any, *, field_name: str) -> str:
    normalized = _non_empty_string(value, field_name=field_name)
    if not _HEADER_NAME_PATTERN.fullmatch(normalized):
        raise ProtocolValidationError(f"{field_name} must be a valid HTTP header name")
    if normalized.lower() in _FORBIDDEN_UPSTREAM_HEADERS:
        raise ProtocolValidationError(f"{field_name} is not allowed")
    return normalized


def _header_value(value: Any, *, field_name: str) -> str:
    if not isinstance(value, str) or "\r" in value or "\n" in value:
        raise ProtocolValidationError(f"{field_name} must be a single-line string")
    return value


def _upstream_headers(value: Any, *, api_key_header: str) -> dict[str, str]:
    if not isinstance(value, dict):
        raise ProtocolValidationError("model_endpoint.headers must be a JSON object")
    normalized_api_key_header = api_key_header.lower()
    for name, header_value in value.items():
        normalized_name = _header_name(name, field_name="model_endpoint.headers key").lower()
        if normalized_name == normalized_api_key_header:
            raise ProtocolValidationError("model_endpoint.headers must not override api_key_header")
        _header_value(header_value, field_name=f"model_endpoint.headers[{name!r}]")
    return value


def _positive_finite(value: Any, *, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ProtocolValidationError(f"{field_name} must be a positive number")
    normalized = float(value)
    if not math.isfinite(normalized) or normalized <= 0:
        raise ProtocolValidationError(f"{field_name} must be a positive finite number")
    return normalized


def _reward_value(value: Any) -> RewardValue:
    if value is None:
        return value
    if isinstance(value, dict):
        _ensure_json_value(value, field_name="reward")
        return value
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ProtocolValidationError("reward must be null, a finite number, or a JSON object")
    normalized = float(value)
    if not math.isfinite(normalized):
        raise ProtocolValidationError("reward must be finite")
    return normalized


def _ensure_json_value(value: Any, *, field_name: str) -> None:
    try:
        json.dumps(value, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ProtocolValidationError(f"{field_name} must contain only finite JSON values") from exc
