# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from __future__ import annotations

import copy
import json
import logging
import os
from typing import Any


logger = logging.getLogger("uvicorn.error")
_SENSITIVE_KEYS = {
    "api_key",
    "api_key_prefix",
    "authorization",
    "password",
    "secret",
    "session_id",
    "access_token",
    "refresh_token",
}


def log_verbose_payload(event: str, payload: Any, **fields: Any) -> None:
    if os.environ.get("NEMO_GYM_VERBOSE", "0") != "1":
        return
    field_text = " ".join(f"{key}={value}" for key, value in fields.items())
    safe_payload = _redact_payload(payload)
    logger.info(
        "nemo_gym_verbose event=%s %s payload=%s",
        event,
        field_text,
        json.dumps(safe_payload, ensure_ascii=False, separators=(",", ":"), default=repr),
    )


def _redact_payload(payload: Any) -> Any:
    if isinstance(payload, dict):
        return {
            key: "***REDACTED***" if _is_sensitive_key(key) else _redact_payload(value)
            for key, value in payload.items()
        }
    if isinstance(payload, list):
        return [_redact_payload(value) for value in payload]
    return copy.deepcopy(payload)


def _is_sensitive_key(key: Any) -> bool:
    normalized = str(key).lower().replace("-", "_")
    return normalized in _SENSITIVE_KEYS or normalized.endswith(("_api_key", "_password", "_secret"))
