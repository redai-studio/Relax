# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Bounded, strict JSON validation for weight-sync configuration."""

import json
from typing import Any


MAX_MANIFEST_BYTES = 16 * 1024 * 1024


class WeightSyncError(ValueError):
    """Invalid, unavailable, or inconsistent weight content."""


def integer(value: Any, minimum: int = 0, maximum: int = 2**63 - 1) -> int:
    if type(value) is not int or not minimum <= value <= maximum:
        raise WeightSyncError(f"Expected integer in [{minimum}, {maximum}]")
    return value


def _unique_pairs(pairs: list[tuple[str, Any]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise WeightSyncError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def parse_json(data: bytes, limit: int = MAX_MANIFEST_BYTES) -> Any:
    if len(data) > limit:
        raise WeightSyncError("JSON exceeds size limit")
    try:
        return json.loads(data, object_pairs_hook=_unique_pairs, parse_constant=_reject_constant)
    except (UnicodeError, json.JSONDecodeError, RecursionError) as error:
        raise WeightSyncError("Invalid JSON") from error


def _reject_constant(value: str) -> None:
    raise WeightSyncError(f"Non-finite JSON number: {value}")
