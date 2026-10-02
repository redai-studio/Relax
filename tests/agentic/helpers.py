# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Shared fixture access for the agentic multi-protocol canonical tests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

from relax.agentic.session.service import (
    _normalized_anthropic_request,
    _normalized_chat_request,
    _normalized_responses_request,
)


FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures"
# The three fields every protocol must agree on before a request may enter a
# Session; they are also the only inputs of the Session state hash.
CANONICAL_FIELDS: tuple[str, ...] = ("messages", "tools", "chat_template_kwargs")
PROTOCOLS: tuple[str, ...] = ("chat_completions", "responses", "anthropic_messages")
PROTOCOL_NORMALIZERS: dict[str, Callable[[dict[str, Any]], dict[str, Any]]] = {
    "chat_completions": _normalized_chat_request,
    "responses": _normalized_responses_request,
    "anthropic_messages": _normalized_anthropic_request,
}


def _read(name: str) -> Any:
    return json.loads((FIXTURE_DIR / name).read_text(encoding="utf-8"))


def protocol_document(protocol: str) -> dict[str, Any]:
    document = _read(f"{protocol}.json")
    if document["protocol"] != protocol:
        raise AssertionError(f"fixture {protocol}.json declares protocol {document['protocol']!r}")
    return document


def protocol_payloads(protocol: str) -> dict[str, dict[str, Any]]:
    """Return one protocol's raw request goldens keyed by ``case_id``."""

    return {case["case_id"]: case["payload"] for case in protocol_document(protocol)["cases"]}


def canonical_expectations() -> dict[str, dict[str, Any]]:
    """Return the pinned canonical goldens keyed by ``case_id``."""

    return _read("canonical.json")


def canonical_case_ids() -> list[str]:
    return list(canonical_expectations())


def normalize(protocol: str, payload: dict[str, Any]) -> dict[str, Any]:
    return PROTOCOL_NORMALIZERS[protocol](payload)


def canonical_of(result: dict[str, Any]) -> dict[str, Any]:
    """Project a normalizer result onto the cross-protocol canonical fields."""

    assert set(CANONICAL_FIELDS) <= set(result), f"normalizer result lost canonical fields: {sorted(result)}"
    return {field: result[field] for field in CANONICAL_FIELDS}
