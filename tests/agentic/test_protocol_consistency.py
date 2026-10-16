# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Golden tests for Chat Completions / Responses / Anthropic Messages
consistency."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import pytest

from relax.agentic.session.service import (
    _normalized_anthropic_request,
    _normalized_chat_request,
    _normalized_responses_request,
)
from relax.agentic.session.state import check_messages, normalize_template_kwargs, normalize_tools


FIXTURES = Path(__file__).resolve().parent / "fixtures"
CANONICAL_FIELDS = ("messages", "tools", "chat_template_kwargs")

PROTOCOL_NORMALIZERS: dict[str, Callable[[dict[str, Any]], dict[str, Any]]] = {
    "chat_completions": _normalized_chat_request,
    "responses": _normalized_responses_request,
    "anthropic_messages": _normalized_anthropic_request,
}

GOLDEN_CASES = ("text_tools", "text_list", "system_text_list", "image_url", "image_base64")


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _canonical_fields(normalized: dict[str, Any]) -> dict[str, Any]:
    return {field: normalized[field] for field in CANONICAL_FIELDS}


@pytest.mark.parametrize("case_name", GOLDEN_CASES)
@pytest.mark.parametrize("protocol", sorted(PROTOCOL_NORMALIZERS))
def test_protocol_golden_fixture_matches_canonical(case_name: str, protocol: str) -> None:
    request = _load_json(FIXTURES / protocol / f"{case_name}.request.json")
    expected = _load_json(FIXTURES / f"{case_name}.canonical.json")
    normalized = PROTOCOL_NORMALIZERS[protocol](request)
    assert _canonical_fields(normalized) == expected


def test_chat_string_image_url_matches_canonical_image_fixture() -> None:
    request = _load_json(FIXTURES / "chat_completions" / "image_url_string.request.json")
    expected = _load_json(FIXTURES / "image_url.canonical.json")
    assert _canonical_fields(_normalized_chat_request(request)) == expected


def test_legal_images_unify_to_canonical_image_url_shape() -> None:
    image_url = "https://example.com/cat.png"
    data_url = (
        "data:image/png;base64,"
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
    )
    chat = _normalized_chat_request(
        {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "see"},
                        {"type": "image_url", "image_url": image_url},
                    ],
                }
            ]
        }
    )
    responses = _normalized_responses_request(
        {
            "input": [
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": "see"},
                        {"type": "input_image", "image_url": image_url},
                    ],
                }
            ]
        }
    )
    anthropic = _normalized_anthropic_request(
        {
            "max_tokens": 16,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "see"},
                        {"type": "image", "source": {"type": "url", "url": image_url}},
                    ],
                }
            ],
        }
    )
    expected_part = {"type": "image_url", "image_url": {"url": image_url}}
    assert chat["messages"][0]["content"][1] == expected_part
    assert responses["messages"][0]["content"][1] == expected_part
    assert anthropic["messages"][0]["content"][1] == expected_part

    anthropic_b64 = _normalized_anthropic_request(
        {
            "max_tokens": 16,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "see"},
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": data_url.split(",", 1)[1],
                            },
                        },
                    ],
                }
            ],
        }
    )
    assert anthropic_b64["messages"][0]["content"][1] == {
        "type": "image_url",
        "image_url": {"url": data_url},
    }


def test_normalize_tools_drops_non_function_fields_and_stable_shape() -> None:
    tools = normalize_tools(
        [
            {
                "type": "function",
                "function": {
                    "name": "search",
                    "description": "Search",
                    "parameters": {"type": "object"},
                    "strict": True,
                },
            },
            {"type": "unsupported", "function": {"name": "nope"}},
        ]
    )
    assert tools == [
        {
            "type": "function",
            "function": {
                "name": "search",
                "parameters": {"type": "object"},
                "description": "Search",
            },
        }
    ]


def test_normalize_template_kwargs_sorts_nested_keys() -> None:
    normalized = normalize_template_kwargs({"z": 1, "nested": {"b": 2, "a": 1}, "enable_thinking": True})
    assert list(normalized) == ["enable_thinking", "nested", "z"]
    assert list(normalized["nested"]) == ["a", "b"]
    assert check_messages([{"role": "user", "content": "hi"}]) == [{"role": "user", "content": "hi"}]
