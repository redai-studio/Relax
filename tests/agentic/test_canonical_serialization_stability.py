# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Stability tests for the canonical serialization behind the Session state
hash.

The state hash is the only identity a Session node has, so it must be
insensitive to key order and JSON number representation while staying sensitive
to actual content, and it must refuse anything that cannot round-trip JSON.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from relax.agentic.session.state import (
    _canonical_tool_arguments,
    _messages_tools_template_state_hash,
    check_messages,
    normalize_template_kwargs,
    normalize_tools,
)
from tests.agentic.helpers import canonical_expectations


_MESSAGES = [
    {"role": "system", "content": "S"},
    {"role": "user", "content": "hi"},
    {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {"id": "call-1", "type": "function", "function": {"name": "run", "arguments": '{"a":"b","c":"d"}'}}
        ],
    },
]
_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
            "description": "Return the weather.",
        },
    }
]
_TEMPLATE_KWARGS = {"enable_thinking": True, "nested": {"z": 1, "a": [2, {"y": 3, "x": 4}]}}


def _reorder(value: Any) -> Any:
    """Return the same JSON value with every object rebuilt in reverse key
    order."""

    if isinstance(value, dict):
        return {key: _reorder(value[key]) for key in reversed(list(value))}
    if isinstance(value, list):
        return [_reorder(item) for item in value]
    return value


def test_serialization_state_hash_ignores_every_key_order() -> None:
    reference = _messages_tools_template_state_hash(_MESSAGES, _TOOLS, _TEMPLATE_KWARGS)

    assert _messages_tools_template_state_hash(_reorder(_MESSAGES), _reorder(_TOOLS), _reorder(_TEMPLATE_KWARGS)) == (
        reference
    )
    assert (
        _messages_tools_template_state_hash(
            json.loads(json.dumps(_MESSAGES)),
            json.loads(json.dumps(_TOOLS)),
            json.loads(json.dumps(_TEMPLATE_KWARGS)),
        )
        == reference
    )


def test_serialization_state_hash_is_a_sha256_hex_digest() -> None:
    digest = _messages_tools_template_state_hash(_MESSAGES, _TOOLS, _TEMPLATE_KWARGS)

    assert len(digest) == 64
    assert all(character in "0123456789abcdef" for character in digest)


@pytest.mark.parametrize(
    ("mutate", "expected_changed_field"),
    [
        pytest.param(
            lambda m, t, k: ([{**m[0], "content": m[0]["content"] + " "}, *m[1:]], t, k),
            "messages",
            id="trailing-space",
        ),
        pytest.param(lambda m, t, k: (m, [], k), "tools", id="tools-dropped"),
        pytest.param(lambda m, t, k: (m, t, {"enable_thinking": False}), "template_kwargs", id="thinking-off"),
    ],
)
def test_serialization_state_hash_is_content_sensitive(mutate: Any, expected_changed_field: str) -> None:
    reference = _messages_tools_template_state_hash(_MESSAGES, _TOOLS, _TEMPLATE_KWARGS)
    mutated = _messages_tools_template_state_hash(*mutate(_MESSAGES, _TOOLS, _TEMPLATE_KWARGS))

    assert mutated != reference, f"{expected_changed_field} change did not move the state hash"


def test_serialization_golden_case_hashes_are_distinct() -> None:
    expectations = canonical_expectations()
    digests = {
        case_id: _messages_tools_template_state_hash(
            golden["messages"],
            golden["tools"],
            golden["chat_template_kwargs"],
        )
        for case_id, golden in expectations.items()
    }

    assert len(set(digests.values())) == len(digests), {
        digest: [case for case, value in digests.items() if value == digest] for digest in set(digests.values())
    }


@pytest.mark.parametrize(
    ("arguments", "expected"),
    [
        pytest.param('{"a":"b","c":"d"}', '{"a":"b","c":"d"}', id="already-sorted"),
        pytest.param('{"c":"d","a":"b"}', '{"a":"b","c":"d"}', id="reversed"),
        pytest.param('{"a": "b", "c": "d"}', '{"a":"b","c":"d"}', id="padded"),
        pytest.param({"c": "d", "a": "b"}, '{"a":"b","c":"d"}', id="native-dict"),
        pytest.param('{"n":{"z":1,"a":[2,{"y":3,"x":4}]}}', '{"n":{"a":[2,{"x":4,"y":3}],"z":1}}', id="nested"),
        pytest.param('{"a":2.0}', '{"a":2}', id="integral-float-matches-int"),
        pytest.param('{"a":2}', '{"a":2}', id="int"),
        pytest.param('{"a":"\\u4e2d"}', '{"a":"中"}', id="non-ascii-kept-literal"),
    ],
)
def test_serialization_tool_arguments_collapse_to_one_json_form(arguments: Any, expected: str) -> None:
    assert _canonical_tool_arguments(arguments, field="messages[0].tool_calls[0].function.arguments") == expected


def test_serialization_equivalent_tool_arguments_share_one_state_hash() -> None:
    def message(arguments: Any) -> list[dict[str, Any]]:
        return check_messages(
            [
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {"id": "call-1", "type": "function", "function": {"name": "run", "arguments": arguments}}
                    ],
                }
            ]
        )

    assert _messages_tools_template_state_hash(
        message('{"c":"d","a":"b"}'), [], {}
    ) == _messages_tools_template_state_hash(message({"a": "b", "c": "d"}), [], {})
    assert _messages_tools_template_state_hash(message('{"a":2.0}'), [], {}) == (
        _messages_tools_template_state_hash(message('{"a":2}'), [], {})
    )


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        pytest.param('{"a":1,"a":2}', "contains duplicate key: a", id="duplicate-key"),
        pytest.param("NaN", "contains non-finite number: NaN", id="nan-literal"),
        pytest.param("Infinity", "contains non-finite number: Infinity", id="infinity-literal"),
        pytest.param('{"a":-Infinity}', "contains non-finite number: -Infinity", id="negative-infinity-literal"),
        pytest.param('{"a":1e999}', "numbers must be finite", id="overflow-to-inf"),
        pytest.param('{"a":0.1+0.2}', "must be valid JSON", id="expression-not-json"),
        pytest.param("{", "must be valid JSON", id="truncated-json"),
        pytest.param(
            "[1,2]", "must be a JSON object or a JSON string encoding an object, got <class 'list'>", id="array"
        ),
        pytest.param(
            '"text"', "must be a JSON object or a JSON string encoding an object, got <class 'str'>", id="string"
        ),
        pytest.param(123, "must be a JSON object or a JSON string encoding an object, got <class 'int'>", id="number"),
        pytest.param(
            None, "must be a JSON object or a JSON string encoding an object, got <class 'NoneType'>", id="null"
        ),
        pytest.param('{"a":"\\ud800"}', "must be valid UTF-8", id="lone-surrogate"),
    ],
)
def test_serialization_tool_arguments_reject_non_json_values(arguments: Any, message: str) -> None:
    with pytest.raises((TypeError, ValueError)) as excinfo:
        _canonical_tool_arguments(arguments, field="tool.arguments")

    assert str(excinfo.value) == f"tool.arguments {message}"


def test_serialization_template_kwargs_sort_nested_keys_recursively() -> None:
    assert normalize_template_kwargs(_TEMPLATE_KWARGS) == _TEMPLATE_KWARGS
    assert normalize_template_kwargs(_reorder(_TEMPLATE_KWARGS)) == normalize_template_kwargs(_TEMPLATE_KWARGS)
    assert normalize_template_kwargs(None) == {}


def test_serialization_template_kwargs_keep_number_types_verbatim() -> None:
    # Only tool arguments get the JavaScript-compatible integral-float rule, so
    # these three representations stay distinct for the state hash.
    digests = {_messages_tools_template_state_hash([], [], {"threshold": value}) for value in (1, 1.0, True)}

    assert len(digests) == 3


def test_serialization_tools_project_only_function_schema_fields() -> None:
    assert normalize_tools(
        [
            {"type": "custom", "name": "skipped"},
            {"type": "function", "function": "not-a-dict"},
            {"type": "function"},
            _TOOLS[0],
            {"type": "function", "function": {"name": "bare", "parameters": {}, "description": None}},
        ]
    ) == [
        _TOOLS[0],
        {"type": "function", "function": {"name": "bare", "parameters": {}}},
    ]
    assert normalize_tools(None) == []
