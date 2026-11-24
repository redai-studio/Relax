# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Error-path coverage for multi-protocol canonical request normalization."""

from __future__ import annotations

import re
from typing import Any, Callable

import pytest

from relax.agentic.session.service import (
    AgenticChatRequestError,
    _normalized_anthropic_request,
    _normalized_chat_request,
    _normalized_responses_request,
)
from relax.agentic.session.state import check_messages


def _raises_with_path(fn: Callable[[], Any], *, path: str) -> None:
    with pytest.raises((ValueError, TypeError, AgenticChatRequestError), match=re.escape(path)):
        fn()


@pytest.mark.parametrize(
    ("messages", "path"),
    [
        ([{"role": "bogus", "content": "x"}], "messages[0].role"),
        ([{"role": 1, "content": "x"}], "messages[0].role"),
        ([{"role": "user", "content": ""}], "messages[0].content"),
        ([{"role": "user", "content": []}], "messages[0].content"),
        ([{"role": "user", "content": [{"type": "text", "text": ""}]}], "messages[0].content[0].text"),
        (
            [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": ""}}]}],
            "messages[0].content[0].image_url.url",
        ),
        (
            [{"role": "user", "content": [{"type": "image_url", "image_url": {}}]}],
            "messages[0].content[0].image_url.url",
        ),
        (
            [{"role": "user", "content": [{"type": "input_image", "image_url": "https://x"}]}],
            "messages[0].content[0].type",
        ),
        (
            [
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [{"type": "function", "function": {"name": "x", "arguments": "{}"}}],
                }
            ],
            "messages[0].tool_calls[0].id",
        ),
        (
            [
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [{"id": "", "type": "function", "function": {"name": "x", "arguments": "{}"}}],
                }
            ],
            "messages[0].tool_calls[0].id",
        ),
        (
            [
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [{"id": "call_1", "type": "function", "function": {"arguments": "{}"}}],
                }
            ],
            "messages[0].tool_calls[0].function.name",
        ),
        (
            [
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {"id": "call_1", "type": "function", "function": {"name": "x", "arguments": "not-json"}}
                    ],
                }
            ],
            "messages[0].tool_calls[0].function.arguments",
        ),
        (
            [
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [{"id": "call_1"}],
                }
            ],
            "messages[0].tool_calls[0].function",
        ),
        ([{"role": "tool", "tool_call_id": "", "content": "result"}], "messages[0].tool_call_id"),
        (
            [
                {
                    "role": "user",
                    "content": "hi",
                    "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "x", "arguments": "{}"}}],
                }
            ],
            "messages[0].tool_calls",
        ),
    ],
)
def test_check_messages_rejects_anomalies_with_stable_field_paths(
    messages: list[dict[str, Any]],
    path: str,
) -> None:
    _raises_with_path(lambda: check_messages(messages), path=path)


def test_check_messages_allows_dataset_image_parts_before_media_transport() -> None:
    messages = check_messages(
        [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "describe"},
                    {"type": "image", "image": "/data/cat.png"},
                ],
            }
        ]
    )
    assert messages[0]["content"][1] == {"type": "image", "image": "/data/cat.png"}


def test_check_messages_allows_tool_observation_without_tool_call_id() -> None:
    assert check_messages([{"role": "tool", "content": "observation"}]) == [{"role": "tool", "content": "observation"}]


def test_check_messages_collapses_text_only_lists_to_string() -> None:
    assert check_messages(
        [{"role": "user", "content": [{"type": "text", "text": "hello"}, {"type": "text", "text": " world"}]}]
    ) == [{"role": "user", "content": "hello world"}]


def test_check_messages_joins_system_text_blocks_with_paragraph_separator() -> None:
    assert check_messages(
        [
            {
                "role": "system",
                "content": [{"type": "text", "text": "Rule A"}, {"type": "text", "text": "Rule B"}],
            },
            {"role": "user", "content": "hi"},
        ]
    ) == [{"role": "system", "content": "Rule A\n\nRule B"}, {"role": "user", "content": "hi"}]


def test_chat_protocol_surfaces_check_messages_field_paths() -> None:
    with pytest.raises(AgenticChatRequestError, match=re.escape("messages[2].tool_calls[0].id")):
        _normalized_chat_request(
            {
                "messages": [
                    {"role": "user", "content": "hi"},
                    {"role": "assistant", "content": "thinking"},
                    {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "type": "function",
                                "function": {"name": "search", "arguments": "{}"},
                            }
                        ],
                    },
                ]
            }
        )


@pytest.mark.parametrize(
    ("payload", "path"),
    [
        (
            {"input": [{"role": "user", "content": [{"type": "input_image", "image_url": ""}]}]},
            "input[0].content[0].image_url",
        ),
        (
            {
                "input": [
                    {
                        "type": "function_call",
                        "name": "search",
                        "arguments": "{}",
                    }
                ]
            },
            "input[0].call_id",
        ),
        (
            {
                "input": [
                    {
                        "type": "function_call_output",
                        "output": "done",
                    }
                ]
            },
            "input[0].call_id",
        ),
        (
            {
                "input": [
                    {
                        "type": "function_call",
                        "call_id": "call_1",
                        "name": "search",
                        "arguments": {"q": "x"},
                    }
                ]
            },
            "input[0].arguments",
        ),
    ],
)
def test_responses_rejects_illegal_inputs_with_field_paths(payload: dict[str, Any], path: str) -> None:
    _raises_with_path(lambda: _normalized_responses_request(payload), path=path)


@pytest.mark.parametrize(
    ("payload", "path"),
    [
        (
            {
                "max_tokens": 16,
                "messages": [
                    {
                        "role": "user",
                        "content": [{"type": "image", "source": {"type": "url", "url": ""}}],
                    }
                ],
            },
            "messages[0].content[0].source.url",
        ),
        (
            {
                "max_tokens": 16,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "source": {"type": "base64", "media_type": "text/plain", "data": "x"}}
                        ],
                    }
                ],
            },
            "messages[0].content[0].source.media_type",
        ),
        (
            {
                "max_tokens": 16,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {"type": "base64", "media_type": "image/png", "data": ""},
                            }
                        ],
                    }
                ],
            },
            "messages[0].content[0].source.data",
        ),
        (
            {
                "max_tokens": 16,
                "messages": [
                    {
                        "role": "user",
                        "content": [{"type": "image", "source": {"type": "file"}}],
                    }
                ],
            },
            "messages[0].content[0].source.type",
        ),
        (
            {
                "max_tokens": 16,
                "messages": [
                    {
                        "role": "assistant",
                        "content": [{"type": "tool_use", "name": "search", "input": {}}],
                    }
                ],
            },
            "messages[0].content[0].id",
        ),
        (
            {
                "max_tokens": 16,
                "messages": [
                    {
                        "role": "user",
                        "content": [{"type": "tool_result", "content": "x"}],
                    }
                ],
            },
            "messages[0].content[0].tool_use_id",
        ),
        (
            {
                "max_tokens": 16,
                "messages": [
                    {
                        "role": "assistant",
                        "content": [{"type": "tool_use", "id": "call_1", "name": "search", "input": "x"}],
                    }
                ],
            },
            "messages[0].content[0].input",
        ),
    ],
)
def test_anthropic_rejects_illegal_inputs_with_field_paths(payload: dict[str, Any], path: str) -> None:
    _raises_with_path(lambda: _normalized_anthropic_request(payload), path=path)


def test_chat_rejects_empty_string_and_empty_list_content() -> None:
    with pytest.raises(AgenticChatRequestError, match=re.escape("messages[0].content must not be empty")):
        _normalized_chat_request({"messages": [{"role": "user", "content": ""}]})
    with pytest.raises(AgenticChatRequestError, match=re.escape("messages[0].content must not be empty")):
        _normalized_chat_request({"messages": [{"role": "user", "content": []}]})
