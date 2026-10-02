# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Golden tests for the three agentic request protocols.

Relax accepts Chat Completions, OpenAI Responses and Anthropic Messages requests
and projects all three onto one canonical ``messages`` / ``tools`` /
``chat_template_kwargs`` triple, which is what SessionForest matches on; see
``docs/zh/guide/agentic-rollout.md``.

Every expected value below was captured from the normalizers themselves, so a
deliberate projection change has to update this file in the same commit. Two
layers refuse a malformed request: the protocol projection names its own field in
``param`` and chains no cause, while the shared canonical validator in
``check_messages`` / ``normalize_tools`` / ``normalize_template_kwargs`` uses
``param="messages"`` and chains a ``ValueError`` or ``TypeError``.

CPU only: no model, network or GPU is involved.
"""

from __future__ import annotations

from typing import Any, Callable

import pytest

from relax.agentic.session.service import (
    AgenticChatRequestError,
    _normalized_anthropic_request,
    _normalized_chat_request,
    _normalized_responses_request,
)
from relax.agentic.session.state import _messages_tools_template_state_hash, check_messages


PROTOCOLS: dict[str, Callable[[dict[str, Any]], dict[str, Any]]] = {
    "chat": _normalized_chat_request,
    "responses": _normalized_responses_request,
    "anthropic": _normalized_anthropic_request,
}

# The three fields the canonical state consists of.
CANONICAL_FIELDS = ("messages", "tools", "chat_template_kwargs")

SYSTEM = "You are a helpful agent."
VISION_SYSTEM = "You are a vision agent."
ASK = "What is the weather in Taiyuan?"
QUESTION = "What is in this image?"
MAX_TOKENS = 512
CALL_ID = "call_weather_1"
TOOL_NAME = "get_weather"
TOOL_DESCRIPTION = "Look up the weather for a city."
TOOL_PARAMETERS = {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}
TOOL_RESULT_TEXT = "18 C, clear"
ASSISTANT_TEXT = "two"
REASONING = "The user is asking about the weather."

# Two assistant turns that differ only in whether they also carry text, so a
# protocol that drops reasoning_content fails on the first and one that drops an
# empty text on the second.
REASONING_ASSISTANT = {"role": "assistant", "content": ASSISTANT_TEXT, "reasoning_content": REASONING}
REASONING_ONLY_ASSISTANT = {"role": "assistant", "content": "", "reasoning_content": REASONING}

# The three spellings of one tool-argument object: the first two differ only in key
# order and whitespace, so the assertions prove the arguments are parsed and
# re-serialized rather than passed through. The third is the canonical form.
ARGS_SPACED = '{"unit": "celsius", "city": "Taiyuan"}'
ARGS_REORDERED = '{"city": "Taiyuan", "unit": "celsius"}'
ARGS_CANONICAL = '{"city":"Taiyuan","unit":"celsius"}'

IMAGE_URL = "https://example.test/cat.png"
IMAGE_DATA_URI = "data:image/png;base64,iVBORw0KGgo="
URL_IMAGE_BLOCK = {"type": "image_url", "image_url": {"url": IMAGE_URL}}
INLINE_IMAGE_BLOCK = {"type": "image_url", "image_url": {"url": IMAGE_DATA_URI}}

SYSTEM_MSG = {"role": "system", "content": SYSTEM}
VISION_SYSTEM_MSG = {"role": "system", "content": VISION_SYSTEM}
USER_MSG = {"role": "user", "content": ASK}
ONE_MSG = {"role": "user", "content": "one"}
TOOL_RESULT_MSG = {"role": "tool", "tool_call_id": CALL_ID, "content": TOOL_RESULT_TEXT}

CANONICAL_TOOL = {
    "type": "function",
    "function": {"name": TOOL_NAME, "description": TOOL_DESCRIPTION, "parameters": TOOL_PARAMETERS},
}
CANONICAL_TOOL_CALL = {
    "role": "assistant",
    "content": "",
    "tool_calls": [{"id": CALL_ID, "type": "function", "function": {"name": TOOL_NAME, "arguments": ARGS_CANONICAL}}],
}


def _text_block(text: str) -> dict[str, Any]:
    return {"type": "text", "text": text}


def _chat_tool_call(arguments: str = ARGS_SPACED, call_id: str = CALL_ID) -> dict[str, Any]:
    tool_call = {"id": call_id, "type": "function", "function": {"name": TOOL_NAME, "arguments": arguments}}
    return {"role": "assistant", "content": "", "tool_calls": [tool_call]}


def _responses_call(arguments: str = ARGS_REORDERED, call_id: str = CALL_ID) -> dict[str, Any]:
    return {"type": "function_call", "call_id": call_id, "name": TOOL_NAME, "arguments": arguments}


def _anthropic_tool_use(call_id: str = CALL_ID, tool_input: dict[str, Any] | None = None) -> dict[str, Any]:
    if tool_input is None:
        tool_input = {"city": "Taiyuan", "unit": "celsius"}
    return {"type": "tool_use", "id": call_id, "name": TOOL_NAME, "input": tool_input}


def _message(text: str) -> dict[str, Any]:
    return {"type": "message", "role": "user", "content": text}


def _output_call(call_id: str = CALL_ID, output: str = TOOL_RESULT_TEXT) -> dict[str, Any]:
    return {"type": "function_call_output", "call_id": call_id, "output": output}


def _tool_result_block(content: str = TOOL_RESULT_TEXT, tool_use_id: str = CALL_ID) -> dict[str, Any]:
    return {"type": "tool_result", "tool_use_id": tool_use_id, "content": content}


def _base64_image_block() -> dict[str, Any]:
    return {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "iVBORw0KGgo="}}


def _anonymous_tool_call() -> dict[str, Any]:
    """An assistant tool call that carries no ``id`` at all."""

    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [{"type": "function", "function": {"name": TOOL_NAME, "arguments": "{}"}}],
    }


def _tool_result_msg(tool_call_id: str) -> dict[str, Any]:
    return {"role": "tool", "tool_call_id": tool_call_id, "content": "out"}


def _input_text_block(text: str) -> dict[str, Any]:
    return {"type": "input_text", "text": text}


def _output_text_block(text: str) -> dict[str, Any]:
    return {"type": "output_text", "text": text}


def _thinking_block(text: str = REASONING) -> dict[str, Any]:
    return {"type": "thinking", "thinking": text}


def _reasoning_item(text: str = REASONING) -> dict[str, Any]:
    """A Responses reasoning Item that carries its text in ``summary``."""

    return {"type": "reasoning", "summary": [{"type": "summary_text", "text": text}]}


def _assistant_message(text: str) -> dict[str, Any]:
    return {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": text}]}


def _anthropic_assistant(*blocks: dict[str, Any]) -> dict[str, Any]:
    return {"role": "assistant", "content": list(blocks)}


def _output_call_blocks(*parts: str) -> dict[str, Any]:
    """A Responses call output written as ``input_text`` blocks instead of a
    string."""

    return {"type": "function_call_output", "call_id": CALL_ID, "output": [_input_text_block(part) for part in parts]}


def _tool_result_blocks(*parts: str) -> dict[str, Any]:
    """A Messages tool result whose content is text blocks instead of a
    string."""

    return {"type": "tool_result", "tool_use_id": CALL_ID, "content": [_text_block(part) for part in parts]}


def _canonical_triple(protocol: str, payload: dict[str, Any]) -> dict[str, Any]:
    normalized = PROTOCOLS[protocol](payload)
    return {field: normalized[field] for field in CANONICAL_FIELDS}


def _state_hash(triple: dict[str, Any]) -> str:
    return _messages_tools_template_state_hash(triple["messages"], triple["tools"], triple["chat_template_kwargs"])


def _reorder_keys(value: Any) -> Any:
    """Reverse dict key insertion order recursively without changing
    content."""

    if isinstance(value, dict):
        return {key: _reorder_keys(value[key]) for key in reversed(list(value))}
    if isinstance(value, list):
        return [_reorder_keys(item) for item in value]
    return value


# One semantic conversation written in all three request shapes, plus the single
# canonical triple they must all project onto.
CANONICAL_SCENARIOS: dict[str, dict[str, Any]] = {
    "text_only": {
        "expected": {"messages": [SYSTEM_MSG, USER_MSG], "tools": [], "chat_template_kwargs": {}},
        "chat": {"messages": [SYSTEM_MSG, USER_MSG]},
        "responses": {"instructions": SYSTEM, "input": [_message(ASK)]},
        "anthropic": {"max_tokens": MAX_TOKENS, "system": SYSTEM, "messages": [USER_MSG]},
    },
    # Covers both halves of a tool exchange: the assistant call and the tool result.
    "tool_call_and_result": {
        "expected": {
            "messages": [SYSTEM_MSG, USER_MSG, CANONICAL_TOOL_CALL, TOOL_RESULT_MSG],
            "tools": [],
            "chat_template_kwargs": {},
        },
        "chat": {"messages": [SYSTEM_MSG, USER_MSG, _chat_tool_call(), TOOL_RESULT_MSG]},
        "responses": {
            "instructions": SYSTEM,
            "input": [_message(ASK), _responses_call(), _output_call()],
        },
        "anthropic": {
            "max_tokens": MAX_TOKENS,
            "system": SYSTEM,
            "messages": [
                USER_MSG,
                {"role": "assistant", "content": [_anthropic_tool_use()]},
                {"role": "user", "content": [_tool_result_block(TOOL_RESULT_TEXT)]},
            ],
        },
    },
    "image_input_url": {
        "expected": {
            "messages": [VISION_SYSTEM_MSG, {"role": "user", "content": [_text_block(QUESTION), URL_IMAGE_BLOCK]}],
            "tools": [],
            "chat_template_kwargs": {},
        },
        "chat": {
            "messages": [VISION_SYSTEM_MSG, {"role": "user", "content": [_text_block(QUESTION), URL_IMAGE_BLOCK]}]
        },
        "responses": {
            "instructions": VISION_SYSTEM,
            "input": [
                {
                    "type": "message",
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": QUESTION},
                        {"type": "input_image", "image_url": IMAGE_URL},
                    ],
                }
            ],
        },
        "anthropic": {
            "max_tokens": MAX_TOKENS,
            "system": VISION_SYSTEM,
            "messages": [
                {
                    "role": "user",
                    "content": [_text_block(QUESTION), {"type": "image", "source": {"type": "url", "url": IMAGE_URL}}],
                }
            ],
        },
    },
    # An Anthropic base64 source has to become the same data URI the other two carry.
    "image_input_base64": {
        "expected": {
            "messages": [{"role": "user", "content": [INLINE_IMAGE_BLOCK]}],
            "tools": [],
            "chat_template_kwargs": {},
        },
        "chat": {"messages": [{"role": "user", "content": [INLINE_IMAGE_BLOCK]}]},
        "responses": {
            "input": [
                {"type": "message", "role": "user", "content": [{"type": "input_image", "image_url": IMAGE_DATA_URI}]}
            ]
        },
        "anthropic": {
            "max_tokens": MAX_TOKENS,
            "messages": [{"role": "user", "content": [_base64_image_block()]}],
        },
    },
    "tools": {
        "expected": {"messages": [USER_MSG], "tools": [CANONICAL_TOOL], "chat_template_kwargs": {}},
        "chat": {"messages": [USER_MSG], "tools": [CANONICAL_TOOL]},
        "responses": {
            "input": [_message(ASK)],
            "tools": [
                {"type": "function", "name": TOOL_NAME, "description": TOOL_DESCRIPTION, "parameters": TOOL_PARAMETERS}
            ],
        },
        "anthropic": {
            "max_tokens": MAX_TOKENS,
            "messages": [USER_MSG],
            "tools": [{"name": TOOL_NAME, "description": TOOL_DESCRIPTION, "input_schema": TOOL_PARAMETERS}],
        },
    },
    # Each protocol expresses "think before answering" with its own native field.
    "thinking_enabled": {
        "expected": {"messages": [ONE_MSG], "tools": [], "chat_template_kwargs": {"enable_thinking": True}},
        "chat": {"messages": [ONE_MSG], "chat_template_kwargs": {"enable_thinking": True}},
        "responses": {"input": [_message("one")], "reasoning": {"effort": "medium"}},
        "anthropic": {"max_tokens": MAX_TOKENS, "messages": [ONE_MSG], "thinking": {"type": "enabled"}},
    },
    "thinking_disabled": {
        "expected": {"messages": [ONE_MSG], "tools": [], "chat_template_kwargs": {"enable_thinking": False}},
        "chat": {"messages": [ONE_MSG], "chat_template_kwargs": {"enable_thinking": False}},
        "responses": {"input": [_message("one")], "reasoning": {"effort": "none"}},
        "anthropic": {"max_tokens": MAX_TOKENS, "messages": [ONE_MSG], "thinking": {"type": "disabled"}},
    },
    # Reasoning is part of the canonical state, so a prior assistant turn has to
    # survive a round trip through every protocol.
    "assistant_reasoning": {
        "expected": {"messages": [ONE_MSG, REASONING_ASSISTANT], "tools": [], "chat_template_kwargs": {}},
        "chat": {"messages": [ONE_MSG, REASONING_ASSISTANT]},
        "responses": {"input": [_message("one"), _reasoning_item(), _assistant_message(ASSISTANT_TEXT)]},
        "anthropic": {
            "max_tokens": MAX_TOKENS,
            "messages": [ONE_MSG, _anthropic_assistant(_thinking_block(), _text_block(ASSISTANT_TEXT))],
        },
    },
    # A reasoning turn with no text at all is legal, because reasoning_content
    # fills the non-empty slot that content would otherwise have to fill.
    "assistant_reasoning_only": {
        "expected": {"messages": [ONE_MSG, REASONING_ONLY_ASSISTANT], "tools": [], "chat_template_kwargs": {}},
        "chat": {"messages": [ONE_MSG, REASONING_ONLY_ASSISTANT]},
        "responses": {
            "input": [
                _message("one"),
                {"type": "reasoning", "content": [{"type": "reasoning_text", "text": REASONING}]},
            ]
        },
        "anthropic": {"max_tokens": MAX_TOKENS, "messages": [ONE_MSG, _anthropic_assistant(_thinking_block())]},
    },
    # Three spellings of one system message: a plain role, a Responses
    # ``developer`` role that has to be renamed, and Messages text blocks.
    "system_message_roles": {
        "expected": {"messages": [SYSTEM_MSG, USER_MSG], "tools": [], "chat_template_kwargs": {}},
        "chat": {"messages": [SYSTEM_MSG, USER_MSG]},
        "responses": {"input": [{"type": "message", "role": "developer", "content": SYSTEM}, _message(ASK)]},
        "anthropic": {"max_tokens": MAX_TOKENS, "system": [_text_block(SYSTEM)], "messages": [USER_MSG]},
    },
}


@pytest.mark.parametrize("protocol", PROTOCOLS)
@pytest.mark.parametrize("scenario", CANONICAL_SCENARIOS)
def test_protocol_projects_onto_the_golden_triple(scenario: str, protocol: str) -> None:
    case = CANONICAL_SCENARIOS[scenario]
    actual = _canonical_triple(protocol, case[protocol])
    for field in CANONICAL_FIELDS:
        assert actual[field] == case["expected"][field], f"{scenario}/{protocol}/{field}"


@pytest.mark.parametrize("scenario", CANONICAL_SCENARIOS)
def test_protocols_agree_on_the_canonical_triple(scenario: str) -> None:
    """Compare the protocols with each other, so a wrong golden triple cannot
    hide a divergence between them."""

    case = CANONICAL_SCENARIOS[scenario]
    triples = {protocol: _canonical_triple(protocol, case[protocol]) for protocol in PROTOCOLS}
    for protocol, triple in triples.items():
        assert triple == triples["chat"], f"{scenario}/{protocol} diverged from chat"

    hashes = {protocol: _state_hash(triple) for protocol, triple in triples.items()}
    assert len(set(hashes.values())) == 1, f"{scenario} state hashes diverged: {hashes}"


@pytest.mark.parametrize("protocol", PROTOCOLS)
@pytest.mark.parametrize("scenario", CANONICAL_SCENARIOS)
def test_state_hash_ignores_request_key_order(scenario: str, protocol: str) -> None:
    payload = CANONICAL_SCENARIOS[scenario][protocol]
    reordered = _canonical_triple(protocol, _reorder_keys(payload))
    assert _state_hash(reordered) == _state_hash(_canonical_triple(protocol, payload))


def test_tool_arguments_collapse_to_one_canonical_string() -> None:
    """Whitespace, key order and the string-versus-object encoding all collapse
    to one serialized form and one state hash."""

    encodings = [
        ARGS_SPACED,
        ARGS_REORDERED,
        '{ "city" : "Taiyuan" , "unit" : "celsius" }',
        {"unit": "celsius", "city": "Taiyuan"},
    ]
    serialized = set()
    hashes = set()
    for arguments in encodings:
        triple = _canonical_triple("chat", {"messages": [ONE_MSG, _chat_tool_call(arguments)]})
        serialized.add(triple["messages"][1]["tool_calls"][0]["function"]["arguments"])
        hashes.add(_state_hash(triple))

    assert serialized == {ARGS_CANONICAL}
    assert len(hashes) == 1


# Projection rules that exist in a single protocol, so they cannot be written as
# a three-protocol scenario. Each row names only the canonical fields it pins.
PROTOCOL_SPECIFIC = [
    (
        "responses-input-as-a-plain-string",
        "responses",
        {"input": "one"},
        {"messages": [ONE_MSG]},
    ),
    (
        "responses-output-text-block",
        "responses",
        {"input": [{"type": "message", "role": "user", "content": [_output_text_block("one")]}]},
        {"messages": [ONE_MSG]},
    ),
    (
        "responses-call-output-as-input-text-blocks",
        "responses",
        {"input": [_message("one"), _responses_call(), _output_call_blocks("18 C", ", clear")]},
        {"messages": [ONE_MSG, CANONICAL_TOOL_CALL, TOOL_RESULT_MSG]},
    ),
    (
        "anthropic-text-only-list-collapses-to-a-string",
        "anthropic",
        {"max_tokens": MAX_TOKENS, "messages": [{"role": "user", "content": [_text_block("one")]}]},
        {"messages": [ONE_MSG]},
    ),
    (
        "anthropic-tool-result-text-blocks-join",
        "anthropic",
        {
            "max_tokens": MAX_TOKENS,
            "messages": [{"role": "user", "content": [_tool_result_blocks("18 C", ", clear")]}],
        },
        {"messages": [{"role": "tool", "content": TOOL_RESULT_TEXT, "tool_call_id": CALL_ID}]},
    ),
    (
        "chat-bare-string-image-url-canonicalizes",
        "chat",
        {"messages": [{"role": "user", "content": [{"type": "image_url", "image_url": IMAGE_URL}]}]},
        {"messages": [{"role": "user", "content": [URL_IMAGE_BLOCK]}]},
    ),
    # A request parameter like OpenAI's ``detail`` rides on the call, not on the
    # session state identity, so canonicalization drops it.
    (
        "chat-image-url-request-only-keys-drop",
        "chat",
        {
            "messages": [
                {"role": "user", "content": [{"type": "image_url", "image_url": {"url": IMAGE_URL, "detail": "low"}}]}
            ]
        },
        {"messages": [{"role": "user", "content": [URL_IMAGE_BLOCK]}]},
    ),
    (
        "anthropic-custom-tool-type-still-becomes-a-function",
        "anthropic",
        {
            "max_tokens": MAX_TOKENS,
            "messages": [USER_MSG],
            "tools": [
                {
                    "type": "custom",
                    "name": TOOL_NAME,
                    "description": TOOL_DESCRIPTION,
                    "input_schema": TOOL_PARAMETERS,
                }
            ],
        },
        {"tools": [CANONICAL_TOOL]},
    ),
    (
        "anthropic-thinking-next-to-a-tool-use",
        "anthropic",
        {
            "max_tokens": MAX_TOKENS,
            "messages": [ONE_MSG, _anthropic_assistant(_thinking_block(), _anthropic_tool_use())],
        },
        {"messages": [ONE_MSG, {**CANONICAL_TOOL_CALL, "reasoning_content": REASONING}]},
    ),
]


@pytest.mark.parametrize(
    ("protocol", "payload", "expected"),
    [case[1:] for case in PROTOCOL_SPECIFIC],
    ids=[case[0] for case in PROTOCOL_SPECIFIC],
)
def test_protocol_specific_rule_matches_the_contract(
    protocol: str, payload: dict[str, Any], expected: dict[str, Any]
) -> None:
    actual = _canonical_triple(protocol, payload)
    for field, value in expected.items():
        assert actual[field] == value, f"{protocol}/{field}"


# Malformed requests that are refused. ``param`` names the field of the protocol
# that received the request; ``cause`` is ``None`` when the projection layer
# refused it and ``ValueError`` / ``TypeError`` when the canonical validator did.
REJECTED = [
    (
        "unknown-role-chat",
        "chat",
        {"messages": [ONE_MSG, {"role": "critic", "content": "two"}]},
        "messages[1].role must be one of: assistant, system, tool, user",
        "messages",
        ValueError,
    ),
    (
        "unknown-role-projects-to-nothing-responses",
        "responses",
        {"input": [{"role": "critic", "type": "message", "content": "two"}]},
        "input projected to zero supported messages",
        "input",
        None,
    ),
    (
        "tool-call-empty-id-chat",
        "chat",
        {"messages": [ONE_MSG, _chat_tool_call(call_id="")]},
        "messages[1].tool_calls[0].id must be a non-empty string",
        "messages",
        ValueError,
    ),
    (
        "tool-call-not-a-dict-chat",
        "chat",
        {"messages": [ONE_MSG, {"role": "assistant", "content": "", "tool_calls": ["x"]}]},
        "messages[1].tool_calls[0] must be a dict, got <class 'str'>",
        "messages",
        TypeError,
    ),
    (
        "tool-call-missing-id-chat",
        "chat",
        {"messages": [ONE_MSG, _anonymous_tool_call()]},
        "messages[1].tool_calls[0].id must be a non-empty string",
        "messages",
        ValueError,
    ),
    (
        "image-block-missing-url-chat",
        "chat",
        {"messages": [{"role": "user", "content": [{"type": "image_url"}]}]},
        "messages[0].content[0].image_url must be a string or a dict, got <class 'NoneType'>",
        "messages",
        TypeError,
    ),
    (
        "image-url-object-without-url-chat",
        "chat",
        {"messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {}}]}]},
        "messages[0].content[0].image_url.url must be a non-empty string",
        "messages",
        ValueError,
    ),
    (
        "image-url-empty-string-chat",
        "chat",
        {"messages": [{"role": "user", "content": [{"type": "image_url", "image_url": ""}]}]},
        "messages[0].content[0].image_url must be a non-empty string",
        "messages",
        ValueError,
    ),
    (
        "tool-call-missing-id-responses",
        "responses",
        {"input": [{"type": "function_call", "name": TOOL_NAME, "arguments": "{}"}]},
        "input[0].call_id must be a non-empty string",
        "input",
        None,
    ),
    (
        "tool-use-missing-id-anthropic",
        "anthropic",
        {
            "max_tokens": MAX_TOKENS,
            "messages": [
                ONE_MSG,
                {"role": "assistant", "content": [{"type": "tool_use", "name": TOOL_NAME, "input": {}}]},
            ],
        },
        "messages[1].content[0].id must be a non-empty string",
        "messages",
        None,
    ),
    (
        "empty-string-content-chat",
        "chat",
        {"messages": [ONE_MSG, {"role": "user", "content": ""}]},
        "messages[1].content must not be empty",
        "messages",
        ValueError,
    ),
    (
        "empty-list-content-chat",
        "chat",
        {"messages": [ONE_MSG, {"role": "user", "content": []}]},
        "messages[1].content must not be empty",
        "messages",
        ValueError,
    ),
    (
        "empty-content-behind-instructions-responses",
        "responses",
        {"instructions": "sys", "input": [_message(ASK), _message("")]},
        "messages[2].content must not be empty",
        "input",
        ValueError,
    ),
    (
        "empty-list-content-projects-to-nothing-anthropic",
        "anthropic",
        {"max_tokens": MAX_TOKENS, "messages": [{"role": "user", "content": []}]},
        "messages[0].content projected to no supported content",
        "messages",
        None,
    ),
    (
        "image-missing-url-responses",
        "responses",
        {"input": [{"type": "message", "role": "user", "content": [{"type": "input_image"}]}]},
        "input[0].content[0].image_url must be a non-empty string",
        "input",
        None,
    ),
    (
        "image-on-assistant-turn-responses",
        "responses",
        {
            "input": [
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "input_image", "image_url": IMAGE_URL}],
                }
            ]
        },
        "input[0].content projected to no supported content",
        "input",
        None,
    ),
    (
        "image-source-not-an-object-anthropic",
        "anthropic",
        {
            "max_tokens": MAX_TOKENS,
            "messages": [{"role": "user", "content": [{"type": "image", "source": IMAGE_URL}]}],
        },
        "messages[0].content[0].source must be a JSON object",
        "messages",
        None,
    ),
    (
        "image-source-type-unsupported-anthropic",
        "anthropic",
        {
            "max_tokens": MAX_TOKENS,
            "messages": [
                {"role": "user", "content": [{"type": "image", "source": {"type": "file", "file_id": "f1"}}]}
            ],
        },
        "messages[0].content[0].source.type is not supported",
        "messages",
        None,
    ),
    (
        "image-url-source-empty-anthropic",
        "anthropic",
        {
            "max_tokens": MAX_TOKENS,
            "messages": [{"role": "user", "content": [{"type": "image", "source": {"type": "url", "url": ""}}]}],
        },
        "messages[0].content[0].source.url must be non-empty",
        "messages",
        None,
    ),
    (
        "tools-entry-not-a-dict-chat",
        "chat",
        {"messages": [ONE_MSG], "tools": ["x"]},
        "tools[0] must be a dict, got <class 'str'>",
        "messages",
        TypeError,
    ),
    (
        "reserved-template-kwargs-chat",
        "chat",
        {"messages": [ONE_MSG], "chat_template_kwargs": {"tools": []}},
        "chat_template_kwargs cannot set reserved keys: tools",
        "chat_template_kwargs",
        None,
    ),
    # Chat Completions has no ``developer`` role, and the documentation requires
    # the harness to rename it; Responses accepts it and renames it itself.
    (
        "developer-role-chat",
        "chat",
        {"messages": [{"role": "developer", "content": SYSTEM}]},
        "messages[0].role must be one of: assistant, system, tool, user",
        "messages",
        ValueError,
    ),
    (
        "assistant-content-projects-to-nothing-responses",
        "responses",
        {"input": [_message(ASK), {"type": "message", "role": "assistant", "content": []}]},
        "input[1].content projected to no supported content",
        "input",
        None,
    ),
    # An empty thinking block is dropped, which leaves an assistant turn with an
    # empty content string for the canonical validator to refuse.
    (
        "empty-thinking-leaves-an-empty-assistant-anthropic",
        "anthropic",
        {"max_tokens": MAX_TOKENS, "messages": [_anthropic_assistant(_thinking_block(""))]},
        "messages[0].content must not be empty",
        "messages",
        ValueError,
    ),
    (
        "empty-text-block-in-assistant-anthropic",
        "anthropic",
        {"max_tokens": MAX_TOKENS, "messages": [_anthropic_assistant(_text_block(""))]},
        "messages[0].content[0].text must be non-empty",
        "messages",
        None,
    ),
]


@pytest.mark.parametrize(
    ("protocol", "payload", "message", "param", "cause"),
    [case[1:] for case in REJECTED],
    ids=[case[0] for case in REJECTED],
)
def test_malformed_request_reports_a_full_field_path(
    protocol: str, payload: dict[str, Any], message: str, param: str, cause: type[BaseException] | None
) -> None:
    with pytest.raises(AgenticChatRequestError) as excinfo:
        PROTOCOLS[protocol](payload)

    error = excinfo.value
    assert error.message == message
    assert error.param == param
    assert (type(error.__cause__) if error.__cause__ is not None else None) is cause


# Anomalies that are not refused, and that the three protocols do not even agree
# on: an unrecognised role is dropped by Responses and Messages but rejected by
# Chat Completions, while an incomplete tool exchange is accepted everywhere. Each
# row pins the canonical state the request actually produces, so an accepted shape
# is documented instead of quietly passing. The gaps these rows expose are listed
# in the PR description.
TOLERATED = [
    (
        "unknown-role-dropped-responses",
        "responses",
        {"input": [_message("one"), {"role": "critic", "type": "message", "content": "two"}]},
        [ONE_MSG],
    ),
    (
        "unknown-role-dropped-anthropic",
        "anthropic",
        {"max_tokens": MAX_TOKENS, "messages": [ONE_MSG, {"role": "critic", "content": "two"}]},
        [ONE_MSG],
    ),
    (
        "tool-result-without-a-call-chat",
        "chat",
        {"messages": [ONE_MSG, _tool_result_msg("never-called")]},
        [ONE_MSG, {"role": "tool", "content": "out", "tool_call_id": "never-called"}],
    ),
    (
        "tool-result-without-a-call-responses",
        "responses",
        {"input": [{"type": "function_call_output", "call_id": "never-called", "output": "out"}]},
        [{"role": "tool", "content": "out", "tool_call_id": "never-called"}],
    ),
    (
        "tool-result-without-a-call-anthropic",
        "anthropic",
        {
            "max_tokens": MAX_TOKENS,
            "messages": [{"role": "user", "content": [_tool_result_block("out", "never-called")]}],
        },
        [{"role": "tool", "content": "out", "tool_call_id": "never-called"}],
    ),
    (
        "tool-call-without-a-result-anthropic",
        "anthropic",
        {
            "max_tokens": MAX_TOKENS,
            "messages": [{"role": "assistant", "content": [_anthropic_tool_use(tool_input={})]}],
        },
        [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"id": CALL_ID, "type": "function", "function": {"name": TOOL_NAME, "arguments": "{}"}}
                ],
            }
        ],
    ),
    (
        "empty-tool-result-text-anthropic",
        "anthropic",
        {"max_tokens": MAX_TOKENS, "messages": [{"role": "user", "content": [_tool_result_block("")]}]},
        [{"role": "tool", "content": "", "tool_call_id": CALL_ID}],
    ),
]


@pytest.mark.parametrize(
    ("protocol", "payload", "expected_messages"),
    [case[1:] for case in TOLERATED],
    ids=[case[0] for case in TOLERATED],
)
def test_anomaly_that_is_not_refused_keeps_a_stable_representation(
    protocol: str, payload: dict[str, Any], expected_messages: list[dict[str, Any]]
) -> None:
    assert PROTOCOLS[protocol](payload)["messages"] == expected_messages


# A dataset prompt reaches ``check_messages`` in ``relax/agentic/pipeline/runtime.py``
# before ``_transport_dataset_message_media`` rewrites its legacy ``{"type": "image"}``
# parts into canonical ``image_url`` ones, so this validator sees a shape no protocol
# produces. Refusing an unknown part type, or demanding a ``tool_call_id`` from every
# observation, would break that path before the rewrite ever runs.
DATASET_TURNS = [
    (
        "legacy-image-part-awaits-media-transport",
        {"role": "user", "content": [_text_block(QUESTION), {"type": "image", "image": "file:///cat.png"}]},
    ),
    ("tool-observation-without-a-call-id", {"role": "tool", "content": TOOL_RESULT_TEXT}),
]


@pytest.mark.parametrize(
    "message",
    [case[1] for case in DATASET_TURNS],
    ids=[case[0] for case in DATASET_TURNS],
)
def test_dataset_turn_survives_the_shared_validator(message: dict[str, Any]) -> None:
    assert check_messages([message]) == [message]
