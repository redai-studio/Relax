# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Rejection tests for the three agentic ingress protocols.

Every case pins the complete error message, not a substring, so a refactor that
reparents a field path or renames a message is caught here. ``param`` is the
protocol-native request field the client sent.
"""

from __future__ import annotations

from typing import Any

import pytest

from relax.agentic.session.service import AgenticChatRequestError
from relax.agentic.session.state import _multimodal_inputs_from_messages
from tests.agentic.helpers import canonical_case_ids, canonical_of, normalize, protocol_payloads


CHAT = "chat_completions"
RESPONSES = "responses"
ANTHROPIC = "anthropic_messages"


def _chat(messages: Any, **extra: Any) -> dict[str, Any]:
    return {"messages": messages, **extra}


def _responses_input(items: Any, **extra: Any) -> dict[str, Any]:
    return {"input": items, **extra}


def _anthropic(messages: Any, **extra: Any) -> dict[str, Any]:
    return {"max_tokens": 8, "messages": messages, **extra}


def _user(content: Any) -> dict[str, Any]:
    return {"role": "user", "content": content}


def _assistant_tool_calls(calls: Any) -> dict[str, Any]:
    return {"role": "assistant", "content": None, "tool_calls": calls}


def _tool_call(call: dict[str, Any]) -> dict[str, Any]:
    return {"type": "function", "function": {"name": "run", "arguments": "{}"}} | call


def assert_rejected(protocol: str, payload: dict[str, Any], message: str, param: str) -> None:
    with pytest.raises(AgenticChatRequestError) as excinfo:
        normalize(protocol, payload)

    error = excinfo.value
    assert error.message == message
    assert error.param == param
    assert error.status_code == 400


@pytest.mark.parametrize(
    ("protocol", "payload", "message", "param"),
    [
        pytest.param(
            CHAT, _chat([{"content": "x"}]), "messages[0] must include role", "messages", id="chat-role-missing"
        ),
        pytest.param(
            CHAT,
            _chat([{"role": 123, "content": "x"}]),
            "messages[0].role must be a string, got <class 'int'>",
            "messages",
            id="chat-role-not-a-string",
        ),
        pytest.param(
            CHAT,
            _chat([{"role": "developer", "content": "x"}]),
            "messages[0].role must be one of: assistant, system, tool, user",
            "messages",
            id="chat-role-unknown",
        ),
        pytest.param(
            CHAT,
            _chat(["hi"]),
            "messages[0] must be a dict, got <class 'str'>",
            "messages",
            id="chat-item-not-a-dict",
        ),
        pytest.param(
            CHAT, _chat("hi"), "messages must be a list, got <class 'str'>", "messages", id="chat-not-a-list"
        ),
        pytest.param(
            CHAT,
            _chat([]),
            "messages projected to zero supported messages",
            "messages",
            id="chat-empty-list",
        ),
        pytest.param(
            CHAT, {}, "messages projected to zero supported messages", "messages", id="chat-messages-missing"
        ),
        pytest.param(
            RESPONSES,
            _responses_input([{"role": "tool", "content": "x"}]),
            "input projected to zero supported messages",
            "input",
            id="responses-role-tool-dropped",
        ),
        pytest.param(
            ANTHROPIC,
            _anthropic([{"role": "system", "content": "x"}]),
            "messages projected to zero supported messages",
            "messages",
            id="anthropic-role-system-dropped",
        ),
    ],
)
def test_canonical_role_errors_report_field_paths(
    protocol: str,
    payload: dict[str, Any],
    message: str,
    param: str,
) -> None:
    assert_rejected(protocol, payload, message, param)


def test_canonical_developer_role_maps_to_system_only_for_responses() -> None:
    # ``developer`` is part of the Responses role vocabulary and maps onto the
    # canonical ``system`` role; the other two protocols must reject it.
    assert canonical_of(normalize(RESPONSES, _responses_input([{"role": "developer", "content": "S"}])))[
        "messages"
    ] == [{"role": "system", "content": "S"}]
    assert_rejected(
        CHAT,
        _chat([{"role": "developer", "content": "x"}]),
        "messages[0].role must be one of: assistant, system, tool, user",
        "messages",
    )


@pytest.mark.parametrize(
    ("protocol", "payload", "message", "param"),
    [
        pytest.param(
            CHAT, _chat([{"role": "user"}]), "messages[0] must include content", "messages", id="chat-content-missing"
        ),
        pytest.param(
            CHAT,
            _chat([_user(None)]),
            "messages[0].content must not be empty",
            "messages",
            id="chat-content-none",
        ),
        pytest.param(
            CHAT,
            _chat([_user("")]),
            "messages[0].content must not be empty",
            "messages",
            id="chat-content-empty-string",
        ),
        pytest.param(
            CHAT,
            _chat([_user([])]),
            "messages[0].content must not be empty",
            "messages",
            id="chat-content-empty-list",
        ),
        pytest.param(
            CHAT,
            _chat([_user(["x"])]),
            "messages[0].content[0] must be a dict, got <class 'str'>",
            "messages",
            id="chat-content-item-not-a-dict",
        ),
        pytest.param(
            CHAT,
            _chat([_user([{"type": "text", "text": ""}])]),
            "messages[0].content[0].text must not be empty",
            "messages",
            id="chat-content-text-block-empty",
        ),
        pytest.param(
            CHAT,
            _chat([_user(123)]),
            "messages[0].content must be a list, string, or None, got <class 'int'>",
            "messages",
            id="chat-content-not-a-string-or-list",
        ),
        pytest.param(
            CHAT,
            _chat([_user("hi"), {"role": "assistant", "content": ""}]),
            "messages[1].content must not be empty",
            "messages",
            id="chat-assistant-empty-without-tool-call",
        ),
        pytest.param(
            RESPONSES,
            _responses_input([{"role": "user", "content": [{"type": "input_text", "text": ""}]}]),
            "input[0].content[0] text must be non-empty",
            "input",
            id="responses-text-block-empty",
        ),
        pytest.param(
            RESPONSES,
            _responses_input([{"role": "user", "content": 123}]),
            "input[0].content must be a string or list",
            "input",
            id="responses-content-not-a-string-or-list",
        ),
        pytest.param(
            ANTHROPIC,
            _anthropic([_user([{"type": "text", "text": ""}])]),
            "messages[0].content[0].text must be non-empty",
            "messages",
            id="anthropic-text-block-empty",
        ),
        pytest.param(
            ANTHROPIC,
            _anthropic([_user(123)]),
            "messages[0].content must be a string or list",
            "messages",
            id="anthropic-content-not-a-string-or-list",
        ),
        pytest.param(
            ANTHROPIC,
            _anthropic([_user([{"type": "document", "x": 1}])]),
            "messages[0].content projected to no supported content",
            "messages",
            id="anthropic-content-projected-empty",
        ),
    ],
)
def test_canonical_content_errors_report_field_paths(
    protocol: str,
    payload: dict[str, Any],
    message: str,
    param: str,
) -> None:
    assert_rejected(protocol, payload, message, param)


@pytest.mark.parametrize(
    ("protocol", "payload", "message", "param"),
    [
        pytest.param(
            CHAT,
            _chat([_user("hi"), {"role": "assistant", "content": None, "tool_calls": "x"}]),
            "messages[1].tool_calls must be a list, got <class 'str'>",
            "messages",
            id="chat-tool-calls-not-a-list",
        ),
        pytest.param(
            CHAT,
            _chat([_user("hi"), _assistant_tool_calls(["x"])]),
            "messages[1].tool_calls[0] must be a dict, got <class 'str'>",
            "messages",
            id="chat-tool-call-not-a-dict",
        ),
        pytest.param(
            CHAT,
            _chat([_user("hi"), _assistant_tool_calls([{"id": "call-1"}])]),
            "messages[1].tool_calls[0].function must be a dict, got <class 'NoneType'>",
            "messages",
            id="chat-function-missing",
        ),
        pytest.param(
            CHAT,
            _chat([_user("hi"), _assistant_tool_calls([{"id": "call-1", "function": "x"}])]),
            "messages[1].tool_calls[0].function must be a dict, got <class 'str'>",
            "messages",
            id="chat-function-not-a-dict",
        ),
        pytest.param(
            CHAT,
            _chat([_user("hi"), _assistant_tool_calls([_tool_call({"function": {"name": 1, "arguments": "{}"}})])]),
            "messages[1].tool_calls[0].function.name must be a string, got <class 'int'>",
            "messages",
            id="chat-function-name-not-a-string",
        ),
        pytest.param(
            CHAT,
            _chat([_user("hi"), _assistant_tool_calls([_tool_call({"function": {"arguments": "{a"}})])]),
            "messages[1].tool_calls[0].function.arguments must be valid JSON",
            "messages",
            id="chat-arguments-invalid-json",
        ),
        pytest.param(
            CHAT,
            _chat([_user("hi"), _assistant_tool_calls([_tool_call({"function": {"arguments": '"x"'}})])]),
            "messages[1].tool_calls[0].function.arguments must be a JSON object or a JSON string encoding an object,"
            " got <class 'str'>",
            "messages",
            id="chat-arguments-json-scalar",
        ),
        pytest.param(
            CHAT,
            _chat([_user("hi"), _assistant_tool_calls([_tool_call({"function": {"arguments": [1]}})])]),
            "messages[1].tool_calls[0].function.arguments must be a JSON object or a JSON string encoding an object,"
            " got <class 'list'>",
            "messages",
            id="chat-arguments-list",
        ),
        pytest.param(
            CHAT,
            _chat([_user("hi"), _assistant_tool_calls([_tool_call({"function": {"arguments": '{"a":1,"a":2}'}})])]),
            "messages[1].tool_calls[0].function.arguments contains duplicate key: a",
            "messages",
            id="chat-arguments-duplicate-key",
        ),
        pytest.param(
            CHAT,
            _chat([_user("hi"), _assistant_tool_calls([_tool_call({"function": {"arguments": "NaN"}})])]),
            "messages[1].tool_calls[0].function.arguments contains non-finite number: NaN",
            "messages",
            id="chat-arguments-non-finite",
        ),
        pytest.param(
            CHAT,
            _chat(
                [
                    {"role": "system", "content": "S"},
                    _user("hi"),
                    _assistant_tool_calls([_tool_call({"id": ""})]),
                ]
            ),
            "messages[2].tool_calls[0].id must be a non-empty string",
            "messages",
            id="chat-tool-call-id-empty",
        ),
        pytest.param(
            CHAT,
            _chat([_user("hi"), _assistant_tool_calls([_tool_call({"id": 123})])]),
            "messages[1].tool_calls[0].id must be a non-empty string",
            "messages",
            id="chat-tool-call-id-not-a-string",
        ),
        pytest.param(
            RESPONSES,
            _responses_input([{"type": "function_call", "name": "run", "arguments": "{}"}]),
            "input[0].call_id must be a non-empty string",
            "input",
            id="responses-call-id-missing",
        ),
        pytest.param(
            RESPONSES,
            _responses_input([{"type": "function_call", "call_id": "call-1", "arguments": "{}"}]),
            "input[0].name must be a non-empty string",
            "input",
            id="responses-function-name-missing",
        ),
        pytest.param(
            RESPONSES,
            _responses_input([{"type": "function_call", "call_id": "call-1", "name": "run", "arguments": {"a": 1}}]),
            "input[0].arguments must be a JSON string",
            "input",
            id="responses-arguments-not-a-string",
        ),
        pytest.param(
            ANTHROPIC,
            _anthropic(
                [_user("hi"), {"role": "assistant", "content": [{"type": "tool_use", "name": "run", "input": {}}]}]
            ),
            "messages[1].content[0].id must be a non-empty string",
            "messages",
            id="anthropic-tool-use-id-missing",
        ),
        pytest.param(
            ANTHROPIC,
            _anthropic([{"role": "assistant", "content": [{"type": "tool_use", "id": "call-1", "input": {}}]}]),
            "messages[0].content[0].name must be a non-empty string",
            "messages",
            id="anthropic-tool-use-name-missing",
        ),
        pytest.param(
            ANTHROPIC,
            _anthropic(
                [
                    {
                        "role": "assistant",
                        "content": [{"type": "tool_use", "id": "call-1", "name": "run", "input": "{}"}],
                    }
                ]
            ),
            "messages[0].content[0].input must be a JSON object",
            "messages",
            id="anthropic-tool-use-input-not-a-dict",
        ),
    ],
)
def test_canonical_tool_call_errors_report_field_paths(
    protocol: str,
    payload: dict[str, Any],
    message: str,
    param: str,
) -> None:
    assert_rejected(protocol, payload, message, param)


@pytest.mark.parametrize(
    ("protocol", "payload", "message", "param"),
    [
        pytest.param(
            CHAT,
            _chat(
                [
                    {"role": "system", "content": "S"},
                    _user("hi"),
                    {"role": "tool", "tool_call_id": "", "content": "ok"},
                ]
            ),
            "messages[2].tool_call_id must be a non-empty string",
            "messages",
            id="chat-tool-call-id-empty",
        ),
        pytest.param(
            CHAT,
            _chat([_user("hi"), {"role": "tool", "tool_call_id": 1, "content": "ok"}]),
            "messages[1].tool_call_id must be a non-empty string",
            "messages",
            id="chat-tool-call-id-not-a-string",
        ),
        pytest.param(
            RESPONSES,
            _responses_input([{"type": "function_call_output", "output": "ok"}]),
            "input[0].call_id must be a non-empty string",
            "input",
            id="responses-output-call-id-missing",
        ),
        pytest.param(
            RESPONSES,
            _responses_input(
                [{"type": "function_call_output", "call_id": "call-1", "output": [{"type": "input_text", "text": ""}]}]
            ),
            "input[0].output[0].text must be non-empty",
            "input",
            id="responses-output-text-empty",
        ),
        pytest.param(
            ANTHROPIC,
            _anthropic([_user("hi"), {"role": "user", "content": [{"type": "tool_result", "content": "ok"}]}]),
            "messages[1].content[0].tool_use_id must be a non-empty string",
            "messages",
            id="anthropic-tool-result-id-missing",
        ),
        pytest.param(
            ANTHROPIC,
            _anthropic([{"role": "user", "content": [{"type": "tool_result", "tool_use_id": "c1", "content": 123}]}]),
            "messages[0].content[0].content must be a string or text block list",
            "messages",
            id="anthropic-tool-result-content-type",
        ),
        pytest.param(
            ANTHROPIC,
            _anthropic(
                [
                    {
                        "role": "user",
                        "content": [
                            {"type": "tool_result", "tool_use_id": "c1", "content": [{"type": "text", "text": ""}]}
                        ],
                    }
                ]
            ),
            "messages[0].content[0].content[0].text must be non-empty",
            "messages",
            id="anthropic-tool-result-text-empty",
        ),
    ],
)
def test_canonical_tool_result_errors_report_field_paths(
    protocol: str,
    payload: dict[str, Any],
    message: str,
    param: str,
) -> None:
    assert_rejected(protocol, payload, message, param)


@pytest.mark.parametrize(
    ("protocol", "payload", "message", "param"),
    [
        pytest.param(
            CHAT,
            _chat([_user([{"type": "image_url"}])]),
            "messages[0].content[0].image_url must be a dict, got <class 'NoneType'>",
            "messages",
            id="chat-image-key-missing",
        ),
        pytest.param(
            CHAT,
            _chat([_user([{"type": "image_url", "image_url": "https://example.invalid/a.png"}])]),
            "messages[0].content[0].image_url must be a dict, got <class 'str'>",
            "messages",
            id="chat-image-object-flattened",
        ),
        pytest.param(
            CHAT,
            _chat([_user([{"type": "image_url", "image_url": {"url": ""}}])]),
            "messages[0].content[0].image_url.url must be a non-empty string",
            "messages",
            id="chat-image-url-empty",
        ),
        pytest.param(
            CHAT,
            _chat([_user([{"type": "image_url", "image_url": {"url": 1}}])]),
            "messages[0].content[0].image_url.url must be a non-empty string",
            "messages",
            id="chat-image-url-not-a-string",
        ),
        pytest.param(
            CHAT,
            _chat(
                [
                    {"role": "system", "content": "S"},
                    _user("hi"),
                    _user([{"type": "image_url", "image_url": {}}]),
                ]
            ),
            "messages[2].content[0].image_url.url must be a non-empty string",
            "messages",
            id="chat-image-url-nested-index-path",
        ),
        pytest.param(
            RESPONSES,
            _responses_input([{"role": "user", "content": [{"type": "input_image", "image_url": ""}]}]),
            "input[0].content[0].image_url must be a non-empty string",
            "input",
            id="responses-image-url-empty",
        ),
        pytest.param(
            RESPONSES,
            _responses_input([{"role": "user", "content": [{"type": "input_image", "image_url": {"url": "u"}}]}]),
            "input[0].content[0].image_url must be a non-empty string",
            "input",
            id="responses-image-url-nested-object",
        ),
        pytest.param(
            ANTHROPIC,
            _anthropic([_user([{"type": "image", "source": "x"}])]),
            "messages[0].content[0].source must be a JSON object",
            "messages",
            id="anthropic-source-not-a-dict",
        ),
        pytest.param(
            ANTHROPIC,
            _anthropic(
                [
                    _user(
                        [{"type": "image", "source": {"type": "base64", "media_type": "application/pdf", "data": "A"}}]
                    )
                ]
            ),
            "messages[0].content[0].source.media_type must be an image type",
            "messages",
            id="anthropic-media-type-not-an-image",
        ),
        pytest.param(
            ANTHROPIC,
            _anthropic(
                [_user([{"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": ""}}])]
            ),
            "messages[0].content[0].source.data must be non-empty",
            "messages",
            id="anthropic-base64-data-empty",
        ),
        pytest.param(
            ANTHROPIC,
            _anthropic([_user([{"type": "image", "source": {"type": "url", "url": ""}}])]),
            "messages[0].content[0].source.url must be non-empty",
            "messages",
            id="anthropic-url-source-empty",
        ),
        pytest.param(
            ANTHROPIC,
            _anthropic([_user([{"type": "image", "source": {"type": "file", "url": "u"}}])]),
            "messages[0].content[0].source.type is not supported",
            "messages",
            id="anthropic-source-type-unknown",
        ),
        pytest.param(
            ANTHROPIC,
            _anthropic(
                [
                    _user("hi"),
                    {"role": "assistant", "content": "ok"},
                    _user([{"type": "image", "source": {"type": "url", "url": ""}}]),
                ]
            ),
            "messages[2].content[0].source.url must be non-empty",
            "messages",
            id="anthropic-image-nested-index-path",
        ),
    ],
)
def test_illegal_image_input_is_rejected_by_every_protocol(
    protocol: str,
    payload: dict[str, Any],
    message: str,
    param: str,
) -> None:
    assert_rejected(protocol, payload, message, param)


def test_illegal_image_never_reaches_multimodal_extraction() -> None:
    # Flattened ``image_url`` used to pass ingress and only blow up with a
    # TypeError while building the training sample.
    assert_rejected(
        CHAT,
        _chat([_user([{"type": "image_url", "image_url": "https://example.invalid/a.png"}])]),
        "messages[0].content[0].image_url must be a dict, got <class 'str'>",
        "messages",
    )
    accepted = normalize(CHAT, _chat([_user([{"type": "image_url", "image_url": {"url": "https://x/a.png"}}])]))

    assert _multimodal_inputs_from_messages(accepted["messages"]) == {"images": ["https://x/a.png"]}


@pytest.mark.parametrize("protocol", (CHAT, RESPONSES, ANTHROPIC))
@pytest.mark.parametrize("case_id", canonical_case_ids())
def test_canonical_image_input_survives_multimodal_extraction(protocol: str, case_id: str) -> None:
    messages = canonical_of(normalize(protocol, protocol_payloads(protocol)[case_id]))["messages"]
    images = _multimodal_inputs_from_messages(messages)

    if "image" in case_id:
        assert images and all(isinstance(url, str) and url for url in images["images"])
    else:
        assert images is None


@pytest.mark.parametrize(
    ("protocol", "payload", "message", "param"),
    [
        pytest.param(
            CHAT,
            _chat([_user("hi")], tools="x"),
            "tools must be a list, got <class 'str'>",
            "messages",
            id="chat-tools-not-a-list",
        ),
        pytest.param(
            CHAT,
            _chat([_user("hi")], tools=["x"]),
            "tools[0] must be a dict, got <class 'str'>",
            "messages",
            id="chat-tool-not-a-dict",
        ),
        pytest.param(
            RESPONSES,
            _responses_input("hi", tools="x"),
            "tools must be a list",
            "tools",
            id="responses-tools-not-a-list",
        ),
        pytest.param(
            RESPONSES,
            _responses_input("hi", tools=[{"type": "function", "name": "", "parameters": {}}]),
            "tools.name must be a non-empty string",
            "tools",
            id="responses-tool-name-empty",
        ),
        pytest.param(
            RESPONSES,
            _responses_input("hi", tools=[{"type": "function", "name": "f", "parameters": []}]),
            "tools.parameters must be a JSON object",
            "tools",
            id="responses-tool-parameters-not-a-dict",
        ),
        pytest.param(
            ANTHROPIC,
            _anthropic([_user("hi")], tools=[{"name": "", "input_schema": {}}]),
            "tools.name must be a non-empty string",
            "tools",
            id="anthropic-tool-name-empty",
        ),
        pytest.param(
            ANTHROPIC,
            _anthropic([_user("hi")], tools=[{"name": "f", "input_schema": "x"}]),
            "tools.input_schema must be a JSON object",
            "tools",
            id="anthropic-tool-schema-not-a-dict",
        ),
    ],
)
def test_canonical_tools_errors_report_field_paths(
    protocol: str,
    payload: dict[str, Any],
    message: str,
    param: str,
) -> None:
    assert_rejected(protocol, payload, message, param)


@pytest.mark.parametrize(
    ("protocol", "payload", "message", "param"),
    [
        pytest.param(
            CHAT,
            _chat([_user("hi")], chat_template_kwargs="x"),
            "chat_template_kwargs must be a dict, got <class 'str'>",
            "messages",
            id="chat-template-kwargs-not-a-dict",
        ),
        pytest.param(
            CHAT,
            _chat([_user("hi")], chat_template_kwargs={"tools": {}}),
            "chat_template_kwargs cannot set reserved keys: tools",
            "chat_template_kwargs",
            id="chat-template-kwargs-reserved",
        ),
        pytest.param(
            CHAT,
            _chat([_user("hi")], chat_template_kwargs={"tokenize": True, "add_generation_prompt": False}),
            "chat_template_kwargs cannot set reserved keys: add_generation_prompt, tokenize",
            "chat_template_kwargs",
            id="chat-template-kwargs-reserved-sorted",
        ),
        pytest.param(
            CHAT,
            _chat([_user("hi")], max_tokens=0),
            "max_tokens must be a positive integer",
            "max_tokens",
            id="chat-max-tokens-zero",
        ),
        pytest.param(
            CHAT,
            _chat([_user("hi")], max_tokens="10"),
            "max_tokens must be a positive integer",
            "max_tokens",
            id="chat-max-tokens-string",
        ),
        pytest.param(
            CHAT,
            _chat([_user("hi")], max_tokens=True),
            "max_tokens must be a positive integer",
            "max_tokens",
            id="chat-max-tokens-bool",
        ),
        pytest.param(
            CHAT,
            _chat([_user("hi")], logprobs="true"),
            "logprobs must be a boolean",
            "logprobs",
            id="chat-logprobs-string",
        ),
        pytest.param(
            CHAT,
            _chat([_user("hi")], stop=["a", 1]),
            "stop must be a string or list of strings",
            "stop",
            id="chat-stop-item-not-a-string",
        ),
        pytest.param(
            RESPONSES,
            _responses_input("hi", max_output_tokens=0),
            "max_output_tokens must be a positive integer",
            "max_output_tokens",
            id="responses-max-output-tokens-zero",
        ),
        pytest.param(
            RESPONSES,
            _responses_input({"type": "message"}),
            "input is required and must be a string or list",
            "input",
            id="responses-input-type",
        ),
        pytest.param(
            RESPONSES,
            _responses_input("hi", instructions=""),
            "instructions must be a non-empty string",
            "instructions",
            id="responses-instructions-empty",
        ),
        pytest.param(
            ANTHROPIC,
            {"messages": [_user("hi")]},
            "max_tokens is required",
            "max_tokens",
            id="anthropic-max-tokens-missing",
        ),
        pytest.param(
            ANTHROPIC,
            _anthropic([_user("hi")], max_tokens="8"),
            "max_tokens must be a positive integer",
            "max_tokens",
            id="anthropic-max-tokens-string",
        ),
        pytest.param(
            ANTHROPIC,
            {"max_tokens": 8, "messages": "x"},
            "messages is required and must be a list",
            "messages",
            id="anthropic-messages-not-a-list",
        ),
        pytest.param(
            ANTHROPIC,
            _anthropic([_user("hi")], stop_sequences="x"),
            "stop_sequences must be a list of strings",
            "stop_sequences",
            id="anthropic-stop-sequences-string",
        ),
        pytest.param(
            ANTHROPIC,
            _anthropic([_user("hi")], system=""),
            "system must not be empty",
            "system",
            id="anthropic-system-empty",
        ),
        pytest.param(
            ANTHROPIC,
            _anthropic([_user("hi")], system=[{"type": "text", "text": ""}]),
            "system[0].text must be non-empty",
            "system",
            id="anthropic-system-text-empty",
        ),
    ],
)
def test_canonical_request_parameter_errors_report_field_paths(
    protocol: str,
    payload: dict[str, Any],
    message: str,
    param: str,
) -> None:
    assert_rejected(protocol, payload, message, param)


# --------------------------------------------------------------------------- #
# Known cross-protocol divergences.
#
# The illegal-image-block divergence was closed by validating the canonical
# block in ``check_messages``. The ones below stay open: tightening them changes
# what an agent process may report through ``runner/ipc.py`` as well, so they
# need a maintainer ruling. ``test_current_divergences_are_as_documented`` is
# the machine-readable list of today's behaviour; each ``*_should_*`` test
# asserts the uniform behaviour instead and stays a strict xfail until the
# production code agrees, so a fix surfaces as XPASS rather than silence.
# --------------------------------------------------------------------------- #

DIVERGENCE_CALL_WITHOUT_ID: dict[str, Any] = {
    "type": "function",
    "function": {"name": "run", "arguments": "{}"},
}
DIVERGENCE_TOOL_WITHOUT_NAME: dict[str, Any] = {"type": "function", "function": {}}


def test_current_divergences_are_as_documented() -> None:
    tool_call = canonical_of(
        normalize(CHAT, _chat([_user("hi"), _assistant_tool_calls([DIVERGENCE_CALL_WITHOUT_ID])]))
    )
    tool_result = canonical_of(normalize(CHAT, _chat([_user("hi"), {"role": "tool", "content": "ok"}])))
    tool = canonical_of(normalize(CHAT, _chat([_user("hi")], tools=[DIVERGENCE_TOOL_WITHOUT_NAME])))
    responses_tool = canonical_of(
        normalize(RESPONSES, _responses_input("hi", tools=[{"name": "f", "parameters": {}}]))
    )
    anthropic_tool = canonical_of(
        normalize(ANTHROPIC, _anthropic([_user("hi")], tools=[{"name": "f", "input_schema": {}}]))
    )
    chat_text_blocks = canonical_of(
        normalize(CHAT, _chat([_user([{"type": "text", "text": "a"}, {"type": "text", "text": "b"}])]))
    )
    responses_text_blocks = canonical_of(
        normalize(
            RESPONSES,
            _responses_input(
                [
                    {
                        "role": "user",
                        "content": [{"type": "input_text", "text": "a"}, {"type": "input_text", "text": "b"}],
                    }
                ]
            ),
        )
    )
    chat_unknown_block = canonical_of(
        normalize(CHAT, _chat([_user([{"type": "text", "text": "a"}, {"type": "document", "source": {}}])]))
    )
    responses_unknown_block = canonical_of(
        normalize(
            RESPONSES,
            _responses_input(
                [{"role": "user", "content": [{"type": "input_text", "text": "a"}, {"type": "document"}]}]
            ),
        )
    )

    # D1/D2: an absent ``id`` / ``tool_call_id`` survives into the canonical form.
    assert "id" not in tool_call["messages"][1]["tool_calls"][0]
    assert "tool_call_id" not in tool_result["messages"][1]
    # D3: Chat keeps an unnamed function tool as an explicit null pair.
    assert tool["tools"] == [{"type": "function", "function": {"name": None, "parameters": None}}]
    # D4: Responses drops a tool that omits ``type``; Anthropic keeps it.
    assert responses_tool["tools"] == []
    assert anthropic_tool["tools"] == [{"type": "function", "function": {"name": "f", "parameters": {}}}]
    # D5: only Chat keeps a multi text-block list; the others join it into a string.
    assert chat_text_blocks["messages"][0]["content"] == [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]
    assert responses_text_blocks["messages"][0]["content"] == "ab"
    # D6: unknown content block types are carried verbatim by Chat, dropped elsewhere.
    assert chat_unknown_block["messages"][0]["content"][1] == {"type": "document", "source": {}}
    assert responses_unknown_block["messages"][0]["content"] == "a"


@pytest.mark.xfail(strict=True, reason="D1: Chat accepts an assistant tool call without id")
def test_tool_call_without_id_should_be_rejected_by_every_protocol() -> None:
    with pytest.raises(AgenticChatRequestError):
        normalize(CHAT, _chat([_user("hi"), _assistant_tool_calls([DIVERGENCE_CALL_WITHOUT_ID])]))


@pytest.mark.xfail(strict=True, reason="D2: Chat accepts a tool result without tool_call_id")
def test_tool_result_without_call_id_should_be_rejected_by_every_protocol() -> None:
    with pytest.raises(AgenticChatRequestError):
        normalize(CHAT, _chat([_user("hi"), {"role": "tool", "content": "ok"}]))


@pytest.mark.xfail(strict=True, reason="D3: Chat accepts a function tool without name")
def test_function_tool_without_name_should_be_rejected_by_every_protocol() -> None:
    with pytest.raises(AgenticChatRequestError):
        normalize(CHAT, _chat([_user("hi")], tools=[DIVERGENCE_TOOL_WITHOUT_NAME]))


@pytest.mark.xfail(strict=True, reason="D5: Chat keeps a multi text-block list the others join")
def test_multiple_text_blocks_should_project_onto_one_canonical_content() -> None:
    chat = canonical_of(
        normalize(CHAT, _chat([_user([{"type": "text", "text": "a"}, {"type": "text", "text": "b"}])]))
    )
    responses = canonical_of(
        normalize(
            RESPONSES,
            _responses_input(
                [
                    {
                        "role": "user",
                        "content": [{"type": "input_text", "text": "a"}, {"type": "input_text", "text": "b"}],
                    }
                ]
            ),
        )
    )
    anthropic = canonical_of(
        normalize(ANTHROPIC, _anthropic([_user([{"type": "text", "text": "a"}, {"type": "text", "text": "b"}])]))
    )

    assert responses == chat
    assert anthropic == chat


@pytest.mark.xfail(strict=True, reason="D6: Chat carries unknown content block types verbatim")
def test_unknown_content_block_type_should_project_the_same_everywhere() -> None:
    chat = canonical_of(
        normalize(CHAT, _chat([_user([{"type": "text", "text": "a"}, {"type": "document", "source": {}}])]))
    )
    responses = canonical_of(
        normalize(
            RESPONSES,
            _responses_input(
                [{"role": "user", "content": [{"type": "input_text", "text": "a"}, {"type": "document"}]}]
            ),
        )
    )
    anthropic = canonical_of(
        normalize(ANTHROPIC, _anthropic([_user([{"type": "text", "text": "a"}, {"type": "document", "source": {}}])]))
    )

    assert responses == chat
    assert anthropic == chat
