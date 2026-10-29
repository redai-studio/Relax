# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Preserve tool-result images across Messages and Responses ingress."""

from __future__ import annotations

from typing import Any

import pytest

from relax.agentic.session.service import (
    AgenticChatRequestError,
    _normalized_anthropic_request,
    _normalized_responses_request,
)
from relax.agentic.session.state import _multimodal_inputs_from_messages


IMAGE_URL = "https://example.com/screenshot.png"
DATA_URL = (
    "data:image/png;base64,"
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
)


def _text(protocol: str, text: str) -> dict[str, Any]:
    return {"type": "text" if protocol == "anthropic" else "input_text", "text": text}


def _image(protocol: str, url: str) -> dict[str, Any]:
    if protocol == "responses":
        return {"type": "input_image", "image_url": url}
    if url.startswith("data:"):
        return {
            "type": "image",
            "source": {"type": "base64", "media_type": "image/png", "data": url.split(",", 1)[1]},
        }
    return {"type": "image", "source": {"type": "url", "url": url}}


def _normalize(protocol: str, outputs: list[Any]) -> dict[str, Any]:
    call_ids = [f"capture-{index}" for index in range(len(outputs))]
    if protocol == "responses":
        return _normalized_responses_request(
            {
                "input": [
                    {"role": "user", "content": "Inspect the screenshots."},
                    *[
                        {"type": "function_call", "call_id": call_id, "name": "capture", "arguments": "{}"}
                        for call_id in call_ids
                    ],
                    *[
                        {"type": "function_call_output", "call_id": call_id, "output": output}
                        for call_id, output in zip(call_ids, outputs)
                    ],
                ]
            }
        )
    return _normalized_anthropic_request(
        {
            "max_tokens": 16,
            "messages": [
                {"role": "user", "content": "Inspect the screenshots."},
                {
                    "role": "assistant",
                    "content": [
                        {"type": "tool_use", "id": call_id, "name": "capture", "input": {}} for call_id in call_ids
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "tool_result", "tool_use_id": call_id, "content": output}
                        for call_id, output in zip(call_ids, outputs)
                    ],
                },
            ],
        }
    )


@pytest.mark.parametrize("protocol", ["anthropic", "responses"])
@pytest.mark.parametrize("url", [IMAGE_URL, DATA_URL], ids=["url", "base64"])
@pytest.mark.parametrize("mixed", [False, True], ids=["image_only", "text_image_text"])
def test_tool_images_preserve_content_and_media(protocol: str, url: str, mixed: bool) -> None:
    content = [_image(protocol, url)]
    expected = [{"type": "image_url", "image_url": {"url": url}}]
    if mixed:
        content = [_text(protocol, "before"), *content, _text(protocol, "after")]
        expected = [{"type": "text", "text": "before"}, *expected, {"type": "text", "text": "after"}]
    normalized = _normalize(protocol, [content])
    assert normalized["messages"][-1] == {"role": "tool", "tool_call_id": "capture-0", "content": expected}
    assert _multimodal_inputs_from_messages(normalized["messages"]) == {"images": [url]}


@pytest.mark.parametrize("protocol", ["anthropic", "responses"])
@pytest.mark.parametrize("kind", ["string", "empty", "text_blocks"])
def test_tool_text_results_keep_existing_shape(protocol: str, kind: str) -> None:
    content = {
        "string": "firstsecond",
        "empty": "",
        "text_blocks": [_text(protocol, "first"), _text(protocol, "second")],
    }[kind]
    normalized = _normalize(protocol, [content])
    assert normalized["messages"][-1] == {
        "role": "tool",
        "tool_call_id": "capture-0",
        "content": "" if kind == "empty" else "firstsecond",
    }
    assert _multimodal_inputs_from_messages(normalized["messages"]) is None


@pytest.mark.parametrize("protocol", ["anthropic", "responses"])
def test_multiple_tool_results_keep_call_ids_and_image_order(protocol: str) -> None:
    normalized = _normalize(
        protocol, [[_image(protocol, DATA_URL), _image(protocol, IMAGE_URL)], [_image(protocol, DATA_URL)]]
    )
    assert [message["tool_call_id"] for message in normalized["messages"] if message["role"] == "tool"] == [
        "capture-0",
        "capture-1",
    ]
    assert _multimodal_inputs_from_messages(normalized["messages"]) == {"images": [DATA_URL, IMAGE_URL, DATA_URL]}


@pytest.mark.parametrize("protocol", ["anthropic", "responses"])
@pytest.mark.parametrize("invalid", ["image", "text"])
def test_invalid_tool_image_or_text_is_rejected(protocol: str, invalid: str) -> None:
    content = [
        _text(protocol, "" if invalid == "text" else "caption"),
        _image(protocol, "" if invalid == "image" else IMAGE_URL),
    ]
    with pytest.raises(AgenticChatRequestError):
        _normalize(protocol, [content])
