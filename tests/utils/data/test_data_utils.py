# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Unit tests for shared data message normalization."""

import pytest

from relax.utils.data.data_utils import build_messages, collect_message_multimodal_data


@pytest.mark.parametrize("structured", [False, True])
def test_build_messages_system_prompt_matches_content_format(structured: bool) -> None:
    content = [{"type": "text", "text": "Question"}] if structured else "Question"
    row = {"messages": [{"role": "user", "content": content}]}
    messages = build_messages(row, "messages", "Answer concisely.", True)
    expected = [{"type": "text", "text": "Answer concisely."}] if structured else "Answer concisely."
    assert messages == [{"role": "system", "content": expected}, {"role": "user", "content": content}]
    assert row["messages"] == [{"role": "user", "content": content}]


def test_build_messages_system_prompt_preserves_tool_call_without_content() -> None:
    message = {"role": "assistant", "tool_calls": [{"type": "function", "function": {"name": "search"}}]}
    messages = build_messages({"messages": [message]}, "messages", "Answer concisely.", True)
    assert messages == [{"role": "system", "content": "Answer concisely."}, message]


@pytest.mark.parametrize(
    "image_url",
    [
        "https://example.test/image.png",
        {"url": "https://example.test/image.png"},
    ],
)
def test_build_messages_normalizes_inline_image_url(image_url):
    row = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe the image."},
                    {"type": "image_url", "image_url": image_url},
                ],
            }
        ]
    }

    messages = build_messages(
        row,
        prompt_key="messages",
        system_prompt=None,
        as_conversation=True,
        multimodal_keys=None,
    )

    assert messages[0]["content"][1] == {
        "type": "image",
        "image": "https://example.test/image.png",
    }
    assert collect_message_multimodal_data(messages)["image"] == ["https://example.test/image.png"]


def test_build_messages_uses_top_level_media_without_mutating_input():
    row = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {}},
                    {"type": "text", "text": "Describe the image."},
                ],
            }
        ],
        "images": ["/data/image.png"],
    }

    first_messages = build_messages(
        row,
        prompt_key="messages",
        system_prompt=None,
        as_conversation=True,
        multimodal_keys={"image": "images"},
    )
    second_messages = build_messages(
        row,
        prompt_key="messages",
        system_prompt=None,
        as_conversation=True,
        multimodal_keys={"image": "images"},
    )

    assert row["images"] == ["/data/image.png"]
    assert first_messages == second_messages
    assert first_messages[0]["content"][0] == {"type": "image", "image": "/data/image.png"}
    assert collect_message_multimodal_data(first_messages)["image"] == ["/data/image.png"]
