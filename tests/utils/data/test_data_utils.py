# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Unit tests for shared data message normalization."""

import pytest

from relax.utils.data.data_utils import build_messages, collect_message_multimodal_data


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
