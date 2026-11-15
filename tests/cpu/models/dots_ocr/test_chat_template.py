# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from pathlib import Path

import pytest
from transformers.utils.chat_template_utils import render_jinja_template


CHAT_TEMPLATE_PATH = Path(__file__).parents[4] / "relax/models/dots_ocr/chat_template.jinja"
CHAT_TEMPLATE_JSON_PATH = Path(__file__).parents[4] / "relax/models/dots_ocr/chat_template.json"
IMAGE_TOKEN = "<|img|><|imgpad|><|endofimg|>"


def render_chat(messages, **kwargs) -> str:
    result = render_jinja_template(
        [messages],
        chat_template=CHAT_TEMPLATE_PATH.read_text(),
        **kwargs,
    )
    if isinstance(result, tuple):
        result = result[0]
    return result[0]


def test_dots_ocr_chat_template_formats_text_only_user_prompt():
    messages = [{"role": "user", "content": "hello"}]

    rendered = render_chat(messages, add_generation_prompt=True)

    assert rendered == "<|user|>hello<|endofuser|><|assistant|>"


def test_dots_ocr_chat_template_has_no_json_sidecar():
    assert not CHAT_TEMPLATE_JSON_PATH.exists()


def test_dots_ocr_chat_template_formats_system_and_multimodal_user_prompt():
    messages = [
        {"role": "system", "content": "sys"},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": "image-1.png"},
                {"type": "text", "text": "OCR this"},
            ],
        },
    ]

    rendered = render_chat(messages, add_generation_prompt=True)

    assert rendered == f"<|system|>sys<|endofsystem|>\n<|user|>{IMAGE_TOKEN}OCR this<|endofuser|><|assistant|>"


def test_dots_ocr_chat_template_supports_text_items_in_content_lists():
    messages = [
        {"role": "system", "content": [{"type": "text", "text": "sys"}]},
        {"role": "user", "content": ["prefix ", {"type": "text", "text": "body"}]},
    ]

    rendered = render_chat(messages, add_generation_prompt=True)

    assert rendered == "<|system|>sys<|endofsystem|>\n<|user|>prefix body<|endofuser|><|assistant|>"


def test_dots_ocr_chat_template_supports_image_url_and_vision_ids():
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": "https://example.test/a.png"}},
                {"type": "image", "image": "image-2.png"},
                {"type": "text", "text": "compare"},
            ],
        }
    ]

    rendered = render_chat(messages, add_generation_prompt=True, add_vision_id=True)

    assert rendered == f"<|user|>Picture 1: {IMAGE_TOKEN}Picture 2: {IMAGE_TOKEN}compare<|endofuser|><|assistant|>"


def test_dots_ocr_chat_template_closes_only_intermediate_assistant_messages():
    messages = [
        {"role": "user", "content": "u1"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "u2"},
        {"role": "assistant", "content": "a2"},
    ]

    rendered = render_chat(messages, add_generation_prompt=True)

    assert rendered == "<|user|>u1<|endofuser|><|assistant|>a1<|endofassistant|><|user|>u2<|endofuser|><|assistant|>a2"


def test_dots_ocr_chat_template_renders_tool_calls_and_responses():
    messages = [
        {"role": "user", "content": "lookup"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [{"function": {"name": "search", "arguments": {"query": "dots", "top_k": 3}}}],
        },
        {"role": "tool", "content": "result"},
        {"role": "user", "content": "answer now"},
    ]
    tools = [{"type": "function", "function": {"name": "search", "parameters": {"type": "object"}}}]

    rendered = render_chat(messages, tools=tools, add_generation_prompt=True)

    assert rendered.startswith("<|system|># Tools\n\nYou have access to the following functions:")
    assert "<|endofsystem|>\n<|user|>lookup<|endofuser|>" in rendered
    assert "<|assistant|><tool_call>\n<function=search>\n<parameter=query>\ndots\n</parameter>" in rendered
    assert "<parameter=top_k>\n3\n</parameter>\n</function>\n</tool_call><|endofassistant|>" in rendered
    assert "<|user|><tool_response>\nresult\n</tool_response><|endofuser|>" in rendered
    assert rendered.endswith("<|user|>answer now<|endofuser|><|assistant|>")


def test_dots_ocr_chat_template_groups_consecutive_tool_responses():
    messages = [
        {"role": "user", "content": "lookup"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [{"function": {"name": "search", "arguments": {"query": "dots"}}}],
        },
        {"role": "tool", "content": "result 1"},
        {"role": "tool", "content": "result 2"},
        {"role": "user", "content": "answer now"},
    ]

    rendered = render_chat(messages, add_generation_prompt=True)

    assert (
        "<|user|><tool_response>\nresult 1\n</tool_response><tool_response>\nresult 2\n</tool_response><|endofuser|>"
        in rendered
    )
    assert rendered.endswith("<|user|>answer now<|endofuser|><|assistant|>")


@pytest.mark.parametrize(
    ("messages", "error"),
    [
        ([], "No messages provided."),
        (
            [
                {"role": "user", "content": "u"},
                {"role": "system", "content": "late"},
            ],
            "System message must be at the beginning.",
        ),
        (
            [
                {"role": "system", "content": [{"type": "image", "image": "image.png"}]},
                {"role": "user", "content": "u"},
            ],
            "System message cannot contain images.",
        ),
        (
            [{"role": "user", "content": [{"type": "video", "video": "video.mp4"}]}],
            "DotsOCR chat template does not support video content.",
        ),
        ([{"role": "developer", "content": "u"}], "Unexpected message role."),
        ([{"role": "user", "content": {"type": "text", "text": "u"}}], "Unexpected content type."),
        ([{"role": "assistant", "content": "a"}], "No user query found in messages."),
    ],
)
def test_dots_ocr_chat_template_rejects_invalid_inputs(messages, error):
    with pytest.raises(Exception, match=error):
        render_chat(messages)
