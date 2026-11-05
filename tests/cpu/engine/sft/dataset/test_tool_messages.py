# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Flattened tool-call canonicalization contracts."""

import json

import pytest

from relax.engine.sft.dataset import streaming as streaming_module
from relax.engine.sft.dataset.streaming import (
    _canonicalize_messages,
)


def test_canonicalize_messages_flattened_tool_calls_preserve_order_and_masks():
    existing = {"type": "function", "function": {"name": "first", "arguments": {}}}
    raw = [
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": "checking", "tool_calls": [existing]},
        {"role": "tool_call", "content": '{"name":"second","arguments":{"q":"中文"}}'},
        {"role": "tool_call", "content": '{"name":"third"}', "learn": False},
        {"role": "tool_response", "content": "result"},
    ]
    original = json.dumps(raw)
    msgs = _canonicalize_messages(raw, require_response=True)
    assert [m.role for m in msgs] == ["user", "assistant", "assistant", "tool"]
    assert [m.learn for m in msgs] == [False, True, False, False]
    assert msgs[1].content == "checking"
    assert [c["function"]["name"] for c in msgs[1].tool_calls] == ["first", "second"]
    assert msgs[1].tool_calls[1]["function"]["arguments"] == {"q": "中文"}
    assert msgs[2].tool_calls[0]["function"]["name"] == "third"
    assert json.dumps(raw) == original


def test_streaming_flattened_tool_calls_with_multimodal_messages():
    row = {
        "messages": [
            {"role": "user", "content": "<image>look"},
            {"role": "tool_call", "content": '{"name":"search","arguments":{"q":"image"}}'},
            {"role": "tool", "content": "found"},
        ],
        "images": ["image.jpg"],
    }
    original = json.dumps(row)
    sample = streaming_module._build_canonical_sample_from_row(
        row,
        row_index=0,
        prompt_key="messages",
        label_key=None,
        multimodal_keys={"image": "images"},
        metadata_key="metadata",
        tool_key="tools",
        system_prompt=None,
        require_response=True,
        source_name="test",
    )
    assert sample.images == ["image.jpg"]
    assert [m.learn for m in sample.messages] == [False, True, False]
    assert sample.messages[1].content == ""
    assert sample.messages[1].tool_calls == [
        {"type": "function", "function": {"name": "search", "arguments": {"q": "image"}}}
    ]
    assert json.dumps(row) == original


@pytest.mark.parametrize("content", ["not json", "[]", "{}", '{"name":null}'])
def test_canonicalize_messages_rejects_invalid_flattened_tool_call(content):
    with pytest.raises(ValueError):
        _canonicalize_messages([{"role": "tool_call", "content": content}], require_response=True)
