# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Tag-extraction unit tests for DeepEyesV2 env adapter.

Pure functions, no sandbox / no network. Smoke tests for the rest live in the
launch script + smoke dataset (Task 6).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


EXAMPLE_DIR = Path(__file__).resolve().parents[3] / "examples" / "deepeyes_v2_agentic"
sys.path.insert(0, str(EXAMPLE_DIR))

from app.env_deepeyes_v2 import extract_answer, extract_code, extract_tool_call  # noqa: E402


def test_extract_answer_basic():
    assert extract_answer("blah <answer>42</answer> end") == "42"


def test_extract_answer_missing_returns_none():
    assert extract_answer("no tag here") is None


def test_extract_code_python_block():
    text = "<code>\n```python\nprint(1)\n```\n</code>"
    assert extract_code(text) == "print(1)"


def test_extract_code_no_python_fence_returns_none():
    assert extract_code("<code>not python</code>") is None


def test_extract_tool_call_search():
    text = '<tool_call>{"name": "search", "arguments": {"query": "x"}}</tool_call>'
    parsed = extract_tool_call(text)
    assert parsed == {"name": "search", "arguments": {"query": "x"}}


def test_extract_tool_call_image_search_no_args():
    parsed = extract_tool_call('<tool_call>{"name": "image_search"}</tool_call>')
    assert parsed == {"name": "image_search", "arguments": None}


def test_extract_tool_call_invalid_json_returns_none():
    assert extract_tool_call("<tool_call>not json</tool_call>") is None


def test_encode_image_data_uri_returns_png_data_url():
    from app.env_deepeyes_v2 import encode_image_data_uri
    from PIL import Image

    uri = encode_image_data_uri(Image.new("RGB", (4, 4)))
    assert uri.startswith("data:image/png;base64,")


def test_extract_tool_call_preserves_open_tag_in_json():
    text = '<tool_call>{"name":"search","arguments":{"query":"explain <tool_call> tags"}}</tool_call>'
    assert extract_tool_call(text)["arguments"]["query"] == "explain <tool_call> tags"
    assert extract_tool_call(text + "<tool_call>{") == extract_tool_call(text)
    code = "print('<tool_call>')"
    text = "<tool_call>" + json.dumps({"name": "python_exec", "arguments": {"code": code}}) + "</tool_call>"
    assert extract_code(text) == code
