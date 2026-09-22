# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Unit tests for ``scripts/tools/process_tool_chat.py``.

The converter takes the flattened tool-chat format (separate ``tool_call`` /
``tool_response`` role messages, ``tools`` as a JSON-string) and emits OpenAI /
Qwen3.5-compatible messages (assistant.tool_calls list, ``tool`` role,
``tools`` as a parsed list).
"""

import importlib.util
import json
from pathlib import Path

import pytest


_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT_PATH = _REPO_ROOT / "scripts" / "tools" / "process_tool_chat.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("process_tool_chat", _SCRIPT_PATH)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


process_tool_chat = _load_module()


# A small record matching the user's actual data format: ``tools`` as a JSON
# string, separate-role ``tool_call`` / ``tool_response`` messages, assistant
# turns with inline ``<think>`` blocks, and an arbitrary meta field
# (``case_id``) that must be preserved through the conversion.
SAMPLE_RAW_RECORD = {
    "tools": json.dumps(
        [
            {
                "type": "function",
                "function": {
                    "name": "activate_skill",
                    "description": "Activate a skill by name.",
                    "parameters": {
                        "type": "object",
                        "properties": {"skill_name": {"type": "string"}},
                        "required": ["skill_name"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "search_request",
                    "description": "Search.",
                    "parameters": {
                        "type": "object",
                        "properties": {"queries": {"type": "array", "items": {"type": "string"}}},
                        "required": ["queries"],
                    },
                },
            },
        ]
    ),
    "messages": [
        {"role": "system", "content": "你是点点。"},
        {"role": "user", "content": "想出去玩"},
        {"role": "assistant", "content": "<think>需要先激活 skill</think>\n\n"},
        {
            "role": "tool_call",
            "content": json.dumps({"name": "activate_skill", "arguments": {"skill_name": "travel"}}),
        },
        {"role": "tool_response", "content": "已加载 travel Skill"},
        {"role": "user", "content": "去厦门 4 天"},
        {"role": "assistant", "content": "<think>同时搜两个</think>\n\n"},
        {
            "role": "tool_call",
            "content": json.dumps({"name": "search_request", "arguments": {"queries": ["厦门 4 天"]}}),
        },
        {
            "role": "tool_call",
            "content": json.dumps({"name": "activate_skill", "arguments": {"skill_name": "route"}}),
        },
        {"role": "tool_response", "content": "[搜索结果...]"},
        {"role": "tool_response", "content": "已加载 route Skill"},
        {"role": "assistant", "content": "推荐你这样玩..."},
    ],
    "case_id": "demo_0001",
    "num_turns_expected": 2,
    "num_turns_got": 2,
    "all_turns_passed": True,
}


def test_convert_record_parses_tools_string_to_list():
    out = process_tool_chat.convert_record(SAMPLE_RAW_RECORD)
    assert isinstance(out["tools"], list)
    assert len(out["tools"]) == 2
    assert out["tools"][0]["function"]["name"] == "activate_skill"
    assert out["tools"][1]["function"]["name"] == "search_request"


def test_convert_record_merges_tool_call_into_preceding_assistant():
    out = process_tool_chat.convert_record(SAMPLE_RAW_RECORD)
    msgs = out["messages"]
    # 1st assistant turn has exactly one tool_call attached.
    asst1 = msgs[2]
    assert asst1["role"] == "assistant"
    assert asst1["content"] == "<think>需要先激活 skill</think>\n\n"
    assert len(asst1["tool_calls"]) == 1
    assert asst1["tool_calls"][0] == {
        "type": "function",
        "function": {"name": "activate_skill", "arguments": {"skill_name": "travel"}},
    }


def test_convert_record_attaches_multiple_consecutive_tool_calls_to_same_assistant():
    out = process_tool_chat.convert_record(SAMPLE_RAW_RECORD)
    msgs = out["messages"]
    # 2nd assistant (after the user "去厦门 4 天") has TWO tool_calls.
    asst2 = next(m for m in msgs if m.get("content", "").startswith("<think>同时搜两个"))
    assert len(asst2["tool_calls"]) == 2
    names = [tc["function"]["name"] for tc in asst2["tool_calls"]]
    assert names == ["search_request", "activate_skill"]


def test_convert_record_renames_tool_response_to_tool():
    out = process_tool_chat.convert_record(SAMPLE_RAW_RECORD)
    msgs = out["messages"]
    roles = [m["role"] for m in msgs]
    assert "tool_response" not in roles
    assert "tool_call" not in roles
    # All original tool_response messages survive as role="tool".
    assert roles.count("tool") == 3


def test_convert_messages_preserves_tool_role():
    msgs = process_tool_chat.convert_messages([{"role": "tool", "content": "result"}])
    assert msgs == [{"role": "tool", "content": "result"}]


def test_convert_record_preserves_other_meta_fields():
    out = process_tool_chat.convert_record(SAMPLE_RAW_RECORD)
    assert out["case_id"] == "demo_0001"
    assert out["num_turns_expected"] == 2
    assert out["num_turns_got"] == 2
    assert out["all_turns_passed"] is True


def test_convert_record_does_not_mutate_input():
    snapshot = json.loads(json.dumps(SAMPLE_RAW_RECORD))
    process_tool_chat.convert_record(SAMPLE_RAW_RECORD)
    assert SAMPLE_RAW_RECORD == snapshot


def test_convert_messages_inserts_empty_assistant_when_tool_call_has_no_predecessor():
    # tool_call appears before any assistant — converter must insert a holder.
    msgs = process_tool_chat.convert_messages(
        [
            {"role": "user", "content": "q"},
            {"role": "tool_call", "content": json.dumps({"name": "f", "arguments": {}})},
        ]
    )
    assert msgs == [
        {"role": "user", "content": "q"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"type": "function", "function": {"name": "f", "arguments": {}}}],
        },
    ]


def test_convert_messages_rejects_unknown_role():
    with pytest.raises(ValueError, match="unknown role"):
        process_tool_chat.convert_messages([{"role": "narrator", "content": "x"}])


def test_converted_output_is_compatible_with_canonicalize_messages():
    """End-to-end: converted record flows cleanly through the SFT canonicalizer
    (which uses the OpenAI-style schema with assistant.tool_calls + role='tool')."""
    from relax.engine.sft.dataset.streaming import _canonicalize_messages, _normalize_tools

    out = process_tool_chat.convert_record(SAMPLE_RAW_RECORD)
    msgs = _canonicalize_messages(out["messages"], require_response=True)
    tools = _normalize_tools(out["tools"])

    # Roles must all be in VALID_ROLES; CanonicalMessage.__post_init__ would
    # have raised otherwise.
    assert {m.role for m in msgs} <= {"system", "user", "assistant", "tool"}
    # tool_calls preserved on the assistant turns that originally carried them.
    assistants_with_calls = [m for m in msgs if m.role == "assistant" and m.tool_calls]
    assert len(assistants_with_calls) == 2
    assert isinstance(tools, list) and tools[0]["function"]["name"] == "activate_skill"


def test_cli_round_trip_writes_jsonl(tmp_path):
    src = tmp_path / "in.jsonl"
    dst = tmp_path / "out.jsonl"
    with src.open("w") as f:
        f.write(json.dumps(SAMPLE_RAW_RECORD, ensure_ascii=False) + "\n")
        f.write(json.dumps(SAMPLE_RAW_RECORD, ensure_ascii=False) + "\n")

    import sys

    argv_backup = sys.argv
    sys.argv = ["process_tool_chat", "--input", str(src), "--output", str(dst)]
    try:
        process_tool_chat.main()
    finally:
        sys.argv = argv_backup

    lines = dst.read_text().splitlines()
    assert len(lines) == 2
    rec = json.loads(lines[0])
    assert isinstance(rec["tools"], list)
    assert all(m["role"] in {"system", "user", "assistant", "tool"} for m in rec["messages"])
