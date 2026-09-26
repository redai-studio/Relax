# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Convert flattened tool-call jsonl to Qwen3.5-compatible chat format.

Input format (per line):
    {
      "tools": "<json-string of tool definitions>",
      "messages": [
        {"role": "system" | "user" | "assistant" | "tool_call" | "tool" | "tool_response", "content": str},
        ...
      ],
      ...other meta fields preserved as-is
    }

Output format (per line):
    {
      "tools": [...],   # parsed list
      "messages": [
        {"role": "system" | "user" | "assistant" | "tool", "content": str, "tool_calls"?: [...]},
        ...
      ],
      ...meta fields preserved
    }

Rules:
  * Each ``tool_call`` message is parsed (``content`` is a JSON string of
    ``{"name", "arguments"}``) and appended to the ``tool_calls`` list of
    the most recently emitted ``assistant`` message. If no assistant exists
    yet, an empty assistant turn is inserted to hold the call.
  * ``tool`` passes through; ``tool_response`` is renamed to ``tool``; content kept as-is.
  * ``system`` / ``user`` / ``assistant`` pass through unchanged.

Usage:
    python scripts/tools/process_tool_chat.py \\
        --input  test_data.jsonl \\
        --output test_data_converted.jsonl
"""

import argparse
import json
from typing import Any


def _parse_tool_call(content: str) -> dict[str, Any]:
    obj = json.loads(content)
    name = obj["name"]
    arguments = obj.get("arguments", {})
    return {"type": "function", "function": {"name": name, "arguments": arguments}}


def convert_messages(raw_messages: list[dict]) -> list[dict]:
    out: list[dict] = []
    for msg in raw_messages:
        role = msg["role"]
        content = msg["content"]
        if role in ("system", "user", "assistant"):
            out.append({"role": role, "content": content})
        elif role in ("tool", "tool_response"):
            out.append({"role": "tool", "content": content})
        elif role == "tool_call":
            call = _parse_tool_call(content)
            if not out or out[-1]["role"] != "assistant":
                out.append({"role": "assistant", "content": "", "tool_calls": []})
            out[-1].setdefault("tool_calls", []).append(call)
        else:
            raise ValueError(f"unknown role: {role!r}")
    return out


def convert_record(record: dict) -> dict:
    tools_raw = record.get("tools")
    if isinstance(tools_raw, str):
        tools = json.loads(tools_raw)
    else:
        tools = tools_raw
    new = dict(record)
    new["tools"] = tools
    new["messages"] = convert_messages(record["messages"])
    return new


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, help="path to source jsonl")
    ap.add_argument("--output", required=True, help="path to write converted jsonl")
    args = ap.parse_args()

    n_in = n_out = 0
    with open(args.input) as src, open(args.output, "w") as dst:
        for line in src:
            line = line.strip()
            if not line:
                continue
            n_in += 1
            record = json.loads(line)
            converted = convert_record(record)
            dst.write(json.dumps(converted, ensure_ascii=False) + "\n")
            n_out += 1
    print(f"wrote {n_out}/{n_in} records to {args.output}")


if __name__ == "__main__":
    main()
