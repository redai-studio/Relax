# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Shared, dependency-light contracts for the OpenSearch-VL comparison."""

import hashlib
import json
import re
from pathlib import Path
from typing import Any


DATA_REVISION = "2c1c460af4fa15bd63210cbf426a96664b959944"
REFERENCE_REVISION = "1ac26e2c35a3e813e022813c88888db457d88f7d"
MODEL_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
SOURCE_FILES = {
    "fvqa": "fvqa_llama_factory_clean.json",
    "livevqa": "livevqa_llama_factory_filtered.json",
    "palace": "palace_llama_factory_filtered.json",
    "webqa": "webqa_llama_factory_filtered.json",
    "wiki_art": "wikiart_llama_factory_filtered.json",
    "wiki_en": "wiki_en_llama_factory_filtered.json",
    "wiki_zh": "wiki_zh_llama_factory_filtered.json",
}


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))


def parse_tools(value: str | list) -> list[dict]:
    # Some releases double-encode the tools column.
    for _ in range(2):
        if isinstance(value, str):
            value = json.loads(value)
    if not isinstance(value, list) or any(not isinstance(tool, dict) for tool in value):
        raise ValueError("tools must decode to a list of schemas")
    return value


def extract_calls(text: str) -> list[dict]:
    calls = []
    for block in re.findall(r"<tool_call>\s*(.*?)\s*</tool_call>", text, re.DOTALL):
        block = re.sub(r"^(?:\\n|\\r|\\t)+|(?:\\n|\\r|\\t)+$", "", block.strip())
        call = json.loads(block)
        if not isinstance(call, dict):
            raise ValueError("Tool call must be an object")
        args = call.get("arguments", call.get("parameters", {}))
        if isinstance(args, str):
            args = json.loads(args)
        if not isinstance(call.get("name"), str) or not isinstance(args, dict):
            raise ValueError("Invalid tool call")
        calls.append({"name": call["name"], "arguments": args})
    if text.count("<tool_call>") != len(calls) or text.count("</tool_call>") != len(calls):
        raise ValueError("Unclosed or ambiguous tool call")
    return calls


def extract_answer(text: str) -> str | None:
    answers = re.findall(r"<response>\s*(.*?)\s*</response>", text, re.DOTALL)
    return answers[-1].strip() if len(answers) == 1 and answers[-1].strip() else None


def normalized_text(text: str) -> str:
    return " ".join(text.casefold().split())


def chat_messages(messages: list[dict]) -> list[dict]:
    """Replace literal image placeholders with Qwen image tokens."""
    return [
        {"role": m["role"], "content": m["content"].replace("<image>", "<|vision_start|><|image_pad|><|vision_end|>")}
        for m in messages
    ]
