#!/usr/bin/env python
# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Convert a chat-prompt RL dataset into the row schema Relax SFT expects.

Why this exists
---------------
``scripts/training/sft/run-qwen3-0.6B-4xgpu-dp4-observer.sh`` needs a tiny,
reproducible SFT set for a *control-variable* timing measurement: four
data-parallel ranks must see equivalent work, so a small fixed dataset is
preferable to the 17k-row source.  The Relax SFT loader
(``relax/engine/sft/dataset/streaming.py``) supports two shapes:

* ``--label-key`` set  -> ``--input-key`` MUST be a plain prompt **string**, and
  the label string is appended as the final ``assistant`` message with
  ``learn=True`` (see ``_build_canonical_sample_from_row``).
* ``--label-key`` unset -> ``--input-key`` must be an OpenAI messages list whose
  assistant turn is marked learnable.

``dapo-math-17k.jsonl`` stores ``prompt`` as a one-turn user message list and
``label`` as the reference answer string, so the string/label shape is the
cheapest conversion and matches the 8-GPU recipe's
``--input-key problem --label-key generated_solution``.

Output format
-------------
JSON Lines.  ``StreamingReader`` accepts ``.jsonl`` and ``.parquet``; JSONL is
chosen because it needs no pyarrow and no extra conversion step.

Usage
-----
    python tools/straggler/make_sft_dataset.py \
        --output scripts/training/sft/data/dapo-math-17k-sft-256.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any


DEFAULT_INPUT = (
    "/root/autodl-tmp/hf-cache/hub/datasets--zhuzilin--dapo-math-17k/"
    "snapshots/2e65612930298bde4c5d58fd97b3f23a483aaff9/dapo-math-17k.jsonl"
)
DEFAULT_OUTPUT = "scripts/training/sft/data/dapo-math-17k-sft-256.jsonl"


def prompt_to_string(prompt: Any) -> str:
    """Flatten a chat message list (or a bare string) into one prompt string.

    Only non-assistant turns are kept: the assistant turn is contributed by
    ``--label-key`` in the SFT loader, so any assistant text present in the
    source prompt is dropped rather than duplicated into the loss mask.
    """
    if isinstance(prompt, str):
        return prompt
    if not isinstance(prompt, list):
        raise TypeError(f"unsupported prompt type: {type(prompt)!r}")
    parts: list[str] = []
    for message in prompt:
        if not isinstance(message, dict):
            raise TypeError(f"unsupported message type: {type(message)!r}")
        if message.get("role") == "assistant":
            continue
        content = message.get("content")
        if isinstance(content, str) and content:
            parts.append(content)
    if not parts:
        raise ValueError("prompt produced no non-assistant text")
    return "\n\n".join(parts)


def convert_row(row: dict[str, Any], *, prompt_key: str, label_key: str, out_keys: tuple[str, str]) -> dict[str, str]:
    if prompt_key not in row:
        raise KeyError(f"row missing prompt key {prompt_key!r}: keys={sorted(row)}")
    if label_key not in row:
        raise KeyError(f"row missing label key {label_key!r}: keys={sorted(row)}")
    label = row[label_key]
    if not isinstance(label, str) or not label.strip():
        raise ValueError(f"row has empty/non-string label under {label_key!r}: {label!r}")
    problem, solution = out_keys
    return {problem: prompt_to_string(row[prompt_key]), solution: label}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", default=DEFAULT_INPUT, help="source jsonl (dapo-math-17k.jsonl)")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="destination .jsonl")
    parser.add_argument("--num-rows", type=int, default=256, help="rows to sample deterministically")
    parser.add_argument("--seed", type=int, default=42, help="sampling seed")
    parser.add_argument("--prompt-key", default="prompt", help="source prompt column")
    parser.add_argument("--label-key", default="label", help="source label column")
    parser.add_argument("--out-prompt-key", default="problem", help="output prompt column (--input-key)")
    parser.add_argument("--out-label-key", default="generated_solution", help="output label column (--label-key)")
    args = parser.parse_args()

    src = Path(args.input)
    if not src.exists():
        raise SystemExit(f"source dataset not found: {src}")

    with src.open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if not rows:
        raise SystemExit(f"source dataset is empty: {src}")
    if args.num_rows > len(rows):
        raise SystemExit(f"--num-rows {args.num_rows} exceeds source rows {len(rows)}")

    rng = random.Random(args.seed)
    indices = sorted(rng.sample(range(len(rows)), args.num_rows))

    out_keys = (args.out_prompt_key, args.out_label_key)
    converted = [
        convert_row(rows[i], prompt_key=args.prompt_key, label_key=args.label_key, out_keys=out_keys) for i in indices
    ]

    dest = Path(args.output)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("w", encoding="utf-8") as handle:
        for record in converted:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    prompt_key, label_key = out_keys
    lengths = [len(r[prompt_key]) for r in converted]
    print(f"source rows      : {len(rows)}")
    print(f"selected rows    : {len(converted)} (seed={args.seed}, sorted indices {indices[:5]}...)")
    print(f"output           : {dest} ({dest.stat().st_size} bytes)")
    print(f"row keys         : {list(converted[0])}")
    print(f"prompt chars     : min={min(lengths)} max={max(lengths)} mean={sum(lengths) // len(lengths)}")
    print(f"first row prompt : {converted[0][prompt_key][:160]!r}")
    print(f"first row label  : {converted[0][label_key]!r}")


if __name__ == "__main__":
    main()
