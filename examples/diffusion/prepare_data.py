#!/usr/bin/env python
# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Convert public datasets to the unified native-generation JSONL (design.

§5.1).

Emits one JSON object per line::

    {"prompt": ..., "images" / "videos": [...], "metadata": {"task": ..., "sample_id": ...}}

Each source dataset has a small converter; add new ones to ``CONVERTERS``. This
is a thin entry point — it only reshapes records, it does not download models.
The record-normalization helpers are pure and unit-tested.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, Iterable, List, Optional


__all__ = ["normalize_record", "write_jsonl", "CONVERTERS", "TASKS"]

# Text-to-image is the only generation task supported on this branch; the i2i /
# t2v / i2v / v2v / t2av adapters live on `backup/diffusion-generative-rl-full`.
TASKS = ("t2i",)


def normalize_record(
    prompt: str,
    task: str,
    sample_id: str,
    *,
    images: Optional[List[str]] = None,
    videos: Optional[List[str]] = None,
    extra_metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build one unified JSONL record; rejects unsupported tasks."""
    if task not in TASKS:
        raise ValueError(f"metadata.task must be one of {TASKS}, got {task!r}.")
    record: Dict[str, Any] = {"prompt": prompt}
    if images:
        record["images"] = list(images)
    if videos:
        record["videos"] = list(videos)
    metadata = {"task": task, "sample_id": sample_id}
    if extra_metadata:
        metadata.update(extra_metadata)
    record["metadata"] = metadata
    return record


def write_jsonl(records: Iterable[Dict[str, Any]], path: str) -> int:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1
    return n


# --- per-dataset converters (prompt/media extraction only) ------------------


def _from_pickapic(src: str, task: str) -> Iterable[Dict[str, Any]]:
    """Pick-a-Pic unique prompts.

    Accepts the HF parquet layout (a `prompt` column) or a JSONL/txt fallback;
    empty prompts are skipped.
    """
    n = 0
    for prompt in _read_prompt_column(src, columns=("prompt", "caption")):
        prompt = prompt.strip()
        if not prompt:
            continue
        yield normalize_record(prompt, task, f"pickapic_{n:07d}")
        n += 1


def _from_prompts(src: str, task: str) -> Iterable[Dict[str, Any]]:
    """Generic prompt source: a parquet with a `prompt`/`caption` column, a
    JSONL with a `prompt` field, or a one-prompt-per-line text file (COCO
    caption list, or any hand-curated prompt file)."""
    n = 0
    for prompt in _read_prompt_column(src, columns=("prompt", "caption")):
        prompt = prompt.strip()
        if not prompt:
            continue
        yield normalize_record(prompt, task, f"{os.path.basename(src.rstrip('/'))}_{n:07d}")
        n += 1


CONVERTERS = {
    "pickapic": _from_pickapic,
    "prompts": _from_prompts,
}


def _read_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def _iter_parquet_files(path: str) -> List[str]:
    """Return parquet files under a dir (recursive), or [path] if it is one."""
    if os.path.isfile(path) and path.endswith(".parquet"):
        return [path]
    hits: List[str] = []
    for dirpath, _dirs, files in os.walk(path):
        if ".cache" in dirpath:
            continue
        for f in files:
            if f.endswith(".parquet"):
                hits.append(os.path.join(dirpath, f))
    return sorted(hits)


def _read_prompt_column(src: str, columns: tuple[str, ...]) -> Iterable[str]:
    """Yield prompt strings from parquet / JSONL / plain-text sources."""
    parquet_files = _iter_parquet_files(src) if os.path.isdir(src) or src.endswith(".parquet") else []
    if parquet_files:
        import pyarrow.parquet as pq

        for pf in parquet_files:
            table = pq.read_table(pf)
            col = next((c for c in columns if c in table.column_names), None)
            if col is None:
                raise ValueError(f"{pf}: none of {columns} in parquet columns {table.column_names}")
            for v in table.column(col).to_pylist():
                yield "" if v is None else str(v)
        return
    if os.path.isfile(src) and src.endswith((".jsonl", ".json")):
        for line in _read_jsonl(src):
            for c in columns:
                if c in line:
                    yield str(line[c])
                    break
        return
    with open(src, encoding="utf-8") as f:  # plain text, one prompt per line
        for line in f:
            yield line.rstrip("\n")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", required=True, choices=sorted(CONVERTERS))
    p.add_argument("--input", required=True, help="Raw dataset path.")
    p.add_argument("--task", default="t2i", choices=list(TASKS))
    p.add_argument("--output", required=True, help="Output unified JSONL path.")
    args = p.parse_args()
    n = write_jsonl(CONVERTERS[args.source](args.input, args.task), args.output)
    print(f"Wrote {n} records to {args.output}")


if __name__ == "__main__":
    main()
