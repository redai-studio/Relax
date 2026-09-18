#!/usr/bin/env python
# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Schema / placeholder / media / split-leakage gate for unified JSONL (design.

§8.4).

Fails loudly on malformed records before a run wastes GPUs: required keys,
valid task, ``<image>``/``<video>`` placeholder ↔ media consistency, media
decodability (optional), and train/eval overlap (split leakage). The per-record
and leakage checks are pure, unit-tested functions.
"""

from __future__ import annotations

import argparse
import json
from typing import Any, Dict, Iterable, List, Set, Tuple


__all__ = ["check_record", "find_split_leakage", "TASKS"]

# Text-to-image is the only generation task supported on this branch; the i2i /
# t2v / i2v / v2v / t2av adapters live on `backup/diffusion-generative-rl-full`.
TASKS = ("t2i",)


def check_record(record: Dict[str, Any]) -> List[str]:
    """Return a list of problems with one record (empty ⇒ valid)."""
    errors: List[str] = []
    if "prompt" not in record or not isinstance(record["prompt"], str):
        errors.append("missing/invalid 'prompt'")
    meta = record.get("metadata", {})
    task = meta.get("task")
    if task not in TASKS:
        errors.append(f"metadata.task must be one of {TASKS}, got {task!r}")

    prompt = record.get("prompt", "")
    images = record.get("images") or []
    videos = record.get("videos") or []
    # placeholder ↔ media consistency
    if "<image>" in prompt and not images:
        errors.append("prompt has <image> placeholder but no images")
    if "<video>" in prompt and not videos:
        errors.append("prompt has <video> placeholder but no videos")
    return errors


def find_split_leakage(train_ids: Iterable[str], eval_ids: Iterable[str]) -> Set[str]:
    """Return sample_ids present in both splits (must be empty)."""
    return set(train_ids) & set(eval_ids)


def _sample_ids(path: str) -> List[str]:
    ids: List[str] = []
    for record in _read_jsonl(path):
        meta = record.get("metadata") or {}
        if isinstance(meta, dict):
            sample_id = meta.get("sample_id")
            if sample_id is not None:
                ids.append(sample_id)
    return ids


def _read_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def _inspect_file(path: str, decode_media: bool) -> Tuple[int, int]:
    total = 0
    bad = 0
    for i, rec in enumerate(_read_jsonl(path)):
        total += 1
        errs = check_record(rec)
        if decode_media:
            errs += _decode_media_errors(rec)
        if errs:
            bad += 1
            print(f"[{path}:{i}] {'; '.join(errs)}")
    return total, bad


def _decode_media_errors(record: Dict[str, Any]) -> List[str]:
    errors: List[str] = []
    for img in record.get("images") or []:
        try:
            from PIL import Image

            Image.open(img).verify()
        except Exception as e:  # pragma: no cover - depends on real media
            errors.append(f"image decode failed {img}: {e}")
    return errors


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train", required=True)
    p.add_argument("--eval", default=None)
    p.add_argument("--decode-media", action="store_true")
    args = p.parse_args()

    total, bad = _inspect_file(args.train, args.decode_media)
    if args.eval:
        et, eb = _inspect_file(args.eval, args.decode_media)
        total += et
        bad += eb
        train_ids = _sample_ids(args.train)
        eval_ids = _sample_ids(args.eval)
        leak = find_split_leakage(train_ids, eval_ids)
        if leak:
            raise SystemExit(f"SPLIT LEAKAGE: {len(leak)} sample_ids in both splits, e.g. {sorted(leak)[:5]}")
    if bad:
        raise SystemExit(f"{bad}/{total} records failed validation.")
    print(f"OK: {total} records valid.")


if __name__ == "__main__":
    main()
