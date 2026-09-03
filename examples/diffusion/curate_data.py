#!/usr/bin/env python
# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Deterministic dedup, media filtering and fixed train/eval split (design.

§8.4).

Pure, reproducible transforms over the unified JSONL: prompts are deduped by a
stable content hash, short prompts and records with missing media are dropped,
and the train/eval split is assigned by hashing the ``sample_id`` (never RNG) so
a rerun yields the identical partition. The split helpers are unit-tested.

The prompt set the shipped launch scripts default to is produced by::

    python3 examples/diffusion/curate_data.py \\
        --input ${DATA_DIR}/raw/t2i/pickapic.jsonl \\
        --out-dir ${DATA_DIR}/processed/t2i \\
        --prefix pickapic_ --min-prompt-words 6 --eval-size 2048 \\
        --eval-subset-sizes 64 256 --no-check-media

which writes ``pickapic_train.jsonl`` / ``pickapic_eval.jsonl`` plus the
``pickapic_eval64.jsonl`` / ``pickapic_eval256.jsonl`` eval subsets.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from typing import Any, Dict, Iterable, List, Optional, Tuple


__all__ = [
    "dedup_key",
    "split_bucket",
    "curate",
    "eval_fraction_to_threshold",
    "prompt_word_count",
    "sample_id_of",
]


def dedup_key(record: Dict[str, Any]) -> str:
    """Stable content hash over prompt + sorted media paths."""
    media = sorted((record.get("images") or []) + (record.get("videos") or []))
    payload = json.dumps({"prompt": record.get("prompt", ""), "media": media}, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def sample_id_of(record: Dict[str, Any]) -> str:
    """The record's sample_id, falling back to its content hash."""
    return str(record.get("metadata", {}).get("sample_id", dedup_key(record)))


def prompt_word_count(record: Dict[str, Any]) -> int:
    """Whitespace-separated word count of the record's prompt."""
    return len(str(record.get("prompt", "")).split())


def eval_fraction_to_threshold(eval_fraction: float) -> int:
    """Map an eval fraction in [0,1] to a 0..9999 hash threshold."""
    return int(max(0.0, min(1.0, eval_fraction)) * 10000)


def split_bucket(sample_id: str, eval_fraction: float) -> str:
    """Assign 'eval' or 'train' by hashing sample_id (deterministic)."""
    h = int(hashlib.sha256(sample_id.encode("utf-8")).hexdigest(), 16) % 10000
    return "eval" if h < eval_fraction_to_threshold(eval_fraction) else "train"


def _media_exists(record: Dict[str, Any]) -> bool:
    for m in (record.get("images") or []) + (record.get("videos") or []):
        if not os.path.exists(m):
            return False
    return True


def curate(
    records: Iterable[Dict[str, Any]],
    eval_fraction: float,
    *,
    check_media: bool = True,
    min_prompt_words: int = 0,
    eval_size: Optional[int] = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Return ``(train, eval)`` after dedup + optional filters + a fixed split.

    ``min_prompt_words`` drops short captions. Under PickScore they score with
    almost no intra-group variance, which starves GRPO of the advantage signal
    it learns from; use ``>= 6`` for the Pick-a-Pic alignment preset.

    ``eval_size`` holds out EXACTLY that many records — ranked by the hash of
    their ``sample_id``, so the choice is reproducible and independent of the
    dataset size — instead of the ``eval_fraction`` hash bucket. Input order is
    preserved within each split either way.
    """
    seen = set()
    kept: List[Dict[str, Any]] = []
    for rec in records:
        key = dedup_key(rec)
        if key in seen:
            continue
        seen.add(key)
        if check_media and not _media_exists(rec):
            continue
        if min_prompt_words and prompt_word_count(rec) < min_prompt_words:
            continue
        kept.append(rec)

    if eval_size is not None:
        ranked = sorted(range(len(kept)), key=lambda i: (hashlib.sha256(sample_id_of(kept[i]).encode()).digest(), i))
        eval_idx = set(ranked[: max(0, min(int(eval_size), len(kept)))])
        train = [r for i, r in enumerate(kept) if i not in eval_idx]
        evalset = [r for i, r in enumerate(kept) if i in eval_idx]
        return train, evalset

    train, evalset = [], []
    for rec in kept:
        (evalset if split_bucket(sample_id_of(rec), eval_fraction) == "eval" else train).append(rec)
    return train, evalset


def _read_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def _write_jsonl(records: List[Dict[str, Any]], path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--eval-fraction", type=float, default=0.02)
    p.add_argument("--no-check-media", action="store_true")
    p.add_argument(
        "--min-prompt-words",
        type=int,
        default=0,
        help="Drop prompts shorter than this many words (use 6 for the Pick-a-Pic alignment preset).",
    )
    p.add_argument(
        "--eval-size",
        type=int,
        default=None,
        help="Hold out exactly this many records instead of --eval-fraction (use 2048 for alignment runs).",
    )
    p.add_argument(
        "--eval-subset-sizes",
        type=int,
        nargs="*",
        default=(),
        metavar="N",
        help="Also write the first N eval records as <prefix>evalN.jsonl, for cheaper periodic eval passes.",
    )
    p.add_argument(
        "--prefix",
        default="",
        help="Output filename prefix, e.g. --prefix pickapic_ writes pickapic_train.jsonl / pickapic_eval.jsonl.",
    )
    args = p.parse_args()
    train, evalset = curate(
        _read_jsonl(args.input),
        args.eval_fraction,
        check_media=not args.no_check_media,
        min_prompt_words=args.min_prompt_words,
        eval_size=args.eval_size,
    )
    _write_jsonl(train, os.path.join(args.out_dir, f"{args.prefix}train.jsonl"))
    _write_jsonl(evalset, os.path.join(args.out_dir, f"{args.prefix}eval.jsonl"))
    for size in args.eval_subset_sizes:
        _write_jsonl(evalset[:size], os.path.join(args.out_dir, f"{args.prefix}eval{size}.jsonl"))
    subsets = " ".join(f"{args.prefix}eval{s}={min(s, len(evalset))}" for s in args.eval_subset_sizes)
    print(f"train={len(train)} eval={len(evalset)} {subsets} -> {args.out_dir}")


if __name__ == "__main__":
    main()
