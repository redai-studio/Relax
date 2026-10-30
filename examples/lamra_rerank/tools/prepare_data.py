#!/usr/bin/env python3
# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Expand M-BEIR query rows into deterministic LamRA rank supervision."""

import argparse
import hashlib
import json
import os
import random
import sys
import time
from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import Any


sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from examples.lamra_rerank.data import (  # noqa: E402
    format_text,
    listwise_messages,
    load_instructions,
    make_entity,
    pointwise_messages,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def make_triplet(
    query: dict[str, Any],
    candidates: dict[str, dict[str, Any]],
    ranking: dict[str, Any],
    instructions: dict[tuple[str, str, str], list[str]],
    image_root: Path,
    truncate: Any,
    rng: random.Random,
) -> list[dict[str, Any]] | None:
    """Exclude every known positive from both pointwise and listwise
    negatives."""
    negative_count = rng.randint(1, 4)
    positive_id = rng.choice(query["pos_cand_list"])
    positive = candidates[positive_id]
    key = (query["qid"].split(":")[0], query["query_modality"], positive["modality"])
    instruction = format_text(rng.choice(instructions[key]))
    query_text = truncate(format_text(f"{instruction} {query.get('query_txt') or ''}"))
    query_entity = make_entity(query_text, query.get("query_img_path"), image_root)
    known_positives = set(ranking["gt_docs"]) | set(query["pos_cand_list"])
    negatives = list(dict.fromkeys(did for did in ranking["top100_docs"] if did not in known_positives))
    if not negatives:
        return None

    def entity(did: str) -> dict[str, str]:
        candidate = candidates[did]
        return make_entity(truncate(format_text(candidate.get("txt"))), candidate.get("img_path"), image_root)

    negative_id = rng.choice(negatives)
    negative_ids = rng.sample(negatives, min(negative_count, len(negatives)))
    answer = rng.randint(1, len(negative_ids) + 1)
    list_ids = negative_ids.copy()
    list_ids.insert(answer - 1, positive_id)
    positive_entity = entity(positive_id)
    shared = {"qid": query["qid"], "positive_id": positive_id}
    return [
        {
            "messages": pointwise_messages(query_entity, positive_entity, "Yes"),
            "metadata": {**shared, "kind": "positive", "candidate_ids": [positive_id]},
        },
        {
            "messages": pointwise_messages(query_entity, entity(negative_id), "No"),
            "metadata": {**shared, "kind": "negative", "candidate_ids": [negative_id]},
        },
        {
            "messages": listwise_messages(query_entity, [entity(did) for did in list_ids], answer),
            "metadata": {**shared, "kind": "listwise", "candidate_ids": list_ids, "answer": answer},
        },
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--instructions", type=Path, required=True)
    parser.add_argument("--rerank-data", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-queries", type=int, help="Explicit small preparation audit; omit for full training")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if not args.image_root.is_dir():
        raise FileNotFoundError(args.image_root)
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=False)

    @lru_cache(maxsize=100000)
    def truncate(text: str) -> str:
        ids = tokenizer(text, truncation=True, max_length=256, padding=False, add_special_tokens=False)["input_ids"]
        return tokenizer.decode(ids)

    with args.candidates.open() as stream:
        candidates = {row["did"]: row for line in stream if (row := json.loads(line))}
    with args.rerank_data.open() as stream:
        rankings = json.load(stream)
    instructions = load_instructions(args.instructions)
    rng = random.Random(args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".partial")
    if temporary.exists():
        raise FileExistsError(f"Partial output requires inspection: {temporary}")
    started = time.time()
    counts = Counter()
    skipped = []
    image_paths: set[str] = set()
    with args.queries.open() as source, temporary.open("x") as destination:
        for line in source:
            if args.max_queries is not None and counts["source_queries"] >= args.max_queries:
                break
            query = json.loads(line)
            triplet = make_triplet(
                query, candidates, rankings[query["qid"]], instructions, args.image_root, truncate, rng
            )
            counts["source_queries"] += 1
            if triplet is None:
                skipped.append(query["qid"])
                counts["queries_without_true_negative"] += 1
                continue
            counts["training_queries"] += 1
            for row in triplet:
                counts[row["metadata"]["kind"]] += 1
                for part in row["messages"][0]["content"]:
                    if part["type"] == "image":
                        image_paths.add(part["image"])
                destination.write(json.dumps(row, ensure_ascii=False) + "\n")
            if counts["source_queries"] % 10000 == 0:
                progress = {"stage": "preparing", "at": time.time(), **counts}
                args.output.with_suffix(".progress.json").write_text(json.dumps(progress, indent=2))
                print(json.dumps(progress), flush=True)
        destination.flush()
        os.fsync(destination.fileno())
    os.replace(temporary, args.output)
    manifest = {
        "status": "prepared",
        "at": time.time(),
        "runtime_s": time.time() - started,
        "seed": args.seed,
        **counts,
        "sequences": 3 * counts["training_queries"],
        "unique_image_paths": len(image_paths),
        "image_root": str(args.image_root),
        "images_validated": False,
        "negative_policy": "Exclude union(gt_docs, pos_cand_list); skip queries with no true negative",
        "skipped_query_ids": skipped,
        "text_truncation_tokens_per_field": 256,
        "output_sha256": sha256(args.output),
        "source_files": {
            str(path): {"bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in [args.queries, args.candidates, args.instructions, args.rerank_data]
        },
    }
    args.output.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2))
    args.output.with_suffix(".image_paths.json").write_text(json.dumps(sorted(image_paths)))
    print(json.dumps({key: value for key, value in manifest.items() if key != "skipped_query_ids"}), flush=True)


if __name__ == "__main__":
    main()
