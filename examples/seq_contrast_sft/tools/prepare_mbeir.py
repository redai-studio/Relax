#!/usr/bin/env python3
# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Prepare M-BEIR queries without fixing runtime shuffle or positive/prompt
draws."""

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from examples.lamra_rerank.tools.prepare_data import sha256  # noqa: E402


def load_instructions(path: Path) -> dict[tuple[str, str, str], list[str]]:
    """Use the native TSV split/strip rules, including literal quotation
    marks."""
    result = {}
    with path.open() as stream:
        next(stream)
        for line in stream:
            fields = line.strip().split("\t")
            if len(fields) < 5:
                raise ValueError("Malformed native query instruction row")
            result[(fields[3], fields[0], fields[1])] = [text for text in fields[4:] if text]
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--instructions", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--expected-query-count", type=int)
    parser.add_argument("--expected-candidate-count", type=int)
    args = parser.parse_args()
    if args.batch_size <= 0:
        raise ValueError("Batch size must be positive")
    if args.output.exists() or args.output.with_suffix(".partial").exists():
        raise FileExistsError("Inspect existing complete/partial preparation before reuse")
    import torch
    from transformers import AutoTokenizer

    if torch.cuda.is_initialized():
        raise RuntimeError("M-BEIR preparation must remain on CPU")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=False, local_files_only=True)
    if "<emb>" not in tokenizer.get_vocab():
        raise ValueError("The supplied PT Retrieve tokenizer must already contain <emb>")
    with args.queries.open() as stream:
        queries = [json.loads(line) for line in stream]
    with args.candidates.open() as stream:
        candidates = {row["did"]: row for line in stream if (row := json.loads(line))}
    if not queries:
        raise ValueError("M-BEIR requires a nonempty training set")
    for actual, expected in (
        (len(queries), args.expected_query_count),
        (len(candidates), args.expected_candidate_count),
    ):
        if expected is not None and actual != expected:
            raise ValueError(f"Source cardinality differs: {actual} != {expected}")
    instructions = load_instructions(args.instructions)
    prepare_pairs(args, queries, candidates, instructions)


def prepare_pairs(
    args: argparse.Namespace,
    queries: list[dict],
    candidates: dict[str, dict],
    instructions: dict[tuple[str, str, str], list[str]],
) -> None:
    """Join lookup data in source order while retaining all runtime choices."""
    started = time.time()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".partial")
    image_paths = set()
    with temporary.open("x") as destination:
        for index, query in enumerate(queries):
            positives = [candidates[did] for did in query["pos_cand_list"]]
            keys = sorted({(query["qid"].split(":")[0], query["query_modality"], p["modality"]) for p in positives})
            source = {
                "query": query,
                "positives": positives,
                "instructions": [[list(key), instructions[key]] for key in keys],
                "image_root": str(args.image_root),
            }
            for image in [query.get("query_img_path"), *(p.get("img_path") for p in positives)]:
                if image:
                    path = args.image_root / image
                    if not path.is_relative_to(args.image_root):
                        raise ValueError(f"Image path escapes image root: {image!r}")
                    image_paths.add(str(path))
            destination.write(json.dumps({"native_mbeir": source, "metadata": {"source_index": index}}) + "\n")
        destination.flush()
        os.fsync(destination.fileno())
    temporary.replace(args.output)
    with args.output.with_suffix(".images.jsonl").open("x") as stream:
        for path in sorted(image_paths):
            stream.write(
                json.dumps({"image": path, "relative_path": str(Path(path).relative_to(args.image_root))}) + "\n"
            )
    visits = math.ceil(len(queries) / args.batch_size) * args.batch_size
    receipt = {
        "stage": "prepared",
        "sampling": "native_runtime",
        "shuffle": "loader default; source rows unchanged",
        "positive_and_instruction_choices": "runtime; not frozen",
        "raw_queries": len(queries),
        "prepared_rows": len(queries),
        "candidates": len(candidates),
        "pair_visits": visits,
        "padding_visits": visits - len(queries),
        "updates": visits // args.batch_size,
        "global_batch_size": args.batch_size,
        "unique_images": len(image_paths),
        "sources_sha256": {str(path): sha256(path) for path in (args.queries, args.candidates, args.instructions)},
        "pairs_sha256": sha256(args.output),
        "CUDA_initialized": False,
        "elapsed_seconds": time.time() - started,
    }
    args.output.with_suffix(".meta.json").write_text(json.dumps(receipt, indent=2) + "\n")


if __name__ == "__main__":
    main()
