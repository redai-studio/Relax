#!/usr/bin/env python3
# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Bind a fixed retrieval top-50 to M-BEIR query/candidate identities."""

import argparse
import json
import math
import random
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from examples.lamra_rerank.data import format_text, load_instructions, make_entity, pointwise_messages  # noqa: E402
from examples.lamra_rerank.tools.prepare_data import sha256  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--instructions", type=Path, required=True)
    parser.add_argument("--qrels", type=Path, required=True)
    parser.add_argument("--query-names", type=Path, required=True)
    parser.add_argument("--candidate-names", type=Path, required=True)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--retrieval-identity", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    query_names = json.loads(args.query_names.read_text())
    candidate_names = json.loads(args.candidate_names.read_text())
    scores = json.loads(args.scores.read_text())
    if len(query_names) != len(set(query_names)) or not len(query_names) == len(candidate_names) == len(scores):
        raise ValueError("Retrieval query/candidate/score coverage mismatch")
    with args.queries.open() as stream:
        queries = {query["qid"]: query for line in stream if (query := json.loads(line))}
    qrels: dict[str, set[str]] = {}
    for line in args.qrels.read_text().splitlines():
        fields = line.split()
        if int(fields[3]) > 0:
            qrels.setdefault(fields[0], set()).add(fields[2])
    if set(query_names) != set(queries) or set(query_names) != set(qrels):
        raise ValueError("Evaluate every test query with its complete ground truth")
    needed = {did for values in candidate_names for did in values[:50]}
    needed.update(query["pos_cand_list"][0] for query in queries.values())
    candidates = {}
    with args.candidates.open() as stream:
        for line in stream:
            candidate = json.loads(line)
            if candidate["did"] in needed:
                candidates[candidate["did"]] = candidate
    if set(candidates) != needed:
        raise ValueError(f"Missing {len(needed - set(candidates))} candidate identities")
    instructions = load_instructions(args.instructions)
    rng = random.Random(args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        for query_index, (qid, names, raw_scores) in enumerate(zip(query_names, candidate_names, scores, strict=True)):
            if len(raw_scores) == 1 and isinstance(raw_scores[0], list):
                raw_scores = raw_scores[0]
            if len(names) < 50 or len(raw_scores) < 50 or len(set(names[:50])) != 50:
                raise ValueError(f"Incomplete or duplicate top-50 for {qid}")
            if not all(math.isfinite(float(value)) for value in raw_scores[:50]):
                raise ValueError(f"Non-finite retrieval scores for {qid}")
            query = queries[qid]
            positive = candidates[query["pos_cand_list"][0]]
            key = (qid.split(":")[0], query["query_modality"], positive["modality"])
            instruction = format_text(rng.choice(instructions[key]))
            query_entity = make_entity(
                format_text(f"{instruction} {query.get('query_txt') or ''}"),
                query.get("query_img_path"),
                args.image_root,
            )
            for position, (did, score) in enumerate(zip(names[:50], raw_scores[:50], strict=True)):
                candidate = candidates[did]
                entity = make_entity(format_text(candidate.get("txt")), candidate.get("img_path"), args.image_root)
                row = {
                    "index": query_index * 50 + position,
                    "qid": qid,
                    "did": did,
                    "candidate_position": position,
                    "ret_score": float(score),
                    "relevant": did in qrels[qid],
                    "messages": pointwise_messages(query_entity, entity),
                }
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    manifest = {
        "status": "prepared",
        "queries": len(query_names),
        "pairs": len(query_names) * 50,
        "topk": 50,
        "seed": args.seed,
        "instructions": "One seeded instruction per query, shared by its 50 candidates",
        "text_truncation": False,
        "retrieval_identity": json.loads(args.retrieval_identity.read_text()),
        "pairs_sha256": sha256(args.output),
        "source_sha256": {
            str(path): sha256(path)
            for path in [
                args.queries,
                args.candidates,
                args.instructions,
                args.qrels,
                args.query_names,
                args.candidate_names,
                args.scores,
                args.retrieval_identity,
            ]
        },
    }
    args.output.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest), flush=True)


if __name__ == "__main__":
    main()
