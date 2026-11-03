#!/usr/bin/env python3
# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Audit complete pair coverage and compare fixed retrieval/rank fusion."""

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any


sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from examples.lamra_rerank.tools.prepare_data import sha256  # noqa: E402


def recall_metrics(pairs: list[dict[str, Any]], predictions: dict[int, dict[str, Any]]) -> dict[str, Any]:
    if set(predictions) != set(range(len(pairs))):
        raise ValueError("Missing, extra or non-contiguous scored pair indices")
    groups: dict[str, list[dict[str, Any]]] = {}
    for index, pair in enumerate(pairs):
        score = predictions[index]
        if pair["index"] != index or (score["qid"], score["did"]) != (pair["qid"], pair["did"]):
            raise ValueError(f"Query/candidate identity mismatch at pair {index}")
        if not math.isfinite(score["p_yes"]) or not 0 <= score["p_yes"] <= 1:
            raise ValueError(f"Invalid Yes probability at pair {index}")
        groups.setdefault(pair["qid"], []).append({**pair, "p_yes": score["p_yes"]})
    hits = {recipe: {k: 0 for k in [1, 5, 10, 20, 50]} for recipe in ["retrieval", "rank_only", "equal_fusion"]}
    for qid, rows in groups.items():
        if len(rows) != 50 or len({row["did"] for row in rows}) != 50:
            raise ValueError(f"Incomplete top-50 candidate coverage for {qid}")
        for recipe in hits:

            def score(row: dict[str, Any]) -> float:
                if recipe == "retrieval":
                    return row["ret_score"]
                if recipe == "rank_only":
                    return row["p_yes"]
                return row["ret_score"] + row["p_yes"]

            ordered = sorted(rows, key=score, reverse=True)
            for k in hits[recipe]:
                hits[recipe][k] += any(row["relevant"] for row in ordered[:k])
    if not groups:
        raise ValueError("No evaluated queries")
    values = {
        recipe: {str(k): 100 * value / len(groups) for k, value in counts.items()} for recipe, counts in hits.items()
    }
    return {
        "queries": len(groups),
        "pairs": len(pairs),
        "topk": 50,
        "fusion_weight": 1.0,
        "recall_percent": values,
        "fusion_minus_retrieval_pp": {
            k: values["equal_fusion"][k] - values["retrieval"][k] for k in values["retrieval"]
        },
        "tie_policy": "Stable original retrieval candidate order",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--world-size", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.pairs.with_suffix(".manifest.json").read_text())
    if sha256(args.pairs) != manifest["pairs_sha256"]:
        raise ValueError("Evaluation pairs changed after preparation")
    with args.pairs.open() as stream:
        pairs = [json.loads(line) for line in stream]
    predictions = {}
    model_paths = set()
    for rank in range(args.world_size):
        loading = json.loads((args.scores / f"loading_rank{rank}.json").read_text())
        complete = json.loads((args.scores / f"completion_rank{rank}.json").read_text())
        path = args.scores / f"scores_rank{rank}.jsonl"
        if loading["rank"] != rank or any(loading["loading_info"].values()):
            raise ValueError(f"Strict model loading failed for rank {rank}")
        if complete["status"] != "passed" or complete["world_size"] != args.world_size:
            raise ValueError(f"Rank {rank} did not complete the expected evaluation")
        if complete["scores_sha256"] != sha256(path) or complete["pairs_sha256"] != manifest["pairs_sha256"]:
            raise ValueError(f"Score/pair identity mismatch for rank {rank}")
        model_paths.add(loading["model"])
        count = 0
        with path.open() as stream:
            for line in stream:
                score = json.loads(line)
                if score["index"] in predictions:
                    raise ValueError(f"Duplicate pair index: {score['index']}")
                predictions[score["index"]] = score
                count += 1
        if count != complete["pairs"]:
            raise ValueError(f"Rank {rank} has an incomplete score file")
    if len(model_paths) != 1:
        raise ValueError("Ranks loaded different model paths")
    metrics = recall_metrics(pairs, predictions)
    if metrics["queries"] != manifest["queries"] or metrics["pairs"] != manifest["pairs"]:
        raise ValueError("Evaluation did not cover the complete test set")
    metrics.update(
        status="complete",
        rank_model=model_paths.pop(),
        retrieval_identity=manifest["retrieval_identity"],
        pairs_sha256=manifest["pairs_sha256"],
        strict_loading_ranks=args.world_size,
    )
    args.output.write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics), flush=True)


if __name__ == "__main__":
    main()
