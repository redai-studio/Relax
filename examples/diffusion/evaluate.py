#!/usr/bin/env python3
# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Offline evaluation for native generative RL checkpoints.

Scores an already-generated artifact set (the eval-summary jsonl the training
loop writes, or any jsonl of ``{prompt, image}`` records) with the same reward
scorers used during training, and prints / writes a per-component summary. This
is the base-vs-RL comparison gate: run it once against the base model's
artifacts and once against the RL checkpoint's, then diff the summaries.

It deliberately does NOT generate images — generation needs the SGLang diffusion
server and a GPU, which the training loop already owns. Point ``--artifacts`` at
what a run produced (``<artifact_root>/eval/...`` plus the eval jsonl), or at
the output of your own generation pass.

Usage::

    python3 examples/diffusion/evaluate.py \\
        --artifacts /path/to/eval_records.jsonl \\
        --reward-scorer-path relax.engine.rewards.pickscore.PickScoreScorer \\
        --reward-model-path /models/PickScore_v1 \\
        --output /path/to/summary.json

Each artifact record must carry a ``prompt`` and at least one image URI, either
as ``image`` / ``uri`` or as a manifest-style ``outputs: [{track, uri}]`` list.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, Iterable, List, Mapping, Sequence


def read_records(path: str) -> List[Dict[str, Any]]:
    """Read a jsonl artifact file into records."""
    records: List[Dict[str, Any]] = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def record_to_request(record: Mapping[str, Any]) -> Dict[str, Any]:
    """Normalize one artifact record into a scorer ``RewardRequest`` payload.

    Accepts either a manifest-style ``outputs`` list or a flat ``image`` /
    ``uri`` field, so both training-emitted records and hand-made ones work.
    """
    outputs = record.get("outputs")
    if not outputs:
        uri = record.get("image") or record.get("uri") or record.get("image_path")
        outputs = [{"track": "image", "uri": uri}] if uri else []
    return {
        "prompt": record.get("prompt", ""),
        "outputs": list(outputs),
        "conditions": record.get("conditions", []),
        "multimodal_inputs": record.get("multimodal_inputs"),
        "metadata": record.get("metadata"),
    }


def summarize(component_rewards: Mapping[str, Sequence[float]], num_records: int) -> Dict[str, Any]:
    """Build the summary dict: count + per-component mean/min/max/std."""
    summary: Dict[str, Any] = {"num_records": int(num_records), "components": {}}
    for name, values in component_rewards.items():
        vals = [float(v) for v in values]
        if not vals:
            continue
        mean = sum(vals) / len(vals)
        variance = sum((v - mean) ** 2 for v in vals) / len(vals)
        summary["components"][name] = {
            "mean": mean,
            "min": min(vals),
            "max": max(vals),
            "std": variance**0.5,
            "count": len(vals),
        }
    return summary


def score_records(args, records: Iterable[Mapping[str, Any]]) -> Dict[str, List[float]]:
    """Score records with the configured scorer (lazy import — needs torch)."""
    from relax.distributed.ray.generative_reward import GenerativeRewardManager

    requests = [record_to_request(r) for r in records]
    if not requests:
        return {}
    manager = GenerativeRewardManager.local(args)
    return manager.score(requests)


def build_scorer_args(parsed: argparse.Namespace) -> argparse.Namespace:
    """Minimal namespace the scorer classes read (mirrors the train-time
    flags)."""
    return argparse.Namespace(
        reward_runtime=parsed.reward_runtime,
        reward_scorer_path=parsed.reward_scorer_path,
        reward_model_path=parsed.reward_model_path,
        reward_endpoint=parsed.reward_endpoint,
    )


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Offline scoring for generative RL artifacts.")
    parser.add_argument("--artifacts", required=True, help="Path to a jsonl of generated artifact records.")
    parser.add_argument("--reward-scorer-path", required=True, help="Dotpath to a GenerativeRewardScorer.")
    parser.add_argument("--reward-model-path", default=None)
    parser.add_argument("--reward-runtime", default="cpu", choices=["cpu", "colocate", "remote"])
    parser.add_argument("--reward-endpoint", default=None)
    parser.add_argument("--output", default=None, help="Write the summary JSON here (default: stdout only).")
    return parser.parse_args(argv)


def main(argv: List[str] | None = None) -> int:
    parsed = parse_args(argv)
    records = read_records(parsed.artifacts)
    if not records:
        print(f"No records found in {parsed.artifacts}")
        return 1

    component_rewards = score_records(build_scorer_args(parsed), records)
    summary = summarize(component_rewards, len(records))

    print(json.dumps(summary, indent=2, sort_keys=True))
    if parsed.output:
        os.makedirs(os.path.dirname(os.path.abspath(parsed.output)), exist_ok=True)
        with open(parsed.output, "w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2, sort_keys=True)
        print(f"Wrote summary to {parsed.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
