# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Send an exactly tokenized request to the example SGLang classifier."""

from __future__ import annotations

import argparse
import json
import math
import urllib.request
from collections.abc import Mapping
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True, help="Exported classifier directory")
    parser.add_argument("--text", required=True, help="User-message content")
    parser.add_argument("--server-url", default="http://127.0.0.1:30000")
    parser.add_argument("--served-model-name", default="qwen3.5-seq-cls")
    parser.add_argument(
        "--problem-type",
        choices=["single_label_classification", "multi_label_classification"],
        default=None,
        help="Defaults to problem_type in config.json",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Multi-label sigmoid decision threshold; it is not used to compute training loss",
    )
    return parser.parse_args()


def post_json(url: str, payload: dict[str, Any]) -> dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(request, timeout=120) as response:
        return json.loads(response.read().decode("utf-8"))


def load_config(model_dir: Path) -> dict[str, Any]:
    config_path = model_dir / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Missing model config: {config_path}")
    return json.loads(config_path.read_text(encoding="utf-8"))


def render_input_ids(model_dir: Path, text: str) -> list[int]:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    encoded = tokenizer.apply_chat_template(
        [{"role": "user", "content": text}],
        tokenize=True,
        add_generation_prompt=False,
    )
    if isinstance(encoded, Mapping):
        input_ids = encoded.get("input_ids")
    else:
        input_ids = encoded
    if hasattr(input_ids, "tolist"):
        input_ids = input_ids.tolist()
    if input_ids and isinstance(input_ids[0], list):
        if len(input_ids) != 1:
            raise ValueError(f"Expected one tokenized input, got batch size {len(input_ids)}")
        input_ids = input_ids[0]
    if not input_ids:
        raise ValueError("The chat template produced no input tokens")
    return [int(token_id) for token_id in input_ids]


def label_mapping(config: dict[str, Any]) -> dict[int, str]:
    raw = config.get("id2label") or {}
    return {int(index): str(label) for index, label in raw.items()}


def sigmoid(value: float) -> float:
    if value >= 0:
        return 1.0 / (1.0 + math.exp(-value))
    exp_value = math.exp(value)
    return exp_value / (1.0 + exp_value)


def classify_single(args: argparse.Namespace, input_ids: list[int]) -> dict[str, Any]:
    return post_json(
        f"{args.server_url.rstrip('/')}/v1/classify",
        {"model": args.served_model_name, "input": input_ids},
    )


def classify_multi(
    args: argparse.Namespace,
    input_ids: list[int],
    config: dict[str, Any],
) -> dict[str, Any]:
    response = post_json(
        f"{args.server_url.rstrip('/')}/v1/score",
        {
            "model": args.served_model_name,
            "query": [],
            "items": [input_ids],
            "apply_softmax": False,
        },
    )
    scores = response.get("scores")
    if not isinstance(scores, list) or len(scores) != 1 or not isinstance(scores[0], list):
        raise ValueError(f"Unexpected /v1/score response: {response}")
    logits = [float(value) for value in scores[0]]
    probabilities = [sigmoid(value) for value in logits]
    labels = label_mapping(config)
    predicted = [
        {"id": index, "label": labels.get(index, f"LABEL_{index}"), "probability": probability}
        for index, probability in enumerate(probabilities)
        if probability >= args.threshold
    ]
    return {
        "logits": logits,
        "probabilities": probabilities,
        "threshold": args.threshold,
        "predicted_labels": predicted,
        "usage": response.get("usage"),
    }


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.threshold <= 1.0:
        raise ValueError("--threshold must be in [0, 1]")
    config = load_config(args.model_dir)
    problem_type = args.problem_type or config.get("problem_type")
    if problem_type not in ("single_label_classification", "multi_label_classification"):
        raise ValueError(f"Unsupported problem_type={problem_type!r}")

    input_ids = render_input_ids(args.model_dir, args.text)
    if problem_type == "single_label_classification":
        result = classify_single(args, input_ids)
    else:
        result = classify_multi(args, input_ids, config)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
