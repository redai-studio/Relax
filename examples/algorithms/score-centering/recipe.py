# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Countdown data preparation, signed rewards and generation length limits.

Data generation follows the author's Countdown environment; this module only
provides signed arithmetic rewards and a total prompt + completion token cap.
"""

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any

from .countdown import generate, messages, reward


def signed_reward(args: Any, sample: Any, **kwargs: Any) -> float | list[float]:
    scores = reward(args, sample, **kwargs)
    return [2 * score - 1 for score in scores] if isinstance(scores, list) else 2 * scores - 1


async def generate_bounded(args: Any, sample: Any, sampling_params: dict, evaluation: bool = False) -> Any:
    from relax.engine.rollout.sglang_rollout import GenerateState
    from relax.engine.rollout.sglang_rollout import generate as generate_sample

    state = GenerateState(args)
    prompt_tokens = state.tokenizer.encode(sample.prompt, add_special_tokens=False)
    remaining = 512 - len(prompt_tokens)
    if remaining <= 0:
        raise ValueError("Countdown prompt exceeds the published 512-token budget")
    sampling_params = dict(sampling_params, max_new_tokens=min(sampling_params["max_new_tokens"], remaining))
    return await generate_sample(args, sample, sampling_params, evaluation=evaluation)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    args = parser.parse_args()
    from datasets import Dataset
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    rng = random.Random(0)
    problems = [generate(rng) for _ in range(20_000)]
    indices = Dataset.from_dict({"index": list(range(len(problems)))}).shuffle(seed=0)["index"]
    rows = []
    for index in indices:
        problem = problems[index]
        prompt = messages(problem)
        tokens = tokenizer.apply_chat_template(
            prompt, tokenize=True, add_generation_prompt=True, return_dict=False, enable_thinking=False
        )
        rows.append(
            dict(
                prompt=prompt,
                label=json.dumps(problem),
                id=f"countdown-reference-{index}",
                max_response_tokens=512 - len(tokens),
                chat_template_kwargs={"enable_thinking": False},
            )
        )
    assert all(row["max_response_tokens"] > 0 for row in rows)
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = {"generation_seed": 0, "shuffle_seed": 0, "total_tokens": 512, "files": {}}
    for split, selected in (("dev", rows[:64]), ("train", rows[64:])):
        content = "".join(json.dumps(row) + "\n" for row in selected)
        path = args.output / f"countdown-{split}.jsonl"
        if path.exists() and path.read_text() != content:
            raise ValueError(f"Refusing to replace a different dataset: {path}")
        path.write_text(content)
        manifest["files"][path.name] = {"rows": len(selected), "sha256": hashlib.sha256(content.encode()).hexdigest()}
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
