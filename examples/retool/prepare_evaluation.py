# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Prepare full official evaluation sets, never execute model programs."""

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model", required=True)
    args = parser.parse_args()
    args.out.mkdir(exist_ok=True)
    from datasets import load_dataset
    from huggingface_hub import HfApi
    from transformers import AutoTokenizer

    from examples.retool.prepare_data import eval_case

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    specification = {
        "gsm8k": ("openai/gsm8k", "main", "test", 1319),
        "math500": ("HuggingFaceH4/MATH-500", None, "test", 500),
        "aime2025": ("yentinglin/aime_2025", None, "train", 30),
        "aime2026": ("math-ai/aime26", None, "test", 30),
    }
    manifest = dict(official_model=args.model, datasets={}, provenance={})
    for name, (repo, subset, split, count) in specification.items():
        revision = HfApi().dataset_info(repo).sha
        dataset = load_dataset(repo, subset, revision=revision)[split]
        assert len(dataset) == count
        rows = [eval_case(dict(row), name) for row in dataset]
        assert len({row["metadata"]["question"] for row in rows}) == count
        for row in rows:
            ids = tokenizer.apply_chat_template(row["prompt"], tokenize=True, add_generation_prompt=True)
            assert len(ids) <= 2048
        output = args.out / f"{name}.jsonl"
        if output.exists():
            assert [json.loads(line) for line in output.read_text().splitlines()] == rows
        else:
            with output.open("x") as stream:
                for row in rows:
                    stream.write(json.dumps(row) + "\n")
        manifest["datasets"][name] = str(output)
        manifest["provenance"][name] = dict(repo=repo, revision=revision, split=split, cases=count)
    with (args.out / "manifest.json").open("x") as stream:
        json.dump(manifest, stream, indent=2)


if __name__ == "__main__":
    main()
