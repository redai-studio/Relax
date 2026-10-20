"""Download DeepCoder training splits and write Relax train.jsonl."""

import argparse
import json
from pathlib import Path
from typing import Any


DATASET = "agentica-org/DeepCoder-Preview-Dataset"
REVISION = "177913a7bd43791646ef6a43645caa3c871ab3db"
TRAIN_SPLITS = {"primeintellect": 16252, "taco": 7436, "lcbv5": 599}

# agentica-project/rllm@8766fffb: rllm/data/utils.py and rllm/system_prompts.py.
LCB_INSTRUCTION = (
    "You are an expert Python programmer. You will be given a question (problem specification) and will generate "
    "a correct Python program that matches the specification and passes all tests."
)
LCB_STARTER_FORMAT = "You will use the following starter code to write the solution to the problem and enclose your code within delimiters."
LCB_STDIO_FORMAT = (
    "Read the inputs from stdin solve the problem and write the answer to stdout (do not directly test on the sample "
    "inputs). Enclose your code within delimiters as follows. Ensure that when the python program runs, it reads the "
    "inputs, runs the algorithm and writes output to STDOUT."
)


def convert_row(row: dict[str, Any], subset: str, index: int) -> dict[str, Any]:
    """Preserve DeepCoder prompts and adapt tests for SandboxFusion."""
    prompt = row["problem"]
    tests = json.loads(row["tests"])
    if subset == "lcbv5":
        starter = row.get("starter_code")
        prompt = f"{LCB_INSTRUCTION}\n\n{prompt}"
        if starter:
            prompt += f"### Format: {LCB_STARTER_FORMAT}\n```python\n{starter}\n```\n\n"
        else:
            prompt += f"### Format: {LCB_STDIO_FORMAT}\n```python\n# YOUR CODE HERE\n```\n\n"
        prompt += "### Answer: (use the provided format with backticks)\n\n"
        function = (row.get("metadata") or {}).get("func_name")
    else:
        function = tests.get("fn_name") if subset == "taco" else tests[0].get("fn_name")

    inputs = tests["inputs"] if subset == "taco" else [test["input"] for test in tests]
    outputs = tests["outputs"] if subset == "taco" else [test["output"] for test in tests]
    assert inputs and len(inputs) == len(outputs), f"Unpaired tests: {subset}/{index}"
    if function and subset != "lcbv5":
        inputs = ["\n".join(json.dumps(argument) for argument in arguments) for arguments in inputs]
        outputs = [json.dumps(output[0] if isinstance(output, list) else output) for output in outputs]
    else:
        inputs = [value if isinstance(value, str) else "\n".join(value) for value in inputs]
        outputs = [value if isinstance(value, str) else "\n".join(value) for value in outputs]
    return {
        "input": [{"role": "user", "content": prompt}],
        "label": json.dumps({"inputs": inputs, "outputs": outputs, "fn_name": function}, ensure_ascii=False),
        "metadata": {"data_source": subset, "task_id": f"{subset}/train/{index}"},
    }


def prepare(output_dir: Path, cache_dir: Path | None = None) -> None:
    """Download train shards and stream every row to train.jsonl."""
    import pyarrow.parquet as parquet
    from huggingface_hub import snapshot_download

    source = Path(
        snapshot_download(
            DATASET,
            repo_type="dataset",
            revision=REVISION,
            allow_patterns=[f"{subset}/train-*.parquet" for subset in TRAIN_SPLITS],
            cache_dir=cache_dir,
            max_workers=2,
        )
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    temporary = output_dir / "train.jsonl.tmp"
    with temporary.open("w", encoding="utf-8") as handle:
        for subset, expected in TRAIN_SPLITS.items():
            count = 0
            columns = ["problem", "tests"] + (["starter_code", "metadata"] if subset == "lcbv5" else [])
            for shard in sorted((source / subset).glob("train-*.parquet")):
                for batch in parquet.ParquetFile(shard).iter_batches(batch_size=8, columns=columns):
                    for row in batch.to_pylist():
                        handle.write(json.dumps(convert_row(row, subset, count), ensure_ascii=False) + "\n")
                        count += 1
            assert count == expected, f"{subset}: {count} != {expected}"
    temporary.replace(output_dir / "train.jsonl")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path)
    args = parser.parse_args()
    prepare(args.output_dir, args.cache_dir)
