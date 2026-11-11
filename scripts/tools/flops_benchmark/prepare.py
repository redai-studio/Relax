# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Freeze real SFT examples in the order consumed by the production loader."""

import argparse
import hashlib
import json
import random
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from relax.utils.logging_utils import get_logger
from scripts.tools.flops_benchmark.common import digest, file_digest, write_json


logger = get_logger(__name__)


def choose_batches(
    rows: list[dict], recipe: str, steps: int, batch_size: int, capacity: int, seed: int
) -> list[list[int]]:
    if min(steps, batch_size, capacity) <= 0:
        raise ValueError("steps, batch size and capacity must be positive")
    rng = random.Random(seed)
    valid = [i for i, row in enumerate(rows) if 1 < row["length"] <= capacity and row["effective_tokens"] > 0]
    if recipe == "stratified":
        if batch_size != 16:
            raise ValueError("The 1-long/15-short recipe requires global batch size 16")
        long = [i for i in valid if 10000 <= rows[i]["length"] <= 11200]
        short = [i for i in valid if 1750 <= rows[i]["length"] <= 2250]
        rng.shuffle(long)
        rng.shuffle(short)
        if len(long) < steps or len(short) < 15 * steps:
            raise ValueError("Insufficient distinct real examples for the requested stratified run")
        batches = [[long[step], *short[step * 15 : (step + 1) * 15]] for step in range(steps)]
        for batch in batches:
            rng.shuffle(batch)
        return batches
    rng.shuffle(valid)
    if len(valid) < steps * batch_size:
        raise ValueError("Insufficient distinct real examples for the requested natural run")
    return [valid[i * batch_size : (i + 1) * batch_size] for i in range(steps)]


def loader_order(size: int, seed: int) -> list[int]:
    from relax.utils.data.streaming_dataset import IndexManager

    manager = IndexManager(size, seed=seed)
    manager.shuffle(0)
    return list(manager.indices)


def _dataset(path: Path, tokenizer: Any, seed: int) -> Any:
    from relax.engine.sft.dataset.streaming import SFTStreamingDataset

    return SFTStreamingDataset(
        str(path),
        tokenizer=tokenizer,
        prompt_key="problem",
        label_key="generated_solution",
        seed=seed,
        prefetch_max_cached=0,
        oversize_strategy="keep",
    )


def _sample_metadata(sample: Any) -> dict:
    if sample is None:
        raise ValueError("The production renderer rejected a selected example")
    return {
        "length": sample.tokens.numel(),
        "effective_tokens": int(sample.loss_mask[1:].sum()),
        "tokens_sha256": hashlib.sha256(memoryview(sample.tokens.numpy())).hexdigest(),
        "mask_sha256": hashlib.sha256(memoryview(sample.loss_mask.numpy())).hexdigest(),
    }


def freeze(args: argparse.Namespace) -> Path:
    from huggingface_hub import HfApi, hf_hub_download
    from transformers import AutoTokenizer

    api = HfApi()
    model_revision = api.model_info(args.model, revision=args.model_revision).sha
    dataset_revision = api.dataset_info(args.dataset, revision=args.dataset_revision).sha
    cache = args.data_dir / "hf-cache"
    source = Path(
        hf_hub_download(
            args.dataset,
            "data/cot-00000-of-00001.parquet",
            repo_type="dataset",
            revision=dataset_revision,
            cache_dir=cache,
        )
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, revision=model_revision, cache_dir=cache, trust_remote_code=True
    )
    dataset = _dataset(source, tokenizer, args.seed)
    from relax.engine.sft.dataset import chat_template, qwen_chat_template_patch, streaming

    rendering = {
        Path(module.__file__).name: file_digest(Path(module.__file__))
        for module in (chat_template, qwen_chat_template_patch, streaming)
    }
    identity = {
        "model": args.model,
        "model_revision": model_revision,
        "dataset": args.dataset,
        "dataset_revision": dataset_revision,
        "source_sha256": file_digest(source),
        "rendering": rendering,
    }
    metadata_path = args.data_dir / f"rendered-{digest(identity)}.json"
    if metadata_path.exists():
        rows = json.loads(metadata_path.read_text())
    else:

        def render(index: int) -> dict:
            return _sample_metadata(dataset._process_one(index))

        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            rows = []
            for row in pool.map(render, range(len(dataset))):
                rows.append(row)
                if len(rows) % 1000 == 0:
                    logger.info("Rendered %s/%s real samples", len(rows), len(dataset))
        write_json(metadata_path, rows)
    output = args.data_dir / args.model.split("/")[-1] / f"{args.recipe}-gbs{args.batch_size}-cap{args.capacity}"
    settings = {
        **identity,
        "recipe": args.recipe,
        "seed": args.seed,
        "batch_size": args.batch_size,
        "capacity": args.capacity,
        "steps": args.steps,
    }
    if (output / "manifest.json").exists():
        existing = json.loads((output / "manifest.json").read_text())
        if (
            any(existing[key] != value for key, value in settings.items())
            or file_digest(output / "samples.jsonl") != existing["data_sha256"]
        ):
            raise ValueError("Existing frozen workload differs; use a separate DATA_DIR")
        logger.info("Reusing verified frozen workload: %s", output)
        return output
    batches = choose_batches(rows, args.recipe, args.steps, args.batch_size, args.capacity, args.seed)
    selected = [i for batch in batches for i in batch]
    order = loader_order(len(selected), args.seed)
    output_rows: list[dict | None] = [None] * len(selected)
    for source_index, physical_index in zip(selected, order, strict=True):
        raw = dataset.reader[source_index]
        output_rows[physical_index] = {
            "problem": raw["problem"],
            "generated_solution": raw["generated_solution"],
            "metadata": {"original_row_index": source_index},
        }
    output.mkdir(parents=True, exist_ok=False)
    path = output / "samples.jsonl"
    with path.open("w") as stream:
        for row in output_rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    # Verify the inverse permutation through the actual production reader/rendering path.
    frozen = _dataset(path, tokenizer, args.seed)
    frozen.shuffle(0)
    batch_records = []
    for step, original_indices in enumerate(batches):
        samples, crossed = frozen.get_batch(args.batch_size)
        physical_indices = order[step * args.batch_size : (step + 1) * args.batch_size]
        if crossed or [sample.source_idx for sample in samples] != physical_indices:
            raise ValueError("Frozen file did not reproduce its declared optimizer batch order")
        actual = [_sample_metadata(sample) for sample in samples]
        if actual != [rows[i] for i in original_indices]:
            raise ValueError("Frozen file changed tokens or loss masks")
        batch_records.append(
            {
                "step": step,
                "original_rows": original_indices,
                "physical_rows": physical_indices,
                "source_ids_sha256": hashlib.sha256(json.dumps(physical_indices).encode()).hexdigest(),
                "input_tokens": sum(row["length"] for row in actual),
                "effective_tokens": sum(row["effective_tokens"] for row in actual),
                "samples": actual,
            }
        )
    lengths = sorted(rows[i]["length"] for i in selected)
    manifest = {
        "version": 1,
        **identity,
        "recipe": args.recipe,
        "seed": args.seed,
        "batch_size": args.batch_size,
        "capacity": args.capacity,
        "steps": args.steps,
        "data_file": path.name,
        "data_sha256": file_digest(path),
        "batches": batch_records,
        "length_distribution": {
            "min": lengths[0],
            "p50": lengths[len(lengths) // 2],
            "p95": lengths[int(len(lengths) * 0.95)],
            "max": lengths[-1],
        },
        "sampling": "without replacement; exact production rendering; inverse epoch-0 loader permutation",
    }
    write_json(output / "manifest.json", manifest)
    logger.info("Frozen workload: %s", output)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-revision", default="main")
    parser.add_argument("--dataset", default="unsloth/OpenMathReasoning-mini")
    parser.add_argument("--dataset-revision", default="main")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--recipe", choices=("stratified", "natural"), required=True)
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--capacity", type=int, default=24576)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--workers", type=int, default=4)
    freeze(parser.parse_args())


if __name__ == "__main__":
    main()
