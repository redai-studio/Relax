# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Freeze original public SFT rows and production-tokenized length statistics.

Selection: seeded permutation of physical source rows, retaining the first N
whose unmodified production rendering fits the configured length interval.
No truncation, repetition or length sorting is used.
"""

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any

import torch


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def freeze_workload(
    source: Path,
    model: Path,
    output: Path,
    *,
    source_revision: str,
    model_revision: str,
    samples: int,
    seed: int = 1605,
    min_length: int = 16,
    max_length: int = 8192,
    input_key: str = "problem",
    label_key: str = "generated_solution",
) -> dict[str, Any]:
    from transformers import AutoTokenizer

    from relax.engine.sft.dataset.streaming import SFTStreamingDataset

    if samples <= 0 or not 1 <= min_length <= max_length:
        raise ValueError("Require positive sample count and 1 <= min_length <= max_length")
    if output.exists():
        raise ValueError("Output directory already exists; frozen workloads must not be overwritten")
    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)
    dataset = SFTStreamingDataset(
        str(source),
        tokenizer=tokenizer,
        prompt_key=input_key,
        label_key=label_key,
        prefetch_max_cached=0,
        capacity=None,
        oversize_strategy="keep",
        seed=seed,
    )
    indexes = list(range(len(dataset.reader)))
    random.Random(seed).shuffle(indexes)
    retained, lengths, mask_sums = [], [], []
    rendered = hashlib.sha256()
    rows = []
    scanned = 0
    try:
        for index in indexes:
            scanned += 1
            sample = dataset.get_batch_by_indices([index])[0]
            if not min_length <= sample.total_length <= max_length:
                continue
            retained.append(index)
            lengths.append(sample.total_length)
            mask_sums.append(int(sample.loss_mask.sum()))
            rendered.update(sample.total_length.to_bytes(8, "little"))
            rendered.update(sample.tokens.to(dtype=torch.int64).contiguous().numpy().tobytes())
            rendered.update(sample.loss_mask.to(dtype=torch.float32).contiguous().numpy().tobytes())
            rows.append(dataset.reader[index])
            if len(rows) == samples:
                break
    finally:
        dataset.stop()
    if len(rows) != samples:
        raise ValueError(f"Only {len(rows)} eligible rows found, requested {samples}")
    output.mkdir(parents=True)
    data_path = output / "train.jsonl"
    with data_path.open("w") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    tokenizer_files = {
        path.name: sha256_file(path)
        for path in sorted(model.iterdir())
        if path.is_file()
        and (path.suffix in (".json", ".jinja", ".txt"))
        and path.name != "model.safetensors.index.json"
    }
    ordered = sorted(lengths)
    manifest = dict(
        source=str(source),
        source_revision=source_revision,
        source_sha256=sha256_file(source),
        model=str(model),
        model_revision=model_revision,
        tokenizer_files=tokenizer_files,
        seed=seed,
        selection=__doc__,
        min_length=min_length,
        max_length=max_length,
        input_key=input_key,
        label_key=label_key,
        scanned=scanned,
        source_rows=len(indexes),
        retained_source_indexes=retained,
        samples=samples,
        total_lengths=lengths,
        loss_mask_sums=mask_sums,
        total_tokens=sum(lengths),
        supervised_tokens=sum(mask_sums),
        rendered_tokens_and_masks_sha256=rendered.hexdigest(),
        train_sha256=sha256_file(data_path),
        length_stats=dict(
            min=ordered[0], p50=ordered[len(ordered) // 2], p90=ordered[int((len(ordered) - 1) * 0.9)], max=ordered[-1]
        ),
    )
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--samples", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=1605)
    parser.add_argument("--min-length", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=8192)
    parser.add_argument("--input-key", default="problem")
    parser.add_argument("--label-key", default="generated_solution")
    args = parser.parse_args()
    manifest = freeze_workload(**vars(args))
    print(json.dumps({key: manifest[key] for key in ("samples", "total_tokens", "length_stats", "train_sha256")}))


if __name__ == "__main__":
    main()
