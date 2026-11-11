# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Prepare complete tool trajectories; leave sequence-length policy to
training."""

import argparse
import hashlib
import json
import math
import multiprocessing
import os
import random
import time
import zipfile
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from pathlib import Path

from common import (
    DATA_REVISION,
    REFERENCE_REVISION,
    SOURCE_FILES,
    digest,
    extract_answer,
    extract_calls,
    parse_tools,
    write_json,
    write_jsonl,
)


@lru_cache(maxsize=8)
def image_bundle(directory: Path) -> zipfile.ZipFile | None:
    archive = directory / "images.zip"
    return zipfile.ZipFile(archive) if archive.is_file() else None


@lru_cache(maxsize=16)
def image_output_dir(output: Path) -> Path:
    directory = output / "images"
    directory.mkdir(parents=True, exist_ok=True)
    return directory


@lru_cache(maxsize=16384)
def prepare_image(path: Path, output: Path, directory: Path) -> tuple[str, str]:
    from PIL import Image

    bundle = image_bundle(directory)
    stream = bundle.open(path.relative_to(directory.resolve()).as_posix()) if bundle is not None else path.open("rb")
    with stream, Image.open(stream) as source:
        image = source.convert("RGB")
        pixel_hash = hashlib.sha256(str(image.size).encode() + image.tobytes()).hexdigest()
        # Match the original recipe's 262144-pixel cap, with Qwen3.5's 32px grid.
        width, height = image.size
        if max(width, height) / min(width, height) > 200:
            raise ValueError(f"Unsupported aspect ratio: {path}")
        w, h = max(32, round(width / 32) * 32), max(32, round(height / 32) * 32)
        if w * h > 262144:
            scale = math.sqrt(width * height / 262144)
            w, h = max(32, math.floor(width / scale / 32) * 32), max(32, math.floor(height / scale / 32) * 32)
        if image.size != (w, h):
            image = image.resize((w, h), Image.Resampling.BICUBIC)
        target = image_output_dir(output) / f"{pixel_hash}.png"
        if not target.exists():
            temporary = target.with_suffix(f".{os.getpid()}.tmp")
            image.save(temporary, format="PNG")
            temporary.replace(target)
    return str(target.resolve()), pixel_hash


def normalize(row: dict, directory: Path, output: Path, domain: str) -> dict:
    roles = {"human": "user", "gpt": "assistant", "observation": "tool"}
    messages = [{"role": "system", "content": row["system"]}]
    for message in row["conversations"]:
        role = roles[message["from"]]
        content = message["value"]
        if not isinstance(content, str) or "<|im_start|>" in content:
            raise ValueError("Expected plain content without injected ChatML boundaries")
        messages.append({"role": role, "content": content})
    if len(messages) < 3 or messages[1]["role"] != "user" or messages[-1]["role"] != "assistant":
        raise ValueError("Expected user-first, assistant-last trajectory")
    calls = sum(len(extract_calls(m["content"])) for m in messages if m["role"] == "assistant")
    if not calls or extract_answer(messages[-1]["content"]) is None:
        raise ValueError("Expected a tool trajectory ending in one <response> answer")
    if sum(m["content"].count("<image>") for m in messages) != len(row["images"]):
        raise ValueError("Image placeholders do not match media list")
    tools = parse_tools(row["tools"])
    paths, hashes = [], []
    for value in row["images"]:
        path = (directory / value).resolve()
        if not path.is_relative_to(directory.resolve()):
            raise ValueError(f"Image outside its source domain: {value}")
        prepared, image_hash = prepare_image(path, output, directory)
        paths.append(prepared)
        hashes.append(image_hash)
    identity = digest([messages, hashes, tools])
    return {
        "id": identity,
        "messages": messages,
        "images": paths,
        "tools": tools,
        "metadata": {"domain": domain, "image_hashes": hashes, "tool_calls": calls},
    }


def select_splits(rows: list[dict], sizes: list[int], seed: int) -> dict[str, list[dict]]:
    """Keep related trajectories together, without discarding repeated train
    images."""
    parents = list(range(len(rows)))

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    owners = {}
    for index, row in enumerate(rows):
        keys = [("image", value) for value in row["metadata"]["image_hashes"]]
        keys.append(("question", digest(row["messages"][1]["content"])))
        for key in keys:
            if key in owners:
                parents[find(index)] = find(owners[key])
            else:
                owners[key] = index
    groups = {}
    for index, row in enumerate(rows):
        groups.setdefault(find(index), []).append(row)
    remaining = list(groups.values())
    random.Random(seed).shuffle(remaining)
    splits = {name: [] for name in ("train", "val", "test")}
    for name, size in zip(("val", "test"), sizes[1:], strict=True):
        unused = []
        for group in remaining:
            if len(splits[name]) + len(group) <= size:
                splits[name].extend(group)
            else:
                unused.append(group)
        if len(splits[name]) != size:
            raise ValueError(f"Cannot form {size} {name} rows without splitting shared images/questions")
        remaining = unused
    splits["train"] = [row for group in remaining for row in group]
    if sizes[0]:
        if len(splits["train"]) < sizes[0]:
            raise ValueError("Insufficient training rows after holding out validation/test groups")
        splits["train"] = splits["train"][: sizes[0]]
    if not splits["train"]:
        raise ValueError("No training rows remain after holding out validation/test groups")
    return splits


def prepare_row(task: tuple) -> tuple[dict | None, dict | None]:
    index, source, directory, output, domain = task
    try:
        return normalize(source, directory, output, domain), None
    except (ValueError, KeyError, OSError) as exc:
        return None, {"domain": domain, "index": index, "reason": str(exc)}


def main() -> None:
    started = time.perf_counter()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", type=Path, help="Legacy argument; model tokenization is deferred to training")
    parser.add_argument("--domains", nargs="+", choices=list(SOURCE_FILES), default=list(SOURCE_FILES))
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--train-size", type=int, default=0, help="0 keeps all training trajectories (default)")
    parser.add_argument("--val-size", type=int, default=32)
    parser.add_argument("--test-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    args = parser.parse_args()
    sizes = [args.train_size, args.val_size, args.test_size]
    if sizes[0] < 0 or any(size < 8 or size % 8 for size in sizes[1:]) or sizes[0] % 8:
        parser.error("Split sizes must be positive multiples of 8; --train-size 0 keeps all rows")
    if args.workers < 1:
        parser.error("--workers must be positive")
    if args.output.exists():
        parser.error("Use a fresh --output directory; existing splits must remain immutable")
    if args.download:
        from huggingface_hub import snapshot_download

        snapshot_download(
            "OpenSearch-VL/Search-VL-SFT-36K",
            repo_type="dataset",
            revision=DATA_REVISION,
            allow_patterns=[
                item
                for domain in args.domains
                for item in (f"{domain}/{SOURCE_FILES[domain]}", f"{domain}/images.zip")
            ],
            local_dir=args.source,
        )
        write_json(args.source / "source-revision.json", {"revision": DATA_REVISION})
    provenance = json.loads((args.source / "source-revision.json").read_text())
    if provenance["revision"] != DATA_REVISION:
        raise ValueError("Source revision differs from the pinned recipe")
    rows, rejected = [], []
    counters = Counter()
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        for domain in args.domains:
            directory = args.source / domain
            raw = json.loads((directory / SOURCE_FILES[domain]).read_text())
            counters["source_rows"] += len(raw)
            tasks = ((index, source, directory, args.output, domain) for index, source in enumerate(raw))
            for row, rejection in pool.map(prepare_row, tasks, chunksize=32):
                if rejection is not None:
                    rejected.append(rejection)
                else:
                    rows.append(row)
    write_json(args.output / "rejected.json", rejected)
    splits = select_splits(rows, sizes, args.seed)
    stats = {}
    for name, selected in splits.items():
        write_jsonl(args.output / f"{name}.jsonl", selected)
        stats[name] = {
            "count": len(selected),
            "ids": [row["id"] for row in selected],
            "tool_calls": dict(
                Counter(
                    call["name"]
                    for r in selected
                    for m in r["messages"]
                    if m["role"] == "assistant"
                    for call in extract_calls(m["content"])
                )
            ),
            "domain_counts": dict(Counter(row["metadata"]["domain"] for row in selected)),
        }
    used_images = {path for selected in splits.values() for row in selected for path in row["images"]}
    for path in (args.output / "images").glob("*.png"):
        if str(path.resolve()) not in used_images:
            path.unlink()
    write_json(
        args.output / "manifest.json",
        {
            "dataset_revision": DATA_REVISION,
            "reference_revision": REFERENCE_REVISION,
            "seed": args.seed,
            "max_length": None,
            "image_max_pixels": 262144,
            "source_counts": dict(counters),
            "rejected_count": len(rejected),
            "splits": stats,
            "test_sha256": digest(splits["test"]),
            "workers": args.workers,
            "elapsed_seconds": time.perf_counter() - started,
        },
    )


if __name__ == "__main__":
    main()
