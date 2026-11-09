# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Prepare immutable 32K comparison splits without trimming trajectories."""

import argparse
import hashlib
import json
import math
import random
import zipfile
from collections import Counter
from pathlib import Path

from common import (
    DATA_REVISION,
    MODEL_REVISION,
    REFERENCE_REVISION,
    SOURCE_FILES,
    canonical_sample,
    digest,
    extract_answer,
    extract_calls,
    parse_tools,
    step_cases,
    write_json,
    write_jsonl,
)


def unpack_images(directory: Path) -> None:
    archive = directory / "images.zip"
    with zipfile.ZipFile(archive) as bundle:
        for entry in bundle.infolist():
            target = (directory / entry.filename).resolve()
            if not target.is_relative_to(directory.resolve()):
                raise ValueError(f"Unsafe archive path: {entry.filename}")
        bundle.extractall(directory)


def prepare_image(path: Path, output: Path) -> tuple[str, str]:
    from PIL import Image

    with Image.open(path) as source:
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
        target = output / "images" / f"{pixel_hash}.png"
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            image.save(target)
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
    if messages[1]["role"] != "user" or messages[-1]["role"] != "assistant":
        raise ValueError("Expected user-first, assistant-last trajectory")
    calls = sum(len(extract_calls(m["content"])) for m in messages if m["role"] == "assistant")
    if not calls or extract_answer(messages[-1]["content"]) is None:
        raise ValueError("Expected a tool trajectory ending in one <response> answer")
    paths, hashes = [], []
    for value in row["images"]:
        path = (directory / value).resolve()
        if not path.is_relative_to(directory.resolve()):
            raise ValueError(f"Image outside its source domain: {value}")
        prepared, image_hash = prepare_image(path, output)
        paths.append(prepared)
        hashes.append(image_hash)
    if sum(m["content"].count("<image>") for m in messages) != len(paths):
        raise ValueError("Image placeholders do not match media list")
    tools = parse_tools(row["tools"])
    identity = digest([messages, hashes, tools])
    return {
        "id": identity,
        "messages": messages,
        "images": paths,
        "tools": tools,
        "metadata": {"domain": domain, "image_hashes": hashes, "tool_calls": calls},
    }


def select_splits(rows: list[dict], sizes: list[int], seed: int) -> dict[str, list[dict]]:
    """Disjoint source images/questions; every split has real long, multi-call rows."""
    rng = random.Random(seed)
    rows = list(rows)
    rng.shuffle(rows)
    # Reserve long/complex rows first, rather than letting short rows exhaust a split.
    used_images, used_questions = set(), set()
    splits = {name: [] for name in ("train", "val", "test")}
    for stage in ("24k-32k", "16k-24k", "remaining"):
        for name, size in zip(splits, sizes, strict=True):
            target = math.ceil(size / 4) if stage == "24k-32k" else size // 2 if stage == "16k-24k" else size
            for row in rows:
                if len(splits[name]) >= target:
                    break
                meta = row["metadata"]
                lower, upper = (24576, 32769) if stage == "24k-32k" else (16384, 24576)
                if stage != "remaining" and not (lower <= meta["prompt_tokens"] < upper and meta["tool_calls"] >= 3):
                    continue
                hashes = set(meta["image_hashes"])
                question = digest(row["messages"][1]["content"])
                if hashes & used_images or question in used_questions:
                    continue
                splits[name].append(row)
                used_images.update(hashes)
                used_questions.add(question)
            if len(splits[name]) != target:
                raise ValueError(f"Insufficient disjoint {stage} / multi-call rows for {name}")
    return splits


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--domains", nargs="+", choices=list(SOURCE_FILES), default=list(SOURCE_FILES))
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--train-size", type=int, default=256)
    parser.add_argument("--val-size", type=int, default=32)
    parser.add_argument("--test-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    sizes = [args.train_size, args.val_size, args.test_size]
    if any(size < 8 or size % 8 for size in sizes):
        parser.error("Split sizes must be positive multiples of 8")
    if args.output.exists():
        parser.error("Use a fresh --output directory; existing splits must remain immutable")
    from PIL import Image
    from transformers import AutoProcessor

    from relax.engine.sft.dataset.chat_template import render_to_text

    if args.download:
        from huggingface_hub import snapshot_download

        snapshot_download(
            "OpenSearch-VL/Search-VL-SFT-36K", repo_type="dataset", revision=DATA_REVISION,
            allow_patterns=[item for domain in args.domains for item in
                            (f"{domain}/{SOURCE_FILES[domain]}", f"{domain}/images.zip")],
            local_dir=args.source,
        )
        write_json(args.source / "source-revision.json", {"revision": DATA_REVISION})
    provenance = json.loads((args.source / "source-revision.json").read_text())
    if provenance["revision"] != DATA_REVISION:
        raise ValueError("Source revision differs from the pinned recipe")
    processor = AutoProcessor.from_pretrained(args.model, local_files_only=True)
    processor.image_processor.min_pixels = 1024
    processor.image_processor.max_pixels = 262144
    rows, rejected = [], []
    counters = Counter()
    for domain in args.domains:
        directory = args.source / domain
        if args.download:
            unpack_images(directory)
        raw = json.loads((directory / SOURCE_FILES[domain]).read_text())
        for index, source in enumerate(raw):
            counters["source_rows"] += 1
            try:
                row = normalize(source, directory, args.output, domain)
                text = render_to_text(
                    canonical_sample(row), tokenizer=processor.tokenizer,
                    apply_chat_template_kwargs={"preserve_thinking": True},
                )
                images = [Image.open(path).convert("RGB") for path in row["images"]]
                inputs = processor(text=[text], images=images, return_tensors=None)
                length = len(inputs["input_ids"][0])
                row["metadata"]["expanded_tokens"] = length
                marker = "<|im_start|>assistant\n"
                boundary = text.rfind(marker)
                if boundary < 0:
                    raise ValueError("Missing final assistant boundary")
                prefix = text[:boundary + len(marker)]
                count = sum(m["content"].count("<image>") for m in row["messages"][:-1])
                row["metadata"]["prompt_tokens"] = len(processor(
                    text=[prefix], **({"images": images[:count]} if count else {}), return_tensors=None,
                )["input_ids"][0])
                if length > 32768:
                    rejected.append({"domain": domain, "index": index, "reason": "over_32k", "tokens": length})
                    continue
                rows.append(row)
            except (ValueError, KeyError, OSError) as exc:
                rejected.append({"domain": domain, "index": index, "reason": str(exc)})
    write_json(args.output / "rejected.json", rejected)
    splits = select_splits(rows, sizes, args.seed)
    stats = {}
    for name, selected in splits.items():
        write_jsonl(args.output / f"{name}.jsonl", selected)
        stats[name] = {
            "count": len(selected), "ids": [row["id"] for row in selected],
            "long_24k_32k": sum(row["metadata"]["prompt_tokens"] >= 24576 for row in selected),
            "length_buckets": dict(Counter(
                "24k-32k" if r["metadata"]["prompt_tokens"] >= 24576 else
                "16k-24k" if r["metadata"]["prompt_tokens"] >= 16384 else "under16k" for r in selected)),
            "tool_calls": dict(Counter(call["name"] for r in selected for m in r["messages"]
                                       if m["role"] == "assistant" for call in extract_calls(m["content"]))),
            "domain_counts": dict(Counter(row["metadata"]["domain"] for row in selected)),
        }
    write_jsonl(args.output / "test_steps.jsonl", step_cases(splits["test"]))
    write_json(args.output / "manifest.json", {
        "dataset_revision": DATA_REVISION, "reference_revision": REFERENCE_REVISION,
        "model_revision": MODEL_REVISION, "seed": args.seed, "max_length": 32768,
        "image_max_pixels": 262144, "source_counts": dict(counters), "rejected_count": len(rejected),
        "splits": stats, "test_sha256": digest(splits["test"]),
    })


if __name__ == "__main__":
    main()
