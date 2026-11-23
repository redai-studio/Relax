# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Prepare the pinned Geometry3K integer-answer subset for multimodal GRPO."""

import argparse
import hashlib
import io
import json
import random
import re
from collections import Counter
from pathlib import Path


REPO = "hiyouga/geometry3k"
REVISION = "fd21e533e1e50d0662a2bf7b223e60511bd5f8b7"
SOURCE_ROWS = {"train": 2101, "validation": 300, "test": 601}
PILOT_SEED = 2401
PILOT_SIZE = 64
INSTRUCTION = "\n\nSolve the problem using the diagram. Put the final integer on its own line as Answer: <integer>."


def write_jsonl(path: Path, rows: list[dict]) -> str:
    data = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows).encode()
    temporary = path.with_suffix(".jsonl.tmp")
    temporary.write_bytes(data)
    temporary.replace(path)
    return hashlib.sha256(data).hexdigest()


def prepare(out: Path) -> None:
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download
    from PIL import Image, ImageOps

    images = out / "images"
    images.mkdir(parents=True, exist_ok=True)
    datasets, report = {}, {}
    eval_hashes: set[str] = set()
    # Give the held-out splits priority when excluding repeated diagrams.
    for split in ("test", "validation", "train"):
        source = Path(
            hf_hub_download(
                REPO,
                f"data/{split}-00000-of-00001.parquet",
                repo_type="dataset",
                revision=REVISION,
                local_dir=out / "raw",
            )
        )
        table = pq.read_table(source)
        if table.num_rows != SOURCE_ROWS[split]:
            raise ValueError(f"Unexpected row count for {split}")
        rows, excluded = [], Counter()
        seen: set[tuple[str, str]] = set()
        split_hashes = set()
        for index, row in enumerate(table.to_pylist()):
            question, answer = row["problem"].strip(), row["answer"].strip()
            media = row["images"]
            if len(media) != 1 or question.count("<image>") != 1 or not media[0]["bytes"]:
                raise ValueError(f"Invalid image or placeholder at {split}:{index}")
            with Image.open(io.BytesIO(media[0]["bytes"])) as raw:
                image = ImageOps.exif_transpose(raw).convert("RGB")
            digest = hashlib.sha256(f"{image.width}x{image.height}:".encode() + image.tobytes()).hexdigest()
            split_hashes.add(digest)
            if not re.fullmatch(r"[+-]?\d+", answer):
                excluded["non_integer_answer"] += 1
                continue
            if digest in eval_hashes:
                excluded["diagram_in_held_out_split"] += 1
                continue
            key = (" ".join(question.split()), digest)
            if key in seen:
                excluded["duplicate_question_and_diagram"] += 1
                continue
            seen.add(key)
            file = images / f"{digest}.png"
            if not file.is_file():
                image.save(file, format="PNG")
            rows.append(
                {
                    "prompt": question + INSTRUCTION,
                    "images": [str(file)],
                    "label": str(int(answer)),
                    "metadata": {"source_id": f"{split}:{index}", "image_sha256": digest},
                }
            )
        if split != "train":
            # Include non-integer diagrams in the leakage check as well.
            eval_hashes.update(split_hashes)
        datasets[split] = rows
        report[split] = {
            "source_rows": table.num_rows,
            "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "rows": len(rows),
            "excluded": dict(excluded),
            "sha256": write_jsonl(out / f"{split}.jsonl", rows),
        }

    pilot = random.Random(PILOT_SEED).sample(datasets["validation"], PILOT_SIZE)
    report["validation-pilot"] = {
        "rows": len(pilot),
        "seed": PILOT_SEED,
        "sha256": write_jsonl(out / "validation-pilot.jsonl", pilot),
    }
    manifest = {"dataset": REPO, "revision": REVISION, "splits": report}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    prepare(parser.parse_args().out.resolve())
