# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Convert downloaded MathVista/MMMU parquet splits; audit train image
overlap."""

import argparse
import ast
import hashlib
import io
import json
import re
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image


def image_hash(value: bytes) -> str:
    image = Image.open(io.BytesIO(value)).convert("RGB")
    return hashlib.sha256(str(image.size).encode() + image.tobytes()).hexdigest()


def prepare(source: Path, train: Path, output: Path) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    train_hashes = set()
    for row in pq.read_table(train, columns=["image"]).to_pylist():
        for value in row["image"]:
            train_hashes.add(image_hash(value))
    report = {}
    schema = pa.schema(
        [
            ("prompt", pa.list_(pa.struct([("role", pa.string()), ("content", pa.string())]))),
            ("image", pa.list_(pa.binary())),
            ("label", pa.string()),
            ("id", pa.string()),
        ]
    )
    for name, pattern in [
        ("mathvista_testmini", "mathvista/data/testmini*.parquet"),
        ("mmmu_validation", "mmmu/*/validation*.parquet"),
    ]:
        records, clean = [], []
        overlap = []
        for path in sorted(source.glob(pattern)):
            for row in pq.read_table(path).to_pylist():
                if name == "mathvista_testmini":
                    images = [row["decoded_image"]["bytes"]]
                    text = "<image>\n" + row["query"]
                    choices = row["choices"] or []
                    spec = dict(
                        answer=row["answer"],
                        choices=choices,
                        answer_type=row["answer_type"],
                        precision=row["precision"],
                    )
                    if choices:
                        spec["correct_index"] = choices.index(row["answer"])
                    key = str(row["pid"])
                else:
                    images = []

                    def replace_image(match: re.Match) -> str:
                        images.append(row[f"image_{match[1]}"]["bytes"])
                        return "<image>"

                    text = re.sub(r"<image\s+(\d+)>", replace_image, row["question"])
                    choices = ast.literal_eval(row["options"]) if row["question_type"] == "multiple-choice" else []
                    # Options can themselves reference images: preserve occurrence order.
                    if choices:
                        text += "\nChoices:\n" + "\n".join(f"{chr(65 + i)}. {v}" for i, v in enumerate(choices))
                        # Question references were already consumed above.
                        text = re.sub(r"<image\s+(\d+)>", replace_image, text)
                    if not images:
                        images = [row[f"image_{i}"]["bytes"] for i in range(1, 8) if row[f"image_{i}"] is not None]
                        text = "<image>\n" * len(images) + text
                    answer = row["answer"]
                    if not choices and answer.startswith("["):
                        answer = ast.literal_eval(answer)
                    spec = dict(answer=answer, choices=choices)
                    if choices:
                        spec["correct_index"] = ord(answer) - ord("A")
                    key = row["id"]
                text += "\nReason step by step, then put only the final answer in <answer>...</answer>."
                if choices:
                    text += " Use the option letter as the final answer."
                assert text.count("<image>") == len(images)
                item = dict(
                    prompt=[dict(role="user", content=text)],
                    image=images,
                    label=json.dumps(spec, ensure_ascii=False),
                    id=key,
                )
                records.append(item)
                if any(image_hash(value) in train_hashes for value in images):
                    overlap.append(key)
                else:
                    clean.append(item)
        assert records and clean
        for suffix, data in [("", records), ("_disjoint", clean)]:
            pq.write_table(pa.Table.from_pylist(data, schema=schema), output / f"{name}{suffix}.parquet")
        report[name] = dict(total=len(records), disjoint=len(clean), train_image_overlap_ids=overlap)
    (output / "overlap-report.json").write_text(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    options = parser.parse_args()
    result = prepare(options.source, options.train, options.output)
    for name, stats in result.items():
        print(name, "total=", stats["total"], "disjoint=", stats["disjoint"])
