# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import argparse
import ast
import hashlib
import io
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image


def convert_row(row):
    problem = row["problem"].strip()
    content = f"<image>{problem}"
    img = row["image"]
    image_field = [img["bytes"]]
    label = row["solution"]

    return {"prompt": [{"role": "user", "content": content}], "image": image_field, "label": label}


def convert_dataset(input_file, output_file):
    df = pd.read_parquet(input_file)
    converted = [convert_row(row) for _, row in df.iterrows()]
    df_out = pd.DataFrame(converted)
    df_out.to_parquet(output_file, index=False)
    print(len(df), len(df_out))


def download_eval(output: Path, mathvista_revision: str, mmmu_revision: str) -> None:
    from huggingface_hub import HfApi, snapshot_download

    output.mkdir(parents=True, exist_ok=True)
    provenance = {}
    for name, repo, revision, patterns in (
        ("mathvista", "AI4Math/MathVista", mathvista_revision, ["data/testmini*.parquet"]),
        ("mmmu", "MMMU/MMMU", mmmu_revision, ["*/validation*.parquet"]),
    ):
        resolved = HfApi().dataset_info(repo, revision=revision).sha
        destination = output / "source" / name
        marker = destination / "source-revision.json"
        expected = {"repo": repo, "revision": resolved}
        if destination.exists() and any(destination.iterdir()):
            if not marker.exists() or json.loads(marker.read_text()) != expected:
                raise ValueError(
                    f"Use a fresh --output-dir directory: {destination} has unknown or different revision data"
                )
        destination.mkdir(parents=True, exist_ok=True)
        # Mark before downloading so an interrupted transfer can resume safely.
        marker.write_text(json.dumps(expected, indent=2) + "\n")
        snapshot_download(
            repo,
            repo_type="dataset",
            revision=resolved,
            allow_patterns=patterns,
            local_dir=destination,
            max_workers=3,
        )
        provenance[name] = {"repo": repo, "revision": resolved, "patterns": patterns}
    (output / "sources.json").write_text(json.dumps(provenance, indent=2) + "\n")


def image_hash(value: bytes) -> str:
    image = Image.open(io.BytesIO(value)).convert("RGB")
    return hashlib.sha256(str(image.size).encode() + image.tobytes()).hexdigest()


def prepare_eval(source: Path, train: Path, output: Path) -> dict[str, Any]:
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
    datasets = []
    for name in report:
        datasets.append(
            {
                "name": name + "_disjoint_local_accuracy",
                "path": str((output / f"{name}_disjoint.parquet").resolve()),
                "input_key": "prompt",
                "label_key": "label",
                "temperature": 0,
                "top_p": 1,
                "top_k": -1,
                "n_samples_per_eval_prompt": 1,
                "max_response_len": 8192,
                "metadata_overrides": {"eval_benchmark": name},
            }
        )
    # JSON is valid YAML and can be loaded by --eval-config without extra dependencies.
    (output / "eval.yaml").write_text(json.dumps({"eval": {"datasets": datasets}}, indent=2) + "\n")
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Prepare OpenR1MM training or MathVista/MMMU evaluation data")
    parser.add_argument("--mode", choices=("train", "download-eval", "prepare-eval"), default="train")
    parser.add_argument("--input-dir", type=Path, help="Original OpenR1MM training parquet (train mode)")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output parquet for train; directory for eval")
    parser.add_argument("--source", type=Path, help="Downloaded eval source directory (default: OUTPUT/source)")
    parser.add_argument("--train", type=Path, help="Converted training parquet for exact image overlap checks")
    parser.add_argument("--mathvista-revision", default="2b6ad69445fbb5695c9b165475e8decdbeb97747")
    parser.add_argument("--mmmu-revision", default="876ce5cb130f7f7e290ce4d9984357737d4db5cf")
    args = parser.parse_args(argv)
    if args.mode == "train":
        if args.input_dir is None:
            parser.error("--input-dir is required in train mode")
        convert_dataset(args.input_dir, args.output_dir)
    elif args.mode == "download-eval":
        download_eval(args.output_dir, args.mathvista_revision, args.mmmu_revision)
    else:
        if args.train is None:
            parser.error("--train is required in prepare-eval mode")
        report = prepare_eval(args.source or args.output_dir / "source", args.train, args.output_dir)
        for name, stats in report.items():
            print(name, "total=", stats["total"], "disjoint=", stats["disjoint"])


if __name__ == "__main__":
    main()
