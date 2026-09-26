#!/usr/bin/env python3
# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import argparse
import json
from pathlib import Path
from typing import Any


DATASETS = {
    "train": ("R2E-Gym-Lite", "data/train-*.parquet"),
    "eval": ("SWE-Bench-Lite", "data/test-*.parquet"),
}


def image_key(docker_image: str) -> str:
    return docker_image.replace("/", "__").replace(":", "__")


def sif_name(docker_image: str) -> str:
    return f"{image_key(docker_image)}.sif"


def _dataset_files(data_path: Path, mode: str) -> list[str]:
    directory, pattern = DATASETS[mode]
    files = sorted((data_path / directory).glob(pattern))
    if not files:
        raise FileNotFoundError(f"No parquet files matched {data_path / directory / pattern}")
    return [str(path) for path in files]


def required_images(data_path: Path, modes: list[str]) -> set[str]:
    import pyarrow.dataset as ds

    images: set[str] = set()
    for mode in modes:
        dataset = ds.dataset(_dataset_files(data_path, mode), format="parquet")
        if "docker_image" not in dataset.schema.names:
            raise KeyError(f"Dataset mode={mode} does not contain docker_image")
        table = dataset.scanner(columns=["docker_image"]).to_table()
        images.update(str(value) for value in table["docker_image"].to_pylist() if value)
    return images


def inspect_sifs(data_path: Path, sif_dir: Path, modes: list[str]) -> dict[str, Any]:
    images = sorted(required_images(data_path, modes))
    existing = {path.name for path in sif_dir.glob("*.sif")}
    missing = [image for image in images if sif_name(image) not in existing]
    return {
        "data_path": str(data_path),
        "sif_dir": str(sif_dir),
        "modes": modes,
        "required_count": len(images),
        "existing_sif_count": len(existing),
        "missing_count": len(missing),
        "missing_images": missing,
        "missing_sifs": [sif_name(image) for image in missing],
    }


def _print_summary(report: dict[str, Any], preview: int) -> None:
    print(
        "R2E SIF coverage: "
        f"required={report['required_count']} "
        f"existing_sifs={report['existing_sif_count']} "
        f"missing={report['missing_count']} "
        f"data_path={report['data_path']} "
        f"sif_dir={report['sif_dir']}"
    )
    for image in report["missing_images"][:preview]:
        print(f"missing: {image} -> {sif_name(image)}")
    remaining = report["missing_count"] - min(preview, report["missing_count"])
    if remaining > 0:
        print(f"... {remaining} more missing SIFs")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--sif-dir", required=True)
    parser.add_argument("--mode", choices=["all", "train", "eval"], default="all")
    parser.add_argument(
        "--output",
        choices=["summary", "json", "missing-images", "missing-sifs"],
        default="summary",
    )
    parser.add_argument("--preview", type=int, default=20)
    parser.add_argument("--fail-if-missing", action="store_true")
    args = parser.parse_args()

    modes = list(DATASETS) if args.mode == "all" else [args.mode]
    report = inspect_sifs(Path(args.data_path), Path(args.sif_dir), modes)

    if args.output == "json":
        print(json.dumps(report, ensure_ascii=False, indent=2))
    elif args.output == "missing-images":
        print("\n".join(report["missing_images"]))
    elif args.output == "missing-sifs":
        print("\n".join(report["missing_sifs"]))
    else:
        _print_summary(report, args.preview)

    if args.fail_if_missing and report["missing_count"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
