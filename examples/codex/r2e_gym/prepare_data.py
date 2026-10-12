# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Convert R2E-Gym train or SWE-Bench-Lite eval Parquet to prompt-data
JSONL."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


_FIELDS = {
    "train": ("problem_statement", "docker_image", "expected_output_json"),
    "eval": ("problem_statement", "docker_image", "instance_id", "repo", "version", "FAIL_TO_PASS", "PASS_TO_PASS"),
}


def _prepared_row(row: dict[str, Any], *, sif_dir: Path, split: str = "train") -> dict[str, Any]:
    fields = _FIELDS[split]
    problem, docker_image = row["problem_statement"], row["docker_image"]
    sif_name = f"{docker_image.replace('/', '__').replace(':', '__')}.sif"
    if not (sif_dir / sif_name).is_file():
        raise FileNotFoundError(f"missing R2E SIF: {sif_dir / sif_name}")
    metadata = {field: row[field] for field in fields[1:]}
    if split == "eval":
        for field in ("FAIL_TO_PASS", "PASS_TO_PASS"):
            value = metadata[field]
            metadata[field] = json.loads(value) if isinstance(value, str) else value
    return {"input": [{"role": "user", "content": problem}], "metadata": {"r2e": metadata}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--sif-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "eval"), default="train")
    args = parser.parse_args()
    pattern = "train-*.parquet" if args.split == "train" else "test-*.parquet"
    files = sorted(args.data_dir.glob(pattern))

    import pyarrow.dataset as ds

    dataset = ds.dataset([str(path) for path in files], format="parquet")
    table = dataset.scanner(columns=list(_FIELDS[args.split])).to_table()

    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if not table.num_rows:
        raise ValueError("R2E conversion produced no rows")
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(
        "".join(
            json.dumps(_prepared_row(row, sif_dir=args.sif_dir, split=args.split), ensure_ascii=False) + "\n"
            for row in table.to_pylist()
        ),
        encoding="utf-8",
    )
    temporary.replace(output)
    print(f"Wrote {table.num_rows} Codex R2E rows to {output}")


if __name__ == "__main__":
    main()
