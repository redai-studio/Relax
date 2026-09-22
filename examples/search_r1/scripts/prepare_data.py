# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Prepare the upstream Search-R1 parquet files for managed agent sessions."""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for split in ("train", "test"):
        source = pq.read_table(
            args.input_dir / f"{split}.parquet",
            columns=["prompt", "data_source", "extra_info", "golden_answers"],
        )
        metadata = [
            {**extra_info, "answers": answers, "data_source": data_source}
            for extra_info, answers, data_source in zip(
                source.column("extra_info").to_pylist(),
                source.column("golden_answers").to_pylist(),
                source.column("data_source").to_pylist(),
            )
        ]
        prepared = pa.table(
            {
                "prompt": source.column("prompt"),
                "data_source": source.column("data_source"),
                "extra_info": pa.array(metadata),
            }
        )
        pq.write_table(prepared, args.output_dir / f"{split}.parquet", compression="snappy")


if __name__ == "__main__":
    main()
