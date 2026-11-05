#!/usr/bin/env python3
"""Count actual adjacent HF/GPU changes, including zero-change steps.

Uses the existing streaming raw-bit comparator. Shared parameter names are
reported both with their export multiplicity and once per GPU alias group.
Requires completed, fully audited training receipts; never modifies weights.
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

from relax.distributed.weight_sync.provenance import runtime_identity
from relax.distributed.weight_sync.store import atomic_write
from relax.distributed.weight_sync.tensor_diff import compare


def delivered_rank(run: Path, version: int) -> Path:
    receipt = json.loads((run / "versions" / f"{version:06d}.json").read_text())
    if receipt["status"] != "ACTIVE":
        raise ValueError(f"Version {version} is not fully installed")
    member = next(item for item in receipt["metrics"]["installation"]["members"] if item["engine"] == 0)
    check = json.loads(Path(member["readback"][0]["diff"]).read_text())
    if check["equal"] is not True:
        raise ValueError("Training GPU readback did not match the independent HF reference")
    path = Path(check["right"])
    if "delivered" not in path.parts:
        raise ValueError("Expected the saved delivered GPU state before reference reload")
    return path


def valid_comparison(check: dict) -> None:
    if check["missing"] or check["extra"] or check["metadata_mismatches"]:
        raise ValueError("Adjacent versions do not have matching tensor metadata")


def count_changes(run: Path, updates: int, output: Path) -> list[dict]:
    output.mkdir(parents=True, exist_ok=False)
    counter = Path(sys.modules[compare.__module__].__file__).resolve()
    atomic_write(
        output / "runtime.json",
        json.dumps(
            {
                "runtime": runtime_identity(),
                "counter_path": str(counter),
                "counter_sha256": hashlib.sha256(counter.read_bytes()).hexdigest(),
            },
            indent=2,
        ).encode(),
    )
    rows = []
    for version in range(1, updates + 1):
        hf = compare(run / "exports" / f"update-{version - 1:06d}", run / "exports" / f"update-{version:06d}")
        valid_comparison(hf)
        left, right = delivered_rank(run, version - 1), delivered_rank(run, version)
        gpu = compare(left, right)
        valid_comparison(gpu)
        for key in ["compared_elements", "compared_bytes", "mismatch_elements", "mismatch_bytes", "mismatch_bits"]:
            if hf[key] != gpu[key]:
                raise ValueError(f"HF/GPU adjacent change counts disagree: {key}")
        coverage = json.loads((left / "coverage.json").read_text())
        if coverage != json.loads((right / "coverage.json").read_text()) or coverage["persistent_buffer_count"]:
            raise ValueError("Unique parameter counting requires identical coverage without persistent buffers")
        duplicate_names = [name for group in coverage["parameter_aliases"] for name in group[1:]]
        maps = [
            json.loads((path / "model.safetensors.index.json").read_text())["weight_map"] for path in [left, right]
        ]
        unique = {
            key: gpu[key]
            for key in ["compared_elements", "compared_bytes", "mismatch_elements", "mismatch_bytes", "mismatch_bits"]
        }
        duplicates = []
        for name in duplicate_names:
            duplicate = compare(left / maps[0][name], right / maps[1][name])
            valid_comparison(duplicate)
            if duplicate["compared_tensors"] != 1:
                raise ValueError("Expected one tensor per GPU dump file")
            for key in unique:
                unique[key] -= duplicate[key]
            duplicates.append({"name": name, "comparison": duplicate})
        if any(value < 0 for value in unique.values()):
            raise ValueError("Invalid alias subtraction")
        row = {
            "version": version,
            "named_hf": hf,
            "named_gpu": gpu,
            "duplicates": duplicates,
            "unique_parameters": unique["compared_elements"],
            "changed_parameters": unique["mismatch_elements"],
            "changed_percent": 100 * unique["mismatch_elements"] / unique["compared_elements"],
            "unique_bytes": unique["compared_bytes"],
            "changed_bytes": unique["mismatch_bytes"],
            "changed_bits": unique["mismatch_bits"],
        }
        atomic_write(output / f"step-{version:06d}.json", json.dumps(row, indent=2).encode())
        rows.append(row)
        atomic_write(output / "summary.json", json.dumps(rows, indent=2).encode())
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--updates", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.updates < 1:
        raise ValueError("updates must be positive")
    count_changes(args.run, args.updates, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
