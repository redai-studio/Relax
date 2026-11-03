#!/usr/bin/env python3
# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Compare complete, committed per-rank reference snapshots across SFT
configurations."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    import torch


def load_snapshots(directory: str | Path, expected_world_size: int = 8) -> list[dict[str, Any]]:
    import torch

    directory = Path(directory)
    if not (directory / "manifest.json").exists() and (directory / "reference-snapshots").is_dir():
        directory /= "reference-snapshots"
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest.get("schema_version") != 1 or manifest.get("world_size") != expected_world_size:
        raise ValueError("Snapshot schema/world size differs from the requested comparison")
    files = manifest["files"]
    if len(files) != expected_world_size or sorted(row["rank"] for row in files) != list(range(expected_world_size)):
        raise ValueError("Snapshot manifest does not cover every rank exactly once")
    snapshots = []
    for row in sorted(files, key=lambda item: item["rank"]):
        if row.get("error") is not None or row["file"] != f"reference-rank{row['rank']:05d}.pt":
            raise ValueError("Invalid rank snapshot descriptor")
        path = directory / row["file"]
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        if path.stat().st_size != row["bytes"] or digest.hexdigest() != row["sha256"]:
            raise ValueError(f"Snapshot size/checksum mismatch: {path}")
        snapshot = torch.load(path, map_location="cpu", weights_only=True)
        if (
            snapshot.get("schema_version") != 1
            or snapshot.get("rank") != row["rank"]
            or snapshot.get("world_size") != expected_world_size
        ):
            raise ValueError(f"Snapshot rank/schema/world size mismatch: {path}")
        snapshots.append(snapshot)
    return snapshots


def compare_tensor_shards(
    reference: list[torch.Tensor], actual: list[torch.Tensor], relative_tolerance: float
) -> dict[str, Any]:
    """Sum squared elementwise differences, never differences between tensor
    norms."""
    import torch

    valid = len(reference) == len(actual)
    reference_squared = actual_squared = difference_squared = maximum = 0.0
    elements = 0
    for ref, value in zip(reference, actual, strict=True):
        if (
            not isinstance(ref, torch.Tensor)
            or not isinstance(value, torch.Tensor)
            or ref.shape != value.shape
            or ref.dtype != torch.float32
            or value.dtype != torch.float32
            or not bool(torch.isfinite(ref).all() and torch.isfinite(value).all())
        ):
            valid = False
            continue
        ref64, value64 = ref.double(), value.double()
        difference = value64 - ref64
        reference_squared += float(ref64.square().sum())
        actual_squared += float(value64.square().sum())
        difference_squared += float(difference.square().sum())
        elements += ref.numel()
        if difference.numel():
            maximum = max(maximum, float(difference.abs().max()))
    reference_norm, actual_norm, difference_norm = map(
        math.sqrt, (reference_squared, actual_squared, difference_squared)
    )
    valid = valid and all(math.isfinite(value) for value in (reference_norm, actual_norm, difference_norm, maximum))
    passed = valid and (
        actual_norm == 0 if reference_norm == 0 else difference_norm <= relative_tolerance * reference_norm
    )
    return {
        "finite_matching_fp32_shapes": valid,
        "elements": elements,
        "reference_l2": reference_norm if valid else None,
        "actual_l2": actual_norm if valid else None,
        "error_l2": difference_norm if valid else None,
        "relative_l2_error": difference_norm / reference_norm if valid and reference_norm else 0.0 if passed else None,
        "max_absolute_error": maximum if valid else None,
        "passed": passed,
    }


def compare_snapshots(
    reference: list[dict[str, Any]], actual: list[dict[str, Any]], relative_tolerance: float = 0.01
) -> dict[str, Any]:
    if not 0 <= relative_tolerance <= 0.01:
        raise ValueError("Relative tolerance must be between zero and one percent")
    if not reference or len(reference) != len(actual):
        raise ValueError("Reference/actual snapshot rank counts differ or are empty")
    mismatches = []
    names = sorted(reference[0]["initial_parameters"])
    for rank, (ref, value) in enumerate(zip(reference, actual, strict=True)):
        for field in (
            "rank",
            "world_size",
            "local_samples",
            "loss_scale",
            "samples_sha256",
            "rng_initial_sha256",
            "rng_after_reference_sha256",
            "optimizer_initial_sha256",
            "optimizer_type",
            "optimizer_parameter_names",
            "initial_parameters",
        ):
            if ref[field] != value[field]:
                mismatches.append(f"rank {rank}: {field}")
        for side, snapshot in (("reference", ref), ("actual", value)):
            for field in ("initial_parameters", "gradients", "parameter_updates"):
                if sorted(snapshot[field]) != names:
                    raise ValueError(f"rank {rank} {side}: {field} parameter names differ")
            for name in names:
                metadata = snapshot["initial_parameters"][name]
                for field in ("gradients", "parameter_updates"):
                    tensor = snapshot[field][name]
                    if list(tensor.shape) != metadata["local_shape"] or str(tensor.dtype) != metadata["dtype"]:
                        raise ValueError(f"rank {rank} {side}: {field} {name} differs from its initial shard layout")
        if ref["global_loss"] != reference[0]["global_loss"] or value["global_loss"] != actual[0]["global_loss"]:
            mismatches.append(f"rank {rank}: inconsistent globally reduced loss")
    reference_loss, actual_loss = reference[0]["global_loss"], actual[0]["global_loss"]
    finite = math.isfinite(reference_loss) and math.isfinite(actual_loss)
    loss_error = abs(actual_loss - reference_loss) if finite else None
    loss = {
        "reference": reference_loss if math.isfinite(reference_loss) else None,
        "actual": actual_loss if math.isfinite(actual_loss) else None,
        "absolute_error": loss_error,
        "finite": finite,
        "passed": finite and loss_error <= relative_tolerance * abs(reference_loss),
    }
    comparisons = {
        field: {
            name: compare_tensor_shards(
                [snapshot[field][name] for snapshot in reference],
                [snapshot[field][name] for snapshot in actual],
                relative_tolerance,
            )
            for name in names
        }
        for field in ("gradients", "parameter_updates")
    }
    return {
        "schema_version": 1,
        "world_size": len(reference),
        "relative_tolerance": relative_tolerance,
        "parameter_count": len(names),
        "compatible_initial_state_and_inputs": not mismatches,
        "precondition_mismatches": mismatches,
        "loss": loss,
        **comparisons,
        "max_absolute_error": {
            field: max((row["max_absolute_error"] or 0.0 for row in rows.values()), default=0.0)
            if all(row["finite_matching_fp32_shapes"] for row in rows.values())
            else None
            for field, rows in comparisons.items()
        },
        "passed": not mismatches
        and loss["passed"]
        and all(row["passed"] for rows in comparisons.values() for row in rows.values()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--actual", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--world-size", type=int, default=8)
    parser.add_argument("--rtol", type=float, default=0.01)
    args = parser.parse_args()
    report = compare_snapshots(
        load_snapshots(args.reference, args.world_size), load_snapshots(args.actual, args.world_size), args.rtol
    )
    report.update(reference_directory=str(args.reference), actual_directory=str(args.actual))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    temporary.replace(args.output)
    print(json.dumps({"passed": report["passed"], "max_absolute_error": report["max_absolute_error"]}))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
