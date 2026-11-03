# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""CPU regression tests for the standalone cross-configuration comparison
tool."""

import copy
import hashlib
import json

import pytest
import torch
from compare_snapshots import compare_snapshots, compare_tensor_shards, load_snapshots


def snapshots(world=8):
    return [
        {
            "schema_version": 1,
            "rank": rank,
            "world_size": world,
            "local_samples": 1,
            "loss_scale": 1.0,
            "samples_sha256": f"input-{rank}",
            "rng_initial_sha256": f"rng-before-{rank}",
            "rng_after_reference_sha256": f"rng-after-{rank}",
            "optimizer_initial_sha256": "optimizer",
            "optimizer_type": "AdamW",
            "optimizer_parameter_names": [["weight", "zero"]],
            "global_loss": 1.0,
            "initial_parameters": {
                "weight": {
                    "sha256": f"initial-{rank}",
                    "dtype": "torch.float32",
                    "local_shape": [2],
                    "global_shape": [2 * world],
                    "placements": ["S(0)"],
                },
                "zero": {
                    "sha256": "initial-zero",
                    "dtype": "torch.float32",
                    "local_shape": [0],
                    "global_shape": [0],
                    "placements": ["S(0)"],
                },
            },
            "gradients": {"weight": torch.tensor([rank + 1.0, -rank - 1.0]), "zero": torch.empty(0)},
            "parameter_updates": {"weight": torch.tensor([0.1, -0.1]), "zero": torch.empty(0)},
        }
        for rank in range(world)
    ]


def test_eight_rank_comparison_checks_tensor_differences_not_equal_norms():
    reference = snapshots()
    actual = copy.deepcopy(reference)
    assert compare_snapshots(reference, actual, 0)["passed"]
    actual[7]["gradients"]["weight"].neg_()
    report = compare_snapshots(reference, actual)
    row = report["gradients"]["weight"]
    assert row["reference_l2"] == row["actual_l2"]
    assert row["error_l2"] > 0 and row["max_absolute_error"] == 16
    assert not report["passed"]
    assert report["parameter_updates"]["weight"]["passed"]


@pytest.mark.parametrize(
    "reference,actual,expected",
    [([0.0], [0.0], True), ([0.0], [1e-30], False), ([1.0], [1.005], True), ([1.0], [float("nan")], False)],
)
def test_zero_norm_tolerance_and_finite_checks(reference, actual, expected):
    report = compare_tensor_shards([torch.tensor(reference)], [torch.tensor(actual)], 0.01)
    assert report["passed"] is expected
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize("field", ["rng_initial_sha256", "rng_after_reference_sha256", "samples_sha256"])
def test_different_effective_input_or_rng_invalidates_numerically_equal_results(field):
    reference = snapshots()
    actual = copy.deepcopy(reference)
    actual[4][field] += "changed"
    report = compare_snapshots(reference, actual)
    assert not report["passed"]
    assert report["precondition_mismatches"] == [f"rank 4: {field}"]


def test_loss_zero_reference_requires_exact_zero():
    reference = snapshots()
    actual = copy.deepcopy(reference)
    for row in reference:
        row["global_loss"] = 0.0
    for row in actual:
        row["global_loss"] = 1e-30
    assert not compare_snapshots(reference, actual)["loss"]["passed"]


def test_loader_requires_complete_manifest_and_checks_each_payload(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_snapshots(tmp_path)
    rows = []
    for snapshot in snapshots():
        filename = f"reference-rank{snapshot['rank']:05d}.pt"
        path = tmp_path / filename
        torch.save(snapshot, path)
        rows.append(
            {
                "rank": snapshot["rank"],
                "file": filename,
                "error": None,
                "bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    manifest = {"schema_version": 1, "world_size": 8, "files": rows}
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    assert compare_snapshots(load_snapshots(tmp_path), load_snapshots(tmp_path), 0)["passed"]
    path = tmp_path / rows[3]["file"]
    path.write_bytes(path.read_bytes() + b"corrupt")
    with pytest.raises(ValueError, match="checksum mismatch"):
        load_snapshots(tmp_path)
    manifest["files"] = rows[:-1]
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="every rank"):
        load_snapshots(tmp_path)
