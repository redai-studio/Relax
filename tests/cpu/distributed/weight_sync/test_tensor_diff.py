# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import json
import struct
from pathlib import Path

import torch
from safetensors.torch import save_file

from relax.distributed.weight_sync.tensor_diff import main


def _run(left: Path, right: Path, result: Path) -> dict:
    status = main([str(left), str(right), "--result", str(result)])
    report = json.loads(result.read_text())
    assert status == (0 if report["equal"] else 1)
    return report


def test_tensor_diff_detects_single_bit_and_special_float_bits(tmp_path):
    left = tmp_path / "left.safetensors"
    right = tmp_path / "right.safetensors"
    # Distinct NaN payload, plus opposite signed zeros; ordinary float equality is insufficient.
    a = torch.tensor([0x7FC00001, 0x00000000, 0x00000001], dtype=torch.int32).view(torch.float32)
    b = torch.tensor([0x7FC00000, -0x80000000, 0x00000000], dtype=torch.int32).view(torch.float32)
    save_file({"weight": a}, left)
    save_file({"weight": b}, right)
    report = _run(left, right, tmp_path / "diff.json")
    assert (report["mismatch_tensors"], report["mismatch_elements"], report["mismatch_bits"]) == (1, 3, 3)
    assert report["first_difference"]["coordinate"] == [0]
    assert report["first_difference"]["left_bits"] == struct.pack("<I", 0x7FC00001).hex()


def test_tensor_diff_rejects_missing_dtype_and_same_inode(tmp_path):
    left = tmp_path / "left.safetensors"
    right = tmp_path / "right.safetensors"
    save_file({"x": torch.ones(2, dtype=torch.int32), "y": torch.ones(1)}, left)
    save_file({"x": torch.ones(2, dtype=torch.int64)}, right)
    report = _run(left, right, tmp_path / "diff.json")
    assert report["missing"] == ["y"]
    assert report["metadata_mismatches"][0]["name"] == "x"
    assert "inode" in _run(left, left, tmp_path / "same.json")["error"]


def test_tensor_diff_sharded_index_and_incomplete_index(tmp_path):
    left = tmp_path / "left"
    right = tmp_path / "right"
    left.mkdir()
    right.mkdir()
    for directory in (left, right):
        save_file({"x": torch.arange(3)}, directory / "a.safetensors")
        save_file({"y": torch.arange(2)}, directory / "b.safetensors")
        (directory / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": {"x": "a.safetensors", "y": "b.safetensors"}})
        )
    assert _run(left, right, tmp_path / "equal.json")["equal"]
    (right / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"x": "a.safetensors"}}))
    assert "Index" in _run(left, right, tmp_path / "bad.json")["error"]
