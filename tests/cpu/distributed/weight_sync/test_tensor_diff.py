# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import json
import random
import struct
from pathlib import Path

import pytest
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


@pytest.mark.parametrize("dtype", [torch.uint8, torch.int16, torch.int32, torch.int64])
def test_tensor_diff_counts_cross_chunk_changes_against_scalar_oracle(tmp_path, monkeypatch, dtype):
    from relax.distributed.weight_sync import tensor_diff

    monkeypatch.setattr(tensor_diff, "_CHUNK_BYTES", 32)
    width = torch.empty((), dtype=dtype).element_size()
    rng = random.Random(1718)
    a = bytearray(rng.randrange(256) for _ in range(257 * width))
    b = a.copy()
    for index in [*range(30, 131, 7), len(b) - 1]:
        b[index] ^= rng.randrange(1, 256)
    left, right = tmp_path / "left.safetensors", tmp_path / "right.safetensors"
    save_file({"x": torch.frombuffer(a, dtype=torch.uint8).view(dtype)}, left)
    save_file({"x": torch.frombuffer(b, dtype=torch.uint8).view(dtype)}, right)
    report = _run(left, right, tmp_path / "diff.json")
    changed = [index for index, (x, y) in enumerate(zip(a, b)) if x != y]
    assert report["mismatch_bytes"] == len(changed)
    assert report["mismatch_elements"] == len({index // width for index in changed})
    assert report["mismatch_bits"] == sum((x ^ y).bit_count() for x, y in zip(a, b))
    first_element = changed[0] // width
    assert report["first_difference"]["coordinate"] == [first_element]
    assert report["first_difference"]["byte_offset"] == changed[0]
    assert report["first_difference"]["left_bits"] == a[first_element * width : (first_element + 1) * width].hex()
    assert report["first_difference"]["right_bits"] == b[first_element * width : (first_element + 1) * width].hex()
