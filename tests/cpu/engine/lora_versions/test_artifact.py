# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import json
from concurrent.futures import ThreadPoolExecutor

import pytest
import torch
from safetensors.torch import load_file, save_file

from relax.engine.lora_versions.artifact import read_adapter_artifact, write_adapter_artifact
from relax.engine.lora_versions.identity import snapshot_adapter


def frozen(version=1, value=1):
    return snapshot_adapter(
        version,
        {
            "model.layers.0.linear_attn.in_proj_qkvz.lora_A.weight": torch.full((8, 16), value, dtype=torch.bfloat16),
            "model.layers.0.linear_attn.in_proj_qkvz.lora_B.weight": torch.arange(32).reshape(4, 8).float(),
        },
        {"r": 8, "lora_alpha": 16, "target_modules": ["in_proj_qkvz"]},
        base_fingerprint="frozen-base-content",
    )


def test_artifact_roundtrip_preserves_engine_layout_and_identity(tmp_path):
    snapshot = frozen()
    artifact = write_adapter_artifact(snapshot, tmp_path)
    assert artifact == read_adapter_artifact(artifact.path)
    assert artifact.identity == snapshot.identity
    assert artifact.config == snapshot.config
    assert artifact.manifest_json == snapshot.manifest_json
    assert artifact.fingerprints == snapshot.fingerprints
    assert artifact.path.is_absolute()
    assert not hasattr(artifact, "_tensors")
    stored = load_file(str(artifact.path / "adapter_model.safetensors"))
    for name, tensor in snapshot.copy_tensors().items():
        assert torch.equal(stored[name], tensor)
        assert stored[name].dtype == tensor.dtype


def test_repeated_and_concurrent_writes_do_not_replace_a_version(tmp_path):
    snapshot = frozen()
    with ThreadPoolExecutor(max_workers=2) as executor:
        artifacts = list(executor.map(lambda _: write_adapter_artifact(snapshot, tmp_path), range(2)))
    assert artifacts[0] == artifacts[1]
    weight = artifacts[0].path / "adapter_model.safetensors"
    inode = weight.stat().st_ino
    assert write_adapter_artifact(snapshot, tmp_path) == artifacts[0]
    with pytest.raises(ValueError, match="VERSION_CONFLICT"):
        write_adapter_artifact(frozen(value=2), tmp_path)
    assert weight.stat().st_ino == inode
    assert list(tmp_path.iterdir()) == [artifacts[0].path]


@pytest.mark.parametrize("component", ["weights", "config", "manifest", "missing"])
def test_existing_corrupt_artifact_is_rejected_without_repair_or_overwrite(tmp_path, component):
    snapshot = frozen()
    artifact = write_adapter_artifact(snapshot, tmp_path)
    if component == "weights":
        tensors = snapshot.copy_tensors()
        tensors[next(iter(tensors))].zero_()
        save_file(tensors, str(artifact.path / "adapter_model.safetensors"))
    elif component == "config":
        (artifact.path / "adapter_config.json").write_text('{"r": 99}')
    elif component == "manifest":
        path = artifact.path / "manifest.json"
        metadata = json.loads(path.read_text())
        metadata["snapshot"]["base_fingerprint"] = "different-base"
        path.write_text(json.dumps(metadata))
    else:
        (artifact.path / "manifest.json").unlink()
    before = {p.name: p.read_bytes() for p in artifact.path.iterdir()}
    with pytest.raises((ValueError, FileNotFoundError)):
        write_adapter_artifact(snapshot, tmp_path)
    assert {p.name: p.read_bytes() for p in artifact.path.iterdir()} == before


def test_failed_write_never_exposes_partial_version_or_removes_previous_version(tmp_path, monkeypatch):
    import safetensors.torch

    first = write_adapter_artifact(frozen(), tmp_path)

    def interrupted(tensors, filename):
        assert not (tmp_path / "version-2").exists()
        from pathlib import Path

        Path(filename).write_bytes(b"partial tensor data")
        raise OSError("injected write failure")

    monkeypatch.setattr(safetensors.torch, "save_file", interrupted)
    with pytest.raises(OSError, match="injected"):
        write_adapter_artifact(frozen(version=2), tmp_path)
    assert list(tmp_path.iterdir()) == [first.path]
    assert read_adapter_artifact(first.path) == first
