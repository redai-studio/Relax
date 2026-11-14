# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Checkpoint content identity covers both flat and Diffusers model layouts."""

import shutil

import pytest

from relax.engine.sft.conditioning import model_fingerprint
from relax.utils.conditioning_cache import build_key, fingerprint_file


def test_flat_model_fingerprint_preserves_previous_identity(tmp_path):
    (tmp_path / "model.safetensors").write_bytes(b"weights")
    (tmp_path / "config.json").write_text('{"dtype":"bfloat16"}')
    (tmp_path / "tokenizer.json").write_text('{"vocab":{"boat":1}}')
    expected = build_key({path.name: fingerprint_file(path) for path in tmp_path.iterdir()})
    assert model_fingerprint(tmp_path) == expected


def test_nested_model_fingerprint_uses_relative_paths_and_actual_contents(tmp_path):
    root = tmp_path / "original"
    for component in ("transformer", "text_encoder", "vae", "tokenizer"):
        directory = root / component
        directory.mkdir(parents=True)
        (directory / "config.json").write_text('{"component":"' + component + '"}')
        if component != "tokenizer":
            (directory / "model.safetensors").write_bytes(component.encode())
    (root / "tokenizer" / "merges.txt").write_text("a b\n")
    reference = model_fingerprint(root)
    copied = tmp_path / "copy"
    shutil.copytree(root, copied)
    assert model_fingerprint(copied) == reference
    for relative in ("text_encoder/model.safetensors", "transformer/config.json", "tokenizer/merges.txt"):
        path = copied / relative
        previous = path.read_bytes()
        path.write_bytes(previous + b"changed")
        assert model_fingerprint(copied) != reference
        path.write_bytes(previous)
        assert model_fingerprint(copied) == reference


def test_fingerprint_ignores_hidden_download_metadata_and_temporary_files(tmp_path):
    (tmp_path / "model.safetensors").write_bytes(b"weights")
    reference = model_fingerprint(tmp_path)
    (tmp_path / ".cache" / "downloads").mkdir(parents=True)
    (tmp_path / ".cache" / "downloads" / "state.json").write_text("{}")
    (tmp_path / "model.safetensors.tmp").write_bytes(b"partial")
    (tmp_path / "README.md").write_text("model card")
    assert model_fingerprint(tmp_path) == reference


def test_fingerprint_requires_real_checkpoint_weights(tmp_path):
    (tmp_path / "config.json").write_text("{}")
    with pytest.raises(ValueError, match="No safetensors checkpoint"):
        model_fingerprint(tmp_path)
