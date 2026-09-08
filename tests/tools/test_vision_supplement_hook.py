# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Tests for the vision-supplement save-hf post-hook."""

from __future__ import annotations

import importlib.util
import json
import os
import pathlib
from types import SimpleNamespace

import pytest


torch = pytest.importorskip("torch")
pytest.importorskip("safetensors")
import safetensors  # noqa: E402
import safetensors.torch  # noqa: E402


_INDEX = "model.safetensors.index.json"
_SHARD = "model-vision-00001-of-00001.safetensors"
_LANG = ["model.language_model.layers.0.self_attn.q_proj.weight"]
_VISION = ["model.vision_tower.encoder.layers.0.mlp.down_proj.weight", "model.embed_vision.projection.weight"]


@pytest.fixture
def hook():
    source = pathlib.Path(__file__).resolve().parents[2] / "scripts/tools/vision_supplement_hook.py"
    spec = importlib.util.spec_from_file_location("vision_supplement_hook_under_test", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_model(directory, keys):
    directory.mkdir(parents=True, exist_ok=True)
    shard = "model-00001-of-00001.safetensors"
    safetensors.torch.save_file(
        {k: torch.ones(4, dtype=torch.bfloat16) for k in keys}, str(directory / shard), metadata={"format": "pt"}
    )
    (directory / _INDEX).write_text(
        json.dumps({"metadata": {"total_size": 8 * len(keys)}, "weight_map": dict.fromkeys(keys, shard)})
    )


@pytest.fixture
def reference(tmp_path):
    _write_model(tmp_path / "base", _LANG + _VISION)
    return str(tmp_path / "base")


@pytest.fixture
def export(tmp_path):
    _write_model(tmp_path / "export", _LANG)
    return tmp_path / "export"


def _index(export):
    return json.loads((export / _INDEX).read_text())


def test_copies_vision_and_extends_the_index(hook, reference, export):
    hook.supplement_vision(SimpleNamespace(hf_checkpoint=reference), str(export), 1, dtype="bf16", is_lora=False)

    index = _index(export)
    assert set(index["weight_map"]) == set(_LANG + _VISION)
    assert all(index["weight_map"][k] == _SHARD for k in _VISION)
    assert index["metadata"]["total_size"] == 8 * len(_LANG + _VISION)


def test_copied_tensors_match_the_reference(hook, reference, export):
    hook.supplement_vision(SimpleNamespace(hf_checkpoint=reference), str(export), 1, dtype="bf16", is_lora=False)

    with safetensors.safe_open(export / _SHARD, framework="pt") as out:
        with safetensors.safe_open(os.path.join(reference, "model-00001-of-00001.safetensors"), framework="pt") as src:
            assert all(torch.equal(out.get_tensor(k), src.get_tensor(k)) for k in _VISION)


def test_second_run_is_a_noop(hook, reference, export):
    args = SimpleNamespace(hf_checkpoint=reference)
    hook.supplement_vision(args, str(export), 1, dtype="bf16", is_lora=False)
    before, mtime = (export / _INDEX).read_text(), (export / _SHARD).stat().st_mtime_ns

    hook.supplement_vision(args, str(export), 2, dtype="bf16", is_lora=False)

    assert (export / _INDEX).read_text() == before
    assert (export / _SHARD).stat().st_mtime_ns == mtime


def test_supplements_only_what_is_missing(hook, reference, tmp_path):
    export = tmp_path / "partial"
    _write_model(export, _LANG + _VISION[:1])

    hook.supplement_vision(SimpleNamespace(hf_checkpoint=reference), str(export), 1, dtype="bf16", is_lora=False)

    with safetensors.safe_open(export / _SHARD, framework="pt") as out:
        assert list(out.keys()) == _VISION[1:]
    assert set(_index(export)["weight_map"]) == set(_LANG + _VISION)


@pytest.mark.parametrize(("dtype", "is_lora"), [("fp8", False), ("bf16", True)])
def test_skipped_for_fp8_and_lora(hook, reference, export, dtype, is_lora):
    hook.supplement_vision(SimpleNamespace(hf_checkpoint=reference), str(export), 1, dtype=dtype, is_lora=is_lora)

    assert not (export / _SHARD).exists()
    assert set(_index(export)["weight_map"]) == set(_LANG)


def test_does_not_mask_a_missing_language_weight(hook, tmp_path, export):
    """A genuine backbone-conversion bug must not be papered over with base
    weights."""
    _write_model(tmp_path / "base", _LANG + ["model.language_model.layers.99.mlp.up_proj.weight"] + _VISION)

    hook.supplement_vision(
        SimpleNamespace(hf_checkpoint=str(tmp_path / "base")), str(export), 1, dtype="bf16", is_lora=False
    )

    with safetensors.safe_open(export / _SHARD, framework="pt") as out:
        assert sorted(out.keys()) == sorted(_VISION)


def test_reference_without_vision_is_a_noop(hook, tmp_path, export):
    _write_model(tmp_path / "text_base", _LANG)

    hook.supplement_vision(
        SimpleNamespace(hf_checkpoint=str(tmp_path / "text_base")), str(export), 1, dtype="bf16", is_lora=False
    )

    assert not (export / _SHARD).exists()
