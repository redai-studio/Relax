# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Validate the execution checkpoint using headers only, without model
dependencies."""

import importlib.util
import json
import struct
import subprocess
import sys
from math import prod
from pathlib import Path

import pytest


_SOURCE = Path(__file__).resolve().parents[3] / "relax/models/deepseekv41/checkpoint.py"
_SPEC = importlib.util.spec_from_file_location("v41_checkpoint_test", _SOURCE)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
validate_bf16_checkpoint = _MODULE.validate_bf16_checkpoint


def _checkpoint(path: Path, *, dtype: str = "BF16", frozen_dtype: str = "F8_E4M3") -> Path:
    path.mkdir()
    config = {
        "model_type": "deepseek_v41",
        "text_config": {"model_type": "deepseek_v41_text", "num_hidden_layers": 2, "engram_layer_ids": [1]},
    }
    (path / "config.json").write_text(json.dumps(config))
    header = {
        "embed.weight": {"dtype": "BF16", "shape": [1, 2], "data_offsets": [0, 4]},
        "layers.1.engram.wkv.weight": {"dtype": dtype, "shape": [1, 2], "data_offsets": [4, 8]},
        "layers.1.engram.embed.weight": {"dtype": frozen_dtype, "shape": [1, 32], "data_offsets": [8, 40]},
        "layers.1.engram.embed.scale": {"dtype": "F8_E8M0", "shape": [1, 1], "data_offsets": [40, 41]},
        "layers.0.ffn.experts.0.w1.weight": {"dtype": dtype, "shape": [1, 2], "data_offsets": [41, 45]},
        "layers.0.hc_attn_fn": {"dtype": "F32", "shape": [1], "data_offsets": [45, 49]},
    }
    offset = 0
    for metadata in header.values():
        size = prod(metadata["shape"]) * {"BF16": 2, "F16": 2, "F32": 4}.get(metadata["dtype"], 1)
        metadata["data_offsets"] = [offset, offset + size]
        offset += size
    blob = json.dumps(header).encode()
    blob += b" " * (-len(blob) % 8)
    filename = "model.safetensors"
    (path / filename).write_bytes(struct.pack("<Q", len(blob)) + blob + bytes(offset))
    (path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": dict.fromkeys(header, filename)}))
    return path


def test_bf16_checkpoint_allows_fp32_constants_and_frozen_quantized_tables(tmp_path: Path) -> None:
    path = _checkpoint(tmp_path / "bf16")
    assert validate_bf16_checkpoint(str(path), require_v41=True)
    before = (path / "config.json").read_bytes()
    assert validate_bf16_checkpoint(str(path))
    assert (path / "config.json").read_bytes() == before


def test_bf16_checkpoint_accepts_flattened_config(tmp_path: Path) -> None:
    path = _checkpoint(tmp_path / "flat")
    config = json.loads((path / "config.json").read_text())["text_config"]
    config["model_type"] = "deepseek_v41"
    (path / "config.json").write_text(json.dumps(config))
    assert validate_bf16_checkpoint(str(path), require_v41=True)


@pytest.mark.parametrize("placement", ["outer", "text"])
def test_bf16_checkpoint_rejects_original_quantization_config(tmp_path: Path, placement: str) -> None:
    path = _checkpoint(tmp_path / "raw")
    config = json.loads((path / "config.json").read_text())
    target = config if placement == "outer" else config["text_config"]
    target["quantization_config"] = {"quant_method": "fp8", "expert_dtype": "fp4"}
    (path / "config.json").write_text(json.dumps(config))
    with pytest.raises(ValueError, match="original quantized checkpoint"):
        validate_bf16_checkpoint(str(path))


@pytest.mark.parametrize("dtype", ["F8_E4M3", "I8", "U8", "F16"])
def test_removing_quantization_config_without_casting_weights_is_rejected(tmp_path: Path, dtype: str) -> None:
    path = _checkpoint(tmp_path / "not_cast", dtype=dtype)
    with pytest.raises(ValueError, match="has dtype"):
        validate_bf16_checkpoint(str(path))


def test_frozen_engram_tables_must_not_be_upcast(tmp_path: Path) -> None:
    path = _checkpoint(tmp_path / "wrong_table", frozen_dtype="BF16")
    with pytest.raises(ValueError, match="engram.embed.weight.*has dtype"):
        validate_bf16_checkpoint(str(path))


def test_bf16_checkpoint_requires_frozen_tables_in_index(tmp_path: Path) -> None:
    path = _checkpoint(tmp_path / "missing_table")
    index_path = path / "model.safetensors.index.json"
    index = json.loads(index_path.read_text())
    del index["weight_map"]["layers.1.engram.embed.scale"]
    index_path.write_text(json.dumps(index))
    with pytest.raises(ValueError, match="missing frozen Engram"):
        validate_bf16_checkpoint(str(path))


def test_bf16_checkpoint_requires_indexed_tensor_to_exist_in_shard(tmp_path: Path) -> None:
    path = _checkpoint(tmp_path / "wrong_index")
    index_path = path / "model.safetensors.index.json"
    index = json.loads(index_path.read_text())
    index["weight_map"]["layers.0.ffn.experts.0.w2.weight"] = "model.safetensors"
    index_path.write_text(json.dumps(index))
    with pytest.raises(ValueError, match="missing from indexed shard"):
        validate_bf16_checkpoint(str(path))


def test_bf16_checkpoint_rejects_obsolete_non_engram_scales(tmp_path: Path) -> None:
    path = _checkpoint(tmp_path / "old_scale")
    index_path = path / "model.safetensors.index.json"
    index = json.loads(index_path.read_text())
    index["weight_map"]["layers.1.engram.wkv.scale"] = "model.safetensors"
    index_path.write_text(json.dumps(index))
    with pytest.raises(ValueError, match="obsolete weight scale"):
        validate_bf16_checkpoint(str(path))


def test_checkpoint_header_validation_does_not_read_tensor_payload(tmp_path: Path, monkeypatch) -> None:
    path = _checkpoint(tmp_path / "metadata_only")
    original = Path.open
    header_size = struct.unpack("<Q", (path / "model.safetensors").read_bytes()[:8])[0]

    class HeaderReader:
        def __init__(self, source) -> None:
            self.source = source

        def __enter__(self):
            self.source.__enter__()
            return self

        def __exit__(self, *args):
            return self.source.__exit__(*args)

        def read(self, count: int = -1) -> bytes:
            assert count in (8, header_size), "Only the safetensors header may be read"
            return self.source.read(count)

    def opened(candidate: Path, mode: str = "r", *args, **kwargs):
        source = original(candidate, mode, *args, **kwargs)
        return HeaderReader(source) if candidate.suffix == ".safetensors" and mode == "rb" else source

    monkeypatch.setattr(Path, "open", opened)
    assert validate_bf16_checkpoint(str(path))


def test_other_models_and_unknown_paths_are_not_restricted(tmp_path: Path) -> None:
    path = tmp_path / "ordinary"
    path.mkdir()
    (path / "config.json").write_text(
        json.dumps({"model_type": "qwen3", "quantization_config": {"quant_method": "fp8"}})
    )
    assert validate_bf16_checkpoint(str(path)) is False
    assert validate_bf16_checkpoint(str(tmp_path / "missing")) is False
    assert validate_bf16_checkpoint(None) is False
    with pytest.raises(ValueError, match="existing local BF16"):
        validate_bf16_checkpoint(str(tmp_path / "missing"), require_v41=True)
    with pytest.raises(ValueError, match="V4.1 BF16 checkpoint directory"):
        validate_bf16_checkpoint(str(path), require_v41=True)


def test_checkpoint_metadata_import_does_not_require_megatron() -> None:
    source = """
import importlib.abc
import sys

class NoMegatron(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'megatron' or fullname.startswith('megatron.'):
            raise ModuleNotFoundError('Megatron is absent in this inference worker', name=fullname)

sys.meta_path.insert(0, NoMegatron())
from relax.models.deepseekv41.checkpoint import validate_bf16_checkpoint
assert validate_bf16_checkpoint(None) is False
assert 'megatron' not in sys.modules
"""
    result = subprocess.run([sys.executable, "-c", source], capture_output=True, text=True, timeout=45)
    assert result.returncode == 0, result.stderr
