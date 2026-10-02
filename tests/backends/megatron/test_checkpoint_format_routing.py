# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Filesystem routing tests; placeholder weights are not loadable models.

Run the same cases against the full module when available and unmodified AST
functions on CPU-only hosts, following test_gdn_cp_mode's isolation pattern.
Heavy loading boundaries are replaced in both modes.
"""

import ast
import os
import re
import string
from contextlib import nullcontext
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest


def _load_functions(filename, names):
    path = Path(__file__).resolve().parents[3] / "relax/backends/megatron" / filename
    tree = ast.parse(path.read_text())
    selected = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    assert {node.name for node in selected} == set(names)
    module = ModuleType("_checkpoint_routing_" + path.stem)
    module.__dict__.update(Path=Path, os=os, re=re, string=string)
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), "exec"), vars(module))
    return module


@pytest.fixture(params=["isolated", "module"])
def checkpoint(request, monkeypatch):
    if request.param == "module":
        pytest.importorskip("megatron.training.checkpointing", exc_type=ImportError)
        from relax.backends.megatron import checkpoint as module
    else:
        module = _load_functions(
            "checkpoint.py", ["load_checkpoint", "_is_megatron_checkpoint", "_is_hf_checkpoint", "_is_dir_nonempty"]
        )
    monkeypatch.setattr(module, "logger", Mock(), raising=False)
    monkeypatch.setattr(module, "_alias_renamed_transfer_queue_enum", lambda: None, raising=False)
    return module


def _files(path, names):
    path.mkdir(parents=True, exist_ok=True)
    for name in names:
        target = path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("{}")
    return path


def _route(checkpoint, monkeypatch, path, expected_path):
    args = SimpleNamespace(load=str(path), hf_checkpoint="base-model")
    monkeypatch.setattr(checkpoint, "get_args", lambda: args, raising=False)
    hf = Mock(return_value=(0, 123))
    native = Mock(side_effect=AssertionError("unexpected Megatron load"))
    metadata = Mock(side_effect=AssertionError("unexpected LoRA metadata read"))
    monkeypatch.setattr(checkpoint, "_load_checkpoint_hf", hf, raising=False)
    monkeypatch.setattr(checkpoint, "_load_checkpoint_megatron", native, raising=False)
    monkeypatch.setattr(checkpoint, "_read_lora_checkpoint_metadata", metadata, raising=False)
    assert checkpoint.load_checkpoint(None, None, None, {}, False) == (0, 123)
    assert hf.call_args.kwargs["load_path"] == expected_path
    assert hf.call_args.kwargs["args"] is args
    native.assert_not_called()
    metadata.assert_not_called()


@pytest.mark.parametrize("name", ["hf_model", "iter_0000082"])
def test_checkpoint_hf_directory_preserves_source(checkpoint, monkeypatch, tmp_path, name):
    path = _files(tmp_path / name, ["config.json", "model.safetensors"])
    _route(checkpoint, monkeypatch, path, str(path))


@pytest.mark.parametrize("template", ["exports", "exports/iter_{rollout_id:07d}"])
def test_checkpoint_hf_export_path_round_trip(checkpoint, monkeypatch, tmp_path, template):
    model = _load_functions("model.py", ["_resolve_save_hf_path"])
    path = model._resolve_save_hf_path(str(tmp_path / template), 82)
    _files(path, ["config.json", "model.safetensors"])
    _route(checkpoint, monkeypatch, path, str(path))


@pytest.mark.parametrize(
    "marker",
    [
        "latest_checkpointed_iteration.txt",
        "metadata.json",  # Megatron torch_dist (also the historical zarr backend).
        ".metadata",  # torch_dcp.
        "mp_rank_00/model_optim_rng.pt",
        "mp_rank_00_000/model_optim_rng.pt",
        "mp_rank_00_000_000/model_optim_rng.pt",
    ],
)
@pytest.mark.parametrize("with_hf", [False, True])
def test_checkpoint_native_markers_take_priority(checkpoint, monkeypatch, tmp_path, marker, with_hf):
    name = "native_root" if marker == "latest_checkpointed_iteration.txt" else "iter_0000082"
    path = _files(tmp_path / name, [marker])
    if marker == "latest_checkpointed_iteration.txt":
        (path / marker).write_text("82")
    elif marker == "metadata.json":
        (path / marker).write_text('{"sharded_backend": "torch_dist", "sharded_backend_version": 1}')
    if with_hf:
        _files(path, ["config.json", "model.safetensors"])
    monkeypatch.setattr(checkpoint, "get_args", lambda: SimpleNamespace(load=str(path)), raising=False)
    metadata = Mock(return_value=None)
    native = Mock(return_value=(82, 456))
    monkeypatch.setattr(checkpoint, "_read_lora_checkpoint_metadata", metadata, raising=False)
    monkeypatch.setattr(checkpoint, "_load_checkpoint_megatron", native, raising=False)
    monkeypatch.setattr(checkpoint, "_load_checkpoint_hf", Mock(side_effect=AssertionError("HF load")), raising=False)
    assert checkpoint.load_checkpoint(None, None, None, {}, False) == (82, 456)
    metadata.assert_called_once_with(str(path))
    native.assert_called_once()


@pytest.mark.parametrize("name,files", [("empty", []), ("iter_0000082", []), ("unknown", ["unrelated.txt"])])
def test_checkpoint_fallback_unchanged(checkpoint, monkeypatch, tmp_path, name, files):
    path = _files(tmp_path / name, files)
    _route(checkpoint, monkeypatch, path, None)


@pytest.mark.parametrize("marker", ["unrelated.txt", "metadata.json", ".metadata"])
def test_checkpoint_incomplete_iteration_propagates_error(checkpoint, monkeypatch, tmp_path, marker):
    path = _files(tmp_path / "iter_0000082", [marker])
    monkeypatch.setattr(checkpoint, "get_args", lambda: SimpleNamespace(load=str(path)), raising=False)
    monkeypatch.setattr(checkpoint, "_read_lora_checkpoint_metadata", lambda _: None, raising=False)
    monkeypatch.setattr(
        checkpoint, "_load_checkpoint_megatron", Mock(side_effect=RuntimeError("incomplete checkpoint")), raising=False
    )
    monkeypatch.setattr(checkpoint, "_load_checkpoint_hf", Mock(side_effect=AssertionError("HF load")), raising=False)
    with pytest.raises(RuntimeError, match="incomplete checkpoint"):
        checkpoint.load_checkpoint(None, None, None, {}, False)


@pytest.mark.parametrize("skip_load", [False, True])
def test_checkpoint_lora_iteration_resume_order(checkpoint, monkeypatch, tmp_path, skip_load):
    path = _files(tmp_path / "iter_0000082", ["metadata.json", "config.json"])
    args = SimpleNamespace(load=str(path), hf_checkpoint="base-model")
    calls = []
    monkeypatch.setattr(checkpoint, "get_args", lambda: args, raising=False)
    monkeypatch.setattr(checkpoint, "_read_lora_checkpoint_metadata", lambda _: {"format_version": 1}, raising=False)
    monkeypatch.setattr(
        checkpoint, "_validate_lora_checkpoint_metadata", lambda *_: calls.append("validate"), raising=False
    )
    monkeypatch.setattr(checkpoint, "_load_checkpoint_hf", lambda **kw: calls.append(kw["load_path"]), raising=False)
    for name in (
        "_patch_lora_checkpoint_state_dict",
        "_strict_lora_checkpoint_dcp_load",
        "_preserve_hybrid_optimizer_steps_on_load",
        "_validate_lora_model_state_load",
    ):
        monkeypatch.setattr(checkpoint, name, lambda *_: nullcontext(), raising=False)
    native = Mock(return_value=(82, 0), side_effect=lambda **kw: calls.append("native") or (82, 0))
    monkeypatch.setattr(checkpoint, "_load_checkpoint_megatron", native, raising=False)
    assert checkpoint.load_checkpoint(None, None, None, {}, skip_load) == (82, 0)
    assert calls == (["validate", "native"] if skip_load else ["validate", "base-model", "native"])
    assert native.call_args.kwargs["strict"] is True
