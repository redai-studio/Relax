# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import json
import os
import subprocess
import sys
import threading
from collections.abc import Callable, Iterator
from dataclasses import replace
from functools import wraps
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
import torch.distributed.checkpoint as dcp
from safetensors import safe_open
from safetensors.torch import save_file
from torch.distributed.checkpoint.metadata import ChunkStorageMetadata, TensorProperties, TensorStorageMetadata

from scripts.tools import convert_checkpoint_reader as range_reader
from scripts.tools import convert_checkpoint_to_hf_cpu as exporter
from scripts.tools.convert_checkpoint_mapping import Conversion, TensorSlice
from scripts.tools.convert_checkpoint_to_hf_cpu import (
    check_chunks,
    export_checkpoint,
    load_slices,
    make_reader,
    resolve_checkpoint,
)


def _cpu_only(test: Callable[..., None]) -> Callable[..., None]:
    @wraps(test)
    def run(*args: Any, **kwargs: Any) -> None:
        if os.environ.get("_RELAX_CPU_EXPORT_TEST_CHILD") == "1":
            assert not torch.cuda.is_available() and not torch.cuda.is_initialized()
            assert not torch.distributed.is_initialized()
            return test(*args, **kwargs)
        if not (torch.cuda.is_available() or torch.cuda.is_initialized() or torch.distributed.is_initialized()):
            return test(*args, **kwargs)
        # GPU CI shares a process with tests that initialize CUDA/process groups.
        # Keep the exporter's checks real and leave that parent state untouched.
        current = os.environ["PYTEST_CURRENT_TEST"].rsplit(" (", 1)[0]
        node = f"{Path(__file__).resolve()}::{current.split('::', 1)[1]}"
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "_RELAX_CPU_EXPORT_TEST_CHILD": "1"}
        for key in (
            "RANK",
            "WORLD_SIZE",
            "LOCAL_RANK",
            "LOCAL_WORLD_SIZE",
            "MASTER_ADDR",
            "MASTER_PORT",
            "PYTEST_ADDOPTS",
        ):
            env.pop(key, None)
        result = subprocess.run(
            [sys.executable, "-m", "pytest", "-q", node, "--tb=short", "--show-capture=no"],
            cwd=Path(__file__).resolve().parents[2],
            env=env,
            capture_output=True,
            text=True,
            timeout=180,
        )
        assert result.returncode == 0, result.stdout + result.stderr

    return run


@pytest.fixture(params=[True, False], ids=["crc", "no-crc"])
def archive_crc32(request: pytest.FixtureRequest) -> Iterator[None]:
    previous = torch.serialization.get_crc32_options()
    torch.serialization.set_crc32_options(request.param)
    try:
        yield
    finally:
        torch.serialization.set_crc32_options(previous)


def _fixture(tmp_path: Path) -> tuple[Path, Path, dict[str, torch.Tensor]]:
    model, checkpoint = tmp_path / "model", tmp_path / "checkpoints" / "iter_0000007"
    model.mkdir()
    config = {
        "model_type": "qwen4_exp",
        "text_config": {
            "hidden_size": 4,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 2,
            "make_ngram_vocab_size_divisible_by": 2,
        },
        "vision_config": {"depth": 2, "hidden_size": 4, "num_heads": 2},
    }
    (model / "config.json").write_text(json.dumps(config))
    (model / "tokenizer.json").write_text("{}")
    state, expected = {}, {}

    def pair(source: str, target: str, tensor: torch.Tensor) -> None:
        state[source] = tensor
        expected[target] = tensor.clone()

    src = "language_model.decoder.layers.0."
    dst = "model.language_model.layers.0."
    pair(
        src + "mlp.experts.experts.linear_fc1.weight",
        dst + "mlp.experts.gate_up_proj",
        torch.arange(32.0).reshape(2, 4, 4),
    )
    pair(
        src + "mlp.experts.experts.linear_fc2.weight",
        dst + "mlp.experts.down_proj",
        torch.arange(16.0).reshape(2, 4, 2),
    )
    for suffix, target in (("in_proj.weight", "in_proj_qkv.weight"), ("conv1d.weight", "conv1d.weight")):
        pieces = []
        for index, part in enumerate(("query", "key", "value")):
            shape = (2, 4) if suffix.startswith("in_proj") else (2, 1, 3)
            tensor = torch.full(shape, float(index + 3))
            state[src + "self_attention." + suffix + "." + part] = tensor
            pieces.append(tensor)
        expected[dst + "linear_attn." + target] = torch.cat(pieces)
    for part, target in (("z", "z"), ("beta", "b"), ("alpha", "a")):
        pair(
            src + "self_attention.in_proj.weight." + part,
            dst + f"linear_attn.in_proj_{target}.weight",
            torch.arange(8.0).reshape(2, 4),
        )
    state[src + "self_attention.out_norm.weight"] = torch.tensor([-1.0, 0.0, 2.0])
    expected[dst + "linear_attn.norm.weight"] = torch.tensor([0.0, 1.0, 3.0])
    for root, target in (
        ("language_model.decoder.layers.1.", "model.language_model.layers.1."),
        ("language_model.mtp.layers.0.mtp_model_layer.layers.0.", "mtp.layers.0."),
    ):
        # Independent HF -> checkpoint packing: per group [Q heads, gates, K, V].
        q = torch.arange(64.0).reshape(4, 2, 2, 4)
        k, v = torch.full((2, 2, 4), 100.0), torch.full((2, 2, 4), 200.0)
        packed = []
        for group in range(2):
            packed.extend(
                (
                    q[2 * group : 2 * group + 2, 0].reshape(4, 4),
                    q[2 * group : 2 * group + 2, 1].reshape(4, 4),
                    k[group],
                    v[group],
                )
            )
        state[root + "self_attention.linear_qkv.weight"] = torch.cat(packed)
        expected[target + "self_attn.q_proj.weight"] = q.reshape(16, 4)
        expected[target + "self_attn.k_proj.weight"] = k.reshape(4, 4)
        expected[target + "self_attn.v_proj.weight"] = v.reshape(4, 4)
    pair("language_model.mtp.layers.0.hnorm.weight", "mtp.pre_fc_norm_hidden.weight", torch.arange(16.0))
    pair(
        src + "self_attention_hyper_connection.hc_norm.weight",
        dst + "attn_hyper_connection.hc_norm.weight",
        torch.arange(4.0),
    )
    state["language_model.embedding.word_embeddings.weight"] = torch.arange(24.0).reshape(6, 4)
    expected["model.language_model.embed_tokens.weight"] = state["language_model.embedding.word_embeddings.weight"][
        :5
    ].clone()
    table = torch.arange(20.0).reshape(10, 2)
    state[src + "per_layer_embedding.ple_embedding.ngram_embedding.weight"] = table
    for index in range(3):
        expected[dst + f"ple.ple_embedding.ngram_embedding.shard_{index}.weight"] = table[
            index * 3 : index * 3 + 3
        ].clone()
    for field, tail in (("weight", (4,)), ("bias", ())):
        layers = []
        for layer in range(2):
            qkv = torch.arange(12 * (4 if tail else 1), dtype=torch.float32).reshape(3, 2, 2, *tail) + layer * 100
            layers.append(qkv.transpose(0, 1).reshape(12, *tail))
            expected[f"model.visual.blocks.{layer}.attn.qkv.{field}"] = qkv.reshape(12, *tail)
        state[f"vision_model.decoder.layers.self_attention.linear_qkv.{field}"] = torch.stack(layers)
    pair("vision_model.merger.patch_norm.weight", "model.visual.merger.norm.weight", torch.arange(4.0))
    state["optimizer.state.exp_avg"] = torch.ones(2)
    dcp.save(state, storage_writer=dcp.FileSystemWriter(checkpoint, single_file_per_rank=False), no_dist=True)
    (checkpoint.parent / "latest_checkpointed_iteration.txt").write_text("7\n")
    base = {key: torch.zeros_like(value) for key, value in expected.items()}
    base[dst + "linear_attn.norm.weight"] = base[dst + "linear_attn.norm.weight"].to(torch.bfloat16)
    save_file(base, model / "model.safetensors")
    return checkpoint, model, expected


@pytest.mark.parametrize(
    "batch_bytes,read_workers,prefetch", [(128, 1, True), (1024**2, 4, True), (128, 4, False), (1024**2, 8, True)]
)
@_cpu_only
def test_cpu_export_round_trip_without_model_or_cuda(
    tmp_path: Path, batch_bytes: int, read_workers: int, prefetch: bool
) -> None:
    checkpoint, model, expected = _fixture(tmp_path)
    output = tmp_path / "export"
    report = export_checkpoint(
        checkpoint.parent,
        model,
        output,
        shard_bytes=512,
        memory_budget=1024**2,
        batch_bytes=batch_bytes,
        read_workers=read_workers,
        prefetch=prefetch,
    )
    index = json.loads((output / "model.safetensors.index.json").read_text())
    assert set(index["weight_map"]) == set(expected)
    for name, value in expected.items():
        with safe_open(output / index["weight_map"][name], framework="pt", device="cpu") as handle:
            assert torch.equal(handle.get_tensor(name), value), name
            assert handle.get_tensor(name).dtype == value.dtype
    assert index["metadata"]["total_size"] == sum(t.numel() * t.element_size() for t in expected.values())
    assert report["max_input_bytes"] < 1024
    assert report["peak_batch_input_bytes"] <= max(batch_bytes, report["max_input_bytes"])
    assert report["planned_buffer_bytes"] <= report["memory_budget_bytes"]
    if batch_bytes > 1024:
        assert report["load_batches"] < report["conversion_groups"]
    assert not report["cuda_initialized"] and not report["distributed_initialized"]
    assert report["dtype_changes"]["model.language_model.layers.0.linear_attn.norm.weight"] == {
        "base": "BF16",
        "export": "F32",
    }
    assert (output / "config.json").read_bytes() == (model / "config.json").read_bytes()
    assert (output / "tokenizer.json").exists()
    assert not list(tmp_path.glob("export.incomplete-*"))


@pytest.mark.parametrize("entrypoint", ["shell", "python", "module"])
@_cpu_only
def test_cpu_export_entrypoint_dry_run(tmp_path: Path, entrypoint: str) -> None:
    checkpoint, model, _ = _fixture(tmp_path)
    output = tmp_path / "export"
    root = Path(__file__).resolve().parents[2]
    script = root / "scripts/tools/convert_checkpoint_to_hf_cpu"
    commands = {
        "shell": ["bash", str(script.with_suffix(".sh"))],
        "python": [sys.executable, str(script.with_suffix(".py"))],
        "module": [sys.executable, "-m", "scripts.tools.convert_checkpoint_to_hf_cpu"],
    }
    env = {
        **os.environ,
        "CHECKPOINT_PATH": str(checkpoint),
        "MODEL_PATH": str(model),
        "OUTPUT_DIR": str(output),
        "PYTHON_BIN": sys.executable,
    }
    env.pop("PYTHONPATH", None)
    result = subprocess.run(
        [*commands[entrypoint], "--dry-run"],
        cwd=root if entrypoint == "module" else tmp_path,
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "CPU export plan" in result.stdout + result.stderr
    assert not output.exists()


@_cpu_only
def test_cpu_export_parameter_layout_hc_ple_mtp_and_frozen_weights(tmp_path: Path) -> None:
    model, checkpoint, output = (tmp_path / name for name in ("model", "checkpoint", "output"))
    model.mkdir()
    (model / "config.json").write_text(json.dumps({"model_type": "qwen4_exp", "text_config": {}}))
    state, expected = {}, {}
    norm = torch.tensor([-1.0, 0.0, 2.0, 3.0], dtype=torch.bfloat16)

    def pair(source: str, target: str, tensor: torch.Tensor = norm) -> None:
        state[source] = tensor.clone()
        expected[target] = tensor.clone()

    hc_fields = {
        "hc_norm_weight": "hc_norm.weight",
        "input_mix_weight_down": "input_mix_weight_down.weight",
        "input_mix_weight_up": "input_mix_weight_up.weight",
        "block_inject_weight": "block_inject_weight.weight",
    }
    for source, target in (
        ("decoder.hc_head_contraction.", "model.language_model.hyper_connection_mixer."),
        ("mtp.layers.0.hc_head_contraction.", "mtp.hyper_connection_mixer."),
    ):
        for field in ("hc_norm_weight", "input_mix_weight_down", "input_mix_weight_up"):
            pair(source + field, target + hc_fields[field])
    for source, target in (
        ("decoder.layers.1.", "model.language_model.layers.1."),
        ("mtp.layers.0.transformer_layer.", "mtp.layers.0."),
    ):
        for src_hc, dst_hc in (
            ("self_attention_hyper_connection.", "attn_hyper_connection."),
            ("mlp_hyper_connection.", "mlp_hyper_connection."),
        ):
            for field, hf_field in hc_fields.items():
                pair(source + src_hc + field, target + dst_hc + hf_field)
        for field in ("q_layernorm", "k_layernorm"):
            pair(source + "self_attention.indexer." + field, target + "self_attn.indexer." + field + ".weight")
    for field, hf_field in (
        ("norm_key", "norm_key.weight"),
        ("norm_query", "norm_query.weight"),
        ("norm_conv", "norm_conv.weight"),
        ("conv1d_weight", "conv1d.weight"),
        ("key_proj.weight", "key_proj.weight"),
        ("value_proj.weight", "value_proj.weight"),
    ):
        pair(
            "decoder.layers.1.self_attention_hyper_connection.ple." + field,
            "model.language_model.layers.1.ple." + hf_field,
        )
    for field in ("norm.weight", "conv1d.weight", "in_proj_qkv.weight", "A_log"):
        pair(
            "decoder.layers.0.self_attention.linear_attn." + field,
            "model.language_model.layers.0.linear_attn." + field,
            norm.float() if field == "A_log" else norm,
        )
    for field, hf_field in (
        ("e_proj.weight", "fc_embedding.weight"),
        ("h_proj.weight", "fc_hidden.weight"),
        ("enorm.weight", "pre_fc_norm_embedding.weight"),
        ("hnorm.weight", "pre_fc_norm_hidden.weight"),
    ):
        pair("mtp.layers.0." + field, "mtp." + hf_field)
    base = {name: torch.zeros_like(value, dtype=torch.bfloat16) for name, value in expected.items()}
    frozen = {
        "model.visual.pos_embed.weight": torch.arange(8, dtype=torch.bfloat16).reshape(2, 4),
        "model.language_model.layers.1.ple.ple_embedding.layer_multipliers": torch.tensor([3, 7, 11]),
        "model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_0.weight": norm,
    }
    base.update(frozen)
    expected.update(frozen)
    dcp.save(state, checkpoint_id=checkpoint, no_dist=True)
    save_file(base, model / "model.safetensors")
    with pytest.raises(ValueError, match="Missing"):
        export_checkpoint(checkpoint, model, output, dry_run=True)
    report = export_checkpoint(checkpoint, model, output, shard_bytes=128, allow_base_fallback=("vision", "ple"))
    assert report["base_fallback_keys"] == sorted(frozen)
    index = json.loads((output / "model.safetensors.index.json").read_text())["weight_map"]
    assert set(index) == set(expected)
    for name, value in expected.items():
        with safe_open(output / index[name], framework="pt", device="cpu") as handle:
            actual = handle.get_tensor(name)
            assert actual.dtype == value.dtype and torch.equal(actual, value), name


@pytest.mark.parametrize("mtp_layer", ["transformer_layer", "mtp_model_layer"])
@_cpu_only
def test_cpu_export_standard_norms_and_individual_mtp_experts(tmp_path: Path, mtp_layer: str) -> None:
    model, checkpoint, output = (tmp_path / name for name in ("model", "checkpoint", "output"))
    model.mkdir()
    config = {"model_type": "qwen3_5_moe", "text_config": {"hidden_size": 4}}
    (model / "config.json").write_text(json.dumps(config))
    src = "language_model."
    layer = f"{src}mtp.layers.0.{mtp_layer}."
    norm = torch.tensor([-1.0, 0.0, 2.0, 3.0], dtype=torch.bfloat16)
    fc1 = torch.arange(32, dtype=torch.bfloat16).reshape(2, 4, 4)
    fc2 = torch.arange(16, dtype=torch.bfloat16).reshape(2, 4, 2)
    state = {
        src + "decoder.final_layernorm.weight": norm,
        src + "decoder.layers.0.pre_mlp_layernorm.weight": norm,
        src + "decoder.layers.0.self_attention.in_proj.layer_norm_weight": norm,
        src + "decoder.layers.0.self_attention.out_norm.weight": norm,
        src + "mtp.layers.0.eh_proj.weight": torch.arange(32, dtype=torch.bfloat16).reshape(4, 8),
        src + "mtp.layers.0.final_layernorm.weight": norm,
        layer + "self_attention.linear_qkv.layer_norm_weight": norm,
        layer + "mlp.experts.experts.linear_fc1.weight": fc1,
        layer + "mlp.experts.experts.linear_fc2.weight": fc2,
    }
    expected = {
        "model.language_model.norm.weight": norm,
        "model.language_model.layers.0.post_attention_layernorm.weight": norm,
        "model.language_model.layers.0.input_layernorm.weight": norm,
        "model.language_model.layers.0.linear_attn.norm.weight": norm + 1,
        "mtp.fc.weight": state[src + "mtp.layers.0.eh_proj.weight"],
        "mtp.norm.weight": norm,
        "mtp.layers.0.input_layernorm.weight": norm,
    }
    for expert in range(2):
        expected[f"mtp.layers.0.mlp.experts.{expert}.gate_proj.weight"] = fc1[expert, :2]
        expected[f"mtp.layers.0.mlp.experts.{expert}.up_proj.weight"] = fc1[expert, 2:]
        expected[f"mtp.layers.0.mlp.experts.{expert}.down_proj.weight"] = fc2[expert]
    dcp.save(state, checkpoint_id=checkpoint, no_dist=True)
    save_file({name: torch.zeros_like(value) for name, value in expected.items()}, model / "model.safetensors")
    export_checkpoint(checkpoint, model, output, shard_bytes=128)
    index = json.loads((output / "model.safetensors.index.json").read_text())["weight_map"]
    assert set(index) == set(expected)
    for name, value in expected.items():
        with safe_open(output / index[name], framework="pt", device="cpu") as handle:
            assert torch.equal(handle.get_tensor(name), value), name


@pytest.mark.parametrize("failure", ["missing", "unknown", "memory", "existing"])
@_cpu_only
def test_cpu_export_rejects_invalid_inputs_before_write(tmp_path: Path, failure: str) -> None:
    checkpoint, model, expected = _fixture(tmp_path)
    output = tmp_path / "export"
    kwargs = {}
    if failure == "missing":
        expected["mtp.fc_hidden.weight"] = torch.ones(4, 4)
        save_file(expected, model / "model.safetensors")
    elif failure == "unknown":
        checkpoint = tmp_path / "unknown"
        dcp.save(
            {"language_model.decoder.layers.0.unsupported.weight": torch.ones(2)},
            checkpoint_id=checkpoint,
            no_dist=True,
        )
    elif failure == "memory":
        kwargs["memory_budget"] = 1
    else:
        output.mkdir()
        (output / "keep").write_text("previous export")
    with pytest.raises((ValueError, FileExistsError)):
        export_checkpoint(checkpoint, model, output, **kwargs)
    assert not list(tmp_path.glob("export.incomplete-*"))
    if failure == "existing":
        assert (output / "keep").read_text() == "previous export"
    else:
        assert not output.exists()


@_cpu_only
def test_cpu_export_missing_weights_require_explicit_group(tmp_path: Path) -> None:
    checkpoint, model, expected = _fixture(tmp_path)
    name = "model.visual.patch_embed.proj.weight"
    expected[name] = torch.full((2, 3), 42.0)
    save_file(expected, model / "model.safetensors")
    with pytest.raises(ValueError, match="Missing"):
        export_checkpoint(checkpoint, model, tmp_path / "rejected", dry_run=True)
    output = tmp_path / "allowed"
    report = export_checkpoint(checkpoint, model, output, allow_base_fallback=("vision",))
    assert report["base_fallback_keys"] == [name]
    index = json.loads((output / "model.safetensors.index.json").read_text())
    with safe_open(output / index["weight_map"][name], framework="pt", device="cpu") as handle:
        assert torch.equal(handle.get_tensor(name), expected[name])


@pytest.mark.parametrize("read_workers", [1, 4])
@pytest.mark.parametrize("ranged", [False, True])
@pytest.mark.parametrize("dimension", [0, 1])
@_cpu_only
def test_cpu_dcp_partial_read_crosses_source_chunks(
    tmp_path: Path, read_workers: int, ranged: bool, dimension: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Write an actual DCP archive with two persistent chunks, as TP shards do.
    from torch.distributed.checkpoint import DefaultSavePlanner
    from torch.distributed.checkpoint.planner import SavePlan, WriteItem, WriteItemType
    from torch.distributed.checkpoint.planner_helpers import _create_write_item_for_tensor

    tensor = torch.arange(64, dtype=torch.bfloat16).reshape(8, 8)

    class ChunkPlanner(DefaultSavePlanner):
        def create_local_plan(self) -> SavePlan:
            items = []
            for index, part in enumerate(tensor.chunk(2, dim=dimension)):
                offset = (index * 4, 0) if dimension == 0 else (0, index * 4)
                item = _create_write_item_for_tensor("table", part)
                # MetadataIndex is frozen; create one with the global offset.
                from torch.distributed.checkpoint.metadata import MetadataIndex

                item = replace(
                    item,
                    type=WriteItemType.SHARD,
                    index=MetadataIndex("table", offset),
                    tensor_data=replace(
                        item.tensor_data,
                        size=tensor.size(),
                        chunk=ChunkStorageMetadata(torch.Size(offset), part.size()),
                    ),
                )
                items.append(item)
            return SavePlan(items)

        def resolve_data(self, write_item: WriteItem) -> torch.Tensor:
            start = write_item.index.offset[dimension]
            return tensor.narrow(dimension, start, 4).clone()

    dcp.save(
        {"table": tensor},
        storage_writer=dcp.FileSystemWriter(tmp_path, single_file_per_rank=False),
        planner=ChunkPlanner(flatten_state_dict=False),
        no_dist=True,
    )
    reader = make_reader(tmp_path, read_workers)
    metadata = reader.read_metadata()
    if ranged:
        monkeypatch.setattr(range_reader, "RANGE_READ_THRESHOLD", 1)
        monkeypatch.setattr(range_reader, "READ_BLOCK_BYTES", 8)
        records = range_reader.prepare_range_reads(tmp_path, metadata, metadata.state_dict_metadata)
        assert len(records) == 2
        reader = make_reader(tmp_path, read_workers, records)

        def reject_full_load(*args: Any, **kwargs: Any) -> None:
            raise AssertionError("Range reads must not deserialize the full tensor")

        monkeypatch.setattr(torch, "load", reject_full_load)
    assert len(metadata.state_dict_metadata["table"].chunks) == 2
    assert len({info.relative_path for info in metadata.storage_data.values()}) == 2
    check_chunks("table", metadata.state_dict_metadata["table"])
    actual = load_slices(reader, (TensorSlice("table", (3, 3), (3, 3)),), metadata)["table"]
    assert torch.equal(actual, tensor[3:6, 3:6])


@_cpu_only
def test_cpu_export_round_trip_with_range_reads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, archive_crc32: None
) -> None:
    checkpoint, model, expected = _fixture(tmp_path)
    monkeypatch.setattr(range_reader, "RANGE_READ_THRESHOLD", 1)
    result = export_checkpoint(checkpoint, model, tmp_path / "export", shard_bytes=512)
    assert result["range_read_chunks"] > 0
    index = json.loads((tmp_path / "export/model.safetensors.index.json").read_text())["weight_map"]
    for key, tensor in expected.items():
        with safe_open(tmp_path / "export" / index[key], framework="pt", device="cpu") as handle:
            assert torch.equal(handle.get_tensor(key), tensor), key


@pytest.mark.parametrize("truncate", [False, True])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@_cpu_only
def test_cpu_range_read_preserves_storage_and_archive_offsets(
    tmp_path: Path, truncate: bool, dtype: torch.dtype, archive_crc32: None
) -> None:
    tensor = torch.arange(64, dtype=dtype).reshape(8, 8)[2:6]
    path = tmp_path / "data.distcp"
    with path.open("wb") as stream:
        stream.write(b"archive-prefix" * 13)
        offset = stream.tell()
        torch.save(tensor, stream)
        length = stream.tell() - offset
    record = range_reader.tensor_record(
        path, SimpleNamespace(offset=offset, length=length), tuple(tensor.shape), tensor.dtype
    )
    target = torch.empty((2, 3), dtype=tensor.dtype)
    request = SimpleNamespace(storage_offsets=(1, 3), lengths=(2, 3))
    if truncate:
        with path.open("r+b") as stream:
            stream.truncate(record.offset + 8)
        with path.open("rb") as stream, pytest.raises(ValueError, match="Truncated"):
            range_reader.read_tensor_range(stream, record, request, target)
    else:
        with path.open("rb") as stream:
            range_reader.read_tensor_range(stream, record, request, target)
        assert torch.equal(target, tensor[1:3, 3:6])


@pytest.mark.parametrize("layout", ["compressed", "transposed"])
@_cpu_only
def test_cpu_range_read_keeps_regular_loader_for_other_archives(
    tmp_path: Path, layout: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    import io
    import zipfile

    tensor = torch.arange(64.0).reshape(8, 8).t()
    dcp.save({"table": tensor}, checkpoint_id=tmp_path, no_dist=True)
    reader = make_reader(tmp_path, 1)
    metadata = reader.read_metadata()
    index, info = next(iter(metadata.storage_data.items()))
    archive = io.BytesIO()
    torch.save(tensor, archive)
    if layout == "compressed":
        compressed = io.BytesIO()
        with zipfile.ZipFile(archive) as source, zipfile.ZipFile(compressed, "w", zipfile.ZIP_DEFLATED) as target:
            for entry in source.infolist():
                target.writestr(entry.filename, source.read(entry))
        archive = compressed
    (tmp_path / info.relative_path).write_bytes(archive.getvalue())
    metadata.storage_data[index] = replace(info, offset=0, length=len(archive.getvalue()))
    monkeypatch.setattr(range_reader, "RANGE_READ_THRESHOLD", 1)
    records = range_reader.prepare_range_reads(tmp_path, metadata, metadata.state_dict_metadata)
    assert not records
    reader.read_metadata = lambda: metadata
    actual = load_slices(reader, (TensorSlice("table", (3, 3), (3, 3)),), metadata)["table"]
    assert torch.equal(actual, tensor[3:6, 3:6])


@pytest.mark.parametrize("corruption", ["shape", "truncated"])
@_cpu_only
def test_cpu_range_read_rejects_invalid_archive_descriptors(tmp_path: Path, corruption: str) -> None:
    tensor = torch.arange(64.0).reshape(8, 8)
    path = tmp_path / "data.distcp"
    torch.save(tensor, path)
    length = path.stat().st_size
    shape = (4, 16) if corruption == "shape" else tuple(tensor.shape)
    if corruption == "truncated":
        with path.open("r+b") as stream:
            stream.truncate(length - 1)
    with pytest.raises(ValueError, match="descriptor|Truncated"):
        range_reader.tensor_record(path, SimpleNamespace(offset=0, length=length), shape, tensor.dtype)


@pytest.mark.parametrize("corruption", ["short-storage", "long-storage", "offset"])
@_cpu_only
def test_cpu_range_read_rejects_invalid_serialized_storage(tmp_path: Path, corruption: str) -> None:
    import io
    import pickle
    import zipfile

    tensor = torch.arange(64.0).reshape(8, 8)

    class TensorDescriptor:
        def __reduce__(self) -> tuple[Any, tuple[Any, ...]]:
            return torch._utils._rebuild_tensor_v2, (
                tensor._typed_storage(),
                1 if corruption == "offset" else 0,
                tuple(tensor.shape),
                tensor.stride(),
                False,
                {},
            )

    class StoragePickler(pickle.Pickler):
        def persistent_id(self, obj: Any) -> Any:
            if isinstance(obj, torch.storage.TypedStorage):
                numel = {"short-storage": 32, "long-storage": 128, "offset": 64}[corruption]
                return "storage", torch.FloatStorage, "0", "cpu", numel
            return None

    descriptor = io.BytesIO()
    StoragePickler(descriptor, protocol=2).dump(TensorDescriptor())
    original = io.BytesIO()
    torch.save(tensor, original)
    path = tmp_path / "data.distcp"
    with zipfile.ZipFile(original) as source, zipfile.ZipFile(path, "w") as target:
        for entry in source.infolist():
            data = descriptor.getvalue() if entry.filename.endswith("/data.pkl") else source.read(entry)
            target.writestr(entry, data)
    with pytest.raises(ValueError, match="storage"):
        range_reader.tensor_record(
            path, SimpleNamespace(offset=0, length=path.stat().st_size), tuple(tensor.shape), tensor.dtype
        )


@_cpu_only
def test_cpu_export_parallel_read_failure_does_not_publish(tmp_path: Path) -> None:
    from torch.distributed.checkpoint.api import CheckpointException

    checkpoint, model, _ = _fixture(tmp_path)
    metadata = dcp.FileSystemReader(checkpoint).read_metadata()
    source = next(info for key, info in metadata.storage_data.items() if key.fqn.startswith("language_model."))
    (checkpoint / source.relative_path).write_bytes(b"corrupt checkpoint")
    output = tmp_path / "export"
    with pytest.raises(CheckpointException):
        export_checkpoint(checkpoint, model, output, shard_bytes=512, read_workers=4)
    assert not output.exists()
    assert not list(tmp_path.glob("export.incomplete-*/model.safetensors.index.json"))


@_cpu_only
def test_cpu_export_batching_respects_memory_budget(tmp_path: Path) -> None:
    checkpoint, model, _ = _fixture(tmp_path)
    output = tmp_path / "export"
    single = export_checkpoint(checkpoint, model, output, shard_bytes=512, batch_bytes=1, prefetch=False, dry_run=True)
    combined = export_checkpoint(checkpoint, model, output, shard_bytes=512, batch_bytes=1024**2, dry_run=True)
    assert combined["planned_buffer_bytes"] > single["planned_buffer_bytes"]
    bounded = export_checkpoint(
        checkpoint,
        model,
        output,
        shard_bytes=512,
        batch_bytes=1024**2,
        memory_budget=single["planned_buffer_bytes"],
        dry_run=True,
    )
    assert bounded["planned_buffer_bytes"] <= single["planned_buffer_bytes"]
    assert bounded["load_batches"] > combined["load_batches"]
    assert bounded["prefetched_batches"] < bounded["load_batches"] - 1
    serial = export_checkpoint(checkpoint, model, output, shard_bytes=512, read_workers=1, dry_run=True)
    assert combined["read_buffer_bytes"] == 4 * serial["read_buffer_bytes"]
    with pytest.raises(ValueError, match="Planned buffers"):
        export_checkpoint(checkpoint, model, output, shard_bytes=512, memory_budget=single["planned_buffer_bytes"] - 1)
    assert not list(tmp_path.glob("export.incomplete-*"))


@_cpu_only
def test_cpu_reader_workspace_counts_source_chunks_and_concurrent_files() -> None:
    from torch.distributed.checkpoint.metadata import MetadataIndex

    properties = TensorProperties(dtype=torch.float32)
    tensors = {
        name: TensorStorageMetadata(
            properties, torch.Size((size,)), [ChunkStorageMetadata(torch.Size((0,)), torch.Size((size,)))]
        )
        for name, size in [("small", 4), ("large", 64), ("unused", 1024**3)]
    }
    storage = {
        MetadataIndex(name, (0,)): SimpleNamespace(relative_path=name, length=length)
        for name, length in [("small", 32), ("large", 512), ("unused", 8 * 1024**3)]
    }
    jobs = [Conversion((TensorSlice(name, (1,), (0,)),), (name,)) for name in ["small", "large"]]
    workspace = exporter._reader_workspace(jobs, tensors, storage, read_workers=8)
    # The one-element destination still needs the entire serialized source
    # chunk, but a single source file cannot occupy eight reader threads.
    assert workspace([0]) == 64
    assert workspace([1]) == 1024
    assert workspace([0, 1]) == 2048
    storage[MetadataIndex("large", (0,))].relative_path = "small"
    shared_file = exporter._reader_workspace(jobs, tensors, storage, read_workers=8)
    assert shared_file([0, 1]) == 1024


@_cpu_only
def test_cpu_batch_plan_recomputes_reader_workspace_when_combining_files() -> None:
    from torch.distributed.checkpoint.metadata import MetadataIndex

    names = ["a", "b", "c"]
    tensors = {
        name: TensorStorageMetadata(
            TensorProperties(dtype=torch.float32),
            torch.Size((4,)),
            [ChunkStorageMetadata(torch.Size((0,)), torch.Size((4,)))],
        )
        for name in names
    }
    storage = {MetadataIndex(name, (0,)): SimpleNamespace(relative_path=name, length=32) for name in names}
    jobs = [Conversion((TensorSlice(name, (1,), (0,)),), (name,)) for name in names]
    workspace = exporter._reader_workspace(jobs, tensors, storage, read_workers=8)
    batches = exporter.plan_batches(jobs, [4, 4, 4], [0, 0, 0], 1024, 100, workspace)
    assert batches == [[0], [1], [2]]
    assert all(sum(4 for _ in batch) + workspace(batch) <= 100 for batch in batches)


@pytest.mark.parametrize("failure", [None, "load", "write"])
@_cpu_only
def test_cpu_export_prefetch_overlaps_and_joins_on_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str | None
) -> None:
    checkpoint, model, _ = _fixture(tmp_path)
    output = tmp_path / "export"
    second_started, appending = threading.Event(), threading.Event()
    lock = threading.Lock()
    original_load, original_add = exporter.load_slices, exporter.ShardWriter.add
    calls = active = peak = 0
    first_append = True

    def load(*args: Any, **kwargs: Any) -> dict[str, torch.Tensor]:
        nonlocal calls, active, peak
        with lock:
            calls += 1
            number = calls
            active += 1
            peak = max(peak, active)
        try:
            if number == 2:
                second_started.set()
                assert appending.wait(10), "Prefetch did not overlap output processing"
                if failure == "load":
                    raise OSError("prefetch failed")
            return original_load(*args, **kwargs)
        finally:
            with lock:
                active -= 1

    def add(self: exporter.ShardWriter, name: str, tensor: torch.Tensor) -> None:
        nonlocal first_append
        if first_append:
            first_append = False
            assert second_started.wait(10), "Second batch was not prefetched"
            appending.set()
            if failure == "write":
                raise OSError("write failed")
        return original_add(self, name, tensor)

    monkeypatch.setattr(exporter, "load_slices", load)
    monkeypatch.setattr(exporter.ShardWriter, "add", add)
    if failure:
        with pytest.raises(OSError, match="failed"):
            export_checkpoint(checkpoint, model, output, batch_bytes=1, shard_bytes=512)
        assert not output.exists()
        assert not list(tmp_path.glob("export.incomplete-*/model.safetensors.index.json"))
    else:
        result = export_checkpoint(checkpoint, model, output, batch_bytes=1, shard_bytes=512)
        assert result["prefetched_batches"] == result["load_batches"] - 1
    assert active == 0 and peak == 1


@_cpu_only
def test_cpu_export_streams_frozen_payloads_without_materializing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import safetensors

    checkpoint, model, expected = _fixture(tmp_path)
    frozen = {
        "model.visual.large": torch.arange(257, dtype=torch.float32),
        "model.visual.empty": torch.empty((0, 3), dtype=torch.bfloat16),
        "model.visual.scalar": torch.tensor(-9, dtype=torch.int64),
        "model.visual.整数": torch.tensor([5, -7], dtype=torch.int32),
        "model.visual.boolean": torch.tensor([True, False]),
    }
    save_file(
        {**{key: torch.zeros_like(value) for key, value in expected.items()}, **frozen}, model / "model.safetensors"
    )
    expected.update(frozen)

    class MetadataOnly:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self.handle = safe_open(*args, **kwargs)

        def __enter__(self) -> "MetadataOnly":
            self.handle.__enter__()
            return self

        def __exit__(self, *args: Any) -> Any:
            return self.handle.__exit__(*args)

        def get_tensor(self, name: str) -> torch.Tensor:
            raise AssertionError(f"Frozen payload was materialized: {name}")

    monkeypatch.setattr(safetensors, "safe_open", MetadataOnly)
    output = tmp_path / "export"
    result = export_checkpoint(
        checkpoint, model, output, shard_bytes=128, memory_budget=1024**2, allow_base_fallback=("vision",)
    )
    assert result["copy_buffer_bytes"] <= 64 * 1024**2
    index = json.loads((output / "model.safetensors.index.json").read_text())
    assert index["metadata"]["total_size"] == sum(value.numel() * value.element_size() for value in expected.values())
    assert index["weight_map"].keys() == expected.keys()
    for name, value in expected.items():
        with safe_open(output / index["weight_map"][name], framework="pt", device="cpu") as handle:
            actual = handle.get_tensor(name)
            assert actual.dtype == value.dtype and torch.equal(actual, value), name


@pytest.mark.parametrize("corruption", ["offset", "length", "overlap", "dtype", "truncated"])
@_cpu_only
def test_cpu_export_rejects_invalid_source_headers_before_raw_copy(tmp_path: Path, corruption: str) -> None:
    import struct

    from safetensors import SafetensorError

    checkpoint, model, _ = _fixture(tmp_path)
    source = model / "model.safetensors"
    data = source.read_bytes()
    length = struct.unpack("<Q", data[:8])[0]
    header, payload = json.loads(data[8 : 8 + length]), data[8 + length :]
    entries = [entry for key, entry in header.items() if key != "__metadata__"]
    if corruption == "offset":
        entries[0]["data_offsets"][0] = -1
    elif corruption == "length":
        entries[0]["data_offsets"][1] += 1
    elif corruption == "overlap":
        entries[1]["data_offsets"] = entries[0]["data_offsets"]
    elif corruption == "dtype":
        entries[0]["dtype"] = "UNKNOWN"
    else:
        payload = payload[:-1]
    raw = json.dumps(header).encode()
    raw += b" " * (-len(raw) % 8)
    source.write_bytes(struct.pack("<Q", len(raw)) + raw + payload)
    with pytest.raises(SafetensorError):
        export_checkpoint(checkpoint, model, tmp_path / "export", allow_base_fallback=("vision",))
    assert not list(tmp_path.glob("export*"))


@_cpu_only
def test_cpu_export_frozen_stream_detects_early_eof(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    checkpoint, model, expected = _fixture(tmp_path)
    expected["model.visual.weight"] = torch.arange(8.0)
    save_file(expected, model / "model.safetensors")
    original_copy = exporter.ShardWriter._copy_ranges

    def truncate_then_copy(
        self: exporter.ShardWriter,
        ranges: list[tuple[str, Path, int, int]],
        schema: dict[str, dict[str, Any]],
        buffer: bytearray,
    ) -> None:
        _, path, offset, size = ranges[0]
        with path.open("r+b") as handle:
            handle.truncate(offset + size - 1)
        return original_copy(self, ranges, schema, buffer)

    monkeypatch.setattr(exporter.ShardWriter, "_copy_ranges", truncate_then_copy)
    output = tmp_path / "export"
    with pytest.raises(ValueError, match="Truncated frozen tensor"):
        export_checkpoint(checkpoint, model, output, allow_base_fallback=("vision",))
    assert not output.exists()
    assert not list(tmp_path.glob("export.incomplete-*/model.safetensors.index.json"))


@_cpu_only
def test_cpu_dcp_rejects_holes_and_overlaps() -> None:
    properties = TensorProperties(dtype=torch.float32)
    hole = TensorStorageMetadata(
        properties, torch.Size((4,)), [ChunkStorageMetadata(torch.Size((0,)), torch.Size((2,)))]
    )
    with pytest.raises(ValueError, match="Incomplete"):
        check_chunks("weight", hole)
    overlap = TensorStorageMetadata(
        properties,
        torch.Size((4,)),
        [
            ChunkStorageMetadata(torch.Size((0,)), torch.Size((2,))),
            ChunkStorageMetadata(torch.Size((1,)), torch.Size((2,))),
        ],
    )
    with pytest.raises(ValueError, match="Overlapping"):
        check_chunks("weight", overlap)


@_cpu_only
def test_cpu_checkpoint_resolves_selected_iteration(tmp_path: Path) -> None:
    checkpoint, _, _ = _fixture(tmp_path)
    assert resolve_checkpoint(checkpoint.parent) == checkpoint
    (checkpoint.parent / "latest_checkpointed_iteration.txt").write_text("999")
    assert resolve_checkpoint(checkpoint) == checkpoint
    with pytest.raises(FileNotFoundError):
        resolve_checkpoint(checkpoint.parent)


def test_cpu_export_isolates_initialized_parent(tmp_path: Path) -> None:
    code = """
import sys
import pytest
import torch

if torch.cuda.is_available():
    torch.cuda.init()
cuda_initialized = torch.cuda.is_initialized()
torch.distributed.init_process_group("gloo", init_method=sys.argv[2], rank=0, world_size=1)
group = torch.distributed.group.WORLD
try:
    result = pytest.main([sys.argv[1] + "::test_cpu_export_round_trip_without_model_or_cuda", "-q", "--tb=short", "--show-capture=no"])
    assert torch.distributed.is_initialized(), "CPU tests changed the parent process group"
    assert torch.distributed.group.WORLD is group, "CPU tests replaced the parent process group"
    torch.distributed.all_reduce(torch.ones(1), group=group)
    assert torch.cuda.is_initialized() == cuda_initialized, "CPU tests changed the parent CUDA state"
finally:
    torch.distributed.destroy_process_group()
sys.exit(result)
"""
    env = dict(os.environ)
    env.pop("_RELAX_CPU_EXPORT_TEST_CHILD", None)
    env.pop("PYTEST_ADDOPTS", None)
    result = subprocess.run(
        [sys.executable, "-c", code, str(Path(__file__).resolve()), (tmp_path / "store").as_uri()],
        cwd=Path(__file__).resolve().parents[2],
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert result.returncode == 0, result.stdout + result.stderr
