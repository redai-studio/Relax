# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Frozen-reference checksum, probe, optimizer and sidecar tests."""

import hashlib
import json
import os
import sys
import types
from argparse import Namespace

import pytest
import torch

from relax.backends.megatron.reference_integrity import (
    DPOReferenceIdentity,
    canonical_optimizer_sha256,
    canonical_tensor_sha256,
    read_reference_identity,
    reference_probe_sha256,
    resolve_dpo_reference_checkpoint,
    write_reference_identity,
)


def test_megatron_resume_detection_ignores_fresh_output_directory(tmp_path):
    pytest.importorskip("megatron.training.checkpointing")
    from relax.backends.megatron.checkpoint import is_megatron_checkpoint

    output = tmp_path / "run"
    output.mkdir()
    (output / "transformer_config.json").write_text("{}", encoding="utf-8")
    assert not is_megatron_checkpoint(output)
    (output / "latest_checkpointed_iteration.txt").write_text("1", encoding="utf-8")
    assert is_megatron_checkpoint(output)
    assert is_megatron_checkpoint(tmp_path / "iter_0000001")


def test_canonical_tensor_digest_is_order_stable_and_byte_sensitive():
    first = canonical_tensor_sha256([("b", torch.tensor([2.0])), ("a", torch.tensor([1.0]))])
    reordered = canonical_tensor_sha256([("a", torch.tensor([1.0])), ("b", torch.tensor([2.0]))])
    changed = canonical_tensor_sha256([("a", torch.tensor([1.0])), ("b", torch.tensor([3.0]))])
    assert first == reordered
    assert first != changed
    assert first != canonical_tensor_sha256([("a", torch.tensor([1], dtype=torch.int64)), ("b", torch.tensor([2.0]))])


def test_optimizer_digest_detects_master_or_state_changes():
    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = torch.optim.Adam([parameter], lr=0.1)
    (parameter.square().sum()).backward()
    optimizer.step()
    baseline = canonical_optimizer_sha256(optimizer)
    master_value = parameter.detach().clone()
    parameter.data.add_(1)
    master_changed = canonical_optimizer_sha256(optimizer)
    assert master_changed != baseline
    parameter.data.copy_(master_value)
    optimizer.state[parameter]["exp_avg"].add_(1)
    assert canonical_optimizer_sha256(optimizer) != baseline


def test_probe_digest_covers_identity_tokens_masks_and_fp32_logprobs():
    args = ([1, 1], [True, False], [[1, 2], [1, 3]], [[0, 1], [0, 1]])
    baseline = reference_probe_sha256(*args, [[0.0, -1.0], [0.0, -2.0]])
    assert baseline != reference_probe_sha256(*args, [[0.0, -1.0], [0.0, -2.1]])
    assert baseline != reference_probe_sha256([2, 2], *args[1:], [[0.0, -1.0], [0.0, -2.0]])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for cross-device probe coverage")
def test_probe_digest_accepts_gpu_logprobs_with_cpu_manifest():
    cpu_digest = reference_probe_sha256([1], [True], [[1, 2]], [[0, 1]], [[0.0, -1.0]])
    gpu_digest = reference_probe_sha256([1], [True], [[1, 2]], [[0, 1]], [torch.tensor([0.0, -1.0], device="cuda")])
    assert gpu_digest == cpu_digest


def test_reference_identity_sidecar_is_required_and_rejects_schema_damage(tmp_path):
    path = tmp_path / "relax_dpo_reference.json"
    with pytest.raises(FileNotFoundError):
        read_reference_identity(path)
    identity = DPOReferenceIdentity(1, "repo", "revision", "loader", "a" * 64, "b" * 64)
    write_reference_identity(path, identity)
    assert read_reference_identity(path) == identity
    payload = identity.to_dict()
    payload["schema_version"] = 99
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="unsupported"):
        read_reference_identity(path)


def _write_local_download_metadata(checkpoint, filename, revision):
    source = checkpoint / filename
    metadata = checkpoint / ".cache" / "huggingface" / "download" / f"{filename}.metadata"
    metadata.parent.mkdir(parents=True, exist_ok=True)
    content = source.read_bytes()
    if source.suffix == ".safetensors":
        etag = hashlib.sha256(content).hexdigest()
    else:
        etag = hashlib.sha1(f"blob {len(content)}\0".encode() + content).hexdigest()
    metadata.write_text(f"{revision}\n{etag}\n{source.stat().st_mtime}\n", encoding="utf-8")


def test_resolve_dpo_reference_checkpoint_uses_the_pinned_configured_local_snapshot(monkeypatch, tmp_path):
    revision = "a" * 40
    snapshot = tmp_path / f"Qwen3-0.6B-{revision}"
    snapshot.mkdir()
    (snapshot / "config.json").write_text("{}", encoding="utf-8")
    (snapshot / "model.safetensors").write_bytes(b"weights")
    _write_local_download_metadata(snapshot, "config.json", revision)
    _write_local_download_metadata(snapshot, "model.safetensors", revision)
    observed = {}

    def snapshot_download(**kwargs):
        observed.update(kwargs)
        return str(snapshot)

    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        types.SimpleNamespace(snapshot_download=snapshot_download),
    )
    assert resolve_dpo_reference_checkpoint("org/model", revision, str(snapshot)) == str(snapshot.resolve())
    assert observed == {
        "repo_id": "org/model",
        "revision": revision,
        "local_dir": str(snapshot.resolve()),
        "local_files_only": True,
    }


def test_resolve_dpo_reference_checkpoint_rejects_missing_local_snapshot(monkeypatch, tmp_path):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()

    def snapshot_download(**_kwargs):
        raise OSError("not cached")

    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        types.SimpleNamespace(snapshot_download=snapshot_download),
    )
    with pytest.raises(RuntimeError, match="hf download org/model --revision"):
        resolve_dpo_reference_checkpoint("org/model", "a" * 40, str(checkpoint))


def test_resolve_dpo_reference_checkpoint_rejects_different_resolved_directory(monkeypatch, tmp_path):
    checkpoint = tmp_path / "checkpoint"
    other = tmp_path / "other"
    checkpoint.mkdir()
    other.mkdir()
    (checkpoint / "config.json").write_text("{}", encoding="utf-8")
    (other / "config.json").write_text("{}", encoding="utf-8")

    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        types.SimpleNamespace(snapshot_download=lambda **_kwargs: str(other)),
    )
    with pytest.raises(RuntimeError, match="different from --hf-checkpoint"):
        resolve_dpo_reference_checkpoint("org/model", "a" * 40, str(checkpoint))


def test_resolve_dpo_reference_checkpoint_rejects_missing_pinned_local_metadata(monkeypatch, tmp_path):
    revision = "a" * 40
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text("{}", encoding="utf-8")
    (checkpoint / "model.safetensors").write_bytes(b"weights")
    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        types.SimpleNamespace(snapshot_download=lambda **_kwargs: str(checkpoint)),
    )
    with pytest.raises(RuntimeError, match="missing valid Hugging Face local-dir metadata"):
        resolve_dpo_reference_checkpoint("org/model", revision, str(checkpoint))


def test_resolve_dpo_reference_checkpoint_rejects_mismatched_file_metadata(monkeypatch, tmp_path):
    revision = "a" * 40
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text("{}", encoding="utf-8")
    _write_local_download_metadata(checkpoint, "config.json", "c" * 40)
    (checkpoint / "model.safetensors").write_bytes(b"weights")
    _write_local_download_metadata(checkpoint, "model.safetensors", revision)

    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        types.SimpleNamespace(snapshot_download=lambda **_kwargs: str(checkpoint)),
    )
    with pytest.raises(RuntimeError, match="metadata does not match the pinned revision"):
        resolve_dpo_reference_checkpoint("org/model", revision, str(checkpoint))


def test_resolve_dpo_reference_checkpoint_rejects_replaced_file_with_restored_mtime(monkeypatch, tmp_path):
    revision = "a" * 40
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    config = checkpoint / "config.json"
    config.write_text("{}", encoding="utf-8")
    _write_local_download_metadata(checkpoint, "config.json", revision)
    weights = checkpoint / "model.safetensors"
    weights.write_bytes(b"weights")
    original_stat = weights.stat()
    _write_local_download_metadata(checkpoint, "model.safetensors", revision)
    weights.write_bytes(b"changed")
    os.utime(weights, ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns))

    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        types.SimpleNamespace(snapshot_download=lambda **_kwargs: str(checkpoint)),
    )
    with pytest.raises(RuntimeError, match="contents do not match its Hugging Face ETag"):
        resolve_dpo_reference_checkpoint("org/model", revision, str(checkpoint))


def test_resolve_dpo_reference_checkpoint_rejects_snapshot_without_supported_weights(monkeypatch, tmp_path):
    revision = "a" * 40
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text("{}", encoding="utf-8")
    _write_local_download_metadata(checkpoint, "config.json", revision)

    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        types.SimpleNamespace(snapshot_download=lambda **_kwargs: str(checkpoint)),
    )
    with pytest.raises(RuntimeError, match="no supported model weights or index"):
        resolve_dpo_reference_checkpoint("org/model", revision, str(checkpoint))


def test_resolve_dpo_reference_checkpoint_accepts_complete_safetensors_index(monkeypatch, tmp_path):
    revision = "a" * 40
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text("{}", encoding="utf-8")
    index_name = "model.safetensors.index.json"
    shard_names = ("model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors")
    weight_map = {f"layer.{index}.weight": shard for index, shard in enumerate(shard_names)}
    (checkpoint / index_name).write_text(json.dumps({"weight_map": weight_map}), encoding="utf-8")
    for shard_name in shard_names:
        (checkpoint / shard_name).write_bytes(shard_name.encode())
    for filename in ("config.json", index_name, *shard_names):
        _write_local_download_metadata(checkpoint, filename, revision)

    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        types.SimpleNamespace(snapshot_download=lambda **_kwargs: str(checkpoint)),
    )
    assert resolve_dpo_reference_checkpoint("org/model", revision, str(checkpoint)) == str(checkpoint.resolve())


def test_resolve_dpo_reference_checkpoint_rejects_missing_indexed_shard_and_metadata(monkeypatch, tmp_path):
    revision = "a" * 40
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text("{}", encoding="utf-8")
    index_name = "model.safetensors.index.json"
    shard_name = "model-00001-of-00001.safetensors"
    (checkpoint / index_name).write_text(json.dumps({"weight_map": {"model.weight": shard_name}}), encoding="utf-8")
    (checkpoint / shard_name).write_bytes(b"weights")
    for filename in ("config.json", index_name, shard_name):
        _write_local_download_metadata(checkpoint, filename, revision)
    (checkpoint / shard_name).unlink()
    (checkpoint / ".cache" / "huggingface" / "download" / f"{shard_name}.metadata").unlink()

    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        types.SimpleNamespace(snapshot_download=lambda **_kwargs: str(checkpoint)),
    )
    with pytest.raises(RuntimeError, match="weight index points to a missing shard"):
        resolve_dpo_reference_checkpoint("org/model", revision, str(checkpoint))


def test_resume_probe_materializes_manifest_lists_as_tensors(monkeypatch):
    try:
        from relax.backends.megatron import actor as actor_module
    except Exception as exc:
        pytest.skip(f"Megatron actor unavailable: {exc}")

    instance = object.__new__(actor_module.MegatronTrainRayActor)
    instance.args = Namespace()
    instance.model = [object()]
    instance._dpo_reference_probe_verified = False
    instance._expected_dpo_reference_identity = DPOReferenceIdentity(
        1,
        "repo",
        "revision",
        "loader",
        "a" * 64,
        "b" * 64,
        {
            "pair_ids": [1, 1],
            "branch_is_chosen": [True, False],
            "tokens": [[1, 2], [1, 3]],
            "loss_masks": [[False, True], [False, True]],
            "total_lengths": [2, 2],
            "response_lengths": [1, 1],
        },
    )

    def inspect_probe_data(_args, _model, probe_data):
        assert all(torch.is_tensor(value) and value.dtype == torch.long for value in probe_data["tokens"])
        assert all(torch.is_tensor(value) and value.dtype == torch.bool for value in probe_data["loss_masks"])
        raise RuntimeError("probe inspected")

    monkeypatch.setattr(actor_module, "get_data_iterator", inspect_probe_data)
    monkeypatch.setattr(actor_module.mpu, "get_data_parallel_world_size", lambda **_kwargs: 1)
    monkeypatch.setattr(actor_module.device_utils, "make_current_torch_device", lambda: torch.device("cpu"))
    with pytest.raises(RuntimeError, match="probe inspected"):
        instance._replay_dpo_reference_probe()


def test_loader_half_write_failure_restores_actor_and_keeps_optimizer(monkeypatch):
    try:
        from relax.backends.megatron import actor as actor_module
    except Exception as exc:
        pytest.skip(f"Megatron actor unavailable: {exc}")

    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = torch.optim.Adam([parameter], lr=0.1)

    class _Backuper:
        def __init__(self):
            self.values = {"actor": {"weight": parameter.detach().clone()}}

        @property
        def backup_tags(self):
            return list(self.values)

        def restore(self, tag):
            parameter.data.copy_(self.values[tag]["weight"])

        def backup(self, tag):
            self.values[tag] = {"weight": parameter.detach().clone()}

        def get(self, tag):
            return self.values[tag]

    instance = object.__new__(actor_module.MegatronTrainRayActor)
    instance.args = Namespace(
        load="checkpoint",
        no_load_optim=False,
        no_load_rng=False,
        finetune=False,
        megatron_to_hf_mode="bridge",
        dpo_reference_repository="repo",
        dpo_reference_revision="revision",
    )
    instance.model = [object()]
    instance.optimizer = optimizer
    instance.weights_backuper = _Backuper()
    instance._active_model_tag = "actor"
    instance._expected_dpo_reference_identity = None
    monkeypatch.setattr(actor_module.device_utils, "maybe_backend_process_on_model_switch", lambda: None)

    def fail_after_half_write(*args, **kwargs):
        parameter.data.fill_(99)
        raise RuntimeError("injected loader failure")

    monkeypatch.setattr(actor_module, "load_checkpoint", fail_after_half_write)
    before = canonical_optimizer_sha256(optimizer)
    with pytest.raises(RuntimeError, match="injected loader failure"):
        instance._rebuild_dpo_reference("hf-path")
    assert parameter.item() == 1.0
    assert instance._active_model_tag == "actor"
    assert "ref" not in instance.weights_backuper.backup_tags
    assert canonical_optimizer_sha256(optimizer) == before
