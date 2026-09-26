# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from relax.engine.lora.snapshot import AdapterConflictError, snapshot_adapter


BASE_DIGEST = "a" * 64


def _export(path: Path, weights: bytes = b"fixture tensor bytes") -> Path:
    path.mkdir(parents=True, exist_ok=True)
    (path / "adapter_config.json").write_text(json.dumps({"r": 4, "lora_alpha": 8}))
    # These CPU tests exercise artifact handling, not the safetensors parser.
    (path / "adapter_model.safetensors").write_bytes(weights)
    return path


def test_snapshot_persists_new_store_ancestor_entries(tmp_path, monkeypatch):
    from relax.engine.lora import snapshot as module

    synced = []
    real_sync = module._sync_directory

    def sync(path):
        synced.append(path)
        real_sync(path)

    monkeypatch.setattr(module, "_sync_directory", sync)
    module.snapshot_adapter(
        _export(tmp_path / "source"),
        tmp_path / "new" / "nested" / "store",
        version_id="A",
        base_model_digest=BASE_DIGEST,
    )
    assert {tmp_path, tmp_path / "new", tmp_path / "new" / "nested"}.issubset(synced)


def test_snapshot_adapter_seals_completed_export_and_normalizes_config(tmp_path):
    source = _export(tmp_path / "export")
    first = snapshot_adapter(source, tmp_path / "store", version_id="A", base_model_digest=BASE_DIGEST)
    (source / "adapter_config.json").write_text('{"lora_alpha":8,"r":4}')
    assert snapshot_adapter(source, tmp_path / "store", version_id="A", base_model_digest=BASE_DIGEST) == first
    second = snapshot_adapter(source, tmp_path / "store", version_id="B", base_model_digest=BASE_DIGEST)
    assert first.digest == second.digest
    assert first.lora_path != second.lora_path
    (source / "adapter_model.safetensors").write_bytes(b"training moved on")
    first.verify()
    assert (first.path / "adapter_model.safetensors").read_bytes() == b"fixture tensor bytes"
    assert not list((tmp_path / "store" / ".staging").iterdir())
    assert first.path == tmp_path / "store" / "versions" / "A"
    assert json.loads((first.path / "manifest.json").read_text())["content_digest"] == first.digest


@pytest.mark.parametrize("change", ["weights", "config", "base"])
def test_snapshot_adapter_rejects_same_id_different_content(tmp_path, change):
    source = _export(tmp_path / "export")
    first = snapshot_adapter(source, tmp_path / "store", version_id="A", base_model_digest=BASE_DIGEST)
    base = BASE_DIGEST
    if change == "weights":
        (source / "adapter_model.safetensors").write_bytes(b"different")
    elif change == "config":
        (source / "adapter_config.json").write_text('{"r": 8}')
    else:
        base = "b" * 64
    with pytest.raises(AdapterConflictError):
        snapshot_adapter(source, tmp_path / "store", version_id="A", base_model_digest=base)
    first.verify()


@pytest.mark.parametrize("name", ["adapter_model.safetensors", "adapter_config.json", "manifest.json"])
def test_snapshot_adapter_detects_tampering(tmp_path, name):
    artifact = snapshot_adapter(
        _export(tmp_path / "export"), tmp_path / "store", version_id="A", base_model_digest=BASE_DIGEST
    )
    target = artifact.path / name
    target.chmod(0o644)
    target.write_bytes(b"{}")
    with pytest.raises(ValueError, match="checksum"):
        artifact.verify()


@pytest.mark.parametrize("version_id", ["", "..", "../A", "/A", "a/b", "a" * 129])
def test_snapshot_adapter_rejects_invalid_version_id(tmp_path, version_id):
    with pytest.raises(ValueError, match="version_id"):
        snapshot_adapter(tmp_path, tmp_path / "store", version_id=version_id, base_model_digest=BASE_DIGEST)


def test_snapshot_adapter_duplicate_concurrent_writers_do_not_replace_artifact(tmp_path):
    source = _export(tmp_path / "export")

    def seal(_):
        return snapshot_adapter(source, tmp_path / "store", version_id="A", base_model_digest=BASE_DIGEST)

    with ThreadPoolExecutor(max_workers=2) as pool:
        first, second = list(pool.map(seal, range(2)))
    assert first == second
    first.verify()
    assert not list((tmp_path / "store" / ".staging").iterdir())


def test_snapshot_adapter_rejects_symlink_source(tmp_path):
    source = _export(tmp_path / "export")
    weights = source / "adapter_model.safetensors"
    weights.rename(source / "other")
    weights.symlink_to(source / "other")
    with pytest.raises(ValueError, match="regular file"):
        snapshot_adapter(source, tmp_path / "store", version_id="A", base_model_digest=BASE_DIGEST)


def test_snapshot_audit_changes_do_not_change_content_identity(tmp_path):
    source = _export(tmp_path / "export")
    metadata = {"contract": "dense-v1", "base_config_digest": "b" * 64, "export_step": 3, "producer": "run-a"}
    (source / "producer_manifest.json").write_text(json.dumps(metadata))
    first = snapshot_adapter(source, tmp_path / "store", version_id="A", base_model_digest=BASE_DIGEST)
    metadata.update(export_step=9, producer="run-b")
    (source / "producer_manifest.json").write_text(json.dumps(metadata))
    again = snapshot_adapter(source, tmp_path / "store", version_id="A", base_model_digest=BASE_DIGEST)
    assert again == first
    assert json.loads((again.path / "producer_manifest.json").read_text())["export_step"] == 3
    # Audit bytes themselves still have integrity protection.
    audit = first.path / "producer_manifest.json"
    audit.chmod(0o644)
    audit.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="checksum"):
        first.verify()


def test_snapshot_contract_changes_conflict_even_with_identical_weights(tmp_path):
    source = _export(tmp_path / "export")
    sidecar = source / "producer_manifest.json"
    sidecar.write_text(json.dumps({"contract": "dense-v1"}))
    snapshot_adapter(source, tmp_path / "store", version_id="A", base_model_digest=BASE_DIGEST)
    sidecar.write_text(json.dumps({"contract": "dense-v2"}))
    with pytest.raises(AdapterConflictError):
        snapshot_adapter(source, tmp_path / "store", version_id="A", base_model_digest=BASE_DIGEST)


def test_snapshot_rename_then_sync_failure_is_unknown_and_retry_repairs(tmp_path, monkeypatch):
    import relax.engine.lora.snapshot as module

    source = _export(tmp_path / "export")
    store = tmp_path / "store"
    sync = module._sync_directory

    def fail_parent(directory):
        if directory == store / "versions":
            raise OSError("injected directory fsync failure")
        sync(directory)

    monkeypatch.setattr(module, "_sync_directory", fail_parent)
    with pytest.raises(module.AdapterSealUnknownError, match="SEAL_COMMIT_UNKNOWN"):
        snapshot_adapter(source, store, version_id="A", base_model_digest=BASE_DIGEST)
    visible = store / "versions" / "A"
    assert visible.is_dir()
    inode = visible.stat().st_ino
    assert list((store / ".staging").iterdir()) == []
    monkeypatch.setattr(module, "_sync_directory", sync)
    repaired = snapshot_adapter(source, store, version_id="A", base_model_digest=BASE_DIGEST)
    assert repaired.path.stat().st_ino == inode
    repaired.confirm_sealed()


def test_snapshot_capacity_rejects_new_version_but_keeps_retry_idempotent(tmp_path):
    from relax.engine.lora.snapshot import AdapterArtifactCapacityError

    source = _export(tmp_path / "export")
    store = tmp_path / "store"
    first = snapshot_adapter(source, store, version_id="A", base_model_digest=BASE_DIGEST)
    assert snapshot_adapter(source, store, version_id="A", base_model_digest=BASE_DIGEST, max_bytes=1) == first
    with pytest.raises(AdapterArtifactCapacityError, match="ARTIFACT_CAPACITY_EXCEEDED"):
        snapshot_adapter(source, store, version_id="B", base_model_digest=BASE_DIGEST, max_bytes=1)
    assert not (store / "versions" / "B").exists()
    first.verify()


def test_snapshot_concurrent_conflicting_writers_keep_exactly_one_version(tmp_path):
    sources = [_export(tmp_path / str(i), bytes([i])) for i in range(2)]

    def seal(source):
        try:
            return snapshot_adapter(source, tmp_path / "store", version_id="A", base_model_digest=BASE_DIGEST)
        except AdapterConflictError as error:
            return error

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(seal, sources))
    assert sum(isinstance(result, AdapterConflictError) for result in results) == 1
    winner = next(result for result in results if not isinstance(result, AdapterConflictError))
    winner.verify()
    assert list((tmp_path / "store" / "versions").iterdir()) == [winner.path]


@pytest.mark.parametrize("format_version", [1, 3, None])
def test_read_snapshot_rejects_unsupported_manifest_format(tmp_path, format_version):
    from relax.engine.lora.snapshot import read_snapshot

    source = _export(tmp_path / "export")
    (source / "manifest.json").write_text(json.dumps({"format_version": format_version}))
    with pytest.raises(ValueError, match="unsupported.*format"):
        read_snapshot(source)


def test_snapshot_rejects_old_store_without_reusing_or_modifying_version(tmp_path):
    store = tmp_path / "store"
    legacy = _export(store / "A")
    original = {p.name: p.read_bytes() for p in legacy.iterdir()}
    source = _export(tmp_path / "source")
    with pytest.raises(ValueError, match="unsupported adapter store layout"):
        snapshot_adapter(source, store, version_id="A", base_model_digest=BASE_DIGEST)
    assert {p.name: p.read_bytes() for p in legacy.iterdir()} == original
    assert not (store / "versions" / "A").exists()
    assert not list((store / ".staging").iterdir())


def test_snapshot_real_export_provenance_and_unknown_step(tmp_path):
    torch = pytest.importorskip("torch", reason="real PEFT writer requires PyTorch")
    pytest.importorskip("safetensors.torch")
    from relax.engine.lora.artifact import PROVENANCE, ModelContract, canonical, provenance
    from relax.utils.megatron_peft_utils import write_hf_peft_adapter

    config = {
        "model_type": "qwen2",
        "hidden_size": 8,
        "intermediate_size": 16,
        "num_attention_heads": 2,
        "num_hidden_layers": 1,
    }
    contract = ModelContract(BASE_DIGEST, config, 2, ("k_proj", "v_proj"))
    weights = {key: torch.ones(shape) for key, shape in contract._shapes(2, ["k_proj", "v_proj"]).items()}
    source = Path(
        write_hf_peft_adapter(
            weights,
            tmp_path / "real-export",
            lora_rank=2,
            lora_alpha=4,
            target_modules=["k_proj", "v_proj"],
            lora_dropout=0.0,
        )
    )
    (source / PROVENANCE).write_bytes(
        canonical(provenance(contract, source, producer="test-existing-writer", export_step=None))
    )
    contract.validate(source)
    first = snapshot_adapter(source, tmp_path / "store", version_id="A", base_model_digest=BASE_DIGEST)
    assert json.loads((first.path / PROVENANCE).read_bytes())["export_step"] is None
    for weight in weights.values():
        weight.add_(1)
    write_hf_peft_adapter(
        weights, source, lora_rank=2, lora_alpha=4, target_modules=["k_proj", "v_proj"], lora_dropout=0.0
    )
    first.verify()
    contract.validate(first.path)
    with pytest.raises(ValueError, match="producer manifest"):
        contract.validate(source)


def test_snapshot_reserves_capacity_before_weight_copy(tmp_path, monkeypatch):
    from relax.engine.lora import snapshot as module

    source = _export(tmp_path / "source")
    original_open = Path.open

    def guard(path, *args, **kwargs):
        if path == source / "adapter_model.safetensors":
            pytest.fail("capacity rejection must precede reading or copying weights")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guard)
    with pytest.raises(module.AdapterArtifactCapacityError):
        snapshot_adapter(source, tmp_path / "store", version_id="A", base_model_digest=BASE_DIGEST, max_bytes=1)
    assert not list((tmp_path / "store" / ".staging").iterdir())


def test_snapshot_concurrent_reservations_cannot_overspend(tmp_path, monkeypatch):
    import threading

    from relax.engine.lora import snapshot as module

    source = _export(tmp_path / "source")
    probe = snapshot_adapter(source, tmp_path / "probe", version_id="A", base_model_digest=BASE_DIGEST)
    budget = sum(p.stat().st_size for p in probe.path.iterdir())
    store = tmp_path / "store"
    entered, release = threading.Event(), threading.Event()
    original_manifest = module._manifest

    def held_manifest(directory, version, base):
        if directory.parent == store / ".staging":
            assert version == "A", "second copy must be rejected before staging"
            entered.set()
            assert release.wait(5)
        return original_manifest(directory, version, base)

    monkeypatch.setattr(module, "_manifest", held_manifest)
    with ThreadPoolExecutor(max_workers=1) as pool:
        pending = pool.submit(
            snapshot_adapter, source, store, version_id="A", base_model_digest=BASE_DIGEST, max_bytes=budget
        )
        try:
            assert entered.wait(5)
            with pytest.raises(module.AdapterArtifactCapacityError):
                snapshot_adapter(source, store, version_id="B", base_model_digest=BASE_DIGEST, max_bytes=budget)
        finally:
            release.set()
        sealed = pending.result(timeout=5)
    assert sum(p.stat().st_size for p in sealed.path.iterdir()) == budget
    assert not (sealed.path / ".reserved_bytes").exists()
    assert not list((store / ".staging").iterdir())


def test_snapshot_failure_releases_reservation_and_retry_needs_no_copy(tmp_path, monkeypatch):
    from relax.engine.lora import snapshot as module

    source = _export(tmp_path / "source")
    store = tmp_path / "store"
    original_manifest = module._manifest

    def fail_after_copy(directory, version, base):
        if directory.parent == store / ".staging":
            raise OSError("injected failure after copy")
        return original_manifest(directory, version, base)

    monkeypatch.setattr(module, "_manifest", fail_after_copy)
    with pytest.raises(OSError, match="injected"):
        snapshot_adapter(source, store, version_id="A", base_model_digest=BASE_DIGEST, max_bytes=10000)
    assert module._store_bytes(store) == 0
    assert not list((store / ".staging").iterdir())
    monkeypatch.setattr(module, "_manifest", original_manifest)
    sealed = snapshot_adapter(source, store, version_id="A", base_model_digest=BASE_DIGEST, max_bytes=10000)
    # A full store must still verify a same-content retry, without a staged copy.
    monkeypatch.setattr(module, "_manifest", fail_after_copy)
    assert snapshot_adapter(source, store, version_id="A", base_model_digest=BASE_DIGEST, max_bytes=1) == sealed


def test_snapshot_keeps_orphan_reservation_charged(tmp_path):
    from relax.engine.lora import snapshot as module

    store = tmp_path / "store"
    orphan = store / ".staging" / "export-interrupted"
    orphan.mkdir(parents=True)
    (orphan / ".reserved_bytes").write_text("10000")
    source = _export(tmp_path / "source")
    with pytest.raises(module.AdapterArtifactCapacityError):
        snapshot_adapter(source, store, version_id="A", base_model_digest=BASE_DIGEST, max_bytes=10000)
    assert (orphan / ".reserved_bytes").read_text() == "10000"


def test_snapshot_source_growth_cannot_exceed_reserved_copy(tmp_path, monkeypatch):
    from relax.engine.lora import snapshot as module

    source = _export(tmp_path / "source")
    store = tmp_path / "store"
    original_open = Path.open

    def grow_on_read(path, *args, **kwargs):
        if path == source / "adapter_model.safetensors" and args == ("rb",):
            with original_open(path, "ab") as writer:
                writer.write(b"unexpected growth")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", grow_on_read)
    with pytest.raises(ValueError, match="changed during snapshot"):
        snapshot_adapter(source, store, version_id="A", base_model_digest=BASE_DIGEST, max_bytes=10000)
    assert module._store_bytes(store) == 0
    assert not list((store / ".staging").iterdir())
