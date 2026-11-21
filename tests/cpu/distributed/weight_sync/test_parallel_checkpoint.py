"""Parallel checkpoint validation retains byte, size, and deadline checks."""

import os
import threading
import time

import pytest

from relax.distributed.weight_sync import checkpoint as module
from relax.distributed.weight_sync.deadline import time_budget
from relax.distributed.weight_sync.store import VersionStore


@pytest.fixture
def checkpoint(tmp_path):
    store = VersionStore(tmp_path / "store", disk_budget=1 << 22, max_weight_bytes=1 << 20)
    tensors = {}
    for n in range(12):
        path = tmp_path / f"tensor-{n}"
        path.write_bytes(bytes([n]) * 4096)
        tensors[f"w{n:02}"] = ("U8", [4096], path)
    manifest = store.publish("s", 0, tensors, model_config={})
    path = module.write_checkpoint(store, "s", 0, tmp_path / "checkpoint")
    return path, manifest


@pytest.mark.parametrize("position", [0, 4095, 4096, 49151])
def test_every_chunk_and_boundary_is_verified(checkpoint, position):
    path, manifest = checkpoint
    module.validate_checkpoint(path, manifest)
    file = path / "model.safetensors"
    value = bytearray(file.read_bytes())
    start = 8 + int.from_bytes(value[:8], "little")
    value[start + position] ^= 1
    file.write_bytes(value)
    with pytest.raises(ValueError, match="content differs"):
        module.validate_checkpoint(path, manifest)


@pytest.mark.parametrize("change", [-1, 1])
def test_checkpoint_exact_size_is_required(checkpoint, change):
    path, manifest = checkpoint
    file = path / "model.safetensors"
    data = file.read_bytes()
    file.write_bytes(data[:-1] if change < 0 else data + b"x")
    with pytest.raises(ValueError, match="size differs"):
        module.validate_checkpoint(path, manifest)


def test_parallelism_is_bounded_and_deadline_reaches_workers(checkpoint, monkeypatch):
    path, manifest = checkpoint
    original = os.pread
    active, peak, lock = 0, 0, threading.Lock()

    def slow_read(*args):
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
        try:
            time.sleep(0.04)
            return original(*args)
        finally:
            with lock:
                active -= 1

    monkeypatch.setattr(module.os, "pread", slow_read)
    module.validate_checkpoint(path, manifest)
    assert 1 < peak <= 8
    with time_budget(0.01), pytest.raises(TimeoutError, match="checkpoint verification"):
        module.validate_checkpoint(path, manifest)
    assert active == 0


def test_worker_io_failure_propagates_and_next_validation_recovers(checkpoint, monkeypatch):
    path, manifest = checkpoint
    original = os.pread

    def fail(*args):
        raise OSError("injected pread failure")

    monkeypatch.setattr(module.os, "pread", fail)
    with pytest.raises(OSError, match="injected pread"):
        module.validate_checkpoint(path, manifest)
    monkeypatch.setattr(module.os, "pread", original)
    module.validate_checkpoint(path, manifest)


def test_checkpoint_retains_source_while_concurrent_collection_runs(tmp_path, monkeypatch):
    store = VersionStore(tmp_path / "store", disk_budget=1 << 22, max_weight_bytes=1 << 20)
    for version in range(3):
        source = tmp_path / f"tensor-{version}"
        source.write_bytes(bytes([version]) * 64)
        store.publish("s", version, {"w": ("U8", [64], source)}, full_only=True, model_config={})

    original_iter = store.iter_tensor
    collections = []

    def collect_before_read(manifest, tensor):
        collections.append(store.collect(retain_anchors=2))
        yield from original_iter(manifest, tensor)

    monkeypatch.setattr(store, "iter_tensor", collect_before_read)
    destination = module.write_checkpoint(store, "s", 0, tmp_path / "checkpoint")

    assert collections[0]["removed_versions"] == 0
    manifest = store.manifest("s", 0)
    module.validate_checkpoint(destination, manifest)

    # Once the checkpoint is durable, its temporary retention can be released.
    assert store.collect(retain_anchors=2)["removed_versions"] == 1
    with pytest.raises(FileNotFoundError):
        store.manifest("s", 0)
