"""Per-file durable IO may overlap encoding, but never commit or cleanup."""

import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from relax.distributed.weight_sync import store as module
from relax.distributed.weight_sync.store import VersionStore


def setup_store(tmp_path, count=24):
    store = VersionStore(tmp_path / "store", disk_budget=1 << 24, max_weight_bytes=1 << 20, canonical_blocks=True)
    inputs = {}
    for i in range(count):
        path = tmp_path / f"input-{i}"
        path.write_bytes(bytes([i]) * 1024)
        inputs[str(i)] = ("U8", [1024], path)
    return store, inputs


def test_file_sync_overlaps_four_encoders_and_waits_before_commit(tmp_path, monkeypatch):
    store, table = setup_store(tmp_path)
    original = os.fsync
    release = threading.Event()
    eight = threading.Event()
    mutex = threading.Lock()
    active = peak = arrived = 0

    def delayed(fd):
        nonlocal active, peak, arrived
        name = os.readlink(f"/proc/self/fd/{fd}")
        if "/objects/.partial-" not in name:
            return original(fd)
        with mutex:
            active += 1
            arrived += 1
            peak = max(peak, active)
            if arrived >= 8:
                eight.set()
        try:
            assert release.wait(5)
            original(fd)
        finally:
            with mutex:
                active -= 1

    monkeypatch.setattr(os, "fsync", delayed)
    with ThreadPoolExecutor(max_workers=1) as caller:
        result = caller.submit(store.publish, "s", 0, table, encode_workers=4)
        try:
            assert eight.wait(5), "fsync still occupies all four encoders"
            assert not store.manifest_path("s", 0).exists()
            assert peak <= 16
        finally:
            release.set()
        manifest = result.result(timeout=10)
    assert active == 0
    assert store.has_verified_version("s", 0, manifest["root"])
    assert not list(store.root.rglob(".partial-*"))


def test_sync_failure_drains_before_cleanup_and_preserves_previous(tmp_path, monkeypatch):
    store, table = setup_store(tmp_path)
    previous = store.publish("s", 0, table)
    for i, (_, _, path) in enumerate(table.values()):
        path.write_bytes(bytes([i + 32]) * 1024)
    original_sync = os.fsync
    original_sweep = store._sweep_unpublished
    active = 0
    mutex = threading.Lock()
    injected = False

    def failing(fd):
        nonlocal active, injected
        name = os.readlink(f"/proc/self/fd/{fd}")
        if "/objects/.partial-" not in name:
            return original_sync(fd)
        with mutex:
            active += 1
            fail = not injected
            injected = True
        try:
            if fail:
                raise OSError("injected file durability failure")
            time.sleep(0.02)
            return original_sync(fd)
        finally:
            with mutex:
                active -= 1

    def sweep():
        assert active == 0
        return original_sweep()

    monkeypatch.setattr(store, "_sweep_unpublished", sweep)
    with monkeypatch.context() as failure:
        failure.setattr(os, "fsync", failing)
        with pytest.raises(OSError, match="file durability failure"):
            store.publish("s", 1, table)
    assert active == 0 and injected
    assert not store.manifest_path("s", 1).exists()
    assert not list(store.root.rglob(".partial-*"))
    assert store.has_verified_version("s", 0, previous["root"])
    assert store.publish("s", 1, table)["version"] == 1


def test_independent_commit_disk_readback_still_rejects_corruption(tmp_path, monkeypatch):
    store, table = setup_store(tmp_path, 4)
    original = store.commit

    def corrupt(manifest):
        key = manifest["tensors"][0]["chunks"][0]["object"]
        path = store.object_path(key)
        with path.open("r+b") as file:
            file.write(b"!")
        return original(manifest)

    monkeypatch.setattr(store, "commit", corrupt)
    with pytest.raises(ValueError, match="corrupt durable object"):
        store.publish("s", 0, table)
    assert not store.manifest_path("s", 0).exists()


def test_pending_duplicate_bytes_are_rechecked_without_duplicate_writes(tmp_path, monkeypatch):
    store, table = setup_store(tmp_path, 24)
    for _, _, path in table.values():
        path.write_bytes(b"a" * 1024)
    real = os.fsync
    files = []

    def observed(fd):
        name = os.readlink(f"/proc/self/fd/{fd}")
        if "/objects/.partial-" in name or "/blocks/.partial-" in name:
            files.append(name)
            time.sleep(0.01)
        return real(fd)

    monkeypatch.setattr(os, "fsync", observed)
    manifest = store.publish("s", 0, table)
    assert len(files) == 2  # encoded + independent canonical durability domains
    assert store.has_verified_version("s", 0, manifest["root"])


def test_public_atomic_write_remains_synchronous_outside_publication(tmp_path, monkeypatch):
    target = tmp_path / "object"
    synced = []
    original = os.fsync

    def observe(fd):
        synced.append(fd)
        return original(fd)

    monkeypatch.setattr(os, "fsync", observe)
    module.atomic_write(target, b"original")
    assert target.read_bytes() == b"original" and len(synced) == 2
    with monkeypatch.context() as failure:
        failure.setattr(os, "fsync", lambda fd: (_ for _ in ()).throw(OSError("sync failed")))
        with pytest.raises(OSError, match="sync failed"):
            module.atomic_write(target, b"replacement")
    assert target.read_bytes() == b"original"
    assert not list(tmp_path.glob(".partial-*"))
