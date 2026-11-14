"""Parallel preparation preserves serial artifacts, bounded work and failure
atomicity."""

import contextvars
import threading
from pathlib import Path

import pytest

from relax.distributed.weight_sync.parallel_io import ordered_map
from relax.distributed.weight_sync.store import VersionStore


def test_batch_fsyncs_each_file_and_each_content_directory_before_manifest(tmp_path, monkeypatch):
    import os
    import stat

    from relax.distributed.weight_sync import store as module

    store = VersionStore(tmp_path / "store", disk_budget=1 << 20, max_weight_bytes=1 << 20, canonical_blocks=True)
    table = {}
    for index in range(8):
        path = tmp_path / f"input-{index}"
        path.write_bytes(bytes([index]) * 1024)
        table[str(index)] = ("U8", [1024], path)
    directories, file_syncs = [], []
    real_sync, real_fsync, real_commit = module.sync_directory, os.fsync, store.commit

    def sync(directory):
        directories.append(directory)
        real_sync(directory)

    def fsync(fd):
        if stat.S_ISREG(os.fstat(fd).st_mode):
            file_syncs.append(fd)
        real_fsync(fd)

    def commit(manifest):
        assert directories.count(store.root / "objects") == 1
        assert directories.count(store.canonical_directory / "blocks") == 1
        assert len(file_syncs) == 16
        real_commit(manifest)

    monkeypatch.setattr(module, "sync_directory", sync)
    monkeypatch.setattr(os, "fsync", fsync)
    monkeypatch.setattr(store, "commit", commit)
    manifest = store.publish("s", 0, table, encode_workers=4)
    assert store.has_verified_version("s", 0, manifest["root"])


def test_directory_sync_failure_reclaims_pending_objects_and_retry_syncs_them(tmp_path, monkeypatch):
    from relax.distributed.weight_sync import store as module

    store = VersionStore(tmp_path / "store", disk_budget=1 << 20, max_weight_bytes=1 << 20, canonical_blocks=True)
    path = tmp_path / "input"
    path.write_bytes(bytes(range(256)) * 4)
    table = {"w": ("U8", [1024], path)}
    real_sync = module.sync_directory

    def failed_sync(directory):
        if directory == store.root / "objects":
            raise OSError("injected immutable-directory sync failure")
        real_sync(directory)

    with monkeypatch.context() as failure:
        failure.setattr(module, "sync_directory", failed_sync)
        with pytest.raises(OSError, match="immutable-directory sync failure"):
            store.publish("s", 0, table, encode_workers=4)
    assert not store.manifest_path("s", 0).exists()
    retained = sorted((store.root / "objects").iterdir())
    assert not retained
    directories = []

    def sync(directory):
        directories.append(directory)
        real_sync(directory)

    monkeypatch.setattr(module, "sync_directory", sync)
    manifest = store.publish("s", 0, table, encode_workers=4)
    assert directories.count(store.root / "objects") == 1
    assert store.has_verified_version("s", 0, manifest["root"])


@pytest.mark.parametrize("compression", [False, True])
def test_parallel_encoder_matches_serial_for_full_sparse_copy_and_dense(tmp_path, compression):
    stores = {
        workers: VersionStore(
            tmp_path / str(workers), disk_budget=1 << 24, max_weight_bytes=1 << 20, canonical_blocks=True
        )
        for workers in (1, 4)
    }
    inputs = {f"w{i}": tmp_path / f"input{i}" for i in range(8)}
    for version in range(4):
        table = {}
        for i, (name, path) in enumerate(inputs.items()):
            value = bytearray(bytes(range(256)) * 32)
            if version in (1, 2):
                value[i] ^= 1
            elif version == 3:
                value = bytearray(bytes([i]) * len(value))
            path.write_bytes(value)
            table[name] = ("BF16", [4096], path)
        manifests = [
            store.publish(
                "s", version, table, compression=compression, encode_workers=workers, chunk_bytes=1024, model_config={}
            )
            for workers, store in stores.items()
        ]
        assert manifests[0] == manifests[1]
        for tensor in manifests[0]["tensors"]:
            for chunk in tensor["chunks"]:
                assert (
                    stores[1].object_path(chunk["object"]).read_bytes()
                    == stores[4].object_path(chunk["object"]).read_bytes()
                )
        for store in stores.values():
            assert store.has_verified_version("s", version, manifests[0]["root"])


def test_failure_in_read_worker_leaves_no_visible_version(tmp_path, monkeypatch):
    path = tmp_path / "input"
    path.write_bytes(b"\0" * 8192)
    store = VersionStore(tmp_path / "store", disk_budget=1 << 20, max_weight_bytes=1 << 20)
    original = Path.open

    def fail_input(self, *args, **kwargs):
        if self == path and threading.current_thread().name.startswith("weight-block-io"):
            raise OSError("injected block read failure")
        return original(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", fail_input)
    with pytest.raises(OSError, match="block read failure"):
        store.publish("s", 0, {"w": ("BF16", [4096], path)}, chunk_bytes=1024, encode_workers=4)
    assert not store.manifest_path("s", 0).exists()


def test_cpu_work_is_bounded_ordered_and_inherits_measurement_context():
    active_context = contextvars.ContextVar("test-update", default=None)
    token = active_context.set("frozen-target")
    consumed = []

    def values():
        for number in range(20):
            consumed.append(number)
            yield number

    def prepare(number):
        assert active_context.get() == "frozen-target"
        return number

    try:
        results = ordered_map(prepare, values(), 4)
        assert next(results) == 0
        assert len(consumed) == 4
        assert list(results) == list(range(1, 20))
    finally:
        active_context.reset(token)


def test_concurrent_duplicate_objects_reserve_and_write_once(tmp_path):
    from relax.distributed.weight_sync.metrics import SyncMetrics
    from relax.distributed.weight_sync.store import digest
    from relax.distributed.weight_sync.traffic import capture

    value = bytes(range(256)) * 16
    store = VersionStore(tmp_path / "store", disk_budget=len(value), max_weight_bytes=len(value))
    metrics = SyncMetrics("same-object")
    with capture(metrics):
        assert list(ordered_map(lambda _: store.put_object(digest(value), value), range(32), 8)) == [None] * 32
    assert metrics.report()["bytes"]["W"] == len(value)
    assert store.object_path(digest(value)).read_bytes() == value
    assert store._remaining_bytes == 0


def test_parallel_full_repair_matches_serial_and_repairs_corrupt_objects(tmp_path, monkeypatch):
    from relax.distributed.weight_sync import store as module

    stores = [
        VersionStore(tmp_path / name, disk_budget=1 << 24, max_weight_bytes=1 << 20, canonical_blocks=True)
        for name in ("serial", "parallel")
    ]
    table = {}
    for version in range(2):
        for index in range(8):
            path = tmp_path / f"input-{index}"
            data = bytearray(bytes(range(256)) * 32)
            data[index] = 255 - version
            path.write_bytes(data)
            table[str(index)] = ("BF16", [4096], path)
        for store in stores:
            store.publish("s", version, table, chunk_bytes=1024)
    with monkeypatch.context() as serial:
        serial.setattr(module, "ordered_map", lambda function, values, workers: map(function, values))
        reference = stores[0].ensure_full("s", 1)
    actual = stores[1].ensure_full("s", 1)
    assert actual == reference
    objects = sorted({c["object"] for t in actual["tensors"] for c in t["chunks"]})
    stores[1].object_path(objects[0]).write_bytes(b"corrupt")
    stores[1].object_path(objects[1]).rename(tmp_path / "withheld-object")
    assert stores[1].ensure_full("s", 1) == reference
    for key in objects:
        assert stores[0].object_path(key).read_bytes() == stores[1].object_path(key).read_bytes()


def test_failed_parallel_full_repair_keeps_delta_visible_and_retries(tmp_path, monkeypatch):
    store = VersionStore(tmp_path / "store", disk_budget=1 << 24, max_weight_bytes=1 << 20, canonical_blocks=True)
    path = tmp_path / "input"
    path.write_bytes(bytes(range(256)) * 32)
    table = {"w": ("BF16", [4096], path)}
    store.publish("s", 0, table, chunk_bytes=1024)
    value = bytearray(path.read_bytes())
    value[0] ^= 1
    path.write_bytes(value)
    delta = store.publish("s", 1, table, chunk_bytes=1024)
    real_put = store.put_object

    def fail_put(key, payload):
        if payload == bytes(value[:1024]):
            raise OSError("injected FULL durable write failure")
        return real_put(key, payload)

    with monkeypatch.context() as failure:
        failure.setattr(store, "put_object", fail_put)
        with pytest.raises(OSError, match="FULL durable write failure"):
            store.ensure_full("s", 1)
    assert store.manifest("s", 1, prefer_full=True) == delta
    assert store.ensure_full("s", 1)["root"] == delta["root"]


def test_failed_parallel_candidate_copy_cannot_publish_and_success_owns_its_bytes(tmp_path, monkeypatch):
    import shutil

    from relax.distributed.weight_sync.publication import VersionPublisher

    store = VersionStore(tmp_path / "store", disk_budget=1 << 24, max_weight_bytes=1 << 20)
    inputs = {}
    for index in range(8):
        path = tmp_path / f"input-{index}"
        path.write_bytes(bytes([index]) * 1024)
        inputs[str(index)] = ("BF16", [512], path)
    publisher = VersionPublisher(store, "s", tmp_path / "scratch", model_config={}, converter_revision="test")
    real_copy = shutil.copyfile

    def fail_copy(source, destination):
        if Path(source).name == "input-2":
            raise OSError("injected candidate copy failure")
        return real_copy(source, destination)

    try:
        with monkeypatch.context() as failure:
            failure.setattr(shutil, "copyfile", fail_copy)
            with pytest.raises(OSError, match="candidate copy failure"):
                publisher.submit("update", inputs)
        assert publisher.pending is None
        assert not (tmp_path / "scratch/publisher-s.json").exists()
        assert not store.manifest_path("s", 0).exists()
        handle = publisher.submit("update", inputs)
        for dtype, shape, path in inputs.values():
            path.write_bytes(b"x" * 1024)
        result = publisher.finalize_pending(handle)
        manifest = store.manifest("s", result.version)
        for tensor in manifest["tensors"]:
            assert b"".join(store.iter_tensor(manifest, tensor)) == bytes([int(tensor["name"])]) * 1024
    finally:
        publisher.close()


@pytest.mark.parametrize("changed", [False, True])
def test_full_repair_preserves_compression_even_when_delta_has_only_copy_chunks(tmp_path, changed):
    delta_store = VersionStore(tmp_path / "delta", disk_budget=1 << 24, max_weight_bytes=1 << 20)
    full_store = VersionStore(tmp_path / "full", disk_budget=1 << 24, max_weight_bytes=1 << 20)
    path = tmp_path / "input"
    table = {"w": ("BF16", [4096], path)}
    for version in range(2):
        value = bytearray(8192)
        if changed and version:
            value[100] = 1
        path.write_bytes(value)
        delta = delta_store.publish("s", version, table, compression=True, chunk_bytes=1024)
        full = full_store.publish("s", version, table, compression=True, full_only=True, chunk_bytes=1024)
    if not changed:
        assert {c["kind"] for c in delta["tensors"][0]["chunks"]} == {"copy"}
    repaired = delta_store.ensure_full("s", 1)
    assert repaired == full
    assert {c["kind"] for c in repaired["tensors"][0]["chunks"]} == {"raw_zlib"}


def pending_commit(tmp_path):
    source = VersionStore(tmp_path / "source", disk_budget=1 << 20, max_weight_bytes=1 << 20)
    target = VersionStore(tmp_path / "target", disk_budget=1 << 20, max_weight_bytes=1 << 20)
    path = tmp_path / "input"
    path.write_bytes(b"".join(bytes([index]) * 64 for index in range(24)))
    manifest = source.publish("s", 0, {"w": ("U8", [1536], path)}, chunk_bytes=64)
    for chunk in manifest["tensors"][0]["chunks"]:
        target.put_object(chunk["object"], source.object_path(chunk["object"]).read_bytes())
    return source, target, manifest


def test_commit_verifies_bounded_concurrent_reads_before_manifest_visibility(tmp_path, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor

    _, target, manifest = pending_commit(tmp_path)
    release, ready = threading.Event(), threading.Event()
    lock = threading.Lock()
    active, peak, completed = 0, 0, set()
    original = Path.read_bytes

    def observed(path):
        nonlocal active, peak
        if path.parent != target.root / "objects":
            return original(path)
        with lock:
            active += 1
            peak = max(peak, active)
            if active == 8:
                ready.set()
        try:
            assert release.wait(10)
            contents = original(path)
            with lock:
                completed.add(path.name)
            return contents
        finally:
            with lock:
                active -= 1

    monkeypatch.setattr(Path, "read_bytes", observed)
    with ThreadPoolExecutor(max_workers=1) as caller:
        future = caller.submit(target.commit, manifest)
        try:
            assert ready.wait(3), "Durable verification still serializes independent reads"
            assert not target.manifest_path("s", 0).exists()
        finally:
            release.set()
        future.result(timeout=10)
    assert peak == 8 and active == 0
    assert completed == {chunk["object"] for chunk in manifest["tensors"][0]["chunks"]}
    assert target.manifest("s", 0) == manifest


@pytest.mark.parametrize("failure", ["missing", "size", "hash"])
def test_commit_worker_failure_keeps_manifest_hidden_and_allows_exact_retry(tmp_path, failure):
    source, target, manifest = pending_commit(tmp_path)
    key = manifest["tensors"][0]["chunks"][7]["object"]
    path = target.object_path(key)
    if failure == "missing":
        path.unlink()
    elif failure == "size":
        path.write_bytes(b"short")
    else:
        path.write_bytes(b"x" * path.stat().st_size)
    with pytest.raises((FileNotFoundError, ValueError)):
        target.commit(manifest)
    assert not target.manifest_path("s", 0).exists()
    target.put_object(key, source.object_path(key).read_bytes())
    target.commit(manifest)
    assert target.manifest("s", 0) == manifest


def test_commit_worker_checks_inherit_the_operation_deadline(tmp_path, monkeypatch):
    from relax.distributed.weight_sync import store as module
    from relax.distributed.weight_sync.deadline import remaining, time_budget

    _, target, manifest = pending_commit(tmp_path)
    original = module.check_deadline
    worker_budgets = []

    def observed(stage):
        if threading.current_thread() is not threading.main_thread():
            worker_budgets.append(remaining(30, stage))
        original(stage)

    monkeypatch.setattr(module, "check_deadline", observed)
    with time_budget(5):
        target.commit(manifest)
    assert len(worker_budgets) == 24 and all(0 < budget < 5 for budget in worker_budgets)
