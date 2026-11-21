"""Completion-driven publication stays bounded and preserves commit
evidence."""

import contextvars
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from relax.distributed.weight_sync import store as module
from relax.distributed.weight_sync.deadline import remaining, time_budget
from relax.distributed.weight_sync.store import VersionStore, digest


def test_slow_first_block_does_not_stall_independent_durable_blocks(tmp_path, monkeypatch):
    store = VersionStore(tmp_path / "store", disk_budget=1 << 22, max_weight_bytes=1 << 20, canonical_blocks=True)
    data = b"".join(bytes([i]) * 1024 for i in range(20))
    source = tmp_path / "input"
    source.write_bytes(data)
    first = digest(data[:1024])
    original_put = store.put_object
    completed, lock = set(), threading.Lock()
    progress = threading.Event()

    def hold_first(key, payload):
        if key == first:
            assert progress.wait(3), "A slow first block still stalls independent publication work"
        original_put(key, payload)
        with lock:
            completed.add(key)
            if len(completed) >= 8:
                progress.set()

    monkeypatch.setattr(store, "put_object", hold_first)
    manifest = store.publish("s", 0, {"w": ("U8", [len(data)], source)}, chunk_bytes=1024, encode_workers=4)
    assert [c["offset"] for c in manifest["tensors"][0]["chunks"]] == list(range(0, len(data), 1024))
    assert b"".join(store.iter_tensor(manifest, manifest["tensors"][0])) == data


def test_completion_pipeline_is_bounded_and_propagates_context():
    tag = contextvars.ContextVar("publication-tag", default=None)
    token = tag.set("frozen-update")
    active = peak = 0
    lock = threading.Lock()
    started = threading.Barrier(4)
    release = threading.Event()

    def work(value):
        nonlocal active, peak
        assert tag.get() == "frozen-update"
        assert 0 < remaining(30, "pipeline worker") <= 5
        with lock:
            active += 1
            peak = max(peak, active)
        try:
            if value < 4:
                started.wait(3)
            assert release.wait(3)
            return value
        finally:
            with lock:
                active -= 1

    try:
        with time_budget(5):
            context = contextvars.copy_context()
            with ThreadPoolExecutor(max_workers=1) as caller:
                future = caller.submit(context.run, lambda: list(module._publish_map(work, range(20), 4)))
                release.set()
                assert sorted(future.result(timeout=5)) == list(range(20))
        assert active == 0 and peak == 4
    finally:
        tag.reset(token)


@pytest.mark.parametrize("workers", [1, 4])
@pytest.mark.parametrize("compression", [False, True])
def test_pipeline_preserves_previous_ordered_manifest_bytes(tmp_path, monkeypatch, workers, compression):
    reference = VersionStore(tmp_path / "before", disk_budget=1 << 24, max_weight_bytes=1 << 20, canonical_blocks=True)
    candidate = VersionStore(tmp_path / "after", disk_budget=1 << 24, max_weight_bytes=1 << 20, canonical_blocks=True)
    for version in range(4):
        source = tmp_path / "input"
        data = bytearray(bytes(range(256)) * 64)
        if version in (1, 2):
            data[3] ^= 128
        elif version == 3:
            data[:] = b"z" * len(data)
        source.write_bytes(data)
        table = {"w": ("BF16", [len(data) // 2], source)}
        with monkeypatch.context() as ordered:
            ordered.setattr(module, "_publish_map", module.ordered_map)
            reference.publish(
                "s", version, table, chunk_bytes=1024, encode_workers=workers, compression=compression, model_config={}
            )
        candidate.publish(
            "s", version, table, chunk_bytes=1024, encode_workers=workers, compression=compression, model_config={}
        )
        assert candidate.manifest_path("s", version).read_bytes() == reference.manifest_path("s", version).read_bytes()
        manifest = candidate.manifest("s", version)
        assert b"".join(candidate.iter_tensor(manifest, manifest["tensors"][0])) == data


def test_pipeline_failure_drains_before_cleanup_and_preserves_prior_version(tmp_path, monkeypatch):
    import time

    store = VersionStore(tmp_path / "store", disk_budget=1 << 24, max_weight_bytes=1 << 20, canonical_blocks=True)
    source = tmp_path / "input"
    source.write_bytes(bytes(8192))
    table = {"w": ("U8", [8192], source)}
    first = store.publish("s", 0, table, chunk_bytes=1024)
    source.write_bytes(b"".join(bytes([i + 1]) * 1024 for i in range(8)))
    put, sweep = store.put_object, store._sweep_unpublished
    active = 0
    lock = threading.Lock()

    def damaged_write(key, payload):
        nonlocal active
        with lock:
            active += 1
        try:
            if payload[0] == 2:
                raise OSError("injected out-of-order block failure")
            time.sleep(0.03)
            put(key, payload)
        finally:
            with lock:
                active -= 1

    def cleanup():
        assert active == 0
        return sweep()

    monkeypatch.setattr(store, "_sweep_unpublished", cleanup)
    with monkeypatch.context() as failing:
        failing.setattr(store, "put_object", damaged_write)
        with pytest.raises(OSError, match="out-of-order block failure"):
            store.publish("s", 1, table, chunk_bytes=1024, encode_workers=4)
    assert not store.manifest_path("s", 1).exists()
    assert store.has_verified_version("s", 0, first["root"])
    live = {c["object"] for t in first["tensors"] for c in t["chunks"]}
    assert {p.name for p in (store.root / "objects").iterdir()} == live
    assert store.publish("s", 1, table, chunk_bytes=1024, encode_workers=4)["version"] == 1


def test_closing_pipeline_drains_running_calls():
    import time

    active = 0
    lock = threading.Lock()

    def work(value):
        nonlocal active
        with lock:
            active += 1
        try:
            time.sleep(0.005 if value == 0 else 0.04)
            return value
        finally:
            with lock:
                active -= 1

    results = module._publish_map(work, range(12), 4)
    next(results)
    results.close()
    assert active == 0
