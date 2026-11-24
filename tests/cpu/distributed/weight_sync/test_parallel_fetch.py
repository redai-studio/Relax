"""Bounded concurrent fetch preserves retries, retention and exact
accounting."""

import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from relax.distributed.weight_sync.deadline import remaining, time_budget
from relax.distributed.weight_sync.metrics import SyncMetrics
from relax.distributed.weight_sync.store import VersionStore, canonical
from relax.distributed.weight_sync.transport import ContentError, ObjectServer, SharedFSTransport, TCPTransport, fetch


def stores(tmp_path, count=8):
    source = VersionStore(tmp_path / "source", disk_budget=1 << 24, max_weight_bytes=1 << 20, canonical_blocks=True)
    target = VersionStore(tmp_path / "target", disk_budget=1 << 24, max_weight_bytes=1 << 20, canonical_blocks=True)
    table = {}
    for n in range(count):
        path = tmp_path / str(n)
        path.write_bytes(bytes([n + 1]) * 1024)
        table[str(n)] = ("U8", [1024], path)
    return source, target, table


def test_fetch_overlaps_four_reads_and_fetches_duplicate_objects_once(tmp_path):
    source, target, table = stores(tmp_path)
    table["duplicate"] = table["0"]
    manifest = source.publish("s", 0, table)
    transport = SharedFSTransport(source)
    original = transport.object
    barrier = threading.Barrier(4, timeout=3)
    calls, concurrent = [], []
    active = 0
    lock = threading.Lock()

    def observed(key):
        nonlocal active
        with lock:
            active += 1
            calls.append(key)
            concurrent.append(active)
        try:
            barrier.wait()
            return original(key)
        finally:
            with lock:
                active -= 1

    transport.object = observed
    assert fetch(transport, target, "s", 0) == manifest
    assert max(concurrent) == 4 and len(calls) == len(set(calls)) == 8
    assert transport.bytes_received == len(canonical(manifest)) + 8 * 1024
    assert target.has_verified_version("s", 0, manifest["root"])


def test_corrupt_read_drains_workers_before_release_and_retry_reuses_verified_objects(tmp_path):
    source, target, table = stores(tmp_path, 4)
    manifest = source.publish("s", 0, table)
    transport = SharedFSTransport(source)
    original, unpin = transport.object, transport.unpin
    barrier, failed, release = threading.Barrier(4, timeout=3), threading.Event(), threading.Event()
    corrupt = manifest["tensors"][0]["chunks"][0]["object"]
    active = 0
    lock = threading.Lock()

    def object_read(key):
        nonlocal active
        with lock:
            active += 1
        try:
            barrier.wait()
            if key == corrupt:
                failed.set()
                return b"corrupt"
            assert release.wait(3)
            return original(key)
        finally:
            with lock:
                active -= 1

    def release_pin(owner):
        assert active == 0
        return unpin(owner)

    transport.object, transport.unpin = object_read, release_pin
    with ThreadPoolExecutor(max_workers=1) as pool:
        task = pool.submit(fetch, transport, target, "s", 0)
        try:
            assert failed.wait(3)
            assert not task.done()
            assert not target.manifest_path("s", 0).exists()
        finally:
            release.set()
        with pytest.raises(ContentError):
            task.result(timeout=3)
    retained = list((target.root / "objects").iterdir())
    assert len(retained) == 3
    calls = []

    def repaired(key):
        calls.append(key)
        return original(key)

    transport.object = repaired
    assert fetch(transport, target, "s", 0) == manifest
    assert calls == [corrupt]
    assert target.has_verified_version("s", 0, manifest["root"])


def test_tcp_cold_chain_counts_both_directions_and_binds_each_object_to_its_version(tmp_path):
    source, target, table = stores(tmp_path, 4)
    manifests = []
    for version in range(3):
        for n, (_, _, path) in enumerate(table.values()):
            path.write_bytes(bytes([n * 4 + version]) * 1024)
        manifests.append(source.publish("s", version, table, anchor_interval=10, update_id=f"update-{version}"))
    metrics = SyncMetrics("server")
    with ObjectServer(("127.0.0.1", 0), source, metrics=metrics) as server:
        worker = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01})
        worker.start()
        transport = TCPTransport(server.server_address)
        try:
            assert fetch(transport, target, "s", 2) == manifests[2]
        finally:
            server.shutdown()
            worker.join()
    client, peer = transport.ledger.report()["bytes"], server.ledger.report()["bytes"]
    assert client["sent"] == peer["received"] and client["received"] == peer["sent"]
    assert sum(client["received"].values()) == transport.bytes_received
    assert sum(client["sent"].values()) == transport.bytes_sent
    for version, manifest in enumerate(manifests):
        keys = {c["object"] for t in manifest["tensors"] for c in t["chunks"]}
        requests = [
            r
            for r in metrics.report()["related_reports"]
            if r["request"]["op"] == "object" and r["request"]["key"] in keys
        ]
        assert requests
        assert all(r["request"]["target_context"] == {"stream": "s", "version": version} for r in requests)
        assert all(r["update_id"] == f"update-{version}" for r in requests)
    assert target.has_verified_version("s", 2, manifests[2]["root"])


def test_fetch_workers_inherit_operation_deadline(tmp_path):
    source, target, table = stores(tmp_path, 4)
    source.publish("s", 0, table)
    transport = SharedFSTransport(source)
    original = transport.object
    seen = []

    def observed(key):
        seen.append(remaining(100))
        return original(key)

    transport.object = observed
    with time_budget(2):
        fetch(transport, target, "s", 0)
    assert len(seen) == 4 and all(0 < seconds <= 2 for seconds in seen)


def test_tcp_identity_lookup_does_not_reparse_manifest_per_object(tmp_path, monkeypatch):
    source, target, table = stores(tmp_path, 32)
    source.publish("s", 0, table, update_id="frozen-target")
    calls = []
    original = source.manifest

    def counted(*args, **kwargs):
        calls.append(args)
        return original(*args, **kwargs)

    monkeypatch.setattr(source, "manifest", counted)
    with ObjectServer(("127.0.0.1", 0), source) as server:
        worker = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01})
        worker.start()
        try:
            fetch(TCPTransport(server.server_address), target, "s", 0)
        finally:
            server.shutdown()
            worker.join()
    # Pin and manifest operations still validate their payloads. The 32 object
    # receipts must reuse a validated update identity instead of 32 full parses.
    assert len(calls) < 10


def test_tcp_identity_cache_revalidates_replacement_and_does_not_hide_deletion(tmp_path, monkeypatch):
    import os

    source, _, table = stores(tmp_path, 1)
    source.publish("s", 0, table, update_id="original-target")
    path = source.manifest_path("s", 0)
    raw = path.read_bytes()
    original = source.manifest
    calls = []

    def counted(*args, **kwargs):
        calls.append(args)
        return original(*args, **kwargs)

    monkeypatch.setattr(source, "manifest", counted)
    with ObjectServer(("127.0.0.1", 0), source) as server:
        assert server.target_update_id("s", 0) == "original-target"
        assert server.target_update_id("s", 0) == "original-target"
        assert len(calls) == 1
        replacement = path.with_suffix(".replacement")
        replacement.write_bytes(raw)
        os.replace(replacement, path)
        assert server.target_update_id("s", 0) == "original-target"
        assert len(calls) == 2
        path.write_bytes(b"invalid manifest")
        with pytest.raises(ValueError):
            server.target_update_id("s", 0)
        path.unlink()
        with pytest.raises(FileNotFoundError):
            server.target_update_id("s", 0)
