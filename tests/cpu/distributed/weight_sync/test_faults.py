# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import socket
import struct
import threading
from pathlib import Path
from time import monotonic

import pytest
import torch

from relax.distributed.weight_sync.consumer import Consumer, verify_checkpoint
from relax.distributed.weight_sync.format import WeightSyncError, canonical_json
from relax.distributed.weight_sync.publisher import Publisher
from relax.distributed.weight_sync.store import CapacityError, SharedFSStore
from relax.distributed.weight_sync.tcp import ObjectServer, TCPTransport, _frame, _read_exact, _receive_frame


def make_store(tmp_path):
    store = SharedFSStore(tmp_path / "store", canonical_root=tmp_path / "canonical")
    pub = Publisher(store, "s", {}, chunk_bytes=256, compression=False)
    tensor = torch.arange(1024, dtype=torch.int32)
    pub.finalize_pending(pub.submit("0", [("x", tensor)]))
    tensor[5] = -5
    target = pub.finalize_pending(pub.submit("1", [("x", tensor)]))
    pub.close()
    return store, target


def test_corrupt_delivery_recovers_same_target_from_independent_canonical(tmp_path):
    store, target = make_store(tmp_path)
    manifest = store.resolve("s", 1)
    changed = next(b for b in manifest["blocks"] if b["encoding"] != "COPY")
    (store.root / "objects" / changed["object"]).write_bytes(b"damaged delivery")
    consumer = Consumer(store, SharedFSStore(tmp_path / "cache"), tmp_path / "ckpt")
    prepared = consumer.prepare("s", 1)
    assert prepared.root == target.root
    assert prepared.artifact_id != target.artifact_id
    assert store.manifest(prepared.artifact_id)["kind"] == "FULL"
    verify_checkpoint(prepared.path, manifest["catalog"])


def test_local_checkpoint_corruption_is_rebuilt_from_validated_objects(tmp_path):
    store, _ = make_store(tmp_path)
    consumer = Consumer(store, SharedFSStore(tmp_path / "cache"), tmp_path / "ckpt")
    prepared = consumer.prepare("s", 1)
    shard = next(Path(prepared.path).glob("*.safetensors"))
    with shard.open("r+b") as output:
        output.seek(-1, 2)
        output.write(b"x")
    repaired = consumer.prepare("s", 1)
    verify_checkpoint(repaired.path, store.resolve("s", 1)["catalog"])


def test_cache_gc_cannot_delete_uncommitted_candidate_objects(tmp_path):
    store, _ = make_store(tmp_path)
    cache = SharedFSStore(tmp_path / "cache")
    downloaded, allow_download = threading.Event(), threading.Event()
    gc_started, gc_done = threading.Event(), threading.Event()

    class SlowTransport:
        def __getattr__(self, name):
            return getattr(store, name)

        def object(self, object_id):
            if downloaded.is_set():
                assert allow_download.wait(3)
            data = store.object(object_id)
            downloaded.set()
            return data

    consumer = Consumer(SlowTransport(), cache, tmp_path / "ckpt", timeout_s=10)
    results, errors = [], []

    def prepare():
        try:
            results.append(consumer.prepare("s", 1))
        except Exception as error:
            errors.append(error)

    def collect():
        gc_started.set()
        cache.collect()
        gc_done.set()

    worker = threading.Thread(target=prepare)
    worker.start()
    assert downloaded.wait(3)
    collector = threading.Thread(target=collect)
    collector.start()
    assert gc_started.wait(3)
    assert not gc_done.wait(0.05)
    allow_download.set()
    worker.join(5)
    collector.join(5)
    assert not errors and len(results) == 1 and gc_done.is_set()
    verify_checkpoint(results[0].path, store.resolve("s", 1)["catalog"])


def test_checkpoint_quota_includes_headers_indexes_and_configuration(tmp_path):
    store = SharedFSStore(tmp_path / "store")
    pub = Publisher(store, "s", {}, chunk_bytes=256)
    weights = [("long-name-" + "x" * 100 + str(index), torch.tensor(index, dtype=torch.uint8)) for index in range(100)]
    pub.finalize_pending(pub.submit("0", weights))
    consumer = Consumer(store, SharedFSStore(tmp_path / "cache"), tmp_path / "ckpt", disk_budget_bytes=30000)
    with pytest.raises(CapacityError, match="CAPACITY"):
        consumer.prepare("s", 0)
    assert not list((tmp_path / "ckpt").glob("*.safetensors"))
    pub.close()


def test_retry_with_same_update_id_cannot_silently_change_target(tmp_path):
    store, _ = make_store(tmp_path)
    pub = Publisher(store, "s", {}, chunk_bytes=256, compression=False)
    with pytest.raises(WeightSyncError, match="different target"):
        pub.finalize_pending(pub.submit("1", [("x", torch.zeros(1024, dtype=torch.int32))]))
    assert store.head("s")["version"] == 1
    pub.close()


@pytest.mark.parametrize("version", [0, 1])
@pytest.mark.parametrize("damage", ["payload", "manifest"])
def test_existing_full_delivery_is_repaired_without_changing_its_identity(tmp_path, version, damage):
    store, _ = make_store(tmp_path)
    full_id = store.ensure_full("s", version)
    manifest = store.manifest(full_id)
    if damage == "payload":
        (store.root / "objects" / manifest["blocks"][0]["object"]).write_bytes(b"damaged")
    else:
        (store.root / "artifacts" / full_id).write_bytes(b"damaged")
    assert store.ensure_full("s", version) == full_id
    consumer = Consumer(store, SharedFSStore(tmp_path / "cache"), tmp_path / "ckpt")
    prepared = consumer.prepare("s", version, full=True)
    assert prepared.artifact_id == full_id and prepared.root == manifest["root"]


def test_invalid_source_base_reseeds_from_current_snapshot(tmp_path):
    store, _ = make_store(tmp_path)
    before = store.resolve("s", 1)
    block = before["catalog"]["tensors"][0]["blocks"][0]
    (store.canonical_root / "blocks" / block["digest"]).unlink()
    pub = Publisher(store, "s", {}, chunk_bytes=256, compression=False)
    result = pub.finalize_pending(pub.submit("2", [("x", torch.ones(1024, dtype=torch.int32))]))
    assert result.version == 2 and store.manifest(result.artifact_id)["kind"] == "FULL"
    pub.close()


def test_gc_remains_usable_after_recovery_from_corrupt_delta_manifest(tmp_path):
    store, target = make_store(tmp_path)
    (store.root / "artifacts" / target.artifact_id).write_bytes(b"corrupt-delta-manifest")
    consumer = Consumer(store, SharedFSStore(tmp_path / "cache"), tmp_path / "ckpt")
    prepared = consumer.prepare("s", 1)
    assert prepared.root == target.root
    assert store.collect()["retained_versions"] == 2
    assert store.manifest(target.artifact_id)["kind"] == "DELTA"


def test_reference_release_remains_available_when_index_copy_cannot_fit(tmp_path):
    store, _ = make_store(tmp_path)
    store.pin("release-me", "s", 0)
    # A quota reduction models the same capacity state as growing into the
    # final free space; releasing protection must remain a usable recovery action.
    store.disk_budget_bytes = store.disk_usage() + 1
    store.release("release-me")
    assert "release-me" not in store.inventory()["references"]


def test_tcp_retries_truncated_object_and_reuses_completed_connection(tmp_path):
    data = b"verified-object-data"
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen(2)
    attempts = []

    def serve():
        try:
            for attempt in range(2):
                with listener.accept()[0] as connection:
                    _receive_frame(connection, monotonic() + 2)
                    attempts.append(attempt)
                    _frame(connection, {"status": "OK", "length": len(data)}, monotonic() + 2)
                    connection.sendall(data[:3] if attempt == 0 else data)
        finally:
            listener.close()

    thread = threading.Thread(target=serve)
    thread.start()
    transport = TCPTransport(*listener.getsockname(), timeout_s=2)
    assert transport.object("a" * 64) == data
    thread.join(3)
    assert attempts == [0, 1] and transport.metrics["retries"] == 1
    assert transport.metrics["payload_bytes"] == 3 + len(data)


def test_tcp_counts_only_accepted_request_bytes_before_interrupted_send(monkeypatch):
    data = b"verified-object-data"
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen(2)
    listener.settimeout(3)
    accepted = []
    server_errors = []

    def serve():
        try:
            with listener.accept()[0] as connection:
                connection.settimeout(2)
                accepted.append(b"".join(iter(lambda: connection.recv(4096), b"")))
            with listener.accept()[0] as connection:
                connection.settimeout(2)
                _receive_frame(connection, monotonic() + 2)
                _frame(connection, {"status": "OK", "length": len(data)}, monotonic() + 2)
                connection.sendall(data)
        except Exception as error:
            server_errors.append(error)
        finally:
            listener.close()

    connect = socket.create_connection
    calls = 0

    class InterruptedSocket:
        def __init__(self, connection):
            self.connection, self.sent = connection, False

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.connection.close()

        def settimeout(self, timeout):
            self.connection.settimeout(timeout)

        def send(self, payload):
            if self.sent:
                raise ConnectionError("injected interruption after three actual socket bytes")
            self.sent = True
            self.connection.sendall(payload[:3])
            return 3

    def interrupted_connect(*args, **kwargs):
        nonlocal calls
        calls += 1
        connection = connect(*args, **kwargs)
        return InterruptedSocket(connection) if calls == 1 else connection

    thread = threading.Thread(target=serve)
    thread.start()
    try:
        monkeypatch.setattr(socket, "create_connection", interrupted_connect)
        transport = TCPTransport(*listener.getsockname(), timeout_s=2)
        assert transport.object("a" * 64) == data
    finally:
        listener.close()
        thread.join(5)
    assert not thread.is_alive() and not server_errors
    assert accepted == [struct.pack("!I", len(canonical_json({"op": "object", "id": "a" * 64})))[:3]]
    complete_request = 4 + len(canonical_json({"op": "object", "id": "a" * 64}))
    complete_response = 4 + len(canonical_json({"status": "OK", "length": len(data)}))
    assert transport.metrics["frame_bytes"] == 3 + complete_request + complete_response
    assert transport.metrics["payload_bytes"] == len(data)
    assert transport.metrics["retries"] == 1


@pytest.mark.parametrize("responds", [True, False])
def test_tcp_read_poll_preserves_total_deadline(responds):
    receiver, sender = socket.socketpair()
    first_timeout, release = threading.Event(), threading.Event()

    class PollingSocket:
        def settimeout(self, timeout):
            receiver.settimeout(min(timeout, 0.01))

        def recv(self, size):
            try:
                return receiver.recv(size)
            except socket.timeout:
                first_timeout.set()
                raise

    def serve():
        try:
            if first_timeout.wait(1) and responds:
                release.wait(0.03)
                sender.sendall(b"target")
            else:
                release.wait(1)
        finally:
            sender.close()

    thread = threading.Thread(target=serve)
    thread.start()
    try:
        if responds:
            assert _read_exact(PollingSocket(), 6, monotonic() + 1) == b"target"
        else:
            with pytest.raises(TimeoutError, match="deadline exceeded"):
                _read_exact(PollingSocket(), 6, monotonic() + 0.06)
        assert first_timeout.is_set()
    finally:
        release.set()
        thread.join(2)
        receiver.close()
    assert not thread.is_alive()


def test_tcp_slow_header_releases_worker_and_other_consumer_progresses(tmp_path):
    store, _ = make_store(tmp_path)
    server = ObjectServer(("127.0.0.1", 0), store, workers=1, request_timeout_s=0.15)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with socket.create_connection(server.server_address) as slow:
            slow.sendall(struct.pack("!I", 100) + b"{")
            transport = TCPTransport(*server.server_address, timeout_s=2, attempts=8)
            assert transport.record("s", 1)["version"] == 1
            assert transport.metrics["retries"] >= 1
    finally:
        server.shutdown()
        server.server_close()
        thread.join(2)


def test_tcp_declared_length_is_rejected_before_payload_allocation(tmp_path):
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)

    def serve():
        try:
            with listener.accept()[0] as connection:
                _receive_frame(connection, monotonic() + 2)
                payload = canonical_json({"status": "OK", "length": 2**40})
                connection.sendall(struct.pack("!I", len(payload)) + payload)
        finally:
            listener.close()

    thread = threading.Thread(target=serve)
    thread.start()
    transport = TCPTransport(*listener.getsockname(), timeout_s=2)
    with pytest.raises(WeightSyncError, match="integer"):
        transport.object("a" * 64)
    thread.join(2)
