import threading

import pytest

from relax.distributed.weight_sync.consumer import VersionConsumer
from relax.distributed.weight_sync.ledger import compare_communication
from relax.distributed.weight_sync.metrics import SyncMetrics
from relax.distributed.weight_sync.store import VersionStore, digest
from relax.distributed.weight_sync.transport import ObjectServer, TCPTransport


def test_full_repair_writes_are_captured_in_server_worker(tmp_path):
    store = VersionStore(tmp_path / "store", disk_budget=1 << 20, max_weight_bytes=1024)
    raw = tmp_path / "input"
    raw.write_bytes(bytes(range(256)))
    store.publish("s", 0, {"w": ("BF16", [128], raw)}, model_config={})
    raw.write_bytes(b"xy" + bytes(range(256))[2:])
    store.publish("s", 1, {"w": ("BF16", [128], raw)}, model_config={})
    metrics = SyncMetrics("repair")
    with ObjectServer(("127.0.0.1", 0), store, metrics=metrics) as server:
        worker = threading.Thread(target=server.serve_forever)
        worker.start()
        try:
            TCPTransport(server.server_address).manifest("s", 1, full=True)
        finally:
            server.shutdown()
            worker.join()
    attempts = metrics.report()["related_reports"]
    assert len(attempts) == 1
    assert attempts[0]["request"] == {"op": "manifest", "stream": "s", "version": 1, "full": True}
    writes = attempts[0]["bytes_by_scope"]["file_write"]["W"]
    assert writes > 256
    assert any("objects" in row["link"] for row in attempts[0]["links"])
    assert server.ledger.report()["bytes"]["sent"]["X"] > 0


def test_socket_ledger_counts_headers_control_and_object_once(tmp_path):
    store = VersionStore(tmp_path, disk_budget=1 << 20, max_weight_bytes=1024)
    data = bytes(range(256))
    store.put_object(digest(data), data)
    with ObjectServer(("127.0.0.1", 0), store) as server:
        thread = threading.Thread(target=server.serve_forever)
        thread.start()
        try:
            transport = TCPTransport(server.server_address)
            assert transport.object(digest(data)) == data
            report = transport.ledger.report()
            assert report["bytes"]["received"]["X"] == len(data)
            assert report["bytes"]["received"]["C"] == 8
            assert report["bytes"]["sent"]["C"] == transport.bytes_sent
            assert sum(report["bytes"]["received"].values()) == transport.bytes_received
        finally:
            server.shutdown()
            thread.join()
    sent = server.ledger.report()["bytes"]["sent"]
    assert sent == report["bytes"]["received"]


def test_preparation_assigns_response_headers_to_control_and_counts_each_direction(tmp_path):
    store = VersionStore(tmp_path / "source", disk_budget=1 << 20, max_weight_bytes=1024)
    raw = tmp_path / "input"
    raw.write_bytes(bytes(range(256)))
    store.publish("s", 0, {"w": ("BF16", [128], raw)}, model_config={})
    with ObjectServer(("127.0.0.1", 0), store) as server:
        thread = threading.Thread(target=server.serve_forever)
        thread.start()
        try:
            transport = TCPTransport(server.server_address)
            cache = VersionStore(tmp_path / "cache", disk_budget=1 << 20, max_weight_bytes=1024)
            consumer = VersionConsumer(transport, cache, tmp_path / "checkpoints", checkpoint_budget=32 << 20)
            report = consumer.prepare("s", 0, "reader").metrics
            assert report["bytes"]["X"] > 256  # Object plus the actual manifest body.
            socket = report["socket_ledger_after"]["bytes"]
            assert report["bytes"]["C"] == socket["sent"]["C"] + socket["received"]["C"]
            assert report["bytes"]["C"] > transport.bytes_sent
            assert not report["counter_coverage_complete"]
        finally:
            server.shutdown()
            thread.join()


def test_comparison_uses_complete_bounds_and_the_same_sequence():
    def make(value):
        return {
            "scope": "socket",
            "sequence": "frozen-run",
            "stages": {
                key: {"lower": value, "upper": value, "evidence": "receipt.json"} for key in ("G", "W", "X", "T", "C")
            },
        }

    baseline, delta = make(100), make(40)
    assert compare_communication(baseline, delta)["meets_50_percent"]
    delta["stages"]["G"]["upper"] = None
    assert compare_communication(baseline, delta)["saving_lower_bound"] is None
    delta["scope"] = "wire"
    with pytest.raises(ValueError, match="same frozen"):
        compare_communication(baseline, delta)


def test_additional_target_and_failed_prepare_retain_socket_bytes(tmp_path, monkeypatch):
    import json

    from relax.distributed.weight_sync import store as store_module

    saved = threading.Event()
    atomic_write = store_module.atomic_write

    def record_write(path, data):
        atomic_write(path, data)
        if path.parent == tmp_path / "checkpoints" / "failed-metrics":
            saved.set()

    monkeypatch.setattr(store_module, "atomic_write", record_write)
    store = VersionStore(tmp_path / "source", disk_budget=1 << 20, max_weight_bytes=1024)
    raw = tmp_path / "input"
    raw.write_bytes(bytes(range(256)))
    store.publish("s", 0, {"w": ("BF16", [128], raw)}, model_config={})
    with ObjectServer(("127.0.0.1", 0), store) as server:
        thread = threading.Thread(target=server.serve_forever)
        thread.start()
        try:
            transport = TCPTransport(server.server_address)
            cache = VersionStore(tmp_path / "cache", disk_budget=1 << 20, max_weight_bytes=1024)
            consumer = VersionConsumer(
                transport, cache, tmp_path / "checkpoints", checkpoint_budget=1, delivery_stage="T"
            )
            with pytest.raises(OSError, match="budget"):
                consumer.prepare("s", 0, "reader")
            # Diagnostic writes may complete after the functional error returns.
            assert saved.wait(5), "Failure traffic observation was not saved"
            failure = json.loads(next((consumer.root / "failed-metrics").glob("*.json")).read_bytes())
            assert failure["failed"] and failure["bytes"]["T"] > 256 and failure["bytes"]["X"] is None
            consumer.checkpoint_budget = 32 << 20
            success = consumer.prepare("s", 0, "reader").metrics
            assert success["bytes"]["T"] > 0 and success["bytes"]["X"] is None
            observed = failure["bytes"]["T"] + success["bytes"]["T"]
            assert observed == transport.ledger.report()["bytes"]["received"]["X"]
            assert server.ledger.report()["bytes"]["sent"] == transport.ledger.report()["bytes"]["received"]
        finally:
            server.shutdown()
            thread.join()
