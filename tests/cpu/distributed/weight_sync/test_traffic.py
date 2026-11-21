from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context

import pytest

from relax.distributed.weight_sync import traffic
from relax.distributed.weight_sync.ledger import aggregate_events
from relax.distributed.weight_sync.metrics import SyncMetrics
from relax.distributed.weight_sync.publication import VersionPublisher
from relax.distributed.weight_sync.store import VersionStore, atomic_write


def test_thread_context_and_nested_context(tmp_path):
    outer, inner = SyncMetrics("a"), SyncMetrics("b")
    with traffic.capture(outer), ThreadPoolExecutor(1) as pool:
        pool.submit(copy_context().run, atomic_write, tmp_path / "a", b"123").result()
        with traffic.capture(inner):
            atomic_write(tmp_path / "b", b"12345")
    atomic_write(tmp_path / "untracked", b"00000000")
    assert outer.report()["bytes_by_scope"]["file_write"]["W"] == 3
    assert inner.report()["bytes_by_scope"]["file_write"]["W"] == 5


def test_publication_thread_counts_actual_objects_and_retry(tmp_path):
    metrics = SyncMetrics("u")
    store = VersionStore(tmp_path / "store", disk_budget=1024**2, max_weight_bytes=1024, canonical_blocks=True)
    source = tmp_path / "source"
    source.write_bytes(b"\x00\x01" * 16)
    with traffic.capture(metrics):
        publisher = VersionPublisher(store, "s", tmp_path / "scratch", model_config={}, converter_revision="test")
        publisher.finalize_pending(publisher.submit("u", {"w": ("BF16", [16], source)}))
        publisher.close()
    links = metrics.report()["links"]
    assert any("objects" in row["link"] for row in links)
    assert any("candidate-" in row["link"] for row in links)
    assert len({row["event_id"] for row in links}) == len(links)
    assert metrics.report()["physical_link_coverage_complete"] is False


def test_async_completion_is_counted_once():
    class Work:
        def wait(self):
            return True

    metrics = SyncMetrics("u")
    wrapped = traffic._MeasuredWork(Work(), 123, "link", metrics)
    wrapped.wait()
    wrapped.wait()
    assert metrics.report()["bytes_by_scope"]["collective_payload"]["G"] == 123


def test_retry_journal_is_counted_without_an_outer_capture(tmp_path, monkeypatch):
    metrics = SyncMetrics("retry")
    store = VersionStore(tmp_path / "store", disk_budget=1024**2, max_weight_bytes=1024)
    raw = tmp_path / "source"
    raw.write_bytes(b"\x00\x01" * 16)
    original = store.publish

    def fail(*args, **kwargs):
        raise OSError("injected storage failure")

    with traffic.capture(metrics):
        publisher = VersionPublisher(store, "s", tmp_path / "scratch", model_config={}, converter_revision="test")
        monkeypatch.setattr(store, "publish", fail)
        handle = publisher.submit("retry", {"w": ("BF16", [16], raw)})
    with pytest.raises(OSError, match="injected"):
        publisher.wait_committed(handle)
    monkeypatch.setattr(store, "publish", original)
    publisher.retry(handle)
    publisher.finalize_pending(handle)
    publisher.close()
    assert any(
        item["attempt"] == 2 and item["stage"] == "W" and item["link"].endswith("publisher-s.json")
        for item in metrics.report()["links"]
    )


def test_wire_observation_does_not_claim_exclusive_attribution():
    observed = traffic.interface_delta(
        {"lo": {"tx_bytes": 5, "rx_bytes": 8}}, {"lo": {"tx_bytes": 20, "rx_bytes": 23}}
    )
    assert observed["interfaces"]["lo"]["tx_bytes"] == 15
    assert observed["attribution"] == "includes_concurrent_traffic"
    with pytest.raises(ValueError):
        traffic.interface_delta({"lo": {"tx_bytes": 50, "rx_bytes": 8}}, {"lo": {"tx_bytes": 20, "rx_bytes": 23}})


def test_recollection_deduplicates_but_retry_remains():
    first, retry = SyncMetrics("u"), SyncMetrics("u")
    first.transferred("G", 128, link="r0->r1", scope="collective_payload")
    retry.transferred("G", 128, link="r0->r1", scope="collective_payload", attempt=2)
    first.transferred("W", 1024, link="local-file", scope="file_write")
    merged = aggregate_events([first.report(), first.report(), retry.report()])
    assert merged["events"] == 3
    assert merged["bytes_by_scope"]["collective_payload"]["G"] == 256
    assert merged["bytes_by_scope"]["file_write"]["W"] == 1024
    assert merged["missing_stages"] == ["C", "T", "X"]


def test_mount_observations_keep_rpc_bytes_separate_from_application():
    raw = "device server:/store mounted on /data with fstype nfs4 statvers=1.1\n"
    raw += " READ: 2 3 0 300 1024 0 0 0\n WRITE: 1 1 0 2048 128 0 0 0\n"
    raw += "device other:/store mounted on /other with fstype nfs4 statvers=1.1\n READ: 90 90 0 999 999 0 0 0\n"
    before = traffic.mount_operations(raw, "/data")
    assert before["READ"]["sent_bytes"] == 300
    after = {key: {field: value * 2 for field, value in values.items()} for key, values in before.items()}
    delta = traffic.mount_delta(before, after)
    assert delta["operations"]["READ"]["received_bytes"] == 1024
    assert delta["attribution"] == "includes_concurrent_traffic"


def test_missing_rank_receipt_is_visible_even_with_duplicate_owner_report():
    owner, other = SyncMetrics("u"), SyncMetrics("u")
    owner.source_participant = {"rank": 0, "world_size": 2}
    other.source_participant = {"rank": 1, "world_size": 2}
    assert aggregate_events([owner.report(), owner.report()])["missing_source_ranks"] == {"u": [1]}
    assert aggregate_events([owner.report(), other.report()])["missing_source_ranks"] == {}


def test_shared_mount_receipts_preserve_overlap_evidence(monkeypatch, tmp_path):
    snapshots = iter(
        [
            {"WRITE": {"sent_bytes": 100}},
            {"WRITE": {"sent_bytes": 160}},
            {"WRITE": {"sent_bytes": 200}},
            {"WRITE": {"sent_bytes": 250}},
        ]
    )
    monkeypatch.setattr(traffic, "process_io", lambda: {"write_bytes": 0})
    monkeypatch.setattr(traffic, "filesystem_info", lambda path: {"type": "nfs4", "mount": "/data"})
    monkeypatch.setattr(traffic, "mount_snapshot", lambda mount: next(snapshots))
    with traffic.IOObservation(path=tmp_path) as outer:
        with traffic.IOObservation(path=tmp_path) as inner:
            pass
    # The inner 40 bytes belong to the outer 150-byte interval too. Raw
    # endpoints allow an auditor to identify this instead of summing 190.
    assert outer.report["filesystem_rpc"]["operations"]["WRITE"]["sent_bytes"] == 150
    assert inner.report["filesystem_rpc"]["operations"]["WRITE"]["sent_bytes"] == 40
    assert outer.report["counter_snapshots"]["filesystem_rpc"]["before"]["WRITE"]["sent_bytes"] == 100
    assert outer.report["counter_snapshots"]["filesystem_rpc"]["after"]["WRITE"]["sent_bytes"] == 250
    assert outer.report["host"] == inner.report["host"]
    assert (
        outer.report["start_unix_ns"]
        <= inner.report["start_unix_ns"]
        <= inner.report["end_unix_ns"]
        <= outer.report["end_unix_ns"]
    )


def test_patch_budget_counts_remote_payload_and_padded_control():
    from relax.distributed.weight_sync.source_patch import replacement_cost

    plans = [{"payload_bytes": 6, "frames": [{"size": 6, "sha256": "0" * 64}]} for _ in range(4)]
    proposals = [{"plans": {rank: plans[rank]}, "error": None, "has_base": True} for rank in range(4)]
    cost = replacement_cost([0, 1, 2, 3], plans, proposals)
    assert cost["payload_bytes"] == 18
    assert cost["control_bytes"] == traffic.object_all_gather_bytes(proposals) + 10 * traffic.object_all_gather_bytes(
        [None] * 4
    )
    assert sum(cost.values()) > 4 * 256
    assert traffic.object_all_gather_bytes([None]) == 0
