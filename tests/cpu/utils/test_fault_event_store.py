# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import base64
import gc
import json
import resource
import sys
import tracemalloc
from concurrent.futures import ThreadPoolExecutor
from dataclasses import FrozenInstanceError, asdict, replace

import pytest

from relax.utils.fault_events import EventDraft, FaultEventStore
from relax.utils.fault_events.models import MAX_EVENT_BYTES, MAX_UINT, encode, new_id


def event(run_id, *, role="actor", fault_id=None):
    return EventDraft(run_id, fault_id or new_id(), new_id(), role, 0, "fault_detected", "reported_error")


def cursor(run_id, after, end, role=None, fault_id=None, version=1):
    payload = [version, run_id, role, fault_id, after, end]
    return base64.urlsafe_b64encode(json.dumps(payload).encode()).decode()


@pytest.mark.parametrize(
    "kind,reason,mode,expected,observed,failed",
    [
        ("fault_detected", "reported_error", None, 0, 0, 0),
        ("fault_detected", "heartbeat_timeout", None, 0, 0, 0),
        ("fault_detected", "fatal_error", None, 0, 0, 0),
        ("fatal_escalated", "fatal_error", None, 0, 0, 0),
        ("handling_started", None, "local", 7, 0, 0),
        ("recovery_handoff", None, "global", 31, 7, 0),
        ("recovery_succeeded", None, "local", 7, 7, 0),
        ("recovery_failed", "step_restore_failed", "local", 7, 1, 2),
        ("recovery_failed", "operation_failed", "global", 31, 7, 8),
        ("recovery_failed", "restart_exception", "global", 5, 0, 0),
        ("recovery_failed", "handoff_exception", "global", 5, 1, 0),
        ("recovery_interrupted", "superseded", "global", 31, 1, 0),
        ("recovery_interrupted", "source_retired", "global", 31, 7, 0),
        ("handling_skipped", "role_unregistered", None, 0, 0, 0),
        ("termination_requested", "fatal_error", None, 0, 0, 0),
        ("termination_requested", "restart_limit", "global", 31, 0, 0),
    ],
)
def test_event_contract_examples(kind, reason, mode, expected, observed, failed):
    draft = replace(
        event(new_id()),
        event_type=kind,
        reason_code=reason,
        recovery_mode=mode,
        recovery_id=new_id() if mode or kind == "handling_skipped" else None,
        expected_evidence=expected,
        observed_evidence=observed,
        failed_evidence=failed,
    )
    draft.validate()
    record = draft.record(MAX_UINT, MAX_UINT)
    assert len(encode(record)) <= MAX_EVENT_BYTES
    assert record["schema_version"] == 1
    assert record["availability"] == ("operations_completed" if kind == "recovery_succeeded" else "unverified")
    if kind == "recovery_handoff":
        assert record["result_scope"] == "control_flow_handoff"
    elif kind in ("recovery_succeeded", "recovery_failed", "recovery_interrupted"):
        assert record["result_scope"] == "recovery_operations"
    else:
        assert record["result_scope"] is None
    assert EventDraft.from_dict(asdict(draft)) == draft


@pytest.mark.parametrize(
    "changes",
    [
        {"role": []},
        {"role": "actor" * 1000},
        {"event_type": {}},
        {"event_type": "UNKNOWN"},
        {"source_generation": True},
        {"source_generation": -1},
        {"source_generation": MAX_UINT + 1},
        {"run_id": "A" * 32},
        {"event_id": "a" * 33},
        {"fault_id": b"a" * 32},
        {"reason_code": "SENSITIVE_INPUT"},
        {"recovery_mode": []},
        {"observed_evidence": True},
        {"observed_evidence": 1},
        {"expected_evidence": 64},
        {"recovery_id": new_id()},
        {"event_type": "recovery_succeeded", "reason_code": None},
        {"event_type": "handling_skipped", "reason_code": "role_unregistered"},
        {
            "event_type": "recovery_failed",
            "reason_code": "fatal_error",
            "recovery_mode": "global",
            "recovery_id": new_id(),
            "expected_evidence": 5,
        },
        {
            "event_type": "termination_requested",
            "reason_code": "restart_limit",
            "recovery_mode": "local",
            "recovery_id": new_id(),
            "expected_evidence": 7,
        },
    ],
)
def test_invalid_events_cannot_consume_sequence_or_evict(changes):
    store = FaultEventStore(new_id(), 1)
    original = event(store.run_id)
    store.append(original)
    with pytest.raises(ValueError, match="^INVALID_EVENT$"):
        store.append(replace(event(store.run_id), **changes))
    assert store.status()["latest_sequence"] == 1
    assert store.query()["events"][0]["event_id"] == original.event_id
    assert set(store._ids) == {original.event_id}


@pytest.mark.parametrize("observed,failed", [(3, 2), (8, 0), (1, 8), (1, 0), (5, 2)])
def test_success_requires_every_expected_bit_and_no_failed_bit(observed, failed):
    draft = replace(
        event(new_id()),
        event_type="recovery_succeeded",
        reason_code=None,
        recovery_id=new_id(),
        recovery_mode="local",
        expected_evidence=7,
        observed_evidence=observed,
        failed_evidence=failed,
    )
    with pytest.raises(ValueError, match="^INVALID_EVENT$"):
        draft.validate()


def test_unknown_fields_and_objects_are_not_serialized_or_echoed():
    class Trap:
        def __str__(self):
            pytest.fail("unknown object string conversion")

        def __repr__(self):
            pytest.fail("unknown object representation")

        def __hash__(self):
            pytest.fail("unknown object hashing")

    draft = event(new_id())
    for field in ("role", "event_type", "reason_code", "run_id", "recovery_mode", "expected_evidence"):
        with pytest.raises(ValueError, match="^INVALID_EVENT$"):
            replace(draft, **{field: Trap()}).validate()
    with pytest.raises(ValueError, match="^INVALID_EVENT$"):
        EventDraft.from_dict({**asdict(draft), "secret": Trap()})
    store = FaultEventStore(draft.run_id)
    for params in ({"role": Trap()}, {"cursor": Trap()}, {"fault_id": Trap()}, {"limit": Trap()}):
        with pytest.raises(ValueError, match="^INVALID_QUERY$"):
            store.query(**params)


def test_size_rejection_is_before_store_mutation(monkeypatch):
    store = FaultEventStore(new_id(), 1)
    first = event(store.run_id)
    store.append(first)
    monkeypatch.setattr("relax.utils.fault_events.models.MAX_EVENT_BYTES", 1)
    with pytest.raises(ValueError, match="^EVENT_TOO_LARGE$"):
        store.append(event(store.run_id))
    assert store.status()["latest_sequence"] == 1
    assert store.query()["events"][0]["event_id"] == first.event_id


def test_deduplication_conflict_and_copies_do_not_rewrite_history():
    store = FaultEventStore(new_id(), 2)
    draft = event(store.run_id)
    assert store.append(draft) == "accepted"
    assert store.append(replace(draft)) == "duplicate"
    with pytest.raises(ValueError, match="^CONFLICT$"):
        store.append(replace(draft, role="critic"))
    record = store.query()["events"][0]
    record["role"] = "critic"
    assert store.query()["events"][0]["role"] == "actor"
    assert store.status()["latest_sequence"] == 1
    with pytest.raises(FrozenInstanceError):
        draft.role = "critic"


def test_consumer_commit_status_distinguishes_retained_ids_without_writing():
    store = FaultEventStore(new_id(), 1)
    draft = event(store.run_id)
    assert store.commit_status(draft) == "absent"
    store.append(draft)
    assert store.commit_status(draft) == "duplicate"
    assert store.commit_status(replace(draft, role="critic")) == "conflict"
    assert store.status()["latest_sequence"] == 1
    store.append(event(store.run_id))
    assert store.commit_status(draft) == "absent"
    with store._lock:
        with pytest.raises(TimeoutError, match="^BUSY$"):
            store.commit_status(draft)
    store.invalidate()
    with pytest.raises(RuntimeError, match="^UNAVAILABLE$"):
        store.commit_status(draft)


def test_concurrent_duplicate_append_allocates_one_sequence():
    store = FaultEventStore(new_id())
    draft = event(store.run_id)
    with ThreadPoolExecutor(max_workers=9) as executor:
        results = list(executor.map(store.append, [draft] * 64))
    assert results.count("accepted") == 1
    assert results.count("duplicate") == 63
    assert store.status()["latest_sequence"] == 1


def test_ring_and_index_evict_together_and_only_retain_bounded_dedup():
    store = FaultEventStore(new_id(), 2)
    drafts = [event(store.run_id) for _ in range(3)]
    for draft in drafts:
        store.append(draft)
    assert set(store._ids) == {draft.event_id for draft in drafts[1:]}
    assert len(store._ring) == len(store._ids) == 2
    assert store.query()["missing_range"] == [1, 1]
    # Cross-retention lifecycle dedup belongs to the later FaultSource layer;
    # the low-level Store never grows an unbounded tombstone index.
    assert store.append(drafts[0]) == "accepted"
    assert [row["sequence"] for row in store.query()["events"]] == [3, 4]


def test_snapshot_gap_boundary_and_all_unread_eviction():
    store = FaultEventStore(new_id(), 5)
    for _ in range(5):
        store.append(event(store.run_id))
    first = store.query(limit=2)
    for _ in range(2):
        store.append(event(store.run_id))
    assert store.query(cursor=first["next_cursor"])["missing_range"] is None
    store.append(event(store.run_id))
    page = store.query(cursor=first["next_cursor"])
    assert page["missing_range"] == [3, 3]
    assert page["snapshot_end_seq"] == 5
    assert [row["sequence"] for row in page["events"]] == [4, 5]
    assert page["done"] and page["next_cursor"] is None
    for _ in range(3):
        store.append(event(store.run_id))
    page = store.query(cursor=first["next_cursor"])
    assert page["events"] == [] and page["missing_range"] == [3, 5]
    assert page["done"] and page["snapshot_end_seq"] == 5


def test_filters_intersect_and_global_missing_range_is_not_a_fault_count():
    store = FaultEventStore(new_id(), 3)
    a, b = new_id(), new_id()
    for role, fault in (("actor", a), ("critic", a), ("actor", b), ("actor", a)):
        store.append(event(store.run_id, role=role, fault_id=fault))
    page = store.query(role="actor", fault_id=a)
    assert [row["sequence"] for row in page["events"]] == [4]
    assert page["missing_range"] == [1, 1]
    assert page["gap_scope"] == "global" and page["filter_match_unknown"]
    assert not store.query()["filter_match_unknown"]


def test_sparse_empty_page_advances_by_scanned_positions():
    store = FaultEventStore(new_id(), 600)
    for _ in range(550):
        store.append(event(store.run_id))
    store.append(event(store.run_id, role="rollout"))
    page = store.query(role="rollout")
    assert page["events"] == [] and not page["done"]
    payload = json.loads(base64.urlsafe_b64decode(page["next_cursor"]))
    assert payload[-2:] == [512, 551]
    next_page = store.query(role="rollout", cursor=page["next_cursor"])
    assert [row["sequence"] for row in next_page["events"]] == [551]
    assert next_page["done"]


@pytest.mark.parametrize(
    "params",
    [
        {"limit": True},
        {"limit": 0},
        {"limit": 101},
        {"limit": "5"},
        {"role": []},
        {"role": "unknown"},
        {"fault_id": "unknown"},
        {"cursor": ""},
        {"cursor": "X" * 1025},
        {"cursor": "汉字"},
        {"cursor": "!!!!"},
    ],
)
def test_invalid_query_is_a_fixed_error(params):
    with pytest.raises(ValueError, match="^INVALID_QUERY$"):
        FaultEventStore(new_id()).query(**params)


@pytest.mark.parametrize(
    "version,after,end",
    [(True, 0, 1), (2, 0, 1), (1, -1, 1), (1, True, 1), (1, 2, 1), (1, 0, 2), (1, 0, MAX_UINT + 1)],
)
def test_invalid_cursor_watermarks_and_version(version, after, end):
    store = FaultEventStore(new_id())
    store.append(event(store.run_id))
    with pytest.raises(ValueError, match="^INVALID_QUERY$"):
        store.query(cursor=cursor(store.run_id, after, end, version=version))


def test_cursor_binds_run_and_filters_but_allows_page_size_change():
    store = FaultEventStore(new_id())
    for _ in range(3):
        store.append(event(store.run_id))
    saved = store.query(role="actor", limit=1)["next_cursor"]
    assert len(store.query(role="actor", cursor=saved, limit=2)["events"]) == 2
    with pytest.raises(ValueError, match="^INVALID_QUERY$"):
        store.query(cursor=saved, role="critic")
    with pytest.raises(RuntimeError, match="^RUN_MISMATCH$"):
        FaultEventStore(new_id()).query(cursor=saved, role="actor")


def test_sequence_exhaustion_never_wraps_and_history_remains_readable():
    store = FaultEventStore(new_id(), 1)
    store._latest = MAX_UINT - 1
    draft = event(store.run_id)
    store.append(draft)
    with pytest.raises(RuntimeError, match="^SEQUENCE_LIMIT$"):
        store.append(event(store.run_id))
    assert store.append(draft) == "duplicate"
    page = store.query(cursor=cursor(store.run_id, MAX_UINT - 1, MAX_UINT))
    assert [row["sequence"] for row in page["events"]] == [MAX_UINT]
    assert page["missing_range"] is None


def test_clock_is_display_only_and_invalid_clock_does_not_evict(monkeypatch):
    store = FaultEventStore(new_id(), 2)
    for timestamp in (20, 10):
        monkeypatch.setattr("relax.utils.fault_events.store.time.time_ns", lambda t=timestamp: t * 1_000_000)
        store.append(event(store.run_id))
    rows = store.query()["events"]
    assert [row["sequence"] for row in rows] == [1, 2]
    assert [row["recorded_at_ms"] for row in rows] == [20, 10]
    monkeypatch.setattr("relax.utils.fault_events.store.time.time_ns", lambda: -1)
    with pytest.raises(ValueError, match="^INVALID_TIME$"):
        store.append(event(store.run_id))
    assert store.query()["events"] == rows
    assert store.append(store._ids[rows[0]["event_id"]].draft) == "duplicate"


def test_query_lock_timeout_and_corruption_do_not_return_a_false_empty_page():
    store = FaultEventStore(new_id())
    store.append(event(store.run_id))
    with store._lock:
        with pytest.raises(TimeoutError, match="^BUSY$"):
            store.query()
    store._ids.clear()
    with pytest.raises(RuntimeError, match="^UNAVAILABLE$"):
        store.query()
    with pytest.raises(RuntimeError, match="^UNAVAILABLE$"):
        store.append(event(store.run_id))
    with pytest.raises(RuntimeError, match="^UNAVAILABLE$"):
        store.status()


def test_query_projection_and_cursor_encoding_run_outside_store_lock(monkeypatch):
    store = FaultEventStore(new_id())
    for _ in range(2):
        store.append(event(store.run_id))
    project, dumps = EventDraft.record, json.dumps

    def checked_project(self, *args):
        assert not store._lock.locked()
        return project(self, *args)

    def checked_dumps(*args, **kwargs):
        assert not store._lock.locked()
        return dumps(*args, **kwargs)

    monkeypatch.setattr(EventDraft, "record", checked_project)
    monkeypatch.setattr("relax.utils.fault_events.store.json.dumps", checked_dumps)
    assert store.query(limit=1)["next_cursor"]


def test_fixed_capacity_long_run_bounds_heap_and_process_growth():
    store = FaultEventStore(new_id(), 32)
    for _ in range(128):
        store.append(event(store.run_id))
    gc.collect()
    rss_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    tracemalloc.start()
    try:
        before = tracemalloc.get_traced_memory()[0]
        for _ in range(4096):
            store.append(event(store.run_id))
        gc.collect()
        growth = tracemalloc.get_traced_memory()[0] - before
        rss_after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        rss_bytes = (rss_after - rss_before) * (1 if sys.platform == "darwin" else 1024)
        assert len(store._ring) == len(store._ids) == 32
        assert growth < 256 * 1024
        assert rss_bytes < 32 * 1024 * 1024
    finally:
        tracemalloc.stop()
