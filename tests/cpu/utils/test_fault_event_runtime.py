# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Real consumer threads with explicit producer/Store barriers; no Ray
needed."""

import threading
from dataclasses import replace
from types import SimpleNamespace

import pytest

from relax.utils.fault_events.models import MAX_UINT, EventDraft
from relax.utils.fault_events.runtime import FaultEventRuntime


def _draft(runtime, index=0):
    return EventDraft(runtime.run_id, "f" * 32, f"{index + 1:032x}", "actor", 0, "fault_detected", "reported_error")


@pytest.fixture
def make_runtime():
    runtimes = []

    def make(**kwargs):
        runtime = FaultEventRuntime(**kwargs)
        runtimes.append(runtime)
        return runtime

    yield make
    for runtime in runtimes:
        runtime.close()
        runtime._consumer.join(3)
        assert not runtime._consumer.is_alive(), "consumer did not stop after its barriers were released"


def _all_events(runtime):
    events, cursor = [], None
    for _ in range(10):
        page = runtime.query(limit=100, cursor=cursor)
        events.extend(page["events"])
        if page["done"]:
            return events
        cursor = page["next_cursor"]
    pytest.fail("bounded test history did not finish pagination")


def test_runtime_nine_producers_preserve_complete_256_event_set(make_runtime, monkeypatch):
    runtime = make_runtime()
    barrier = threading.Barrier(10)
    committed = threading.Event()
    original_append = runtime.store.append
    recorded_ids = set()
    results = [[] for _ in range(9)]

    def append(draft):
        assert threading.current_thread() is runtime._consumer
        result = original_append(draft)
        recorded_ids.add(draft.event_id)
        if len(recorded_ids) == 256:
            committed.set()
        return result

    def produce(worker):
        barrier.wait(3)
        for index in range(worker, 256, 9):
            results[worker].append(runtime.record(_draft(runtime, index)))

    monkeypatch.setattr(runtime.store, "append", append)
    threads = [threading.Thread(target=produce, args=(index,), daemon=True) for index in range(9)]
    for thread in threads:
        thread.start()
    barrier.wait(3)
    for thread in threads:
        thread.join(3)
        assert not thread.is_alive()
    assert sum(map(len, results)) == 256
    assert all(accepted for worker in results for accepted in worker)
    assert committed.wait(3)
    events = _all_events(runtime)
    assert {event["event_id"] for event in events} == {f"{index + 1:032x}" for index in range(256)}
    assert [event["sequence"] for event in events] == list(range(1, 257))
    diagnostics = runtime.diagnostics()
    assert diagnostics["admission_rejections"] == diagnostics["known_lost_events"] == 0
    assert not diagnostics["coverage_unknown"]
    with runtime._lock:
        assert len(runtime._pending) <= 1


@pytest.mark.parametrize("queue_capacity", [2, 512])
def test_runtime_fifo_full_and_only_one_inflight(make_runtime, monkeypatch, queue_capacity):
    runtime = make_runtime(queue_capacity=queue_capacity)
    entered, release, committed = threading.Event(), threading.Event(), threading.Event()
    original_append = runtime.store.append
    calls, active, maximum_active = [], 0, 0

    def append(draft):
        nonlocal active, maximum_active
        active += 1
        maximum_active = max(maximum_active, active)
        calls.append(draft.event_id)
        try:
            if len(calls) == 1:
                entered.set()
                assert release.wait(3)
            result = original_append(draft)
            if len(calls) == queue_capacity + 1:
                committed.set()
            return result
        finally:
            active -= 1

    monkeypatch.setattr(runtime.store, "append", append)
    try:
        assert runtime.record(_draft(runtime, 0))
        assert entered.wait(3)
        for index in range(1, queue_capacity + 1):
            assert runtime.record(_draft(runtime, index))
        assert runtime.record(_draft(runtime, queue_capacity + 1)) is False
        assert runtime.record(_draft(runtime, 0))  # Pending duplicates do not need another queue slot.
        diagnostics = runtime.diagnostics()
        assert diagnostics["queued"] == queue_capacity
        assert diagnostics["inflight"] == 1
        assert diagnostics["admission_rejections"] == 1
        assert diagnostics["known_lost_events"] == 0
        assert len(runtime._pending) == queue_capacity + 1
    finally:
        release.set()
    assert committed.wait(3)
    assert calls == [f"{index + 1:032x}" for index in range(queue_capacity + 1)]
    assert maximum_active == 1


def test_runtime_pending_state_releases_ids_across_retention_evictions(make_runtime, monkeypatch):
    runtime = make_runtime(capacity=8)
    committed_batch = threading.Barrier(2)
    original_append = runtime.store.append
    count = 0

    def append(draft):
        nonlocal count
        result = original_append(draft)
        count += 1
        if count % 128 == 0:
            committed_batch.wait(3)
        return result

    monkeypatch.setattr(runtime.store, "append", append)
    for batch in range(16):
        for index in range(batch * 128, (batch + 1) * 128):
            assert runtime.record(_draft(runtime, index))
        committed_batch.wait(3)
        with runtime._lock:
            assert len(runtime._pending) <= 1
        assert runtime.store.status()["event_count"] == 8
    assert count == 2048
    assert runtime.query()["retention_truncated"] is True
    assert runtime.diagnostics()["admission_rejections"] == 0
    runtime.close()
    runtime._consumer.join(3)
    assert not runtime._consumer.is_alive()
    assert not runtime._pending


def test_runtime_pending_duplicates_merge_and_failure_settles_once(make_runtime, monkeypatch):
    runtime = make_runtime(queue_capacity=1)
    entered, release, settled = threading.Event(), threading.Event(), threading.Event()
    original_submit, original_status = runtime._submit, runtime.store.commit_status
    calls = []
    draft = _draft(runtime)

    def append(value):
        calls.append(value)
        entered.set()
        assert release.wait(3)
        raise TimeoutError("BUSY")

    def commit_status(value):
        assert threading.current_thread() is runtime._consumer
        assert runtime._lock.acquire(blocking=False), "settlement performed Store work under admission"
        runtime._lock.release()
        return original_status(value)

    def submit(value):
        original_submit(value)
        settled.set()

    monkeypatch.setattr(runtime.store, "append", append)
    monkeypatch.setattr(runtime.store, "commit_status", commit_status)
    monkeypatch.setattr(runtime, "_submit", submit)
    try:
        assert runtime.record(draft)
        assert entered.wait(3)
        for _ in range(8):
            assert runtime.record(draft)
        assert len(runtime._pending) == 1
        assert runtime.diagnostics()["queued"] == 0
        assert runtime.diagnostics()["inflight"] == 1
    finally:
        release.set()
    assert settled.wait(3)
    assert len(calls) == 2
    assert runtime.diagnostics()["known_lost_events"] == 1
    assert runtime.diagnostics()["admission_rejections"] == 0
    assert runtime.diagnostics()["coverage_unknown"] is False


def test_runtime_pending_conflicting_content_is_rejected(make_runtime, monkeypatch):
    runtime = make_runtime()
    entered, release, committed = threading.Event(), threading.Event(), threading.Event()
    original_append = runtime.store.append
    draft = _draft(runtime)

    def append(value):
        entered.set()
        assert release.wait(3)
        result = original_append(value)
        committed.set()
        return result

    monkeypatch.setattr(runtime.store, "append", append)
    try:
        assert runtime.record(draft)
        assert entered.wait(3)
        assert runtime.record(replace(draft, reason_code="heartbeat_timeout")) is False
        assert runtime.diagnostics()["queued"] == 0
        assert runtime.diagnostics()["admission_rejections"] == 1
        assert runtime.diagnostics()["unknown_reasons"] == ["OBSERVATION_ERROR"]
    finally:
        release.set()
    assert committed.wait(3)
    assert _all_events(runtime)[0]["reason_code"] == "reported_error"
    assert runtime.diagnostics()["known_lost_events"] == 0


def test_runtime_close_does_not_count_retained_duplicate_as_lost(make_runtime, monkeypatch):
    runtime = make_runtime()
    committed, entered, release = threading.Event(), threading.Event(), threading.Event()
    original_append, original_status = runtime.store.append, runtime.store.commit_status
    draft, next_draft = _draft(runtime), _draft(runtime, 1)

    def append(value):
        if value is next_draft:
            entered.set()
            assert release.wait(3)
        result = original_append(value)
        if value is draft:
            committed.set()
        return result

    def commit_status(value):
        assert runtime._lock.acquire(blocking=False), "shutdown settlement held admission while reading Store"
        runtime._lock.release()
        return original_status(value)

    monkeypatch.setattr(runtime.store, "append", append)
    monkeypatch.setattr(runtime.store, "commit_status", commit_status)
    try:
        assert runtime.record(draft)
        assert committed.wait(3)
        assert runtime.record(next_draft)
        assert entered.wait(3)  # The first ID has left the pending table.
        assert runtime.record(draft)
        assert runtime.diagnostics()["queued"] == 1
        runtime.close()
    finally:
        release.set()
    runtime._consumer.join(3)
    assert not runtime._consumer.is_alive()
    assert {event["event_id"] for event in runtime.store.query()["events"]} == {draft.event_id, next_draft.event_id}
    assert runtime.diagnostics()["known_lost_events"] == 0
    assert runtime.diagnostics()["coverage_unknown"] is False
    assert not runtime._pending


def test_runtime_retained_conflict_preserves_original_without_guessing_loss(make_runtime, monkeypatch):
    runtime = make_runtime()
    draft = _draft(runtime)
    assert runtime.store.append(draft) == "accepted"
    settled = threading.Event()
    original_submit = runtime._submit

    def submit(value):
        original_submit(value)
        settled.set()

    monkeypatch.setattr(runtime, "_submit", submit)
    assert runtime.record(replace(draft, reason_code="heartbeat_timeout"))
    assert settled.wait(3)
    assert runtime.diagnostics()["known_lost_events"] == 0
    assert runtime.diagnostics()["unknown_reasons"] == ["OBSERVATION_ERROR"]
    assert runtime.query()["events"][0]["reason_code"] == "reported_error"


def test_runtime_retained_duplicate_with_busy_append_is_not_lost(make_runtime, monkeypatch):
    runtime = make_runtime()
    draft = _draft(runtime)
    assert runtime.store.append(draft) == "accepted"
    settled = threading.Event()
    original_submit = runtime._submit
    calls = []

    def append(value):
        calls.append(value)
        raise TimeoutError("BUSY")

    def submit(value):
        original_submit(value)
        settled.set()

    monkeypatch.setattr(runtime.store, "append", append)
    monkeypatch.setattr(runtime, "_submit", submit)
    assert runtime.record(draft)
    assert settled.wait(3)
    assert len(calls) == 2
    assert runtime.diagnostics()["known_lost_events"] == 0
    assert runtime.diagnostics()["coverage_unknown"] is False
    assert len(runtime.query()["events"]) == 1


def test_runtime_unverifiable_abandonment_reports_unknown_instead_of_lost(make_runtime, monkeypatch):
    runtime = make_runtime()
    settled = threading.Event()
    original_submit = runtime._submit

    def busy(_draft):
        raise TimeoutError("BUSY")

    def submit(value):
        original_submit(value)
        settled.set()

    monkeypatch.setattr(runtime.store, "append", busy)
    monkeypatch.setattr(runtime.store, "commit_status", busy)
    monkeypatch.setattr(runtime, "_submit", submit)
    assert runtime.record(_draft(runtime))
    assert settled.wait(3)
    assert runtime.diagnostics()["known_lost_events"] == 0
    assert runtime.diagnostics()["unknown_reasons"] == ["STORE_COMMIT_UNCERTAIN"]
    assert runtime.query()["events"] == []


@pytest.mark.parametrize("retry_succeeds", [True, False])
def test_runtime_busy_retries_same_draft_once(make_runtime, monkeypatch, retry_succeeds):
    runtime = make_runtime()
    entered, release, settled = threading.Event(), threading.Event(), threading.Event()
    original_append, original_submit = runtime.store.append, runtime._submit
    calls = []
    draft = _draft(runtime)

    def append(value):
        calls.append(value)
        if len(calls) == 1:
            entered.set()
            assert release.wait(3)
            raise TimeoutError("BUSY")
        if retry_succeeds:
            return original_append(value)
        raise TimeoutError("BUSY")

    def submit(value):
        original_submit(value)
        settled.set()

    monkeypatch.setattr(runtime.store, "append", append)
    monkeypatch.setattr(runtime, "_submit", submit)
    try:
        assert runtime.record(draft)
        assert entered.wait(3)
        assert runtime.diagnostics()["inflight"] == 1
    finally:
        release.set()
    assert settled.wait(3)
    assert len(calls) == 2 and all(value is draft for value in calls)
    diagnostics = runtime.diagnostics()
    assert diagnostics["known_lost_events"] == int(not retry_succeeds)
    assert not diagnostics["coverage_unknown"]
    assert len(_all_events(runtime)) == int(retry_succeeds)


@pytest.mark.parametrize("code", ["INVALID_EVENT", "RUN_MISMATCH", "INVALID_TIME"])
def test_runtime_permanent_store_rejection_is_known_lost(make_runtime, monkeypatch, code):
    runtime = make_runtime()
    settled = threading.Event()
    original_submit = runtime._submit
    calls = []

    def append(draft):
        calls.append(draft)
        raise ValueError(code)

    def submit(draft):
        original_submit(draft)
        settled.set()

    monkeypatch.setattr(runtime.store, "append", append)
    monkeypatch.setattr(runtime, "_submit", submit)
    assert runtime.record(_draft(runtime))
    assert settled.wait(3)
    assert len(calls) == 1
    assert runtime.diagnostics()["known_lost_events"] == 1
    assert not runtime.diagnostics()["coverage_unknown"]


@pytest.mark.parametrize("commit_first", [False, True])
@pytest.mark.parametrize("failure", ["exception", "unrecognized_value_error", "invalid_result"])
def test_runtime_uncertain_store_disables_original_history(make_runtime, monkeypatch, commit_first, failure):
    runtime = make_runtime(queue_capacity=2)
    entered, release = threading.Event(), threading.Event()
    store = runtime.store
    original_append = store.append
    calls = []

    def append(draft):
        calls.append(draft)
        entered.set()
        assert release.wait(3)
        if commit_first:
            assert original_append(draft) == "accepted"
        if failure == "invalid_result":
            return None
        if failure == "unrecognized_value_error":
            raise ValueError("untrusted details")
        raise RuntimeError("untrusted details")

    monkeypatch.setattr(store, "append", append)
    try:
        assert runtime.record(_draft(runtime))
        assert entered.wait(3)
        assert runtime.record(_draft(runtime, 1))
        assert runtime.record(_draft(runtime, 2))
    finally:
        release.set()
    runtime._consumer.join(3)
    assert not runtime._consumer.is_alive()
    assert runtime.store is store
    assert len(calls) == 1
    assert runtime.record(_draft(runtime, 3)) is False
    diagnostics = runtime.diagnostics()
    # With an invalid Store even a queued ID might repeat a previously
    # committed fact; do not invent a definite loss count.
    assert diagnostics["known_lost_events"] == 0
    assert diagnostics["unknown_reasons"] == ["STORE_COMMIT_UNCERTAIN"]
    assert diagnostics["queued"] == diagnostics["inflight"] == 0
    for query in (runtime.query, runtime.status, store.query, store.status):
        with pytest.raises(RuntimeError, match="^UNAVAILABLE$"):
            query()


def test_runtime_sequence_limit_stops_admission_but_keeps_history_readable(make_runtime, monkeypatch):
    runtime = make_runtime(queue_capacity=2)
    entered, release = threading.Event(), threading.Event()
    calls = []

    def append(draft):
        calls.append(draft)
        entered.set()
        assert release.wait(3)
        raise RuntimeError("SEQUENCE_LIMIT")

    monkeypatch.setattr(runtime.store, "append", append)
    try:
        assert runtime.record(_draft(runtime))
        assert entered.wait(3)
        assert runtime.record(_draft(runtime, 1))
        assert runtime.record(_draft(runtime, 2))
    finally:
        release.set()
    runtime._consumer.join(3)
    assert not runtime._consumer.is_alive()
    assert len(calls) == 1
    assert runtime.query()["events"] == []
    assert runtime.status()["known_lost_events"] == 3
    assert not runtime.diagnostics()["coverage_unknown"]
    assert runtime.record(_draft(runtime, 3)) is False


def test_runtime_validation_precedes_admission_lock(make_runtime, monkeypatch):
    runtime = make_runtime()
    entered, release = threading.Event(), threading.Event()
    original_validate = EventDraft.validate
    results = []

    def validate(draft):
        entered.set()
        assert release.wait(3)
        original_validate(draft)

    monkeypatch.setattr(EventDraft, "validate", validate)
    producer = threading.Thread(target=lambda: results.append(runtime.record(_draft(runtime))), daemon=True)
    producer.start()
    try:
        assert entered.wait(3)
        assert runtime._lock.acquire(timeout=0.05)
        runtime._lock.release()
        assert runtime.diagnostics()["queued"] == 0
    finally:
        release.set()
        producer.join(3)
    assert results == [True]


@pytest.mark.parametrize(
    "change",
    [None, {"run_id": "a" * 32}, {"role": "unregistered"}, {"source_generation": True}, {"event_id": "x" * 4096}],
)
def test_runtime_rejects_invalid_drafts_before_queue(make_runtime, change):
    runtime = make_runtime()
    draft = None if change is None else replace(_draft(runtime), **change)
    assert runtime.record(draft) is False
    diagnostics = runtime.diagnostics()
    assert diagnostics["admission_rejections"] == 1
    assert diagnostics["queued"] == diagnostics["inflight"] == diagnostics["known_lost_events"] == 0
    assert not diagnostics["coverage_unknown"]


def test_runtime_validation_exception_never_reaches_business(make_runtime, monkeypatch):
    runtime = make_runtime()

    def validate(_draft):
        raise OSError("private input")

    monkeypatch.setattr(EventDraft, "validate", validate)
    assert runtime.record(_draft(runtime)) is False
    assert runtime.diagnostics()["unknown_reasons"] == ["OBSERVATION_ERROR"]


@pytest.mark.parametrize("initial", [0, MAX_UINT - 3])
def test_runtime_concurrent_lock_rejections_are_atomic_and_saturate(make_runtime, initial):
    runtime = make_runtime()
    barrier = threading.Barrier(10)
    results = [[] for _ in range(9)]
    with runtime._diagnostic_lock:
        runtime._rejections = initial

    def produce(index):
        barrier.wait(3)
        for attempt in range(8):
            results[index].append(runtime.record(_draft(runtime, index * 8 + attempt)))

    with runtime._lock:
        threads = [threading.Thread(target=produce, args=(index,), daemon=True) for index in range(9)]
        for thread in threads:
            thread.start()
        barrier.wait(3)
        for thread in threads:
            thread.join(3)
            assert not thread.is_alive(), "record waited for the held admission lock after its 5 ms budget"
    assert sum(map(len, results)) == 72
    assert not any(result for worker in results for result in worker)
    diagnostics = runtime.diagnostics()
    assert diagnostics["admission_rejections"] == min(MAX_UINT, initial + 72)
    assert diagnostics["known_lost_events"] == 0


@pytest.mark.parametrize("failure, lost, unknown", [(None, 2, False), ("busy", 3, False), ("uncertain", 0, True)])
def test_runtime_close_does_not_wait_or_double_settle_inflight(make_runtime, monkeypatch, failure, lost, unknown):
    runtime = make_runtime(queue_capacity=2)
    entered, release = threading.Event(), threading.Event()
    original_append = runtime.store.append
    calls, closes = [], []
    runtime.query_server = SimpleNamespace(available=True, close=lambda: closes.append("closed"))

    def append(draft):
        calls.append(draft)
        entered.set()
        assert release.wait(3)
        if failure == "busy":
            raise TimeoutError("BUSY")
        if failure == "uncertain":
            raise OSError("commit status unavailable")
        return original_append(draft)

    monkeypatch.setattr(runtime.store, "append", append)
    try:
        assert runtime.record(_draft(runtime))
        assert entered.wait(3)
        assert runtime.record(_draft(runtime, 1))
        assert runtime.record(_draft(runtime, 2))
        runtime.close()
        runtime.close()
        assert not release.is_set()
        assert closes == ["closed"]
        assert runtime.record(_draft(runtime, 3)) is False
        assert runtime.record(_draft(runtime, 0)) is False
        assert runtime.diagnostics()["known_lost_events"] == 0
        assert runtime.diagnostics()["inflight"] == 1
        with pytest.raises(RuntimeError, match="^UNAVAILABLE$"):
            runtime.query()
    finally:
        release.set()
    runtime._consumer.join(3)
    assert not runtime._consumer.is_alive()
    assert len(calls) == 1
    diagnostics = runtime.diagnostics()
    assert diagnostics["known_lost_events"] == lost
    assert diagnostics["coverage_unknown"] is unknown
    assert diagnostics["queued"] == diagnostics["inflight"] == 0
    runtime.close()
    assert runtime.diagnostics()["known_lost_events"] == lost


def test_runtime_close_never_waits_for_admission_and_isolates_query_close(make_runtime):
    runtime = make_runtime()

    def fail_close():
        raise OSError("private server details")

    runtime.query_server = SimpleNamespace(close=fail_close)
    with runtime._lock:
        runtime.close()
        assert runtime._stop.is_set()
    assert runtime.diagnostics()["unknown_reasons"] == ["OBSERVATION_ERROR"]


def test_runtime_unexpected_consumer_failure_disables_store_and_settles_queue(make_runtime, monkeypatch):
    runtime = make_runtime(queue_capacity=2)
    entered, release = threading.Event(), threading.Event()

    def submit(_draft):
        entered.set()
        assert release.wait(3)
        raise OSError("consumer failure")

    monkeypatch.setattr(runtime, "_submit", submit)
    try:
        assert runtime.record(_draft(runtime))
        assert entered.wait(3)
        assert runtime.record(_draft(runtime, 1))
        assert runtime.record(_draft(runtime, 2))
    finally:
        release.set()
    runtime._consumer.join(3)
    assert not runtime._consumer.is_alive()
    assert runtime.diagnostics()["known_lost_events"] == 0
    assert runtime.diagnostics()["unknown_reasons"] == ["STORE_COMMIT_UNCERTAIN"]
    with pytest.raises(RuntimeError, match="^UNAVAILABLE$"):
        runtime.store.query()


def test_runtime_diagnostics_use_only_fixed_reasons_and_saturating_counters(make_runtime):
    runtime = make_runtime()
    reasons = {
        "SOURCE_RETIRED",
        "STORE_COMMIT_UNCERTAIN",
        "OUTCOME_UNOBSERVED",
        "CONTEXT_CAPACITY",
        "OBSERVATION_ERROR",
        "RECEIPT_CONFLICT",
    }
    for reason in reasons:
        runtime.note_unknown(reason)
    for value in ("arbitrary secret", [], {}, None):
        runtime.note_unknown(value)
    runtime._increment("_lost", MAX_UINT)
    runtime._increment("_lost")
    diagnostics = runtime.diagnostics()
    assert diagnostics["known_lost_events"] == MAX_UINT
    assert diagnostics["unknown_reasons"] == sorted(reasons)
    assert diagnostics["coverage_unknown"] is True


@pytest.mark.parametrize("operation", ["query", "status"])
def test_runtime_unavailable_store_read_sets_unknown_and_disables_admission(make_runtime, operation):
    runtime = make_runtime()
    runtime.store.invalidate()
    with pytest.raises(RuntimeError, match="^UNAVAILABLE$"):
        getattr(runtime, operation)()
    assert runtime.record(_draft(runtime)) is False
    assert runtime.diagnostics()["unknown_reasons"] == ["STORE_COMMIT_UNCERTAIN"]


@pytest.mark.parametrize("capacity", [0, 513, True, "512"])
def test_runtime_rejects_out_of_contract_queue_capacity(capacity):
    with pytest.raises(ValueError, match="^INVALID_CAPACITY$"):
        FaultEventRuntime(queue_capacity=capacity)
