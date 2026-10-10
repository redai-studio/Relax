import pytest

from relax.utils.failure_events import FailureEvent, FailureEventStore


def make_event(
    fault_id: str,
    phase: str,
    *,
    role: str = "actor",
    attempt: int | None = None,
) -> FailureEvent:
    return FailureEvent(
        fault_id=fault_id,
        role=role,
        phase=phase,
        occurred_at_ms=1,
        attempt=attempt,
    )


def test_append_assigns_monotonic_sequence():
    store = FailureEventStore(capacity=4)

    first = store.append(make_event("fault-1", "detected"))
    second = store.append(make_event("fault-1", "handling_started", attempt=1))

    assert first == {"seq": 1, "appended": True}
    assert second == {"seq": 2, "appended": True}
    assert [event["seq"] for event in store.query()["events"]] == [1, 2]


def test_duplicate_event_is_not_appended_twice():
    store = FailureEventStore()

    event = make_event("fault-1", "detected")

    first = store.append(event)
    duplicate = store.append(event)

    assert first == {"seq": 1, "appended": True}
    assert duplicate == {"seq": 1, "appended": False}
    assert len(store.query()["events"]) == 1


def test_capacity_evicts_oldest_event():
    store = FailureEventStore(capacity=2)

    store.append(make_event("fault-1", "detected"))
    store.append(make_event("fault-1", "handling_started", attempt=1))
    store.append(make_event("fault-1", "recovery_succeeded", attempt=1))

    page = store.query()

    assert [event["seq"] for event in page["events"]] == [2, 3]
    assert page["oldest_seq"] == 2
    assert page["latest_seq"] == 3


def test_sequence_continues_after_eviction():
    store = FailureEventStore(capacity=2)

    store.append(make_event("fault-1", "detected"))
    store.append(make_event("fault-2", "detected"))
    third = store.append(make_event("fault-3", "detected"))

    assert third == {"seq": 3, "appended": True}


def test_query_filters_by_role_and_fault_id():
    store = FailureEventStore()

    store.append(make_event("fault-a", "detected", role="actor"))
    store.append(make_event("fault-b", "detected", role="critic"))
    store.append(
        make_event(
            "fault-a",
            "handling_started",
            role="actor",
            attempt=1,
        )
    )

    by_role = store.query(role="actor")
    assert [event["seq"] for event in by_role["events"]] == [1, 3]

    by_fault = store.query(fault_id="fault-a")
    assert [event["seq"] for event in by_fault["events"]] == [1, 3]

    intersection = store.query(role="critic", fault_id="fault-a")
    assert intersection["events"] == []


def test_query_paginates_with_after_seq():
    store = FailureEventStore()

    for i in range(5):
        store.append(make_event(f"fault-{i}", "detected"))

    first = store.query(limit=2)
    assert [event["seq"] for event in first["events"]] == [1, 2]
    assert first["next_after_seq"] == 2

    second = store.query(after_seq=first["next_after_seq"], limit=2)
    assert [event["seq"] for event in second["events"]] == [3, 4]
    assert second["next_after_seq"] == 4

    third = store.query(after_seq=second["next_after_seq"], limit=2)
    assert [event["seq"] for event in third["events"]] == [5]
    assert third["next_after_seq"] is None


def test_filtered_pagination_advances_over_nonmatching_events():
    store = FailureEventStore()

    store.append(make_event("fault-1", "detected", role="critic"))
    store.append(make_event("fault-2", "detected", role="actor"))
    store.append(make_event("fault-3", "detected", role="critic"))
    store.append(make_event("fault-4", "detected", role="actor"))

    first = store.query(role="actor", limit=1)

    assert [event["seq"] for event in first["events"]] == [2]
    assert first["next_after_seq"] == 2

    second = store.query(
        role="actor",
        after_seq=first["next_after_seq"],
        limit=1,
    )

    assert [event["seq"] for event in second["events"]] == [4]
    assert second["next_after_seq"] is None


def test_history_truncated_only_when_requested_history_was_evicted():
    store = FailureEventStore(capacity=2)

    store.append(make_event("fault-1", "detected"))
    store.append(make_event("fault-2", "detected"))
    store.append(make_event("fault-3", "detected"))

    # Retained sequence is [2, 3].
    assert store.query(after_seq=0)["history_truncated"] is True
    assert store.query(after_seq=1)["history_truncated"] is False
    assert store.query(after_seq=2)["history_truncated"] is False


def test_event_id_is_derived_from_fault_phase_and_attempt():
    event = make_event(
        "fault-1",
        "handling_started",
        attempt=2,
    )

    assert event.event_id == "fault-1:handling_started:2"


@pytest.mark.parametrize("capacity", [0, -1])
def test_invalid_capacity_is_rejected(capacity):
    with pytest.raises(ValueError, match="capacity must be > 0"):
        FailureEventStore(capacity=capacity)


def test_invalid_query_pagination_is_rejected():
    store = FailureEventStore()

    with pytest.raises(ValueError, match="after_seq must be >= 0"):
        store.query(after_seq=-1)

    with pytest.raises(ValueError, match="limit must be > 0"):
        store.query(limit=0)
