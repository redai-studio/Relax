# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Unit tests for the straggler wire protocol and its bounded deduplicator."""

import json
from types import SimpleNamespace

from relax.utils.straggler.observer import TimingEnvelope
from relax.utils.straggler.protocol import (
    DEDUP_MAX_ENTRIES,
    DEDUP_TTL_S,
    MEASUREMENT_KINDS,
    PROTOCOL_NAME,
    SCHEMA_VERSION,
    WORKLOAD_FIELDS,
    BoundedDedup,
    dedup_key,
    validate,
)


class FakeClock:
    """A manually advanced clock, so TTL behaviour is deterministic."""

    def __init__(self, now: float = 0.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def make_envelope(rank: int = 0, seq: int = 1) -> TimingEnvelope:
    """Build a minimal real envelope for the object-path checks."""
    return TimingEnvelope(
        run_id="run-1",
        rank=rank,
        cohort="0:0:0:0:0",
        label=f"rank{rank}",
        world_size=4,
        name="forward-compute",
        log_level=2,
        seq=seq,
        host_start=1.0,
        host_end=1.1,
        device_ms=None,
        barrier=False,
        reason="device",
    )


class TestDedupKey:
    def test_reads_a_mapping(self) -> None:
        payload = {"run_id": "run-1", "topology_epoch": "topo2", "global_rank": 5, "sample_seq": 9}

        assert dedup_key(payload) == ("run-1", "topo2", 5, 9)

    def test_reads_an_object(self) -> None:
        payload = SimpleNamespace(run_id="run-1", topology_epoch="topo2", global_rank=5, sample_seq=9)

        assert dedup_key(payload) == ("run-1", "topo2", 5, 9)

    def test_falls_back_to_rank_and_seq(self) -> None:
        assert dedup_key({"run_id": "run-1", "rank": 3, "seq": 7}) == ("run-1", "", 3, 7)

    def test_prefers_the_explicit_fields(self) -> None:
        payload = {"run_id": "run-1", "rank": 1, "seq": 2, "global_rank": 5, "sample_seq": 9}

        assert dedup_key(payload) == ("run-1", "", 5, 9)

    def test_uses_the_envelope_fields_when_present(self) -> None:
        assert dedup_key(make_envelope(rank=2, seq=12)) == ("run-1", "", 2, 12)

    def test_junk_keys_never_raise(self) -> None:
        assert dedup_key(None) == ("", "", -1, 0)
        assert dedup_key(object()) == ("", "", -1, 0)
        assert dedup_key({"rank": "not-an-int", "seq": "not-an-int"}) == ("", "", -1, 0)
        assert dedup_key({"run_id": "r", "global_rank": None, "rank": None, "seq": None}) == ("r", "", -1, 0)


class TestValidate:
    def test_accepts_a_well_formed_packet(self) -> None:
        payload = {"schema_version": SCHEMA_VERSION, "rank": 0, "name": "forward-compute"}

        assert validate(payload) == (True, "")

    def test_accepts_every_measurement_kind(self) -> None:
        for kind in MEASUREMENT_KINDS:
            payload = {
                "schema_version": SCHEMA_VERSION,
                "rank": 0,
                "name": "forward-compute",
                "measurement_kind": kind,
            }
            assert validate(payload)[0] is True

    def test_missing_schema_version_is_rejected(self) -> None:
        assert validate({}) == (False, "missing_schema_version")
        assert validate({"rank": 0, "name": "forward-compute"}) == (False, "missing_schema_version")

    def test_bad_rank_is_rejected(self) -> None:
        for rank in (-1, True, "0", 0.0, None):
            payload = {"schema_version": SCHEMA_VERSION, "rank": rank, "name": "forward-compute"}
            assert validate(payload) == (False, "invalid_rank")

    def test_bad_name_is_rejected(self) -> None:
        for name in ("", None, 3, b"forward"):
            payload = {"schema_version": SCHEMA_VERSION, "rank": 0, "name": name}
            assert validate(payload) == (False, "invalid_name")

    def test_unknown_measurement_kind_is_rejected(self) -> None:
        payload = {"schema_version": SCHEMA_VERSION, "rank": 0, "name": "forward-compute", "measurement_kind": "gpu"}

        assert validate(payload) == (False, "invalid_measurement_kind")

    def test_empty_measurement_kind_counts_as_unpopulated(self) -> None:
        payload = {"schema_version": SCHEMA_VERSION, "rank": 0, "name": "forward-compute", "measurement_kind": ""}

        assert validate(payload) == (True, "")

    def test_workload_counters_are_advisory(self) -> None:
        payload = {"schema_version": SCHEMA_VERSION, "rank": 0, "name": "forward-compute", "tokens": "lots"}

        assert validate(payload)[0] is True

    def test_local_envelope_without_schema_version_is_trusted(self) -> None:
        # The observer owns the dataclass; this module must keep working before
        # it grows a schema_version attribute.
        assert validate(make_envelope())[0] is True
        assert validate(SimpleNamespace(rank=0, name="forward-compute"))[0] is True

    def test_local_envelope_with_a_bad_rank_is_still_rejected(self) -> None:
        assert validate(SimpleNamespace(rank=-1, name="forward-compute")) == (False, "invalid_rank")

    def test_never_raises_on_junk(self) -> None:
        for junk in (None, 42, "x", [], object(), {"rank": {}, "name": []}):
            result = validate(junk)
            assert isinstance(result, tuple) and len(result) == 2
            assert isinstance(result[0], bool) and isinstance(result[1], str)


class TestBoundedDedup:
    def test_new_then_duplicate_within_the_ttl(self) -> None:
        tracker = BoundedDedup(clock=FakeClock())
        key = dedup_key({"run_id": "r", "rank": 0, "seq": 1})

        assert tracker.check(key) == "new"
        assert tracker.check(key) == "duplicate"
        assert tracker.stats()["new"] == 1
        assert tracker.stats()["duplicate"] == 1

    def test_a_resend_after_the_ttl_is_new_again(self) -> None:
        clock = FakeClock()
        tracker = BoundedDedup(ttl_s=10.0, clock=clock)
        key = dedup_key({"run_id": "r", "rank": 0, "seq": 1})

        assert tracker.check(key) == "new"
        clock.advance(5.0)
        assert tracker.check(key) == "duplicate"
        clock.advance(6.0)
        assert tracker.check(key) == "new"
        assert tracker.stats()["expired"] >= 1

    def test_out_of_order_arrival_is_late(self) -> None:
        tracker = BoundedDedup(clock=FakeClock())
        newer = dedup_key({"run_id": "r", "rank": 0, "seq": 10})
        older = dedup_key({"run_id": "r", "rank": 0, "seq": 5})

        assert tracker.check(newer) == "new"
        assert tracker.check(older) == "late"
        # A late key is remembered too, so its resend is a duplicate.
        assert tracker.check(older) == "duplicate"
        assert tracker.stats()["late"] == 1

    def test_ordering_is_tracked_per_rank(self) -> None:
        tracker = BoundedDedup(clock=FakeClock())
        rank0 = dedup_key({"run_id": "r", "rank": 0, "seq": 5})
        rank1 = dedup_key({"run_id": "r", "rank": 1, "seq": 5})

        assert tracker.check(rank0) == "new"
        assert tracker.check(rank1) == "new"
        assert tracker.check(dedup_key({"run_id": "r", "rank": 0, "seq": 3})) == "late"

    def test_topology_epoch_separates_the_ordering(self) -> None:
        tracker = BoundedDedup(clock=FakeClock())

        assert tracker.check(("r", "topo0", 0, 10)) == "new"
        assert tracker.check(("r", "topo1", 0, 5)) == "new"

    def test_cap_evicts_the_oldest_and_never_grows(self) -> None:
        tracker = BoundedDedup(max_entries=4, ttl_s=3600.0, clock=FakeClock())

        for seq in range(1, 11):
            assert tracker.check(dedup_key({"run_id": "r", "rank": 0, "seq": seq})) == "new"

        stats = tracker.stats()
        assert stats["entries"] == 4
        assert stats["evicted"] >= 6
        # The oldest key was evicted, so re-seeing it is a late sample, not a
        # duplicate.
        assert tracker.check(dedup_key({"run_id": "r", "rank": 0, "seq": 1})) == "late"
        assert tracker.stats()["entries"] <= 4

    def test_stats_are_json_serialisable(self) -> None:
        tracker = BoundedDedup(clock=FakeClock())
        tracker.check(dedup_key({"run_id": "r", "rank": 0, "seq": 1}))

        payload = json.loads(json.dumps(tracker.stats()))

        assert payload["max_entries"] == DEDUP_MAX_ENTRIES
        assert payload["ttl_s"] == DEDUP_TTL_S
        assert payload["entries"] == 1
        assert payload["newest_ranks"] == 1

    def test_junk_keys_never_raise(self) -> None:
        tracker = BoundedDedup()

        for junk in (None, (), "x", 3, {"a": 1}, ("r", "", 0)):
            assert tracker.check(junk) in ("new", "duplicate", "late")


def test_protocol_constants_document_the_wire() -> None:
    assert SCHEMA_VERSION == 2
    assert PROTOCOL_NAME
    assert MEASUREMENT_KINDS == ("device", "host_only", "unknown")
    assert WORKLOAD_FIELDS == ("tokens", "sequences", "microbatches")
    assert DEDUP_MAX_ENTRIES == 8192
    assert DEDUP_TTL_S == 120.0
