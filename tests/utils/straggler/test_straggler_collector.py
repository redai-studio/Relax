# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Unit tests for the collector, its JSONL sinks and the same-host
transport."""

import json
import threading
import time
from itertools import count
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

from relax.utils.straggler import protocol
from relax.utils.straggler.collector import (
    DEDUP_MAX_ENTRIES,
    EnvelopeReceiver,
    EnvelopeSender,
    TimingCollector,
    parse_address,
)
from relax.utils.straggler.config import StragglerConfig
from relax.utils.straggler.detector import VERDICT_STRAGGLER
from relax.utils.straggler.identity import RuntimeIdentity
from relax.utils.straggler.observer import TimingEnvelope


#: A real observer stamps a distinct ``seq`` on every interval. The fixture has
#: to model that now that the collector dedups on ``(run, epoch, rank, seq)``;
#: otherwise every envelope of one rank would be a legitimate duplicate.
_SEQ = count(1)


def make_envelope(
    rank: int,
    host_ms: float,
    window_start_s: float = 0.0,
    device_ms: Optional[float] = None,
    name: str = "forward-compute",
    cohort: str = "0:0:0:0:0",
    world_size: int = 4,
    seq: Optional[int] = None,
) -> TimingEnvelope:
    """Build a realistic envelope whose host interval is ``host_ms`` long."""
    return TimingEnvelope(
        run_id="run-1",
        rank=rank,
        cohort=cohort,
        label=f"rank{rank}/tp0/pp0",
        world_size=world_size,
        name=name,
        log_level=2,
        seq=next(_SEQ) if seq is None else seq,
        host_start=window_start_s + 0.1,
        host_end=window_start_s + 0.1 + host_ms / 1000.0,
        device_ms=device_ms,
        barrier=False,
        reason="device",
    )


def make_collector(output_dir: Optional[str] = None, **overrides: Any) -> TimingCollector:
    settings: Dict[str, Any] = {
        "enabled": True,
        "window_seconds": 1.0,
        "warmup_windows": 0,
        "work_tolerance": 0.05,
        "persist_windows": 1,
        "min_cohort_size": 2,
        "report_interval_seconds": 3600.0,
    }
    settings.update(overrides)
    settings["output_dir"] = output_dir
    return TimingCollector(
        StragglerConfig(**settings),
        identity=RuntimeIdentity(run_id="run-1", rank=0, world_size=4),
    )


class TestAddressParsing:
    def test_parses_host_and_port(self) -> None:
        assert parse_address("127.0.0.1:39999") == ("127.0.0.1", 39999)

    @pytest.mark.parametrize("address", ["127.0.0.1", "localhost:", ":1234", "host:port", ""])
    def test_rejects_malformed_address(self, address: str) -> None:
        with pytest.raises(ValueError):
            parse_address(address)


class TestTimingCollector:
    def test_ingest_produces_verdicts(self) -> None:
        collector = make_collector()

        verdicts: List[Any] = []
        for window in range(3):
            for rank, host_ms in ((0, 100.0), (1, 100.0), (2, 300.0)):
                verdicts.extend(collector.ingest(make_envelope(rank, host_ms, window_start_s=window)))

        assert [verdict for verdict in verdicts if verdict.kind == VERDICT_STRAGGLER]
        assert collector.status()["envelopes"] == 9

    def test_verdict_callback_receives_every_verdict(self) -> None:
        seen: List[Any] = []
        config = StragglerConfig(enabled=True, window_seconds=1.0, warmup_windows=0, persist_windows=1)
        collector = TimingCollector(config, identity=None, on_verdict=seen.append)

        for window in range(3):
            for rank, host_ms in ((0, 100.0), (1, 100.0), (2, 300.0)):
                collector.ingest(make_envelope(rank, host_ms, window_start_s=window))

        assert seen
        assert collector.status()["verdicts"] == len(seen)

    def test_failing_verdict_callback_is_counted(self) -> None:
        def explode(verdict: Any) -> None:
            raise RuntimeError("callback exploded")

        config = StragglerConfig(enabled=True, window_seconds=1.0, warmup_windows=0, persist_windows=1)
        collector = TimingCollector(config, identity=None, on_verdict=explode)

        for window in range(3):
            for rank, host_ms in ((0, 100.0), (1, 100.0), (2, 300.0)):
                collector.ingest(make_envelope(rank, host_ms, window_start_s=window))

        assert collector.status()["verdict_callback_errors"] > 0

    def test_jsonl_sinks_are_written(self, tmp_path: Path) -> None:
        collector = make_collector(output_dir=str(tmp_path))

        for window in range(3):
            for rank, host_ms in ((0, 100.0), (1, 100.0), (2, 300.0)):
                collector.ingest(make_envelope(rank, host_ms, window_start_s=window))
        collector.flush()

        envelopes = (tmp_path / "straggler_envelopes.jsonl").read_text(encoding="utf-8").strip().splitlines()
        verdicts = (tmp_path / "straggler_verdicts.jsonl").read_text(encoding="utf-8").strip().splitlines()
        assert len(envelopes) == 9
        assert verdicts
        assert json.loads(envelopes[0])["rank"] == 0
        assert json.loads(verdicts[0])["kind"] == VERDICT_STRAGGLER

    def test_unwritable_output_dir_disables_persistence_only(self, tmp_path: Path) -> None:
        blocker = tmp_path / "blocker"
        blocker.write_text("not a directory", encoding="utf-8")
        collector = make_collector(output_dir=str(blocker / "nested"))

        verdicts = collector.ingest(make_envelope(0, 100.0))

        assert verdicts == []
        assert collector.status()["write_errors"] >= 1
        assert collector.status()["envelopes"] == 1

    def test_periodic_report_emits_a_summary(self, caplog: pytest.LogCaptureFixture) -> None:
        collector = make_collector(report_interval_seconds=0.0)

        with caplog.at_level("INFO"):
            collector.ingest(make_envelope(0, 100.0))

        assert "straggler[" in caplog.text
        assert collector.status()["reports"] >= 1

    def test_status_is_json_serialisable(self) -> None:
        collector = make_collector()
        collector.ingest(make_envelope(0, 100.0))

        payload = json.loads(json.dumps(collector.status()))

        assert payload["identity"] == RuntimeIdentity(run_id="run-1", rank=0, world_size=4).label
        assert payload["active_stragglers"] == []

    def test_flush_closes_open_windows(self) -> None:
        collector = make_collector()
        for rank, host_ms in ((0, 100.0), (1, 100.0), (2, 300.0)):
            collector.ingest(make_envelope(rank, host_ms))

        collector.flush()

        assert collector.status()["open_windows"] == 0

    def test_drain_verdicts_is_destructive(self) -> None:
        collector = make_collector()
        for window in range(3):
            for rank, host_ms in ((0, 100.0), (1, 100.0), (2, 300.0)):
                collector.ingest(make_envelope(rank, host_ms, window_start_s=window))

        assert collector.drain_verdicts()
        assert collector.drain_verdicts() == []

    def test_ingest_accepts_a_rebuilt_envelope(self) -> None:
        collector = make_collector()
        original = make_envelope(2, 300.0)

        rebuilt = TimingEnvelope.from_dict(json.loads(original.to_json()))
        collector.ingest(rebuilt)

        assert collector.status()["envelopes"] == 1
        assert rebuilt.host_ms == pytest.approx(original.host_ms)


class TestIngestProtocolGates:
    """Validation and idempotency run before the detector, and never raise."""

    def test_packet_without_schema_version_is_invalid(self) -> None:
        collector = make_collector()

        verdicts = collector.ingest({"rank": 0, "name": "forward-compute"})

        assert verdicts == []
        assert collector.status()["invalid_packets"] == 1
        assert collector.status()["judged_packets"] == 0

    def test_invalid_packet_is_counted_not_judged_or_persisted(self, tmp_path: Path) -> None:
        collector = make_collector(output_dir=str(tmp_path))

        verdicts = collector.ingest({"schema_version": protocol.SCHEMA_VERSION, "rank": -1, "name": "forward-compute"})
        collector.flush()

        status = collector.status()
        assert verdicts == []
        assert status["invalid_packets"] == 1
        assert status["judged_packets"] == 0
        assert (tmp_path / "straggler_envelopes.jsonl").read_text(encoding="utf-8") == ""

    def test_ingest_never_raises_on_junk(self) -> None:
        collector = make_collector()

        for junk in (None, 42, "not-an-envelope", [], object()):
            assert collector.ingest(junk) == []

        assert collector.status()["envelopes"] == 5
        assert collector.status()["invalid_packets"] == 5

    def test_duplicate_packet_is_judged_once(self) -> None:
        collector = make_collector()
        packet = make_envelope(2, 300.0, seq=41)

        first = collector.ingest(packet)
        second = collector.ingest(packet)

        status = collector.status()
        assert first == [] and second == []
        assert status["envelopes"] == 2
        assert status["judged_packets"] == 1
        assert status["duplicate_packets"] == 1
        assert status["dedup"]["duplicate"] == 1

    def test_two_host_only_intervals_from_the_real_observer_are_both_judged(self) -> None:
        """Regression for the seq=0 host-only defect, end to end.

        The observer used to give every host-only interval ``seq=0``; under the
        protocol's dedup key the collector then counted the second (and every
        later) host-only packet as a duplicate and never judged it. Two real
        host-only intervals must now both reach the detector.
        """
        from tests.utils.straggler.test_straggler_observer import FakeEventBackend, make_observer

        observed: List[TimingEnvelope] = []
        observer = make_observer(FakeEventBackend(available=False), consumer=observed.append)
        observer.complete_interval(None, "forward-compute", 2, 1.0, 1.25, False)
        observer.complete_interval(None, "forward-compute", 2, 2.0, 2.25, False)
        assert len(observed) == 2
        assert observed[0].seq != observed[1].seq

        collector = make_collector()
        collector.ingest(observed[0])
        collector.ingest(observed[1])

        status = collector.status()
        assert status["envelopes"] == 2
        assert status["judged_packets"] == 2
        assert status["duplicate_packets"] == 0
        assert status["dedup"]["duplicate"] == 0

    def test_out_of_order_packet_is_counted_late(self) -> None:
        collector = make_collector()

        collector.ingest(make_envelope(0, 100.0, seq=10))
        verdicts = collector.ingest(make_envelope(0, 100.0, seq=5))

        status = collector.status()
        assert verdicts == []
        assert status["judged_packets"] == 1
        assert status["late_packets"] == 1
        assert status["dedup"]["late"] == 1

    def test_duplicate_and_late_counters_survive_a_summary(self) -> None:
        collector = make_collector(report_interval_seconds=0.0)
        collector.ingest(make_envelope(0, 100.0, seq=10))
        collector.ingest(make_envelope(0, 100.0, seq=10))
        collector.ingest(make_envelope(0, 100.0, seq=5))

        status = collector.report()

        assert status["duplicate_packets"] == 1
        assert status["late_packets"] == 1
        assert status["reports"] >= 1

    def test_status_reports_dedup_state_and_is_json_serialisable(self) -> None:
        collector = make_collector()
        collector.ingest(make_envelope(0, 100.0, seq=1))

        payload = json.loads(json.dumps(collector.status()))

        assert payload["dedup"]["max_entries"] == DEDUP_MAX_ENTRIES
        assert payload["dedup"]["entries"] == 1
        assert payload["dedup"]["new"] == 1


class TestTransport:
    def test_sender_stamps_the_schema_version(self) -> None:
        payload = json.loads(EnvelopeSender._encode(make_envelope(0, 100.0)))

        assert payload["schema_version"] == protocol.SCHEMA_VERSION
        assert payload["protocol"] == protocol.PROTOCOL_NAME
        assert protocol.validate(payload) == (True, "")

    def test_round_trip_over_loopback(self) -> None:
        received: List[Dict[str, Any]] = []
        receiver = EnvelopeReceiver("127.0.0.1:0", received.append)
        receiver.start()
        if not receiver.stats()["listening"]:
            pytest.skip("loopback TCP is unavailable in this environment")
        host, port = receiver.address
        sender = EnvelopeSender(f"{host}:{port}", queue_max=64)
        try:
            for rank in range(3):
                sender.send(make_envelope(rank, 100.0 + rank))
            assert _wait_until(lambda: len(received) == 3)
        finally:
            sender.close()
            receiver.close()

        assert {payload["rank"] for payload in received} == {0, 1, 2}
        assert sender.stats()["sent"] == 3
        assert sender.stats()["send_errors"] == 0

    def test_full_sender_queue_drops_and_counts(self) -> None:
        sender = EnvelopeSender("127.0.0.1:1", queue_max=2, reconnect_interval_s=0.2)
        try:
            for rank in range(10):
                sender.send(make_envelope(rank, 100.0))
            stats = sender.stats()
        finally:
            sender.close()

        assert stats["dropped_queue_full"] > 0
        assert stats["queued"] + stats["dropped_queue_full"] == 10

    def test_sender_survives_an_unserialisable_envelope(self) -> None:
        sender = EnvelopeSender("127.0.0.1:1", queue_max=2, reconnect_interval_s=0.2)

        class Broken:
            def to_json(self) -> str:
                raise RuntimeError("boom")

        try:
            sender.send(Broken())
            stats = sender.stats()
        finally:
            sender.close()

        assert stats["send_errors"] == 1
        assert stats["queued"] == 0

    def test_receiver_counts_malformed_lines(self) -> None:
        received: List[Any] = []
        receiver = EnvelopeReceiver("127.0.0.1:0", received.append)

        receiver._handle_line(b"{not json}\n")
        receiver._handle_line(b'{"rank": 1}\n')

        assert receiver.stats()["parse_errors"] == 1
        assert receiver.stats()["lines"] == 1

    def test_receiver_counts_invalid_packets_without_raising(self) -> None:
        receiver = EnvelopeReceiver("127.0.0.1:0", lambda payload: None)

        receiver._handle_line(b'{"schema_version": 2, "rank": -1, "name": "forward-compute"}\n')
        receiver._handle_line(b'{"schema_version": 2, "rank": 0, "name": ""}\n')
        receiver._handle_line(b'{"schema_version": 2, "rank": 0, "name": "ok", "measurement_kind": "bogus"}\n')

        stats = receiver.stats()
        assert stats["invalid_packets"] == 3
        assert stats["parse_errors"] == 0
        assert stats["lines"] == 3

    def test_receiver_forwards_a_versioned_packet(self) -> None:
        received: List[Any] = []
        receiver = EnvelopeReceiver("127.0.0.1:0", received.append)

        receiver._handle_line(b'{"schema_version": 2, "rank": 3, "name": "forward-compute"}\n')

        assert receiver.stats()["invalid_packets"] == 0
        assert received and received[0]["rank"] == 3

    def test_receiver_counts_callback_failures(self) -> None:
        def explode(payload: Any) -> None:
            raise RuntimeError("boom")

        receiver = EnvelopeReceiver("127.0.0.1:0", explode)

        receiver._handle_line(b'{"rank": 1}\n')

        assert receiver.stats()["callback_errors"] == 1

    def test_receiver_close_is_idempotent(self) -> None:
        receiver = EnvelopeReceiver("127.0.0.1:0", lambda payload: None)

        receiver.start()
        receiver.close()
        receiver.close()

        assert receiver.stats()["listening"] is False

    def test_receiver_reports_unavailable_bind(self) -> None:
        # TEST-NET-3 is never assigned to a local interface, so the bind fails
        # for every user; a privileged port would succeed when running as root.
        receiver = EnvelopeReceiver("203.0.113.1:0", lambda payload: None)

        receiver.start()

        assert receiver.stats()["listening"] is False
        assert receiver.stats()["accept_errors"] >= 1
        receiver.close()


def _wait_until(predicate: Any, timeout: float = 3.0, interval: float = 0.01) -> bool:
    """Poll ``predicate`` until it is true or ``timeout`` elapses."""
    deadline = time.perf_counter() + timeout
    while time.perf_counter() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return predicate()


class TestBufferedPersistence:
    """The consumer can run on the training thread, so writes are batched."""

    def test_lines_stay_buffered_until_flush(self, tmp_path: Path) -> None:
        collector = make_collector(output_dir=str(tmp_path))

        collector.ingest(make_envelope(0, 100.0))

        envelope_file = tmp_path / "straggler_envelopes.jsonl"
        assert envelope_file.read_text(encoding="utf-8") == ""
        assert collector.status()["pending_lines"][str(envelope_file)] == 1

        collector.flush()

        assert envelope_file.read_text(encoding="utf-8").strip()
        assert collector.status()["pending_lines"][str(envelope_file)] == 0

    def test_report_flushes_buffered_lines(self, tmp_path: Path) -> None:
        collector = make_collector(output_dir=str(tmp_path))

        collector.ingest(make_envelope(0, 100.0))
        collector.report()

        assert (tmp_path / "straggler_envelopes.jsonl").read_text(encoding="utf-8").strip()

    def test_batch_bound_triggers_a_write(self, tmp_path: Path) -> None:
        from relax.utils.straggler.collector import WRITE_BATCH

        collector = make_collector(output_dir=str(tmp_path))
        # Ingest from a non-training thread: the batch bound flushes only off
        # the training thread (see the red-team regression that a host-only
        # ingest path must not write files). The batch itself is unchanged.
        worker = threading.Thread(
            target=lambda: [collector.ingest(make_envelope(0, 100.0)) for _ in range(WRITE_BATCH)]
        )
        worker.start()
        worker.join()

        lines = (tmp_path / "straggler_envelopes.jsonl").read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == WRITE_BATCH
        assert collector.status()["flushed_lines"] >= WRITE_BATCH

    def test_summary_does_not_flush_but_report_does(self, tmp_path: Path) -> None:
        collector = make_collector(output_dir=str(tmp_path))
        envelope_file = tmp_path / "straggler_envelopes.jsonl"

        collector.ingest(make_envelope(0, 100.0))
        assert envelope_file.read_text(encoding="utf-8") == ""

        collector.summary()
        assert envelope_file.read_text(encoding="utf-8") == ""
        assert collector.status()["pending_lines"][str(envelope_file)] == 1

        collector.report()
        assert envelope_file.read_text(encoding="utf-8").strip()
