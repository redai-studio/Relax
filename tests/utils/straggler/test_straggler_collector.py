# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Unit tests for the collector, its JSONL sinks and the same-host
transport."""

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

from relax.utils.straggler.collector import (
    EnvelopeReceiver,
    EnvelopeSender,
    TimingCollector,
    parse_address,
)
from relax.utils.straggler.config import StragglerConfig
from relax.utils.straggler.detector import VERDICT_STRAGGLER
from relax.utils.straggler.identity import RuntimeIdentity
from relax.utils.straggler.observer import TimingEnvelope


def make_envelope(
    rank: int,
    host_ms: float,
    window_start_s: float = 0.0,
    device_ms: Optional[float] = None,
    name: str = "forward-compute",
    cohort: str = "0:0:0:0:0",
    world_size: int = 4,
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
        seq=rank,
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
        config = StragglerConfig(enabled=True, window_seconds=1.0, persist_windows=1)
        collector = TimingCollector(config, identity=None, on_verdict=seen.append)

        for window in range(3):
            for rank, host_ms in ((0, 100.0), (1, 100.0), (2, 300.0)):
                collector.ingest(make_envelope(rank, host_ms, window_start_s=window))

        assert seen
        assert collector.status()["verdicts"] == len(seen)

    def test_failing_verdict_callback_is_counted(self) -> None:
        def explode(verdict: Any) -> None:
            raise RuntimeError("callback exploded")

        config = StragglerConfig(enabled=True, window_seconds=1.0, persist_windows=1)
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


class TestTransport:
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
