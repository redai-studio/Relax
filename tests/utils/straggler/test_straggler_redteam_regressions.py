# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Red-team regression tests for the Task 11 straggler profiler.

Each test here is the minimal counterexample for one confirmed defect from the
independent adversarial review (``RED_TEAM_REPORT.md``). They are deliberately
separate from the subsystem's own suites so the invariant each one pins is
unambiguous:

* RT-01 -- the host-only / pool-exhausted ingest path must not write files on
  the training thread;
* RT-02 -- concurrent ingest must persist every judged envelope exactly once
  (the old snapshot-and-clear flush duplicated or dropped lines);
* RT-03 -- the receiver's connection cap must count *live* connections, not
  every connection ever accepted;
* RT-04 -- a NaN/inf/negative timing sample must not become the peer reference
  and silence a whole window;
* RT-05 -- free-text fields must be length-capped so the dedup entry cap stays
  the binding memory constraint;
* RT-06 -- the shim's timer-name table must be bounded;
* RT-07 -- ranks with different expert-parallel roles must not share a cohort;
* RT-08 -- workload counters must survive the wire and reach the detector;
* RT-09 -- a rank starting three windows late must still be compared (fixed by
  the shared cohort anchor), while its own cold-start windows stay warmup;
* RT-10 -- concurrent ingest must conserve every tally exactly: an unsynchronised
  ``BoundedDedup`` expiry walk and a window close racing ``_Window.add`` both
  used to lose evidence silently.
"""

import socket
import sys
import threading
import time
from pathlib import Path
from typing import Any, List

import pytest

from relax.utils.straggler import collector as collector_module
from relax.utils.straggler import megatron_timer_shim as shim_module
from relax.utils.straggler import protocol
from relax.utils.straggler.collector import EnvelopeReceiver, TimingCollector
from relax.utils.straggler.config import StragglerConfig
from relax.utils.straggler.detector import VERDICT_STRAGGLER, StragglerDetector
from relax.utils.straggler.identity import RuntimeIdentity
from relax.utils.straggler.megatron_timer_shim import StragglerTimers
from relax.utils.straggler.observer import StragglerObserver, TimingEnvelope


IDENTITY = RuntimeIdentity(run_id="rt", rank=0, world_size=8, tensor_parallel_rank=0, data_parallel_rank=0)
COHORT = "topo0:dense:0:0:-1:-1:-1"


def envelope(
    rank: int,
    seq: int,
    host_ms: float = 100.0,
    cohort: str = COHORT,
    name: str = "forward-compute",
    start_s: float = 0.0,
) -> Any:
    """A locally built envelope with a chosen start time and host duration."""
    return TimingEnvelope(
        run_id="rt",
        rank=rank,
        cohort=cohort,
        label=f"rank{rank}",
        world_size=8,
        name=name,
        log_level=2,
        seq=seq,
        host_start=start_s,
        host_end=start_s + host_ms / 1000.0,
        device_ms=None,
        barrier=False,
        reason="no_event_pair",
    )


def make_collector(tmp_path: Path, **overrides: Any) -> TimingCollector:
    settings: dict = {
        "enabled": True,
        "window_seconds": 3600.0,
        "warmup_windows": 0,
        "persist_windows": 1,
    }
    settings.update(overrides)
    return TimingCollector(StragglerConfig(**settings), identity=IDENTITY)


class _NoDeviceBackend:
    """Event backend that reports no CUDA: every interval is host-only."""

    def available(self) -> bool:
        return False

    def create_event(self) -> Any:  # pragma: no cover - never called
        raise AssertionError("no event may be created without a device")

    def record(self, event: Any) -> None:  # pragma: no cover - never called
        raise AssertionError("no event may be recorded without a device")

    def is_complete(self, event: Any) -> bool:  # pragma: no cover - never called
        return True

    def elapsed_ms(self, start: Any, end: Any) -> float:  # pragma: no cover
        return 0.0


# --- RT-01: no file I/O on the training thread ---------------------------------


def test_host_only_ingest_never_writes_files_on_the_training_thread(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The no-device path delivers synchronously on the training thread."""
    monkeypatch.setattr(collector_module, "WRITE_BATCH", 4)
    collector = make_collector(tmp_path, output_dir=str(tmp_path))
    flushes: List[str] = []
    real_flush = collector._flush_path

    def recording_flush(path: str) -> None:
        flushes.append(threading.current_thread().name)
        real_flush(path)

    monkeypatch.setattr(collector, "_flush_path", recording_flush)

    observer = StragglerObserver(
        collector._config, identity=IDENTITY, backend=_NoDeviceBackend(), consumer=collector.ingest
    )
    timers = StragglerTimers(collector._config, sink=observer)
    for _ in range(10):
        handle = timers("forward-compute", 2)
        handle.start()
        handle.stop()
    observer.close()

    envelope_path = tmp_path / "straggler_envelopes.jsonl"
    assert flushes == [], f"training-thread file write: {flushes}"
    assert envelope_path.read_text(encoding="utf-8") == ""
    assert collector.status()["pending_lines"][str(envelope_path)] == 10


def test_off_thread_ingest_still_batches_and_flushes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The batch write still happens -- but only off the training thread."""
    monkeypatch.setattr(collector_module, "WRITE_BATCH", 4)
    collector = make_collector(tmp_path, output_dir=str(tmp_path))
    flushes: List[str] = []
    real_flush = collector._flush_path

    def recording_flush(path: str) -> None:
        flushes.append(threading.current_thread().name)
        real_flush(path)

    monkeypatch.setattr(collector, "_flush_path", recording_flush)

    def ingest_batch() -> None:
        for i in range(8):
            collector.ingest(envelope(0, i + 1))

    worker = threading.Thread(target=ingest_batch, name="rt-worker")
    worker.start()
    worker.join()

    assert flushes, "the batch bound never triggered a write"
    assert all(name != "MainThread" for name in flushes), flushes
    lines = (tmp_path / "straggler_envelopes.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 8
    assert collector.status()["pending_lines"].get(str(tmp_path / "straggler_envelopes.jsonl"), 0) == 0


# --- RT-02: concurrent persistence is exact ------------------------------------


def test_concurrent_ingest_persists_every_envelope_exactly_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(collector_module, "WRITE_BATCH", 16)
    collector = make_collector(tmp_path, output_dir=str(tmp_path))
    writer_threads = 8
    per_thread = 1500
    barrier = threading.Barrier(writer_threads)

    def worker(rank: int) -> None:
        barrier.wait()
        for i in range(per_thread):
            collector.ingest(envelope(rank, i + 1))

    threads = [threading.Thread(target=worker, args=(rank,), name=f"rt-r{rank}") for rank in range(writer_threads)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    collector.flush()
    collector.report()

    status = collector.status()
    expected = writer_threads * per_thread
    assert status["judged_packets"] == expected
    lines = (tmp_path / "straggler_envelopes.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == expected, f"persisted {len(lines)} lines for {expected} judged envelopes"
    assert status["flushed_lines"] == expected


# --- RT-03: the connection cap counts live connections --------------------------


def test_receiver_releases_finished_connections(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(collector_module, "MAX_CONNECTIONS", 4)
    collector = make_collector(Path("/tmp"))
    receiver = EnvelopeReceiver("127.0.0.1:0", lambda payload: collector.ingest(TimingEnvelope.from_dict(payload)))
    receiver.start()
    if not receiver.stats()["listening"]:
        pytest.skip("loopback TCP is unavailable in this environment")
    host, port = receiver.address
    total = 10
    try:
        for i in range(total):
            payload = envelope(1, i + 1).to_json() + "\n"
            connection = socket.create_connection((host, port), timeout=2.0)
            connection.sendall(payload.encode("utf-8"))
            connection.close()
            deadline = time.perf_counter() + 2.0
            while time.perf_counter() < deadline and receiver.stats()["lines"] < i + 1:
                time.sleep(0.005)
    finally:
        receiver.close()

    status = receiver.stats()
    assert status["lines"] == total
    assert status["accept_errors"] == 0


# --- RT-04: a bad numeric sample cannot silence a window ------------------------


@pytest.mark.parametrize("bad_host_ms", [float("nan"), float("inf"), -50.0])
def test_non_finite_or_negative_sample_cannot_suppress_a_window(bad_host_ms: float) -> None:
    detector = StragglerDetector(
        StragglerConfig(enabled=True, window_seconds=1.0, warmup_windows=0, persist_windows=1, work_tolerance=0.05)
    )
    verdicts: List[Any] = []
    # rank0 is the malformed/hostile sample; rank1 is normal; rank2 is 3x slow.
    verdicts += detector.observe(envelope(0, 1, host_ms=bad_host_ms))
    verdicts += detector.observe(envelope(1, 2, host_ms=100.0))
    verdicts += detector.observe(envelope(2, 3, host_ms=300.0))
    for rank in (0, 1, 2):
        verdicts += detector.observe(envelope(rank, 10 + rank, host_ms=100.0, start_s=2.0))
    verdicts += detector.flush()

    stragglers = [verdict for verdict in verdicts if verdict.kind == VERDICT_STRAGGLER]
    assert [verdict.rank for verdict in stragglers] == [2]
    assert detector.stats()["invalid_samples"] == 1


# --- RT-05: text fields are length-capped ---------------------------------------


def test_over_long_text_field_is_rejected_before_dedup() -> None:
    hostile = {
        "schema_version": protocol.SCHEMA_VERSION,
        "rank": 1,
        "name": "forward-compute",
        "run_id": "x" * (protocol.MAX_TEXT_CHARS + 1),
    }
    ok, reason = protocol.validate(hostile)
    assert ok is False
    assert reason == "text_field_too_long"

    collector = make_collector(Path("/tmp"))
    for i in range(50):
        packet = dict(hostile, run_id="x" * (protocol.MAX_TEXT_CHARS + 1) + str(i))
        collector.ingest(packet)
    status = collector.status()
    assert status["invalid_packets"] == 50
    assert status["dedup"]["entries"] == 0


# --- RT-06: the timer-name table is bounded -------------------------------------


def test_timer_name_table_is_bounded(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(shim_module, "MAX_TIMER_NAMES", 4)
    timers = StragglerTimers(StragglerConfig(enabled=False))

    for i in range(10):
        handle = timers(f"stage-{i}", log_level=2)
        handle.start()
        handle.stop()

    stats = timers.stats()
    assert stats["timer_names"] == 4
    assert stats["name_evictions"] == 6
    # The four retained names still work.
    assert timers("stage-0", log_level=2).name == "stage-0"


# --- RT-07: expert-parallel roles are not compared ------------------------------


def test_cohort_separates_expert_parallel_roles() -> None:
    base = dict(run_id="rt", world_size=8, tensor_parallel_rank=0, pipeline_parallel_rank=0, data_parallel_rank=0)
    dense = RuntimeIdentity(rank=0, expert_parallel_rank=0, expert_tensor_parallel_rank=0, **base)
    other_expert = RuntimeIdentity(rank=1, expert_parallel_rank=1, expert_tensor_parallel_rank=0, **base)
    other_shard = RuntimeIdentity(rank=2, expert_parallel_rank=0, expert_tensor_parallel_rank=1, **base)
    same_role_other_data = RuntimeIdentity(
        rank=3, expert_parallel_rank=0, expert_tensor_parallel_rank=0, expert_data_parallel_rank=1, **base
    )

    assert dense.cohort != other_expert.cohort
    assert dense.cohort != other_shard.cohort
    # Expert-data-parallel replicas run the same program and stay comparable.
    assert dense.cohort == same_role_other_data.cohort


def test_two_expert_roles_do_not_produce_a_false_accusation() -> None:
    """A 3x timing difference across EP roles must not be judged a
    straggler."""
    left = RuntimeIdentity(
        run_id="rt", rank=0, world_size=8, tensor_parallel_rank=0, expert_parallel_rank=0, data_parallel_rank=0
    )
    right = RuntimeIdentity(
        run_id="rt", rank=1, world_size=8, tensor_parallel_rank=0, expert_parallel_rank=1, data_parallel_rank=0
    )
    detector = StragglerDetector(
        StragglerConfig(enabled=True, window_seconds=1.0, warmup_windows=0, persist_windows=1, work_tolerance=0.05)
    )
    for index in range(2):
        detector.observe(envelope(0, index + 1, host_ms=100.0, cohort=left.cohort))
        detector.observe(envelope(1, index + 1, host_ms=300.0, cohort=right.cohort))
    verdicts = detector.flush()
    assert [v for v in verdicts if v.kind == VERDICT_STRAGGLER] == []


# --- RT-08: workload survives the wire ------------------------------------------


def test_workload_survives_the_wire_and_is_bounded() -> None:
    tagged = envelope(1, 1)
    object.__setattr__(tagged, "workload", {"tokens": 700, "sequences": 3})
    restored = TimingEnvelope.from_dict(tagged.to_dict())
    assert restored.workload == {"tokens": 700, "sequences": 3}

    hostile = TimingEnvelope.from_dict(
        {
            "schema_version": protocol.SCHEMA_VERSION,
            "rank": 1,
            "name": "forward-compute",
            "run_id": "rt",
            "host_start": 0.0,
            "host_end": 0.1,
            "workload": {"tokens": float("nan"), "junk": "x", "sequences": 5, "microbatches": float("inf")},
        }
    )
    assert hostile.workload == {"sequences": 5}


def test_workload_delta_is_reported_over_the_wire_envelope() -> None:
    detector = StragglerDetector(
        StragglerConfig(enabled=True, window_seconds=1.0, warmup_windows=0, persist_windows=1, work_tolerance=0.05)
    )
    verdicts: List[Any] = []
    for rank, (host_ms, tokens, start_s) in enumerate(((100.0, 10.0, 0.0), (100.0, 10.0, 0.0), (200.0, 20.0, 0.0))):
        tagged = envelope(rank, rank + 1, host_ms=host_ms, start_s=start_s)
        object.__setattr__(tagged, "workload", {"tokens": tokens})
        verdicts += detector.observe(TimingEnvelope.from_dict(tagged.to_dict()))
    for rank in range(3):
        tagged = envelope(rank, 10 + rank, host_ms=100.0, start_s=2.0)
        object.__setattr__(tagged, "workload", {"tokens": 10.0})
        verdicts += detector.observe(TimingEnvelope.from_dict(tagged.to_dict()))
    verdicts += detector.flush()

    straggler = next(v for v in verdicts if v.kind == VERDICT_STRAGGLER and v.rank == 2)
    assert straggler.facts["workload_delta"] == pytest.approx(1.0)
    assert straggler.facts["workload_delta_beyond_tolerance"] is True


# --- RT-09: a rank that starts >= 3 windows late is still compared --------------


def test_rank_starting_three_windows_late_is_still_compared() -> None:
    """The straight-line form of the original defect.

    rank 1 starts 15 s (three 5 s windows) after rank 0 and is 3x slow. Windows
    are anchored to the cohort, so both ranks land in the same wall-clock
    window once rank 1 is up; before the fix rank 0 closed every shared window
    before rank 1's first sample and the straggler was never named.
    """
    detector = StragglerDetector(
        StragglerConfig(enabled=True, window_seconds=5.0, warmup_windows=0, persist_windows=1, work_tolerance=0.05)
    )
    verdicts: List[Any] = []
    seq = {0: 0, 1: 0}
    t = 0.0
    while t < 60.0:
        seq[0] += 1
        verdicts += detector.observe(envelope(0, seq[0], host_ms=100.0, start_s=t))
        if t >= 15.0:
            seq[1] += 1
            verdicts += detector.observe(envelope(1, seq[1], host_ms=300.0, start_s=t))
        t += 0.5
    verdicts += detector.flush()
    assert [v.rank for v in verdicts if v.kind == VERDICT_STRAGGLER] == [1]


def test_late_rank_cold_start_is_warmup_not_a_straggler() -> None:
    """Per-rank warmup: a late rank's own first windows must not be a finding.

    A shared anchor alone would judge rank 1 from its very first window. rank 1
    starts 20 s in and is 4x slow only for its own first two windows, then
    matches rank 0 exactly: nothing may be reported, and the excluded samples
    must be visible in the counters.
    """
    detector = StragglerDetector(
        StragglerConfig(enabled=True, window_seconds=5.0, warmup_windows=2, persist_windows=1, work_tolerance=0.05)
    )
    verdicts: List[Any] = []
    seq = {0: 0, 1: 0}
    t = 0.0
    while t < 55.0:
        seq[0] += 1
        verdicts += detector.observe(envelope(0, seq[0], host_ms=100.0, start_s=t))
        if t >= 20.0:
            seq[1] += 1
            # Slow for its own first two windows (t in [20, 30)), then normal.
            host_ms = 400.0 if t < 30.0 else 100.0
            verdicts += detector.observe(envelope(1, seq[1], host_ms=host_ms, start_s=t))
        t += 0.5
    verdicts += detector.flush()

    assert [v.rank for v in verdicts if v.kind == VERDICT_STRAGGLER] == []
    stats = detector.stats()
    assert stats["warmup_samples_skipped"] > 0
    assert stats["warmup_windows_skipped"] >= 1


# --- RT-10: every shared tally is conserved under concurrent ingest -------------


def test_concurrent_ingest_conserves_every_tally(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Hammer the real ingest path from several threads and require exact
    tallies.

    The collector is fed by one thread per accepted connection plus the
    observer readout thread, and its state used to be lock-free. Two silent
    losses were reproducible under a 1 us interpreter switch interval:
    ``BoundedDedup`` raised ``OrderedDict mutated during iteration`` inside its
    expiry walk and swallowed it (13-15 keys per run were never recorded, so a
    resend of those keys could be judged twice), and a window close racing a
    concurrent ``_Window.add`` raised ``StatisticsError: no median for empty
    data`` and dropped that window's verdicts with only a log line. Neither
    showed up in the counters, so the test asserts conservation rather than
    "did not crash".
    """
    monkeypatch.setattr(collector_module, "WRITE_BATCH", 8)
    collector = make_collector(tmp_path, output_dir=str(tmp_path), window_seconds=0.1)
    writer_threads = 8
    per_thread = 250
    expected = writer_threads * per_thread
    errors: List[str] = []
    stop = threading.Event()

    def writer(rank: int) -> None:
        try:
            for seq in range(per_thread):
                # One window per sequence index, so windows keep closing while
                # other threads are still adding samples to them.
                collector.ingest(envelope(rank, seq + 1, host_ms=100.0, start_s=seq * 0.1 + 0.05))
        except BaseException as exc:  # noqa: BLE001
            errors.append(f"writer {type(exc).__name__}: {exc}")

    def reader() -> None:
        while not stop.is_set():
            try:
                collector.status()
            except BaseException as exc:  # noqa: BLE001
                errors.append(f"reader {type(exc).__name__}: {exc}")
                return

    previous_interval = sys.getswitchinterval()
    sys.setswitchinterval(1e-6)
    try:
        writers = [threading.Thread(target=writer, args=(rank,)) for rank in range(writer_threads)]
        observer = threading.Thread(target=reader)
        observer.start()
        for thread in writers:
            thread.start()
        for thread in writers:
            thread.join()
        stop.set()
        observer.join(timeout=5.0)
    finally:
        stop.set()
        sys.setswitchinterval(previous_interval)

    collector.flush()
    status = collector.status()
    assert errors == []
    assert status["envelopes"] == expected
    assert status["judged_packets"] == expected
    assert status["invalid_packets"] == 0
    assert status["duplicate_packets"] == 0
    assert status["late_packets"] == 0
    assert status["ingest_errors"] == 0
    assert status["dedup"]["new"] == expected
    assert status["dedup"]["entries"] == expected
    assert status["windows_closed"] > 0
    assert status["flushed_lines"] == expected
    lines = (tmp_path / "straggler_envelopes.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == expected
