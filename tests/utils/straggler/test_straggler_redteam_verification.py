# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Independent adversarial verification of the straggler durability work.

This file is written by the red-team reviewer, not by the subsystem author. It
does not import helpers from the author's tests and it never mutates production
state outside the test: every claim is reproduced from scratch so a summary
cannot be trusted by accident.

Coverage:

* durability under abrupt death (SIGTERM/SIGKILL) -- at most one flush interval
  may be missing, never the whole run;
* per-rank status paths unique under four simulated ranks, with exactly one
  writer for the unsuffixed names ``analyze_run.py`` reads;
* status snapshots are atomic: a concurrent reader never observes partial JSON,
  even when the writer is SIGKILLed mid-cycle;
* output paths are run-scoped;
* training-thread purity: the host-only and event-pool-exhausted ingest paths
  perform zero file writes on the training thread;
* conservation: accepted / duplicate / late / malformed / dedup-eviction /
  closed-window tallies are exactly conserved after concurrent ingest;
* regression re-verification of RT-09 (staggered start) and the A7 pickle walker.
"""

import copy
import dataclasses
import functools
import json
import os
import pathlib
import signal
import subprocess
import sys
import textwrap
import threading
import time
import types
from typing import Any, Dict, List, Optional

import pytest

from relax.utils.straggler import collector as collector_module
from relax.utils.straggler.collector import TimingCollector
from relax.utils.straggler.config import StragglerConfig
from relax.utils.straggler.detector import VERDICT_STRAGGLER, StragglerDetector
from relax.utils.straggler.identity import RuntimeIdentity
from relax.utils.straggler.observer import StragglerObserver, TimingEnvelope
from relax.utils.straggler.runtime import StragglerRuntime, _sanitise


WORKTREE = pathlib.Path(__file__).resolve().parents[3]
COHORT = "topo0:dense:0:0:-1:-1:-1"
IDENTITY = RuntimeIdentity(run_id="verify", rank=0, world_size=8, tensor_parallel_rank=0, data_parallel_rank=0)

#: The collector flushes its write batch from a non-training thread once this
#: many lines are buffered; the durability window is therefore bounded by both
#: this and the status-writer interval.
WRITE_BATCH = collector_module.WRITE_BATCH
#: Child cadence and periodic-flush cadence used by the kill experiments. One
#: flush interval therefore covers at most ``FLUSH_INTERVAL / CADENCE`` packets.
CADENCE_S = 0.01
FLUSH_INTERVAL_S = 0.1
PER_INTERVAL = int(FLUSH_INTERVAL_S / CADENCE_S) + 1


def _envelope(
    rank: int,
    seq: int,
    host_ms: float = 100.0,
    start_s: float = 0.0,
    cohort: str = COHORT,
    name: str = "forward-compute",
    device_ms: Optional[float] = None,
) -> TimingEnvelope:
    """Build a local envelope without going through the observer."""
    return TimingEnvelope(
        run_id="verify",
        rank=rank,
        cohort=cohort,
        label=f"rank{rank}",
        world_size=8,
        name=name,
        log_level=2,
        seq=seq,
        host_start=start_s,
        host_end=start_s + host_ms / 1000.0,
        device_ms=device_ms,
        barrier=False,
        reason="verification",
    )


def _runtime(tmp_path: pathlib.Path, run_id: str = "verify", rank: int = 0, **overrides: Any) -> StragglerRuntime:
    settings: Dict[str, Any] = {
        "enabled": True,
        "output_dir": str(tmp_path),
        "window_seconds": 3600.0,
        "warmup_windows": 0,
        "persist_windows": 1,
        "report_interval_seconds": 10.0,  # silence the periodic writer in-process
    }
    settings.update(overrides)
    identity = RuntimeIdentity(run_id=run_id, rank=rank, world_size=4, data_parallel_rank=rank)
    return StragglerRuntime(StragglerConfig(**settings), identity=identity, register_atexit=False)


class _NoDeviceBackend:
    """Event backend that reports no CUDA: every interval is host-only."""

    def available(self) -> bool:
        return False


class _SmallEvent:
    def __init__(self) -> None:
        self.completed = False


class _SmallPoolBackend:
    """Event backend that is available but never completes, so the pool
    drains."""

    def available(self) -> bool:
        return True

    def create_event(self) -> _SmallEvent:
        return _SmallEvent()

    def record(self, event: _SmallEvent) -> None:
        return None

    def is_complete(self, event: _SmallEvent) -> bool:
        return False

    def elapsed_ms(self, start: _SmallEvent, end: _SmallEvent) -> float:
        return 1.0


# --- durability under abrupt death ---------------------------------------------


_KILL_CHILD = textwrap.dedent(
    """
    import sys, time
    from relax.utils.straggler.config import StragglerConfig
    from relax.utils.straggler.identity import RuntimeIdentity
    from relax.utils.straggler.runtime import StragglerRuntime

    cfg = StragglerConfig(
        enabled=True,
        output_dir=sys.argv[1],
        window_seconds=1.0,
        warmup_windows=0,
        persist_windows=1,
        report_interval_seconds=0.1,
    )
    ident = RuntimeIdentity(run_id="kill-run", rank=0, world_size=1, data_parallel_rank=0)
    runtime = StragglerRuntime(cfg, identity=ident, register_atexit=False).start()
    for n in range(1, 401):
        handle = runtime.timers("forward-compute", log_level=2)
        handle.start()
        time.sleep(0.01)
        handle.stop()
        print(f"COUNT {n}", flush=True)
    time.sleep(60)
    """
)


def _child_env() -> Dict[str, str]:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(WORKTREE) + os.pathsep + env.get("PYTHONPATH", "")
    return env


def _run_until_delivered(tmp_path: pathlib.Path, delivery: int, sig: int) -> int:
    process = subprocess.Popen(
        [sys.executable, "-c", _KILL_CHILD, str(tmp_path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=_child_env(),
    )
    delivered = 0
    try:
        assert process.stdout is not None
        deadline = time.monotonic() + 30.0
        while time.monotonic() < deadline:
            line = process.stdout.readline()
            if not line:
                break
            if line.startswith("COUNT "):
                delivered = int(line.split()[1])
                if delivered >= delivery:
                    break
        process.send_signal(sig)
        process.wait(timeout=15)
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=15)
    return delivered


def _persisted(run_id: str, tmp_path: pathlib.Path) -> List[Any]:
    path = tmp_path / f"run_{_sanitise(run_id)}" / "straggler_envelopes.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


@pytest.mark.parametrize("sig", [signal.SIGTERM, signal.SIGKILL])
def test_abrupt_death_loses_at_most_one_flush_interval(tmp_path: pathlib.Path, sig: int) -> None:
    """Ray SIGTERMs actors; the whole run's JSONL must not be lost.

    The child delivers one interval every ``CADENCE_S`` and never calls
    ``close()``. It is killed as soon as the parent sees a fresh delivery, so
    whatever is missing can only be what one periodic flush had not yet
    written.
    """
    delivered = _run_until_delivered(tmp_path, delivery=120, sig=sig)
    assert delivered >= 120, f"child only delivered {delivered} intervals"
    lines = _persisted("kill-run", tmp_path)
    assert lines, "an abrupt kill lost the entire run's evidence"
    missing = delivered - len(lines)
    assert missing <= PER_INTERVAL + 3, (
        f"lost {missing} of {delivered} intervals on signal {sig} (one flush interval covers ~{PER_INTERVAL})"
    )
    # Every persisted envelope is complete and well formed.
    assert all("rank" in entry and "host_start" in entry for entry in lines)


def test_kill_during_a_status_write_leaves_only_valid_json(tmp_path: pathlib.Path) -> None:
    """``_write_json`` must be rename-atomic: a kill mid-write cannot corrupt
    it."""
    child = textwrap.dedent(
        """
        import sys, time
        from relax.utils.straggler.config import StragglerConfig
        from relax.utils.straggler.identity import RuntimeIdentity
        from relax.utils.straggler.runtime import StragglerRuntime

        cfg = StragglerConfig(enabled=True, output_dir=sys.argv[1], window_seconds=1.0)
        ident = RuntimeIdentity(run_id="atomic-run", rank=0, world_size=1, data_parallel_rank=0)
        runtime = StragglerRuntime(cfg, identity=ident, register_atexit=False).start()
        print("READY", flush=True)
        while True:
            runtime._write_status()  # noqa: SLF001 - deliberate stress of the writer
        """
    )
    process = subprocess.Popen(
        [sys.executable, "-c", child, str(tmp_path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=_child_env(),
    )
    try:
        assert process.stdout is not None
        assert process.stdout.readline().strip() == "READY"
        time.sleep(0.7)  # let several full write cycles land
        process.kill()
        process.wait(timeout=15)
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=15)

    run_dir = tmp_path / "run_atomic-run"
    finals = sorted(run_dir.glob("*.json"))
    assert finals, "no completed status snapshot survived the kill"
    for path in finals:
        text = path.read_text(encoding="utf-8")
        assert text.strip(), f"{path.name} is empty"
        json.loads(text)  # raises if a partial file was ever published
    assert (run_dir / "runtime_status.json").exists()
    assert (run_dir / "collector_status.json").exists()


def test_four_ranks_write_unique_status_paths_and_one_unsuffixed_writer(tmp_path: pathlib.Path) -> None:
    """Four simulated ranks must not overwrite each other's status file."""
    runtimes = [_runtime(tmp_path, run_id="four", rank=rank) for rank in range(4)]
    for runtime in runtimes:
        runtime.start()
        runtime._write_status()  # noqa: SLF001 - deterministic snapshot
        runtime.close()

    run_dir = tmp_path / "run_four"
    per_rank = sorted(path.name for path in run_dir.glob("runtime_status_*.json"))
    assert len(per_rank) == 4, per_rank
    assert len(set(per_rank)) == 4
    # Exactly one writer for the unsuffixed names analyze_run.py reads (rank 0).
    assert (run_dir / "runtime_status.json").exists()
    assert (run_dir / "collector_status.json").exists()
    for path in run_dir.glob("*.json"):
        json.loads(path.read_text(encoding="utf-8"))


def test_a_concurrent_reader_never_observes_partial_status_json(tmp_path: pathlib.Path) -> None:
    """Poll the published file while the writer loops: every read must
    parse."""
    runtime = _runtime(tmp_path, run_id="atomic2")
    runtime.start()
    run_dir = tmp_path / "run_atomic2"
    final = run_dir / "runtime_status.json"
    failures: List[str] = []
    stop = threading.Event()
    reads = 0

    def reader() -> None:
        nonlocal reads
        while not stop.is_set():
            if final.exists():
                try:
                    text = final.read_text(encoding="utf-8")
                    json.loads(text)
                    reads += 1
                except Exception as exc:  # noqa: BLE001
                    failures.append(f"{type(exc).__name__}: {exc}")

    worker = threading.Thread(target=reader, name="verify-status-reader")
    worker.start()
    try:
        for _ in range(120):
            runtime._write_status()  # noqa: SLF001
    finally:
        stop.set()
        worker.join(timeout=5.0)
        runtime.close()

    assert reads > 20, f"only {reads} successful reads; test did not exercise the writer"
    assert failures == []


def test_close_does_not_race_the_status_writer(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """close() must not overlap the periodic status writer on the same
    files."""
    state = {"active": 0, "max": 0}
    guard = threading.Lock()
    original = StragglerRuntime._write_status

    def slow_write_status(self: StragglerRuntime) -> None:
        with guard:
            state["active"] += 1
            state["max"] = max(state["max"], state["active"])
        try:
            time.sleep(1.5)  # a flush + serialisation slower than close()'s 1.0 s join
            original(self)
        finally:
            with guard:
                state["active"] -= 1

    monkeypatch.setattr(StragglerRuntime, "_write_status", slow_write_status)
    runtime = _runtime(tmp_path, run_id="close-race", report_interval_seconds=0.1)
    runtime.start()
    time.sleep(0.3)  # let the writer enter its slow iteration
    runtime.close()
    assert state["max"] == 1, f"close() overlapped the status writer: {state['max']} writers at once"


def test_write_json_is_not_safe_for_two_concurrent_writers(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The published snapshot must stay valid even with two concurrent
    writers."""
    original_dumps = json.dumps
    barrier = threading.Barrier(2, timeout=5.0)

    def half_step_dump(obj: Any, fp: Any, **kwargs: Any) -> None:
        text = original_dumps(obj, **kwargs)
        fp.write(text[: len(text) // 2])
        fp.flush()
        try:
            barrier.wait()
        except threading.BrokenBarrierError:
            pass
        fp.write(text[len(text) // 2 :])

    monkeypatch.setattr(json, "dump", half_step_dump)
    payloads = (
        {"a": list(range(50)), "pad": "x" * 800},
        {"b": list(range(70)), "pad": "y" * 1200},
    )
    errors: List[str] = []

    def writer(payload: Dict[str, Any]) -> None:
        try:
            StragglerRuntime._write_json(str(tmp_path), "snapshot.json", payload)  # noqa: SLF001
        except BaseException as exc:  # noqa: BLE001
            errors.append(f"{type(exc).__name__}: {exc}")

    threads = [threading.Thread(target=writer, args=(payload,)) for payload in payloads]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10.0)

    published = (tmp_path / "snapshot.json").read_text(encoding="utf-8")
    json.loads(published)  # the unique temp path keeps the published snapshot valid


def test_output_paths_are_run_scoped(tmp_path: pathlib.Path) -> None:
    """Two run ids must not interleave their evidence."""
    for run_id in ("run-alpha", "run-beta"):
        runtime = _runtime(tmp_path, run_id=run_id)
        runtime.start()
        runtime.collector.ingest(_envelope(0, 1))  # type: ignore[union-attr]
        runtime._write_status()  # noqa: SLF001
        runtime.close()

    alpha = tmp_path / "run_run-alpha"
    beta = tmp_path / "run_run-beta"
    assert alpha.is_dir() and beta.is_dir()
    assert (alpha / "straggler_envelopes.jsonl").exists()
    assert (beta / "straggler_envelopes.jsonl").exists()
    # Nothing is written directly in the shared base directory.
    assert not (tmp_path / "straggler_envelopes.jsonl").exists()


def test_run_scoping_is_not_injective_for_punctuation() -> None:
    """Documented invariant check: punctuation-distinct run ids collapse.

    ``_run_dir`` promises two runs "can never interleave", but ``_sanitise``
    maps ``job/a`` and ``job_a`` to the same component, so those two run ids
    share one directory. Pinned here as a known limit of the scoping claim.
    """
    assert _sanitise("job/a") == _sanitise("job_a") == "job_a"


# --- training-thread purity -----------------------------------------------------


def _collector(tmp_path: pathlib.Path, **overrides: Any) -> TimingCollector:
    settings: Dict[str, Any] = {
        "enabled": True,
        "output_dir": str(tmp_path),
        "window_seconds": 3600.0,
        "warmup_windows": 0,
        "persist_windows": 1,
    }
    settings.update(overrides)
    return TimingCollector(StragglerConfig(**settings), identity=IDENTITY)


def _record_writes(monkeypatch: pytest.MonkeyPatch) -> List[str]:
    writes: List[str] = []
    original = TimingCollector._write_batch

    def spy(self: TimingCollector, path: str, lines: List[str]) -> None:
        writes.append(threading.current_thread().name)
        original(self, path, lines)

    monkeypatch.setattr(TimingCollector, "_write_batch", spy)
    return writes


def test_host_only_ingest_never_writes_on_the_training_thread(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The RT-01 branch: no CUDA, so every interval is judged on the caller."""
    config = StragglerConfig(enabled=True, output_dir=str(tmp_path), window_seconds=3600.0)
    collector = _collector(tmp_path)
    writes = _record_writes(monkeypatch)
    observer = StragglerObserver(config, identity=IDENTITY, backend=_NoDeviceBackend(), consumer=collector.ingest)

    for i in range(WRITE_BATCH + 20):
        token = observer.acquire_interval("forward-compute", 2)
        assert token is None, "the no-device branch must return a host-only token"
        observer.complete_interval(token, "forward-compute", 2, 100.0 + i, 100.1 + i, False)

    assert writes == [], f"training thread wrote files: {writes}"
    assert (tmp_path / "straggler_envelopes.jsonl").read_text(encoding="utf-8") == ""

    off = threading.Thread(target=collector.flush, name="verify-off-thread")
    off.start()
    off.join(timeout=10.0)
    assert writes, "the off-thread flush never wrote"
    assert all(name == "verify-off-thread" for name in writes)


def test_pool_exhausted_interval_never_writes_on_the_training_thread(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The RT-01 pool-exhausted branch: an interval without events is judged
    inline."""
    config = StragglerConfig(enabled=True, output_dir=str(tmp_path), window_seconds=3600.0, event_pool=2)
    collector = _collector(tmp_path, event_pool=2)
    writes = _record_writes(monkeypatch)
    observer = StragglerObserver(config, identity=IDENTITY, backend=_SmallPoolBackend(), consumer=collector.ingest)

    tokens = [observer.acquire_interval("forward-compute", 2) for _ in range(3)]
    # event_pool=2 is exactly one interval pair, so the second acquire exhausts.
    assert tokens[0] is not None
    assert tokens[1] is None and tokens[2] is None, "the event pool did not exhaust as expected"

    for i in range(WRITE_BATCH + 20):
        observer.complete_interval(None, "forward-compute", 2, 500.0 + i, 500.1 + i, False)

    assert writes == [], f"training thread wrote files: {writes}"
    assert (tmp_path / "straggler_envelopes.jsonl").read_text(encoding="utf-8") == ""

    off = threading.Thread(target=collector.flush, name="verify-off-thread")
    off.start()
    off.join(timeout=10.0)
    assert writes and all(name == "verify-off-thread" for name in writes)
    observer.close()


# --- conservation after the locking lands ---------------------------------------


def test_concurrent_ingest_conserves_every_packet_class() -> None:
    """Every accepted/duplicate/late/malformed packet is accounted for once."""
    collector = TimingCollector(
        StragglerConfig(enabled=True, window_seconds=3600.0, warmup_windows=0, persist_windows=1),
        identity=IDENTITY,
    )
    writers = 7
    per_thread = 200
    malformed = 37
    first_barrier = threading.Barrier(writers)
    second_barrier = threading.Barrier(writers)
    errors: List[str] = []

    def phase(seq_offset: int, barrier: threading.Barrier) -> None:
        def worker(rank: int) -> None:
            try:
                barrier.wait(timeout=10.0)
                for seq in range(per_thread):
                    collector.ingest(_envelope(rank, seq + seq_offset))
            except BaseException as exc:  # noqa: BLE001
                errors.append(f"{type(exc).__name__}: {exc}")

        threads = [threading.Thread(target=worker, args=(rank,)) for rank in range(writers)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=30.0)

    phase(0, first_barrier)  # all new
    phase(0, second_barrier)  # all resends -> duplicates
    collector.ingest(_envelope(7, 10))  # a new key for a new rank
    collector.ingest(_envelope(7, 5))  # behind rank 7's newest -> late
    for _ in range(malformed):
        collector.ingest({})
    collector.flush()

    status = collector.status()
    judged = writers * per_thread + 1
    assert errors == []
    assert status["envelopes"] == writers * per_thread * 2 + 2 + malformed
    assert status["judged_packets"] == judged
    assert status["duplicate_packets"] == writers * per_thread
    assert status["late_packets"] == 1
    assert status["invalid_packets"] == malformed
    assert status["ingest_errors"] == 0
    assert status["dedup"]["new"] == judged
    assert status["dedup"]["duplicate"] == writers * per_thread
    assert status["dedup"]["late"] == 1
    assert status["dedup"]["entries"] == judged + 1
    # One window, fully closed, judged exactly once.
    assert status["open_windows"] == 0
    assert status["windows_closed"] == 1
    assert status["cohort_stage_pairs"] == 1


def test_dedup_evictions_are_counted_and_conserved(monkeypatch: pytest.MonkeyPatch) -> None:
    """A small dedup cap must report every eviction, not silently forget."""
    monkeypatch.setattr(collector_module, "DEDUP_MAX_ENTRIES", 32)
    collector = TimingCollector(StragglerConfig(enabled=True, window_seconds=3600.0), identity=IDENTITY)
    total = 100
    for seq in range(total):
        collector.ingest(_envelope(0, seq))

    status = collector.status()
    assert status["judged_packets"] == total
    assert status["dedup"]["new"] == total
    assert status["dedup"]["entries"] == 32
    assert status["dedup"]["evicted"] == total - 32
    assert status["ingest_errors"] == 0


# --- regression re-verification -------------------------------------------------


def test_rt09_staggered_start_is_compared_on_the_current_tip() -> None:
    """Independent re-run of the RT-09 scenario (rank 1 joins 3 windows
    late)."""
    detector = StragglerDetector(
        StragglerConfig(enabled=True, window_seconds=5.0, warmup_windows=0, persist_windows=1, work_tolerance=0.05)
    )
    verdicts: List[Any] = []
    seq = {0: 0, 1: 0}
    t = 0.0
    while t < 60.0:
        seq[0] += 1
        verdicts += detector.observe(_envelope(0, seq[0], host_ms=100.0, start_s=t))
        if t >= 15.0:
            seq[1] += 1
            verdicts += detector.observe(_envelope(1, seq[1], host_ms=300.0, start_s=t))
        t += 0.5
    verdicts += detector.flush()

    assert [verdict.rank for verdict in verdicts if verdict.kind == VERDICT_STRAGGLER] == [1]
    stats = detector.stats()
    assert stats["incomplete_windows"] > 0, "the pre-start windows must be counted, not guessed"


def _local_remove_non_pickleables(obj: Any, max_depth: int = 3, current_depth: int = 0) -> Any:
    """Dependency-free copy of upstream ``remove_non_pickleables``.

    ``copy.copy`` + ``setattr`` over ``vars()`` is what turned the pre-fix
    slotted-less shim into a ``FrozenInstanceError`` on a
    ``TransformerConfig``.
    """
    if current_depth >= max_depth:
        return obj
    if obj is None:
        return obj
    if callable(obj):
        if isinstance(obj, type):
            return obj
        if isinstance(obj, (types.FunctionType, types.MethodType, functools.partial)) or hasattr(obj, "__self__"):
            return None
    if hasattr(obj, "__dict__"):
        cleaned = copy.copy(obj)
        for name, value in vars(obj).items():
            setattr(cleaned, name, _local_remove_non_pickleables(value, max_depth, current_depth + 1))
        return cleaned
    if isinstance(obj, (list, tuple)):
        return type(obj)(_local_remove_non_pickleables(item, max_depth, current_depth + 1) for item in obj)
    if isinstance(obj, dict):
        return {key: _local_remove_non_pickleables(value, max_depth, current_depth + 1) for key, value in obj.items()}
    return obj


def test_a7_local_walker_still_reproduces_the_pre_fix_crash() -> None:
    """The A7 invariant must remain enforced without a Megatron checkout."""

    class _PreFixTimers:
        """The pre-fix shape: an instance ``__dict__`` holding a frozen
        config."""

        def __init__(self) -> None:
            self.__dict__["_config"] = StragglerConfig(enabled=True)
            self.__dict__["_log_levels"] = {}

        def __call__(self, name: str, log_level: Optional[int] = None) -> Any:
            return None

    with pytest.raises(dataclasses.FrozenInstanceError):
        _local_remove_non_pickleables(_PreFixTimers())

    # The current shim is slotted, so the same walk leaves it untouched and
    # functional instead of rewriting it into a disabled copy.
    from relax.utils.straggler.megatron_timer_shim import StragglerTimers

    shim = StragglerTimers(StragglerConfig(enabled=True), sink=lambda envelope: None)
    assert not hasattr(shim, "__dict__")
    assert _local_remove_non_pickleables(shim) is shim
    assert shim("forward-compute", log_level=2) is not None
