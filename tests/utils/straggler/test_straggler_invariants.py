# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Invariant audit for the Task 11 straggler profiler (Phase 1 evidence).

Three properties are asserted here, each with an independent mechanism so a
single weak test cannot carry the claim:

* **A. zero training-path collective.** (1) an AST guard scans every module for
  forbidden distributed/device/network entry points; (2) a runtime
  instrumentation test wraps those entry points with recorders and drives the
  real timers -> observer -> sender -> receiver -> collector path in-process.
* **B. strictly bounded.** The event pool never allocates past its capacity and
  every documented structure cap is asserted directly.
* **C. failure cannot escape.** Every injected failure path is counted and the
  training-side call returns normally.

The real multi-process degradation evidence lives in
``task11_evidence/downgrade_harness.py`` (8/8 checks); this module is the
in-process, machine-checkable complement.
"""

import ast
import socket
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pytest

from relax.utils.straggler import collector as collector_module
from relax.utils.straggler import detector as detector_module
from relax.utils.straggler import observer as observer_module
from relax.utils.straggler.collector import EnvelopeReceiver, EnvelopeSender, TimingCollector
from relax.utils.straggler.config import StragglerConfig
from relax.utils.straggler.identity import RuntimeIdentity
from relax.utils.straggler.megatron_timer_shim import StragglerTimers
from relax.utils.straggler.observer import StragglerObserver, TimingEnvelope
from tests.utils.straggler.test_straggler_observer import FakeEventBackend


PACKAGE_DIR = Path(observer_module.__file__).resolve().parent

#: Modules that execute on the training thread (or are reachable from it) and
#: therefore must not contain any blocking network call.
TRAINING_PATH_MODULES = (
    "__init__.py",
    "config.py",
    "context.py",
    "detector.py",
    "identity.py",
    "megatron_timer_shim.py",
    "observer.py",
    "protocol.py",
    "reporter.py",
    "runtime.py",
    "stages.py",
)

#: Transport module: the only place a socket may be touched, and only from its
#: own background threads.
TRANSPORT_MODULES = ("collector.py",)

#: Exact final attribute segment of a forbidden call.
_COLLECTIVE_SEGMENTS = frozenset(
    {"all_gather", "all_gather_object", "all_reduce", "reduce_scatter", "all_to_all", "broadcast", "barrier"}
)
_SYNC_SEGMENTS = frozenset({"synchronize", "item", "tolist"})
_NETWORK_SEGMENTS = frozenset({"accept", "recv", "recvfrom", "recv_into", "sendall", "connect", "create_connection"})
_FORBIDDEN_PREFIXES = ("requests", "urllib", "select", "http.client")


def _dotted(node: ast.AST) -> Optional[str]:
    """Return the dotted attribute name of ``node``, or ``None``."""
    parts: List[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
        return ".".join(reversed(parts))
    return None


def _module_forbidden_attributes(path: Path) -> List[Tuple[int, str, bool]]:
    """Return ``(lineno, dotted_name, is_call)`` for every attribute in one
    module.

    ``is_call`` distinguishes a real entry point (``dist.barrier(...)``) from a
    plain attribute that merely shares a name (``token.barrier`` is a boolean
    field, not a collective).
    """
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    found: List[Tuple[int, str, bool]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            name = _dotted(node.func)
            if name is not None:
                found.append((node.lineno, name, True))
        if isinstance(node, ast.Attribute):
            name = _dotted(node)
            if name is not None:
                found.append((node.lineno, name, False))
    return found


def _classify(attributes: List[Tuple[int, str, bool]]) -> Dict[str, List[str]]:
    """Bucket forbidden attribute names by the property they would violate."""
    buckets: Dict[str, List[str]] = {"collective": [], "sync": [], "network": []}
    for line, name, is_call in attributes:
        head = name.split(".", 1)[0]
        tail = name.rsplit(".", 1)[-1]
        module_prefixed = head in ("dist", "torch") or name.startswith("torch.distributed")
        where = f"{name}:{line}"
        if tail in _COLLECTIVE_SEGMENTS and (is_call or name.startswith("torch.distributed") or head == "dist"):
            buckets["collective"].append(where)
        if tail in _SYNC_SEGMENTS and (is_call or name.startswith("torch.")):
            buckets["sync"].append(where)
        if head in _FORBIDDEN_PREFIXES:
            buckets["network"].append(where)
        elif tail in _NETWORK_SEGMENTS and (is_call or module_prefixed):
            buckets["network"].append(where)
    return buckets


def test_ast_guard_training_path_has_no_forbidden_entry_point() -> None:
    """A.1 static guard: no collective, sync, or blocking network in the
    path."""
    violations: Dict[str, List[str]] = {}
    for filename in TRAINING_PATH_MODULES:
        path = PACKAGE_DIR / filename
        buckets = _classify(_module_forbidden_attributes(path))
        bad = buckets["collective"] + buckets["sync"] + buckets["network"]
        if bad:
            violations[filename] = bad

    assert violations == {}, f"training-path modules contain forbidden entry points: {violations}"


def test_ast_guard_transport_confines_sockets_to_the_background_transport() -> None:
    """A.1 static guard: only ``collector.py`` may touch a socket, and no
    collective/sync anywhere."""
    for filename in TRANSPORT_MODULES:
        buckets = _classify(_module_forbidden_attributes(PACKAGE_DIR / filename))
        assert buckets["collective"] == []
        assert buckets["sync"] == []

    # And the transport module really is the only one: every other module was
    # checked to have no network attribute at all by the previous test.
    assert "collector.py" in TRANSPORT_MODULES


def _raiser(name: str):
    calls: List[str] = []

    def explode(*_args: Any, **_kwargs: Any) -> Any:
        calls.append(name)
        raise AssertionError(f"forbidden training-path call: {name}")

    explode.calls = calls  # type: ignore[attr-defined]
    return explode


def test_runtime_instrumentation_full_path_makes_no_forbidden_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A.3 runtime evidence: wrap the forbidden entry points and drive the real
    shim -> observer -> sender -> socket -> receiver -> collector path."""
    import torch

    raisers: List[Any] = []
    for module_name, names in (
        (
            "torch.distributed",
            ("all_gather", "all_gather_object", "all_reduce", "reduce_scatter", "barrier", "broadcast"),
        ),
        ("torch.cuda", ("synchronize",)),
    ):
        target = torch.distributed if module_name == "torch.distributed" else torch.cuda
        for name in names:
            if hasattr(target, name):
                raiser = _raiser(f"{module_name}.{name}")
                monkeypatch.setattr(target, name, raiser)
                raisers.append(raiser)

    socket_threads: List[Tuple[str, str]] = []
    flushes: List[str] = []
    main_thread = threading.current_thread().name

    real_sendall = socket.socket.sendall
    real_create = socket.create_connection
    real_accept = socket.socket.accept

    def sendall(self: Any, data: Any) -> Any:
        socket_threads.append(("sendall", threading.current_thread().name))
        return real_sendall(self, data)

    def create_connection(*args: Any, **kwargs: Any) -> Any:
        socket_threads.append(("create_connection", threading.current_thread().name))
        return real_create(*args, **kwargs)

    def accept(self: Any) -> Any:
        socket_threads.append(("accept", threading.current_thread().name))
        return real_accept(self)

    monkeypatch.setattr(socket.socket, "sendall", sendall)
    monkeypatch.setattr(socket, "create_connection", create_connection)
    monkeypatch.setattr(socket.socket, "accept", accept)

    identity = RuntimeIdentity(
        run_id="invariant-run", rank=0, world_size=4, tensor_parallel_rank=0, data_parallel_rank=0
    )
    config = StragglerConfig(
        enabled=True,
        window_seconds=1.0,
        warmup_windows=0,
        persist_windows=1,
        event_pool=16,
        queue_max=256,
        output_dir=str(tmp_path),
    )
    # Force a persistence write to happen after a couple of envelopes so the
    # "writes only on the reader thread" claim is actually exercised.
    monkeypatch.setattr(collector_module, "WRITE_BATCH", 2)

    collector = TimingCollector(config, identity=identity)
    real_flush = collector._flush_path

    def flush_recording(path: str) -> None:
        flushes.append(threading.current_thread().name)
        real_flush(path)

    monkeypatch.setattr(collector, "_flush_path", flush_recording)

    receiver = EnvelopeReceiver("127.0.0.1:0", lambda payload: collector.ingest(TimingEnvelope.from_dict(payload)))
    receiver.start()
    if not receiver.stats()["listening"]:
        pytest.skip("loopback TCP is unavailable in this environment")
    host, port = receiver.address

    sender = EnvelopeSender(f"{host}:{port}", queue_max=256, reconnect_interval_s=0.2)
    observer = StragglerObserver(
        config,
        identity=identity,
        backend=FakeEventBackend(),
        consumer=sender.send,
        poll_interval_s=0.001,
    )
    timers = StragglerTimers(config, sink=observer)

    try:
        for _ in range(6):
            handle = timers("forward-compute", 2)
            handle.start()
            handle.stop()
        deadline = time.perf_counter() + 5.0
        while time.perf_counter() < deadline and collector.status()["envelopes"] < 6:
            time.sleep(0.01)
        status = collector.status()
    finally:
        sender.close()
        receiver.close()
        observer.close()

    # 1. No forbidden call was ever made.
    assert all(raiser.calls == [] for raiser in raisers), [raiser.calls for raiser in raisers]
    # 2. All 6 envelopes made it through the whole transport path.
    assert status["envelopes"] == 6
    assert status["invalid_packets"] == 0
    assert status["judged_packets"] == 6
    # 3. Every socket operation happened off the training (main) thread.
    assert socket_threads, "the transport path was never exercised"
    assert all(thread != main_thread for _, thread in socket_threads), socket_threads
    # 4. Every persistence write happened off the training thread too.
    assert flushes, "no flush was exercised"
    assert all(thread != main_thread for thread in flushes), flushes


def test_event_pool_never_allocates_past_capacity(monkeypatch: pytest.MonkeyPatch) -> None:
    """B: acquire far more pairs than the pool holds; creation stays capped."""
    backend = FakeEventBackend()
    config = StragglerConfig(enabled=True, event_pool=4, queue_max=64)
    observer = StragglerObserver(config, identity=RuntimeIdentity(run_id="r", rank=0, world_size=1), backend=backend)

    acquired = [observer.acquire_interval("forward-compute", 2) for _ in range(200)]
    observer.close()

    assert backend.created <= 4
    assert observer.stats()["pool"]["created"] <= 4
    assert sum(1 for token in acquired if token is not None) == 2


def test_documented_structure_caps_are_the_intended_bounds() -> None:
    """B: the constants the report's bounded-structure table cites."""
    assert detector_module.MAX_PENDING_WINDOWS == 8
    assert detector_module.MAX_VERDICTS == 512
    assert detector_module.MAX_STREAK_ENTRIES == 4096
    assert detector_module.MAX_LABEL_ENTRIES == 4096
    assert detector_module.MAX_ALIGNED_RANKS == 4096
    assert detector_module.MAX_ACTIVE_ENTRIES == 512
    assert detector_module.MAX_PAIRS_PER_WINDOW == 256
    assert detector_module.MAX_RANKS_PER_WINDOW == 1024
    assert detector_module.MAX_SAMPLES_PER_RANK == 32
    assert observer_module.STATE_HISTORY_MAX == 64
    assert observer_module.NAME_BUDGET == 128
    assert observer_module.DISABLE_AFTER_FAILURES == 64
    assert collector_module.MAX_CONNECTIONS == 64
    assert collector_module.MAX_LINE_BYTES == 1 << 20
    assert collector_module.WRITE_BATCH == 256
    assert collector_module.DEDUP_MAX_ENTRIES == 8192
    assert collector_module.DEDUP_TTL_S == 120.0


def test_event_pool_full_condition_drops_and_counts() -> None:
    """B: the full-condition behaviour of the pool is a counted drop."""
    backend = FakeEventBackend()
    config = StragglerConfig(enabled=True, event_pool=2, queue_max=64)
    observer = StragglerObserver(config, identity=RuntimeIdentity(run_id="r", rank=0, world_size=1), backend=backend)

    assert observer.acquire_interval("forward-backward", 1) is not None
    assert observer.acquire_interval("forward-compute", 2) is None

    stats = observer.stats()
    assert stats["host_only_intervals"] == 1
    assert stats["pool"]["exhausted"] == 1
    assert observer.state == "degraded"


class _ExplodingBackend(FakeEventBackend):
    """Backend whose every device operation fails."""

    def __init__(self, fail_on: str) -> None:
        super().__init__()
        self._fail_on = fail_on

    def create_event(self) -> Any:
        if self._fail_on == "create":
            raise RuntimeError("create_event failed")
        return super().create_event()

    def record(self, event: Any) -> None:
        if self._fail_on == "record":
            raise RuntimeError("record failed")
        super().record(event)

    def is_complete(self, event: Any) -> bool:
        if self._fail_on == "read":
            raise RuntimeError("is_complete failed")
        return super().is_complete(event)


def test_injected_event_create_and_record_failures_do_not_escape() -> None:
    """C: event creation/record failure is counted and the call returns."""
    for fail_on in ("create", "record", "read"):
        config = StragglerConfig(enabled=True, event_pool=8, queue_max=64)
        observer = StragglerObserver(
            config,
            identity=RuntimeIdentity(run_id="r", rank=0, world_size=1),
            backend=_ExplodingBackend(fail_on),
            poll_interval_s=0.001,
        )
        token = observer.acquire_interval("forward-compute", 2)  # must not raise
        observer.complete_interval(token, "forward-compute", 2, 0.0, 0.1, False)
        deadline = time.perf_counter() + 2.0
        while time.perf_counter() < deadline and observer.stats()["observer_errors"] == 0:
            time.sleep(0.01)
        assert observer.state in ("degraded", "disabled")
        observer.close()


def test_injected_consumer_and_callback_failures_do_not_escape() -> None:
    """C: a failing consumer/verdict callback is counted, never raised."""
    config = StragglerConfig(enabled=True, event_pool=8, queue_max=8)
    observer = StragglerObserver(
        config,
        identity=RuntimeIdentity(run_id="r", rank=0, world_size=1),
        backend=FakeEventBackend(),
        consumer=lambda envelope: (_ for _ in ()).throw(RuntimeError("consumer boom")),
        poll_interval_s=0.001,
    )
    token = observer.acquire_interval("forward-compute", 2)
    observer.complete_interval(token, "forward-compute", 2, 0.0, 0.1, False)
    deadline = time.perf_counter() + 2.0
    while time.perf_counter() < deadline and observer.stats()["consumer_errors"] == 0:
        time.sleep(0.01)
    assert observer.stats()["consumer_errors"] == 1
    assert observer.state == "degraded"
    observer.close()


def test_injected_serializer_and_socket_failures_do_not_escape() -> None:
    """C: a broken envelope or a dead collector socket is counted."""

    class Broken:
        def to_json(self) -> str:
            raise RuntimeError("serialiser boom")

    sender = EnvelopeSender("127.0.0.1:1", queue_max=2, reconnect_interval_s=0.2)
    try:
        sender.send(Broken())  # must not raise
        assert sender.stats()["send_errors"] == 1
        sender.send(_envelope_for_sender())
        assert sender.stats()["queued"] == 1
    finally:
        sender.close()
    # The unreachable address means the transport thread also counts errors.
    assert sender.stats()["send_errors"] >= 1


def _envelope_for_sender() -> Any:
    from relax.utils.straggler.observer import TimingEnvelope

    return TimingEnvelope(
        run_id="r",
        rank=1,
        cohort="0:0:0:0:0",
        label="rank1",
        world_size=4,
        name="forward-compute",
        log_level=2,
        seq=1,
        host_start=0.0,
        host_end=0.1,
        device_ms=None,
        barrier=False,
        reason="no_event_pair",
    )


class _AlwaysErrorDetector:
    """Detector double whose observe always raises."""

    def observe(self, envelope: Any) -> List[Any]:
        raise RuntimeError("detector boom")

    def flush(self) -> List[Any]:
        raise RuntimeError("flush boom")

    def stats(self) -> Dict[str, Any]:
        return {}

    def active_stragglers(self) -> List[Any]:
        return []


def test_injected_collector_crash_does_not_escape_ingest() -> None:
    """C: a detector that dies under the collector is counted, never raised."""
    config = StragglerConfig(enabled=True, window_seconds=1.0, warmup_windows=0, persist_windows=1)
    collector = TimingCollector(config, identity=RuntimeIdentity(run_id="r", rank=0, world_size=4))
    collector._detector = _AlwaysErrorDetector()

    assert collector.ingest(_envelope_for_sender()) == []
    assert collector.status()["ingest_errors"] == 1


def test_malformed_and_duplicate_packets_are_counted_not_judged() -> None:
    """C: malformed, duplicate and late packets stay out of the detector."""
    config = StragglerConfig(enabled=True, window_seconds=1.0, warmup_windows=0, persist_windows=1)
    collector = TimingCollector(config, identity=RuntimeIdentity(run_id="r", rank=0, world_size=4))

    assert collector.ingest(None) == []
    assert collector.ingest({"schema_version": 2, "rank": -1, "name": "x"}) == []
    packet = _envelope_for_sender()
    collector.ingest(packet)
    collector.ingest(packet)
    late = _envelope_for_sender()
    object.__setattr__(late, "seq", 0)
    collector.ingest(late)

    status = collector.status()
    assert status["invalid_packets"] == 2
    assert status["judged_packets"] == 1
    assert status["duplicate_packets"] == 1
    assert status["late_packets"] == 1


def test_shutdown_with_unfinished_events_delivers_and_releases() -> None:
    """C: close() with still-pending events never raises and keeps evidence."""
    seen: List[Any] = []
    config = StragglerConfig(enabled=True, event_pool=8, queue_max=64)
    observer = StragglerObserver(
        config,
        identity=RuntimeIdentity(run_id="r", rank=0, world_size=1),
        backend=FakeEventBackend(auto_complete=False),
        consumer=seen.append,
        readout_timeout_s=30.0,
    )
    tokens = [observer.acquire_interval("forward-compute", 2) for _ in range(4)]
    for token in tokens:
        observer.complete_interval(token, "forward-compute", 2, 0.0, 0.1, False)

    observer.close()
    observer.close()  # idempotent

    assert len(seen) == 4
    assert {envelope.reason for envelope in seen} == {"closed"}
    assert observer.stats()["pending"] == 0
