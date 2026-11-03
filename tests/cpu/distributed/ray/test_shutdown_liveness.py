# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Shutdown retires servers even when abort or a sender cannot finish."""

from types import SimpleNamespace

import pytest
import ray

from relax.distributed.ray.multi_engine_manager import MultiEngineManager
from relax.distributed.ray.rollout import RolloutEngineRuntime
from relax.engine.inference.admission import AdmissionDrainError, AdmissionGate
from relax.engine.inference.holder import HolderIdentity
from relax.engine.inference.routing import ModelUnavailableError
from tests.cpu.engine.inference.test_request_dispatch import snapshot


@pytest.mark.parametrize("runtime_kind", ["rollout", "static"])
@pytest.mark.parametrize("failure", ["dead-engine-abort", "abort-timeout", "queued-sender"])
def test_shutdown_stops_engines_without_waiting_for_sender_quiescence(monkeypatch, runtime_kind, failure):
    monkeypatch.setenv("RELAX_INFERENCE_ADMISSION_GRACE_S", "0")
    gate = AdmissionGate()
    permits = [
        gate.acquire(snapshot, f"req-{i}", holder=HolderIdentity(actor_id=f"holder-{i}"), dispatch_tracked=True)
        for i in range(7)
    ]
    generation = gate._generation
    events = []
    warnings = []
    monkeypatch.setattr(
        "relax.engine.inference.admission.logger.warning",
        lambda message, *args, **kwargs: warnings.append(message % args),
    )

    def abort(*, timeout):
        assert timeout == 30
        events.append("abort submitted")
        return object()

    def wait(refs, *, timeout):
        assert len(refs) == 1 and timeout == 35
        assert gate._closed
        events.append("abort waited")
        if failure == "dead-engine-abort":
            raise ray.exceptions.ActorDiedError()
        if failure == "abort-timeout":
            raise ray.exceptions.GetTimeoutError()

    monkeypatch.setattr(ray, "get", wait)
    monkeypatch.setattr("relax.engine.inference.admission.holder_alive", lambda holder: True)
    wait_for = gate._condition.wait_for
    waits = []

    def record_wait(*args, **kwargs):
        waits.append(kwargs.get("timeout"))
        return wait_for(*args, **kwargs)

    monkeypatch.setattr(gate._condition, "wait_for", record_wait)
    engine = SimpleNamespace(abort_requests=SimpleNamespace(remote=abort))
    cls = RolloutEngineRuntime if runtime_kind == "rollout" else MultiEngineManager
    runtime = object.__new__(cls)
    runtime._admission = gate
    if runtime_kind == "rollout":
        runtime.servers = {"default": SimpleNamespace(engines=[engine])}
        runtime._stop_eviction_monitor = lambda **kwargs: None
        runtime._health_monitors = []
        runtime._shutdown_all_engines = lambda: events.append("engines stopped")
        monkeypatch.setattr("relax.distributed.ray.rollout.stop_launched_routers", lambda: None)
        shutdown = runtime.dispose
    else:
        runtime.all_engines, runtime.nodes_per_engine = [engine], 1
        runtime._pool = SimpleNamespace(shutdown=lambda: events.append("engines stopped"))
        shutdown = runtime.shutdown
    shutdown()
    assert events == ["abort submitted", "abort waited", "engines stopped"]
    assert waits == [], "shutdown must not wait for the admission grace or sender receipts"
    assert gate._closed and not gate._permits
    logged = "\n".join(warnings)
    assert all(f"req-{i}" in logged and f"holder-{i}" in logged for i in range(5))
    assert "req-5" not in logged and "req-6" not in logged
    if failure != "queued-sender":
        assert "abort" in logged.lower()
    for permit in permits:
        gate.release(permit)
    with pytest.raises(ModelUnavailableError):
        gate.acquire(snapshot, "late")
    with pytest.raises(AdmissionDrainError):
        gate.open()
    assert gate._generation == generation


@pytest.mark.parametrize("runtime_kind", ["rollout", "static"])
def test_shutdown_still_requires_process_exit_proof(monkeypatch, runtime_kind):
    gate = AdmissionGate()
    cls = RolloutEngineRuntime if runtime_kind == "rollout" else MultiEngineManager
    runtime = object.__new__(cls)
    runtime._admission = gate

    def unconfirmed():
        raise RuntimeError("server tree is still alive")

    if runtime_kind == "rollout":
        runtime.servers = {}
        runtime._stop_eviction_monitor = lambda **kwargs: None
        runtime._health_monitors = []
        runtime._shutdown_all_engines = unconfirmed
        shutdown = runtime.dispose
    else:
        runtime.all_engines, runtime.nodes_per_engine = [], 1
        runtime._pool = SimpleNamespace(shutdown=unconfirmed)
        shutdown = runtime.shutdown
    with pytest.raises(RuntimeError, match="server tree is still alive"):
        shutdown()
    assert gate._closed
