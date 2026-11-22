# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Receipt/atomic handoff contracts independent of arrival order and
transport."""

import itertools
import threading
import time
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from relax.utils.fault_events.models import (
    MAX_UINT,
    FaultContext,
    encode,
    new_id,
)
from relax.utils.fault_events.receipts import OPERATIONS, ReceiptInbox, validate_receipt, validate_token
from relax.utils.fault_events.recovery import RecoveryPlan
from relax.utils.fault_events.runtime import FaultEventRuntime
from relax.utils.health_system import HealthStatus


Status = HealthStatus.__ray_metadata__.modified_class


@pytest.fixture
def runtime():
    runtime = FaultEventRuntime()
    runtime.bind_source()
    yield runtime
    runtime.close()
    runtime._consumer.join(3)
    assert not runtime._consumer.is_alive()


def snapshot(runtime, *faults, receipts=None):
    return {
        "roles": [fault.role for fault in faults],
        "source_generation": runtime.generation,
        "faults": [asdict(fault) for fault in faults],
        "observation_error": False,
        "receipts": receipts,
    }


def start(runtime, plan=None):
    plan = plan or RecoveryPlan(("actor", "actor_fwd", "reference"), "pure_async")
    fault = FaultContext("actor", new_id(), runtime.generation, new_id(), "reported_error")
    runtime.observe_snapshot(snapshot(runtime, fault))
    recovery = runtime.start_recovery("actor", "global", plan.expected, plan=plan)
    assert recovery is not None
    generation, token = runtime.bind_source(recovery)
    assert token == runtime.token(recovery) and generation == recovery.receiver_generation
    return fault, recovery, ReceiptInbox(token)


def drain(runtime):
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        state = runtime.diagnostics()
        if not state["queued"] and not state["inflight"]:
            return runtime.query(limit=100)["events"]
        time.sleep(0.001)
    pytest.fail("consumer did not drain")


@pytest.mark.parametrize("order", list(itertools.permutations(("handoff", *OPERATIONS))))
def test_three_receipts_and_handoff_all_24_orders(runtime, order):
    _, recovery, inbox = start(runtime)
    for index, item in enumerate(order):
        if item == "handoff":
            runtime.complete_handoff(recovery, recovery.plan, 7)
        else:
            assert inbox.receive(runtime.token(recovery), item, "succeeded") == "accepted"
            runtime.observe_snapshot(snapshot(runtime, receipts=inbox.snapshot()))
        assert recovery.outcome == ("recovery_succeeded" if index == 3 else None)
    events = drain(runtime)
    assert [e["event_type"] for e in events] == [
        "fault_detected",
        "handling_started",
        "recovery_handoff",
        "recovery_succeeded",
    ]
    assert events[-1]["expected_evidence"] == events[-1]["observed_evidence"] == 63
    assert events[-1]["failed_evidence"] == 0


@pytest.mark.parametrize("failed_operation", OPERATIONS)
def test_failed_receipt_is_operation_failure_without_waiting_for_handoff(runtime, failed_operation):
    _, recovery, inbox = start(runtime)
    runtime.evidence(recovery, observed=1)
    inbox.receive(runtime.token(recovery), failed_operation, "failed")
    runtime.observe_snapshot(snapshot(runtime, receipts=inbox.snapshot()))
    terminal = drain(runtime)[-1]
    assert terminal["event_type"] == "recovery_failed" and terminal["reason_code"] == "operation_failed"
    assert terminal["failed_evidence"] == OPERATIONS[failed_operation]
    assert terminal["observed_evidence"] == 1
    runtime.complete_handoff(recovery, recovery.plan, 7)
    assert len(drain(runtime)) == 3


def test_missing_receipt_stays_unverified_and_post_terminal_conflict_keeps_history(runtime):
    _, recovery, inbox = start(runtime)
    runtime.complete_handoff(recovery, recovery.plan, 7)
    for operation in ("actor_send", "actor_fwd_recv"):
        inbox.receive(runtime.token(recovery), operation, "succeeded")
    for _ in range(10):
        runtime.observe_snapshot(snapshot(runtime, receipts=inbox.snapshot()))
    assert recovery.outcome is None and len(drain(runtime)) == 3
    assert runtime.diagnostics()["known_lost_events"] == 0
    assert inbox.receive(runtime.token(recovery), "reference_recv", "succeeded") == "accepted"
    runtime.observe_snapshot(snapshot(runtime, receipts=inbox.snapshot()))
    before = drain(runtime)
    assert before[-1]["event_type"] == "recovery_succeeded"
    assert inbox.receive(runtime.token(recovery), "actor_send", "failed") == "conflict"
    runtime.observe_snapshot(snapshot(runtime, receipts=inbox.snapshot()))
    assert drain(runtime) == before
    assert "RECEIPT_CONFLICT" in runtime.diagnostics()["unknown_reasons"]


def test_conflict_before_handoff_prevents_success(runtime):
    _, recovery, inbox = start(runtime)
    for operation in OPERATIONS:
        inbox.receive(runtime.token(recovery), operation, "succeeded")
    inbox.receive(runtime.token(recovery), "actor_send", "failed")
    runtime.observe_snapshot(snapshot(runtime, receipts=inbox.snapshot()))
    runtime.complete_handoff(recovery, recovery.plan, 7)
    assert recovery.conflict and recovery.outcome is None
    assert drain(runtime)[-1]["event_type"] == "recovery_handoff"


@pytest.mark.parametrize(
    "first,opposite,merged_first",
    [
        ("succeeded", "failed", False),
        ("succeeded", "failed", True),
        ("failed", "succeeded", False),
    ],
)
def test_conflicting_slot_does_not_hide_an_independent_failed_operation(runtime, first, opposite, merged_first):
    _, recovery, inbox = start(runtime)
    token = runtime.token(recovery)
    inbox.receive(token, "actor_send", first)
    if merged_first:
        runtime.observe_snapshot(snapshot(runtime, receipts=inbox.snapshot()))
    assert inbox.receive(token, "actor_send", opposite) == "conflict"
    assert not inbox.snapshot()["failed"]  # The contradictory slot alone is ambiguous.
    inbox.receive(token, "actor_fwd_recv", "failed")
    runtime.observe_snapshot(snapshot(runtime, receipts=inbox.snapshot()))
    terminal = drain(runtime)[-1]
    assert terminal["event_type"] == "recovery_failed" and terminal["failed_evidence"] == 16
    assert "RECEIPT_CONFLICT" in runtime.diagnostics()["unknown_reasons"]


def test_source_observation_retirement_keeps_independent_receipt_slots(runtime):
    _, recovery, _ = start(runtime)
    status = Status(fault_generation=runtime.generation, fault_seed=runtime.token(recovery))
    status._fault_source.clear = lambda role: (_ for _ in ()).throw(RuntimeError("source unavailable"))
    status.mark_healthy("actor")
    assert status._fault_source is None
    assert status.record_recovery_receipt(runtime.token(recovery), "actor_send", "failed") == "accepted"
    runtime.observe_snapshot(status.get_unhealthy_services(include_faults=True))
    assert recovery.outcome == "recovery_failed"
    assert drain(runtime)[-1]["failed_evidence"] == 8


def test_receiver_rejects_seed_for_a_different_generation(runtime):
    _, recovery, _ = start(runtime)
    status = Status(fault_generation=runtime.generation + 1, fault_seed=runtime.token(recovery))
    assert status.record_recovery_receipt(runtime.token(recovery), "actor_send", "succeeded") == "invalid"
    envelope = status.get_unhealthy_services(include_faults=True)
    assert envelope["observation_error"] and "receipts" not in envelope


@pytest.mark.parametrize("drift", ["branch", "roles", "sync"])
def test_plan_drift_is_checked_atomically_even_with_identical_masks(runtime, drift):
    plan = RecoveryPlan(("actor", "critic", "rollout"), "colocate", True)
    _, recovery, _ = start(runtime, plan)
    actual = {
        "branch": replace(plan, branch="hybrid"),
        "roles": replace(plan, roles=("actor", "advantages", "rollout")),
        "sync": replace(plan, sync_weights=False),
    }[drift]
    assert actual.expected == plan.expected == 7
    runtime.complete_handoff(recovery, actual, 7)
    assert not recovery.handoff and not recovery.observed and not recovery.plan_verified
    assert recovery.outcome is None
    assert "OBSERVATION_ERROR" in runtime.diagnostics()["unknown_reasons"]


def test_handoff_lock_failure_commits_none_of_plan_evidence_or_handoff(runtime):
    _, recovery, _ = start(runtime, RecoveryPlan(("actor",), "colocate"))
    with runtime._lock:
        runtime.complete_handoff(recovery, recovery.plan, 7)
    assert not recovery.plan_verified and not recovery.handoff and recovery.observed == 0
    runtime.complete_handoff(recovery, recovery.plan, 7)
    assert recovery.outcome == "recovery_succeeded"


def test_rejected_startup_handoff_is_applied_by_the_next_admission(runtime):
    fault, recovery, _ = start(runtime, RecoveryPlan(("actor",), "colocate"))
    config = SimpleNamespace(debug_train_only=False, debug_rollout_only=False, hybrid=False, fully_async=False)
    with runtime._lock:
        runtime.startup_handoff(recovery, config, ["actor"], 7)
    assert not recovery.handoff and recovery.outcome is None
    # The next admitted transaction, here a health snapshot, settles it.
    runtime.observe_snapshot(snapshot(runtime, fault))
    assert recovery.plan_verified and recovery.handoff
    assert recovery.outcome == "recovery_succeeded"
    assert recovery.recovery_id not in runtime._recoveries


def test_deferred_startup_handoff_after_receiver_retirement_is_not_success(runtime):
    _, recovery, _ = start(runtime, RecoveryPlan(("actor",), "colocate"))
    config = SimpleNamespace(debug_train_only=False, debug_rollout_only=False, hybrid=False, fully_async=False)
    with runtime._lock:
        runtime.startup_handoff(recovery, config, ["actor"], 7)
        runtime.retire_source(recovery.receiver_generation)
    # Any later admitted transaction settles both pending fences.
    runtime.evidence(recovery)
    assert recovery.outcome == "recovery_interrupted" and not recovery.handoff
    assert "recovery_succeeded" not in {event["event_type"] for event in drain(runtime)}


def test_deferred_startup_handoff_without_bound_receiver_releases_recovery(runtime):
    fault = FaultContext("actor", new_id(), runtime.generation, new_id(), "reported_error")
    runtime.observe_snapshot(snapshot(runtime, fault))
    plan = RecoveryPlan(("actor",), "colocate")
    recovery = runtime.start_recovery("actor", "global", plan.expected, plan=plan)
    config = SimpleNamespace(debug_train_only=False, debug_rollout_only=False, hybrid=False, fully_async=False)
    # The rebuilt receiver was never bound, so this generation cannot complete.
    runtime.startup_handoff(recovery, config, ["actor"], 7)
    assert not recovery.handoff and recovery.outcome is None
    assert recovery.recovery_id not in runtime._recoveries
    assert "OUTCOME_UNOBSERVED" in runtime.diagnostics()["unknown_reasons"]
    assert "recovery_succeeded" not in {event["event_type"] for event in drain(runtime)}


def test_retirement_lock_failure_is_settled_before_later_handoff(runtime):
    _, recovery, _ = start(runtime, RecoveryPlan(("actor",), "colocate"))
    with runtime._lock:
        runtime.retire_source(1)
    assert recovery.outcome is None
    runtime.complete_handoff(recovery, recovery.plan, 7)
    assert recovery.outcome == "recovery_interrupted" and not recovery.handoff
    assert drain(runtime)[-1]["reason_code"] == "source_retired"


@pytest.mark.parametrize("complete_first", [False, True])
def test_two_generations_retire_only_the_real_receiver_and_reject_old_tokens(runtime, complete_first):
    first_fault, first, inbox = start(runtime, RecoveryPlan(("actor",), "pure_async"))
    runtime.complete_handoff(first, first.plan, 7)
    if complete_first:
        inbox.receive(runtime.token(first), "actor_send", "succeeded")
        runtime.observe_snapshot(snapshot(runtime, receipts=inbox.snapshot()))
    second_fault = FaultContext("actor", new_id(), runtime.generation, new_id(), "reported_error")
    runtime.observe_snapshot(snapshot(runtime, second_fault, receipts=inbox.snapshot()))
    second = runtime.start_recovery("actor", "global", 15, plan=first.plan)
    assert first.outcome == ("recovery_succeeded" if complete_first else None)
    runtime.retire_source(1)
    assert first.outcome == ("recovery_succeeded" if complete_first else "recovery_interrupted")
    assert second.outcome is None
    _, token = runtime.bind_source(second)
    receiver = ReceiptInbox(token)
    assert receiver.receive(runtime.token(first), "actor_send", "succeeded") == "stale"
    runtime.observe_snapshot(snapshot(runtime, receipts=inbox.snapshot()))
    assert second.observed == 0
    runtime.complete_handoff(second, second.plan, 7)
    receiver.receive(token, "actor_send", "succeeded")
    runtime.observe_snapshot(snapshot(runtime, receipts=receiver.snapshot()))
    terminals = [e for e in drain(runtime) if e["event_type"] in ("recovery_succeeded", "recovery_interrupted")]
    assert [e["fault_id"] for e in terminals] == [first_fault.fault_id, second_fault.fault_id]
    assert second.outcome == "recovery_succeeded" and runtime.generation == 2


def test_health_status_seeded_before_first_report_and_empty_query_returns_receipts(runtime):
    _, recovery, _ = start(runtime)
    token = runtime.token(recovery)
    status = Status(fault_generation=runtime.generation, fault_seed=token)
    assert status.record_recovery_receipt(token, "actor_send", "succeeded") == "accepted"
    assert status.record_recovery_receipt(token, "actor_send", "succeeded") == "duplicate"
    envelope = status.get_unhealthy_services(include_faults=True)
    assert envelope["roles"] == envelope["faults"] == []
    assert envelope["receipts"]["observed"] == 8
    assert status.get_unhealthy_services() == []
    runtime.observe_snapshot(envelope)
    assert recovery.observed == 8


@pytest.mark.parametrize(
    "invalid",
    [None, {}, {"run_id": "bad"}, {"run_id": new_id(), "recovery_id": new_id(), "recovery_generation": True}],
)
def test_fixed_token_boundary_rejects_invalid_types(invalid):
    with pytest.raises(ValueError, match="INVALID_TOKEN"):
        validate_token(invalid)


def test_receipt_encoding_and_fixed_slots_are_bounded_at_uint64_limit():
    token = {"run_id": new_id(), "recovery_id": new_id(), "recovery_generation": MAX_UINT}
    inbox = ReceiptInbox(token)
    for _ in range(1000):
        for operation in OPERATIONS:
            validate_receipt(token, operation, "succeeded")
            inbox.receive(token, operation, "succeeded")
            assert len(encode({**token, "operation_id": operation, "outcome": "succeeded"})) <= 256
    assert inbox.observed == 56 and inbox.failed == 0 and len(inbox.__dict__) == 4


def test_repeated_global_cycles_bound_history_seed_contexts_and_threads():
    import gc
    import tracemalloc

    runtime = FaultEventRuntime(capacity=16)
    runtime.bind_source()
    plan = RecoveryPlan(("actor",), "debug")
    threads = {t.ident for t in threading.enumerate() if t.name == "fault-events"}
    tracemalloc.start()
    checkpoint = None
    try:
        for cycle in range(600):
            _, recovery, _ = start(runtime, plan)
            runtime.complete_handoff(recovery, plan, 5)
            assert recovery.outcome == "recovery_succeeded"
            if cycle % 32 == 0:
                drain(runtime)
            if cycle == 100:
                gc.collect()
                checkpoint = tracemalloc.get_traced_memory()[0]
            assert len(runtime._recoveries) == 0 and len(runtime._publications) <= 18
            assert len(runtime._source_epoch[2]) == 18 and len(runtime._receipt_seed) == 3
        drain(runtime)
        status = runtime.status()
        assert status["event_count"] == 16 and status["retention_truncated"]
        assert status["known_lost_events"] == 0
        assert {t.ident for t in threading.enumerate() if t.name == "fault-events"} == threads
        gc.collect()
        assert tracemalloc.get_traced_memory()[0] - checkpoint < 512_000
    finally:
        tracemalloc.stop()
        runtime.close()
        runtime._consumer.join(3)


@pytest.mark.parametrize("enabled", [False, True])
def test_performance_timeline_and_fault_history_keep_independent_data(tmp_path, enabled):
    import json

    from relax.utils.metrics.timeline_trace import TimelineTraceAdapter
    from relax.utils.timer import TimelineEvent

    timeline = TimelineTraceAdapter(str(tmp_path))
    timeline.add_events([TimelineEvent("train_step", 1.0, 1.5, pid=1, tid=2, step=3)])
    runtime = FaultEventRuntime() if enabled else None
    try:
        if runtime is not None:
            runtime.bind_source()
            start(runtime, RecoveryPlan(("actor",), "debug"))
            assert all("train_step" not in str(e) for e in drain(runtime))
        timeline.dump(3)
        assert json.loads((tmp_path / "timeline_step_3.json").read_text()) == [
            {"name": "train_step", "ph": "X", "ts": 1000000, "dur": 500000, "pid": 1, "tid": 2, "args": {"step": 3}}
        ]
    finally:
        if runtime is not None:
            runtime.close()
            runtime._consumer.join(3)


def test_sender_never_replaces_unconfirmed_request_and_queue_stays_bounded(monkeypatch):
    import ray

    from relax.utils.fault_events.receipts import ReceiptSender

    entered, release = threading.Event(), threading.Event()
    submissions = []

    def submit(*item):
        submissions.append(item)
        return object()

    def pending(refs, timeout):
        entered.set()
        release.wait(0.01)
        return [], refs

    monkeypatch.setattr(ray, "wait", pending)
    sender = ReceiptSender(SimpleNamespace(record_recovery_receipt=SimpleNamespace(remote=submit)))
    token = {"run_id": new_id(), "recovery_id": new_id(), "recovery_generation": 1}
    try:
        sender.emit(token, "actor_send", "succeeded")
        assert entered.wait(2)
        for _ in range(1000):
            sender.emit(token, "actor_send", "succeeded")
        assert len(submissions) == 1 and sender._queue.qsize() == 8 and sender.dropped == 992
    finally:
        sender.close()
        release.set()
        sender._thread.join(2)
    assert not sender._thread.is_alive()


@pytest.mark.parametrize("failure", ["creation", "enqueue"])
def test_ordinary_component_calls_start_no_sender_and_observer_failure_keeps_exception(monkeypatch, failure):
    import ast

    from relax.utils.fault_events import receipts

    ROOT = Path(__file__).resolve().parents[3]
    source = ast.parse((ROOT / "relax/components/actor.py").read_text())
    cls = next(n for n in source.body if isinstance(n, ast.ClassDef) and n.name == "Actor")
    fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "update_weights_fully_async")
    ns = {}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[])), "Actor", "exec"), ns)
    calls = []
    owner = SimpleNamespace(actor_model=SimpleNamespace(update_weights_fully_async=lambda *a, **k: calls.append(k)))
    ns[fn.name](owner)
    assert calls == [{"rollout_only": False, "actor_fwd_only": False}]
    assert not hasattr(owner, "_fault_receipt_sender")

    def broken_sender(*args):
        raise RuntimeError("observer transport failed")

    def broken_worker(*args, **kwargs):
        raise ValueError("original worker error")

    if failure == "creation":
        monkeypatch.setattr(receipts, "ReceiptSender", broken_sender)
    else:
        owner._fault_receipt_sender = SimpleNamespace(emit=broken_sender)
    token = {"run_id": new_id(), "recovery_id": new_id(), "recovery_generation": 1}
    ns[fn.name](owner, recovery_token=token)
    assert len(calls) == 2
    owner.actor_model.update_weights_fully_async = broken_worker
    with pytest.raises(ValueError, match="original worker error"):
        ns[fn.name](owner, recovery_token=token)


@pytest.mark.parametrize("field,value", [("token", {}), ("operation", "unknown"), ("outcome", True)])
def test_health_rpc_rejects_invalid_receipt_without_changing_business_state(runtime, field, value):
    _, recovery, _ = start(runtime)
    token = runtime.token(recovery)
    status = Status(fault_generation=runtime.generation, fault_seed=token)
    status.mark_unhealthy("actor", "business error")
    original = status.get_service_health("actor", include_fault=True)
    request = {"token": token, "operation": "actor_send", "outcome": "succeeded", field: value}
    assert status.record_recovery_receipt(**request) == "invalid"
    assert status.get_service_health("actor", include_fault=True) == original
    assert status._fault_receipts.snapshot()["observed"] == 0
    status.mark_healthy("actor")
    assert status.get_service_health("actor")["healthy"]
    assert status.record_recovery_receipt(token, "actor_send", "succeeded") == "accepted"


def test_malformed_receipt_seed_keeps_valid_fault_source():
    status = Status(fault_generation=0, fault_seed={})
    status.report_error("actor", "business error")
    assert status._fault_observation_error and status._fault_receipts is None
    assert status.get_service_health("actor", include_fault=True)["fault"]["role"] == "actor"


def test_sender_counts_invalid_receipt_without_submitting(monkeypatch):
    from relax.utils.fault_events.receipts import ReceiptSender

    # No worker is needed to test admission; real transport and queue bounds
    # are exercised by test_sender_never_replaces_unconfirmed_request.
    monkeypatch.setattr(threading.Thread, "start", lambda self: None)
    sender = ReceiptSender(object())
    sender.emit({}, "actor_send", "succeeded")
    assert sender.dropped == 1 and sender._queue.empty()
