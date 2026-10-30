# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Bounded source publication and recovery state with a real consumer
thread."""

import threading
from dataclasses import asdict, replace
from enum import Enum

import pytest

from relax.utils.fault_events import runtime as runtime_module
from relax.utils.fault_events.models import MAX_UINT, ROLES, EventDraft, FaultContext, new_id
from relax.utils.fault_events.runtime import FaultEventRuntime


class _ConsumerProbe:
    def __init__(self, store):
        self.append = store._append_committed
        self.condition = threading.Condition()
        self.events = []
        self.paused_id = None
        self.entered = threading.Event()
        self.release = threading.Event()

    def __call__(self, draft):
        if draft.event_id == self.paused_id:
            self.entered.set()
            assert self.release.wait(5), "test did not release the consumer barrier"
        result = self.append(draft)
        with self.condition:
            self.events.append(draft)
            self.condition.notify_all()
        return result

    def wait_ids(self, *identities):
        with self.condition:
            assert self.condition.wait_for(
                lambda: set(identities) <= {draft.event_id for draft in self.events}, timeout=5
            ), "expected stable event identities were not committed"

    def wait_count(self, count):
        with self.condition:
            assert self.condition.wait_for(lambda: len(self.events) >= count, timeout=5)


@pytest.fixture
def make_runtime(monkeypatch):
    created = []

    def make(**kwargs):
        runtime = FaultEventRuntime(**kwargs)
        probe = _ConsumerProbe(runtime.store)
        monkeypatch.setattr(runtime.store, "_append_committed", probe)
        created.append((runtime, probe))
        assert runtime.bind_source() == (0, None)
        return runtime, probe

    yield make
    for runtime, probe in created:
        probe.release.set()
        runtime.close()
        runtime._consumer.join(5)
        assert not runtime._consumer.is_alive()


def _fault(role="actor", generation=0):
    return FaultContext(role, new_id(), generation, new_id(), "reported_error")


def _upgrade(fault):
    return replace(fault, fatal=True, fatal_event_id=new_id())


def _snapshot(runtime, *faults, roles=None, generation=None, observation_error=False):
    return {
        "roles": [fault.role for fault in faults] if roles is None else roles,
        "source_generation": runtime.generation if generation is None else generation,
        "faults": [asdict(fault) for fault in faults],
        "observation_error": observation_error,
    }


def _event(runtime):
    return EventDraft(runtime.run_id, new_id(), new_id(), "actor", 0, "fault_detected", "reported_error")


def _fill_queue(runtime, probe):
    blocked, queued = _event(runtime), _event(runtime)
    probe.paused_id = blocked.event_id
    assert runtime.record(blocked)
    assert probe.entered.wait(5)
    assert runtime.record(queued)
    assert runtime.diagnostics()["queued"] == 1
    return queued.event_id


def _events(runtime, **kwargs):
    return runtime.query(limit=100, **kwargs)["events"]


def test_publication_discovery_and_upgrade_do_not_reappear_after_eviction(make_runtime):
    runtime, probe = make_runtime(capacity=2)
    fault = _fault()
    runtime.observe_snapshot(_snapshot(runtime, fault))
    probe.wait_ids(fault.event_id)
    fillers = [_event(runtime), _event(runtime)]
    for event in fillers:
        assert runtime.record(event)
    probe.wait_ids(*(event.event_id for event in fillers))
    assert runtime.store.status()["latest_sequence"] == 3
    assert runtime.query()["missing_range"] == [1, 1]
    runtime.observe_snapshot(_snapshot(runtime, fault))
    runtime.observe_fault(asdict(fault), role="actor")
    assert runtime.store.status()["latest_sequence"] == 3
    upgraded = _upgrade(fault)
    runtime.observe_fault(asdict(upgraded), role="actor")
    probe.wait_ids(upgraded.fatal_event_id)
    runtime.observe_snapshot(_snapshot(runtime, upgraded))
    runtime.observe_fault(asdict(upgraded), role="actor")
    assert runtime.store.status()["latest_sequence"] == 4
    assert [event["event_type"] for event in _events(runtime, fault_id=fault.fault_id)] == ["fatal_escalated"]
    assert runtime.diagnostics()["known_lost_events"] == 0


def test_publication_queued_old_source_event_still_commits_after_retirement(make_runtime):
    runtime, probe = make_runtime()
    fault = _fault()
    probe.paused_id = fault.event_id
    runtime.observe_snapshot(_snapshot(runtime, fault))
    assert probe.entered.wait(5)
    runtime.retire_source(0)
    runtime.retire_source(0)
    runtime.observe_snapshot(_snapshot(runtime, generation=0))
    runtime.observe_snapshot(_snapshot(runtime, fault, generation=0))
    runtime.observe_fault(asdict(fault), role="actor")
    assert runtime._snapshot == ()
    assert runtime.diagnostics()["known_lost_events"] == 0
    assert runtime.bind_source() == (1, None)
    probe.release.set()
    probe.wait_ids(fault.event_id)
    assert [event["event_id"] for event in _events(runtime)] == [fault.event_id]
    assert runtime.diagnostics()["unknown_reasons"] == ["SOURCE_RETIRED"]


def test_publication_rejected_upgrade_retires_exactly_once(make_runtime):
    runtime, probe = make_runtime(queue_capacity=1)
    fault = _fault()
    runtime.observe_snapshot(_snapshot(runtime, fault))
    probe.wait_ids(fault.event_id)
    queued = _fill_queue(runtime, probe)
    upgraded = _upgrade(fault)
    runtime.observe_snapshot(_snapshot(runtime, upgraded))
    runtime.observe_fault(asdict(upgraded), role="actor")
    runtime.observe_fault(asdict(upgraded), role="actor")
    assert runtime.diagnostics()["admission_rejections"] == 1
    assert runtime.diagnostics()["known_lost_events"] == 0
    runtime.retire_source(0)
    runtime.retire_source(0)
    runtime.observe_snapshot(_snapshot(runtime, upgraded, generation=0))
    assert runtime.diagnostics()["known_lost_events"] == 1
    assert runtime.bind_source() == (1, None)
    runtime.retire_source(0)
    runtime.observe_fault(asdict(upgraded), role="actor")
    assert runtime.diagnostics()["known_lost_events"] == 1
    probe.release.set()
    probe.wait_ids(queued)
    assert [event["event_type"] for event in _events(runtime, fault_id=fault.fault_id)] == ["fault_detected"]


def test_publication_replacement_settles_upgrade_before_any_old_retry(make_runtime):
    runtime, probe = make_runtime(queue_capacity=1)
    fault = _fault()
    runtime.observe_snapshot(_snapshot(runtime, fault))
    probe.wait_ids(fault.event_id)
    queued = _fill_queue(runtime, probe)
    upgraded = _upgrade(fault)
    runtime.observe_snapshot(_snapshot(runtime, upgraded))
    replacement = _fault()
    runtime.observe_snapshot(_snapshot(runtime, replacement))
    assert runtime.diagnostics()["known_lost_events"] == 1
    assert runtime.diagnostics()["admission_rejections"] == 2
    assert runtime._publications[(0, "actor")].fault == replacement
    probe.release.set()
    probe.wait_ids(queued)
    runtime.observe_snapshot(_snapshot(runtime, replacement))
    probe.wait_ids(replacement.event_id)
    runtime.retire_source(0)
    runtime.retire_source(0)
    assert runtime.diagnostics()["known_lost_events"] == 1
    assert not any(event["event_id"] == upgraded.fatal_event_id for event in _events(runtime))


@pytest.mark.parametrize("first_fatal, expected", [(False, 2), (True, 1)])
def test_publication_every_unadmitted_stable_identity_is_settled(make_runtime, first_fatal, expected):
    runtime, probe = make_runtime(queue_capacity=1)
    _fill_queue(runtime, probe)
    fault = _fault()
    fault = replace(fault, reason="fatal_error", fatal=True) if first_fatal else _upgrade(fault)
    runtime.observe_snapshot(_snapshot(runtime, fault))
    runtime.observe_fault(asdict(fault))
    assert runtime.diagnostics()["admission_rejections"] == expected
    assert runtime.diagnostics()["known_lost_events"] == 0
    runtime.retire_source(0)
    runtime.retire_source(0)
    runtime.observe_snapshot(_snapshot(runtime, fault, generation=0))
    assert runtime.diagnostics()["known_lost_events"] == expected


@pytest.mark.parametrize("healthy", [False, True])
def test_publication_rejected_upgrade_retries_in_next_existing_snapshot(make_runtime, healthy):
    runtime, probe = make_runtime(queue_capacity=1)
    fault = _fault()
    runtime.observe_snapshot(_snapshot(runtime, fault))
    probe.wait_ids(fault.event_id)
    queued = _fill_queue(runtime, probe)
    upgraded = _upgrade(fault)
    runtime.observe_snapshot(_snapshot(runtime, upgraded))
    assert runtime.diagnostics()["admission_rejections"] == 1
    probe.release.set()
    probe.wait_ids(queued)
    runtime.observe_snapshot(_snapshot(runtime, *(() if healthy else (upgraded,))))
    probe.wait_ids(upgraded.fatal_event_id)
    if healthy:
        assert runtime._snapshot == ()
        assert runtime.start_recovery("actor", "local", 7) is None
    runtime.retire_source(0)
    assert runtime.diagnostics()["known_lost_events"] == 0
    assert runtime.diagnostics()["admission_rejections"] == 1
    assert sum(event["event_id"] == upgraded.fatal_event_id for event in _events(runtime)) == 1


def test_publication_same_round_rejections_are_not_retried_but_new_upgrade_is(make_runtime):
    runtime, probe = make_runtime(queue_capacity=1)
    _fill_queue(runtime, probe)
    fault = _fault()
    runtime.observe_snapshot(_snapshot(runtime, fault))
    for _ in range(5):
        runtime.observe_fault(asdict(fault), role="actor")
    assert runtime.diagnostics()["admission_rejections"] == 1
    upgraded = _upgrade(fault)
    runtime.observe_fault(asdict(upgraded), role="actor")
    runtime.observe_fault(asdict(upgraded), role="actor")
    assert runtime.diagnostics()["admission_rejections"] == 2
    runtime.observe_snapshot(_snapshot(runtime, upgraded))
    assert runtime.diagnostics()["admission_rejections"] == 4


@pytest.mark.parametrize("change", ["discovery", "reason", "remove_upgrade", "change_upgrade", "clear_fatal"])
def test_publication_identity_changes_preserve_slot_and_remove_candidate(make_runtime, change):
    runtime, probe = make_runtime()
    original = _fault()
    if change in ("remove_upgrade", "change_upgrade", "clear_fatal"):
        original = _upgrade(original)
    runtime.observe_snapshot(_snapshot(runtime, original))
    probe.wait_ids(original.event_id)
    if change == "discovery":
        invalid = replace(original, event_id=new_id())
    elif change == "reason":
        invalid = replace(original, reason="heartbeat_timeout")
    elif change == "change_upgrade":
        invalid = replace(original, fatal_event_id=new_id())
    elif change == "remove_upgrade":
        invalid = replace(original, fatal_event_id=None)
    else:
        invalid = replace(original, fatal=False, fatal_event_id=None)
    runtime.observe_snapshot(_snapshot(runtime, invalid))
    assert runtime._publications[(0, "actor")].fault == original
    assert runtime._snapshot == ()
    assert runtime.start_recovery("actor", "local", 7) is None
    assert "OBSERVATION_ERROR" in runtime.diagnostics()["unknown_reasons"]


def test_publication_invalid_upgrade_removal_keeps_observed_upgrade_for_retry(make_runtime):
    runtime, probe = make_runtime(queue_capacity=1)
    fault = _fault()
    runtime.observe_snapshot(_snapshot(runtime, fault))
    probe.wait_ids(fault.event_id)
    queued = _fill_queue(runtime, probe)
    upgraded = _upgrade(fault)
    runtime.observe_snapshot(_snapshot(runtime, upgraded))
    runtime.observe_fault(asdict(fault), role="actor")
    assert runtime._snapshot == ()
    assert runtime._publications[(0, "actor")].fault == upgraded
    probe.release.set()
    probe.wait_ids(queued)
    runtime.observe_snapshot(_snapshot(runtime))
    probe.wait_ids(upgraded.fatal_event_id)
    assert runtime._snapshot == ()
    assert runtime.diagnostics()["known_lost_events"] == 0


@pytest.mark.parametrize(
    "changes",
    [
        {"fatal": False, "fatal_event_id": "1" * 32},
        {"fatal": True, "fatal_event_id": None},
        {"fatal": False, "reason": "fatal_error"},
        {"fatal": True, "reason": "fatal_error", "fatal_event_id": "1" * 32},
        {"source_generation": True},
        {"extra_field": "private payload"},
    ],
)
def test_publication_invalid_payload_never_replaces_identity_or_enters_candidates(make_runtime, changes):
    runtime, _ = make_runtime()
    original = _fault()
    runtime.observe_snapshot(_snapshot(runtime, original))
    payload = {**asdict(original), **changes}
    runtime.observe_fault(payload, role="actor")
    assert runtime._publications[(0, "actor")].fault == original
    assert runtime._snapshot == ()
    assert runtime.diagnostics()["unknown_reasons"] == ["OBSERVATION_ERROR"]


def test_publication_invalid_details_clear_only_expected_role_and_keep_retry_slot(make_runtime):
    runtime, _ = make_runtime()
    actor, critic = _fault(), _fault("critic")
    runtime.observe_snapshot(_snapshot(runtime, actor, critic))
    runtime.observe_fault(None, role="actor")
    assert runtime._snapshot == (critic,)
    assert runtime._publications[(0, "actor")].fault == actor
    runtime.observe_fault(asdict(critic), role="actor")
    assert runtime._snapshot == (critic,)
    recovery = runtime.start_recovery("critic", "global", 63)
    assert recovery is not None and recovery.members == (critic,)
    runtime.finish(recovery, "recovery_failed", "restart_exception")


def test_publication_safe_string_enum_roles_and_stale_path_addition(make_runtime):
    class Role(str, Enum):
        ACTOR = "actor"
        CRITIC = "critic"

    runtime, _ = make_runtime()
    actor, critic = _fault(), _fault("critic")
    runtime.observe_snapshot(_snapshot(runtime, actor, roles=[Role.ACTOR]))
    runtime.observe_fault(asdict(critic), role=Role.CRITIC)
    runtime.observe_fault(asdict(actor), role=Role.ACTOR)
    assert runtime._snapshot == (actor, critic)
    recovery = runtime.start_recovery(Role.ACTOR, "global", 63)
    assert recovery is not None and recovery.members == (actor, critic)
    runtime.finish(recovery, "recovery_failed", "restart_exception")


def test_publication_old_or_retired_empty_snapshot_never_changes_new_state(make_runtime):
    runtime, _ = make_runtime()
    runtime.retire_source(0)
    retired = _snapshot(runtime, generation=0)
    retired["observation_error"] = "untrusted invalid field"
    runtime.observe_snapshot(retired)
    assert runtime.diagnostics()["unknown_reasons"] == ["SOURCE_RETIRED"]
    assert runtime.bind_source() == (1, None)
    current = _fault(generation=1)
    runtime.observe_snapshot(_snapshot(runtime, current))
    publication = runtime._publications[(1, "actor")]
    runtime.observe_snapshot(retired)
    runtime.retire_source(0)
    assert runtime._snapshot == (current,)
    assert runtime._publications[(1, "actor")] is publication
    assert runtime.diagnostics()["unknown_reasons"] == ["SOURCE_RETIRED"]


@pytest.mark.parametrize("invalid", [{}, {"roles": ["actor", "actor"]}, {"faults": []}, {"observation_error": 1}])
def test_publication_invalid_snapshot_cannot_reuse_prior_candidates(make_runtime, invalid):
    runtime, _ = make_runtime()
    fault = _fault()
    runtime.observe_snapshot(_snapshot(runtime, fault))
    payload = invalid if not invalid or "faults" in invalid else {**_snapshot(runtime, fault), **invalid}
    runtime.observe_snapshot(payload)
    assert runtime._snapshot == ()
    assert runtime.start_recovery("actor", "local", 7) is None
    assert runtime._publications[(0, "actor")].fault == fault


def test_publication_lock_failure_fences_stale_candidates_until_next_snapshot(make_runtime):
    runtime, _ = make_runtime()
    fault = _fault()
    runtime.observe_snapshot(_snapshot(runtime, fault))
    with runtime._lock:
        runtime.observe_fault(None, role="actor")
    assert runtime.start_recovery("actor", "local", 7) is None
    runtime.observe_snapshot(_snapshot(runtime, fault))
    recovery = runtime.start_recovery("actor", "local", 7)
    assert recovery is not None
    runtime.finish(recovery, "recovery_failed", "restart_exception")


def test_publication_retirement_lock_timeout_still_fences_the_old_epoch(make_runtime):
    runtime, probe = make_runtime(queue_capacity=1)
    fault = _fault()
    runtime.observe_snapshot(_snapshot(runtime, fault))
    probe.wait_ids(fault.event_id)
    queued = _fill_queue(runtime, probe)
    upgraded = _upgrade(fault)
    runtime.observe_snapshot(_snapshot(runtime, upgraded))
    old_epoch = runtime._source_epoch
    with runtime._lock:
        runtime.retire_source(0)
    assert old_epoch[1].is_set()
    assert runtime.diagnostics()["known_lost_events"] == 0
    rejections = runtime.diagnostics()["admission_rejections"]
    runtime.observe_snapshot(_snapshot(runtime, generation=0))
    runtime.observe_snapshot(_snapshot(runtime, upgraded, generation=0))
    runtime.observe_fault(asdict(upgraded), role="actor")
    assert runtime.diagnostics()["admission_rejections"] == rejections
    assert runtime.start_recovery("actor", "local", 7) is None
    assert runtime.bind_source() == (1, None)
    assert runtime.diagnostics()["known_lost_events"] == 1
    old_epoch[1].set()  # An old caller cannot fence the newly bound receiver.
    assert not runtime._source_epoch[1].is_set()
    runtime.retire_source(0)
    assert runtime.diagnostics()["known_lost_events"] == 1
    probe.release.set()
    probe.wait_ids(queued)


def test_recovery_own_receiver_survives_retirement_lock_timeout(make_runtime):
    runtime, _ = make_runtime()
    fault = _fault()
    runtime.observe_snapshot(_snapshot(runtime, fault))
    recovery = runtime.start_recovery("actor", "global", 63)
    with runtime._lock:
        runtime.retire_source(0)
    assert recovery.outcome is None
    assert runtime.bind_source() == (1, None)
    assert recovery.outcome is None and recovery.receiver_generation == 1
    runtime.finish(recovery, "recovery_failed", "restart_exception")


def test_recovery_local_members_are_not_adopted_again_by_global_same_round(make_runtime):
    runtime, probe = make_runtime()
    local_fault, global_fault = _fault("advantages"), _fault("actor")
    runtime.observe_snapshot(_snapshot(runtime, local_fault, global_fault))
    local = runtime.start_recovery("advantages", "local", 7)
    assert local is not None and local.members == (local_fault,)
    runtime.evidence(local, observed=1)
    runtime.evidence(local, observed=2)
    assert local.outcome is None
    runtime.evidence(local, observed=4, handoff=True)
    assert local.outcome == "recovery_succeeded"
    global_ = runtime.start_recovery("actor", "global", 63)
    assert global_ is not None and global_.members == (global_fault,)
    runtime.finish(global_, "recovery_failed", "restart_exception")
    probe.wait_count(7)
    terminal = [
        event for event in _events(runtime) if event["event_type"] in ("recovery_succeeded", "recovery_failed")
    ]
    assert [(event["fault_id"], event["event_type"]) for event in terminal] == [
        (local_fault.fault_id, "recovery_succeeded"),
        (global_fault.fault_id, "recovery_failed"),
    ]
    assert runtime.diagnostics()["active_recoveries"] == 0


def test_recovery_step_failure_is_single_terminal_and_later_round_can_retry(make_runtime):
    runtime, probe = make_runtime()
    fault = _fault("advantages")
    snapshot = _snapshot(runtime, fault)
    runtime.observe_snapshot(snapshot)
    first = runtime.start_recovery("advantages", "local", 7)
    runtime.evidence(first, observed=5, failed=2, handoff=True, reason="step_restore_failed")
    assert first.outcome == "recovery_failed"
    runtime.finish(first, "recovery_failed", "restart_exception")
    assert runtime.start_recovery("advantages", "local", 7) is None
    runtime.observe_snapshot(snapshot)
    second = runtime.start_recovery("advantages", "local", 7)
    assert second is not None and second.recovery_id != first.recovery_id
    runtime.evidence(second, observed=7, handoff=True)
    runtime.finish(second, "recovery_failed", "restart_exception")
    probe.wait_count(7)
    events = _events(runtime)
    assert sum(event["event_type"] == "fault_detected" for event in events) == 1
    failures = [event for event in events if event["event_type"] == "recovery_failed"]
    assert len(failures) == 1
    assert failures[0]["reason_code"] == "step_restore_failed"
    assert (failures[0]["expected_evidence"], failures[0]["observed_evidence"], failures[0]["failed_evidence"]) == (
        7,
        5,
        2,
    )
    assert sum(event["event_type"] == "recovery_succeeded" for event in events) == 1


def test_local_completion_uses_collected_evidence_and_emits_one_terminal(make_runtime):
    runtime, probe = make_runtime()
    fault = _fault()
    runtime.observe_snapshot(_snapshot(runtime, fault))
    recovery = runtime.start_recovery("actor", "local", 7)
    runtime.evidence(recovery, observed=7)
    assert recovery.outcome is None and not recovery.handoff
    runtime.complete_local(recovery)
    runtime.complete_local(recovery)
    assert recovery.outcome == "recovery_succeeded"
    assert runtime.diagnostics()["active_recoveries"] == 0
    probe.wait_count(4)
    assert [event["event_type"] for event in _events(runtime)] == [
        "fault_detected",
        "handling_started",
        "recovery_handoff",
        "recovery_succeeded",
    ]


def test_local_completion_with_missing_evidence_releases_without_inventing_terminal(make_runtime):
    runtime, probe = make_runtime()
    fault = _fault()
    snapshot = _snapshot(runtime, fault)
    runtime.observe_snapshot(snapshot)
    recovery = runtime.start_recovery("actor", "local", 7)
    runtime.evidence(recovery, observed=1)
    runtime.complete_local(recovery)
    assert recovery.outcome is None and recovery.handoff
    assert runtime.diagnostics()["active_recoveries"] == 0
    assert "OUTCOME_UNOBSERVED" in runtime.diagnostics()["unknown_reasons"]
    assert runtime.start_recovery("actor", "local", 7) is None
    probe.wait_count(3)
    assert [event["event_type"] for event in _events(runtime)] == [
        "fault_detected",
        "handling_started",
        "recovery_handoff",
    ]
    runtime.observe_snapshot(snapshot)
    next_attempt = runtime.start_recovery("actor", "local", 7)
    assert next_attempt is not None and next_attempt.recovery_id != recovery.recovery_id
    runtime.complete_local(next_attempt, failed=True)


@pytest.mark.parametrize("step_already_failed", [False, True])
def test_local_completion_exception_preserves_prior_step_failure(make_runtime, step_already_failed):
    runtime, probe = make_runtime()
    fault = _fault()
    runtime.observe_snapshot(_snapshot(runtime, fault))
    recovery = runtime.start_recovery("actor", "local", 7)
    if step_already_failed:
        runtime.evidence(recovery, observed=1, failed=2, reason="step_restore_failed")
    runtime.complete_local(recovery, failed=True)
    runtime.complete_local(recovery)
    probe.wait_count(3)
    assert recovery.outcome == "recovery_failed"
    events = _events(runtime)
    assert [event["event_type"] for event in events] == ["fault_detected", "handling_started", "recovery_failed"]
    assert events[-1]["reason_code"] == ("step_restore_failed" if step_already_failed else "restart_exception")


def test_local_completion_lock_timeout_is_settled_before_next_round(make_runtime, monkeypatch):
    runtime, probe = make_runtime()
    fault = _fault()
    snapshot = _snapshot(runtime, fault)
    runtime.observe_snapshot(snapshot)
    recovery = runtime.start_recovery("actor", "local", 7)
    runtime.evidence(recovery, observed=7)
    reserved_ids = set(recovery._local_event_ids)
    with runtime._lock:
        runtime.complete_local(recovery)
    assert recovery.outcome is None
    assert runtime.diagnostics()["active_recoveries"] == 1

    def unexpected_random_source():
        raise AssertionError("deferred local completion must use its reserved event identities")

    with monkeypatch.context() as deferred:
        deferred.setattr(runtime_module, "new_id", unexpected_random_source)
        runtime.observe_snapshot(snapshot)
    assert recovery.outcome == "recovery_succeeded"
    probe.wait_count(4)
    assert {event["event_id"] for event in _events(runtime)[-2:]} <= reserved_ids
    next_attempt = runtime.start_recovery("actor", "local", 7)
    assert next_attempt is not None and next_attempt.recovery_id != recovery.recovery_id
    runtime.complete_local(next_attempt, failed=True)


@pytest.mark.parametrize("failed", [False, True])
def test_local_completion_is_drained_before_new_context_capacity_check(make_runtime, failed):
    runtime, _ = make_runtime()
    faults = [_fault(role) for role in ("actor", "critic", "advantages")]
    runtime.observe_snapshot(_snapshot(runtime, *faults))
    first = runtime.start_recovery("actor", "local", 7)
    second = runtime.start_recovery("critic", "local", 7)
    with runtime._lock:
        runtime.complete_local(first, failed=failed)
    assert runtime.diagnostics()["active_recoveries"] == 2
    third = runtime.start_recovery("advantages", "local", 7)
    assert third is not None
    assert first.outcome == ("recovery_failed" if failed else None)
    assert first.recovery_id not in runtime._recoveries
    assert runtime.diagnostics()["active_recoveries"] == 2
    runtime.complete_local(second, failed=True)
    runtime.complete_local(third, failed=True)


@pytest.mark.parametrize("transition", ["retire", "bind"])
def test_local_completion_is_settled_before_source_retirement(make_runtime, transition):
    runtime, probe = make_runtime()
    actor, critic = _fault(), _fault("critic")
    runtime.observe_snapshot(_snapshot(runtime, actor, critic))
    local = runtime.start_recovery("actor", "local", 7)
    global_ = runtime.start_recovery("critic", "global", 63)
    runtime.evidence(local, observed=7)
    with runtime._lock:
        runtime.complete_local(local)
    if transition == "retire":
        runtime.retire_source(0)
    else:
        assert runtime.bind_source() == (1, None)
    assert local.outcome == "recovery_succeeded"
    assert global_.outcome is None
    runtime.finish(global_, "recovery_failed", "restart_exception")
    probe.wait_count(7)
    assert not any(event["event_type"] == "recovery_interrupted" for event in _events(runtime))


def test_local_completion_rejects_global_and_foreign_attempt_objects(make_runtime):
    runtime, _ = make_runtime()
    actor, critic = _fault(), _fault("critic")
    runtime.observe_snapshot(_snapshot(runtime, actor, critic))
    local = runtime.start_recovery("actor", "local", 7)
    global_ = runtime.start_recovery("critic", "global", 63)
    foreign = replace(local)
    runtime.complete_local(foreign, failed=True)
    runtime.complete_local(global_)
    runtime.observe_snapshot(_snapshot(runtime, actor, critic))
    assert local.outcome is None and global_.outcome is None
    assert runtime.diagnostics()["active_recoveries"] == 2
    runtime.complete_local(local, failed=True)
    runtime.finish(global_, "recovery_failed", "restart_exception")


def test_recovery_active_fault_cannot_reenter_on_a_new_round(make_runtime):
    runtime, _ = make_runtime()
    fault = _fault()
    snapshot = _snapshot(runtime, fault)
    runtime.observe_snapshot(snapshot)
    first = runtime.start_recovery("actor", "local", 7)
    runtime.observe_snapshot(snapshot)
    assert runtime.start_recovery("actor", "local", 7) is None
    assert runtime.start_recovery("actor", "global", 63) is None
    assert runtime._recoveries == {first.recovery_id: first}
    runtime.finish(first, "recovery_failed", "restart_exception")


def test_recovery_freezes_identity_and_ignores_later_replacement(make_runtime):
    runtime, probe = make_runtime()
    original = _fault()
    runtime.observe_snapshot(_snapshot(runtime, original))
    recovery = runtime.start_recovery("actor", "local", 7)
    replacement = _fault()
    runtime.observe_snapshot(_snapshot(runtime, replacement))
    runtime.finish(recovery, "recovery_failed", "restart_exception")
    probe.wait_count(4)
    terminal = next(event for event in _events(runtime) if event["event_type"] == "recovery_failed")
    assert terminal["fault_id"] == original.fault_id
    assert runtime._snapshot == (replacement,)
    assert recovery.members == (original,)


def test_recovery_global_waits_for_real_receiver_retirement(make_runtime):
    runtime, probe = make_runtime()
    first_fault = _fault()
    runtime.observe_snapshot(_snapshot(runtime, first_fault))
    first = runtime.start_recovery("actor", "global", 63)
    assert first.receiver_generation == 1
    runtime.retire_source(0)
    assert first.outcome is None
    runtime.retire_source(0)
    assert runtime.bind_source() == (1, None)
    assert first.outcome is None
    second_fault = _fault(generation=1)
    runtime.observe_snapshot(_snapshot(runtime, second_fault))
    second = runtime.start_recovery("actor", "global", 63)
    assert second.receiver_generation == 2 and first.outcome is None
    runtime.retire_source(1)
    assert first.outcome == "recovery_interrupted"
    assert second.outcome is None
    runtime.retire_source(1)
    runtime.finish(second, "recovery_failed", "restart_exception")
    probe.wait_count(6)
    terminal = [
        event for event in _events(runtime) if event["event_type"] in ("recovery_interrupted", "recovery_failed")
    ]
    assert [(event["fault_id"], event["source_generation"]) for event in terminal] == [
        (first_fault.fault_id, 0),
        (second_fault.fault_id, 1),
    ]
    assert runtime.diagnostics()["active_recoveries"] == 0


def test_recovery_global_cannot_claim_handoff_or_success_in_this_batch(make_runtime):
    runtime, probe = make_runtime()
    fault = _fault()
    runtime.observe_snapshot(_snapshot(runtime, fault))
    recovery = runtime.start_recovery("actor", "global", 63)
    runtime.evidence(recovery, observed=63, handoff=True)
    runtime.finish(recovery, "recovery_succeeded", None)
    assert recovery.outcome is None and not recovery.handoff and recovery.expected == 63
    runtime.finish(recovery, "recovery_failed", "restart_exception")
    probe.wait_count(3)
    assert [event["event_type"] for event in _events(runtime)] == [
        "fault_detected",
        "handling_started",
        "recovery_failed",
    ]


def test_recovery_conflicting_evidence_cannot_reverse_a_terminal(make_runtime):
    runtime, probe = make_runtime()
    fault = _fault()
    runtime.observe_snapshot(_snapshot(runtime, fault))
    recovery = runtime.start_recovery("actor", "local", 7)
    runtime.evidence(recovery, observed=7, handoff=True)
    probe.wait_count(4)
    before = _events(runtime)[-1]
    runtime.evidence(recovery, failed=2)
    runtime.finish(recovery, "recovery_failed", "step_restore_failed")
    assert recovery.outcome == "recovery_succeeded"
    assert _events(runtime)[-1] == before
    assert "RECEIPT_CONFLICT" in runtime.diagnostics()["unknown_reasons"]


def test_recovery_budget_projects_all_members_and_releases_context_once(make_runtime):
    runtime, probe = make_runtime()
    actor, critic = _fault(), _fault("critic")
    runtime.observe_snapshot(_snapshot(runtime, actor, critic))
    recovery = runtime.start_recovery("actor", "global", 63)
    runtime.special("actor", "termination_requested", "restart_limit", recovery)
    runtime.special("actor", "termination_requested", "restart_limit", recovery)
    runtime.finish(recovery, "recovery_failed", "restart_exception")
    assert recovery.outcome == "termination_requested"
    assert runtime.diagnostics()["active_recoveries"] == 0
    probe.wait_count(6)
    terminals = [event for event in _events(runtime) if event["event_type"] == "termination_requested"]
    assert {event["fault_id"] for event in terminals} == {actor.fault_id, critic.fault_id}
    assert {event["recovery_id"] for event in terminals} == {recovery.recovery_id}
    assert all(event["expected_evidence"] == 63 and event["recovery_mode"] == "global" for event in terminals)


@pytest.mark.parametrize("fatal", [False, True])
def test_recovery_global_freezes_bounded_nonfatal_members(make_runtime, fatal):
    runtime, probe = make_runtime()
    faults = [_fault(role) for role in sorted(ROLES)]
    if fatal:
        faults[0] = replace(faults[0], reason="fatal_error", fatal=True)
    runtime.observe_snapshot(_snapshot(runtime, *faults))
    expected_members = tuple(fault for fault in faults if not fault.fatal)
    recovery = runtime.start_recovery(expected_members[0].role, "global", 63)
    assert recovery is not None and recovery.members == expected_members
    assert len(recovery.members) == 9 - int(fatal)
    runtime.retire_source(0)
    runtime.finish(recovery, "recovery_failed", "restart_exception")
    probe.wait_count(9 + 2 * len(expected_members))
    terminals = [event for event in _events(runtime) if event["event_type"] == "recovery_failed"]
    assert {event["fault_id"] for event in terminals} == {fault.fault_id for fault in expected_members}
    assert all(event["expected_evidence"] == 63 and event["source_generation"] == 0 for event in terminals)


def test_recovery_terminal_selection_does_not_roll_back_when_queue_is_full(make_runtime):
    runtime, probe = make_runtime(queue_capacity=1)
    _fill_queue(runtime, probe)
    fault = _fault()
    runtime.observe_snapshot(_snapshot(runtime, fault))
    recovery = runtime.start_recovery("actor", "local", 7)
    runtime.evidence(recovery, observed=7, handoff=True)
    assert recovery.outcome == "recovery_succeeded"
    assert runtime.diagnostics()["active_recoveries"] == 0
    assert runtime.diagnostics()["known_lost_events"] == 3  # started, handoff, terminal
    runtime.finish(recovery, "recovery_failed", "restart_exception")
    assert runtime.diagnostics()["known_lost_events"] == 3
    runtime.retire_source(0)
    assert runtime.diagnostics()["known_lost_events"] == 4  # unadmitted D is settled separately
    assert runtime.diagnostics()["admission_rejections"] == 4


def test_recovery_skip_is_per_round_and_fatal_is_per_fault(make_runtime):
    runtime, probe = make_runtime()
    skipped = _fault("advantages")
    fatal = replace(_fault(), reason="fatal_error", fatal=True)
    snapshot = _snapshot(runtime, skipped, fatal)
    runtime.observe_snapshot(snapshot)
    runtime.special("advantages", "handling_skipped", "role_unregistered")
    runtime.special("advantages", "handling_skipped", "role_unregistered")
    runtime.special("actor", "termination_requested", "fatal_error")
    runtime.special("actor", "termination_requested", "fatal_error")
    runtime.observe_snapshot(snapshot)
    runtime.special("advantages", "handling_skipped", "role_unregistered")
    runtime.special("actor", "termination_requested", "fatal_error")
    probe.wait_count(5)
    events = _events(runtime)
    skips = [event for event in events if event["event_type"] == "handling_skipped"]
    assert len(skips) == 2 and skips[0]["recovery_id"] != skips[1]["recovery_id"]
    terminals = [event for event in events if event["event_type"] == "termination_requested"]
    assert len(terminals) == 1 and terminals[0]["recovery_id"] is None
    assert terminals[0]["expected_evidence"] == 0


def test_recovery_two_context_bound_preserves_existing_members(make_runtime):
    runtime, _ = make_runtime()
    faults = [_fault(role) for role in ("actor", "critic", "advantages")]
    runtime.observe_snapshot(_snapshot(runtime, *faults))
    first = runtime.start_recovery("actor", "local", 7)
    second = runtime.start_recovery("critic", "local", 7)
    assert runtime.start_recovery("advantages", "local", 7) is None
    assert runtime.diagnostics()["active_recoveries"] == 2
    assert "CONTEXT_CAPACITY" in runtime.diagnostics()["unknown_reasons"]
    runtime.finish(first, "recovery_failed", "restart_exception")
    third = runtime.start_recovery("advantages", "local", 7)
    assert third is not None
    runtime.finish(second, "recovery_failed", "restart_exception")
    runtime.finish(third, "recovery_failed", "restart_exception")


def test_publication_and_snapshot_state_stay_bounded_across_generations(make_runtime):
    runtime, probe = make_runtime()
    for generation in range(16):
        if generation:
            assert runtime.bind_source() == (generation, None)
        faults = [_fault(role, generation) for role in sorted(ROLES)]
        runtime.observe_snapshot(_snapshot(runtime, *faults))
        assert len(runtime._publications) == 9
        assert runtime.diagnostics()["publications"] <= 18
        assert runtime.diagnostics()["snapshot_size"] <= 9
    probe.wait_ids(*(fault.event_id for fault in faults))
    assert runtime.diagnostics()["known_lost_events"] == 0


def test_publication_capacity_failure_disables_only_that_source(make_runtime, monkeypatch):
    runtime, _ = make_runtime()
    monkeypatch.setattr(runtime_module, "_MAX_PUBLICATIONS", 2)
    faults = [_fault(role) for role in ("actor", "critic", "advantages")]
    runtime.observe_snapshot(_snapshot(runtime, *faults))
    assert len(runtime._publications) == 2 and runtime._snapshot == ()
    runtime.observe_fault(asdict(_fault("rollout")), role="rollout")
    assert len(runtime._publications) == 2
    assert "CONTEXT_CAPACITY" in runtime.diagnostics()["unknown_reasons"]
    assert runtime.bind_source() == (1, None)
    new_fault = _fault(generation=1)
    runtime.observe_snapshot(_snapshot(runtime, new_fault))
    assert runtime._snapshot == (new_fault,)


def test_publication_generation_does_not_wrap_after_uint64_limit(make_runtime):
    runtime, probe = make_runtime()
    with runtime._lock:
        runtime.generation = MAX_UINT - 1
    assert runtime.bind_source() == (MAX_UINT, None)
    fault = _fault(generation=MAX_UINT)
    runtime.observe_snapshot(_snapshot(runtime, fault))
    probe.wait_ids(fault.event_id)
    assert runtime.start_recovery("actor", "global", 63) is None
    assert runtime.bind_source() == (None, None)
    assert runtime.bind_source() == (None, None)
    runtime.observe_snapshot(_snapshot(runtime, fault))
    assert runtime.generation == MAX_UINT
    assert runtime._snapshot == ()
    assert runtime.store.status()["latest_sequence"] == 1
