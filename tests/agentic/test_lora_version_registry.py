# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Fault and race tests for the immutable LoRA version registry (Task 7).

The registry is the single linearization point for "which version does a new
Session bind" and "which retired version may still be reclaimed"; every case
here is one of the RFC §21 fault scenarios, expressed directly against the
state machine (no Ray, no engines).
"""

import pytest

from relax.agentic.session.lora_version import (
    LoRAVersionError,
    LoRAVersionRegistry,
    VersionState,
)
from tests.agentic.lora_helpers import commit_ready


DIGEST_A = "a" * 64
DIGEST_B = "b" * 64
DIGEST_C = "c" * 64


@pytest.fixture()
def registry() -> LoRAVersionRegistry:
    return LoRAVersionRegistry(logical_capacity=2, deployment_epoch="epoch0")


def publish(registry: LoRAVersionRegistry, digest: str, *, version_id: int):
    """Reserve + commit a version, the way the publisher does."""

    publication = registry.allocate(digest, version_id=version_id)
    commit_ready(registry, publication.version_id, publication.attempt_id)
    return publication


class TestPublicationIdentity:
    def test_same_version_replay_is_an_exact_no_op(self, registry):
        first = publish(registry, DIGEST_A, version_id=1)
        replay = registry.allocate(DIGEST_A, version_id=1)
        assert replay.no_op and replay.version_id == first.version_id
        assert replay.lora_name == first.lora_name
        assert registry.status().capacity_owning == 1

    def test_duplicate_publication_while_loading_reuses_the_attempt(self, registry):
        first = registry.allocate(DIGEST_A, version_id=1)
        duplicate = registry.allocate(DIGEST_A, version_id=1)
        assert duplicate.version_id == first.version_id
        assert duplicate.attempt_id == first.attempt_id
        assert not duplicate.no_op
        assert registry.status().capacity_owning == 1

    def test_same_version_with_different_digest_conflicts(self, registry):
        first = registry.allocate(DIGEST_A, version_id=1)
        registry.mark_retryable_failure(first.version_id, first.attempt_id)
        with pytest.raises(LoRAVersionError) as error:
            registry.retry_publication(first.version_id, DIGEST_B)
        assert error.value.code == "VERSION_CONFLICT"

    def test_names_are_immutable_and_epoch_scoped(self, registry):
        first = registry.allocate(DIGEST_A, version_id=1)
        assert first.lora_name == f"relax_policy_lora@epoch0-1-{'a' * 16}"
        other = LoRAVersionRegistry(deployment_epoch="epoch1").allocate(DIGEST_A, version_id=1)
        assert other.lora_name != first.lora_name

    def test_status_reports_every_version_and_binding(self, registry):
        publish(registry, DIGEST_A, version_id=1)
        registry.bind_latest("s1")
        status = registry.status()
        assert status.default_version == 1
        assert status.default_revision == 1
        assert status.capacity_owning == 1
        assert status.session_bindings == {"epoch0:s1": 1}
        assert status.versions[1]["state"] == VersionState.PUBLISHED.value


class TestFleetCommit:
    def test_loading_candidate_does_not_move_the_default(self, registry):
        """F1: only one engine has READY_LOCAL, so B was never committed."""

        a = publish(registry, DIGEST_A, version_id=1)
        registry.bind_latest("s_old")
        b = registry.allocate(DIGEST_B, version_id=2)
        assert registry.bind_latest("s_mid").lora_name == a.lora_name
        assert registry.status().default_version == a.version_id

        commit_ready(registry, b.version_id, b.attempt_id)
        assert registry.bind_latest("s_new").lora_name == b.lora_name
        assert registry.status().versions[a.version_id]["state"] == VersionState.RETIRED.value
        assert registry.bind_latest("s_old").lora_name == a.lora_name

    def test_commit_replay_is_idempotent(self, registry):
        a = publish(registry, DIGEST_A, version_id=1)
        b = registry.allocate(DIGEST_B, version_id=2)
        commit_ready(registry, b.version_id, b.attempt_id)
        revision = registry.status().default_revision
        again = commit_ready(registry, b.version_id, b.attempt_id)
        assert again.version_id == b.version_id
        assert registry.status().default_revision == revision
        assert registry.status().versions[a.version_id]["state"] == VersionState.RETIRED.value

    def test_late_commit_of_a_retired_version_never_makes_it_default_again(self, registry):
        a = publish(registry, DIGEST_A, version_id=1)
        publish(registry, DIGEST_B, version_id=2)
        late = commit_ready(registry, a.version_id, a.attempt_id)
        assert late.version_id == a.version_id
        assert registry.status().default_version != a.version_id

    def test_stale_attempt_replies_are_rejected(self, registry):
        """13.8: late replies from an old attempt must not mutate state."""

        a = registry.allocate(DIGEST_A, version_id=1)
        registry.mark_retryable_failure(a.version_id, a.attempt_id)
        registry.retry_publication(a.version_id, DIGEST_A)  # attempt 2 is now current
        with pytest.raises(LoRAVersionError) as error:
            commit_ready(registry, a.version_id, a.attempt_id)
        assert error.value.code == "ATTEMPT_CONFLICT"
        with pytest.raises(LoRAVersionError) as error:
            registry.mark_retryable_failure(a.version_id, a.attempt_id)
        assert error.value.code == "ATTEMPT_CONFLICT"
        assert registry.status().versions[a.version_id]["state"] == VersionState.LOADING.value


class TestCapacity:
    def test_third_version_is_refused_before_any_transport(self, registry):
        a = publish(registry, DIGEST_A, version_id=1)
        b = registry.allocate(DIGEST_B, version_id=2)
        with pytest.raises(LoRAVersionError) as error:
            registry.allocate(DIGEST_C, version_id=3)
        assert error.value.code == "CAPACITY_ERROR"
        assert registry.status().capacity_owning == 2
        assert registry.status().versions[a.version_id]["state"] == VersionState.PUBLISHED.value
        assert registry.status().versions[b.version_id]["state"] == VersionState.LOADING.value

    def test_retired_version_keeps_its_slot_while_a_session_is_bound(self, registry):
        a = publish(registry, DIGEST_A, version_id=1)
        registry.bind_latest("s_old")
        publish(registry, DIGEST_B, version_id=2)
        with pytest.raises(LoRAVersionError) as error:
            registry.allocate(DIGEST_C, version_id=3)
        assert error.value.code == "CAPACITY_ERROR"
        assert registry.status().versions[a.version_id]["state"] == VersionState.RETIRED.value

    def test_retryable_failure_releases_capacity_and_retry_re_admits(self, registry):
        a = registry.allocate(DIGEST_A, version_id=1)
        registry.mark_retryable_failure(a.version_id, a.attempt_id)
        assert registry.status().capacity_owning == 0

        retry = registry.retry_publication(a.version_id, DIGEST_A)
        assert retry.version_id == a.version_id
        assert retry.attempt_id == a.attempt_id + 1
        assert registry.status().versions[a.version_id]["state"] == VersionState.LOADING.value
        assert registry.status().capacity_owning == 1

    def test_retry_is_refused_once_the_slots_are_gone(self, registry):
        a = registry.allocate(DIGEST_A, version_id=1)
        registry.mark_retryable_failure(a.version_id, a.attempt_id)
        publish(registry, DIGEST_B, version_id=2)
        publish(registry, DIGEST_C, version_id=3)
        with pytest.raises(LoRAVersionError) as error:
            registry.retry_publication(a.version_id, DIGEST_A)
        assert error.value.code == "CAPACITY_ERROR"
        assert registry.status().versions[a.version_id]["state"] == VersionState.FAILED_RETRYABLE.value
        assert registry.status().versions[a.version_id]["attempt_id"] == 1

    def test_fatal_failure_is_never_retryable(self, registry):
        a = registry.allocate(DIGEST_A, version_id=1)
        registry.mark_fatal_failure(a.version_id, a.attempt_id)
        with pytest.raises(LoRAVersionError) as error:
            registry.retry_publication(a.version_id, DIGEST_A)
        assert error.value.code == "INVALID_STATE"


class TestSessionBinding:
    def test_bind_requires_a_published_version(self, registry):
        registry.allocate(DIGEST_A, version_id=1)
        with pytest.raises(LoRAVersionError) as error:
            registry.bind_latest("s1")
        assert error.value.code == "NO_PUBLISHED_VERSION"

    def test_bind_is_idempotent_and_keeps_the_first_version(self, registry):
        a = publish(registry, DIGEST_A, version_id=1)
        first = registry.bind_latest("s1")
        publish(registry, DIGEST_B, version_id=2)
        again = registry.bind_latest("s1")
        assert again == first
        assert again.lora_name == a.lora_name
        assert registry.status().capacity_owning == 2

    def test_release_is_idempotent(self, registry):
        publish(registry, DIGEST_A, version_id=1)
        registry.bind_latest("s1")
        assert registry.release("s1") is True
        assert registry.release("s1") is False
        assert registry.status().session_bindings == {}

    def test_release_before_bind_keeps_the_session_closed(self, registry):
        """16: close may arrive before the first bind; the ref must not appear later."""

        publish(registry, DIGEST_A, version_id=1)
        assert registry.release("s1") is False
        with pytest.raises(LoRAVersionError) as error:
            registry.bind_latest("s1")
        assert error.value.code == "SESSION_CLOSED"
        assert registry.status().session_bindings == {}

    def test_client_session_id_is_scoped_to_the_deployment_epoch(self, registry):
        publish(registry, DIGEST_A, version_id=1)
        registry.bind_latest("s1")
        other = LoRAVersionRegistry(logical_capacity=2, deployment_epoch="epoch1")
        publish(other, DIGEST_A, version_id=1)
        assert other.bind_latest("s1").version_id == 1
        assert other.release("s1") is True

    def test_forget_sessions_prunes_only_requested_closed_sessions(self, registry):
        publish(registry, DIGEST_A, version_id=1)
        registry.bind_latest("s1")
        registry.release("s1")
        registry.release("s2")
        registry.forget_sessions(["s1", "s1", "unknown"])
        registry.forget_sessions(["s1"])
        assert registry.closed_sessions == {"epoch0:s2"}
        with pytest.raises(LoRAVersionError) as error:
            registry.bind_latest("s2")
        assert error.value.code == "SESSION_CLOSED"
        registry.forget_sessions(["s2"])
        assert not registry.closed_sessions

    def test_forget_sessions_rejects_active_binding_without_pruning_any_tombstones(self, registry):
        publish(registry, DIGEST_A, version_id=1)
        registry.bind_latest("active")
        registry.release("closed")
        before = registry.status()
        with pytest.raises(LoRAVersionError) as error:
            registry.forget_sessions(["closed", "active"])
        assert error.value.code == "SESSION_STILL_BOUND"
        assert registry.status() == before
        assert registry.closed_sessions == {"epoch0:closed"}


class TestReclaim:
    def test_reclaim_waits_for_the_last_session(self, registry):
        a = publish(registry, DIGEST_A, version_id=1)
        registry.bind_latest("s_old")
        publish(registry, DIGEST_B, version_id=2)
        assert registry.claim_reclaimable() is None

        registry.release("s_old")
        claim = registry.claim_reclaimable()
        assert claim is not None and claim.version_id == a.version_id
        assert registry.status().versions[a.version_id]["state"] == VersionState.RECLAIMING.value
        assert registry.status().capacity_owning == 2  # still owns its slot while RECLAIMING

    def test_claim_is_exactly_once(self, registry):
        a = publish(registry, DIGEST_A, version_id=1)
        publish(registry, DIGEST_B, version_id=2)
        assert registry.claim_reclaimable().version_id == a.version_id
        assert registry.claim_reclaimable() is None

    def test_mark_reclaimed_releases_the_slot(self, registry):
        a = publish(registry, DIGEST_A, version_id=1)
        publish(registry, DIGEST_B, version_id=2)
        registry.claim_reclaimable()
        registry.mark_reclaimed(a.version_id)
        assert registry.status().capacity_owning == 1
        assert registry.allocate(DIGEST_C, version_id=3).version_id == 3

    def test_ambiguous_unload_keeps_capacity_and_refuses_late_ack(self, registry):
        a = publish(registry, DIGEST_A, version_id=1)
        publish(registry, DIGEST_B, version_id=2)
        registry.claim_reclaimable()
        registry.mark_reclaim_fatal(a.version_id)
        assert registry.status().versions[a.version_id]["reclaim_fatal"] is True
        assert registry.claim_reclaimable() is None
        with pytest.raises(LoRAVersionError) as error:
            registry.mark_reclaimed(a.version_id)
        assert error.value.code == "RECLAIM_AMBIGUOUS"
        assert registry.status().capacity_owning == 2

    def test_no_reclaim_for_a_default_that_is_still_published(self, registry):
        publish(registry, DIGEST_A, version_id=1)
        assert registry.claim_reclaimable() is None


class TestPublicationSafety:
    def test_fatal_candidate_keeps_capacity_and_blocks_new_publication(self, registry):
        publish(registry, DIGEST_A, version_id=1)
        b = registry.allocate(DIGEST_B, version_id=2)
        registry.mark_fatal_failure(b.version_id, b.attempt_id)
        assert registry.status().capacity_owning == 2
        with pytest.raises(LoRAVersionError, match="unconfirmed"):
            registry.allocate(DIGEST_C, version_id=3)

    def test_only_one_driver_can_claim_an_attempt(self, registry):
        p = registry.allocate(DIGEST_A, version_id=1)
        registry.claim_publication(p.version_id, p.attempt_id)
        replay = registry.allocate(DIGEST_A, version_id=1)
        with pytest.raises(LoRAVersionError) as error:
            registry.claim_publication(replay.version_id, replay.attempt_id)
        assert error.value.code == "PUBLICATION_IN_PROGRESS"

    def test_commit_requires_both_ready_receipts(self, registry):
        p = registry.allocate(DIGEST_A, version_id=1)
        with pytest.raises(LoRAVersionError) as error:
            registry.mark_published(p.version_id, p.attempt_id)
        assert error.value.code == "NOT_READY"
        assert registry.default_version is None

    def test_ready_cannot_change_engine_incarnation_or_digest(self, registry):
        from tests.agentic.lora_helpers import fleet_receipts

        p = registry.allocate(DIGEST_A, version_id=1)
        entry = registry.versions[p.version_id]
        registry.claim_publication(p.version_id, p.attempt_id)
        registry.record_prepared(p.version_id, p.attempt_id, fleet_receipts(entry))
        for field, value in (("engine_incarnation", "restarted"), ("digest", DIGEST_B)):
            receipts = fleet_receipts(entry, "READY_LOCAL")
            receipts["engine1"][field] = value
            with pytest.raises(LoRAVersionError):
                registry.record_ready(p.version_id, p.attempt_id, receipts)
        assert registry.default_version is None


def test_direct_retry_cannot_bypass_unconfirmed_other_version(registry):
    a = registry.allocate(DIGEST_A, version_id=1)
    registry.mark_retryable_failure(a.version_id, a.attempt_id)
    b = registry.allocate(DIGEST_B, version_id=2)
    registry.mark_fatal_failure(b.version_id, b.attempt_id)
    with pytest.raises(LoRAVersionError) as error:
        registry.retry_publication(a.version_id, DIGEST_A)
    assert error.value.code == "PUBLICATION_BLOCKED"
    assert registry.versions[a.version_id].current_attempt_id == 1


@pytest.mark.parametrize("state", ["retired", "reclaiming", "reclaimed"])
def test_completed_publication_replay_never_rolls_back_default(registry, state):
    a = publish(registry, DIGEST_A, version_id=1)
    b = publish(registry, DIGEST_B, version_id=2)
    if state != "retired":
        registry.claim_reclaimable()
    if state == "reclaimed":
        registry.mark_reclaimed(a.version_id)
    before = registry.status().capacity_owning
    for replay in (registry.allocate(DIGEST_A, version_id=1), registry.retry_publication(a.version_id, DIGEST_A)):
        assert replay.no_op
        assert replay.version_id == a.version_id
        assert registry.default_version == b.version_id
        assert registry.status().capacity_owning == before
    with pytest.raises(LoRAVersionError, match="digest"):
        registry.retry_publication(a.version_id, DIGEST_C)


def test_new_version_can_return_to_historical_content_after_capacity_releases(registry):
    a = publish(registry, DIGEST_A, version_id=11)
    registry.bind_latest("old-A")
    b = publish(registry, DIGEST_B, version_id=22)
    registry.bind_latest("old-B")
    with pytest.raises(LoRAVersionError) as error:
        registry.allocate(DIGEST_A, version_id=33)
    assert error.value.code == "CAPACITY_ERROR"
    assert 33 not in registry.versions

    registry.release("old-A")
    assert registry.claim_reclaimable().version_id == a.version_id
    registry.mark_reclaimed(a.version_id)
    c = registry.allocate(DIGEST_A, version_id=33)
    assert not c.no_op and c.version_id == 33
    assert c.lora_name != a.lora_name and c.digest == a.digest
    assert registry.bind_latest("during").version_id == b.version_id
    commit_ready(registry, c.version_id, c.attempt_id)
    assert registry.bind_latest("new").version_id == c.version_id
    assert registry.bind_latest("old-B").version_id == b.version_id
    assert registry.bind_latest("during").version_id == b.version_id
    assert registry.allocate(DIGEST_A, version_id=11).no_op
    assert registry.default_version == 33


@pytest.mark.parametrize("state", list(VersionState))
def test_reserved_version_rejects_changed_digest_in_every_state(registry, state):
    a = registry.allocate(DIGEST_A, version_id=17)
    if state in {VersionState.FAILED_RETRYABLE, VersionState.FAILED_FATAL}:
        failure = (
            registry.mark_retryable_failure if state is VersionState.FAILED_RETRYABLE else registry.mark_fatal_failure
        )
        failure(a.version_id, a.attempt_id)
    elif state is not VersionState.LOADING:
        commit_ready(registry, a.version_id, a.attempt_id)
        if state is not VersionState.PUBLISHED:
            publish(registry, DIGEST_B, version_id=18)
            if state in {VersionState.RECLAIMING, VersionState.RECLAIMED}:
                registry.claim_reclaimable()
            if state is VersionState.RECLAIMED:
                registry.mark_reclaimed(a.version_id)
    before = registry.status()
    with pytest.raises(LoRAVersionError) as error:
        registry.allocate(DIGEST_B, version_id=17)
    assert error.value.code == "VERSION_CONFLICT"
    assert registry.status() == before


def test_identical_content_with_new_id_is_not_a_duplicate_loading_attempt(registry):
    first = registry.allocate(DIGEST_A, version_id=1)
    assert registry.allocate(DIGEST_A, version_id=1) == first
    with pytest.raises(LoRAVersionError) as error:
        registry.allocate(DIGEST_A, version_id=2)
    assert error.value.code == "PUBLICATION_IN_PROGRESS"
    assert 2 not in registry.versions


@pytest.mark.parametrize("version_id", [0, -1, True, "1", None])
def test_allocate_requires_a_positive_integer_identity(registry, version_id):
    with pytest.raises(LoRAVersionError) as error:
        registry.allocate(DIGEST_A, version_id=version_id)
    assert error.value.code == "INVALID_VERSION"
    assert not registry.versions
