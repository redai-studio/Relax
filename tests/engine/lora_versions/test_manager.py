# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""CPU publication scenarios and seeded event replay, with no Ray/GPU
runtime."""

import hashlib
import os
import pickle
import random
from collections import Counter
from dataclasses import replace

import pytest

from relax.engine.lora_versions.identity import VersionIdentity
from relax.engine.lora_versions.manager import (
    CleanupObservation,
    LoRAVersionManager,
    PublicationError,
    Replica,
    SessionKey,
    VersionState,
)


FINGERPRINTS = {"a": 123, "b": -456}
ENGINES = {"e0": "boot0", "e1": "boot1"}


def identity(version_id):
    return VersionIdentity(version_id, hashlib.sha256(str(version_id).encode()).hexdigest())


def verify(manager, publication, engine_id):
    replica = Replica(engine_id, dict(publication.targets)[engine_id], f"{publication.attempt_id}-{engine_id}")
    assert manager.record_verified(
        publication.identity.version_id, publication.attempt_id, replica, FINGERPRINTS, pinned=True
    )
    return replica


def publish(manager, version_id):
    publication = manager.reserve(identity(version_id), FINGERPRINTS).publication
    for engine, _ in publication.targets:
        verify(manager, publication, engine)
    return manager.commit(version_id, publication.attempt_id, publication.expected_default)


def cleanup_observation(publication, engine_id, *, lora_id=None, requests=0, fenced=True, closed=True):
    return CleanupObservation(engine_id, dict(publication.targets)[engine_id], lora_id, requests, closed, fenced)


@pytest.fixture
def manager():
    result = LoRAVersionManager(ENGINES)
    result.register_shard("s", "s0")
    return result


def test_first_publication_and_partial_readiness(manager):
    key = SessionKey("s", "s0", 0)
    with pytest.raises(PublicationError, match="NO_PUBLISHED_VERSION"):
        manager.bind(key)
    a = manager.reserve(identity(0), FINGERPRINTS).publication
    assert manager.occupied_versions == 1
    assert manager.default_version is None  # Bootstrap is the exception to RFC I6.
    verify(manager, a, "e0")
    with pytest.raises(PublicationError, match="VERSION_NOT_READY"):
        manager.commit(0, a.attempt_id, None)
    verify(manager, a, "e1")
    manager.commit(0, a.attempt_id, None)
    assert manager.bind(key) == a.identity


def test_commit_switches_new_sessions_only_and_bind_retry_keeps_old_version(manager):
    publish(manager, 0)
    old = SessionKey("s", "s0", 0)
    assert manager.bind(old) == identity(0)
    b = manager.reserve(identity(1), FINGERPRINTS).publication
    verify(manager, b, "e0")
    mid = SessionKey("s", "s0", 1)
    assert manager.bind(mid) == identity(0)
    with pytest.raises(PublicationError, match="VERSION_NOT_READY"):
        manager.commit(1, b.attempt_id, 0)
    verify(manager, b, "e1")
    with pytest.raises(PublicationError, match="DEFAULT_CHANGED"):
        manager.commit(1, b.attempt_id, None)
    assert manager.default_version == 0
    manager.commit(1, b.attempt_id, 0)
    assert manager.bind(SessionKey("s", "s0", 2)) == identity(1)
    for key in (old, mid):
        assert manager.bind(key) == identity(0)
    assert manager.publication(0).session_count == 2
    assert manager.publication(0).state is VersionState.RETIRED
    assert manager.release(old)
    assert not manager.release(old)
    assert manager.publication(0).state is VersionState.RETIRED
    manager.release(mid)
    assert manager.publication(0).state is VersionState.DRAINING


def test_duplicate_publication_and_conflict_precede_capacity_checks(manager):
    publish(manager, 0)
    manager.bind(SessionKey("s", "s0", 0))
    b = publish(manager, 1)
    result = manager.reserve(identity(1), FINGERPRINTS)
    assert not result.should_load
    assert result.publication == b
    assert manager.occupied_versions == 2
    conflict = replace(identity(1), digest="f" * 64)
    with pytest.raises(PublicationError, match=f"VERSION_CONFLICT.*{identity(1).digest}.*{'f' * 64}"):
        manager.reserve(conflict, FINGERPRINTS)
    assert manager.default_version == 1
    with pytest.raises(PublicationError, match="CAPACITY_EXHAUSTED"):
        manager.reserve(identity(2), FINGERPRINTS)


def test_only_one_attempt_can_load(manager):
    first = manager.reserve(identity(0), FINGERPRINTS)
    duplicate = manager.reserve(identity(0), FINGERPRINTS)
    assert duplicate.publication == first.publication
    assert not duplicate.should_load
    with pytest.raises(PublicationError, match="PUBLICATION_IN_PROGRESS"):
        manager.reserve(identity(1), FINGERPRINTS)


@pytest.mark.parametrize("cancel_first", [True, False])
def test_cancel_commit_race_has_one_winner(manager, cancel_first):
    publish(manager, 0)
    b = manager.reserve(identity(1), FINGERPRINTS).publication
    for engine in ENGINES:
        verify(manager, b, engine)
    if cancel_first:
        assert manager.cancel(1, b.attempt_id)
        with pytest.raises(PublicationError, match="PUBLICATION_CANCELLED"):
            manager.commit(1, b.attempt_id, 0)
        assert manager.default_version == 0
    else:
        manager.commit(1, b.attempt_id, 0)
        assert not manager.cancel(1, b.attempt_id)
        assert manager.commit(1, b.attempt_id, 0) == manager.publication(1)
        assert manager.default_version == 1


def test_failed_publication_fences_all_targets_even_without_load_receipts(manager):
    publish(manager, 0)
    b = manager.reserve(identity(1), FINGERPRINTS).publication
    verify(manager, b, "e0")
    manager.cancel(1, b.attempt_id)  # e1 timed out; it may still be loading.
    assert not manager.record_verified(1, b.attempt_id, Replica("e1", "boot1", "late"), FINGERPRINTS, pinned=True)
    manager.observe_cleanup(1, b.attempt_id, cleanup_observation(b, "e0"))
    manager.observe_cleanup(1, b.attempt_id, cleanup_observation(b, "e1", fenced=False))
    assert manager.occupied_versions == 2
    assert manager.publication(1).state is VersionState.ABORTING
    # A delayed load is visible in inventory even though its receipt was lost.
    manager.observe_cleanup(1, b.attempt_id, cleanup_observation(b, "e1", lora_id="late", requests=1))
    assert not manager.unload_candidates(1)
    manager.observe_cleanup(1, b.attempt_id, cleanup_observation(b, "e1", lora_id="late"))
    assert manager.unload_candidates(1) == (Replica("e1", "boot1", "late"),)
    manager.observe_cleanup(1, b.attempt_id, cleanup_observation(b, "e1"))
    assert manager.publication(1).state is VersionState.ABORTED
    assert manager.occupied_versions == 1
    assert manager.default_version == 0


def test_retry_fences_old_receipts_cancel_commit_and_cleanup(manager):
    publish(manager, 0)
    old = manager.reserve(identity(1), FINGERPRINTS).publication
    manager.cancel(1, old.attempt_id)
    for engine in ENGINES:
        manager.observe_cleanup(1, old.attempt_id, cleanup_observation(old, engine))
    with pytest.raises(PublicationError, match="REPLICA_INVALID"):
        manager.reserve(identity(1), {"a": 999})
    retry = manager.reserve(identity(1), FINGERPRINTS)
    assert retry.should_load
    assert retry.publication.attempt_id != old.attempt_id
    assert not manager.cancel(1, old.attempt_id)
    assert not manager.record_verified(1, old.attempt_id, Replica("e0", "boot0", "old"), FINGERPRINTS, pinned=True)
    assert not manager.observe_cleanup(1, old.attempt_id, cleanup_observation(old, "e0"))
    with pytest.raises(PublicationError, match="STALE_ATTEMPT"):
        manager.commit(1, old.attempt_id, 0)
    for engine in ENGINES:
        verify(manager, retry.publication, engine)
    manager.commit(1, retry.publication.attempt_id, 0)
    assert manager.default_version == 1


@pytest.mark.parametrize("invalid", ["fingerprint", "pinned", "lora_id"])
def test_invalid_receipt_aborts_without_switching_default(manager, invalid):
    publish(manager, 0)
    b = manager.reserve(identity(1), FINGERPRINTS).publication
    with pytest.raises(PublicationError, match="REPLICA_INVALID"):
        manager.record_verified(
            1,
            b.attempt_id,
            Replica("e0", "boot0", "" if invalid == "lora_id" else "b0"),
            {"a": 999} if invalid == "fingerprint" else FINGERPRINTS,
            pinned=invalid != "pinned",
        )
    assert manager.publication(1).state is VersionState.ABORTING
    assert manager.default_version == 0


def test_two_level_reclamation_and_capacity(manager):
    a = publish(manager, 0)
    session = SessionKey("s", "s0", 0)
    manager.bind(session)
    publish(manager, 1)
    assert not manager.observe_cleanup(0, a.attempt_id, cleanup_observation(a, "e0"))
    assert not manager.unload_candidates(0)
    with pytest.raises(PublicationError, match="CAPACITY_EXHAUSTED"):
        manager.reserve(identity(2), FINGERPRINTS)
    manager.release(session)
    for replica in a.verified:
        manager.observe_cleanup(
            0, a.attempt_id, cleanup_observation(a, replica.engine_id, lora_id=replica.lora_id, requests=1)
        )
    assert not manager.unload_candidates(0)
    for replica in a.verified:
        manager.observe_cleanup(0, a.attempt_id, cleanup_observation(a, replica.engine_id, lora_id=replica.lora_id))
        # Lost unload replies keep the same fenced target eligible for retry.
        # Physical exactly-once behavior is the engine's contract (PR2).
        assert manager.unload_candidates(0) == (replica,)
        assert manager.unload_candidates(0) == (replica,)
        assert manager.occupied_versions == 2
        manager.observe_cleanup(0, a.attempt_id, cleanup_observation(a, replica.engine_id))
    assert manager.publication(0).state is VersionState.RECLAIMED
    assert not manager.unload_candidates(0)
    assert manager.occupied_versions == 1
    assert manager.reserve(identity(2), FINGERPRINTS).should_load
    assert not manager.reserve(identity(0), FINGERPRINTS).should_load
    manager.commit(0, a.attempt_id, None)  # Lost original commit response, retried much later.
    assert manager.default_version == 1


def test_cleanup_requires_stop_fence_and_matching_instance(manager):
    a = publish(manager, 0)
    publish(manager, 1)
    replica = a.verified[0]
    for closed, fenced in [(False, False), (True, False), (False, True)]:
        manager.observe_cleanup(
            0,
            a.attempt_id,
            cleanup_observation(a, "e0", lora_id=replica.lora_id, closed=closed, fenced=fenced),
        )
        assert not manager.unload_candidates(0)
    assert not manager.observe_cleanup(0, a.attempt_id, cleanup_observation(a, "e0", lora_id="different-load"))
    wrong_boot = replace(cleanup_observation(a, "e0"), instance_id="old-boot")
    assert not manager.observe_cleanup(0, a.attempt_id, wrong_boot)
    manager.observe_cleanup(0, a.attempt_id, cleanup_observation(a, "e0"))
    assert not manager.observe_cleanup(0, a.attempt_id, cleanup_observation(a, "e0", lora_id=replica.lora_id))
    assert manager.publication(0).absent_engines == ("e0",)


@pytest.mark.parametrize("regression", ["fence", "admission", "requests"])
def test_cleanup_rejects_whole_regressive_observations(manager, regression):
    a = publish(manager, 0)
    publish(manager, 1)
    replica = a.verified[0]
    draining = cleanup_observation(a, "e0", lora_id=replica.lora_id, requests=1)
    assert manager.observe_cleanup(0, a.attempt_id, draining)
    stale = {
        "fence": replace(draining, attempt_fenced=False, requests=0),
        "admission": replace(draining, admission_closed=False, requests=0),
        "requests": replace(draining, requests=2),
    }[regression]
    assert not manager.observe_cleanup(0, a.attempt_id, stale)
    assert not manager.unload_candidates(0)
    assert manager.observe_cleanup(0, a.attempt_id, replace(draining, requests=0))
    assert manager.unload_candidates(0) == (replica,)
    assert not manager.observe_cleanup(0, a.attempt_id, draining)
    assert manager.unload_candidates(0) == (replica,)
    assert manager.occupied_versions == 2


@pytest.mark.parametrize(("closed", "fenced"), [(False, True), (True, False)])
def test_cleanup_keeps_snapshot_fields_together_and_allows_pre_drain_growth(manager, closed, fenced):
    a = publish(manager, 0)
    publish(manager, 1)
    replica = a.verified[0]
    before_stop = cleanup_observation(a, "e0", lora_id=replica.lora_id, requests=0, closed=closed, fenced=fenced)
    assert manager.observe_cleanup(0, a.attempt_id, before_stop)
    assert not manager.unload_candidates(0)
    stopped = replace(before_stop, requests=2, admission_closed=True, attempt_fenced=True)
    assert manager.observe_cleanup(0, a.attempt_id, stopped)
    assert not manager.unload_candidates(0)
    # The earlier zero count cannot be combined with the later closed flag.
    assert not manager.observe_cleanup(0, a.attempt_id, before_stop)
    assert not manager.unload_candidates(0)
    assert manager.observe_cleanup(0, a.attempt_id, replace(stopped, requests=0))
    assert manager.unload_candidates(0) == (replica,)


def test_cleanup_does_not_merge_complementary_close_and_fence_flags(manager):
    a = publish(manager, 0)
    publish(manager, 1)
    replica = a.verified[0]
    closed_only = cleanup_observation(a, "e0", lora_id=replica.lora_id, requests=2, fenced=False)
    fenced_only = replace(closed_only, requests=0, admission_closed=False, attempt_fenced=True)
    assert manager.observe_cleanup(0, a.attempt_id, closed_only)
    assert not manager.observe_cleanup(0, a.attempt_id, fenced_only)
    assert not manager.unload_candidates(0)
    assert manager.observe_cleanup(0, a.attempt_id, replace(closed_only, attempt_fenced=True))
    assert not manager.unload_candidates(0)
    assert manager.observe_cleanup(0, a.attempt_id, replace(closed_only, attempt_fenced=True, requests=0))
    assert manager.unload_candidates(0) == (replica,)


def test_engine_restart_invalidates_receipts_and_requires_recovery(manager):
    a = publish(manager, 0)
    key = SessionKey("s", "s0", 0)
    manager.bind(key)
    b = manager.reserve(identity(1), FINGERPRINTS).publication
    old = verify(manager, b, "e0")
    assert manager.replace_engine("e0", "boot0", "boot2")
    assert not manager.replace_engine("e0", "boot0", "boot3")
    assert manager.serving_engines == ("e1",)
    assert manager.publication(1).state is VersionState.ABORTING
    assert not manager.record_verified(1, b.attempt_id, old, FINGERPRINTS, pinned=True)
    assert not manager.observe_cleanup(1, b.attempt_id, cleanup_observation(b, "e0"))
    assert not manager.record_verified(0, a.attempt_id, a.verified[0], FINGERPRINTS, pinned=True)
    manager.record_verified(0, a.attempt_id, Replica("e0", "boot2", "restored-a"), FINGERPRINTS, pinned=True)
    assert manager.serving_engines == ("e0", "e1")
    assert manager.bind(key) == identity(0)


def test_no_fallback_when_every_engine_is_recovering(manager):
    publish(manager, 0)
    bound = SessionKey("s", "s0", 0)
    manager.bind(bound)
    for engine, instance in ENGINES.items():
        manager.replace_engine(engine, instance, f"{instance}-new")
    assert manager.bind(bound) == identity(0)
    with pytest.raises(PublicationError, match="LORA_VERSION_UNAVAILABLE"):
        manager.bind(SessionKey("s", "s0", 1))
    assert manager.publication(0).session_count == 1


def test_conflicting_replica_receipt_revokes_serving_readiness(manager):
    a = publish(manager, 0)
    with pytest.raises(PublicationError, match="LORA_ID_MISMATCH"):
        manager.record_verified(0, a.attempt_id, Replica("e0", "boot0", "unexpected"), FINGERPRINTS, pinned=True)
    assert manager.serving_engines == ("e1",)
    with pytest.raises(PublicationError, match="REPLICA_INVALID"):
        manager.record_verified(0, a.attempt_id, a.verified[1], {"a": 999}, pinned=True)
    assert not manager.serving_engines
    assert manager.default_version == 0


def test_release_before_bind_and_watermark_prevent_resurrection(manager):
    publish(manager, 0)
    key = SessionKey("s", "s0", 4)
    assert manager.release(key)
    assert not manager.release(key)
    with pytest.raises(PublicationError, match="SESSION_CLOSED"):
        manager.bind(key)
    assert manager.advance_watermark("s", "s0", 4)
    assert not manager._shards["s"].closed
    with pytest.raises(PublicationError, match="SESSION_CLOSED"):
        manager.bind(key)
    assert not manager.release(key)
    assert not manager.advance_watermark("s", "s0", 3)
    assert manager.publication(0).session_count == 0
    manager.bind(SessionKey("s", "s0", 5))
    with pytest.raises(PublicationError, match="SESSION_STILL_ACTIVE"):
        manager.advance_watermark("s", "s0", 5)


def test_shard_death_releases_all_bindings_without_reviving_old_instances(manager):
    publish(manager, 0)
    old = [SessionKey("s", "s0", n) for n in range(3)]
    for key in old:
        manager.bind(key)
    publish(manager, 1)
    manager.register_shard("s", "s1", expected_instance="s0")
    assert manager.publication(0).session_count == 0
    assert manager.publication(0).state is VersionState.DRAINING
    for key in old:
        assert not manager.release(key)
        with pytest.raises(PublicationError, match="SESSION_CLOSED"):
            manager.bind(key)
    with pytest.raises(PublicationError, match="SHARD_INSTANCE_CHANGED"):
        manager.register_shard("s", "s0")
    assert not manager.close_shard("s", "s0")
    assert not manager.advance_watermark("s", "s0", 100)
    assert manager.bind(SessionKey("s", "s1", 0)) == identity(1)
    assert manager.close_shard("s", "s1")
    assert not manager.close_shard("s", "s1")
    with pytest.raises(PublicationError, match="SESSION_CLOSED"):
        manager.register_shard("s", "s1")


def test_reclamation_waits_for_references_from_every_shard(manager):
    a = publish(manager, 0)
    manager.register_shard("t", "t0")
    first = SessionKey("s", "s0", 0)
    other = SessionKey("t", "t0", 0)
    assert manager.bind(first) == manager.bind(other) == identity(0)
    assert manager.publication(0).session_count == 2
    publish(manager, 1)
    assert manager.close_shard("s", "s0")
    assert manager.publication(0).session_count == 1
    assert manager.publication(0).state is VersionState.RETIRED
    assert not manager.observe_cleanup(0, a.attempt_id, cleanup_observation(a, "e0"))
    with pytest.raises(PublicationError, match="CAPACITY_EXHAUSTED"):
        manager.reserve(identity(2), FINGERPRINTS)
    manager.register_shard("s", "s1", expected_instance="s0")
    assert manager.bind(SessionKey("s", "s1", 0)) == identity(1)
    assert manager.bind(other) == identity(0)
    assert not manager.release(first)
    assert manager.publication(0).session_count == 1
    assert manager.release(other)
    assert manager.publication(0).state is VersionState.DRAINING
    assert manager.publication(1).session_count == 1


def test_aborted_old_version_cannot_roll_default_back(manager):
    publish(manager, 0)
    b = manager.reserve(identity(1), FINGERPRINTS).publication
    manager.cancel(1, b.attempt_id)
    for engine in ENGINES:
        manager.observe_cleanup(1, b.attempt_id, cleanup_observation(b, engine))
    publish(manager, 2)
    with pytest.raises(PublicationError, match="VERSION_SUPERSEDED"):
        manager.reserve(identity(1), FINGERPRINTS)


def test_publication_results_and_errors_are_serializable_and_detached(manager):
    publication = publish(manager, 0)
    assert pickle.loads(pickle.dumps(publication)) == publication
    error = pickle.loads(pickle.dumps(PublicationError("CAPACITY_EXHAUSTED", "full")))
    assert error.code == "CAPACITY_EXHAUSTED"
    manager.bind(SessionKey("s", "s0", 0))
    assert publication.session_count == 0
    assert manager.publication(0).session_count == 1


def test_seeded_event_interleavings():
    """Replay with RELAX_LORA_TEST_SEED_START and RELAX_LORA_TEST_SEEDS.

    Keep normal CI bounded; the RFC acceptance run uses 100000 seeds. An
    independent binding ledger checks counts after every reordered event, with
    two shards sharing versions and restarting independently.
    """
    start = int(os.environ.get("RELAX_LORA_TEST_SEED_START", "0"))
    seeds = int(os.environ.get("RELAX_LORA_TEST_SEEDS", "128"))
    for seed in range(start, start + seeds):
        rng = random.Random(seed)
        manager = LoRAVersionManager(ENGINES)
        publish(manager, 0)
        bindings = {}
        shard_epochs = {"s": 0, "t": 0}
        for shard_id in shard_epochs:
            manager.register_shard(shard_id, f"{shard_id}0")
            key = SessionKey(shard_id, f"{shard_id}0", 0)
            assert manager.bind(key) == identity(0)
            bindings[key] = 0
        session_keys = list(bindings)
        closed = set()
        attempts = [manager.publication(0)]
        trace = []
        next_version = 1
        previous_default = 0
        engine_epochs = dict(ENGINES)
        for step in range(64):
            action = rng.choice(
                ["reserve"] * 3
                + ["receipt"] * 6
                + ["commit"] * 4
                + ["cancel"]
                + ["bind"] * 4
                + ["release"] * 4
                + ["cleanup"] * 4
                + ["restart", "death"]
            )
            shard_id = rng.choice(tuple(shard_epochs))
            shard_instance = f"{shard_id}{shard_epochs[shard_id]}"
            key = SessionKey(shard_id, shard_instance, rng.randrange(8))
            if action in ("bind", "release") and rng.random() < 0.2:
                key = rng.choice(session_keys)
            session_keys.append(key)
            current = [manager.publication(v) for v in sorted({p.identity.version_id for p in attempts})]
            eligible = [
                p
                for p in current
                if (action in ("commit", "cancel") and p.state is VersionState.LOADING)
                or (action == "cleanup" and p.state in (VersionState.DRAINING, VersionState.ABORTING))
                or (
                    action == "receipt"
                    and p.state in (VersionState.LOADING, VersionState.PUBLISHED, VersionState.RETIRED)
                )
            ]
            # Mix useful current work with old/lost replies. Restarted engines
            # get recovery receipts, so the trace can continue making progress.
            publication = rng.choice(eligible if eligible and rng.random() < 0.8 else attempts)
            version_id = publication.identity.version_id
            engine = rng.choice(tuple(ENGINES))
            trace.append((action, version_id, publication.attempt_id, engine, key))
            try:
                try:
                    if action == "reserve":
                        # Include known IDs to exercise duplicate reservations and retries.
                        requested = rng.choice([next_version, version_id])
                        result = manager.reserve(identity(requested), FINGERPRINTS)
                        if result.should_load:
                            attempts.append(result.publication)
                            next_version = max(next_version, requested + 1)
                    elif action == "receipt":
                        manager.record_verified(
                            version_id,
                            publication.attempt_id,
                            Replica(engine, dict(publication.targets)[engine], f"{publication.attempt_id}-{engine}"),
                            FINGERPRINTS,
                            pinned=True,
                        )
                    elif action == "commit":
                        manager.commit(version_id, publication.attempt_id, publication.expected_default)
                    elif action == "cancel":
                        manager.cancel(version_id, publication.attempt_id)
                    elif action == "bind":
                        bound = manager.bind(key)
                        assert key not in closed
                        if key in bindings:
                            assert bound.version_id == bindings[key]
                        else:
                            assert bound.version_id == manager.default_version
                            bindings[key] = bound.version_id
                    elif action == "release":
                        manager.release(key)
                        closed.add(key)
                        bindings.pop(key, None)
                    elif action == "cleanup":
                        present = rng.choice([True, False])
                        manager.observe_cleanup(
                            version_id,
                            publication.attempt_id,
                            cleanup_observation(
                                publication,
                                engine,
                                lora_id=f"{publication.attempt_id}-{engine}" if present else None,
                                requests=rng.randrange(3) if present else 0,
                                fenced=rng.choice([True, False]),
                                closed=rng.choice([True, False]),
                            ),
                        )
                    elif action == "restart":
                        new_instance = f"boot-{seed}-{step}"
                        manager.replace_engine(engine, engine_epochs[engine], new_instance)
                        engine_epochs[engine] = new_instance
                    else:
                        manager.close_shard(shard_id, shard_instance)
                        retired = [key for key in bindings if key.shard_id == shard_id]
                        for retired_key in retired:
                            del bindings[retired_key]
                        closed.update(retired)
                        shard_epochs[shard_id] += 1
                        manager.register_shard(
                            shard_id, f"{shard_id}{shard_epochs[shard_id]}", expected_instance=shard_instance
                        )
                except PublicationError as error:
                    assert error.code in {
                        "CAPACITY_EXHAUSTED",
                        "PUBLICATION_IN_PROGRESS",
                        "VERSION_SUPERSEDED",
                        "STALE_ATTEMPT",
                        "PUBLICATION_CANCELLED",
                        "DEFAULT_CHANGED",
                        "VERSION_NOT_READY",
                        "SESSION_CLOSED",
                        "LORA_VERSION_UNAVAILABLE",
                    }
                assert manager.default_version >= previous_default
                previous_default = manager.default_version
                assert manager.occupied_versions <= 2
                assert manager.publication(manager.default_version).state is VersionState.PUBLISHED
                counts = Counter(bindings.values())
                for version_id in sorted({p.identity.version_id for p in attempts}):
                    status = manager.publication(version_id)
                    assert status.session_count == counts[status.identity.version_id]
                    if status.session_count:
                        assert status.state in (VersionState.PUBLISHED, VersionState.RETIRED)
                    if status.state in (VersionState.ABORTED, VersionState.RECLAIMED):
                        assert len(status.absent_engines) == len(status.targets)
                    for candidate in manager.unload_candidates(status.identity.version_id):
                        assert not status.session_count
                        assert candidate.instance_id == engine_epochs[candidate.engine_id]
                    if status.state in (VersionState.PUBLISHED, VersionState.RETIRED):
                        for serving in manager.serving_engines:
                            assert any(
                                r.engine_id == serving and r.instance_id == engine_epochs[serving]
                                for r in status.verified
                            )
            except Exception as error:
                raise AssertionError(f"seed={seed}, step={step}, events={trace!r}") from error
