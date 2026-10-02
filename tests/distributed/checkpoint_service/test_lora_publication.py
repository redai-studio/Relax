# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Fault tests for the immutable LoRA publisher (Task 7, RFC §12/§13/§17).

The publisher is driven against a real ``LoRAVersionRegistry`` and a scripted
engine double, so every case asserts both what the engines were told and where
the version ended up — no engines, NCCL or Megatron involved.
"""

import logging
from types import SimpleNamespace
from typing import Any, Callable, Dict, List, Sequence, Tuple
from unittest.mock import Mock

import pytest
import torch

from relax.agentic.session.lora_version import (
    LoRAVersionError,
    LoRAVersionRegistry,
    VersionState,
)
from relax.distributed.checkpoint_service.lora_publication import (
    AdapterSnapshot,
    EngineReply,
    LoRAPublicationError,
    LoRAPublisher,
    RayLoRAVersionRegistryClient,
    classify_engine_response,
    materialize_adapter_snapshot,
)
from tests.agentic.lora_helpers import commit_ready


DIGEST_A = "a" * 64
DIGEST_B = "b" * 64
DIGEST_C = "c" * 64


# --------------------------------------------------------------------------------------
# Doubles
# --------------------------------------------------------------------------------------


class _LocalRegistry:
    """Publisher-side facade over an in-process Registry (same call shape as
    the Ray one)."""

    def __init__(self, registry: LoRAVersionRegistry) -> None:
        self._registry = registry

    def __getattr__(self, name: str) -> Callable:
        return getattr(self._registry, name)


class _FakeEngines:
    """Scripted fleet: records the protocol in order and answers per (engine,
    op)."""

    def __init__(self, engine_ids: Sequence[str] = ("engine0", "engine1")) -> None:
        self.engine_ids = list(engine_ids)
        self.events: List[Tuple[str, str, Any]] = []
        self.replies: Dict[Tuple[str, str], EngineReply] = {}
        self.identities = {}
        self.collective_participants = set()
        self.silent: set = set()  # engines that never answer

    def fail(self, op: str, engine_id: str, ambiguous: bool = False, message: str = "nope") -> None:
        self.replies[(engine_id, op)] = EngineReply(False, message, ambiguous=ambiguous)

    def fire(self, endpoint: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        op = payload.get("op", "legacy")
        if op == "bucket":
            self.collective_participants = {
                engine
                for engine in self.engine_ids
                if engine not in self.silent
                and (self.replies.get((engine, op), EngineReply(True)).success or self.replies[(engine, op)].ambiguous)
            }
        for engine_id in self.engine_ids:
            if op == "begin":
                self.identities[engine_id] = dict(payload)
            self.events.append(("fire", f"{engine_id}:{op}", payload))
        return {engine_id: (engine_id, op) for engine_id in self.engine_ids}

    def broadcast(self, names: Sequence[str], index: int) -> None:
        self.events.append(("broadcast", str(index), tuple(names)))
        if self.collective_participants != set(self.engine_ids):
            # A pre-receive refusal means one NCCL participant never enters.
            # Model its failure instead of allowing an impossible successful broadcast.
            raise RuntimeError("collective missing a receiver")

    def collect(self, pending: Dict[str, Any]) -> Dict[str, EngineReply]:
        replies: Dict[str, EngineReply] = {}
        for engine_id, (_, op) in pending.items():
            self.events.append(("collect", f"{engine_id}:{op}", None))
            if engine_id in self.silent:
                continue
            identity = self.identities.get(engine_id, {})
            receipt = {key: identity.get(key) for key in ("version_id", "digest", "lora_name", "attempt_id")}
            receipt.update(
                engine_incarnation=f"{engine_id}-boot1", state="PREPARED" if op == "begin" else "READY_LOCAL"
            )
            replies[engine_id] = self.replies.get((engine_id, op), EngineReply(True, receipt=receipt))
        return replies

    # -- assertions helpers --
    def ops(self) -> List[str]:
        return [op for kind, op, _ in self.events if kind == "fire"]


def _snapshot(values: Dict[str, torch.Tensor] | None = None) -> AdapterSnapshot:
    tensors = values or {"lora_a": torch.arange(6, dtype=torch.bfloat16), "lora_b": torch.ones(4)}
    return materialize_adapter_snapshot({"lora_rank": 8, "lora_alpha": 16}, tensors)


@pytest.fixture()
def registry() -> LoRAVersionRegistry:
    return LoRAVersionRegistry(logical_capacity=2, deployment_epoch="epoch0")


def _publisher(
    engines: _FakeEngines,
    registry: LoRAVersionRegistry,
    *,
    bucket_cap: int = 8,
    group_name: str = "slime-pp_0",
) -> LoRAPublisher:
    return LoRAPublisher(
        fire=engines.fire,
        collect=engines.collect,
        broadcast=engines.broadcast,
        registry=_LocalRegistry(registry),
        bucket_cap_bytes=bucket_cap,
        group_name=group_name,
    )


def _publish_a(registry: LoRAVersionRegistry) -> int:
    publication = registry.allocate(DIGEST_A, version_id=1)
    commit_ready(registry, publication.version_id, publication.attempt_id)
    return publication.version_id


# --------------------------------------------------------------------------------------
# Identity
# --------------------------------------------------------------------------------------


class TestSnapshotIdentity:
    def test_digest_ignores_bucketing_but_not_bytes(self):
        """F32: a sub-tolerance byte difference is still a different version."""

        snapshot = _snapshot()
        nudged = _snapshot({"lora_a": torch.arange(6, dtype=torch.bfloat16) + 1e-7, "lora_b": torch.ones(4)})
        assert snapshot.digest != nudged.digest
        assert snapshot.manifest["lora_a"] != nudged.manifest["lora_a"]

    def test_snapshot_owns_its_storage(self):
        """F40: hashing then transmitting must both see the frozen bytes."""

        live = {"lora_a": torch.zeros(4, dtype=torch.float32)}
        snapshot = _snapshot(live)
        live["lora_a"][0] = 5.0
        assert torch.equal(snapshot.tensors["lora_a"], torch.zeros(4, dtype=torch.float32))
        assert snapshot.manifest["lora_a"] == materialize_adapter_snapshot({}, snapshot.tensors).manifest["lora_a"]

    def test_config_participates_in_the_digest(self):
        tensors = {"lora_a": torch.zeros(4)}
        assert (
            materialize_adapter_snapshot({"r": 1}, tensors).digest
            != materialize_adapter_snapshot({"r": 2}, tensors).digest
        )


class TestReplyClassification:
    def test_only_an_explicit_success_counts(self):
        assert classify_engine_response(200, {"success": True}).success
        assert classify_engine_response(200, {"success": True}).message == ""

    def test_client_error_is_a_clean_refusal(self):
        reply = classify_engine_response(400, {"success": False, "error_message": "stale attempt"})
        assert not reply.success and not reply.ambiguous
        assert "stale attempt" in reply.message

    def test_server_error_and_missing_success_are_ambiguous(self):
        assert classify_engine_response(500, {"success": False}).ambiguous
        assert classify_engine_response(200, {"weird": 1}).ambiguous


# --------------------------------------------------------------------------------------
# Protocol
# --------------------------------------------------------------------------------------


class TestPublication:
    def test_happy_path_streams_buckets_between_begin_and_end(self, registry):
        engines = _FakeEngines()
        outcome = _publisher(engines, registry).publish(_snapshot(), [1, 1], version_id=1)

        # One request per engine per phase, and the collective sits between fire and collect.
        assert engines.ops() == [
            "engine0:begin",
            "engine1:begin",
            "engine0:bucket",
            "engine1:bucket",
            "engine0:bucket",
            "engine1:bucket",
            "engine0:end",
            "engine1:end",
            "engine0:status",
            "engine1:status",
        ]
        assert [event[1] for event in engines.events if event[0] == "broadcast"] == ["0", "1"]
        assert outcome.status == "PUBLISHED" and outcome.bucket_count == 2
        assert registry.status().default_version == outcome.version_id
        assert registry.status().versions[outcome.version_id]["state"] == VersionState.PUBLISHED.value

    def test_bucket_payload_carries_identity_and_metadata(self, registry):
        engines = _FakeEngines()
        outcome = _publisher(engines, registry).publish(_snapshot(), [1, 1], version_id=1)
        bucket = next(payload for kind, op, payload in engines.events if op == "engine0:bucket")
        assert bucket["lora_name"] == outcome.lora_name
        assert bucket["attempt_id"] == 1
        assert bucket["group_name"] == "slime-pp_0"
        assert len(bucket["names"]) == len(bucket["dtypes"]) == len(bucket["shapes"]) == bucket["bucket_sizes"][0]

    def test_same_version_replay_is_a_no_op_without_any_transport(self, registry):
        """F19: replaying the same publication sends nothing."""

        engines = _FakeEngines()
        snapshot = _snapshot()
        first = _publisher(engines, registry).publish(snapshot, [1, 1], version_id=1)
        engines.events.clear()
        second = _publisher(engines, registry).publish(snapshot, [1, 1], version_id=1)

        assert second.status == "NO_OP" and second.version_id == first.version_id
        assert engines.events == []
        assert registry.status().capacity_owning == 1

    def test_publishing_while_a_retired_version_has_no_sessions_reclaims_it(self, registry):
        """The third distinct version fits only because the retired one is
        reclaimed first."""

        engines = _FakeEngines()
        a = registry.allocate(DIGEST_A, version_id=1)
        commit_ready(registry, a.version_id, a.attempt_id)
        b = registry.allocate(DIGEST_B, version_id=2)
        commit_ready(registry, b.version_id, b.attempt_id)

        outcome = _publisher(engines, registry).publish(_snapshot(), [1, 1], version_id=3)
        assert outcome.status == "PUBLISHED"
        # A was reclaimed before the allocation and B right after the commit, so only C owns a slot.
        assert registry.status().versions[a.version_id]["state"] == VersionState.RECLAIMED.value
        assert registry.status().versions[b.version_id]["state"] == VersionState.RECLAIMED.value
        assert registry.status().capacity_owning == 1
        assert engines.ops().count("engine0:unload") == 2

    def test_single_bucket_warns(self, registry, caplog):
        """F23: one data-bearing bucket must be reported, not silently accepted."""

        engines = _FakeEngines()
        with caplog.at_level(logging.WARNING):
            _publisher(engines, registry, bucket_cap=1024).publish(_snapshot(), [2], version_id=1)
        assert "overlap is not structurally guaranteed" in caplog.text


class TestFailureSemantics:
    def test_begin_failure_cleans_up_and_stays_retryable(self, registry):
        """F1/§13.1: default stays A, candidate back to ABSENT everywhere."""

        a = _publish_a(registry)
        engines = _FakeEngines()
        engines.fail("begin", "engine1", message="staged slot busy")

        with pytest.raises(LoRAPublicationError) as error:
            _publisher(engines, registry).publish(_snapshot(), [1, 1], version_id=2)
        assert error.value.kind == "RETRYABLE"
        # No collective was ever started, and the cleanup covered both engines.
        assert "bucket" not in " ".join(engines.ops())
        assert "engine0:unload" in engines.ops() and "engine1:unload" in engines.ops()
        assert registry.status().default_version == a
        failed = [entry for entry in registry.status().versions.values() if entry["state"] != "PUBLISHED"]
        assert failed[0]["state"] == VersionState.FAILED_RETRYABLE.value

    def test_unreachable_engine_makes_the_attempt_fatal(self, registry):
        """F34: cleanup that cannot be confirmed never becomes FAILED_RETRYABLE."""

        _publish_a(registry)
        engines = _FakeEngines()
        engines.silent.add("engine1")
        publisher = _publisher(engines, registry)

        with pytest.raises(LoRAPublicationError) as error:
            publisher.publish(_snapshot(), [1, 1], version_id=2)
        assert error.value.kind == "FATAL"
        assert publisher._registry.status().versions[2]["state"] == VersionState.FAILED_FATAL.value
        with pytest.raises(LoRAVersionError):
            registry.retry_publication(2, _snapshot().digest)

    def test_partial_end_success_cleans_up_the_ready_engine_too(self, registry):
        """F9/§13.3: a never-fleet-published READY_LOCAL candidate is unloaded, not reused."""

        a = _publish_a(registry)
        engines = _FakeEngines()
        engines.fail("end", "engine1", message="checksum mismatch")

        with pytest.raises(LoRAPublicationError) as error:
            _publisher(engines, registry).publish(_snapshot(), [1, 1], version_id=2)
        assert error.value.kind == "RETRYABLE"
        assert engines.ops()[-2:] == ["engine0:unload", "engine1:unload"]
        assert registry.status().default_version == a

    def test_unknown_end_state_is_fatal_and_not_retried(self, registry):
        """§13.5: a failed load may have written backend state, so the version never retries."""

        _publish_a(registry)
        engines = _FakeEngines()
        engines.fail("end", "engine1", ambiguous=True, message="HTTP 500: load crashed")
        publisher = _publisher(engines, registry)

        with pytest.raises(LoRAPublicationError) as error:
            publisher.publish(_snapshot(), [1, 1], version_id=2)
        assert error.value.kind == "FATAL"
        # The healthy engine was still cleaned up (best effort), but nothing is called ABSENT.
        assert "engine0:unload" in engines.ops()
        assert registry.status().versions[2]["state"] == VersionState.FAILED_FATAL.value

    def test_bucket_rejected_before_receive_is_fatal_and_keeps_capacity(self, registry):
        """A receiver that skips NCCL cannot yield a clean, retryable
        transfer."""

        a = _publish_a(registry)
        engines = _FakeEngines()
        engines.fail("bucket", "engine0", message="stale attempt id")

        with pytest.raises(LoRAPublicationError) as error:
            _publisher(engines, registry).publish(_snapshot(), [1, 1], version_id=2)
        assert error.value.kind == "FATAL"
        assert "engine0:end" not in engines.ops()
        assert "engine0:unload" not in engines.ops()
        assert registry.status().default_version == a
        assert registry.status().capacity_owning == 2

    def test_capacity_is_refused_before_any_begin(self, registry):
        """F13: two live versions, so a third never reaches the wire."""

        _publish_a(registry)
        registry.allocate(DIGEST_B, version_id=2)  # a second live version: no slot is free any more
        engines = _FakeEngines()

        with pytest.raises(LoRAVersionError) as error:
            _publisher(engines, registry).publish(_snapshot(), [1, 1], version_id=3)
        assert error.value.code == "CAPACITY_ERROR"
        assert engines.events == []

    def test_clean_failure_retries_in_place_with_a_new_attempt(self, registry):
        """F24/F30: same version, same name, next attempt — every engine Begins again."""

        _publish_a(registry)
        engines = _FakeEngines()
        engines.fail("end", "engine1", message="manifest mismatch")
        snapshot = _snapshot()
        with pytest.raises(LoRAPublicationError):
            _publisher(engines, registry).publish(snapshot, [1, 1], version_id=2)
        # The retry reuses the immutable name, so the cleanup must say *which* attempt
        # it is undoing; otherwise a late duplicate could strip the retry's instance.
        unloads = [payload for kind, op, payload in engines.events if kind == "fire" and op == "engine0:unload"]
        assert unloads and all(payload["attempt_id"] == 1 for payload in unloads)

        engines.replies.clear()
        engines.events.clear()
        outcome = _publisher(engines, registry).publish(snapshot, [1, 1], version_id=2)
        assert outcome.version_id == 2
        begins = [payload for kind, op, payload in engines.events if op == "engine0:begin"]
        assert begins[0]["attempt_id"] == 2
        assert begins[0]["lora_name"] == outcome.lora_name
        assert registry.status().versions[2]["state"] == VersionState.PUBLISHED.value


class TestReclaim:
    def test_reclaim_waits_for_the_bound_session(self, registry):
        """F14/F15: no unload while a Session can still request the version."""

        engines = _FakeEngines()
        publisher = _publisher(engines, registry)
        a = registry.allocate(DIGEST_A, version_id=1)
        commit_ready(registry, a.version_id, a.attempt_id)
        registry.bind_latest("s_old")
        b = registry.allocate(DIGEST_B, version_id=2)
        commit_ready(registry, b.version_id, b.attempt_id)

        assert publisher.reclaim_once() is False
        assert "engine0:unload" not in engines.ops()

        registry.release("s_old")
        assert publisher.reclaim_once() is True
        assert engines.ops().count("engine0:unload") == 1
        assert registry.status().versions[a.version_id]["state"] == VersionState.RECLAIMED.value
        # Reclaim stays name-scoped on purpose: a published version never gets a new
        # attempt, so there is nothing an attempt_id could fence against.
        reclaim = [payload for kind, op, payload in engines.events if kind == "fire" and op == "engine0:unload"][0]
        assert reclaim.get("attempt_id") is None

    def test_unconfirmed_unload_is_never_retried(self, registry):
        """F36/§17.3: ambiguous unload keeps the slot and fails the run closed."""

        engines = _FakeEngines()
        publisher = _publisher(engines, registry)
        a = registry.allocate(DIGEST_A, version_id=1)
        commit_ready(registry, a.version_id, a.attempt_id)
        b = registry.allocate(DIGEST_B, version_id=2)
        commit_ready(registry, b.version_id, b.attempt_id)

        engines.fail("unload", "engine1", ambiguous=True, message="timeout")
        with pytest.raises(LoRAPublicationError) as error:
            publisher.reclaim_once()
        assert error.value.kind == "FATAL"
        assert engines.ops().count("engine1:unload") == 1
        assert registry.status().versions[a.version_id]["state"] == VersionState.RECLAIMING.value
        assert registry.status().versions[a.version_id]["reclaim_fatal"] is True
        assert publisher.reclaim_once() is False  # never handed out again


class TestFleetContract:
    def test_empty_fleet_never_publishes(self, registry):
        """§11.4: a publication with no reachable engine must fail closed, not commit."""

        engines = _FakeEngines(engine_ids=())
        with pytest.raises(LoRAPublicationError) as error:
            _publisher(engines, registry).publish(_snapshot(), [1, 1], version_id=1)
        assert error.value.kind == "FATAL"
        assert registry.status().default_version is None


class TestBootstrap:
    def test_first_sync_uses_the_same_staged_protocol(self, registry):
        engines = _FakeEngines()
        outcome = _publisher(engines, registry).publish(_snapshot(), [1, 1], version_id=1)
        assert outcome.status == "PUBLISHED"
        assert engines.ops()[:2] == ["engine0:begin", "engine1:begin"]
        assert engines.ops()[-2:] == ["engine0:status", "engine1:status"]
        assert registry.bind_latest("s1").lora_name == outcome.lora_name


class TestReplaySafety:
    def test_reentrant_publish_cannot_rebroadcast_loading_attempt(self, registry):
        engines = _FakeEngines()
        snapshot = _snapshot()
        publisher = _publisher(engines, registry)
        original = publisher._broadcast

        def broadcast(names, index):
            before = list(engines.events)
            with pytest.raises(LoRAVersionError) as error:
                _publisher(engines, registry).publish(snapshot, [1, 1], version_id=1)
            assert error.value.code == "PUBLICATION_IN_PROGRESS"
            assert engines.events == before
            original(names, index)

        publisher._broadcast = broadcast
        assert publisher.publish(snapshot, [1, 1], version_id=1).status == "PUBLISHED"
        assert sum(event[0] == "broadcast" for event in engines.events) == 2

    def test_ambiguous_bucket_cannot_become_retryable_after_cleanup(self, registry):
        engines = _FakeEngines()
        engines.fail("bucket", "engine1", ambiguous=True, message="collective completion unknown")
        with pytest.raises(LoRAPublicationError) as error:
            _publisher(engines, registry).publish(_snapshot(), [1, 1], version_id=1)
        assert error.value.kind == "FATAL"
        assert registry.status().capacity_owning == 1
        assert registry.versions[1].state is VersionState.FAILED_FATAL
        assert "engine0:unload" not in engines.ops()

    def test_missing_bucket_reply_is_not_success(self, registry):
        engines = _FakeEngines()
        collect = engines.collect

        def lose_bucket_reply(pending):
            replies = collect(pending)
            if next(iter(pending.values()))[1] == "bucket":
                replies.pop("engine1")
            return replies

        engines.collect = lose_bucket_reply
        with pytest.raises(LoRAPublicationError):
            _publisher(engines, registry).publish(_snapshot(), [1, 1], version_id=1)
        assert registry.default_version is None
        assert registry.versions[1].state is VersionState.FAILED_FATAL

    def test_single_engine_cannot_publish(self, registry):
        engines = _FakeEngines(engine_ids=("engine0",))
        with pytest.raises(LoRAPublicationError):
            _publisher(engines, registry).publish(_snapshot(), [1, 1], version_id=1)
        assert registry.default_version is None
        assert not any(event[0] == "broadcast" for event in engines.events)

    def test_restart_between_end_and_commit_blocks_default_switch(self, registry):
        engines = _FakeEngines()
        collect = engines.collect

        def restarted(pending):
            replies = collect(pending)
            if next(iter(pending.values()))[1] == "status":
                replies["engine1"].receipt["engine_incarnation"] = "restarted"
            return replies

        engines.collect = restarted
        with pytest.raises(LoRAPublicationError):
            _publisher(engines, registry).publish(_snapshot(), [1, 1], version_id=1)
        assert registry.default_version is None
        assert registry.versions[1].state is VersionState.FAILED_FATAL


def test_replaying_reclaimed_snapshot_does_not_broadcast_or_change_default(registry):
    engines = _FakeEngines()
    publisher = _publisher(engines, registry)
    snapshot = _snapshot()
    a = publisher.publish(snapshot, [1, 1], version_id=1)
    b = publisher.publish(_snapshot({"changed": torch.ones(2)}), [1], version_id=2)
    assert registry.versions[a.version_id].state is VersionState.RECLAIMED
    engines.events.clear()
    replay = publisher.publish(snapshot, [1, 1], version_id=1)
    assert replay.status == "NO_OP"
    assert replay.version_id == a.version_id
    assert registry.default_version == b.version_id
    assert engines.events == []


@pytest.mark.parametrize("retain_a", [False, True])
def test_new_publication_can_restore_identical_historical_adapter(registry, retain_a):
    engines = _FakeEngines()
    publisher = _publisher(engines, registry)
    snapshot = _snapshot()
    a = publisher.publish(snapshot, [1, 1], version_id=1)
    if retain_a:
        registry.bind_latest("old-A")
    b = publisher.publish(_snapshot({"changed": torch.ones(2)}), [1], version_id=2)
    registry.bind_latest("old-B")
    engines.events.clear()
    if retain_a:
        with pytest.raises(LoRAVersionError) as error:
            publisher.publish(snapshot, [1, 1], version_id=3)
        assert error.value.code == "CAPACITY_ERROR"
        assert engines.events == []
        registry.release("old-A")
    collect = engines.collect

    def check_default_before_commit(pending):
        assert registry.default_version == b.version_id
        return collect(pending)

    engines.collect = check_default_before_commit
    c = publisher.publish(snapshot, [1, 1], version_id=3)
    assert c.status == "PUBLISHED" and c.version_id == 3
    assert c.lora_name != a.lora_name
    assert engines.ops().count("engine0:begin") == 1
    assert sum(event[0] == "broadcast" for event in engines.events) == 2
    assert registry.bind_latest("new").version_id == c.version_id
    assert registry.bind_latest("old-B").version_id == b.version_id
    assert registry.versions[a.version_id].state is VersionState.RECLAIMED
    engines.events.clear()
    for version_id in (1, 3):
        assert publisher.publish(snapshot, [1, 1], version_id=version_id).status == "NO_OP"
    assert registry.default_version == 3 and not engines.events


def test_conflict_and_completed_replay_do_not_reclaim_unrelated_versions(registry):
    engines = _FakeEngines()
    publisher = _publisher(engines, registry)
    snapshot = _snapshot()
    publisher.publish(snapshot, [1, 1], version_id=1)
    registry.bind_latest("old-A")
    publisher.publish(_snapshot({"changed": torch.ones(2)}), [1], version_id=2)
    registry.release("old-A")
    engines.events.clear()
    before = registry.status()
    with pytest.raises(LoRAVersionError) as error:
        publisher.publish(snapshot, [1, 1], version_id=2)
    assert error.value.code == "VERSION_CONFLICT"
    assert publisher.publish(snapshot, [1, 1], version_id=1).status == "NO_OP"
    assert registry.status() == before
    assert engines.events == []


def _device_direct_backend(registry, monkeypatch, *, base_sync_done=True, rank=0):
    from relax.distributed.checkpoint_service.backends import device_direct

    backend = object.__new__(device_direct.DeviceDirectBackend)
    backend.lock = SimpleNamespace(
        acquire=SimpleNamespace(remote=Mock(return_value=True)), release=SimpleNamespace(remote=Mock())
    )
    monkeypatch.setattr(device_direct.ray, "get", lambda result: result)
    monkeypatch.setattr(device_direct.dist, "get_rank", lambda: rank)
    group = object()
    monkeypatch.setattr(device_direct, "get_gloo_group", lambda: group)
    monkeypatch.setattr(device_direct.dist, "barrier", Mock())
    monkeypatch.setattr(device_direct.dist, "all_reduce", Mock())
    monkeypatch.setattr(device_direct.device_module, "empty_cache", Mock())
    backend.args = SimpleNamespace()
    backend.model = []
    backend.weight_version = 0
    backend._lora_merge_mode = False
    backend._lora_adapter_mode = True
    backend._versioned_lora = True
    backend._is_pp_src_rank = False
    backend._lora_sync = SimpleNamespace(base_sync_done=base_sync_done, adapter_loaded=base_sync_done)
    backend._megatron = SimpleNamespace(named_params_and_buffers=lambda *args: ())
    backend.rollout_engines = {rank: SimpleNamespace(flush_cache=SimpleNamespace(remote=Mock())) for rank in (0, 1)}
    backend._batch_request = Mock(return_value=[])
    engines = _FakeEngines()
    backend._lora_publication_bucket_cap = lambda: 8
    backend._new_lora_publisher = lambda snapshot, cap: _publisher(engines, registry, bucket_cap=cap)
    backend._materialize_adapter_snapshot = lambda: _snapshot() if rank == 0 else None
    return backend, engines


def test_device_direct_uses_sync_identity_for_new_and_replayed_publications(registry, monkeypatch):
    backend, engines = _device_direct_backend(registry, monkeypatch)
    snapshots = [_snapshot(), _snapshot({"changed": torch.ones(2)})]
    for version_id, snapshot in ((11, snapshots[0]), (11, snapshots[0]), (22, snapshots[1]), (33, snapshots[0])):
        backend.weight_version = version_id
        backend._materialize_adapter_snapshot = lambda: snapshot
        backend._publish_lora_adapter_versioned()
        assert registry.default_version == version_id
    begins = [payload for kind, op, payload in engines.events if kind == "fire" and op == "engine0:begin"]
    assert [payload["version_id"] for payload in begins] == [11, 22, 33]
    assert begins[0]["digest"] == begins[2]["digest"]
    assert begins[0]["lora_name"] != begins[2]["lora_name"]


@pytest.mark.parametrize("rank", [0, 1])
def test_device_direct_skips_capacity_and_publishes_after_session_release(registry, monkeypatch, caplog, rank):
    from relax.distributed.checkpoint_service.backends import device_direct

    backend, engines = _device_direct_backend(registry, monkeypatch, rank=rank)
    publisher = _publisher(engines, registry)
    publisher.publish(_snapshot(), [1, 1], version_id=1)
    old = registry.bind_latest("old-session")
    publisher.publish(_snapshot({"changed": torch.ones(2)}), [1], version_id=2)
    before = registry.status()
    engines.events.clear()
    backend.weight_version = 2

    with caplog.at_level(logging.WARNING):
        backend.update_weights_for_rollout(rollout_only=True)

    assert backend.weight_version == 3
    assert registry.status() == before
    assert registry.bind_latest("old-session") == old
    assert engines.events == []  # No Begin or NCCL for the refused candidate.
    assert backend._lora_sync.base_sync_done and backend._lora_sync.adapter_loaded
    device_direct.dist.all_reduce.assert_called_once()
    assert device_direct.dist.all_reduce.call_args.args[0].item() == 0
    device_direct.device_module.empty_cache.assert_called_once()
    if rank == 0:
        assert "sync v=3 SKIPPED: CAPACITY_ERROR" in caplog.text
        backend.lock.release.remote.assert_called_once()
        registry.release("old-session")
        backend.update_weights_for_rollout(rollout_only=True)
        assert registry.default_version == 4
        assert 3 not in registry.versions
        assert registry.versions[1].state is VersionState.RECLAIMED


@pytest.mark.parametrize(
    ("base_sync_done", "error"),
    [
        (False, LoRAVersionError("CAPACITY_ERROR")),
        (True, LoRAVersionError("PUBLICATION_BLOCKED")),
        (True, LoRAVersionError("VERSION_CONFLICT")),
        (True, LoRAPublicationError("FATAL", "ambiguous engine state")),
    ],
)
def test_device_direct_keeps_bootstrap_and_protocol_errors_fatal(registry, monkeypatch, base_sync_done, error):
    from relax.distributed.checkpoint_service.backends import device_direct

    backend, _ = _device_direct_backend(registry, monkeypatch, base_sync_done=base_sync_done)
    backend._new_lora_publisher = lambda snapshot, cap: SimpleNamespace(publish=Mock(side_effect=error))
    with pytest.raises(RuntimeError, match="LoRA adapter push.*failed") as raised:
        backend.update_weights_for_rollout(rollout_only=True)
    assert raised.value.__cause__ is error
    assert backend._lora_sync.base_sync_done is base_sync_done
    assert backend._lora_sync.adapter_loaded is base_sync_done
    backend.lock.release.remote.assert_called_once()
    device_direct.dist.all_reduce.assert_called_once()
    assert device_direct.dist.all_reduce.call_args.args[0].item() == 1


def test_registry_client_forwards_explicit_publication_identity(registry, monkeypatch):
    import ray

    handle = SimpleNamespace(allocate=SimpleNamespace(remote=registry.allocate))
    monkeypatch.setattr(ray, "get", lambda result: result)
    client = RayLoRAVersionRegistryClient(handle)
    publication = client.allocate(DIGEST_A, version_id=41)
    assert publication.version_id == 41
    assert client.allocate(DIGEST_A, version_id=41) == publication
    with pytest.raises(LoRAVersionError) as error:
        client.allocate(DIGEST_B, version_id=41)
    assert error.value.code == "VERSION_CONFLICT"
