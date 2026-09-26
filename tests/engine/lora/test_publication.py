# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Publication linearization and ownership; these are not native/GPU tests."""

import asyncio
from collections import Counter, defaultdict
from dataclasses import replace

import pytest

from relax.engine.lora.publication import (
    AdapterPublicationError,
    AdapterVersionManager,
    EngineIdentity,
    EngineReceipt,
    SessionReceipt,
    native_instance_id,
)
from relax.engine.lora.snapshot import snapshot_adapter


BASE = "a" * 64


class Engine:
    def __init__(self, name):
        self.identity = EngineIdentity(name, "boot", f"http://{name}")
        self.loaded = {}
        self.fenced = set()
        self.loads = Counter()
        self.unloads = Counter()
        self.prepare_started = defaultdict(asyncio.Event)
        self.prepare_gate = {}
        self.close_gate = {}
        self.fail_prepare = set()
        self.fail_retire = set()
        self.lose_unload_ack = set()
        self.closed = set()
        self.receipt_changes = {}

    def receipt(self, snapshot, cohort, operation, state):
        return EngineReceipt(
            self.identity,
            cohort,
            operation,
            snapshot.version_id,
            snapshot.digest,
            native_instance_id(self.identity, cohort, operation),
            state,
            pinned=state == "READY",
            resident=state == "READY",
            fenced=state == "ABSENT",
        )

    async def prepare(self, snapshot, cohort, operation):
        self.loads[snapshot.version_id] += 1
        self.prepare_started[snapshot.version_id].set()
        if snapshot.version_id in self.prepare_gate:
            await self.prepare_gate[snapshot.version_id].wait()
        if operation in self.fenced:
            raise RuntimeError("VERSION_FENCED")
        self.loaded[operation] = snapshot.version_id
        if snapshot.version_id in self.fail_prepare:
            raise RuntimeError("partial load failed")
        return replace(self.receipt(snapshot, cohort, operation, "READY"), **self.receipt_changes)

    async def retire(self, snapshot, cohort, operation):
        self.fenced.add(operation)
        if snapshot.version_id in self.fail_retire:
            raise TimeoutError("unknown cleanup")
        if self.loaded.pop(operation, None) is not None:
            self.unloads[snapshot.version_id] += 1
        if snapshot.version_id in self.lose_unload_ack:
            self.lose_unload_ack.remove(snapshot.version_id)
            raise TimeoutError("lost completed unload ACK")
        return self.receipt(snapshot, cohort, operation, "ABSENT")

    async def close_session(self, cohort, owner, sid):
        self.closed.add((owner, sid))
        if sid in self.close_gate:
            await self.close_gate[sid].wait()
        return SessionReceipt(self.identity, cohort, owner, sid, "SESSION_DRAINED")


def artifact(tmp_path, version, content=None):
    source = tmp_path / f"source-{version}-{content}"
    source.mkdir(exist_ok=True)
    (source / "adapter_config.json").write_text('{"r":4}')
    (source / "adapter_model.safetensors").write_text(content or version)
    return snapshot_adapter(source, tmp_path / f"store-{content}", version_id=version, base_model_digest=BASE)


def manager(engines, **kwargs):
    return AdapterVersionManager(engines, cohort_id="cohort", capacity=2, base_model_digest=BASE, **kwargs)


async def wait_state(owner, operation, state):
    for _ in range(1000):
        await owner.collect()
        observed = owner.status(operation)
        if observed["state"] == state:
            return observed
        await asyncio.sleep(0)
    raise AssertionError(owner.status())


async def publish(owner, snapshot, **kwargs):
    result = await owner.publish(snapshot, snapshot.version_id, **kwargs)
    return await wait_state(owner, result["operation_id"], "PUBLISHED")


@pytest.mark.parametrize("engine_count", [2, 3, 8])
async def test_all_ready_linearization_session_affinity_capacity_and_once_only_unload(tmp_path, engine_count):
    engines = [Engine(f"E{i}") for i in range(engine_count)]
    owner = manager(engines)
    a, b, c = [artifact(tmp_path, name) for name in "ABC"]
    await publish(owner, a)
    old = owner.bind_session("shard", "old")
    engines[1].prepare_gate["B"] = asyncio.Event()
    op = (await owner.publish(b, "B"))["operation_id"]
    await engines[1].prepare_started["B"].wait()
    between = owner.bind_session("shard", "between")
    assert between["binding"]["version_id"] == "A"
    assert owner.status()["default"]["version_id"] == "A"
    engines[1].prepare_gate["B"].set()
    await wait_state(owner, op, "PUBLISHED")
    assert owner.bind_session("shard", "old") == old
    assert owner.bind_session("shard", "new")["binding"]["version_id"] == "B"
    with pytest.raises(AdapterPublicationError, match="ADAPTER_CAPACITY_EXCEEDED"):
        await owner.publish(c, "C")
    assert all(engine.loads["C"] == 0 for engine in engines)
    gate = engines[1].close_gate["old"] = asyncio.Event()
    await owner.close_session("shard", "old")
    await owner.close_session("shard", "between")
    for _ in range(15):
        await asyncio.sleep(0)
    assert owner.status()["versions"]["A"]["session_refs"] >= 1
    assert all(engine.unloads["A"] == 0 for engine in engines)
    gate.set()
    await wait_state(owner, old["binding"]["publication_id"], "RETIRED")
    await owner.close_session("shard", "old")
    assert all(engine.unloads["A"] == 1 for engine in engines)
    await publish(owner, c)
    assert owner.status()["default"]["version_id"] == "C"


async def test_intent_replay_conflict_explicit_retry_and_old_operation_isolation(tmp_path):
    engines = [Engine("E1"), Engine("E2")]
    owner = manager(engines)
    b = artifact(tmp_path, "B")
    engines[1].fail_prepare.add("B")
    op1 = (await owner.publish(b, "intent-1"))["operation_id"]
    await wait_state(owner, op1, "ABORTED")
    assert (await owner.publish(b, "intent-1"))["operation_id"] == op1
    assert (await owner.publish(b, "ordinary-replay"))["state"] == "ABORTED"
    with pytest.raises(AdapterPublicationError, match="REQUEST_ID_CONFLICT"):
        await owner.publish(b, "intent-1", retry_of=op1)
    with pytest.raises(AdapterPublicationError, match="VERSION_CONTENT_CONFLICT"):
        await owner.publish(artifact(tmp_path, "B", "different"), "conflict")
    engines[1].fail_prepare.clear()
    op2 = (await owner.publish(b, "retry", retry_of=op1))["operation_id"]
    assert op2 != op1
    with pytest.raises(AdapterPublicationError, match="INVALID_PUBLICATION_RETRY"):
        await owner.publish(b, "competing-retry", retry_of=op1)
    await wait_state(owner, op2, "PUBLISHED")
    await owner.cancel_publication(op1)
    for engine in engines:
        await engine.retire(b, "cohort", op1)
        assert op2 in engine.loaded
    assert owner.status()["default"]["publication_id"] == op2


async def test_failure_unknown_cleanup_and_lost_ack_keep_slot_until_both_absent(tmp_path):
    engines = [Engine("E1"), Engine("E2")]
    owner = manager(engines)
    await publish(owner, artifact(tmp_path, "A"))
    owner.bind_session("shard", "old")
    engines[1].fail_prepare.add("B")
    engines[1].fail_retire.add("B")
    engines[0].lose_unload_ack.add("B")
    op = (await owner.publish(artifact(tmp_path, "B"), "B"))["operation_id"]
    await wait_state(owner, op, "RETIRING")
    for _ in range(30):
        await owner.collect()
        await asyncio.sleep(0)
    assert owner.status()["occupied"] == 2
    assert owner.status()["default"]["version_id"] == "A"
    assert owner.status(op)["absent"] == ["E1"]
    engines[1].fail_retire.clear()
    await wait_state(owner, op, "ABORTED")
    assert owner.status()["occupied"] == 1
    assert [engine.unloads["B"] for engine in engines] == [1, 1]


async def test_cancel_fences_delayed_prepare_and_cannot_rollback_committed_default(tmp_path):
    engines = [Engine("E1"), Engine("E2")]
    owner = manager(engines)
    a = await publish(owner, artifact(tmp_path, "A"))
    gate = engines[0].prepare_gate["B"] = asyncio.Event()
    op = (await owner.publish(artifact(tmp_path, "B"), "B"))["operation_id"]
    await engines[0].prepare_started["B"].wait()
    await owner.cancel_publication(op)
    for _ in range(20):
        await asyncio.sleep(0)
    assert all(op in engine.fenced for engine in engines)
    assert owner.status()["occupied"] == 2
    gate.set()
    await wait_state(owner, op, "ABORTED")
    assert all(op not in engine.loaded for engine in engines)
    with pytest.raises(AdapterPublicationError, match="ALREADY_COMMITTED"):
        await owner.cancel_publication(a["operation_id"])
    assert owner.status()["default"]["version_id"] == "A"


async def test_close_before_bind_idempotent_ref_release_and_metadata_exhaustion(tmp_path):
    engines = [Engine("E1"), Engine("E2")]
    owner = manager(engines, max_lifecycle_records=5)
    await publish(owner, artifact(tmp_path, "A"))  # version, operation, intent
    await owner.close_session("shard", "before-bind")
    with pytest.raises(AdapterPublicationError, match="SESSION_CLOSED"):
        owner.bind_session("shard", "before-bind")
    owner.bind_session("shard", "last")
    owner.bind_session("shard", "last")
    assert owner.status()["versions"]["A"]["session_refs"] == 1
    with pytest.raises(AdapterPublicationError, match="LIFECYCLE_CAPACITY_EXCEEDED"):
        await owner.close_session("shard", "no-space-for-fence")
    assert not owner.status()["accepting"]
    await owner.close_session("shard", "last")
    for _ in range(30):
        await owner.collect()
        await asyncio.sleep(0)
    assert owner.session_status("shard", "last")["state"] == "CLOSED"
    assert owner.status()["versions"]["A"]["session_refs"] == 0
    assert all(engine.unloads["A"] == 0 for engine in engines)  # default retains A


@pytest.mark.parametrize(
    "changes",
    [
        {"resident": False},
        {"pinned": False},
        {"cohort_id": "old"},
        {"native_lora_id": None},
        {"digest": "b" * 64},
    ],
)
async def test_invalid_readiness_never_commits(tmp_path, changes):
    engines = [Engine("E1"), Engine("E2")]
    owner = manager(engines)
    await publish(owner, artifact(tmp_path, "A"))
    previous_default = owner.status()["default"]
    engines[1].receipt_changes = changes
    op = (await owner.publish(artifact(tmp_path, "B"), "B"))["operation_id"]
    result = await wait_state(owner, op, "ABORTED")
    assert owner.status()["default"] == previous_default
    assert owner.status()["occupied"] == 1
    assert set(result["absent"]) == {"E1", "E2"}
    assert all(op not in engine.loaded for engine in engines)
    assert [engine.unloads["B"] for engine in engines] == [1, 1]
    assert all(engine.unloads["A"] == 0 for engine in engines)


async def test_unhealthy_fixed_target_is_still_required_for_cleanup(tmp_path):
    engines = [Engine("E1"), Engine("E2")]
    owner = manager(engines)
    a = await publish(owner, artifact(tmp_path, "A"))
    owner.bind_session("shard", "old")
    await publish(owner, artifact(tmp_path, "B"))
    owner.mark_engine_unavailable(engines[1].identity)
    engines[1].fail_retire.add("A")
    await owner.close_session("shard", "old")
    await wait_state(owner, a["operation_id"], "RETIRING")
    for _ in range(20):
        await owner.collect()
        await asyncio.sleep(0)
    assert owner.status()["occupied"] == 2
    assert owner.status(a["operation_id"])["absent"] == ["E1"]


async def test_deadline_is_checked_at_commit_even_if_event_loop_was_blocked(tmp_path):
    import time

    engines = [Engine("E1"), Engine("E2")]
    original = engines[1].prepare

    async def delayed(*args):
        receipt = await original(*args)
        time.sleep(0.02)  # Model a late callback occupying the owner's loop.
        return receipt

    engines[1].prepare = delayed
    owner = manager(engines, prepare_timeout_seconds=0.005)
    op = (await owner.publish(artifact(tmp_path, "A"), "A"))["operation_id"]
    await wait_state(owner, op, "ABORTED")
    assert owner.status()["default"] is None
    assert "PREPARE_DEADLINE_EXCEEDED" in owner.status(op)["errors"]["prepare"]


async def test_target_verification_can_exceed_status_poll_budget(tmp_path):
    engines = [Engine("E1"), Engine("E2")]
    original = engines[0].prepare

    async def verify_then_prepare(*args):
        # No native load exists until the target has verified the artifact.
        await asyncio.sleep(5.2)
        return await original(*args)

    engines[0].prepare = verify_then_prepare
    owner = manager(engines, prepare_timeout_seconds=15)
    result = await owner.publish(artifact(tmp_path, "A"), "A")
    await asyncio.wait_for(owner._operations[result["operation_id"]].prepare_task, 10)
    assert owner.status(result["operation_id"])["state"] == "PUBLISHED"
    assert owner.status()["default"]["version_id"] == "A"
    assert all(engine.loads["A"] == 1 and not engine.unloads for engine in engines)


async def test_lost_prepare_response_queries_same_operation_within_total_budget(tmp_path):
    engines = [Engine("E1"), Engine("E2")]
    original = engines[0].prepare

    async def lost(*args):
        await original(*args)
        raise TimeoutError("HTTP response lost")

    async def status(snapshot, cohort, operation):
        assert operation in engines[0].loaded
        return engines[0].receipt(snapshot, cohort, operation, "READY")

    engines[0].prepare = lost
    engines[0].status = status
    owner = manager(engines)
    result = await owner.publish(artifact(tmp_path, "A"), "A")
    # Wait for the accepted native prepare, then the bounded status poll.
    await asyncio.wait_for(owner._operations[result["operation_id"]].prepare_task, 1)
    assert owner.status(result["operation_id"])["state"] == "PUBLISHED"
    assert engines[0].loads["A"] == 1


async def test_accepted_intent_lookup_needs_no_artifact_read_and_still_checks_payload(tmp_path):
    owner = manager([Engine("E1"), Engine("E2")])
    snapshot = artifact(tmp_path, "A")
    result = await publish(owner, snapshot)
    assert owner.publication_intent("A", "A") == result
    with pytest.raises(AdapterPublicationError, match="REQUEST_ID_CONFLICT"):
        owner.publication_intent("A", "B")
    with pytest.raises(AdapterPublicationError, match="REQUEST_ID_CONFLICT"):
        owner.publication_intent("A", "A", "f" * 64)
    assert owner.publication_intent("unseen", "A") is None


def test_history_does_not_enter_bind_or_background_collection_hot_path(tmp_path):
    class History(dict):
        def values(self):
            raise AssertionError("hot path scanned historical records")

    async def scenario():
        engines = [Engine("E1"), Engine("E2")]
        owner = manager(engines)
        await publish(owner, artifact(tmp_path, "A"))
        owner._sessions = History(owner._sessions)
        owner._operations = History(owner._operations)
        first = owner.bind_session("owner", "first")
        second = owner.bind_session("owner", "second")
        assert first["engine"] != second["engine"]
        await owner.close_session("owner", "first")
        for _ in range(10):
            await owner.collect(report=False)
            await asyncio.sleep(0)
        assert owner.bind_session("owner", "third")["engine"] == first["engine"]
        # A repeated close must not return a second unit of engine capacity.
        await owner.close_session("owner", "first")
        assert owner._engine_sessions == {"E1": 1, "E2": 1}

    asyncio.run(scenario())


async def test_memory_handoff_keeps_bindings_and_capacity_while_deferring_cleanup(tmp_path):
    engines = [Engine("E0"), Engine("E1")]
    owner = manager(engines)
    a, b, c = (artifact(tmp_path, version) for version in ("A", "B", "C"))
    await publish(owner, a)
    binding = owner.bind_session("owner", "old")
    gate = engines[1].prepare_gate["B"] = asyncio.Event()
    operation = await owner.publish(b, "B")
    await engines[1].prepare_started["B"].wait()
    owner.suspend()
    settled = asyncio.create_task(owner.settle_control())
    await asyncio.sleep(0)
    assert not settled.done()
    with pytest.raises(AdapterPublicationError, match="ENGINE_SUSPENDED"):
        owner.bind_session("owner", "new")
    gate.set()
    await settled
    assert owner.status(operation["operation_id"])["state"] == "PUBLISHED"
    assert owner.bind_session("owner", "old") == binding
    await owner.close_session("owner", "old")
    await owner.collect()
    assert all(not engine.closed and not engine.unloads for engine in engines)
    with pytest.raises(AdapterPublicationError, match="ENGINE_SUSPENDED"):
        await owner.publish(c, "C")
    owner.resume()
    await owner.collect()
    await wait_state(owner, binding["binding"]["publication_id"], "RETIRED")
    assert all(engine.unloads["A"] == 1 for engine in engines)
    await publish(owner, c)


async def test_join_waits_for_all_retained_versions_and_preserves_publication_targets(tmp_path):
    e1, e2, e3 = Engine("e1"), Engine("e2"), Engine("e3")
    owner = manager([e1, e2])
    a = await owner.publish(artifact(tmp_path, "A"), "a")
    await wait_state(owner, a["operation_id"], "PUBLISHED")
    old = owner.bind_session("owner", "old")
    b = await owner.publish(artifact(tmp_path, "B"), "b")
    await wait_state(owner, b["operation_id"], "PUBLISHED")
    e3.prepare_gate["B"] = asyncio.Event()
    joining = asyncio.create_task(owner.join_engines([e3]))
    await e3.prepare_started["B"].wait()
    for i in range(5):
        assert owner.bind_session("owner", f"during-{i}")["engine"]["engine_id"] != "e3"
    assert owner.status(b["operation_id"])["publish_targets"] == ["e1", "e2"]
    assert owner.status()["default_epoch"] == 2
    e3.prepare_gate["B"].set()
    await joining
    assert owner.bind_session("owner", "after")["engine"]["engine_id"] == "e3"
    assert owner.bind_session("owner", "old") == old
    await owner.close_session("owner", "old")
    await wait_state(owner, a["operation_id"], "RETIRED")
    assert e3.unloads["A"] == 1
    assert set(owner.status(a["operation_id"])["absent"]) == {"e1", "e2", "e3"}


async def test_remove_waits_for_tool_session_and_native_close_after_waiter_cancel(tmp_path):
    engines = [Engine(f"e{i}") for i in range(3)]
    owner = manager(engines)
    a = await owner.publish(artifact(tmp_path, "A"), "a")
    await wait_state(owner, a["operation_id"], "PUBLISHED")
    owner.bind_session("o", "zero")
    owner.bind_session("o", "one")
    retained = owner.bind_session("o", "two")
    assert retained["engine"]["engine_id"] == "e2"
    engines[2].close_gate["two"] = asyncio.Event()
    waiter = asyncio.create_task(owner.remove_engines([engines[2].identity]))
    await asyncio.sleep(0)
    waiter.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiter
    assert owner.bind_session("o", "two") == retained
    assert owner.bind_session("o", "next")["engine"]["engine_id"] != "e2"
    assert engines[2].unloads["A"] == 0
    await owner.close_session("o", "two")
    await asyncio.sleep(0)
    assert engines[2].unloads["A"] == 0
    engines[2].close_gate["two"].set()
    await owner.remove_engines([engines[2].identity])
    assert engines[2].unloads["A"] == 1
    assert owner.status()["occupied"] == 1
    # Other Sessions' close inventories retain the detached engine; its
    # exact-instance ABSENT proof substitutes for contacting a dead process.
    await owner.close_session("o", "zero")
    for _ in range(100):
        await owner.collect()
        if owner.session_status("o", "zero")["state"] == "CLOSED":
            break
        await asyncio.sleep(0)
    assert owner.session_status("o", "zero")["state"] == "CLOSED"
    with pytest.raises(AdapterPublicationError, match="MINIMUM_ENGINE_CAPACITY"):
        await owner.remove_engines([engines[0].identity])


async def test_failed_join_keeps_cleanup_owner_until_absent_and_never_becomes_route(tmp_path):
    e1, e2, e3 = Engine("e1"), Engine("e2"), Engine("e3")
    owner = manager([e1, e2])
    a = await owner.publish(artifact(tmp_path, "A"), "a")
    await wait_state(owner, a["operation_id"], "PUBLISHED")
    e3.fail_prepare.add("A")
    e3.fail_retire.add("A")
    joining = asyncio.create_task(owner.join_engines([e3]))
    await e3.prepare_started["A"].wait()
    for _ in range(20):
        await asyncio.sleep(0)
    assert not joining.done()
    assert owner.status()["membership_pending"]
    assert owner.status()["serving_engines"] == ["e1", "e2"]
    assert "e3" in owner.engine_identities
    with pytest.raises(AdapterPublicationError, match="MEMBERSHIP_BUSY"):
        await owner.publish(artifact(tmp_path, "B"), "b")
    e3.fail_retire.clear()
    with pytest.raises(RuntimeError, match="partial load failed"):
        await joining
    assert e3.unloads["A"] == 1
    assert "e3" not in owner.engine_identities
    b = await owner.publish(artifact(tmp_path, "B"), "b")
    await wait_state(owner, b["operation_id"], "PUBLISHED")
    assert owner.status(b["operation_id"])["publish_targets"] == ["e1", "e2"]


async def test_offload_during_scale_in_keeps_parked_session_and_defers_cleanup(tmp_path):
    engines = [Engine(f"e{i}") for i in range(3)]
    owner = manager(engines)
    result = await owner.publish(artifact(tmp_path, "A"), "a")
    await wait_state(owner, result["operation_id"], "PUBLISHED")
    for sid in ("zero", "one", "parked"):
        owner.bind_session("o", sid)
    leaving = asyncio.create_task(owner.remove_engines([engines[2].identity]))
    for _ in range(100):
        if owner.status()["membership_phase"] == "DRAINING":
            break
        await asyncio.sleep(0)
    owner.suspend()
    await owner.close_session("o", "parked")
    assert owner.status(result["operation_id"])["session_refs"] == 3
    assert not leaving.done() and not engines[2].unloads
    owner.resume()
    await owner.collect()
    await leaving
    assert engines[2].unloads["A"] == 1
