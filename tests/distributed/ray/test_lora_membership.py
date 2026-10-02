# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Actual RM integration; skipped explicitly when its optional runtime is
absent."""

import asyncio
import threading
from types import SimpleNamespace

import pytest
from conftest import HAS_DEPS, create_test_manager

from relax.engine.lora.publication import EngineIdentity


pytestmark = pytest.mark.skipif(not HAS_DEPS, reason="requires Ray/SGLang to import the real RolloutManager")


def remote(function):
    return SimpleNamespace(remote=function)


@pytest.mark.parametrize("exit_confirmed", [True, False])
async def test_managed_removal_requires_native_drain_then_process_exit(exit_confirmed, monkeypatch):
    owner = create_test_manager()
    owner._lora_profile = object()
    owner._lora_process_exit_proofs = set()
    from relax.distributed.ray import rollout

    killed = []
    monkeypatch.setattr(rollout.ray, "kill", lambda actor, **kwargs: killed.append(actor))
    owner._lora_loop = SimpleNamespace(loop=asyncio.get_running_loop())
    identity = EngineIdentity("e3", "boot", "http://e3")
    events = []
    gate = asyncio.Event()

    async def detach(identities):
        assert identities == [identity]
        events.append("wait-session-drain")
        await gate.wait()
        events.append("absent")

    async def capacity():
        events.append("capacity")

    owner._sync_lora_permit_capacity = capacity

    async def shutdown():
        events.append("shutdown")
        return {"processes_exited": exit_confirmed}

    engine = SimpleNamespace(shutdown=remote(shutdown))
    group = SimpleNamespace(engines=[engine], all_engines=[engine], rank_offset=3)
    owner._engine_lifecycle_lock = threading.RLock()
    owner._lora_clients = {"e3": SimpleNamespace(handle=engine, identity=identity)}
    owner._lora_manager = SimpleNamespace(engine_identities={"e3": identity}, remove_engines=detach)
    owner._get_live_engine_actors = lambda group, index: [(0, engine)]
    owner._cleanup_engine_groups = lambda srv: events.append("check-empty-groups")
    waiter = asyncio.create_task(
        owner._remove_live_engines(object(), [(group, 0)], drain_timeout=0, shutdown_timeout=1, force=True)
    )
    for _ in range(30):
        if events:
            break
        await asyncio.sleep(0)
    assert events == ["wait-session-drain"]  # force and zero timeout cannot skip it.
    gate.set()
    removed, failed = await waiter
    assert events == ["wait-session-drain", "absent", "capacity", "shutdown", "check-empty-groups"]
    label = "group_3_engine_0"
    if exit_confirmed:
        assert removed == [label] and not failed
        assert group.all_engines == [None]
        assert killed == [engine]
    else:
        assert failed == [label] and not removed
        assert group.all_engines == [engine]
        assert not killed
        assert owner._lora_clients["e3"].handle is engine


async def test_memory_handoff_uses_each_boots_sequence_after_scale_out():
    owner = create_test_manager()
    owner._lora_profile = SimpleNamespace(cleanup_timeout_seconds=1)
    identities = {str(i): EngineIdentity(str(i), f"b{i}", f"http://e{i}") for i in range(3)}
    calls = []
    owner._lora_clients = {}
    # e2 just joined; it has not seen the old cohort's eight handoffs.
    owner._lora_memory_sequences = {"0": 8, "1": 8, "2": 0}
    native = {}
    for key, identity in identities.items():
        native[key] = owner._lora_memory_sequences[key]

        async def release(*, memory_sequence, memory_owner, key=key, identity=identity):
            assert memory_owner == ("cohort", identity.boot_id)
            assert memory_sequence == native[key] + 1
            native[key] = memory_sequence
            calls.append((key, memory_sequence))

        async def capability(key=key, identity=identity):
            return {
                "engine_id": key,
                "engine_boot_id": identity.boot_id,
                "memory_sequence": native[key],
                "memory_state": "SUSPENDED",
            }

        owner._lora_clients[key] = SimpleNamespace(
            handle=SimpleNamespace(
                release_memory_occupation=remote(release), lora_publication_identity=remote(capability)
            )
        )

    async def settled():
        pass

    owner._lora_manager = SimpleNamespace(
        cohort_id="cohort",
        engine_identities=identities,
        suspended=False,
        suspend=lambda: None,
        settle_control=settled,
    )
    await owner._change_lora_memory("suspend")
    assert sorted(calls) == [("0", 9), ("1", 9), ("2", 1)]
    assert owner.status == "offload"
