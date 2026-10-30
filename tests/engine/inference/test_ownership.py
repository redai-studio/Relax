# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import asyncio
from types import SimpleNamespace

import pytest

from relax.distributed.ray.placement_planner import Phase, PlacementClaim, PlacementPlan
from relax.engine.inference.ownership import OwnershipError, OwnershipLedger, ResourceBlockedError, StaleLeaseError
from relax.engine.inference.phase_client import gather_owned


def claim(role="rollout", model="default", start=0, phase=Phase.GENERATE):
    return PlacementClaim(role, model, "actor", start, 1, frozenset({phase}))


async def test_ownership_interleaved_callers_and_disjoint_resources():
    a, b = claim(), claim("teacher", start=1)
    ledger = OwnershipLedger(PlacementPlan((), (a, b)))
    first = await ledger.acquire("workload", "first", [a])
    blocked = asyncio.create_task(ledger.acquire("trainer", "second", [a]))
    independent = await ledger.acquire("teacher", "other", [b])
    await asyncio.sleep(0)
    assert not blocked.done()
    await ledger.release(first)
    second = await blocked
    assert second.generation > first.generation
    with pytest.raises(StaleLeaseError):
        ledger.check(first)
    ledger.check(independent)
    ledger.check(second)


async def test_ownership_retries_do_not_acquire_a_new_generation():
    c = claim()
    ledger = OwnershipLedger(PlacementPlan((), (c,)))
    one, two = await asyncio.gather(ledger.acquire("worker", "op", [c]), ledger.acquire("worker", "op", [c]))
    assert one == two
    with pytest.raises(OwnershipError, match="operation ID"):
        await ledger.acquire("other", "op", [c])
    await ledger.release(one)
    assert await ledger.acquire("worker", "op", [c]) == one
    with pytest.raises(StaleLeaseError):
        ledger.check(one)


async def test_ownership_unconfirmed_release_quarantines_the_original_claims():
    c = claim()
    ledger = OwnershipLedger(PlacementPlan((), (c,)))
    lease = await ledger.acquire("worker", "op", [c])
    waiter = asyncio.create_task(ledger.acquire("new", "new-op", [c]))
    await ledger.block(lease, "old backend still resident")
    await ledger.release(lease)
    with pytest.raises(ResourceBlockedError, match="still resident"):
        await waiter
    assert ledger.snapshot()["blocked"] == {"op": "old backend still resident"}


async def test_ownership_release_waits_for_inflight_switch_and_old_commands_are_fenced():
    c = claim()
    ledger = OwnershipLedger(PlacementPlan((), (c,)))
    entered, finish = asyncio.Event(), asyncio.Event()
    events = []

    async def apply(model, action, lease, command, tags=None, parent_command=None):
        ledger.check(lease, [c])
        events.append(action)
        entered.set()
        await finish.wait()

    ledger.register_model("rollout", "default", SimpleNamespace(apply=SimpleNamespace(remote=apply)), c)
    lease = await ledger.acquire("worker", "op", [c])
    switch = asyncio.create_task(ledger.command(lease, "switch", "generate"))
    await entered.wait()
    returned = asyncio.create_task(ledger.release(lease))
    await asyncio.sleep(0)
    assert not returned.done()
    finish.set()
    await asyncio.gather(switch, returned)
    await ledger.command(lease, "switch", "generate")  # cached receipt, no new onload
    with pytest.raises(StaleLeaseError):
        await ledger.command(lease, "late", "generate")
    assert events == ["onload"]


async def test_ownership_partial_activation_is_compensated_before_next_owner():
    rollout, judge = claim(), claim("genrm", "judge", phase=Phase.SCORE)
    ledger = OwnershipLedger(PlacementPlan((), (rollout, judge)))
    events = []

    async def apply(model, action, lease, command, tags=None, parent_command=None):
        events.append((model, action))
        if model == "judge" and action == "onload":
            raise RuntimeError("partially allocated")

    handle = SimpleNamespace(apply=SimpleNamespace(remote=apply))
    ledger.register_model("rollout", "default", handle, rollout)
    ledger.register_model("genrm", "judge", handle, judge)
    lease = await ledger.acquire("worker", "op", [rollout, judge])
    with pytest.raises(OwnershipError, match="partially allocated"):
        await ledger.command(lease, "score", "score")
    assert events == [("default", "offload"), ("judge", "onload"), ("judge", "offload")]


async def test_owned_batch_finishes_siblings_even_when_cleanup_is_cancelled_again():
    entered, cleaning, finish = asyncio.Event(), asyncio.Event(), asyncio.Event()
    events = []

    async def slow():
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            cleaning.set()
            await finish.wait()
            events.append("closed")

    async def fail():
        await entered.wait()
        raise ValueError("scoring failed")

    batch = asyncio.create_task(gather_owned([slow(), fail()]))
    await cleaning.wait()
    batch.cancel()
    await asyncio.sleep(0)
    assert not batch.done()
    finish.set()
    with pytest.raises(ValueError, match="scoring failed"):
        await batch
    assert events == ["closed"]


async def test_cancelled_acquire_returns_late_grant_without_leaking_ownership():
    from relax.engine.inference.phase_client import PhaseSession

    c = claim()
    ledger = OwnershipLedger(PlacementPlan((), (c,)))
    held = await ledger.acquire("held", "held", [c])

    async def claims(role):
        return [c]

    authority = SimpleNamespace(
        claims_for=SimpleNamespace(remote=claims),
        acquire=SimpleNamespace(remote=ledger.acquire),
        release=SimpleNamespace(remote=ledger.release),
    )
    session = PhaseSession(authority, "rollout", "cancelled")
    pending = asyncio.create_task(session.__aenter__())
    await asyncio.sleep(0)
    pending.cancel()
    await asyncio.sleep(0)
    assert not pending.done()
    await ledger.release(held)
    with pytest.raises(asyncio.CancelledError):
        await pending
    assert ledger.snapshot()["owners"] == {}
    assert session.closed


async def test_late_direct_mutation_is_fenced_when_return_begins():
    c = claim()
    ledger = OwnershipLedger(PlacementPlan((), (c,)))
    lease = await ledger.acquire("worker", "op", [c])
    ledger.begin(lease, "inflight")
    returned = asyncio.create_task(ledger.release(lease))
    await asyncio.sleep(0)
    with pytest.raises(StaleLeaseError, match="being returned"):
        ledger.begin(lease, "late")
    await ledger.finish(lease, "inflight")
    await returned


async def test_valid_lease_cannot_activate_overlapping_models_without_release_receipt():
    rollout, judge = claim(), claim("genrm", "judge", phase=Phase.SCORE)
    ledger = OwnershipLedger(PlacementPlan((), (rollout, judge)))
    ledger.register_model("rollout", "default", None, rollout)
    ledger.register_model("genrm", "judge", None, judge)
    lease = await ledger.acquire("workload", "op", [rollout, judge])
    ledger.reserve_activation(lease, "rollout", "default")
    with pytest.raises(OwnershipError, match="must confirm release"):
        ledger.reserve_activation(lease, "genrm", "judge")
    ledger.confirm_deactivation(lease, "rollout", "default")
    ledger.reserve_activation(lease, "genrm", "judge")
    assert len(ledger.snapshot()["resident"]) == 1
