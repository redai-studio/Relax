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


async def test_dedicated_leases_do_not_queue_same_role_operations():
    from dataclasses import replace

    c = replace(claim(), pool="rollout")
    ledger = OwnershipLedger(PlacementPlan((), (c,)))
    first = await ledger.acquire("generate", "one", [c], role="rollout")
    second = await asyncio.wait_for(ledger.acquire("recover", "two", [c], role="rollout"), timeout=1)
    ledger.check(first, [c])
    ledger.check(second, [c])


async def test_training_roles_do_not_conflict_with_each_other():
    c = claim()
    ledger = OwnershipLedger(PlacementPlan((), (c,)))
    first = await ledger.acquire("train", "actor", [c], role="actor")
    second = await asyncio.wait_for(ledger.acquire("train", "critic", [c], role="critic"), timeout=1)
    ledger.check(first)
    ledger.check(second)


async def test_same_role_shared_lease_waits_while_peer_is_scoring():
    rollout, scorer = claim(), claim("genrm", phase=Phase.SCORE)
    ledger = OwnershipLedger(PlacementPlan((), (rollout, scorer)))
    scoring, finish_score, checked = asyncio.Event(), asyncio.Event(), asyncio.Event()

    async def apply(model, action, lease, command, **kwargs):
        if model == "judge" and action == "onload":
            scoring.set()
            await finish_score.wait()

    ledger.register_model("genrm", "judge", SimpleNamespace(apply=SimpleNamespace(remote=apply)), scorer)
    first = await ledger.acquire("generate", "one", [rollout, scorer], role="rollout")
    switching = asyncio.create_task(ledger.command(first, "score", "score"))
    await scoring.wait()

    conflicts = ledger._conflicts

    def check_conflict(role, other):
        checked.set()
        return conflicts(role, other)

    ledger._conflicts = check_conflict
    second = asyncio.create_task(ledger.acquire("eval", "two", [rollout, scorer], role="rollout"))
    await checked.wait()
    assert "two" not in ledger.snapshot()["owners"]
    finish_score.set()
    await switching
    assert not second.done()
    await ledger.release(first)
    ledger.check(await asyncio.wait_for(second, timeout=1))


async def test_maintenance_defers_only_for_other_resident_roles():
    rollout, teacher = claim(), claim("teacher", phase=Phase.SCORE)
    ledger = OwnershipLedger(PlacementPlan((), (rollout, teacher)))
    ledger.register_model("rollout", "default", None, rollout)
    ledger.register_model("teacher", "default", None, teacher)
    lease = await ledger.acquire("generate", "one", [rollout, teacher], role="rollout")
    ledger.reserve_activation(lease, "teacher", "default")
    assert not ledger.reserve_maintenance("rollout", "default")
    ledger.confirm_deactivation(lease, "teacher", "default")
    assert ledger.reserve_maintenance("rollout", "default")


async def test_manager_maintenance_does_not_acquire_workload_lease():
    from relax.distributed.ray.inference_manager import InferenceManager

    entered, complete = asyncio.Event(), asyncio.Event()

    async def forbidden(*args, **kwargs):
        pytest.fail("maintenance waited for workload lease")

    async def reserve(*args):
        return True

    authority = SimpleNamespace(
        claims_for=SimpleNamespace(remote=forbidden),
        acquire=SimpleNamespace(remote=forbidden),
        reserve_maintenance=SimpleNamespace(remote=reserve),
    )
    cls = InferenceManager.__ray_metadata__.modified_class
    manager = cls(SimpleNamespace(_inference_authority=authority), "rollout")
    manager._locks["default"] = asyncio.Lock()
    manager._runtimes["default"] = SimpleNamespace(recover=lambda: "recovered", scale_out=lambda: "scaled")

    async def generate():
        entered.set()
        await complete.wait()

    work = asyncio.create_task(generate())
    await entered.wait()
    try:
        assert await asyncio.wait_for(manager.maintenance("default", "recover"), timeout=1) == "recovered"
        assert await asyncio.wait_for(manager.maintenance("default", "scale_out"), timeout=1) == "scaled"
        assert not work.done()
    finally:
        complete.set()
        await work


async def test_training_failure_can_restart_only_after_every_rank_dies(monkeypatch):
    import ray

    c = claim()
    ledger = OwnershipLedger(PlacementPlan((), (c,)))
    death = asyncio.Event()
    killed = []

    class Actor:
        def __init__(self, rank):
            self.rank = rank
            self.termination_probe = SimpleNamespace(remote=self.probe)

        async def probe(self):
            await death.wait()
            raise ray.exceptions.RayActorError()

    actors = [Actor(0), Actor(1)]
    ledger.register_training_actors("actor", actors)
    monkeypatch.setattr(ray, "kill", lambda actor, **kwargs: killed.append(actor))
    lease = await ledger.acquire("train:actor", "failed", [c], role="actor")
    await ledger.suspect(lease, "NCCL peer lost")
    assert ledger.snapshot()["blocked"] == {}
    assert ledger.snapshot()["suspect"] == {"failed": "NCCL peer lost"}
    with pytest.raises(ResourceBlockedError):
        await ledger.acquire("generate", "waiting", [c], role="rollout")
    stopping = asyncio.create_task(ledger.stop_training(timeout=1))
    while len(killed) != 2:
        await asyncio.sleep(0)  # Yield, no timing assumption; the death event is the barrier.
    assert not stopping.done()
    death.set()
    await stopping
    old_epoch = ledger.epoch
    await ledger.new_epoch()
    assert ledger.epoch != old_epoch
    assert ledger.snapshot()["suspect"] == {}
    with pytest.raises(StaleLeaseError):
        ledger.check(lease)
    with pytest.raises(StaleLeaseError):
        await ledger.command(lease, "late", "train")
    replacement = await ledger.acquire("generate", "new", [c], role="rollout")
    ledger.check(replacement)


async def test_restart_does_not_discharge_unproven_training_exit(monkeypatch):
    import ray

    c = claim()
    ledger = OwnershipLedger(PlacementPlan((), (c,)))

    async def probe():
        return None  # A normal reply is not evidence of actor death.

    actor = SimpleNamespace(termination_probe=SimpleNamespace(remote=probe))
    ledger.register_training_actors("actor", [actor])
    monkeypatch.setattr(ray, "kill", lambda *args, **kwargs: None)
    lease = await ledger.acquire("train:actor", "failed", [c], role="actor")
    await ledger.suspect(lease, "NCCL peer lost")
    with pytest.raises(ResourceBlockedError, match="training.*exit"):
        await ledger.stop_training(timeout=1)
    assert "failed" in ledger.snapshot()["suspect"]


async def test_acquire_timeout_names_holder_and_never_grants_late(monkeypatch):
    monkeypatch.setenv("RELAX_INFERENCE_LEASE_WAIT_S", "0.01")
    c = claim()
    ledger = OwnershipLedger(PlacementPlan((), (c,)))
    first = await ledger.acquire("busy-worker", "held", [c])
    with pytest.raises(OwnershipError, match="busy-worker"):
        await asyncio.wait_for(ledger.acquire("waiting", "pending", [c]), timeout=1)
    await ledger.release(first)
    assert ledger.snapshot()["owners"] == {}


async def test_dead_holder_requires_manager_release_receipt(monkeypatch):
    monkeypatch.setenv("RELAX_INFERENCE_HOLDER_POLL_S", "0.001")
    from relax.engine.inference.holder import HolderIdentity

    c = claim()
    ledger = OwnershipLedger(PlacementPlan((), (c,)))
    entered, confirmed = asyncio.Event(), asyncio.Event()

    async def reconcile(name, lease):
        entered.set()
        await confirmed.wait()
        return True

    async def dead(holder):
        return False

    monkeypatch.setattr(ledger, "_holder_alive", dead)
    manager = SimpleNamespace(reconcile_dead_holder=SimpleNamespace(remote=reconcile))
    ledger.register_model("rollout", "default", manager, c)
    held = await ledger.acquire("dead-worker", "held", [c], holder=HolderIdentity(actor_id="dead"))
    waiting = asyncio.create_task(ledger.acquire("next", "next", [c]))
    await asyncio.wait_for(entered.wait(), timeout=1)
    assert not waiting.done()
    confirmed.set()
    current = await asyncio.wait_for(waiting, timeout=1)
    assert current.generation > held.generation
    with pytest.raises(StaleLeaseError):
        ledger.check(held)


async def test_release_has_a_deadline_for_inflight_commands(monkeypatch):
    monkeypatch.setenv("RELAX_INFERENCE_CONTROL_TIMEOUT_S", "0.01")
    c = claim()
    ledger = OwnershipLedger(PlacementPlan((), (c,)))
    lease = await ledger.acquire("busy-worker", "held", [c])
    ledger.begin(lease, "unfinished-onload")
    with pytest.raises(OwnershipError, match="busy-worker.*unfinished-onload"):
        await asyncio.wait_for(ledger.release(lease), timeout=1)
    assert "held" in ledger.snapshot()["owners"]
    await ledger.finish(lease, "unfinished-onload")
    await ledger.release(lease)


async def test_dead_holder_reconciliation_retries_after_inflight_work_settles(monkeypatch):
    from relax.engine.inference.holder import HolderIdentity

    monkeypatch.setenv("RELAX_INFERENCE_HOLDER_POLL_S", "0.001")
    monkeypatch.setenv("RELAX_INFERENCE_CONTROL_TIMEOUT_S", "0.01")
    c = claim()
    ledger = OwnershipLedger(PlacementPlan((), (c,)))

    async def dead(holder):
        return False

    async def confirmed(name, lease):
        return True

    monkeypatch.setattr(ledger, "_holder_alive", dead)
    ledger.register_model(
        "rollout", "default", SimpleNamespace(reconcile_dead_holder=SimpleNamespace(remote=confirmed)), c
    )
    lease = await ledger.acquire("dead", "held", [c], holder=HolderIdentity(actor_id="dead"))
    ledger.begin(lease, "unfinished-onload")
    with pytest.raises(OwnershipError, match="unfinished-onload"):
        await ledger.acquire("waiting", "first-attempt", [c])
    await ledger.finish(lease, "unfinished-onload")
    current = await asyncio.wait_for(ledger.acquire("waiting", "retry", [c]), timeout=1)
    ledger.check(current)


async def test_dead_narrow_teacher_does_not_quarantine_offloaded_rollout(monkeypatch):
    from dataclasses import replace

    from relax.engine.inference.holder import HolderIdentity

    monkeypatch.setenv("RELAX_INFERENCE_HOLDER_POLL_S", "0.001")
    rollout, teacher = replace(claim(), size=8), replace(claim("teacher", phase=Phase.SCORE), size=2)
    ledger = OwnershipLedger(PlacementPlan((), (rollout, teacher)))
    calls = []

    async def reconcile(name, lease):
        calls.append(name)
        ledger.check(lease, [teacher if name == "teacher" else rollout])
        return True

    async def dead(holder):
        return False

    monkeypatch.setattr(ledger, "_holder_alive", dead)
    manager = SimpleNamespace(reconcile_dead_holder=SimpleNamespace(remote=reconcile))
    ledger.register_model("rollout", "rollout", manager, rollout)
    ledger.register_model("teacher", "teacher", manager, teacher)
    lease = await ledger.acquire("teacher", "lost", [teacher], role="teacher", holder=HolderIdentity(actor_id="dead"))
    ledger.reserve_activation(lease, "teacher", "teacher")
    current = await ledger.acquire("rollout", "next", [rollout, teacher], role="rollout")
    ledger.check(current)
    assert calls == ["teacher"]
    assert ledger.snapshot()["blocked"] == {}


async def test_control_rpc_deadline_does_not_abandon_owned_rollout_thread(monkeypatch):
    import threading

    from relax.engine.inference.phase_client import control_call

    monkeypatch.setenv("RELAX_INFERENCE_CONTROL_TIMEOUT_S", "0.01")
    entered, finish = threading.Event(), threading.Event()

    def work():
        entered.set()
        finish.wait()
        return "settled"

    task = asyncio.create_task(control_call(asyncio.to_thread(work)))
    assert await asyncio.to_thread(entered.wait, 1)
    try:
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(asyncio.shield(task), timeout=0.03)
        assert not task.done()
    finally:
        finish.set()
    assert await task == "settled"


async def test_acquire_lost_reply_abandons_its_unused_grant(monkeypatch):
    from relax.engine.inference.phase_client import PhaseSession

    monkeypatch.setenv("RELAX_INFERENCE_LEASE_WAIT_S", "0.01")
    monkeypatch.setenv("RELAX_INFERENCE_HOLDER_CHECK_S", "0.01")
    c = claim()
    ledger = OwnershipLedger(PlacementPlan((), (c,)))
    reply = asyncio.Event()
    transport = []

    async def acquire(*args, **kwargs):
        lease = await ledger.acquire(*args, **kwargs)
        await reply.wait()
        return lease

    def remote(*args, **kwargs):
        task = asyncio.create_task(acquire(*args, **kwargs))
        transport.append(task)
        return asyncio.shield(task)  # Ray work survives local wait cancellation.

    async def claims(role):
        return [c]

    authority = SimpleNamespace(
        claims_for=SimpleNamespace(remote=claims),
        acquire=SimpleNamespace(remote=remote),
        abandon=SimpleNamespace(remote=ledger.abandon),
    )
    session = PhaseSession(authority, "rollout", "lost-reply")
    try:
        with pytest.raises(OwnershipError, match="lease grant"):
            await session.__aenter__()
        assert ledger.snapshot()["owners"] == {}
        with pytest.raises(StaleLeaseError, match="abandoned"):
            await ledger.acquire(session.owner, session.operation_id, [c], role="rollout", holder=session.holder)
    finally:
        reply.set()
        await asyncio.gather(*transport)


async def test_late_reconciliation_cannot_quarantine_new_epoch():
    c = claim()
    ledger = OwnershipLedger(PlacementPlan((), (c,)))
    entered, reply = asyncio.Event(), asyncio.Event()

    async def reconcile(name, lease):
        entered.set()
        await reply.wait()
        raise RuntimeError("old response arrived late")

    ledger.register_model(
        "rollout", "default", SimpleNamespace(reconcile_dead_holder=SimpleNamespace(remote=reconcile)), c
    )
    lease = await ledger.acquire("old", "old", [c])
    task = asyncio.create_task(ledger._reconcile_dead_holder(lease))
    await entered.wait()
    await ledger.new_epoch()  # No resident resource remains after owner shutdown.
    reply.set()
    await task
    assert ledger.snapshot()["blocked"] == {}


async def test_role_maintenance_rejects_a_returned_explicit_lease():
    from relax.distributed.ray.inference_manager import InferenceManager

    c = claim()
    ledger = OwnershipLedger(PlacementPlan((), (c,)))

    async def check(*args, **kwargs):
        return ledger.check(*args, **kwargs)

    manager = InferenceManager.__ray_metadata__.modified_class(
        SimpleNamespace(_inference_authority=SimpleNamespace(check=SimpleNamespace(remote=check))), "rollout"
    )
    called = []
    manager._runtimes["default"] = SimpleNamespace(scale_in=lambda: called.append("scaled"))
    manager._claims["default"] = (c,)
    manager._locks["default"] = asyncio.Lock()
    lease = await ledger.acquire("old", "old", [c], role="rollout")
    await ledger.release(lease)
    with pytest.raises(StaleLeaseError):
        await manager.call("default", "scale_in", lease=lease)
    assert called == []
