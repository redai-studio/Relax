# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import asyncio
import threading
from types import SimpleNamespace

import pytest

from relax.distributed.ray.placement_planner import plan_placement
from relax.engine.inference import deferred
from relax.engine.inference.lifecycle import LifecycleCoordinator, LifecycleParticipant
from tests.engine.inference.test_deferred_scoring import _args, _groups


@pytest.mark.parametrize("trigger", ["entry_failure", "entry_cancel", "cleanup_cancel", "cancel_twice"])
async def test_deferred_switch_settles_under_lock_before_next_batch(monkeypatch, trigger):
    loop = asyncio.get_running_loop()
    entered, cleaning, queued = asyncio.Event(), asyncio.Event(), asyncio.Event()
    release_entry, release_cleanup = threading.Event(), threading.Event()
    events = []

    class Lock(asyncio.Lock):
        async def acquire(self):
            if self.locked():
                queued.set()
            return await super().acquire()

    lock = Lock()
    monkeypatch.setattr(deferred, "_scoring_lock", lambda: lock)

    def enter():
        events.append("old enter")
        loop.call_soon_threadsafe(entered.set)
        assert release_entry.wait(5)
        events.append("old onload")
        if trigger == "entry_failure":
            raise RuntimeError("second scorer failed")

    def leave():
        assert lock.locked()
        loop.call_soon_threadsafe(cleaning.set)
        assert release_cleanup.wait(5)
        events.append("old offload")

    async def score(*args):
        return None

    monkeypatch.setattr(deferred, "_score_group", score)
    old = asyncio.create_task(
        deferred.run_deferred_scoring(
            _args(), _groups(1), coordinator=SimpleNamespace(enter_score=enter, leave_score=leave)
        )
    )
    new = None
    try:
        await asyncio.wait_for(entered.wait(), 5)
        if trigger in {"entry_cancel", "cancel_twice"}:
            old.cancel()
        new = asyncio.create_task(
            deferred.run_deferred_scoring(
                _args(),
                _groups(1),
                coordinator=SimpleNamespace(
                    enter_score=lambda: events.append("new enter"),
                    leave_score=lambda: events.append("new offload"),
                ),
            )
        )
        await asyncio.wait_for(queued.wait(), 5)
        assert events == ["old enter"]
        assert not old.done() and not new.done()
        release_entry.set()
        await asyncio.wait_for(cleaning.wait(), 5)
        assert events == ["old enter", "old onload"]
        if trigger in {"cleanup_cancel", "cancel_twice"}:
            old.cancel()
        assert lock.locked() and not old.done() and not new.done()
        release_cleanup.set()
        error = RuntimeError if trigger == "entry_failure" else asyncio.CancelledError
        with pytest.raises(error):
            await old
        await new
        assert events == ["old enter", "old onload", "old offload", "new enter", "new offload"]
    finally:
        release_entry.set()
        release_cleanup.set()
        await asyncio.gather(*(task for task in (old, new) if task is not None), return_exceptions=True)


@pytest.mark.parametrize("failure", ["submit", "wait"])
def test_standalone_enter_score_compensates_every_attempted_activation(failure):
    events = []
    args = _args(_genrm_instances_resolved={"first": {"num_gpus": 4}, "second": {"num_gpus": 4}})

    def activate(name):
        events.append(f"{name} onload issued")
        if name == "second" and failure == "submit":
            raise RuntimeError("onload failed")
        return [name]

    def wait(refs):
        for ref in refs:
            events.append(f"{ref} settled")
            if ref == "second":
                raise RuntimeError("onload failed")

    def offload(name):
        events.append(f"{name} offload")
        return []

    participants = [
        LifecycleParticipant("genrm", name, lambda name=name: offload(name), lambda name=name: activate(name))
        for name in ("first", "second")
    ]
    coordinator = LifecycleCoordinator(plan_placement(args), participants, wait=wait)
    with pytest.raises(RuntimeError, match="onload failed"):
        coordinator.enter_score()
    assert "first settled" in events
    assert events.index("first settled") < events.index("first offload")
    assert events[-2:] == ["second offload", "first offload"]
