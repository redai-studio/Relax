# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Unit tests for the GenRM scale watcher (Task 4).

The watcher maps the manager's physical-lifecycle progress onto the
registry's monotonic state machine, closes the component-side admission
cache for draining victims, proves the drain via the per-engine in-flight
counter and confirms it to the manager. These tests drive the real watcher
against a scripted manager speaking the real progress protocol.

Run: python -m unittest tests.components.test_genrm_scale_watcher -v
"""

import asyncio
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from tests.utils._dep_stubs import import_genrm_component


genrm_module = import_genrm_component()

import ray  # noqa: E402  (real or stubbed -- guaranteed importable after stub install)


# The scripted manager's ``.remote()`` facade returns plain dicts, never real
# ObjectRefs.  A real ``ray.get`` rejects non-ObjectRef inputs, but only after
# spinning up a local Ray instance (~10s per call), which starves the event
# barriers below.  Pass ``ray.get`` through for the duration of this module,
# following the direct-attribute-replacement pattern of
# tests/distributed/ray/conftest.py (patch("ray.get", ...) is unreliable once
# Ray is initialised).
_orig_ray_get = ray.get


def _passthrough_ray_get(ref, timeout=None, **kwargs):
    return ref


def setUpModule():
    ray.get = _passthrough_ray_get


def tearDownModule():
    ray.get = _orig_ray_get


_GenRM = genrm_module.GenRM.func_or_class
_GenRMEngineCacheState = genrm_module._EngineCacheState
_GenRMScaleRegistry = genrm_module.GenRMScaleRegistry


class _ScriptedManager:
    """Manager fake speaking the real progress protocol.

    Scale-in behaves like the real manager: the victim stays DRAINING until
    ``confirm_scale_drained`` arrives, then REMOVING, then COMPLETED.
    """

    def __init__(self, direction: str, script=None):
        self.direction = direction
        self.begin_calls = []
        self.hook_calls = []
        self.confirms = []
        self.aborts = []
        self.drained = False
        self.state = {
            "phase": "CREATING" if direction == "scale_out" else "DRAINING",
            "created": 0,
            "removed": 0,
            "failed": 0,
            "current": 1,
            "ready": 1,
            "victim": None,
            "error": None,
            # The production protocol requires an explicit physical
            # completion fence.  Ordinary scripted terminal states
            # are complete unless a test deliberately holds them.
            "physical_done": True,
        }
        self.script = script  # optional list of states overriding self.state

    # -- protocol surface -------------------------------------------------

    def begin_scale_op(self, request_id, direction, target):
        self.begin_calls.append((request_id, direction, target))

    def execute_genrm_scale_out(self, request_id):
        self.hook_calls.append(("scale_out", request_id))

    def execute_genrm_scale_in(self, request_id):
        self.hook_calls.append(("scale_in", request_id))

    def confirm_scale_drained(self, request_id):
        self.confirms.append(request_id)
        self.drained = True

    def abort_scale_op(self, request_id):
        self.aborts.append(request_id)

    def get_scale_progress(self, request_id):
        if self.script is not None:
            return dict(self.state if self.state.get("hold") else self._advance_script())
        if self.direction == "scale_in":
            if not self.drained:
                return dict(self.state)
            if self.state["phase"] == "DRAINING":
                self.state = dict(self.state, phase="REMOVING")
            elif self.state["phase"] == "REMOVING":
                self.state = dict(self.state, phase="COMPLETED", removed=1, current=1, ready=1, victim=None)
            return dict(self.state)
        return dict(self.state)

    def _advance_script(self):
        return self.state


def _remote_wrap(manager):
    """Wrap each protocol method with a .remote(arg) facade like a Ray
    handle."""

    def wrap(fn):
        def remote(*args, **kwargs):
            return fn(*args, **kwargs)

        return SimpleNamespace(remote=remote)

    for name in (
        "begin_scale_op",
        "execute_genrm_scale_out",
        "execute_genrm_scale_in",
        "confirm_scale_drained",
        "abort_scale_op",
        "get_scale_progress",
    ):
        setattr(manager, name, wrap(getattr(manager, name)))
    return manager


def _replica(manager):
    replica = object.__new__(_GenRM)
    replica.genrm_managers = {"__default__": manager}
    replica._scale_registry = _GenRMScaleRegistry()
    replica._scale_registry.register_initial("__default__", 1)
    replica._logger_instance = None
    replica._engine_inflight = {}
    replica._engine_served = {}
    replica._engine_caches = {"__default__": _GenRMEngineCacheState()}
    replica._genrm_engine_list = lambda key: []
    return replica


class _FastSleep:
    """Collapse watcher poll sleeps so tests run in milliseconds."""

    def __enter__(self):
        self._patcher = patch("asyncio.sleep", new=lambda *_a, **_k: _immediate())
        self._patcher.start()

    def __exit__(self, *exc):
        self._patcher.stop()


async def _immediate():
    return None


def _run(coro):
    return asyncio.run(coro)


class TestScaleOutWatcher(unittest.TestCase):
    def _submit(self, replica, target=2):
        decision = replica._scale_registry.submit(
            "scale_out",
            model_name="__default__",
            target=target,
            timeout_secs=60.0,
            idempotency_key=None,
            current=1,
            ready=1,
        )
        return decision["request_id"]

    def test_active_maps_monotonic_chain_and_finishes(self):
        manager = _remote_wrap(_ScriptedManager("scale_out"))
        manager.state = dict(manager.state, phase="ACTIVE", created=1, current=2, ready=2)
        replica = _replica(manager)
        request_id = self._submit(replica)
        with _FastSleep():
            _run(replica._watch_scale_operation("scale_out", "__default__", request_id))
        result = replica._scale_registry.get_status("scale_out", request_id)
        self.assertEqual(result["status"], "ACTIVE")
        self.assertEqual(result["created"], 1)
        self.assertEqual(result["current"], 2)

    def test_partial_after_one_published(self):
        manager = _remote_wrap(_ScriptedManager("scale_out"))
        manager.state = dict(
            manager.state, phase="PARTIAL", created=1, failed=1, current=2, ready=2, error="second engine failed"
        )
        replica = _replica(manager)
        request_id = self._submit(replica, target=3)
        with _FastSleep():
            _run(replica._watch_scale_operation("scale_out", "__default__", request_id))
        result = replica._scale_registry.get_status("scale_out", request_id)
        self.assertEqual(result["status"], "PARTIAL")
        self.assertEqual(result["created"], 1)
        self.assertEqual(result["failed"], 1)

    def test_failed_from_creating_is_legal(self):
        manager = _remote_wrap(_ScriptedManager("scale_out"))
        manager.state = dict(manager.state, phase="FAILED", failed=1, error="pg timeout")
        replica = _replica(manager)
        request_id = self._submit(replica)
        with _FastSleep():
            _run(replica._watch_scale_operation("scale_out", "__default__", request_id))
        result = replica._scale_registry.get_status("scale_out", request_id)
        self.assertEqual(result["status"], "FAILED")

    def test_terminal_manager_phase_waits_for_physical_completion(self):
        manager = _remote_wrap(_ScriptedManager("scale_out"))
        manager.state = dict(manager.state, phase="FAILED", failed=1, physical_done=False, hold=True)
        replica = _replica(manager)
        request_id = self._submit(replica)

        original_sleep = asyncio.sleep

        async def yielding_sleep(*_args, **_kwargs):
            await original_sleep(0)

        async def scenario():
            with patch("asyncio.sleep", new=yielding_sleep):
                task = asyncio.create_task(replica._watch_scale_operation("scale_out", "__default__", request_id))
                await original_sleep(0)
                task.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await task

        _run(scenario())
        result = replica._scale_registry.get_status("scale_out", request_id)
        self.assertEqual(result["status"], "PENDING")

    def test_unfinished_terminal_keeps_model_exclusive_until_physical_completion(self):
        """A reported FAILED phase must not release a still-running manager.

        This is intentionally not a sleep-only test: the event-loop yield is
        explicit, then we submit a second request while the first manager
        reports ``physical_done=False``.  It caught the prior bug where the
        watcher advanced the registry to FAILED outside its completion fence.
        """
        manager = _remote_wrap(_ScriptedManager("scale_out"))
        manager.state = dict(manager.state, phase="FAILED", failed=1, physical_done=False, hold=True)
        replica = _replica(manager)
        request_id = self._submit(replica)
        original_sleep = asyncio.sleep

        async def yielding_sleep(*_args, **_kwargs):
            await original_sleep(0)

        async def scenario():
            with patch("asyncio.sleep", new=yielding_sleep):
                task = asyncio.create_task(replica._watch_scale_operation("scale_out", "__default__", request_id))
                await original_sleep(0)
                status = replica._scale_registry.get_status("scale_out", request_id)
                self.assertEqual(status["status"], "PENDING")
                blocked = replica._scale_registry.submit(
                    "scale_out",
                    model_name="__default__",
                    target=2,
                    timeout_secs=60.0,
                    idempotency_key=None,
                    current=1,
                    ready=1,
                )
                self.assertEqual(blocked["http"], 409)
                manager.state = dict(manager.state, physical_done=True)
                await task

        _run(scenario())
        result = replica._scale_registry.get_status("scale_out", request_id)
        self.assertEqual(result["status"], "FAILED")
        admitted = replica._scale_registry.submit(
            "scale_out",
            model_name="__default__",
            target=2,
            timeout_secs=60.0,
            idempotency_key=None,
            current=1,
            ready=1,
        )
        self.assertEqual(admitted["http"], 200)


class TestScaleOutWatcherFenceRace(unittest.TestCase):
    """Deterministic regression for the watcher's physical-completion fence.

    The sleep-yield tests above can pass while the watcher coroutine has never
    processed a single poll (deleting the fence keeps them green). Here the
    barrier is the watcher's own poll sleep: the watcher only reaches
    ``asyncio.sleep`` inside its loop after it has fully processed a poll that
    reported a terminal phase with ``physical_done=False`` and deliberately
    declined to finish the registry.  A fence-less watcher finishes on that
    same poll and never parks, so the barrier wait fails.
    """

    def _submit(self, replica, target=2):
        decision = replica._scale_registry.submit(
            "scale_out",
            model_name="__default__",
            target=target,
            timeout_secs=60.0,
            idempotency_key=None,
            current=1,
            ready=1,
        )
        return decision["request_id"]

    def _run_fence_scenario(self, final_progress):
        """Hold the manager at FAILED/physical_done=False across processed
        polls, then release the fence with ``final_progress`` extras."""
        manager = _remote_wrap(_ScriptedManager("scale_out"))
        manager.state = dict(manager.state, phase="FAILED", failed=1, physical_done=False, hold=True)
        replica = _replica(manager)
        request_id = self._submit(replica)

        original_sleep = asyncio.sleep
        watcher_parked = threading.Event()

        async def recording_sleep(*_args, **_kwargs):
            # Reached only after one poll cycle was fully processed.
            watcher_parked.set()
            await original_sleep(0)

        async def scenario():
            with patch("asyncio.sleep", new=recording_sleep):
                task = asyncio.create_task(replica._watch_scale_operation("scale_out", "__default__", request_id))
                # Real event barrier: waits for the watcher's first poll sleep.
                await asyncio.to_thread(watcher_parked.wait, 5.0)
                self.assertTrue(
                    watcher_parked.is_set(),
                    "watcher never parked on a poll sleep; it finished the registry "
                    "without the physical-completion fence",
                )
                self.assertFalse(task.done())
                # The poll that just parked reported FAILED + physical_done=False;
                # the registry must still be non-terminal and hold the model gate.
                status = replica._scale_registry.get_status("scale_out", request_id)
                self.assertEqual(status["status"], "PENDING")
                blocked = replica._scale_registry.submit(
                    "scale_out",
                    model_name="__default__",
                    target=2,
                    timeout_secs=60.0,
                    idempotency_key=None,
                    current=1,
                    ready=1,
                )
                self.assertEqual(blocked["http"], 409)
                self.assertEqual(blocked["status"], "CONFLICT")
                # Release the fence: the lifecycle thread has stopped.
                manager.state = dict(manager.state, physical_done=True, **final_progress)
                await asyncio.wait_for(task, timeout=15.0)

        _run(scenario())
        return replica, request_id

    def test_fence_holds_across_processed_polls_then_releases(self):
        """Terminal phase + physical_done=False blocks the finish for as long
        as the state persists; physical_done=True finishes and, with a clean
        progress snapshot, releases the per-model mutual exclusion."""
        replica, request_id = self._run_fence_scenario({})
        result = replica._scale_registry.get_status("scale_out", request_id)
        self.assertEqual(result["status"], "FAILED")
        self.assertFalse(result["cleanup_required"])
        admitted = replica._scale_registry.submit(
            "scale_out",
            model_name="__default__",
            target=2,
            timeout_secs=60.0,
            idempotency_key=None,
            current=1,
            ready=1,
        )
        self.assertEqual(admitted["http"], 200)

    def test_fence_release_with_cleanup_required_keeps_model_exclusive(self):
        """When the finished progress is dirty (cleanup_required), the terminal
        operation keeps blocking new scale requests until reconcile clears it
        (registry finish semantics)."""
        replica, request_id = self._run_fence_scenario({"cleanup_required": True})
        result = replica._scale_registry.get_status("scale_out", request_id)
        self.assertEqual(result["status"], "FAILED")
        self.assertTrue(result["cleanup_required"])
        still_blocked = replica._scale_registry.submit(
            "scale_out",
            model_name="__default__",
            target=2,
            timeout_secs=60.0,
            idempotency_key=None,
            current=1,
            ready=1,
        )
        self.assertEqual(still_blocked["http"], 409)
        self.assertEqual(still_blocked["status"], "CONFLICT")


class TestScaleInWatcher(unittest.TestCase):
    def _submit(self, replica, target=1):
        decision = replica._scale_registry.submit(
            "scale_in",
            model_name="__default__",
            target=target,
            timeout_secs=60.0,
            idempotency_key=None,
            current=2,
            ready=2,
        )
        return decision["request_id"]

    def _victim(self):
        return ["192.0.2.9", 16009]

    def test_drain_proof_waits_for_inflight_zero_then_confirms(self):
        manager = _remote_wrap(_ScriptedManager("scale_in"))
        manager.state = dict(manager.state, phase="DRAINING", current=2, ready=2, victim=self._victim())
        replica = _replica(manager)
        # Admission cache still holds the victim and one request is in flight.
        replica._engine_caches["__default__"].refresh([tuple(self._victim()), ("192.0.2.1", 16001)])
        replica._engine_inflight[("__default__", *self._victim())] = 1
        request_id = self._submit(replica)

        async def scenario():
            # First poll cycle: victim seen, cache closed, in-flight > 0: no confirm.
            task = asyncio.ensure_future(replica._watch_scale_operation("scale_in", "__default__", request_id))
            await asyncio.sleep(0)
            # The watcher parks on its (patched) sleep; drop in-flight to zero.
            replica._engine_inflight[("__default__", *self._victim())] = 0
            await task

        with _FastSleep():
            _run(scenario())

        self.assertEqual(manager.confirms, [request_id])
        result = replica._scale_registry.get_status("scale_in", request_id)
        self.assertEqual(result["status"], "COMPLETED")
        self.assertEqual(result["removed"], 1)
        self.assertEqual(result["current"], 1)
        # The stale cache holding the victim was dropped.
        self.assertIsNone(replica._engine_caches["__default__"].hosts_ports)

    def test_no_confirm_while_inflight_positive(self):
        manager = _remote_wrap(_ScriptedManager("scale_in"))
        manager.state = dict(manager.state, phase="DRAINING", current=2, ready=2, victim=self._victim(), hold=True)
        replica = _replica(manager)
        replica._engine_inflight[("__default__", *self._victim())] = 3
        request_id = self._submit(replica)

        async def scenario():
            task = asyncio.ensure_future(replica._watch_scale_operation("scale_in", "__default__", request_id))
            for _ in range(20):
                await asyncio.sleep(0)
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        with _FastSleep():
            _run(scenario())
        self.assertEqual(manager.confirms, [])

    def test_victim_leaves_engine_table_on_completion(self):
        manager = _remote_wrap(_ScriptedManager("scale_in"))
        manager.state = dict(manager.state, phase="DRAINING", current=2, ready=2, victim=self._victim())
        replica = _replica(manager)
        replica._engine_caches["__default__"].refresh([("192.0.2.1", 16001), tuple(self._victim())])
        request_id = self._submit(replica)
        with _FastSleep():
            _run(replica._watch_scale_operation("scale_in", "__default__", request_id))
        # force_refresh dropped the list; the manager no longer reports the victim.
        self.assertIsNone(replica._engine_caches["__default__"].hosts_ports)


class TestDispatchWiring(unittest.TestCase):
    def test_dispatch_registers_target_and_fires_hook(self):
        manager = _remote_wrap(_ScriptedManager("scale_out"))
        replica = _replica(manager)
        replica._genrm_engine_list = lambda key: [("192.0.2.1", 16001)]
        response = _run(replica.scale_out(genrm_module.GenRMScaleRequest(num_replicas=2)))
        self.assertEqual(response.status, "PENDING")
        self.assertIsNone(response.detail)
        self.assertEqual(len(manager.begin_calls), 1)
        request_id, direction, target = manager.begin_calls[0]
        self.assertEqual(direction, "scale_out")
        self.assertEqual(target, 2)
        self.assertEqual(manager.hook_calls, [("scale_out", request_id)])


if __name__ == "__main__":
    unittest.main()
