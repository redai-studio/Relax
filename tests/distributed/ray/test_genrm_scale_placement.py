# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""GenRM manager placement hygiene after a failed scale-out (Task 4).

A failed scale-out candidate whose placement-group release Ray has *confirmed*
must not leave a resolvable stale placement behind: ``recover()`` rebuilds
every non-excluded dead slot via ``_resolve_placement``, and scheduling that
rebuild on an already-REMOVED PG fails on every recovery cycle, permanently
degrading the slot. These tests drive the real ``_scale_out_add_one`` failure
path (and its reconcile counterpart) against a stubbed Ray placement-group
table and assert the exclusion-set / placement bookkeeping: a released failed
candidate is retired exactly like a scale-in victim, so recovery never
resolves the deleted PG. The default-placement fallback is not an option for
scale-out ranks (their gpu-index math only covers initial slots), so the
correct post-release semantics are "explicitly never rebuild".

Run: python -m pytest tests/distributed/ray/test_genrm_scale_placement.py -q
"""

import threading
import time
import unittest
from contextlib import ExitStack, contextmanager
from unittest.mock import MagicMock, patch

from tests.utils._dep_stubs import import_genrm_manager


genrm_module = import_genrm_manager()

# Under a real Ray install, ``GenRMManager`` comes back as an ActorClass
# wrapper; unwrap the original Python class (same pattern as
# tests/distributed/ray/conftest.py) so ``object.__new__`` works everywhere.
_GenRMManager = (
    getattr(getattr(genrm_module.GenRMManager, "__ray_metadata__", None), "modified_class", None)
    or genrm_module.GenRMManager
)

import importlib  # noqa: E402

import ray as _real_ray  # noqa: E402  seam for _create_scale_pg's function-scope ``import ray``

import relax.distributed.ray.placement_group as _relax_pg_mod  # noqa: E402
import relax.utils.utils as _relax_utils  # noqa: E402


# ``import ray.util.placement_group as X`` binds X to the *factory function*
# of the same name when ray.util's __init__ re-exports it, not to the module
# (verified: ``X is sys.modules['ray.util.placement_group']`` is False under a
# real Ray install).  Resolve the real module object explicitly so the
# patches below actually steer the production function-scope imports.
_ray_pg_module = importlib.import_module("ray.util.placement_group")


def _manager(num_slots: int = 1):
    """Bare GenRMManager: elastic-scaling state only, no Ray construction."""
    manager = object.__new__(_GenRMManager)
    manager._scale_lock = threading.RLock()
    manager._scale_progress = {}
    manager._scale_op_params = {}
    manager._scale_abort = {}
    manager._scale_added_ranks = []
    manager._candidate_ranks = set()
    manager._draining_ranks = set()
    manager._scale_placements = {}
    manager._scale_drain_confirmed = {}
    manager._scale_physical_done = {}
    manager._retired_scale_ranks = set()
    manager._failed_holding_ranks = set()
    manager._unreleased_candidate_ranks = set()
    manager._scale_drain_timeout_s = 600.0
    manager.all_engines = [object() for _ in range(num_slots)]
    manager._engine_placements = {}
    manager._engine_addr_and_ports = {}
    manager._pending_pg_cleanup = set()
    manager.num_gpu_per_engine = 1
    manager.nodes_per_engine = 1
    manager.args = type(
        "Args",
        (),
        {"fully_async": True, "rollout_num_gpus": 0, "_genrm_colocate_with_rollout": False},
    )()
    manager.pg = "BASE_PG"
    return manager


class _PgTablePatch:
    """Make Ray's placement-group table report a fixed state.

    ``_remove_owned_pg`` imports ``placement_group_table`` /
    ``remove_placement_group`` from ``ray.util.placement_group`` at call time,
    so patching the (real or stubbed) module attribute steers both the release
    confirmation and the asynchronous removal submit.
    """

    def __init__(self, state: str):
        self._state = state

    def __enter__(self):
        self._patches = [
            patch.object(_ray_pg_module, "placement_group_table", lambda pg: {"state": self._state}, create=True),
            patch.object(_ray_pg_module, "remove_placement_group", lambda pg: None, create=True),
        ]
        for patcher in self._patches:
            patcher.start()
        return self

    def __exit__(self, *exc):
        for patcher in self._patches:
            patcher.stop()
        return False


class TestFailedScaleOutCandidateRelease(unittest.TestCase):
    def _prepare_failed_scale_out(self, manager, stale_pg):
        """Register a scale-out op whose engine init always fails after the
        slot has reserved its owned placement group."""
        manager.begin_scale_op("req-1", "scale_out", 2)
        manager._scale_abort["req-1"] = threading.Event()
        manager._scale_progress["req-1"] = manager._new_progress("CREATING")

        def _failing_init(ranks):
            # Mimic the real ``_init_engines`` failure: the slot reserved its
            # owned PG, actor creation failed, the slot is left None.
            for rank in ranks:
                manager._engine_placements[rank] = (stale_pg, True)
                manager.all_engines[rank] = None
            raise RuntimeError("engine init failed")

        manager._create_scale_pg = lambda rank: stale_pg
        manager._init_engines = _failing_init

    def _recording_init(self, manager):
        """Replace ``_init_engines`` with one that records and 'succeeds'."""
        calls = []

        def _init(ranks):
            calls.append(list(ranks))
            for rank in ranks:
                manager.all_engines[rank] = object()
            return len(ranks)

        manager._init_engines = _init
        return calls

    def test_confirmed_release_retires_failed_candidate(self):
        """Scale-out fails, Ray confirms the PG release: the slot must be
        retired (never rebuilt by recover()) and its stale placement must be
        gone; a genuinely dead initial engine is still rebuilt."""
        manager = _manager(num_slots=1)
        stale_pg = ("SCALE_PG", [0], [0])
        self._prepare_failed_scale_out(manager, stale_pg)

        with _PgTablePatch("REMOVED"):
            with self.assertRaises(RuntimeError):
                manager._scale_out_add_one("req-1")

        self.assertIsNone(manager.all_engines[1])
        self.assertNotIn(1, manager._candidate_ranks)
        self.assertNotIn(1, manager._unreleased_candidate_ranks)
        self.assertFalse(manager.get_scale_progress("req-1")["cleanup_required"])
        # The released failed candidate is retired like a scale-in victim and
        # its REMOVED placement is no longer resolvable.
        self.assertIn(1, manager._retired_scale_ranks)
        self.assertNotIn(1, manager._scale_placements)

        # recover() rebuilds real dead engines but never the failed candidate.
        manager.all_engines[0] = None
        calls = self._recording_init(manager)
        rebuilt = manager.recover()
        self.assertEqual(calls, [[0]])
        self.assertEqual(rebuilt, {0})

    def test_unreleased_candidate_excluded_until_reconcile_confirms_release(self):
        """Scale-out fails and Ray does not confirm the release in the bounded
        wait: the slot blocks via ``_unreleased_candidate_ranks``; once
        reconcile confirms the release the slot is retired, so recover() never
        resolves the deleted PG on any cycle."""
        manager = _manager(num_slots=1)
        stale_pg = ("SCALE_PG", [0], [0])
        self._prepare_failed_scale_out(manager, stale_pg)

        clock = {"now": 1000.0}

        def fast_monotonic():
            # Each call advances past the previous one so the 60s bounded
            # wait expires after a couple of iterations.
            clock["now"] += 30.0
            return clock["now"]

        with (
            _PgTablePatch("PENDING"),
            patch.object(time, "monotonic", fast_monotonic),
            patch.object(time, "sleep", lambda _s: None),
        ):
            with self.assertRaises(RuntimeError):
                manager._scale_out_add_one("req-1")

        self.assertIn(1, manager._unreleased_candidate_ranks)
        self.assertTrue(manager.get_scale_progress("req-1")["cleanup_required"])
        calls = self._recording_init(manager)
        self.assertEqual(manager.recover(), set())
        self.assertEqual(calls, [])

        # Lifecycle thread finishes; reconcile retries and Ray now confirms.
        manager._scale_physical_done["req-1"] = True
        with _PgTablePatch("REMOVED"):
            result = manager.reconcile_scale_op("req-1")
        self.assertFalse(result["cleanup_required"])
        self.assertNotIn(1, manager._unreleased_candidate_ranks)
        self.assertIn(1, manager._retired_scale_ranks)
        self.assertNotIn(1, manager._scale_placements)
        # And recovery still ignores the released failed candidate.
        self.assertEqual(manager.recover(), set())
        self.assertEqual(calls, [])


class _FakePg:
    def ready(self):
        return "pg-ready-ref"


class TestCreateScalePgOwnershipFence(unittest.TestCase):
    """Every exit between PG creation and engine registration must be fenced.

    ``_create_scale_pg`` registers ownership immediately after the PG is
    created, so a readiness-wait timeout, an InfoActor creation failure, a
    probe failure and a kill failure all reach ``_scale_out_add_one``'s fenced
    release: the PG is either confirmed REMOVED or left in pending cleanup
    blocking the model's next scale operation. (Before this fix those exits
    propagated before any ownership map was populated, and the readiness-
    timeout path removed the PG without REMOVED confirmation -- stranding the
    bundle or racing a replacement actor.)
    """

    def _prepare(self):
        manager = _manager(num_slots=1)
        manager.begin_scale_op("req-1", "scale_out", 2)
        manager._scale_abort["req-1"] = threading.Event()
        manager._scale_progress["req-1"] = manager._new_progress("CREATING")
        return manager

    @contextmanager
    def _seamed(self, *, ready, pg_state, probe=None, kill=None, info_actor_class=None, fast=False):
        """Patch every Ray seam the real ``_create_scale_pg`` touches.

        ``fast`` additionally fast-forwards the 60 s bounded release wait so an
        unconfirmed removal expires after a couple of iterations.
        """
        clock = {"now": 1000.0}

        def fast_monotonic():
            clock["now"] += 30.0
            return clock["now"]

        with ExitStack() as stack:
            stack.enter_context(_PgTablePatch(pg_state))
            stack.enter_context(
                patch.object(_ray_pg_module, "placement_group", lambda bundles, strategy=None: _FakePg(), create=True)
            )
            stack.enter_context(
                patch.object(
                    _real_ray, "wait", lambda refs, timeout=None: (list(refs), []) if ready else ([], list(refs))
                )
            )
            stack.enter_context(patch.object(_relax_utils, "get_ray_accelerator_kwargs", lambda n: {"num_gpus": n}))
            if info_actor_class is not None:
                stack.enter_context(patch.object(_relax_pg_mod, "InfoActor", info_actor_class))
            if probe is not None:
                stack.enter_context(patch.object(_real_ray, "get", probe))
            if kill is not None:
                stack.enter_context(patch.object(_real_ray, "kill", kill))
            if fast:
                stack.enter_context(patch.object(time, "monotonic", fast_monotonic))
                stack.enter_context(patch.object(time, "sleep", lambda _s: None))
            yield

    def _assert_pending_cleanup(self, manager):
        self.assertIn(1, manager._unreleased_candidate_ranks)
        self.assertIn(1, manager._pending_pg_cleanup)
        self.assertTrue(manager.get_scale_progress("req-1")["cleanup_required"])
        self.assertIsNone(manager.all_engines[1])
        self.assertNotIn(1, manager._candidate_ranks)

    def _reconcile_confirms(self, manager):
        manager._scale_physical_done["req-1"] = True
        with _PgTablePatch("REMOVED"):
            result = manager.reconcile_scale_op("req-1")
        self.assertFalse(result["cleanup_required"])
        self.assertNotIn(1, manager._unreleased_candidate_ranks)
        self.assertIn(1, manager._retired_scale_ranks)
        self.assertNotIn(1, manager._scale_placements)

    def test_readiness_timeout_is_fenced(self):
        manager = self._prepare()
        with self._seamed(ready=False, pg_state="PENDING", fast=True):
            with self.assertRaises(RuntimeError):
                manager._scale_out_add_one("req-1")
        self._assert_pending_cleanup(manager)
        self._reconcile_confirms(manager)

    def test_infoactor_create_failure_is_fenced(self):
        manager = self._prepare()
        info_actor_class = MagicMock()
        info_actor_class.options.side_effect = RuntimeError("info actor create failed")
        with self._seamed(ready=True, info_actor_class=info_actor_class, pg_state="PENDING", fast=True):
            with self.assertRaises(RuntimeError):
                manager._scale_out_add_one("req-1")
        self._assert_pending_cleanup(manager)
        self._reconcile_confirms(manager)

    def test_probe_failure_is_fenced(self):
        manager = self._prepare()
        info_actor = MagicMock()
        info_actor.get_ip_and_gpu_id.remote.return_value = "probe-ref"
        info_actor_class = MagicMock()
        info_actor_class.options.return_value.remote.return_value = info_actor

        def probe_fails(ref, timeout=None):
            raise RuntimeError("probe failed")

        with self._seamed(
            ready=True,
            probe=probe_fails,
            kill=lambda actor: None,
            info_actor_class=info_actor_class,
            pg_state="PENDING",
            fast=True,
        ):
            with self.assertRaises(RuntimeError):
                manager._scale_out_add_one("req-1")
        self._assert_pending_cleanup(manager)
        self._reconcile_confirms(manager)

    def test_kill_failure_does_not_mask_probe_error_or_leak(self):
        manager = self._prepare()
        info_actor = MagicMock()
        info_actor.get_ip_and_gpu_id.remote.return_value = "probe-ref"
        info_actor_class = MagicMock()
        info_actor_class.options.return_value.remote.return_value = info_actor

        def probe_fails(ref, timeout=None):
            raise RuntimeError("probe failed")

        def kill_fails(actor):
            raise RuntimeError("kill failed")

        with self._seamed(
            ready=True,
            probe=probe_fails,
            kill=kill_fails,
            info_actor_class=info_actor_class,
            pg_state="PENDING",
            fast=True,
        ):
            with self.assertRaises(RuntimeError) as ctx:
                manager._scale_out_add_one("req-1")
        # The original probe error survives; the kill failure was only logged.
        self.assertEqual(str(ctx.exception), "probe failed")
        self._assert_pending_cleanup(manager)

    def test_abort_between_pg_and_engine_is_fenced(self):
        manager = self._prepare()
        manager._scale_abort["req-1"].set()
        info_actor = MagicMock()
        info_actor.get_ip_and_gpu_id.remote.return_value = "probe-ref"
        info_actor_class = MagicMock()
        info_actor_class.options.return_value.remote.return_value = info_actor
        with self._seamed(
            ready=True,
            probe=lambda ref, timeout=None: ("10.0.0.1", 2),
            kill=lambda actor: None,
            info_actor_class=info_actor_class,
            pg_state="REMOVED",
        ):
            with self.assertRaises(RuntimeError):
                manager._scale_out_add_one("req-1")
        # Confirmed release: retired, placement dropped, no mutex kept.
        self.assertIn(1, manager._retired_scale_ranks)
        self.assertNotIn(1, manager._scale_placements)
        self.assertNotIn(1, manager._unreleased_candidate_ranks)
        self.assertFalse(manager.get_scale_progress("req-1")["cleanup_required"])


if __name__ == "__main__":
    unittest.main()
