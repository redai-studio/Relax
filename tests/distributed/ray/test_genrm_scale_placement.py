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

from relax.inference.manager import InferenceManager
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
    manager._pg_remove_submitted = set()
    manager._initializing_ranks = set()
    manager._cleanup_pending = set()
    manager.inference = InferenceManager("genrm")
    manager._log_prefix = "test-genrm-manager"
    manager.num_gpu_per_engine = 1
    manager.nodes_per_engine = 1
    manager.args = type(
        "Args",
        (),
        {
            "fully_async": True,
            "rollout_num_gpus": 0,
            "_genrm_colocate_with_rollout": False,
            "genrm_num_gpus_per_engine": 1,
        },
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
        with self.assertRaisesRegex(RuntimeError, "cleanup pending"):
            manager.recover()
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
                # Other CPU fixtures can replace the import shim after
                # collection. Patch the module the production import sees.
                pg_module = importlib.import_module("relax.distributed.ray.placement_group")
                stack.enter_context(patch.object(pg_module, "InfoActor", info_actor_class))
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


class TestVictimBoundDrainConfirm(unittest.TestCase):
    """Review finding: a multi-victim scale-in reuses the request id and swaps
    the drain event per victim.

    A delayed duplicate confirmation for the previous victim must not release
    the current one.
    """

    def _manager(self):
        manager = _manager(num_slots=4)
        manager._scale_op_params["req-1"] = {"target": 1, "direction": "scale_in"}
        return manager

    def _install_victim(self, manager, rank, host="10.0.0.1", port=16001):
        event = threading.Event()
        with manager._scale_lock:
            manager._scale_drain_confirmed["req-1"] = event
            manager._draining_ranks.add(rank)
        manager._scale_progress["req-1"] = manager._new_progress("DRAINING")
        manager._scale_progress["req-1"].update(victim=[host, port], victim_rank=rank)
        return event

    def test_confirm_bound_to_current_victim(self):
        manager = self._manager()
        event = self._install_victim(manager, rank=3)
        manager.confirm_scale_drained("req-1", victim=["10.0.0.1", 16001], victim_rank=3)
        self.assertTrue(event.is_set())

    def test_stale_confirm_for_previous_victim_is_ignored(self):
        manager = self._manager()
        # Victim A (rank 3) was confirmed and retired; the lifecycle moved on
        # to victim B (rank 2) with a fresh event.
        event_a = self._install_victim(manager, rank=3, port=16001)
        manager.confirm_scale_drained("req-1", victim=["10.0.0.1", 16001], victim_rank=3)
        self.assertTrue(event_a.is_set())
        event_b = self._install_victim(manager, rank=2, port=16002)
        # The delayed duplicate confirm for A arrives while B owns the op.
        manager.confirm_scale_drained("req-1", victim=["10.0.0.1", 16001], victim_rank=3)
        self.assertFalse(event_b.is_set())
        # A fresh, correctly-bound confirm still releases B.
        manager.confirm_scale_drained("req-1", victim=["10.0.0.1", 16002], victim_rank=2)
        self.assertTrue(event_b.is_set())

    def test_stale_confirm_between_victims_is_ignored(self):
        manager = self._manager()
        self._install_victim(manager, rank=3)
        manager.confirm_scale_drained("req-1", victim=["10.0.0.1", 16001], victim_rank=3)
        # Lifecycle retired A; progress momentarily carries no victim.
        manager._scale_progress["req-1"].update(victim=None, victim_rank=None)
        event_b = self._install_victim(manager, rank=2, port=16002)
        manager.confirm_scale_drained("req-1", victim=["10.0.0.1", 16001], victim_rank=3)
        self.assertFalse(event_b.is_set())

    def test_identity_less_confirm_is_rejected(self):
        """The drain-proof API carries no legacy unbound form: the victim
        identity (address + rank) is a required part of the contract, and a
        mismatched identity never releases the fence."""
        manager = self._manager()
        event = self._install_victim(manager, rank=3)
        # Identity-less invocation is a contract violation, not a legacy
        # fallback: the signature makes it impossible to call without the
        # victim binding.
        with self.assertRaises(TypeError):
            manager.confirm_scale_drained("req-1")
        self.assertFalse(event.is_set())
        # A wrong rank is logged and dropped (fail-closed): the fence then
        # only releases via a correctly bound proof or its timeout.
        manager.confirm_scale_drained("req-1", victim=["10.0.0.1", 16001], victim_rank=99)
        self.assertFalse(event.is_set())
        # The correctly bound proof still releases the fence.
        manager.confirm_scale_drained("req-1", victim=["10.0.0.1", 16001], victim_rank=3)
        self.assertTrue(event.is_set())
        self.assertTrue(event.is_set())


class TestPgRemoveResubmission(unittest.TestCase):
    """Review finding: an exception from remove_placement_group() on the first
    attempt used to put the rank into _pending_pg_cleanup, which then
    suppressed every resubmission -- the PG stayed CREATED forever.

    The fix separates "deletion submitted" from "release unconfirmed".
    """

    def _manager_with_pg(self):
        manager = _manager(num_slots=2)
        pg = MagicMock()
        manager._engine_placements[1] = ((pg, [0], [1]), True)
        return manager, pg

    def test_first_submit_failure_resubmits_on_retry(self):
        manager, pg = self._manager_with_pg()
        submits = []
        state = {"calls": 0}

        def _remove(pg_arg):
            submits.append(pg_arg)
            if len(submits) == 1:
                raise RuntimeError("submission failed before reaching Ray")
            # Second submission accepted.

        def _table(pg_arg):
            state["calls"] += 1
            return {"state": "REMOVED" if len(submits) >= 2 else "CREATED"}

        with (
            patch.object(_ray_pg_module, "remove_placement_group", _remove, create=True),
            patch.object(_ray_pg_module, "placement_group_table", _table, create=True),
        ):
            self.assertFalse(manager._remove_owned_pg(1))
            self.assertIn(1, manager._pending_pg_cleanup)
            # The failed submission must NOT be recorded as submitted.
            self.assertNotIn(1, manager._pg_remove_submitted)
            # Retry: deletion is resubmitted, then confirmed REMOVED.
            self.assertTrue(manager._remove_owned_pg(1))
        self.assertEqual(len(submits), 2)
        self.assertNotIn(1, manager._pending_pg_cleanup)
        self.assertNotIn(1, manager._engine_placements)

    def test_accepted_submission_is_not_resubmitted(self):
        manager, pg = self._manager_with_pg()
        submits = []

        def _remove(pg_arg):
            submits.append(pg_arg)

        def _table(pg_arg):
            return {"state": "CREATED"}  # deletion accepted, not yet confirmed

        with (
            patch.object(_ray_pg_module, "remove_placement_group", _remove, create=True),
            patch.object(_ray_pg_module, "placement_group_table", _table, create=True),
        ):
            self.assertFalse(manager._remove_owned_pg(1))  # pending, not REMOVED
            self.assertFalse(manager._remove_owned_pg(1))  # confirmation poll only
        self.assertEqual(len(submits), 1)  # never resubmitted an accepted delete
        self.assertIn(1, manager._pending_pg_cleanup)


if __name__ == "__main__":
    unittest.main()


class TestUnifiedInferenceIntegration(unittest.TestCase):
    def test_discovery_and_legacy_addresses_hide_candidates_and_draining_engines(self):
        from relax.inference.routing import select_endpoint

        manager = _manager(4)
        manager._engine_addr_and_ports = {rank: {"host": "192.0.2.1", "port": 16000 + rank} for rank in range(4)}
        manager._candidate_ranks.add(1)
        manager._draining_ranks.add(2)
        manager._retired_scale_ranks.add(3)
        manager.all_engines[3] = None
        snapshot = manager.get_inference_snapshot()
        self.assertEqual(manager.get_engine_hosts_ports(), [("192.0.2.1", 16000)])
        self.assertEqual(select_endpoint(snapshot, "default", 99), "http://192.0.2.1:16000")
        self.assertEqual(
            [e["state"] for e in snapshot["models"]["default"]["engines"]], ["ready", "starting", "draining", "dead"]
        )
        manager._candidate_ranks.clear()
        ready = manager.get_inference_snapshot()
        self.assertGreater(ready["topology_revision"], snapshot["topology_revision"])
        self.assertEqual(select_endpoint(ready, "default", 1), "http://192.0.2.1:16001")
        manager.inference.state = "sleeping"
        self.assertEqual(manager.get_engine_hosts_ports(), [])
        self.assertEqual(manager.get_engine_capacity()["ready"], 0)

    def test_initial_split_placement_and_owned_elastic_placement_coexist(self):
        manager = _manager()
        manager.bundle_offset = 1
        manager.args._genrm_bundle_start = 3
        manager._scale_placements[1] = ("ELASTIC_PG", [0], [7])
        self.assertEqual(manager._resolve_placement(0), ("BASE_PG", False, 4))
        self.assertEqual(manager._resolve_placement(1), (("ELASTIC_PG", [0], [7]), True, 0))

    def test_retired_slot_does_not_trigger_recovery_or_repeat_onload(self):
        manager = _manager(2)
        manager.all_engines[1] = None
        manager._retired_scale_ranks.add(1)
        manager._onload_engines = MagicMock()
        manager.onload()
        manager.onload()
        manager._onload_engines.assert_not_called()
        self.assertEqual(manager.inference.state, "ready")

    def test_failed_recovery_closes_admission_even_with_a_surviving_engine(self):
        manager = _manager(2)
        manager.all_engines[1] = None
        manager._init_engines = MagicMock(side_effect=RuntimeError("rebuild failed"))
        with self.assertRaisesRegex(RuntimeError, "rebuild failed"):
            manager.onload()
        self.assertEqual(manager.inference.state, "failed")
        self.assertEqual(manager.get_engine_hosts_ports(), [])

    def test_unsupported_scaling_allocates_no_gpu(self):
        for deferred, state, gpus, nodes in [(True, "ready", 1, 1), (False, "sleeping", 1, 1), (False, "ready", 2, 2)]:
            with self.subTest(deferred=deferred, state=state, gpus=gpus):
                manager = _manager(2 if nodes == 2 else 1)
                manager.args.defer_reward_to_post_process = deferred
                manager.args.genrm_num_gpus_per_engine = gpus
                manager.nodes_per_engine = nodes
                manager.inference.state = state
                manager.begin_scale_op("scale", "scale_out", 2)
                manager._scale_abort["scale"] = threading.Event()
                manager._scale_progress["scale"] = manager._new_progress("CREATING")
                manager._scale_out_add_one = MagicMock()
                manager._scale_out_lifecycle("scale")
                manager._scale_out_add_one.assert_not_called()
                self.assertEqual(manager.get_scale_progress("scale")["phase"], "FAILED")
                self.assertTrue(manager.get_scale_progress("scale")["physical_done"])

    def test_offload_does_not_block_ray_drain_confirmation_during_scaling(self):
        manager = _manager()
        entered = threading.Event()
        release = threading.Event()

        def scaling():
            with manager._exclusive_lifecycle():
                entered.set()
                release.wait(timeout=5)

        thread = threading.Thread(target=scaling)
        thread.start()
        try:
            self.assertTrue(entered.wait(timeout=2))
            with self.assertRaisesRegex(RuntimeError, "busy"):
                manager.offload()
            self.assertEqual(manager.get_inference_snapshot()["role"], "genrm")
        finally:
            release.set()
            thread.join(timeout=2)
        self.assertFalse(thread.is_alive())

    def test_training_offload_pool_rejects_scaling_before_allocating_resources(self):
        for direction in ("scale_out", "scale_in"):
            with self.subTest(direction=direction):
                manager = _manager()
                manager.args.offload_rollout = True
                manager.begin_scale_op("scale", direction, 2)
                manager._scale_abort["scale"] = threading.Event()
                manager._scale_progress["scale"] = manager._new_progress("CREATING")
                manager._scale_out_add_one = MagicMock()
                manager._exclusive_lifecycle = MagicMock(wraps=manager._exclusive_lifecycle)
                getattr(manager, f"_{direction}_lifecycle")("scale")
                manager._scale_out_add_one.assert_not_called()
                manager._exclusive_lifecycle.assert_not_called()
                self.assertEqual(manager.get_scale_progress("scale")["phase"], "FAILED")
                self.assertIn("training offload", manager.get_scale_progress("scale")["error"])
                self.assertEqual(manager.inference.state, "ready")
                manager._offload_engines = MagicMock()
                manager.offload()
                manager._offload_engines.assert_called_once()

    def test_pending_pg_is_reconciled_before_reusing_lifecycle(self):
        manager = _manager(2)
        engine = MagicMock()
        engine.shutdown.remote.return_value = True
        manager.all_engines[1] = engine
        manager._scale_placements[1] = ("SCALE_PG", [0], [0])
        manager._engine_placements[1] = (manager._scale_placements[1], True)
        manager._scale_added_ranks = [1]
        manager._draining_ranks.add(1)
        manager.begin_scale_op("scale", "scale_in", 1)
        manager._scale_progress["scale"] = manager._new_progress("FAILED")
        manager._scale_progress["scale"].update(victim_rank=1, victim=["192.0.2.1", 16001])
        manager._scale_physical_done["scale"] = True
        # Exercise actual shutdown/PG bookkeeping, with the bounded wait reduced
        # to one poll so a pending Ray deletion is deterministic in this test.
        manager._wait_for_owned_pg_release = lambda rank: manager._remove_owned_pg(rank)
        with (
            patch.object(genrm_module.ray, "get", lambda ref, **kw: ref),
            patch.object(genrm_module.ray, "kill", lambda actor: None),
            _PgTablePatch("PENDING"),
        ):
            with self.assertRaisesRegex(RuntimeError, "cleanup unconfirmed"):
                manager._retire_scaled_engines([1])
        self.assertEqual(manager.inference.state, "ready")
        self.assertEqual(manager._cleanup_pending, {1})
        self.assertEqual(manager.get_engine_capacity()["pending_cleanup"], 1)
        with self.assertRaisesRegex(RuntimeError, "reconciled"):
            manager.onload()
        with _PgTablePatch("REMOVED"):
            result = manager.reconcile_scale_op("scale")
        self.assertTrue(result["victim_cleared"])
        self.assertFalse(result["cleanup_required"])
        self.assertFalse(manager._cleanup_pending)
        self.assertFalse(manager._pending_pg_cleanup)
        manager.onload()
        self.assertEqual(manager.inference.state, "ready")

    def test_failed_published_elastic_engine_gets_a_new_pg_for_recovery(self):
        manager = _manager(2)
        engine = MagicMock()
        engine.shutdown.remote.return_value = True
        manager.all_engines[1] = engine
        manager._scale_placements[1] = ("OLD_PG", [0], [0])
        manager._engine_placements[1] = (manager._scale_placements[1], True)
        manager._scale_added_ranks = [1]
        with (
            patch.object(genrm_module.ray, "get", lambda ref, **kw: ref),
            patch.object(genrm_module.ray, "kill", lambda actor: None),
            _PgTablePatch("REMOVED"),
        ):
            manager._retire_engines([1])
        self.assertNotIn(1, manager._scale_placements)
        self.assertNotIn(1, manager._retired_scale_ranks)
        manager._create_scale_pg = MagicMock(return_value=("NEW_PG", [0], [3]))
        self.assertEqual(manager._resolve_placement(1), (("NEW_PG", [0], [3]), True, 0))
        manager._create_scale_pg.assert_called_once_with(1)
