# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""GenRM Manager for Generative Reward Model Service.

This module implements a simplified manager for genRM engines, built on top of
``MultiEngineManager`` (parallel bring-up, health check, dead-engine recovery,
onload/offload) with GenRM-specific placement and engine wiring.
"""

import logging
import threading
import time

import ray

from relax.backends.sglang.sglang_engine import GenRMEngine
from relax.core.node_group_affinity import with_control_plane_affinity
from relax.distributed.ray.multi_engine_manager import MultiEngineManager, _is_engine_dead  # noqa: F401
from relax.distributed.ray.utils import NOSET_VISIBLE_DEVICES_ENV_VARS_LIST, Lock
from relax.utils.http_utils import init_http_client
from relax.utils.logging_utils import get_logger


logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

logger = get_logger(__name__)

_GENRM_PORT_BASE = 16000
_GENRM_PORT_WINDOW_SIZE = 1000
_MAX_PORT = 65535


@ray.remote
class GenRMManager(MultiEngineManager):
    """Manager for GenRM engines.

    This is a simplified version of RolloutManager focused on:
    - Initializing genRM engines
    - Health checking
    - Onload/offload operations
    """

    def __init__(self, args, pg, bundle_offset: int = 0, port_window_index: int = 0):
        init_http_client(args)

        num_gpu_per_engine = min(args.genrm_num_gpus_per_engine, args.num_gpus_per_node)
        num_slots = 0 if args.debug_train_only else args.genrm_num_gpus // num_gpu_per_engine
        nodes_per_engine = max(1, args.genrm_num_gpus_per_engine // args.num_gpus_per_node)

        self.pg = pg
        self.num_gpu_per_engine = num_gpu_per_engine
        self.bundle_offset = bundle_offset
        self.port_window_index = port_window_index

        # Task 4 elastic-scaling state must exist BEFORE super().__init__():
        # MultiEngineManager's constructor initializes engines, which calls
        # _resolve_placement / get_engine_hosts_ports on this instance while
        # the subclass body has not run yet (classic init-order hazard).
        self._scale_lock = threading.RLock()
        self._scale_progress: dict = {}
        self._scale_op_params: dict = {}
        self._scale_abort: dict = {}
        self._scale_added_ranks: list = []
        self._candidate_ranks: set = set()
        self._draining_ranks: set = set()
        self._scale_placements: dict = {}
        self._scale_drain_confirmed: dict = {}
        # A lifecycle can finish its public state only after this event is
        # recorded.  In particular, aborting a blocking PG wait is not the
        # same as the waiting thread having observed the abort and cleaned up.
        self._scale_physical_done: dict = {}
        # Retired elastic slots are intentionally not candidates for the base
        # manager's best-effort recovery path.
        self._retired_scale_ranks: set = set()
        # Unreleased resources are tracked separately from service capacity:
        # a failed scale-out candidate consumes a PG but was never current.
        self._failed_holding_ranks: set = set()
        self._unreleased_candidate_ranks: set = set()
        self._scale_drain_timeout_s = float(getattr(args, "genrm_scale_drain_timeout_s", 600.0))

        super().__init__(
            args,
            num_slots=num_slots,
            nodes_per_engine=nodes_per_engine,
            engine_actor_cls=GenRMEngine,
            skip_init=args.debug_train_only,
            log_prefix="GenRM",
        )
        self.num_new_engines = len(self.engines)
        self.genrm_engine_lock = Lock.options(
            **with_control_plane_affinity(self.args, {"num_cpus": 1, "num_gpus": 0})
        ).remote()

    def get_genrm_engines_and_lock(self):
        return self.engines, self.genrm_engine_lock, self.num_new_engines

    def get_engine_hosts_ports(self):
        """Return a list of (host, port) tuples for each live, routable genRM
        engine.

        This is used by the GenRM service to send HTTP generation requests
        directly to the underlying SGLang servers.

        The host/port information is captured during engine initialization
        from the addr_and_ports dict passed to ``_allocate_engine_addr_and_ports``.

        ``all_engines`` holds one entry per *node*, so a multi-node engine
        (genrm_num_gpus_per_engine > num_gpus_per_node) occupies
        ``nodes_per_engine`` consecutive ranks. Only node_rank 0 of each group
        runs the SGLang HTTP server -- the followers are compute-only workers
        and answer /generate with 404 -- so stride the same way the
        ``engines`` property does and return head nodes only.

        The list is also compacted over dead engines, so callers must swap it
        and any derived round-robin state together. Scale-out candidates that
        have not passed their health check and scale-in victims that are
        draining are not routable and are skipped here -- this is the
        admission boundary of a scale operation.
        """
        results = []
        for rank in range(0, len(self.all_engines), self.nodes_per_engine):
            engine = self.all_engines[rank]
            with self._scale_lock:
                hidden = rank in self._candidate_ranks or rank in self._draining_ranks
            if engine is not None and not hidden and rank in self._engine_addr_and_ports:
                info = self._engine_addr_and_ports[rank]
                results.append((info["host"], info["port"]))
        return results

    # ------------------------------------------------------------------
    # Task 4 elastic scaling: manager-side physical lifecycle.
    #
    # The contract with the component is intentionally small:
    #   begin_scale_op(request_id, direction, target)   - register parameters
    #   execute_genrm_scale_out/in(request_id)          - fire-and-forget hook
    #   get_scale_progress(request_id)                  - polled by the watcher
    #   confirm_scale_drained(request_id)               - component drain proof
    #   abort_scale_op(request_id)                      - watcher deadline
    #
    # The component owns the operation registry (HTTP status queries); this
    # manager owns the physical lifecycle and reports progress only.
    # ------------------------------------------------------------------

    def begin_scale_op(self, request_id: str, direction: str, target: int) -> None:
        """Register the parameters of one scale operation before the hook."""
        with self._scale_lock:
            self._scale_op_params[request_id] = {
                "direction": direction,
                "target": int(target),
            }

    def execute_genrm_scale_out(self, request_id: str) -> None:
        """Scale-out hook: bring up (target - current) new engines, one at a
        time, each on its own placement group, published only after its health
        check passes. Runs in a daemon thread; progress via
        ``get_scale_progress``."""
        with self._scale_lock:
            self._scale_progress[request_id] = self._new_progress("CREATING")
            self._scale_abort[request_id] = threading.Event()
            self._scale_physical_done[request_id] = False
        threading.Thread(
            target=self._scale_out_lifecycle,
            args=(request_id,),
            daemon=True,
            name=f"genrm-scale-out-{request_id}",
        ).start()

    def execute_genrm_scale_in(self, request_id: str) -> None:
        """Scale-in hook: drain and remove (current - target) newest scale-added
        engines one at a time. Admission closes as soon as a victim enters
        ``_draining_ranks``; actual removal waits for the component's drain
        confirmation (its in-flight counter for the victim reached zero)."""
        with self._scale_lock:
            self._scale_progress[request_id] = self._new_progress("DRAINING")
            self._scale_abort[request_id] = threading.Event()
            self._scale_physical_done[request_id] = False
        threading.Thread(
            target=self._scale_in_lifecycle,
            args=(request_id,),
            daemon=True,
            name=f"genrm-scale-in-{request_id}",
        ).start()

    def get_scale_progress(self, request_id: str) -> dict:
        """Snapshot of one operation's progress (empty dict while unknown)."""
        with self._scale_lock:
            return dict(self._scale_progress.get(request_id) or {})

    def confirm_scale_drained(self, request_id: str) -> None:
        """Component-side drain proof: no in-flight request targets the current
        victim any more."""
        with self._scale_lock:
            event = self._scale_drain_confirmed.get(request_id)
        if event is not None:
            event.set()

    def abort_scale_op(self, request_id: str) -> None:
        """Ask the lifecycle thread to stop at the next safe point.

        Draining victims stay unroutable and keep their resources (failure
        isolation); no new engine is created after the abort.
        """
        with self._scale_lock:
            event = self._scale_abort.get(request_id)
        if event is not None:
            event.set()

    # -- internals -------------------------------------------------------

    def _capacity_counts(self) -> tuple:
        """RFC capacity semantics: ``current`` counts live published engines
        plus draining victims; ``ready`` counts only routable engines
        (admission open).

        Failed cleanup is resource occupancy, not service capacity, so
        unpublished candidates never inflate ``current``.
        """
        current = 0
        for rank in range(0, len(self.all_engines), self.nodes_per_engine):
            with self._scale_lock:
                is_candidate = rank in self._candidate_ranks
            if self.all_engines[rank] is not None and not is_candidate:
                current += 1
        # A failed scale-in victim was published capacity before its removal.
        # Until its actor and PG are physically released it remains current
        # (but is not ready because it stays draining).
        with self._scale_lock:
            holding = len(self._failed_holding_ranks)
        return current + holding, self._live_published_count()

    def _occupied_count(self) -> int:
        """Count slots/resources still held, including unpublished failures."""
        live = sum(
            self.all_engines[rank] is not None for rank in range(0, len(self.all_engines), self.nodes_per_engine)
        )
        with self._scale_lock:
            stranded = self._failed_holding_ranks | self._unreleased_candidate_ranks
        return live + len(stranded)

    def get_engine_capacity(self) -> dict:
        """Capacity snapshot for the component's /engines endpoint:

        ``current`` (service capacity), ``ready`` (routable capacity), and
        separately observable physical resource occupancy.
        """
        current, ready = self._capacity_counts()
        with self._scale_lock:
            pending_cleanup = len(self._failed_holding_ranks | self._unreleased_candidate_ranks)
        return {
            "current": current,
            "ready": ready,
            "occupied": self._occupied_count(),
            "pending_cleanup": pending_cleanup,
        }

    def _wait_for_owned_pg_release(self, rank: int, timeout_s: float = 60.0) -> bool:
        """Wait for Ray to confirm an owned elastic PG is REMOVED.

        ``remove_placement_group`` is asynchronous.  This bounded wait is used
        only on lifecycle/reconcile paths, never on request routing; a timeout
        leaves the owner in ``_pending_pg_cleanup`` for retry.
        """
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if self._remove_owned_pg(rank):
                return True
            time.sleep(0.25)
        return self._remove_owned_pg(rank)

    def reconcile_scale_op(self, request_id: str) -> dict:
        """Retry unfinished cleanup for one operation (RFC reconcile).

        scale-out: retry removal of placement groups whose release failed;
        no new replica is created. scale-in: retire the draining victim the
        operation already selected (never a new one) once the component has
        proven its in-flight count at zero, then retry its PG release. On
        success the component clears the registry's cleanup_required flag;
        the operation's terminal status itself is immutable.
        """
        with self._scale_lock:
            progress = dict(self._scale_progress.get(request_id) or {})
        result = {
            "request_id": request_id,
            "known": bool(progress),
            "victim_cleared": False,
            "cleanup_required": False,
            "in_progress": False,
        }
        if not progress:
            return result
        with self._scale_lock:
            physical_done = self._scale_physical_done.get(request_id, False)
        if not physical_done:
            result["cleanup_required"] = True
            result["in_progress"] = True
            return result
        direction = (self._scale_target(request_id) or (None, None))[1]
        victim_rank = progress.get("victim_rank")
        if direction == "scale_in" and victim_rank is not None:
            with self._scale_lock:
                draining = victim_rank in self._draining_ranks
                alive = self.all_engines[victim_rank] is not None if victim_rank < len(self.all_engines) else False
            if draining and alive:
                # Component verified in-flight zero before calling reconcile.
                self._retire_scaled_engines([victim_rank])
                with self._scale_lock:
                    self._failed_holding_ranks.add(victim_rank)
        elif direction == "scale_out":
            pass  # nothing to do beyond the pending-PG retry below
        # Retry every failed PG release; anything still pending keeps
        # blocking new scale operations for this manager.
        # The request owns one fixed victim.  Do not claim it clear until the
        # asynchronous Ray removal has reached REMOVED.
        if direction == "scale_in" and victim_rank is not None:
            self._wait_for_owned_pg_release(victim_rank)
        still_pending = self.retry_pending_pg_cleanup()
        with self._scale_lock:
            # Failed scale-out candidates whose PG release is now confirmed
            # leave the unreleased set: retire them exactly like freshly
            # failed candidates so recover() never resolves their stale
            # (REMOVED) scale placement.
            for rank in list(self._unreleased_candidate_ranks):
                if rank not in self._pending_pg_cleanup:
                    self._retired_scale_ranks.add(rank)
                    self._scale_placements.pop(rank, None)
            self._failed_holding_ranks = {r for r in self._failed_holding_ranks if r in self._pending_pg_cleanup}
            self._unreleased_candidate_ranks = {
                r for r in self._unreleased_candidate_ranks if r in self._pending_pg_cleanup
            }
            still_holding = set(self._failed_holding_ranks)
            victim_released = victim_rank is not None and victim_rank not in self._failed_holding_ranks
            if direction == "scale_in" and victim_released:
                self._draining_ranks.discard(victim_rank)
                if victim_rank in self._scale_added_ranks:
                    self._scale_added_ranks.remove(victim_rank)
        result["victim_cleared"] = bool(direction == "scale_in" and victim_rank is not None and victim_released)
        result["cleanup_required"] = bool(still_pending) or bool(still_holding)
        self._update_progress(request_id, cleanup_required=result["cleanup_required"])
        return result

    def _new_progress(self, phase: str) -> dict:
        current, ready = self._capacity_counts()
        return {
            "phase": phase,
            "created": 0,
            "removed": 0,
            "failed": 0,
            "current": current,
            "ready": ready,
            "cleanup_required": False,
            "physical_done": False,
            "victim": None,
            "victim_rank": None,
            "error": None,
        }

    def _live_published_count(self) -> int:
        count = 0
        for rank in range(0, len(self.all_engines), self.nodes_per_engine):
            with self._scale_lock:
                hidden = rank in self._candidate_ranks or rank in self._draining_ranks
            if self.all_engines[rank] is not None and not hidden:
                count += 1
        return count

    def _update_progress(self, request_id: str, **fields) -> None:
        with self._scale_lock:
            progress = self._scale_progress.get(request_id)
            if progress is None:
                return
            progress.update(fields)
            current, ready = self._capacity_counts()
            progress["current"] = current
            progress["ready"] = ready

    def _scale_target(self, request_id: str):
        with self._scale_lock:
            params = self._scale_op_params.get(request_id) or {}
        return params.get("target"), params.get("direction")

    def _create_scale_pg(self, rank: int):
        """Create one dedicated single-GPU placement group for a scale-out
        engine (RFC: new replicas request free resources on their own PG).

        The PG is registered in the ownership maps immediately after creation
        -- before the readiness wait and the physical-GPU probe -- so any
        failure on those paths reaches the caller's fenced release instead of
        stranding an unowned PG (RFC: register the cleanup handle at PG
        request time).
        """
        import ray
        from ray.util.placement_group import placement_group
        from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

        from relax.distributed.ray.placement_group import InfoActor
        from relax.utils.utils import get_ray_accelerator_kwargs

        pg = placement_group([{"GPU": 1.0, "CPU": 2.0}], strategy="STRICT_PACK")
        # Ownership before readiness.  The placeholder gpu id is never used
        # for scheduling: _resolve_placement is only reached via _init_engines
        # (which runs after the probed tuple below is stored) and recover()
        # skips candidate ranks; _engine_placements is what makes
        # _remove_owned_pg able to see the PG from this moment on.
        pg_tuple = (pg, [0], [0])
        with self._scale_lock:
            self._scale_placements[rank] = pg_tuple
            self._engine_placements[rank] = (pg_tuple, True)
        # ray.util.placement_group exports no wait_for_ready in ray 2.5x;
        # pg.ready() + ray.wait(timeout=...) is the supported readiness wait
        # (same pattern as the rollout scale-out PG polling).
        ready, _ = ray.wait([pg.ready()], timeout=120)
        if not ready:
            # The caller's fenced release owns PG removal; removing here
            # would bypass the REMOVED-confirmation bookkeeping.
            raise RuntimeError("scale-out placement group did not become ready within 120s")
        # Probe the physical GPU inside the bundle.  Engine env vars keep
        # RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1 (multi-GPU engines
        # address physical GPUs via base_gpu_id), so the PG's gpu id list
        # must contain the *physical* GPU id -- a local ``0`` here would
        # point the scaled-out engine at physical GPU 0 and collide with
        # the initial engine.  Same InfoActor probing rollout uses for its
        # per-replica PGs.
        info = InfoActor.options(
            scheduling_strategy=PlacementGroupSchedulingStrategy(
                placement_group=pg,
                placement_group_bundle_index=0,
            ),
            **get_ray_accelerator_kwargs(1),
        ).remote()
        try:
            _, gpu_id = ray.get(info.get_ip_and_gpu_id.remote(), timeout=60)
        finally:
            # The InfoActor is scheduled on the PG bundle, so PG removal
            # reaps it anyway; a kill failure must not mask the original
            # error nor skip the caller's fenced release.
            try:
                ray.kill(info)
            except Exception:  # noqa: BLE001
                logger.warning("GenRM scale-out: failed to kill PG probe InfoActor", exc_info=True)
        pg_tuple = (pg, [0], [int(gpu_id)])
        with self._scale_lock:
            self._scale_placements[rank] = pg_tuple
            self._engine_placements[rank] = (pg_tuple, True)
        return pg_tuple

    def _scale_out_lifecycle(self, request_id: str) -> None:
        abort = self._scale_abort[request_id]
        target, direction = self._scale_target(request_id)
        created = 0
        try:
            if direction != "scale_out" or target is None:
                self._update_progress(request_id, phase="FAILED", error="scale-out parameters were not registered")
                return
            if self.num_gpu_per_engine != 1:
                self._update_progress(
                    request_id,
                    phase="FAILED",
                    error=f"elastic scale-out supports single-GPU engines only (num_gpu_per_engine={self.num_gpu_per_engine})",
                )
                return

            while self._live_published_count() < target:
                if abort.is_set():
                    raise RuntimeError("aborted")
                self._update_progress(request_id, phase="CREATING")
                self._scale_out_add_one(request_id)
                created += 1
                self._update_progress(request_id, created=created)
            self._update_progress(request_id, phase="ACTIVE", created=created, failed=0)
        except Exception as exc:
            logger.exception(f"GenRM scale-out {request_id} failed")
            # First failure stops the operation (RFC): keep what was
            # published, report PARTIAL when at least one replica made it.
            self._update_progress(
                request_id, phase="PARTIAL" if created > 0 else "FAILED", created=created, failed=1, error=str(exc)
            )
        finally:
            with self._scale_lock:
                self._scale_physical_done[request_id] = True
            self._update_progress(request_id, physical_done=True)

    def _scale_out_add_one(self, request_id: str) -> int:
        """Bring up one candidate engine and publish it after its health check.

        Raises on failure after cleaning the candidate up. Progress phases may
        oscillate per engine (CREATING -> HEALTH_CHECKING); the component's
        watcher maps them onto the registry's monotonic chain.
        """
        abort = self._scale_abort[request_id]
        # Allocate the rank before creating the PG: _create_scale_pg
        # registers ownership under this rank immediately after creation, so
        # every failure exit (readiness wait, InfoActor create/probe/kill)
        # flows through the fenced release in the except handler below
        # instead of stranding an unowned PG.
        with self._scale_lock:
            rank = len(self.all_engines)
            self.all_engines.append(None)
            self._candidate_ranks.add(rank)
        try:
            pg_tuple = self._create_scale_pg(rank)
            with self._scale_lock:
                self._scale_placements[rank] = pg_tuple
            # Record ownership before checking abort: PG creation is not
            # interruptible, so an abort immediately after it must flow
            # through the normal confirmed-cleanup path.
            if abort.is_set():
                raise RuntimeError("aborted while waiting for the placement group")
            # _init_engines creates the actor on the scale PG, inits it, and
            # on failure kills it and removes the owned PG for us.
            self._init_engines([rank])
            if abort.is_set():
                raise RuntimeError("aborted after engine init (late init must not publish)")
            self._update_progress(request_id, phase="HEALTH_CHECKING")
            engine = self.all_engines[rank]
            healthy = False
            try:
                healthy = ray.get(engine.health_generate.remote(), timeout=120)
            except Exception as exc:
                logger.warning(f"GenRM scale-out health check raised for rank={rank}: {exc}")
            if abort.is_set():
                raise RuntimeError("aborted after health check (late engine must not publish)")
            if not healthy:
                raise RuntimeError(f"health check failed for scaled-out engine rank={rank}")
            with self._scale_lock:
                self._candidate_ranks.discard(rank)
                self._scale_added_ranks.append(rank)
            logger.info(f"GenRM scale-out {request_id}: engine rank={rank} published")
            return rank
        except Exception:
            with self._scale_lock:
                self._candidate_ranks.discard(rank)
                engine = self.all_engines[rank]
                self.all_engines[rank] = None
            if engine is not None:
                try:
                    ray.get(engine.shutdown.remote(), timeout=30)
                except Exception:
                    pass
                try:
                    ray.kill(engine)
                except Exception:
                    pass
            self._release_owned_scale_pg(rank, request_id)
            raise

    def _release_owned_scale_pg(self, rank: int, request_id: str, *, timeout_s: float = 60.0) -> bool:
        """Fenced release of a scale-out PG owned by ``rank``: submit removal,
        poll until Ray confirms REMOVED, then deregister the owner.

        Returns True when the release is confirmed. On timeout the rank stays
        in ``_pending_pg_cleanup`` and ``_unreleased_candidate_ranks`` with
        ``cleanup_required`` set on the operation, keeping the model's scale
        mutex until reconcile confirms the release.
        """
        if self._wait_for_owned_pg_release(rank, timeout_s=timeout_s):
            # Release confirmed: the failed candidate's slot is gone for
            # good.  Retire it (the same recovery exemption a scale-in
            # victim gets) and drop the placement so recover() can neither
            # rebuild the slot on the stale REMOVED PG nor on an
            # out-of-range default placement -- a failed scale-out attempt
            # is not recoverable capacity; the next scale-out appends a
            # fresh slot with a fresh PG instead.
            with self._scale_lock:
                self._retired_scale_ranks.add(rank)
                self._scale_placements.pop(rank, None)
            return True
        # PG release failed: the rank still occupies resources and must keep
        # blocking new scale operations until reconcile.
        self._update_progress(request_id, cleanup_required=True)
        with self._scale_lock:
            self._unreleased_candidate_ranks.add(rank)
        return False

    def _scale_in_lifecycle(self, request_id: str) -> None:
        abort = self._scale_abort[request_id]
        target, direction = self._scale_target(request_id)
        removed = 0
        try:
            if direction != "scale_in" or target is None:
                raise RuntimeError("scale-in parameters were not registered")
            while self._live_published_count() > target:
                if abort.is_set():
                    raise RuntimeError("aborted")
                with self._scale_lock:
                    candidates = [
                        r
                        for r in self._scale_added_ranks
                        if r not in self._draining_ranks and self.all_engines[r] is not None
                    ]
                if not candidates:
                    raise RuntimeError("no scale-added engine left to remove")
                victim = max(candidates)  # newest first (RFC)
                # Register the drain event before the victim becomes
                # visible: the watcher can observe DRAINING + victim and
                # confirm within the same poll cycle, so the event must
                # already exist (a dropped confirm would stall the lifecycle
                # until the drain timeout).
                event = threading.Event()
                with self._scale_lock:
                    self._scale_drain_confirmed[request_id] = event
                    self._draining_ranks.add(victim)
                    info = self._engine_addr_and_ports.get(victim) or {}
                victim_addr = [info.get("host"), info.get("port")]
                self._update_progress(request_id, victim=victim_addr, victim_rank=victim)
                # admission is closed for the victim (get_engine_hosts_ports
                # skips it from here on); wait for the component's drain proof.
                if not event.wait(timeout=self._scale_drain_timeout_s):
                    # Keep the victim unroutable with resources retained;
                    # recovery is reconcile / shutdown, never auto-restore.
                    self._update_progress(request_id, cleanup_required=True)
                    raise RuntimeError("drain confirmation timed out")
                if abort.is_set():
                    self._update_progress(request_id, cleanup_required=True)
                    raise RuntimeError("aborted")
                self._update_progress(request_id, phase="REMOVING", victim=victim_addr, victim_rank=victim)
                self._retire_scaled_engines([victim])
                self._wait_for_owned_pg_release(victim)
                with self._scale_lock:
                    victim_pg_pending = victim in self._pending_pg_cleanup
                    if victim_pg_pending:
                        # Retire succeeded but PG release failed: the victim
                        # keeps its slot accounted until reconcile releases it.
                        self._failed_holding_ranks.add(victim)
                if victim_pg_pending:
                    self._update_progress(request_id, cleanup_required=True)
                    # Do not choose another victim or claim this one removed:
                    # its old capacity remains occupied until Ray confirms PG
                    # deletion and reconcile clears this exact owner.
                    raise RuntimeError("victim placement-group cleanup is pending")
                with self._scale_lock:
                    self._draining_ranks.discard(victim)
                    if victim in self._scale_added_ranks:
                        self._scale_added_ranks.remove(victim)
                removed += 1
                self._update_progress(request_id, phase="DRAINING", removed=removed, victim=None, victim_rank=None)
            self._update_progress(request_id, phase="COMPLETED", removed=removed, failed=0)
        except Exception as exc:
            logger.exception(f"GenRM scale-in {request_id} failed")
            self._update_progress(request_id, phase="FAILED", removed=removed, failed=1, error=str(exc))
        finally:
            with self._scale_lock:
                self._scale_drain_confirmed.pop(request_id, None)
                self._scale_physical_done[request_id] = True
            self._update_progress(request_id, physical_done=True)

    def _retire_scaled_engines(self, ranks: list[int]) -> None:
        """Retire elastic replicas without making them recoverable holes."""
        with self._scale_lock:
            self._retired_scale_ranks.update(ranks)
        super()._retire_engines(ranks)

    def recover(self) -> set:
        """Recover failures, but never recreate intentionally scaled-in
        slots."""
        with self._scale_lock:
            excluded = (
                set(self._retired_scale_ranks)
                | set(self._candidate_ranks)
                | set(self._draining_ranks)
                | set(self._failed_holding_ranks)
                | set(self._unreleased_candidate_ranks)
            )
        dead = [i for i, engine in enumerate(self.all_engines) if engine is None and i not in excluded]
        if not dead:
            return set()
        logger.info(f"GenRM recovering {len(dead)} engine(s): ranks={dead}")
        try:
            self._init_engines(dead)
        except Exception as exc:
            logger.exception(f"GenRM engine rebuild failed for ranks={dead}: {exc}")
        rebuilt = {i for i in dead if self.all_engines[i] is not None}
        still_dead = [i for i in dead if i not in rebuilt]
        live = [engine for i, engine in enumerate(self.all_engines) if i not in excluded and engine is not None]
        if still_dead:
            if not live:
                raise RuntimeError(f"All recoverable GenRM engines are dead (ranks={still_dead})")
            logger.error(f"GenRM engines still dead after recovery, continuing degraded: ranks={still_dead}")
        return rebuilt

    # ------------------------------------------------------------------
    # MultiEngineManager hooks.
    # ------------------------------------------------------------------

    def _resolve_placement(self, rank):
        with self._scale_lock:
            scale_pg = self._scale_placements.get(rank)
        if scale_pg is not None:
            # Engine slot created by a scale-out operation: it owns its own
            # dedicated placement group (removed on retire/shutdown).
            return scale_pg, True, 0
        gpu_idx = rank * self.num_gpu_per_engine + self.bundle_offset
        shared_with_rollout = getattr(self.args, "_genrm_colocate_with_rollout", False)
        if not self.args.fully_async and not shared_with_rollout:
            gpu_idx += self.args.rollout_num_gpus

        return self.pg, False, gpu_idx

    def _ray_resource_kwargs(self, rank):
        # Lower default fractional-GPU footprint when sharing bundles with
        # rollout (rollout uses 0.2 per actor; 0.2 + 0.2 risks Ray scheduler
        # rejection).
        shared_with_rollout = getattr(self.args, "_genrm_colocate_with_rollout", False)
        default_ray_num_gpus = 0.1 if shared_with_rollout else 0.2
        num_gpus = getattr(self.args, "genrm_ray_num_gpus", default_ray_num_gpus)
        return {"num_cpus": num_gpus, "num_gpus": num_gpus}

    def _build_engine_env_vars(self):
        env_vars = {name: "1" for name in NOSET_VISIBLE_DEVICES_ENV_VARS_LIST} | {
            "SGL_JIT_DEEPGEMM_PRECOMPILE": "false",
            "SGLANG_JIT_DEEPGEMM_PRECOMPILE": "false",
            # See rollout.py: recent SGLang reads SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK
            # (default True) and the deprecation shim value-copies SGL_DISABLE_* into it,
            # so the old DISABLE vars re-enable the check. Set ENABLE=false directly.
            "SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK": "false",
            "SGLANG_MEMORY_SAVER_CUDA_GRAPH": "true",
            "SGLANG_BATCH_INVARIANT_OPS_ENABLE_MM_FALLBACK_VARIANT": "true",
            "SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION": "false",
            "SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_IDLE": "false",
            # NOTE: disable custom all-reduce-v2, same as rollout.py — avoids
            # custom_all_reduce.cuh:37: CUDA error: invalid argument during CUDA graph capture.
            "SGLANG_OPT_USE_CUSTOM_ALL_REDUCE_V2": "0",
        }
        if getattr(self.args, "fp16", False):
            env_vars["SGLANG_MAMBA_CONV_DTYPE"] = "float16"
        return env_vars

    def _allocate_engine_addr_and_ports(self, *, new_engines):
        return _allocate_genrm_engine_addr_and_ports(
            args=self.args,
            new_engines=new_engines,
            port_window_index=self.port_window_index,
        )


def _allocate_genrm_engine_addr_and_ports(*, args, new_engines, port_window_index=0):
    """Allocate network addresses and ports for genRM engines.

    Similar to _allocate_rollout_engine_addr_and_ports_normal but for genRM.

    ``port_window_index`` selects a disjoint range per GenRM instance (multi-instance
    --genrm-instances): every instance is a separate GenRMManager that probes
    for free ports independently and in parallel during Serve replica
    initialization, so two instances starting from the same base port can both
    see a given port as free (probe-then-bind race) and then collide when
    their SGLang servers actually bind it.
    """
    window_start = _GENRM_PORT_BASE + port_window_index * _GENRM_PORT_WINDOW_SIZE
    window_end = window_start + _GENRM_PORT_WINDOW_SIZE - 1
    if port_window_index < 0 or window_end > _MAX_PORT:
        raise ValueError(
            f"GenRM port window index {port_window_index} is out of range; "
            f"at most {(_MAX_PORT - _GENRM_PORT_BASE + 1) // _GENRM_PORT_WINDOW_SIZE} instances are supported."
        )

    num_engines_per_node = max(1, min(args.num_gpus_per_node, args.genrm_num_gpus) // args.genrm_num_gpus_per_engine)
    addr_and_ports = {}

    visited_nodes = set()
    for rank, engine in new_engines:
        if rank // num_engines_per_node in visited_nodes:
            continue
        visited_nodes.add(rank // num_engines_per_node)
        num_engines_on_this_node = num_engines_per_node - (rank % num_engines_per_node)

        def get_addr_and_ports(engine):
            # Use a different, bounded port range from rollout (15000).
            start_port = window_start

            def port(consecutive=1):
                nonlocal start_port
                _, port = ray.get(
                    engine._get_current_node_ip_and_free_port.remote(
                        start_port=start_port,
                        consecutive=consecutive,
                        max_port=window_end,
                    )
                )
                start_port = port + consecutive
                return port

            def addr():
                addr, _ = ray.get(engine._get_current_node_ip_and_free_port.remote())
                return addr

            return addr, port

        get_addr, get_port = get_addr_and_ports(engine)

        for i in range(num_engines_on_this_node):
            current_rank = rank + i
            addr_and_ports.setdefault(current_rank, {})
            addr_and_ports[current_rank]["host"] = get_addr()
            addr_and_ports[current_rank]["port"] = get_port()
            addr_and_ports[current_rank]["nccl_port"] = get_port()

        if args.genrm_num_gpus_per_engine > args.num_gpus_per_node:
            num_node_per_engine = args.genrm_num_gpus_per_engine // args.num_gpus_per_node
            if rank % num_node_per_engine == 0:
                # First node in the engine, allocate dist_init_addr port
                dist_init_addr = f"{get_addr()}:{get_port(30 + args.sglang_dp_size)}"
                for i in range(num_node_per_engine):
                    addr_and_ports.setdefault(rank + i, {})
                    addr_and_ports[rank + i]["dist_init_addr"] = dist_init_addr
        else:
            for i in range(num_engines_on_this_node):
                addr_and_ports.setdefault(rank + i, {})
                addr_and_ports[rank + i]["dist_init_addr"] = f"{get_addr()}:{get_port(30 + args.sglang_dp_size)}"

    for rank, _ in new_engines:
        for key in ["port", "nccl_port", "dist_init_addr"]:
            assert key in addr_and_ports[rank], f"GenRM engine rank={rank} {key} is not set."
        logger.info(f"Ports for genRM engine rank={rank}: {addr_and_ports[rank]}")

    return addr_and_ports
