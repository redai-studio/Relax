# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Common lifecycle skeleton for managers that own a pool of SGLang engine
replicas (GenRM judges, OPD teachers, ...).

Concrete managers subclass ``MultiEngineManager`` (in addition to their own
``@ray.remote`` decorator) and implement the hooks below to plug in their
engine actor class, placement, GPU/port allocation, and env vars. The base
class owns: parallel engine bring-up, health checking, dead-engine
detection/retirement, recovery, and onload/offload with idempotency tracking.

Placement is resolved per engine (not once per manager): a manager may put
all engines on one shared placement group (e.g. GenRM colocated with
rollout), or give each engine its own dedicated placement group that it
creates and tears down itself (e.g. a non-colocated OPD teacher). Subclasses
express this via ``_resolve_placement``.
"""

from typing import Any, Optional

import ray
import requests
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from relax.utils.logging_utils import get_logger


logger = get_logger(__name__)

# An engine process can die on its own (e.g. SGLang's scheduler watchdog
# SIGQUITs the server after a CUDA-level hang). The next call into it then
# raises one of these. Everything else is a real bug and must propagate.
#   - ConnectionError / TimeoutError: raised by the engine's
#     release_memory_occupation when a drain loop hits its dead-server
#     fast-fail or its deadline.
#   - requests.exceptions.{ConnectionError,Timeout}: raised by _make_request,
#     i.e. the resume_memory_occupation path. These are OSError subclasses but
#     NOT builtin ConnectionError/TimeoutError, so they must be listed
#     explicitly -- otherwise an engine that died during the offloaded window
#     (only observable at onload) escalates to a global restart.
#   - RayActorError: the Ray actor itself is gone.
# ray.get re-raises as a class inheriting from BOTH RayTaskError and the
# original cause (ray/exceptions.py::as_instanceof_cause), so isinstance works.
_ENGINE_DEAD_EXCEPTIONS = (
    ConnectionError,
    TimeoutError,
    requests.exceptions.ConnectionError,
    requests.exceptions.Timeout,
    ray.exceptions.RayActorError,
)

# Rebuilding one engine is ~1.5 min (weight load + cuda graph capture). Bound it
# so a dead *node* -- whose placement-group bundle can never be filled -- degrades
# to "run with N-1 engines" instead of hanging the training step forever.
_ENGINE_REBUILD_TIMEOUT_S = 900.0
_ENGINE_SHUTDOWN_TIMEOUT_S = 60.0


def _is_engine_dead(exc: BaseException) -> bool:
    return isinstance(exc, _ENGINE_DEAD_EXCEPTIONS)


class MultiEngineManager:
    """Base class for managers of a fixed-size pool of engine replicas.

    Not a Ray actor itself -- subclasses apply ``@ray.remote`` so this class
    can be unit-tested without a Ray runtime. Engine *slots* are tracked as a
    flat list indexed by rank; a ``None`` slot means "dead, needs rebuild".

    A slot is the unit of scheduling (one Ray actor, one placement-group
    bundle range). ``nodes_per_engine > 1`` lets one *logical* engine span
    multiple slots/nodes (e.g. a TP group larger than one node): ``num_slots``
    passed to ``__init__`` already accounts for this (it is node-count, not
    logical-engine-count), and only the head slot of each group
    (``rank % nodes_per_engine == 0``) runs the HTTP server, so ``engines``
    strides over the followers.
    """

    def __init__(
        self,
        args: Any,
        *,
        num_slots: int,
        nodes_per_engine: int = 1,
        engine_actor_cls: type,
        skip_init: bool = False,
        log_prefix: str = "",
    ) -> None:
        self.args = args
        self.engine_actor_cls = engine_actor_cls
        self.nodes_per_engine = max(1, nodes_per_engine)
        self._log_prefix = log_prefix

        self.all_engines: list[Any] = [None] * num_slots
        # Per-slot (pg_tuple, owns_pg) so shutdown()/_retire_engines() only
        # remove placement groups this manager itself created.
        self._engine_placements: dict[int, tuple] = {}
        self._engine_addr_and_ports: dict[int, dict] = {}
        # Ranks whose owned placement-group removal failed and whose handle is
        # retained for retry (Task 4: unconfirmed resource release keeps
        # blocking new scale operations until reconcile succeeds).
        self._pending_pg_cleanup: set[int] = set()
        # Track memory-occupation state so repeated onload/offload calls become
        # safe no-ops. Engines start onloaded; callers may immediately offload.
        self._onloaded = True

        if not skip_init:
            self._init_engines(list(range(num_slots)))

    @property
    def engines(self) -> list[Any]:
        """Return the head-node slot of each logical engine."""
        return self.all_engines[:: self.nodes_per_engine]

    # ------------------------------------------------------------------
    # Hooks -- subclasses must implement these.
    # ------------------------------------------------------------------

    def _resolve_placement(self, rank: int) -> tuple[tuple, bool, int]:
        """Return ``(pg_tuple, owns_pg, gpu_index)`` for the slot at ``rank``.

        ``pg_tuple`` is ``(pg, reordered_bundle_indices, reordered_gpu_ids)``
        as returned by ``create_placement_group``. ``owns_pg`` marks whether
        this manager created ``pg_tuple`` itself (and must remove it on
        shutdown/retirement) or is borrowing a placement group it does not
        own. ``gpu_index`` is this slot's starting index into
        ``reordered_gpu_ids``/``reordered_bundle_indices``.
        """
        raise NotImplementedError

    def _ray_resource_kwargs(self, rank: int) -> dict:
        """Return the num_cpus/num_gpus fractional Ray resource request for one
        slot."""
        raise NotImplementedError

    def _allocate_engine_addr_and_ports(self, *, new_engines: list[tuple]) -> dict[int, dict]:
        """Allocate host/port/dist_init_addr for the given (rank, engine)
        pairs.

        Returns a dict keyed by rank; each value must contain at least
        ``host``, ``port``, ``nccl_port``, ``dist_init_addr``.
        """
        raise NotImplementedError

    def _build_engine_env_vars(self) -> dict[str, str]:
        """Return the runtime_env env vars for a new engine actor."""
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Hooks -- subclasses may override; sane defaults provided.
    # ------------------------------------------------------------------

    def _engine_ctor_args(self, rank: int) -> Any:
        """First positional argument passed to the engine actor constructor."""
        return self.args

    def _build_engine_ctor_kwargs(self, rank: int) -> dict:
        """Extra keyword arguments passed to the engine actor constructor
        (beyond rank/worker_type/base_gpu_id)."""
        return {}

    def _build_engine_init_kwargs(self, rank: int, addr_and_ports: dict) -> dict:
        """Keyword arguments passed to ``engine.init.remote(...)``."""
        return dict(addr_and_ports)

    # ------------------------------------------------------------------
    # Engine bring-up.
    # ------------------------------------------------------------------

    def _init_engines(self, ranks: list[int]) -> int:
        """Create actors for the given slot ranks, fire init.remote() for all
        of them without blocking, then await everything in one ray.get so a
        large engine doesn't pay N x cold-load latency.

        On failure, kill any newly created engines and leave their slots None
        so the caller sees them as still-dead rather than silently healthy.
        """
        EngineActor = ray.remote(self.engine_actor_cls)
        new_engines: list[tuple[int, Any]] = []
        for rank in ranks:
            if self.all_engines[rank] is not None:
                continue

            pg_tuple, owns_pg, gpu_index = self._resolve_placement(rank)
            pg, reordered_bundle_indices, reordered_gpu_ids = pg_tuple
            base_gpu_id = int(reordered_gpu_ids[gpu_index])
            scheduling_strategy = PlacementGroupSchedulingStrategy(
                placement_group=pg,
                placement_group_capture_child_tasks=True,
                placement_group_bundle_index=reordered_bundle_indices[gpu_index],
            )

            engine = EngineActor.options(
                **self._ray_resource_kwargs(rank),
                scheduling_strategy=scheduling_strategy,
                runtime_env={"env_vars": self._build_engine_env_vars()},
            ).remote(
                self._engine_ctor_args(rank),
                rank=rank,
                worker_type="regular",
                base_gpu_id=base_gpu_id,
                **self._build_engine_ctor_kwargs(rank),
            )
            new_engines.append((rank, engine))
            self.all_engines[rank] = engine
            self._engine_placements[rank] = (pg_tuple, owns_pg)

        num_new_engines = len(new_engines)
        if num_new_engines == 0:
            return num_new_engines

        addr_and_ports = self._allocate_engine_addr_and_ports(new_engines=new_engines)
        for rank, _ in new_engines:
            self._engine_addr_and_ports[rank] = addr_and_ports[rank]

        init_handles = [
            engine.init.remote(**self._build_engine_init_kwargs(rank, addr_and_ports[rank]))
            for rank, engine in new_engines
        ]
        try:
            # Bounded: a bundle on a dead node never schedules, and an unbounded
            # ray.get would block the training step forever.
            ray.get(init_handles, timeout=_ENGINE_REBUILD_TIMEOUT_S)
        except Exception:
            for rank, engine in new_engines:
                try:
                    ray.kill(engine)
                except Exception:
                    pass
                self.all_engines[rank] = None
                self._remove_owned_pg(rank)
            raise

        return num_new_engines

    def _remove_owned_pg(self, rank: int) -> bool:
        """Remove the placement group this manager owns for ``rank``.

        Returns True when the slot had no owned PG or removal succeeded. On
        removal failure the handle is retained (put back into
        ``_engine_placements``) and the rank is recorded in
        ``_pending_pg_cleanup`` for reconcile retries, so the resource stays
        accounted-for instead of silently leaking.
        """
        # ``remove_placement_group`` only submits an asynchronous deletion.
        # Keep ownership until Ray reports REMOVED; treating an accepted RPC as
        # resource release lets a replacement actor race the old PG.
        placement = self._engine_placements.get(rank)
        if placement is None:
            return True
        pg_tuple, owns_pg = placement
        if not owns_pg:
            self._engine_placements.pop(rank, None)
            return True
        try:
            from ray.util.placement_group import placement_group_table, remove_placement_group

            if rank not in self._pending_pg_cleanup:
                remove_placement_group(pg_tuple[0])
                self._pending_pg_cleanup.add(rank)
            state = placement_group_table(pg_tuple[0]).get("state")
            if state == "REMOVED":
                self._engine_placements.pop(rank, None)
                self._pending_pg_cleanup.discard(rank)
                return True
            return False
        except Exception as exc:
            logger.warning(f"{self._log_prefix} remove placement group for rank={rank} failed: {exc}")
            # Retain the handle and mark the rank for reconcile retries. An
            # unknown table state is not evidence of resource release.
            self._pending_pg_cleanup.add(rank)
            return False

    def retry_pending_pg_cleanup(self) -> list[int]:
        """Retry PG removal for all failed-cleanup ranks.

        Returns the ranks still failing cleanup after this attempt.
        """
        still_failing: list[int] = []
        for rank in sorted(self._pending_pg_cleanup):
            if self._remove_owned_pg(rank):
                self._pending_pg_cleanup.discard(rank)
            else:
                still_failing.append(rank)
        return still_failing

    # ------------------------------------------------------------------
    # Health / lifecycle.
    # ------------------------------------------------------------------

    def health_check(self) -> bool:
        """Perform a health check on every engine."""
        health_results = []
        for engine in self.engines:
            if engine is not None:
                try:
                    health_results.append(ray.get(engine.health_generate.remote(), timeout=5.0))
                except Exception as e:
                    logger.warning(f"{self._log_prefix} engine health check failed: {e}")
                    health_results.append(False)
            else:
                health_results.append(False)
        return all(health_results)

    def onload(self, tags: Optional[list[str]] = None) -> None:
        """Load engine weights to GPU.

        Also the recovery point for engines that died since the last step: a
        freshly built engine comes up onloaded, which is exactly the state this
        phase wants.
        """
        rebuilt = self.recover()
        if self._onloaded and tags is None:
            logger.info(f"{self._log_prefix} engines already onloaded; skipping")
            return
        logger.info(f"{self._log_prefix} engines onload started with tags={tags}")
        # Engines rebuilt just above are already onloaded -- resuming them
        # again would be a double-resume, so only touch the ones that survived.
        dead = self._fanout("resume_memory_occupation", skip_ranks=rebuilt, tags=tags)
        if dead:
            # An engine that died while offloaded is only discovered here
            # (offload() short-circuits when already offloaded), so it missed
            # the recover() above. Rebuild now rather than leaving the pool a
            # man down for the whole next phase.
            self._retire_engines(dead)
            self.recover()
        self._onloaded = True
        logger.info(f"{self._log_prefix} engines onload completed")

    def offload(self) -> None:
        """Offload engine weights from GPU to free memory.

        Dead engines are retired here but NOT rebuilt: this typically runs
        while other ranks wait on a barrier, so keep it short and leave the
        rebuild to the next onload().
        """
        if not self._onloaded:
            logger.info(f"{self._log_prefix} engines already offloaded; skipping")
            return
        logger.info(f"{self._log_prefix} engines offload started")
        dead = self._fanout("release_memory_occupation")
        self._retire_engines(dead)
        # Unconditional: the surviving engines did release, so the manager
        # must not claim to still be onloaded just because one engine died.
        self._onloaded = False
        logger.info(f"{self._log_prefix} engines offload completed (retired {len(dead)} dead)")

    def _fanout(self, method: str, *, skip_ranks: Optional[set] = None, **kwargs) -> list[int]:
        """Call ``method`` on every live engine; return the ranks that are
        dead.

        Per-handle ray.get rather than one ray.get over the list: the batched
        form aborts on the first failure and loses which engine raised.
        """
        skip = skip_ranks or set()
        handles = {}
        for rank in range(0, len(self.all_engines), self.nodes_per_engine):
            engine = self.all_engines[rank]
            if engine is None or rank in skip:
                continue
            handles[rank] = getattr(engine, method).remote(**kwargs)

        dead = []
        for rank, handle in handles.items():
            try:
                ray.get(handle)
            except Exception as exc:
                if not _is_engine_dead(exc):
                    raise
                logger.warning(f"{self._log_prefix} engine rank={rank} died during {method}: {exc}")
                dead.append(rank)
        return dead

    def _retire_engines(self, ranks: list[int]) -> None:
        """Tear down dead engines and null their slots so recover() rebuilds
        them."""
        for rank in ranks:
            for i in range(rank, rank + self.nodes_per_engine):
                engine = self.all_engines[i]
                if engine is None:
                    continue
                try:
                    # shutdown() kill_process_tree's the SGLang server. Must run
                    # before ray.kill or the scheduler subprocesses are orphaned
                    # and keep holding GPU memory, so the rebuild can't fit.
                    ray.get(engine.shutdown.remote(), timeout=_ENGINE_SHUTDOWN_TIMEOUT_S)
                except Exception as exc:
                    logger.warning(f"{self._log_prefix} engine rank={i} shutdown failed (killing anyway): {exc}")
                try:
                    ray.kill(engine)
                except Exception as exc:
                    logger.warning(f"{self._log_prefix} engine rank={i} ray.kill failed: {exc}")
                self.all_engines[i] = None
                self._remove_owned_pg(i)
                logger.info(f"{self._log_prefix} engine rank={i} retired")

    def recover(self) -> set:
        """Rebuild engines whose slot is None. Returns the ranks rebuilt.

        ``_init_engines`` already skips non-None slots, so it rebuilds exactly
        the holes, reusing the same placement-group bundles (or creating fresh
        dedicated ones) and probing fresh ports (surviving engines' ports are
        bound, so they're skipped).
        """
        dead = [i for i, engine in enumerate(self.all_engines) if engine is None]
        if not dead:
            return set()

        logger.info(f"{self._log_prefix} recovering {len(dead)} engine(s): ranks={dead}")
        try:
            self._init_engines(dead)
        except Exception as exc:
            logger.exception(f"{self._log_prefix} engine rebuild failed for ranks={dead}: {exc}")

        rebuilt = {i for i in dead if self.all_engines[i] is not None}
        still_dead = [i for i in dead if i not in rebuilt]
        if still_dead:
            # Degrade rather than escalate: running on N-1 engines beats a
            # global restart. Only a total wipeout is unrecoverable here.
            if all(engine is None for engine in self.all_engines):
                raise RuntimeError(f"All engines are dead and could not be rebuilt (ranks={still_dead})")
            logger.error(
                f"{self._log_prefix} engines still dead after recovery, continuing degraded: ranks={still_dead}"
            )
        if rebuilt:
            logger.info(f"{self._log_prefix} recovered engine ranks={sorted(rebuilt)}")
        return rebuilt

    def is_onloaded(self) -> bool:
        return self._onloaded

    def shutdown(self) -> None:
        """Tear down every engine and remove any placement group this manager
        created for it."""
        for rank in range(0, len(self.all_engines), self.nodes_per_engine):
            engine = self.all_engines[rank]
            if engine is not None:
                try:
                    ray.get(engine.shutdown.remote(), timeout=_ENGINE_SHUTDOWN_TIMEOUT_S)
                except Exception as exc:
                    logger.warning(f"{self._log_prefix} engine rank={rank} shutdown failed (killing anyway): {exc}")
                try:
                    ray.kill(engine)
                except Exception as exc:
                    logger.warning(f"{self._log_prefix} engine rank={rank} ray.kill failed: {exc}")
                self.all_engines[rank] = None
            self._remove_owned_pg(rank)
        logger.info(f"{self._log_prefix} shutdown complete.")
