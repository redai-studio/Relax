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

from relax.inference.manager import InferenceManager
from relax.inference.placement import validate_pg_span
from relax.inference.routing import engine_record
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


def _is_actor_dead(exc: BaseException) -> bool:
    """Return whether Ray has already lost the engine actor itself.

    A transport timeout is ambiguous: the SGLang process may still own GPU
    memory and must stay pending until ``shutdown`` succeeds.  RayActorError is
    different.  It means the actor process is gone, so a cleanup attempt can
    acknowledge that slot and let recovery rebuild it.
    """

    return isinstance(exc, ray.exceptions.RayActorError)


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
        role: str = "genrm",
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
        self._initializing_ranks: set[int] = set()
        self._cleanup_pending: set[int] = set()
        # Track memory-occupation state so repeated onload/offload calls become
        # safe no-ops. Engines start onloaded; callers may immediately offload.
        self._onloaded = True
        self.inference = InferenceManager(role, {"default": self})

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
        self._require_cleanup_complete()
        EngineActor = ray.remote(self.engine_actor_cls)
        new_engines: list[tuple[int, Any]] = []
        pending_ranks = [rank for rank in ranks if self.all_engines[rank] is None]
        self._initializing_ranks.update(pending_ranks)
        try:
            resolved = {}
            for rank in pending_ranks:
                resolved[rank] = self._resolve_placement(rank)
                pg_tuple, owns_pg, gpu_index = resolved[rank]
                self._engine_placements[rank] = (pg_tuple, owns_pg)
                if getattr(self.args, "_inference_placement", None):
                    validate_pg_span(pg_tuple, gpu_index, self.num_gpu_per_engine)
            for rank in pending_ranks:
                pg_tuple, owns_pg, gpu_index = resolved[rank]
                pg, reordered_bundle_indices, reordered_gpu_ids = pg_tuple
                self._engine_placements[rank] = (pg_tuple, owns_pg)
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
            ray.get(init_handles, timeout=_ENGINE_REBUILD_TIMEOUT_S)
            return num_new_engines
        except BaseException:
            self._cleanup_slots(pending_ranks)
            raise
        finally:
            self._initializing_ranks.difference_update(pending_ranks)

    def _remove_owned_pg(self, rank: int) -> None:
        placement = self._engine_placements.get(rank)
        if placement is None:
            return
        pg_tuple, owns_pg = placement
        if not owns_pg:
            self._engine_placements.pop(rank)
            return
        if any(i != rank and other[0][0] == pg_tuple[0] for i, other in self._engine_placements.items()):
            self._engine_placements.pop(rank)
            return
        from ray.util.placement_group import remove_placement_group

        remove_placement_group(pg_tuple[0])
        self._engine_placements.pop(rank)

    def _require_cleanup_complete(self) -> None:
        if self._cleanup_pending:
            raise RuntimeError(f"Inference cleanup pending for ranks={sorted(self._cleanup_pending)}")

    def _cleanup_slots(self, ranks: list[int]) -> None:
        ranks = sorted(set(ranks))
        self._cleanup_pending.update(ranks)
        failures = []
        for rank in ranks:
            try:
                engine = self.all_engines[rank]
                if engine is not None:
                    # Killing only the Ray actor cannot confirm its SGLang
                    # children released GPU memory. Keep the handle until
                    # shutdown acknowledges cleanup, including on retry.
                    ray.get(engine.shutdown.remote(), timeout=_ENGINE_SHUTDOWN_TIMEOUT_S)
                    ray.kill(engine)
                    self.all_engines[rank] = None
                self._remove_owned_pg(rank)
                self._cleanup_pending.discard(rank)
            except Exception as exc:
                if engine is not None and _is_actor_dead(exc):
                    # The Ray actor has already exited, so its SGLang child
                    # cannot retain the GPU lease.  ``ray.kill`` is kept as a
                    # best-effort acknowledgement because providers differ
                    # on whether killing an already-dead actor raises.
                    try:
                        ray.kill(engine)
                    except Exception as kill_exc:
                        if not _is_actor_dead(kill_exc):
                            failures.append(kill_exc)
                            logger.warning(
                                f"{self._log_prefix} actor death cleanup remains pending for rank={rank}: {kill_exc}"
                            )
                            continue
                    try:
                        self._remove_owned_pg(rank)
                    except Exception as pg_exc:
                        failures.append(pg_exc)
                        logger.warning(
                            f"{self._log_prefix} actor death PG cleanup remains pending for rank={rank}: {pg_exc}"
                        )
                        continue
                    self.all_engines[rank] = None
                    self._cleanup_pending.discard(rank)
                    logger.warning(
                        f"{self._log_prefix} acknowledged dead Ray actor rank={rank}; slot is eligible for recovery"
                    )
                    continue
                failures.append(exc)
                logger.warning(f"{self._log_prefix} cleanup remains pending for rank={rank}: {exc}")
        if failures:
            self.inference.state = "failed"
            raise RuntimeError(
                f"Inference cleanup unconfirmed for ranks={sorted(self._cleanup_pending)}"
            ) from failures[0]

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
        self._require_cleanup_complete()
        if self.inference.state != "dead" and any(engine is None for engine in self.all_engines):
            self.inference.state = "failed"
        self.inference.transition("activate", lambda: self._onload_engines(tags), tags)

    def _onload_engines(self, tags: Optional[list[str]] = None) -> None:
        """Load engine weights to GPU.

        Also the recovery point for engines that died since the last step: a
        freshly built engine comes up onloaded, which is exactly the state this
        phase wants.
        """
        rebuilt = self.recover()
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
        self.inference.transition("deactivate", self._offload_engines)

    def _offload_engines(self) -> None:
        """Offload engine weights from GPU to free memory.

        Dead engines are retired here but NOT rebuilt: this typically runs
        while other ranks wait on a barrier, so keep it short and leave the
        rebuild to the next onload().
        """
        logger.info(f"{self._log_prefix} engines offload started")
        # A previous attempt may have removed the head but retained a follower.
        # Retrying only the head RPCs would incorrectly publish SLEEPING.
        self._cleanup_slots(list(self._cleanup_pending))
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
        failures = []
        for rank, handle in handles.items():
            try:
                ray.get(handle, timeout=_ENGINE_REBUILD_TIMEOUT_S)
            except Exception as exc:
                if not _is_engine_dead(exc):
                    failures.append(exc)
                    continue
                logger.warning(f"{self._log_prefix} engine rank={rank} died during {method}: {exc}")
                dead.append(rank)
        if failures:
            self._retire_engines(dead)
            raise failures[0]
        return dead

    def _retire_engines(self, ranks: list[int]) -> None:
        """Tear down dead engines and null their slots so recover() rebuilds
        them."""
        self._cleanup_slots([i for rank in ranks for i in range(rank, rank + self.nodes_per_engine)])

    def recover(self) -> set:
        """Rebuild engines whose slot is None. Returns the ranks rebuilt.

        ``_init_engines`` already skips non-None slots, so it rebuilds exactly
        the holes, reusing the same placement-group bundles (or creating fresh
        dedicated ones) and probing fresh ports (surviving engines' ports are
        bound, so they're skipped).
        """
        self._require_cleanup_complete()
        dead = [i for i, engine in enumerate(self.all_engines) if engine is None]
        if not dead:
            return set()

        logger.info(f"{self._log_prefix} recovering {len(dead)} engine(s): ranks={dead}")
        try:
            self._init_engines(dead)
        except Exception as exc:
            logger.exception(f"{self._log_prefix} engine rebuild failed for ranks={dead}: {exc}")
            self._require_cleanup_complete()
            # A partial pool must never be published as READY. The caller can
            # retry recovery after the placement or node failure is resolved,
            # but routing and training must observe the failed state first.
            self.inference.state = "failed"
            raise

        rebuilt = {i for i in dead if self.all_engines[i] is not None}
        still_dead = [i for i in dead if i not in rebuilt]
        if still_dead:
            self.inference.state = "failed"
            raise RuntimeError(f"Engines are still dead after recovery (ranks={still_dead})")
        if rebuilt:
            logger.info(f"{self._log_prefix} recovered engine ranks={sorted(rebuilt)}")
        return rebuilt

    def is_onloaded(self) -> bool:
        return self._onloaded

    def shutdown(self) -> None:
        self.inference.transition("shutdown", self._shutdown_engines)

    def _shutdown_engines(self) -> None:
        """Tear down every engine and remove any placement group this manager
        created for it."""
        self._cleanup_slots(list(range(len(self.all_engines))))
        logger.info(f"{self._log_prefix} shutdown complete.")

    def activate(self, tags: Optional[list[str]] = None) -> None:
        self.onload(tags)

    def deactivate(self) -> None:
        self.offload()

    def drain(self) -> None:
        def pause() -> None:
            dead = self._fanout("drain")
            self._retire_engines(dead)

        self.inference.transition("drain", pause)

    @ray.method(concurrency_group="discovery")
    def get_inference_snapshot(self) -> dict:
        engines = []
        for rank in range(0, len(self.all_engines), self.nodes_per_engine):
            live = all(
                self.all_engines[i] is not None and i not in self._initializing_ranks
                for i in range(rank, rank + self.nodes_per_engine)
            )
            address = self._engine_addr_and_ports.get(rank, {})
            host = address.get("host")
            if host and ":" in host and not host.startswith("["):
                host = f"[{host}]"
            url = f"http://{host}:{address['port']}" if host and "port" in address else None
            engines.append(engine_record(f"default/{rank // self.nodes_per_engine}", url, "ready" if live else "dead"))
        state = "ready" if any(e["direct_eligible"] for e in engines) else "dead"
        aliases = []
        for attr in ("genrm_model_path", "teacher_hf_checkpoint", "sglang_hf_checkpoint", "hf_checkpoint"):
            value = getattr(self.args, attr, None)
            if isinstance(value, str) and value and value not in aliases:
                aliases.append(value)
        return self.inference.snapshot(
            {"default": {"state": state, "router_url": None, "engines": engines, "model_aliases": aliases}},
            default_model="default",
        )
