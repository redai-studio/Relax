# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Common lifecycle skeleton for managers that own a pool of SGLang engine
replicas (GenRM judges, OPD teachers, ...).

Concrete managers subclass ``MultiEngineManager`` (in addition to their own
``@ray.remote`` decorator) and implement the hooks below to plug in their
engine actor class, placement, GPU/port allocation, and env vars. The
mechanics -- parallel engine bring-up, health checking, dead-engine
detection/retirement, recovery, and onload/offload with idempotency tracking
-- live in ``EnginePool``, shared with rollout; this class turns the hooks
into the pool's spec and keeps the manager API.

Placement is resolved per engine (not once per manager): a manager may put
all engines on one shared placement group (e.g. GenRM colocated with
rollout), or give each engine its own dedicated placement group that it
creates and tears down itself (e.g. a non-colocated OPD teacher). Subclasses
express this via ``_resolve_placement``.
"""

from typing import Any, Optional

import ray
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from relax.distributed.ray.engine_pool import ENGINE_DEAD_EXCEPTIONS as _ENGINE_DEAD_EXCEPTIONS  # noqa: F401
from relax.distributed.ray.engine_pool import EnginePool, EnginePoolSpec
from relax.distributed.ray.engine_pool import is_engine_dead as _is_engine_dead  # noqa: F401
from relax.distributed.ray.placement_physical import validate_physical_placement
from relax.distributed.ray.placement_planner import plan_placement
from relax.engine.inference.discovery import TopologyRevision, build_model_snapshot, format_base_url
from relax.utils.logging_utils import get_logger


logger = get_logger(__name__)

# Rebuilding one engine is ~1.5 min (weight load + cuda graph capture). Bound it
# so a dead *node* -- whose placement-group bundle can never be filled -- degrades
# to "run with N-1 engines" instead of hanging the training step forever.
_ENGINE_REBUILD_TIMEOUT_S = 900.0
_ENGINE_SHUTDOWN_TIMEOUT_S = 60.0

# Concurrency group a manager actor answers ``get_inference_snapshot`` in, so
# discovery is not queued behind an offload drain or an engine rebuild that
# keeps the actor's default group busy for minutes. An actor class opts in by
# declaring the group and tagging its override of the method:
#
#     @ray.remote(concurrency_groups=SNAPSHOT_CONCURRENCY_GROUPS)
#     class MyManager(MultiEngineManager):
#         @ray.method(concurrency_group=SNAPSHOT_CONCURRENCY_GROUP)
#         def get_inference_snapshot(self) -> dict:
#             return super().get_inference_snapshot()
#
# The base method stays untagged: Ray rejects a tagged method on an actor that
# does not declare the group.
SNAPSHOT_CONCURRENCY_GROUP = "inference_snapshot"
SNAPSHOT_CONCURRENCY_GROUPS = {SNAPSHOT_CONCURRENCY_GROUP: 1}


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
        # The hooks are looked up at call time, so subclasses only override them.
        self._pool = EnginePool(
            EnginePoolSpec(
                actor_class=lambda: ray.remote(self.engine_actor_cls),
                resolve_placement=lambda rank: self._resolve_placement(rank),
                actor_options=self._engine_actor_options,
                ctor=self._engine_ctor,
                init_kwargs=lambda rank, addr_and_ports: self._build_engine_init_kwargs(rank, addr_and_ports),
                allocate_addresses=lambda new_engines: self._allocate_engine_addr_and_ports(new_engines=new_engines),
                teardown_timeout_s=_ENGINE_SHUTDOWN_TIMEOUT_S,
                start_timeout_s=_ENGINE_REBUILD_TIMEOUT_S,
                log_prefix=log_prefix,
            ),
            self.all_engines,
            nodes_per_engine=self.nodes_per_engine,
        )
        # Shared with the pool: the address each slot's engine was given. Kept
        # across a rebuild so a subclass can hand the same endpoint out again.
        self._engine_addr_and_ports: dict[int, dict] = self._pool.addresses
        self._topology_revision = TopologyRevision()

        if not skip_init:
            # Before any engine exists: an engine on the wrong GPUs must not start.
            self._validate_physical_placement()
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

    def _physical_placement(self) -> Optional[tuple[str, tuple]]:
        """Return ``(pool name, pg_tuple)`` of the placement group this
        manager's engines are planned on, so the plan can be checked against
        where the bundles really are before any engine starts.

        ``None`` (the default) skips the check, e.g. for a manager whose
        placement groups do not exist yet.
        """
        return None

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

    def _validate_physical_placement(self) -> None:
        scope = self._physical_placement()
        if scope is None:
            return
        pool, pg_tuple = scope
        validate_physical_placement(
            # The logical layout was validated when the run started.
            plan_placement(self.args, validate=False),
            pool,
            pg_tuple,
            num_gpus_per_node=getattr(self.args, "num_gpus_per_node", None),
        )

    def _engine_actor_options(self, rank: int, pg: Any, bundle_index: int) -> dict:
        return {
            **self._ray_resource_kwargs(rank),
            "scheduling_strategy": PlacementGroupSchedulingStrategy(
                placement_group=pg,
                placement_group_capture_child_tasks=True,
                placement_group_bundle_index=bundle_index,
            ),
            "runtime_env": {"env_vars": self._build_engine_env_vars()},
        }

    def _engine_ctor(self, rank: int, base_gpu_id: int) -> tuple[tuple, dict]:
        return (self._engine_ctor_args(rank),), {
            "rank": rank,
            "worker_type": "regular",
            "base_gpu_id": base_gpu_id,
            **self._build_engine_ctor_kwargs(rank),
        }

    def _init_engines(self, ranks: list[int]) -> int:
        """Create actors for the given slot ranks, fire init.remote() for all
        of them without blocking, then await everything in one ray.get so a
        large engine doesn't pay N x cold-load latency.

        On failure, kill any newly created engines and leave their slots None
        so the caller sees them as still-dead rather than silently healthy.
        """
        return len(self._pool.start(ranks))

    # ------------------------------------------------------------------
    # Health / lifecycle.
    # ------------------------------------------------------------------

    def health_check(self) -> bool:
        """Perform a health check on every engine."""
        heads = self._pool.head_slots()
        if any(self.all_engines[rank] is None for rank in heads):
            return False
        return not self._pool.failed_health_checks(heads, get_timeout=5.0)

    def onload(self, tags: Optional[list[str]] = None) -> None:
        """Load engine weights to GPU.

        Also the recovery point for engines that died since the last step: a
        freshly built engine comes up onloaded, which is exactly the state this
        phase wants.
        """
        self._pool.activate(tags)

    def offload(self) -> None:
        """Offload engine weights from GPU to free memory.

        Dead engines are retired here but NOT rebuilt: this typically runs
        while other ranks wait on a barrier, so keep it short and leave the
        rebuild to the next onload().
        """
        self._pool.deactivate()

    def _fanout(self, method: str, *, skip_ranks: Optional[set] = None, **kwargs) -> list[int]:
        """Call ``method`` on every live engine; return the ranks that are
        dead."""
        return self._pool.call_all(method, skip=skip_ranks, **kwargs)

    def _retire_engines(self, ranks: list[int]) -> None:
        """Tear down dead engines and null their slots so recover() rebuilds
        them."""
        self._pool.retire(ranks)

    def recover(self) -> set:
        """Rebuild engines whose slot is None. Returns the ranks rebuilt.

        Only the holes are rebuilt, reusing the same placement-group bundles
        (or creating fresh dedicated ones) and probing fresh ports (surviving
        engines' ports are bound, so they're skipped).
        """
        return self._pool.recover()

    def is_onloaded(self) -> bool:
        return self._pool.is_active()

    def get_inference_snapshot(self) -> dict:
        """Describe this manager's logical replicas for discovery.

        Only head slots are listed: followers of a multi-node engine serve no
        HTTP. Replicas are reported by index; the owner that knows which model
        this manager serves names them (``model_snapshot_from_payload``).

        Only reads pool state, one slot at a time, so it may run while another
        thread of the actor is switching or rebuilding engines.
        """
        rows = []
        incarnations = {}
        for index, rank in enumerate(range(0, len(self.all_engines), self.nodes_per_engine)):
            address = self._engine_addr_and_ports.get(rank) or {}
            state = self._pool.state(rank)
            base_url = None
            if self.all_engines[rank] is not None and "host" in address:
                base_url = format_base_url(address["host"], address["port"])
            rows.append((index, base_url, state))
            incarnations[f"_/{index}"] = self._pool.incarnations.get(rank, 0)

        model = build_model_snapshot("_", rows)
        revision = self._topology_revision.observe([model], incarnations)
        return {
            "topology_revision": revision,
            "state": model.state.value,
            "router_url": None,
            "engines": [
                {"index": index, "base_url": base_url, "state": state.value} for index, base_url, state in rows
            ],
        }

    def shutdown(self) -> None:
        """Tear down every engine and remove any placement group this manager
        created for it."""
        self._pool.shutdown()
