# Copyright (c) 2026 Relax Authors. All Rights Reserved.


import threading
import time
from typing import Any, Optional

import ray
import requests
from ray.util.placement_group import remove_placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from relax.inference.registry import InferenceRegistry
from relax.inference.specs import RoutingSpec
from relax.utils.logging_utils import get_logger


logger = get_logger(__name__)

_ENGINE_DEAD_EXCEPTIONS = (
    ConnectionError,
    TimeoutError,
    requests.exceptions.ConnectionError,
    requests.exceptions.Timeout,
    ray.exceptions.RayActorError,
)

_ENGINE_REBUILD_TIMEOUT_S = 900.0
_ENGINE_SHUTDOWN_TIMEOUT_S = 60.0
_ENGINE_OPERATION_TIMEOUT_S = 180.0
_DISCOVERY_TIMEOUT_S = 5.0
_RESIDENT_TAGS = frozenset({"weights", "kv_cache", "cuda_graph"})


class _InferenceObservation:
    def __init__(self, role: str) -> None:

        self.lock = threading.RLock()
        self.registry = InferenceRegistry(role)
        self.entries: dict[str, dict[str, Any]] = {}

    def observe(self, key: str, workers: list[Any]) -> dict[str, Any]:
        identity = tuple(id(worker) if worker is not None else None for worker in workers)
        previous = self.entries.get(key)
        if previous is None or previous["identity"] != identity:
            self.registry.invalidate()
            self.entries[key] = {
                "identity": identity,
                "workers": tuple(workers),
                "generation": 1 if previous is None else previous["generation"] + 1,
                "state": "STARTING",
                "tags": set(),
                "weights_ready": False,
                "admission": False,
            }
        return self.entries[key]

    def initialized(self, key: str, workers: list[Any], *, weights_ready: bool) -> None:
        with self.lock:
            entry = self.observe(key, workers)
            entry.update(tags=set(_RESIDENT_TAGS), weights_ready=weights_ready, admission=True)
            self._refresh_state(entry)
            self.registry.invalidate()

    @staticmethod
    def _refresh_state(entry: dict[str, Any]) -> None:
        entry["state"] = (
            "READY"
            if _RESIDENT_TAGS.issubset(entry["tags"]) and entry["weights_ready"] and entry["admission"]
            else "ONLOADING"
        )

    def begin(self, replicas: dict[str, list[Any]], state: str) -> dict[str, dict[str, Any]]:
        with self.lock:
            entries = {key: self.observe(key, workers) for key, workers in replicas.items()}
            for entry in entries.values():
                entry["state"] = state
                entry["admission"] = False
            if entries:
                self.registry.invalidate()
            return entries

    def failed(self, entries: dict[str, dict[str, Any]]) -> None:
        with self.lock:
            for entry in entries.values():
                entry.update(state="FAILED", admission=False)

    def offloaded(self, entries: dict[str, dict[str, Any]], *, preserve_weights: bool) -> None:
        with self.lock:
            for entry in entries.values():
                entry.update(
                    state="SLEEPING",
                    tags=set(),
                    admission=False,
                    weights_ready=entry["weights_ready"] and preserve_weights,
                )

    def onloaded(self, entries: dict[str, dict[str, Any]], tags: Optional[list[str]]) -> None:
        with self.lock:
            for entry in entries.values():
                entry["tags"].update(_RESIDENT_TAGS if tags is None else tags)
                if tags is None:
                    entry["admission"] = True
                self._refresh_state(entry)

    def replica(
        self, key: str, workers: list[Any], *, direct: bool, expected_workers: int, process_probe: bool = True
    ) -> dict[str, Any]:
        entry = self.observe(key, workers)
        state = entry["state"]
        base_url = None
        if len(workers) != expected_workers or any(worker is None for worker in workers):
            state = "FAILED"
        elif state == "READY":
            try:
                refs = [worker.health_process.remote() for worker in workers] if process_probe else []
                refs.extend([workers[0].health_generate.remote(), workers[0].get_url.remote()])
                results = ray.get(refs, timeout=_DISCOVERY_TIMEOUT_S)
                if any(result is not True for result in results[:-1]):
                    state = "FAILED" if False in results[:-1] else "STARTING"
                else:
                    base_url = results[-1]
                    if not base_url:
                        state = "STARTING"
                    elif entry.get("endpoint") != base_url:
                        if entry.get("endpoint") is not None:
                            entry["generation"] += 1
                        entry["endpoint"] = base_url
            except Exception:
                state = "FAILED"
        return {
            "engine_id": key,
            "generation": entry["generation"],
            "base_url": base_url,
            "state": state,
            "direct_eligible": direct and state == "READY",
        }


def _model_discovery_state(replicas: list[dict[str, Any]]) -> str:
    states = {replica["state"] for replica in replicas}
    if "READY" in states:
        return "READY"
    if len(states) == 1:
        return next(iter(states))
    return "STARTING"


def _is_engine_dead(exc: BaseException) -> bool:
    return isinstance(exc, _ENGINE_DEAD_EXCEPTIONS)


def _is_actor_confirmed_dead(exc: BaseException) -> bool:
    return isinstance(exc, ray.exceptions.RayActorError) and not isinstance(exc, ray.exceptions.ActorUnavailableError)


class InferenceCleanupError(RuntimeError):
    pass


class InferenceManager:
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
        if nodes_per_engine < 1 or num_slots < 0 or num_slots % nodes_per_engine:
            raise ValueError("Engine slots must form complete logical replicas")
        self.nodes_per_engine = nodes_per_engine
        self._log_prefix = log_prefix

        self.all_engines: list[Any] = [None] * num_slots
        self._engine_placements: dict[int, tuple] = {}
        self._borrowed_pgs: list[Any] = []
        self._engine_addr_and_ports: dict[int, dict] = {}
        self._lifecycle_lock = threading.RLock()
        self._cleanup_pending: set[int] = set()
        self._shutdown_confirmed: set[int] = set()
        self._onloaded = True
        self._inference_observation = _InferenceObservation(getattr(self, "_inference_role", "genrm"))

        if not skip_init:
            self._init_engines(list(range(num_slots)))

    @classmethod
    def for_rollout(cls, args: Any = None, servers: dict | None = None) -> "InferenceManager":
        owner = cls(args, num_slots=0, engine_actor_cls=object, skip_init=True, log_prefix="Rollout")
        owner._inference_role = "rollout"
        owner._inference_observation = _InferenceObservation("rollout")
        owner.servers = servers if servers is not None else {}
        owner.status = None
        owner.health_monitors = []
        owner._rollout_initialized = servers is not None
        return owner

    def initialize_rollout(self, start_factory: Any, pg: Any) -> dict:
        with self._lifecycle_lock:
            if self._rollout_initialized:
                return self.servers
            try:
                self.servers = start_factory(self.args, pg)
            except InferenceCleanupError as exc:
                self.servers = getattr(exc, "rollout_servers", {})
                self.status = "failed"
                self._failed_startup_owner = getattr(exc, "cleanup_owner", None)
                raise
            self._rollout_initialized = True
            for server in self.servers.values():
                for group in server.engine_groups:
                    group.pg_owned = False
            for key, workers in self.rollout_replicas().items():
                if workers and all(worker is not None for worker in workers):
                    self._inference_observation.initialized(key, workers, weights_ready=False)
                    self._inference_observation.entries[key]["initial_weight_probe"] = True
            return self.servers

    def rollout_replicas(self, model_name: str | None = None) -> dict[str, list[Any]]:
        replicas = {}
        for name, server in self.servers.items():
            if model_name is not None and name != model_name:
                continue
            for group in self._active_rollout_groups(server):
                for head in range(0, len(group.all_engines), group.nodes_per_engine):
                    key = f"{name}/group-{group.rank_offset}/replica-{head // group.nodes_per_engine}"
                    replicas[key] = group.all_engines[head : head + group.nodes_per_engine]
        return replicas

    @staticmethod
    def _active_rollout_groups(server: Any) -> list[Any]:
        groups = []
        for group in server.engine_groups:
            if not getattr(group, "is_scaled_out", False):
                groups.append(group)
                continue
            status = getattr(group, "lifecycle_status", "ACTIVE")
            if getattr(status, "value", status) == "ACTIVE":
                groups.append(group)
        return groups

    def get_rollout_engines_and_lock(self, model_name: str | None, lock: Any) -> tuple:
        server = self.servers.get(model_name) if model_name is not None else next(iter(self.servers.values()), None)
        if server is None:
            return [], lock, 0, [], []
        groups = self._active_rollout_groups(server)
        engines = [engine for group in groups for engine in group.engines]
        counts = [group.num_gpus_per_engine for group in groups for _engine in group.engines]
        offsets = [
            group.gpu_offset + index * group.num_gpus_per_engine
            for group in groups
            for index, _engine in enumerate(group.engines)
        ]
        return engines, lock, sum(group.num_new_engines for group in groups), counts, offsets

    def adopt_group(self, model_id: str, group: Any, *, owned_pg: bool) -> None:
        with self._lifecycle_lock:
            server = self.servers[model_id]
            if group in server.engine_groups:
                if getattr(group, "pg_owned", False) != owned_pg:
                    raise ValueError("Engine group ownership cannot change after publication")
                return
            group.pg_owned = owned_pg
            server.engine_groups.append(group)
            self._inference_observation.registry.invalidate()

    def remove_group(self, model_id: str, group: Any) -> None:
        with self._lifecycle_lock:
            server = self.servers[model_id]
            if group not in server.engine_groups:
                return
            if any(engine is not None for engine in group.all_engines):
                raise InferenceCleanupError("Cannot remove a group with live or unconfirmed workers")
            if getattr(group, "pg_owned", False) and group.pg is not None:
                pg = group.pg[0]
                other_live = any(
                    other is not group and other.pg is not None and other.pg[0] == pg
                    for pool in self.servers.values()
                    for other in pool.engine_groups
                )
                if not other_live:
                    remove_placement_group(pg)
            server.engine_groups.remove(group)
            self._inference_observation.registry.invalidate()

    def _rollout_memory(self, method: str, tags: Optional[list[str]] = None) -> None:
        with self._lifecycle_lock:
            observation = self._inference_observation
            requested = set(_RESIDENT_TAGS if tags is None else tags)
            if not requested.issubset(_RESIDENT_TAGS):
                raise ValueError("Unknown inference memory tags")
            errors = []
            restored = []
            for key, workers in self.rollout_replicas().items():
                if not workers or all(worker is None for worker in workers):
                    continue
                if any(worker is None for worker in workers):
                    errors.append(
                        InferenceCleanupError("Incomplete Rollout replica requires confirmed cleanup before reuse")
                    )
                    continue
                entry = observation.observe(key, workers)
                if method == "release_memory_occupation" and entry["state"] == "SLEEPING":
                    continue
                missing = requested - entry["tags"]
                if method == "resume_memory_occupation" and entry["state"] == "FAILED":
                    errors.append(
                        InferenceCleanupError(
                            "Failed Rollout memory operation requires confirmed offload before restore"
                        )
                    )
                    continue
                if method == "resume_memory_occupation" and not missing:
                    continue
                observed = observation.begin(
                    {key: workers}, "DRAINING" if method == "release_memory_occupation" else "ONLOADING"
                )
                try:
                    kwargs = (
                        {}
                        if method == "release_memory_occupation"
                        else {"tags": None if missing == _RESIDENT_TAGS else sorted(missing)}
                    )
                    ray.get(getattr(workers[0], method).remote(**kwargs), timeout=_ENGINE_OPERATION_TIMEOUT_S)
                    if method == "release_memory_occupation":
                        observation.offloaded(
                            observed, preserve_weights=getattr(self.args, "_inference_preserve_rollout_weights", False)
                        )
                        entry["initial_weight_probe"] = False
                    else:
                        observation.onloaded(observed, kwargs["tags"])
                        restored.append((key, workers, entry))
                        if entry["weights_ready"] and _RESIDENT_TAGS.issubset(entry["tags"]):
                            ray.get(workers[0].continue_generation.remote(), timeout=_ENGINE_OPERATION_TIMEOUT_S)
                            entry["admission"] = True
                            observation._refresh_state(entry)
                except Exception as exc:
                    observation.failed(observed)
                    errors.append(exc)
            if errors:
                if method == "resume_memory_occupation":
                    for key, workers, entry in restored:
                        observed = observation.begin({key: workers}, "DRAINING")
                        try:
                            ray.get(workers[0].release_memory_occupation.remote(), timeout=_ENGINE_OPERATION_TIMEOUT_S)
                            observation.offloaded(
                                observed,
                                preserve_weights=getattr(self.args, "_inference_preserve_rollout_weights", False),
                            )
                        except Exception:
                            observation.failed(observed)
                self.status = "failed"
                raise errors[0]
            if method == "release_memory_occupation":
                self.status = "offload"
            elif tags is None:
                self.status = "onload"

    def recover_rollout(self, model_name: str | None = None) -> None:
        with self._lifecycle_lock:
            server = (
                self.servers.get(model_name) if model_name is not None else next(iter(self.servers.values()), None)
            )
            if server is None:
                return
            observation = self._inference_observation
            before = observation.begin(self.rollout_replicas(server.model_name), "ONLOADING")
            try:
                server.recover()
            except Exception:
                observation.failed(before)
                raise
            for key, workers in self.rollout_replicas(server.model_name).items():
                entry = observation.observe(key, workers)
                if key not in before or entry is not before[key]:
                    entry["tags"] = {"weights"} if self.args.offload_rollout else set(_RESIDENT_TAGS)
                entry.update(weights_ready=False, admission=False, initial_weight_probe=False)
                observation._refresh_state(entry)

    def shutdown_rollout(self, timeout: float = _ENGINE_SHUTDOWN_TIMEOUT_S) -> None:
        with self._lifecycle_lock:
            errors, owned_pgs = [], []
            seen = set()
            refs = []
            for server in self.servers.values():
                for group in server.engine_groups:
                    if getattr(group, "pg_owned", False) and group.pg is not None and group.pg[0] not in owned_pgs:
                        owned_pgs.append(group.pg[0])
                    for index, engine in enumerate(group.all_engines):
                        if engine is None or id(engine) in seen:
                            continue
                        seen.add(id(engine))
                        try:
                            refs.append((group, index, engine, engine.shutdown.remote()))
                        except Exception as exc:
                            if _is_actor_confirmed_dead(exc):
                                refs.append((group, index, engine, None))
                                continue
                            errors.append(exc)
            for group, index, engine, ref in refs:
                try:
                    if ref is not None:
                        try:
                            ray.get(ref, timeout=timeout)
                        except Exception as exc:
                            if not _is_actor_confirmed_dead(exc):
                                raise
                    ray.kill(engine)
                    group.all_engines[index] = None
                except Exception as exc:
                    errors.append(exc)
            if errors:
                self.status = "failed"
                raise InferenceCleanupError("Rollout shutdown remains unconfirmed") from errors[0]

            for pg in owned_pgs:
                remove_placement_group(pg)
            self._inference_observation.registry.invalidate()
            self.servers.clear()
            self.status = "offload"

    @property
    def engines(self) -> list[Any]:
        return self.all_engines[:: self.nodes_per_engine]

    def _inference_replicas(self) -> dict[str, list[Any]]:
        model_id = getattr(self, "_inference_model_id", "__default__")
        return {
            f"{model_id}/replica-{rank // self.nodes_per_engine}": self.all_engines[
                rank : rank + self.nodes_per_engine
            ]
            for rank in range(0, len(self.all_engines), self.nodes_per_engine)
        }

    def get_inference_snapshot(self) -> dict[str, Any]:

        observation = self._inference_observation
        model_id = getattr(self, "_inference_model_id", "__default__")
        with observation.lock:
            replicas = [
                observation.replica(key, workers, direct=True, expected_workers=self.nodes_per_engine)
                for key, workers in self._inference_replicas().items()
            ]
            model = {
                "model_id": model_id,
                "state": _model_discovery_state(replicas),
                "weight_source": "STATIC",
                "weight_version": None,
                "served_model_name": getattr(self, "_inference_served_model_name", None),
                "route_mode": "DIRECT",
                "router_url": None,
                "engines": replicas,
                "capabilities": ["generate", "chat"],
            }
            return observation.registry.publish({model_id: model}, routing=RoutingSpec(default_model=model_id))

    def _resolve_placement(self, rank: int) -> tuple[tuple, bool, int]:
        raise NotImplementedError

    def _ray_resource_kwargs(self, rank: int) -> dict:
        raise NotImplementedError

    def _allocate_engine_addr_and_ports(self, *, new_engines: list[tuple]) -> dict[int, dict]:
        raise NotImplementedError

    def _build_engine_env_vars(self) -> dict[str, str]:
        raise NotImplementedError

    def _engine_ctor_args(self, rank: int) -> Any:
        return self.args

    def _build_engine_ctor_kwargs(self, rank: int) -> dict:
        return {}

    def _build_engine_init_kwargs(self, rank: int, addr_and_ports: dict) -> dict:
        return dict(addr_and_ports)

    def _init_engines(self, ranks: list[int]) -> int:
        with self._lifecycle_lock:
            if self._cleanup_pending:
                raise InferenceCleanupError("Cannot initialize engines while prior cleanup is unconfirmed")
            EngineActor = ray.remote(self.engine_actor_cls)
            new_engines: list[tuple[int, Any]] = []
            allocated_ranks: list[int] = []
            try:
                for rank in ranks:
                    if self.all_engines[rank] is not None:
                        continue
                    pg_tuple, owns_pg, gpu_index = self._resolve_placement(rank)
                    self._engine_placements[rank] = (pg_tuple, owns_pg)
                    if not owns_pg and pg_tuple[0] not in self._borrowed_pgs:
                        self._borrowed_pgs.append(pg_tuple[0])
                    allocated_ranks.append(rank)
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
                if not new_engines:
                    return 0
                addr_and_ports = self._allocate_engine_addr_and_ports(new_engines=new_engines)
                for rank, _ in new_engines:
                    self._engine_addr_and_ports[rank] = addr_and_ports[rank]
                init_handles = [
                    engine.init.remote(**self._build_engine_init_kwargs(rank, addr_and_ports[rank]))
                    for rank, engine in new_engines
                ]
                ray.get(init_handles, timeout=_ENGINE_REBUILD_TIMEOUT_S)
            except Exception as startup_error:
                self._onloaded = False
                try:
                    self._cleanup_slots(allocated_ranks)
                except InferenceCleanupError as cleanup_error:
                    raise InferenceCleanupError(f"Engine startup failed; {cleanup_error}") from startup_error
                raise
            new_ranks = {rank for rank, _engine in new_engines}
            for index, (key, workers) in enumerate(self._inference_replicas().items()):
                replica_ranks = set(range(index * self.nodes_per_engine, (index + 1) * self.nodes_per_engine))
                if replica_ranks.issubset(new_ranks):
                    self._inference_observation.initialized(key, workers, weights_ready=True)
            self._update_onloaded_state()
            return len(new_engines)

    def _remove_owned_pg(self, rank: int) -> None:
        placement = self._engine_placements.get(rank)
        if placement is None:
            return
        pg_tuple, owns_pg = placement
        if not owns_pg or pg_tuple[0] in self._borrowed_pgs:
            self._engine_placements.pop(rank, None)
            return
        related = [slot for slot, (other, _owned) in self._engine_placements.items() if other[0] == pg_tuple[0]]
        if any(self.all_engines[slot] is not None for slot in related):
            return
        if any(not self._engine_placements[slot][1] for slot in related):
            raise InferenceCleanupError("Placement group has conflicting owned and borrowed records")

        remove_placement_group(pg_tuple[0])
        for slot in related:
            self._engine_placements.pop(slot, None)

    def _update_onloaded_state(self) -> None:
        observation = self._inference_observation
        with observation.lock:
            entries = [observation.observe(key, workers) for key, workers in self._inference_replicas().items()]
            self._onloaded = (
                bool(entries) and not self._cleanup_pending and all(entry["state"] == "READY" for entry in entries)
            )

    def _cleanup_slots(self, ranks: list[int]) -> None:
        ranks = sorted(set(ranks))
        affected = {
            key: workers
            for index, (key, workers) in enumerate(self._inference_replicas().items())
            if any(index * self.nodes_per_engine <= rank < (index + 1) * self.nodes_per_engine for rank in ranks)
        }
        observed = self._inference_observation.begin(affected, "STOPPING")
        self._cleanup_pending.update(ranks)
        errors = []
        shutdown_refs = {}
        deadline = time.monotonic() + _ENGINE_SHUTDOWN_TIMEOUT_S
        for rank in ranks:
            engine = self.all_engines[rank]
            if engine is None or rank in self._shutdown_confirmed:
                continue
            try:
                shutdown_refs[rank] = engine.shutdown.remote()
            except Exception as exc:
                if _is_actor_confirmed_dead(exc):
                    self._shutdown_confirmed.add(rank)
                    continue
                errors.append(f"rank={rank} shutdown submission: {exc}")
        for rank, ref in shutdown_refs.items():
            try:
                ray.get(ref, timeout=max(0.01, deadline - time.monotonic()))
                self._shutdown_confirmed.add(rank)
            except Exception as exc:
                if _is_actor_confirmed_dead(exc):
                    self._shutdown_confirmed.add(rank)
                    continue
                errors.append(f"rank={rank} shutdown: {exc}")
        for rank in ranks:
            engine = self.all_engines[rank]
            if engine is not None and rank not in self._shutdown_confirmed:
                continue
            if engine is not None:
                try:
                    ray.kill(engine)
                except Exception as exc:
                    errors.append(f"rank={rank} actor kill: {exc}")
                    continue
                self.all_engines[rank] = None
                self._shutdown_confirmed.discard(rank)
            try:
                self._remove_owned_pg(rank)
            except Exception as exc:
                errors.append(f"rank={rank} placement cleanup: {exc}")
                continue
            self._cleanup_pending.discard(rank)
        if errors:
            self._inference_observation.failed(observed)
            raise InferenceCleanupError("Cleanup remains unconfirmed: " + "; ".join(errors))

    def health_check(self) -> bool:
        healthy = True
        dead_heads = []
        for index, (key, workers) in enumerate(self._inference_replicas().items()):
            head_rank = index * self.nodes_per_engine
            if not workers or any(worker is None for worker in workers):
                healthy = False
                continue
            refs = []
            try:
                refs = [workers[0].health_generate.remote()]
                refs.extend(worker.health_process.remote() for worker in workers)
            except Exception as exc:
                logger.warning(f"{self._log_prefix} replica {key} health submission failed: {exc}")
                healthy = False
                if _is_actor_confirmed_dead(exc):
                    dead_heads.append(head_rank)
                continue
            replica_dead = False
            for position, ref in enumerate(refs):
                try:
                    result = ray.get(ref, timeout=5.0)
                except Exception as exc:
                    logger.warning(f"{self._log_prefix} replica {key} health check failed: {exc}")
                    healthy = False
                    replica_dead = replica_dead or _is_actor_confirmed_dead(exc)
                    continue
                if result is not True:
                    healthy = False
                    replica_dead = replica_dead or (position > 0 and result is False)
            if replica_dead:
                dead_heads.append(head_rank)
        if dead_heads:
            logger.warning(f"{self._log_prefix} retiring replicas with dead workers: heads={dead_heads}")
            try:
                self._retire_engines(dead_heads)
            except InferenceCleanupError as exc:
                logger.warning(f"{self._log_prefix} dead replica cleanup remains pending: {exc}")
        return healthy

    def onload(self, tags: Optional[list[str]] = None) -> None:
        if getattr(self, "_inference_role", None) == "rollout":
            return self._rollout_memory("resume_memory_occupation", tags)
        requested = set(_RESIDENT_TAGS if tags is None else tags)
        if not requested.issubset(_RESIDENT_TAGS):
            raise ValueError(f"Unknown inference memory tags: {sorted(requested - _RESIDENT_TAGS)}")
        if not requested:
            return
        with self._lifecycle_lock:
            if self._cleanup_pending:
                self._cleanup_slots(list(self._cleanup_pending))
            self.recover()
            observation = self._inference_observation
            dead = []
            for index, (key, workers) in enumerate(self._inference_replicas().items()):
                rank = index * self.nodes_per_engine
                if any(worker is None for worker in workers):
                    continue
                with observation.lock:
                    entry = observation.observe(key, workers)
                    if entry["state"] == "FAILED":
                        raise RuntimeError(
                            "Cannot onload a failed replica before offload or cleanup confirms its state"
                        )
                    missing = requested - entry["tags"]
                    reopen = _RESIDENT_TAGS.issubset(entry["tags"]) and not entry["admission"]
                    if not missing and not reopen:
                        continue
                    before = set(entry["tags"])
                observed = observation.begin({key: workers}, "ONLOADING")
                self._onloaded = False
                try:
                    if missing:
                        call_tags = None if not before and missing == _RESIDENT_TAGS else sorted(missing)
                        ray.get(
                            workers[0].resume_memory_occupation.remote(tags=call_tags),
                            timeout=_ENGINE_OPERATION_TIMEOUT_S,
                        )
                        observation.onloaded(observed, call_tags)
                    else:
                        call_tags = []
                    if _RESIDENT_TAGS.issubset(entry["tags"]) and call_tags is not None:
                        ray.get(workers[0].continue_generation.remote(), timeout=_ENGINE_OPERATION_TIMEOUT_S)
                        with observation.lock:
                            entry["admission"] = True
                            observation._refresh_state(entry)
                except Exception as exc:
                    observation.failed(observed)
                    if not _is_engine_dead(exc):
                        self._retire_engines([rank])
                        self._update_onloaded_state()
                        raise
                    dead.append(rank)
            if dead:
                self._retire_engines(dead)
                self.recover()
            self._update_onloaded_state()
            logger.info(f"{self._log_prefix} engines onload completed")

    def offload(self) -> None:
        if getattr(self, "_inference_role", None) == "rollout":
            return self._rollout_memory("release_memory_occupation")
        with self._lifecycle_lock:
            if self._cleanup_pending:
                self._cleanup_slots(list(self._cleanup_pending))
            observation = self._inference_observation
            replicas = {}
            skip_ranks = set()
            for index, (key, workers) in enumerate(self._inference_replicas().items()):
                entry = observation.observe(key, workers)
                if entry["state"] == "SLEEPING":
                    skip_ranks.add(index * self.nodes_per_engine)
                else:
                    replicas[key] = workers
            if not replicas:
                return
            observed = observation.begin(replicas, "DRAINING")
            self._last_fanout_succeeded = set()
            try:
                dead = self._fanout("release_memory_occupation", skip_ranks=skip_ranks)
                self._retire_engines(dead)
            except Exception:
                succeeded = {
                    key: observed[key]
                    for index, key in enumerate(self._inference_replicas())
                    if key in observed and index * self.nodes_per_engine in self._last_fanout_succeeded
                }
                observation.offloaded(succeeded, preserve_weights=getattr(self, "_inference_preserves_weights", False))
                observation.failed({key: entry for key, entry in observed.items() if key not in succeeded})
                self._update_onloaded_state()
                raise
            observation.offloaded(observed, preserve_weights=getattr(self, "_inference_preserves_weights", False))
            self._onloaded = False
            logger.info(f"{self._log_prefix} engines offload completed (retired {len(dead)} dead)")

    def _fanout(self, method: str, *, skip_ranks: Optional[set] = None, **kwargs) -> list[int]:
        skip = skip_ranks or set()
        handles = {}
        self._last_fanout_succeeded: set[int] = set()
        dead = []
        errors = []
        deadline = time.monotonic() + _ENGINE_OPERATION_TIMEOUT_S
        for rank in range(0, len(self.all_engines), self.nodes_per_engine):
            engine = self.all_engines[rank]
            if engine is None or rank in skip:
                continue
            try:
                handles[rank] = getattr(engine, method).remote(**kwargs)
            except Exception as exc:
                if _is_engine_dead(exc):
                    dead.append(rank)
                else:
                    errors.append(exc)
        for rank, handle in handles.items():
            try:
                ray.get(handle, timeout=max(0.01, deadline - time.monotonic()))
                self._last_fanout_succeeded.add(rank)
            except Exception as exc:
                if not _is_engine_dead(exc):
                    errors.append(exc)
                    continue
                logger.warning(f"{self._log_prefix} engine rank={rank} died during {method}: {exc}")
                dead.append(rank)
        if errors:
            raise errors[0]
        return dead

    def _retire_engines(self, ranks: list[int]) -> None:
        with self._lifecycle_lock:
            slots = [
                slot
                for rank in ranks
                for slot in range(
                    (rank // self.nodes_per_engine) * self.nodes_per_engine,
                    min((rank // self.nodes_per_engine + 1) * self.nodes_per_engine, len(self.all_engines)),
                )
            ]
            self._cleanup_slots(slots)

    def recover(self) -> set:
        with self._lifecycle_lock:
            if self._cleanup_pending:
                self._cleanup_slots(list(self._cleanup_pending))
            dead = [i for i, engine in enumerate(self.all_engines) if engine is None]
            if not dead:
                return set()
            heads = sorted({rank // self.nodes_per_engine * self.nodes_per_engine for rank in dead})
            self._retire_engines(heads)
            dead = [
                rank
                for head in heads
                for rank in range(head, min(head + self.nodes_per_engine, len(self.all_engines)))
            ]
            try:
                self._init_engines(dead)
            except InferenceCleanupError:
                raise
            except Exception as exc:
                logger.exception(f"{self._log_prefix} engine rebuild failed for ranks={dead}: {exc}")
            rebuilt = {rank for rank in dead if self.all_engines[rank] is not None}
            if all(engine is None for engine in self.all_engines):
                raise RuntimeError(f"All engines are dead and could not be rebuilt (ranks={dead})")
            self._update_onloaded_state()
            return rebuilt

    def is_onloaded(self) -> bool:
        return self._onloaded

    def shutdown(self) -> None:
        with self._lifecycle_lock:
            observed = self._inference_observation.begin(self._inference_replicas(), "STOPPING")
            try:
                self._cleanup_slots(list(range(len(self.all_engines))))
            except Exception:
                self._inference_observation.failed(observed)
                self._onloaded = False
                raise
            self._onloaded = False
            logger.info(f"{self._log_prefix} shutdown complete.")
