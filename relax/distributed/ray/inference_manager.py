# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Task-local engine owner; blocking GPU RPCs never occupy the actor loop."""

import asyncio
import inspect
from copy import deepcopy
from dataclasses import replace
from typing import Any
from uuid import uuid4

import ray

from relax.engine.inference.discovery import new_manager_epoch
from relax.engine.inference.lifecycle import AdmissionLedger, AdmissionScope
from relax.engine.inference.placement import PlacementPlanner
from relax.engine.inference.types import LifecycleState, Role, RoleSnapshot, RoutingSpec
from relax.engine.rollout.base_types import RolloutCleanupUnconfirmed
from relax.utils.logging_utils import get_logger


logger = get_logger(__name__)

# Failed rollback owners must outlive the failing factory stack. In particular,
# unnamed Teacher actors would otherwise lose their last driver reference.
_FAILED_ROLLBACK_OWNERS: dict[int, Any] = {}


def rollback_static_inference_manager(manager: Any) -> None:
    """Release an owner only after its engines have confirmed disposal."""
    key = id(manager)
    _FAILED_ROLLBACK_OWNERS[key] = manager
    try:
        if isinstance(manager, ray.actor.ActorHandle):
            ray.get(manager.invalidate_startup.remote(), timeout=10)
        ray.get(manager.dispose.remote(), timeout=60)
    except BaseException:
        logger.exception("Static inference cleanup unconfirmed; retaining owner %s", key)
        raise
    from relax.distributed.ray.inference_handle import InferenceRoleHandle

    if not isinstance(manager, InferenceRoleHandle):
        ray.kill(manager)
    _FAILED_ROLLBACK_OWNERS.pop(key, None)


# Explicit surface: private/workload methods cannot bypass the engine owner.
POOL_METHODS = frozenset(
    {
        "get_rollout_engines_and_lock",
        "get_weight_sync_lock",
        "get_router_address",
        "get_engines_info",
        "get_status",
        "get_lora_training_config",
        "get_lora_version_runtime",
        "configure_lora_training",
        "configure_lora_versions",
        "publish_training_lora",
        "get_training_weight_updating",
        "check_weights",
        "health_monitoring_pause",
        "health_monitoring_resume",
        "recover_rollout_engines",
        "register_recovered_engines",
        "clear_num_new_engines",
        "set_force_unhealthy",
        "create_scale_out_request",
        "execute_scale_out",
        "get_scale_out_status",
        "cancel_scale_out_request",
        "list_all_scale_out_requests",
        "cancel_all_scale_out_requests",
        "create_scale_in_request",
        "execute_scale_in",
        "get_scale_in_status",
        "list_all_scale_in_requests",
        "sync_weights_for_scaled_out_engines",
        "cancel_scale_out",
    }
)


async def _blocking(fn: Any, *args: Any, **kwargs: Any) -> Any:
    """Cancellation must not release ownership while a thread still mutates
    engines."""

    def run() -> Any:
        result = fn(*args, **kwargs)
        return asyncio.run(result) if inspect.isawaitable(result) else result

    work = asyncio.create_task(asyncio.to_thread(run))
    cancelled = False
    while True:
        try:
            result = await asyncio.shield(work)
            break
        except asyncio.CancelledError:
            if work.cancelled():
                # The operation itself was cancelled, rather than its caller.
                raise
            cancelled = True
    if cancelled:
        raise asyncio.CancelledError
    return result


class _ModelRuntime:
    def __init__(self, ledger: Any = None, placement: Any = None) -> None:
        self.pool: Any = None
        self.role = Role.ROLLOUT
        self._last_health = 0.0
        self.placement = placement if placement is not None else PlacementPlanner()
        self.ledger = ledger if ledger is not None else AdmissionLedger()
        self.epoch = new_manager_epoch()
        self.revision = 0
        self._topology: Any = None
        self._ready_generations: dict[str, Any] = {}
        self.state = LifecycleState.STARTING
        self.memory_ready = False
        self.resident = False
        self.required_weight_version: str | None = None
        self._policy_version_pending = False
        self._last_scaled_version_check = 0.0
        self._sync_generations: dict[str, tuple[Any, int, tuple[str, ...]]] = {}
        self._transition = asyncio.Lock()
        self._updates: dict[str, str] = {}
        self._disposed = False
        self._closing = False
        self._cleanup_task: asyncio.Task | None = None
        self._active_calls: set[asyncio.Future] = set()
        self._lora_publication = asyncio.Lock()
        self._recipients: set[str] = set()

    async def start(self, args: Any, binding: Any) -> dict[str, Any]:
        from relax.distributed.ray.inference_pool import RolloutPool, plan_rollout

        self.shared_memory = bool(getattr(args, "colocate", False)) and not getattr(args, "hybrid", False)
        async with self._transition:
            if self._closing or self._disposed:
                raise RuntimeError("Inference manager is closed")
            if self.pool is None:
                await _blocking(plan_rollout, args, binding, self.placement)

                def create_pool() -> None:
                    # Publish ownership inside the operation, even if the RPC
                    # caller is cancelled before receiving the startup reply.
                    self.pool = RolloutPool.__new__(RolloutPool)
                    RolloutPool.__init__(self.pool, args, binding)

                await _blocking(create_pool)
                self._recipients = set(self.pool.synced_actor_ids)
                self.memory_ready = True
                self.resident = True
                self.state = LifecycleState.READY
            primary = next(iter(self.pool.backend.servers))
            return self.pool.backend.get_router_address(primary)

    async def start_static(self, role: str, args: Any, binding: Any, options: dict[str, Any]) -> None:
        from relax.distributed.ray.static_inference_pool import StaticInferencePool

        async with self._transition:
            if self.pool is not None or self._closing:
                raise RuntimeError("Inference manager is already started or closed")
            self.role = Role(role)
            if self.role is Role.ROLLOUT:
                raise ValueError("Static pools only support auxiliary roles")

            def create_pool() -> None:
                self.pool = StaticInferencePool.__new__(StaticInferencePool)
                StaticInferencePool.__init__(self.pool, self.role, args, binding, options)
                self.pool.record_placement(self.placement)

            await _blocking(create_pool)
            self.memory_ready = not self.pool.start_sleeping
            self.resident = self.memory_ready
            self.state = LifecycleState.READY if self.memory_ready else LifecycleState.SLEEPING
            if not self.memory_ready:
                self.ledger.fences.add("memory")
            self._last_health = asyncio.get_running_loop().time()

    async def recover_static(self) -> None:
        async with self._transition:
            if self._closing or self.pool is None:
                raise RuntimeError("Inference manager is closed")
            if not self.memory_ready:
                return
            dead = await _blocking(self.pool.unhealthy_ranks)
            router_healthy = await _blocking(self.pool.backend.router.is_healthy)
            self._last_health = asyncio.get_running_loop().time()
            if not dead and router_healthy:
                return
            await self.ledger.drain("health", timeout=120)
            self.state = LifecycleState.DRAINING
            try:
                await _blocking(self.pool.recover, dead)
                await _blocking(self.pool.record_placement, self.placement)
            except BaseException:
                self.state = LifecycleState.DEAD
                raise
            self.state = LifecycleState.READY
            self._last_health = asyncio.get_running_loop().time()
            self.ledger.fences.discard("health")

    async def _confirm_scaled_out_versions(self) -> None:
        if self.role is not Role.ROLLOUT or self.pool is None or self.required_weight_version is None:
            return
        backend = self.pool.backend
        if not hasattr(backend, "servers") or not hasattr(backend, "_engine_lifecycle_lock"):
            return
        from relax.distributed.ray.policy_engine_pool import EngineGroupLifecycle

        receipts = getattr(backend, "_seed_sync_receipts", {})
        with backend._engine_lifecycle_lock:
            for server in backend.servers.values():
                for group in server.engine_groups:
                    if not group.is_scaled_out or group.lifecycle_status is not EngineGroupLifecycle.ACTIVE:
                        continue
                    for first in range(0, len(group.all_engines), group.nodes_per_engine):
                        slots = tuple(group.all_engines[first : first + group.nodes_per_engine])
                        if len(slots) != group.nodes_per_engine or any(engine is None for engine in slots):
                            continue
                        identities = tuple(engine._actor_id.hex() for engine in slots)
                        receipt = receipts.get(identities[0])
                        if receipt == (identities, self.required_weight_version):
                            self.pool.mark_synced(set(identities), self.required_weight_version)

    def _process_generations(self) -> dict[str, tuple[Any, ...]]:
        """Read process identities; memory and weight residency are not
        topology."""
        backend = self.pool.backend
        result = {}

        def actor_id(engine: Any) -> str | None:
            identity = getattr(engine, "_actor_id", None)
            return identity.hex() if identity is not None else None

        if self.role is Role.ROLLOUT and hasattr(backend, "servers"):
            lock = backend._engine_lifecycle_lock
            if not lock.acquire(blocking=False):
                return result
            try:
                for name, server in backend.servers.items():
                    pd = []
                    for group in server.engine_groups:
                        for first in range(0, len(group.all_engines), group.nodes_per_engine):
                            tokens = []
                            for slot in range(first, min(first + group.nodes_per_engine, len(group.all_engines))):
                                engine = group.all_engines[slot]
                                value = getattr(group.engine_init_records.get(slot), "value", None)
                                pid = value.get("pid") if isinstance(value, dict) else None
                                tokens.append((actor_id(engine), pid))
                            marker = tuple(tokens)
                            result[f"{name}/engine-{group.rank_offset + first}"] = marker
                            if group.worker_type in {"prefill", "decode"}:
                                pd.extend(marker)
                    if pd:
                        result[f"{name}/pd-service"] = tuple(pd)
            finally:
                lock.release()
        elif hasattr(backend, "all_engines") and hasattr(self.pool, "model_id"):
            for first in range(0, len(backend.all_engines), backend.nodes_per_engine):
                result[f"{self.pool.model_id}/replica-{first}"] = tuple(
                    actor_id(engine) for engine in backend.all_engines[first : first + backend.nodes_per_engine]
                )
        return result

    async def snapshot(self, role: Role | str = Role.ROLLOUT) -> RoleSnapshot:
        if Role(role) is not self.role or self.pool is None:
            raise RuntimeError("Inference role is not registered")
        if getattr(self.pool.backend, "_scale_out_cleanup_failed", False):
            self.state = LifecycleState.DEAD
        if (
            self.role is not Role.ROLLOUT
            and not self._closing
            and not self.ledger.fences
            and not self._transition.locked()
            and not self.ledger.permits
        ):
            if asyncio.get_running_loop().time() - self._last_health >= 5:
                await self.recover_static()
        if self.role is Role.ROLLOUT and self.required_weight_version is not None:
            now = asyncio.get_running_loop().time()
            if now - self._last_scaled_version_check >= 5:
                self._last_scaled_version_check = now
                await self._confirm_scaled_out_versions()
        root_fences = getattr(getattr(self.ledger, "root", None), "fences", set())
        ready = self.memory_ready and not self.ledger.fences and not root_fences
        if self.role is Role.ROLLOUT and getattr(self.pool, "requires_weight_confirmation", False):
            models = self.pool.observe(
                ready=ready, state=self.state, required_weight_version=self.required_weight_version
            )
        else:
            models = self.pool.observe(ready=ready, state=self.state)
        stable = not any(reason == "topology_busy" for model in models for _, reason in model.issues)
        if stable:
            generations = self._process_generations()
            for model in models:
                for replica in (*model.replicas, *(worker for _, worker in model.pd_workers)):
                    if replica.state is LifecycleState.READY and replica.engine_id in generations:
                        self._ready_generations[replica.engine_id] = generations[replica.engine_id]
            topology = tuple(
                (
                    model.model_id,
                    model.router_url,
                    tuple(
                        (replica.engine_id, replica.base_url, self._ready_generations.get(replica.engine_id))
                        for replica in model.replicas
                    ),
                    tuple(
                        (kind, worker.engine_id, worker.base_url, self._ready_generations.get(worker.engine_id))
                        for kind, worker in model.pd_workers
                    ),
                )
                for model in models
            )
            if topology != self._topology:
                self.revision += 1
                self._topology = topology
        names = tuple(model.model_id for model in models)
        return RoleSnapshot(
            self.role,
            self.epoch,
            self.revision,
            "inference",
            models,
            RoutingSpec(default_model=names[0] if names else None, route_key_to_model=tuple((n, n) for n in names)),
        )

    async def admit_request(self, role: Role | str, model_id: str, trace_id: str | None = None) -> str:
        # No await between the final observation and recording the permit.
        snapshot = await self.snapshot(role)
        model = next((model for model in snapshot.models if model.model_id == model_id), None)
        if model is None or not model.admission:
            raise RuntimeError("Inference model is not ready")
        return self.ledger.admit(model_id)

    async def wait_ready(self, role: Role | str, model_id: str | None = None) -> None:
        """Internal generation pauses across normal weight/memory
        transitions."""
        while True:
            if self._closing or self._disposed:
                raise RuntimeError("Inference manager is closing")
            if self.state is LifecycleState.DEAD:
                if "unconfirmed_dispatch" in self.ledger.fences:
                    raise RuntimeError("Inference manager cannot confirm distributed POST dispatch has stopped")
                if self.pool is not None and getattr(self.pool.backend, "_scale_out_cleanup_failed", False):
                    raise RuntimeError("Inference manager cannot confirm scale-out rollback")
                raise RuntimeError("Inference manager cannot restore engine residency")
            if any(
                model.admission and (model_id is None or model.model_id == model_id)
                for model in (await self.snapshot(role)).models
            ):
                return
            await asyncio.sleep(0.05)

    async def complete_request(self, permit: str) -> None:
        await self.ledger.complete(permit)

    async def fail_request_transport(self, permit: str) -> None:
        if permit in self.ledger.permits:
            self.ledger.fences.add("unconfirmed_dispatch")
            self.state = LifecycleState.DEAD

    def _capture_sync_generations(self) -> None:
        self._sync_generations = {}
        if self.pool is None or not hasattr(self.pool.backend, "servers"):
            return
        backend = self.pool.backend
        with backend._engine_lifecycle_lock:
            for server in backend.servers.values():
                for group in server.engine_groups:
                    for first in range(0, len(group.all_engines), group.nodes_per_engine):
                        slots = tuple(group.all_engines[first : first + group.nodes_per_engine])
                        if len(slots) == group.nodes_per_engine and all(engine is not None for engine in slots):
                            identities = tuple(engine._actor_id.hex() for engine in slots)
                            self._sync_generations[identities[0]] = (group, first, identities)

    def _confirmed_sync_generations(self, heads: set[str]) -> set[str]:
        confirmed = set()
        if self.pool is None or not hasattr(self.pool.backend, "servers"):
            return confirmed
        with self.pool.backend._engine_lifecycle_lock:
            for head in heads:
                captured = self._sync_generations.get(head)
                if captured is None:
                    continue
                group, first, identities = captured
                current = tuple(
                    engine._actor_id.hex() if engine is not None else None
                    for engine in group.all_engines[first : first + len(identities)]
                )
                if current == identities:
                    confirmed.update(identities)
        return confirmed

    async def begin_weight_update(self, source: str) -> str:
        if self._closing or self._disposed or self.pool is None:
            raise RuntimeError("Inference manager is not running")
        if self.role is not Role.ROLLOUT:
            raise ValueError("Static models do not receive policy weight updates")
        token = uuid4().hex
        if source == "dcs":
            self._policy_version_pending = True
            self.ledger.fences.add("transport_version")
        self._updates[token] = source
        await self.ledger.drain(token, timeout=120)
        return token

    async def capture_dcs_recipients(self, token: str) -> None:
        """Record replica generations while the trainer owns the DCS lock."""
        if self._closing or self._disposed or self._updates.get(token) != "dcs":
            raise RuntimeError("DCS weight update is not active")
        if getattr(self.pool.backend, "_scale_out_cleanup_failed", False):
            raise RuntimeError("DCS weight update cannot proceed after unconfirmed scale-out rollback")
        self._capture_sync_generations()

    async def end_weight_update(self, token: str, weight_version: str | int | None = None) -> None:
        if token not in self._updates:
            raise ValueError("Unknown weight-update token")
        if self.pool is not None and getattr(self.pool, "requires_weight_confirmation", False):
            if weight_version is None:
                raise ValueError("A completed policy sync must supply its actual committed weight version")
            self.required_weight_version = str(weight_version)
            if self._updates[token] != "dcs":
                self.pool.mark_synced(self._recipients, self.required_weight_version)
        elif self.pool is not None:
            self.pool.mark_synced(self._recipients)
        self._updates.pop(token)
        self.ledger.fences.discard(token)
        self.ledger.fences.discard("recovery")

    async def begin_rollout(self, rollout_id: int | None) -> None:
        if self.pool is None or self._closing or self._disposed:
            raise RuntimeError("Inference manager is not running")
        backend = self.pool.backend
        await _blocking(backend._begin_generation)
        if self._closing:
            raise RuntimeError("Inference manager is not running")
        if rollout_id is not None:
            backend.rollout_id = rollout_id
        await _blocking(backend.health_monitoring_resume)

    async def inject_ci_fault(self) -> None:
        if self.pool is None or self._closing or self._disposed:
            raise RuntimeError("Inference manager is not running")
        await _blocking(self.pool.backend._try_ci_fault_injection)

    async def get_engine_hosts_ports(self) -> list[tuple[str, int]]:
        async with self._transition:
            return await _blocking(self.pool.backend.get_engine_hosts_ports)

    async def shutdown(self) -> None:
        await self.dispose()

    async def get_urls(self) -> list[str]:
        return [
            replica.base_url + "/generate"
            for model in (await self.snapshot(self.role)).models
            for replica in model.replicas
            if replica.direct_eligible
        ]

    async def health_check(self) -> bool:
        await self.recover_static()
        return any(model.admission for model in (await self.snapshot(self.role)).models)

    async def onload(self) -> None:
        await self.pool_call("onload")

    async def offload(self) -> None:
        await self.pool_call("offload")

    async def pool_call(self, method: str, *args: Any, **kwargs: Any) -> Any:
        releasing_weights = method == "set_weight_updating" and not (args[0] if args else kwargs["is_updating"])
        if self._closing and not releasing_weights and not method.startswith("cancel_"):
            raise RuntimeError("Inference manager is closing")
        completed = asyncio.get_running_loop().create_future()
        self._active_calls.add(completed)
        try:
            if method in {"configure_lora_training", "configure_lora_versions", "publish_training_lora"}:
                async with self._lora_publication:
                    return await self._pool_call(method, *args, **kwargs)
            if method == "recover_rollout_engines":
                # Recovery cannot mutate engine residency during memory transitions.
                async with self._transition, self._lora_publication:
                    return await self._pool_call(method, *args, **kwargs)
            return await self._pool_call(method, *args, **kwargs)
        finally:
            self._active_calls.discard(completed)
            completed.set_result(None)

    async def _pool_call(self, method: str, *args: Any, **kwargs: Any) -> Any:
        if self.pool is None or self._disposed:
            raise RuntimeError("Inference manager is not running")
        if self.role is not Role.ROLLOUT and method not in {"onload", "offload"}:
            raise ValueError("Static models only support memory lifecycle operations")
        backend = self.pool.backend
        if method in {"offload", "onload", "onload_weights", "onload_kv"}:
            async with self._transition:
                if method == "offload":
                    await self.ledger.drain("memory", timeout=600)
                    self.state = LifecycleState.DRAINING
                    try:
                        await _blocking(backend.offload)
                    except BaseException:
                        self.state = LifecycleState.DEAD
                        raise
                    self.memory_ready = False
                    self.resident = False
                    self.state = LifecycleState.SLEEPING
                else:
                    if method == "onload" and not args and not kwargs and self.role is Role.ROLLOUT:
                        if self.state is LifecycleState.READY and self.memory_ready:
                            return None
                        if self.state is LifecycleState.SLEEPING and self.resident:
                            method = "onload_kv"
                    # Close admission before even a partial memory transition.
                    await self.ledger.drain("memory", timeout=600)
                    self.state = LifecycleState.ONLOADING
                    self.resident = True
                    try:
                        await _blocking(getattr(backend, method), *args, **kwargs)
                    except BaseException:
                        self.state = LifecycleState.DEAD
                        raise
                    tags = kwargs.get("tags", args[0] if args else None)
                    self.memory_ready = method == "onload_kv" or (
                        method == "onload" and (tags is None or "kv_cache" in tags)
                    )
                    self.state = LifecycleState.READY if self.memory_ready else LifecycleState.SLEEPING
                    if self.memory_ready:
                        self.ledger.fences.discard("memory")
                return None
        if method == "set_weight_updating":
            updating = args[0] if args else kwargs["is_updating"]
            async with self._transition:
                if updating:
                    self.ledger.fences.add("fully_async")
                    accepted = await _blocking(backend.set_weight_updating, True)
                    if not accepted:
                        self.ledger.fences.discard("fully_async")
                        return False
                    if getattr(self.pool, "requires_weight_confirmation", False):
                        self._capture_sync_generations()
                        self._policy_version_pending = True
                    await self.ledger.drain("fully_async", timeout=120)
                else:
                    await _blocking(backend.set_weight_updating, False)
                    if getattr(self.pool, "requires_weight_confirmation", False) and self._policy_version_pending:
                        return True  # Keep admission closed until transport publication is confirmed.
                    self.pool.mark_synced(self._recipients)
                    self.ledger.fences.discard("fully_async")
                    self.ledger.fences.discard("recovery")
                return True
        if method not in POOL_METHODS:
            raise ValueError(f"Unsupported inference pool operation: {method}")
        if method == "recover_rollout_engines":
            backend._require_unmanaged_engine_pool()
            await self.ledger.drain("recovery", timeout=120)
        confirmed_version = kwargs.pop("weight_version", None) if method == "register_recovered_engines" else None
        if method == "register_recovered_engines":
            base_confirmed = kwargs.pop("base_weights_confirmed", True)
            if not base_confirmed:
                # Adapter-only sync confirms previously loaded generations, never
                # a replacement that has not received its base checkpoint.
                current = set(getattr(self.pool, "actor_weight_versions", {}))
                selected = set(args[0] if args else kwargs["synced_actor_ids"]) & current
                if args:
                    args = (sorted(selected), *args[1:])
                else:
                    kwargs["synced_actor_ids"] = sorted(selected)
        if method == "register_recovered_engines" and getattr(self.pool, "requires_weight_confirmation", False):
            heads = set(args[0] if args else kwargs["synced_actor_ids"])
            confirmed = self._confirmed_sync_generations(heads)
            if args:
                args = (sorted(confirmed), *args[1:])
            else:
                kwargs["synced_actor_ids"] = sorted(confirmed)
        if method == "sync_weights_for_scaled_out_engines" and getattr(
            self.pool, "requires_weight_confirmation", False
        ):
            version = self.required_weight_version
            kwargs["required_version"] = version
            kwargs["confirmed_actor_ids"] = {
                actor_id for actor_id, confirmed in self.pool.actor_weight_versions.items() if confirmed == version
            }
        result = await _blocking(getattr(backend, method), *args, **kwargs)
        if method in {"get_rollout_engines_and_lock", "recover_rollout_engines"}:
            # The sync transport exposes only each replica's first node. Capture
            # every node in those exact replica generations before publishing
            # success; a later replacement must not inherit this confirmation.
            recipients = {engine._actor_id.hex() for engine in result[0] if engine is not None}
            with backend._engine_lifecycle_lock:
                for server in backend.servers.values():
                    for group in server.engine_groups:
                        slots = group.all_engines
                        for first in range(0, len(slots), group.nodes_per_engine):
                            head = slots[first]
                            if head is not None and head._actor_id.hex() in recipients:
                                recipients.update(
                                    engine._actor_id.hex()
                                    for engine in slots[first : first + group.nodes_per_engine]
                                    if engine is not None
                                )
            self._recipients = recipients
            if method == "recover_rollout_engines" and self._recipients <= self.pool.synced_actor_ids:
                self.ledger.fences.discard("recovery")
        if method == "register_recovered_engines":
            recipients = set(args[0] if args else kwargs["synced_actor_ids"])
            if confirmed_version is not None:
                self.required_weight_version = str(confirmed_version)
                self.pool.mark_synced(recipients, self.required_weight_version)
                self._policy_version_pending = False
                self.ledger.fences.discard("transport_version")
            else:
                self.pool.mark_synced(recipients)
        if method == "execute_scale_out" or (
            method == "sync_weights_for_scaled_out_engines" and isinstance(result, dict) and result.get("success")
        ):
            await self._confirm_scaled_out_versions()
        return result

    async def dispose(self, worker: Any = None) -> None:
        self._closing = True
        self.ledger.fences.add("shutdown")
        if self._cleanup_task is None or (
            self._cleanup_task.done()
            and (self._cleanup_task.cancelled() or self._cleanup_task.exception() is not None)
        ):
            self._cleanup_task = asyncio.create_task(self._dispose(worker))
        cleanup = self._cleanup_task
        cancelled = False
        while True:
            try:
                await asyncio.shield(cleanup)
                break
            except asyncio.CancelledError:
                if cleanup.cancelled():
                    raise
                cancelled = True
        if cancelled:
            raise asyncio.CancelledError

    async def _dispose(self, worker: Any = None) -> None:
        self._closing = True
        self.ledger.fences.add("shutdown")
        if self.pool is not None:
            cancel = getattr(self.pool.backend, "cancel_all_scale_out_requests", None)
            if cancel is not None:
                try:
                    await _blocking(cancel)
                except Exception:
                    logger.exception("Could not cancel scale-out requests; waiting for their owners before cleanup")
        # A running thread may still publish newly created engines. Preserve the
        # owner until it finishes; weight-update release and permits stay live.
        while self._active_calls:
            await asyncio.gather(*(asyncio.shield(task) for task in tuple(self._active_calls)), return_exceptions=True)
        async with self._transition:
            self._disposed = True
            self.ledger.fences.add("shutdown")
            worker_cleanup_unconfirmed = False
            try:
                try:
                    if self.pool is not None:
                        # Dead engines leave empty slots in the backend topology.
                        handles = (
                            self.pool.backend.rollout_engines
                            if self.role is Role.ROLLOUT
                            else self.pool.backend.engines
                        )
                        engines = [engine for engine in handles if engine is not None]
                        if engines:
                            await _blocking(
                                ray.get, [engine.abort_requests.remote() for engine in engines], timeout=10
                            )
                except Exception:
                    logger.exception("Could not abort residual requests during shutdown")
                if worker is not None:
                    try:
                        await asyncio.wait_for(worker.close.remote(), timeout=30)
                    except ray.exceptions.ActorDiedError:
                        pass
                    except RolloutCleanupUnconfirmed:
                        worker_cleanup_unconfirmed = True
                        raise
                    except (asyncio.TimeoutError, ray.exceptions.ActorUnavailableError) as exc:
                        worker_cleanup_unconfirmed = True
                        raise RolloutCleanupUnconfirmed("Rollout worker cleanup is unconfirmed") from exc
                await self.ledger.drain("shutdown", timeout=10)
            finally:
                if not worker_cleanup_unconfirmed:
                    try:
                        if self.pool is not None:
                            await _blocking(self.pool.dispose)
                            self.pool = None
                            self.state = LifecycleState.DEAD
                            self.memory_ready = False
                            self.resident = False
                    finally:
                        if worker is not None:
                            ray.kill(worker)


@ray.remote(num_cpus=1, num_gpus=0, max_concurrency=4096)
class InferenceManager(_ModelRuntime):
    """One task actor owns all role pools, permits, placement and handoffs."""

    def __init__(self, startup_token: str | None = None) -> None:
        super().__init__()
        self._startup_token = startup_token
        self._startup_valid = True
        self.requests = self.ledger
        self.ledger = AdmissionScope(self.requests, "rollout/default")
        self.models: dict[tuple[Role, str], _ModelRuntime] = {}
        self._phase_coordinator: Any = None
        self._phase_switch = asyncio.Lock()
        self._registration = asyncio.Lock()
        self._phase_models: dict[str, tuple[tuple[str, str], ...]] = {}

    def _check_startup(self, token: str | None) -> None:
        if not self._startup_valid or self._closing:
            raise RuntimeError("Inference deployment startup has been retired")
        if self._startup_token is not None and token != self._startup_token:
            raise RuntimeError("Inference deployment startup token does not match")

    async def invalidate_startup(self) -> None:
        self._startup_valid = False
        self._closing = True
        self.requests.fences.add("shutdown")

    async def start(self, args: Any, binding: Any, startup_token: str | None = None) -> dict[str, Any]:
        self._check_startup(startup_token)
        address = await _ModelRuntime.start(self, args, binding)
        self._check_startup(startup_token)
        return address

    async def attach_workload(self, worker: Any, startup_token: str | None) -> None:
        self._check_startup(startup_token)
        self.pool.backend.rollout_worker = worker

    async def register_static(
        self, role: str, args: Any, binding: Any, options: dict[str, Any], startup_token: str | None = None
    ) -> str:
        async with self._registration:
            return await self._register_static(role, args, binding, options, startup_token)

    async def _register_static(
        self, role: str, args: Any, binding: Any, options: dict[str, Any], startup_token: str | None = None
    ) -> str:
        self._check_startup(startup_token)
        key = (Role(role), options.get("model_id", "default"))
        if key in self.models or self._closing or self._phase_coordinator is not None:
            raise RuntimeError("Inference model already registered or startup complete")
        if args is not None and binding is not None:
            from relax.engine.inference.config_adapters import static_role_spec
            from relax.engine.inference.task_placement import StaticPlacement, plan_static_placement
            from relax.engine.inference.types import DeploymentMode

            spec = static_role_spec(Role(role), args, options)
            offset = options.get("bundle_offset", 0)
            if spec.deployment.mode is DeploymentMode.SPLIT:
                offset += args.rollout_num_gpus
            plan_static_placement(args, StaticPlacement(spec, binding, offset), self.placement, dry_run=True)
        runtime = _ModelRuntime(AdmissionScope(self.requests, f"{key[0].value}/{key[1]}"), self.placement)
        runtime.epoch = self.epoch
        self.models[key] = runtime
        await runtime.start_static(role, args, binding, options)
        self._check_startup(startup_token)
        return key[1]

    async def dispatch(self, role: str, model_id: str, method: str, *args: Any, **kwargs: Any) -> Any:
        allowed = {
            "snapshot",
            "admit_request",
            "complete_request",
            "wait_ready",
            "onload",
            "offload",
            "dispose",
            "shutdown",
            "health_check",
            "recover_static",
            "get_urls",
            "get_engine_hosts_ports",
            "pool_call",
        }
        if method not in allowed:
            raise ValueError(f"Unsupported role operation: {method}")
        if self._closing and method not in {"complete_request", "dispose", "shutdown"}:
            raise RuntimeError("Task inference manager is closing")
        runtime = self.models[(Role(role), model_id)]
        if method == "snapshot":
            requested_role = Role(args[0] if args else kwargs.get("role", role))
            if requested_role is not Role(role):
                raise ValueError("Role routing handle cannot request another inference role")
            snapshot = await self.snapshot(requested_role)
            return replace(
                snapshot,
                models=tuple(model for model in snapshot.models if model.model_id == model_id),
                routing=RoutingSpec(
                    default_model=model_id,
                    route_key_to_model=tuple(
                        (key, target) for key, target in snapshot.routing.route_key_to_model if target == model_id
                    ),
                    config_version=snapshot.routing.config_version,
                ),
            )
        if self._phase_coordinator is not None and getattr(self, "shared_memory", False):
            memory_method = args[0] if method == "pool_call" and args else method
            if memory_method in {"onload", "offload"}:
                phase = runtime.pool.spec.deployment.phase if memory_method == "onload" else "startup"
                return await self.switch(phase)
        return await getattr(runtime, method)(*args, **kwargs)

    async def snapshot(self, role: Role | str = Role.ROLLOUT) -> RoleSnapshot:
        role = Role(role)
        if role is self.role and self.pool is not None:
            snapshot = await _ModelRuntime.snapshot(self, role)
            return replace(
                snapshot, phase=self._phase_coordinator._phase if self._phase_coordinator else snapshot.phase
            )
        entries = [runtime for (item_role, _), runtime in self.models.items() if item_role is role]
        if not entries:
            raise RuntimeError("Inference role is not registered")
        snapshots = [await runtime.snapshot(role) for runtime in entries]
        models = tuple(model for snapshot in snapshots for model in snapshot.models)
        return RoleSnapshot(
            role,
            self.epoch,
            sum(item.topology_revision for item in snapshots),
            self._phase_coordinator._phase if self._phase_coordinator else "inference",
            models,
            RoutingSpec(
                default_model=models[0].model_id,
                route_key_to_model=tuple((model.model_id, model.model_id) for model in models),
            ),
        )

    async def admit_request(self, role: Role | str, model_id: str, trace_id: str | None = None) -> str:
        runtime = self if Role(role) is self.role and self.pool is not None else self.models[(Role(role), model_id)]
        return await _ModelRuntime.admit_request(runtime, role, model_id, trace_id)

    async def complete_request(self, permit: str) -> None:
        await self.requests.complete(permit)

    async def wait_ready(self, role: Role | str, model_id: str | None = None) -> None:
        if Role(role) is self.role and self.pool is not None:
            return await _ModelRuntime.wait_ready(self, role, model_id)
        runtime = self.models[(Role(role), model_id or "default")]
        await runtime.wait_ready(role, model_id)

    async def pool_call(self, method: str, *args: Any, **kwargs: Any) -> Any:
        if self._phase_coordinator is not None and getattr(self, "shared_memory", False):
            if method == "onload_weights":
                await self.switch("startup")
            elif method in {"onload", "onload_kv"}:
                return await self.switch("inference")
            elif method == "offload":
                return await self.switch("startup")
        return await _ModelRuntime.pool_call(self, method, *args, **kwargs)

    async def configure_phases(self, phases: dict[str, tuple[tuple[str, str], ...]], initial_phase: str) -> None:
        if self._closing or not self._startup_valid:
            raise RuntimeError("Inference deployment startup has been retired")
        from relax.engine.inference.phases import PhaseCoordinator

        known = {(role.value, model) for role, model in self.models}
        if self.pool is not None:
            known.add(("rollout", "default"))
        configured = [key for owners in phases.values() for key in owners]
        if len(configured) != len(set(configured)) or set(configured) != known:
            raise ValueError("Each task inference model must belong to exactly one lifecycle phase")

        async def call(key: tuple[str, str], method: str) -> None:
            runtime = self if key[0] == "rollout" else self.models[(Role(key[0]), key[1])]
            await _ModelRuntime.pool_call(runtime, method)

        self._phase_models = phases
        self._phase_coordinator = PhaseCoordinator(phases, initial_phase, call)

    async def switch(self, phase: str) -> None:
        if (self._closing or not self._startup_valid) and phase != "startup":
            raise RuntimeError("Inference deployment startup has been retired")
        if self._phase_coordinator is None:
            raise RuntimeError("Task phases have not been configured")
        async with self._phase_switch:
            # A queued handoff must re-check retirement after acquiring the lock.
            if (self._closing or not self._startup_valid) and phase != "startup":
                raise RuntimeError("Inference deployment startup has been retired")
            if phase not in self._phase_models:
                raise ValueError(f"Unknown inference phase: {phase}")
            target = set(self._phase_models[phase])
            source = set(self._phase_models[self._phase_coordinator._phase])
            persistent = set(self._phase_models.get("resident", ()))
            entries = [
                (("rollout", "default"), self),
                *(((role.value, model), runtime) for (role, model), runtime in self.models.items()),
            ]
            try:
                for key, runtime in entries:
                    if (
                        key not in target
                        and key not in source
                        and key not in persistent
                        and runtime.pool is not None
                        and runtime.resident
                    ):
                        await _ModelRuntime.pool_call(runtime, "offload")
                await self._phase_coordinator.switch(phase)
            except BaseException as exc:
                self._phase_coordinator._failure = exc
                raise

    async def dispose(self, worker: Any = None) -> None:
        self._startup_valid = False
        self._closing = True
        self.requests.fences.add("shutdown")
        for runtime in self.models.values():
            await runtime.dispose()
        if worker is None and self.pool is not None:
            worker = getattr(self.pool.backend, "rollout_worker", None)
        await _ModelRuntime.dispose(self, worker)


@ray.remote(num_cpus=0)
def dispose_managed_rollout(manager: Any, worker: Any) -> None:
    """Release the CPU owner after cleanup, including on global restarts."""
    try:
        ray.get(manager.dispose.remote(worker))
    except BaseException:
        _FAILED_ROLLBACK_OWNERS[id(manager)] = manager
        raise
    ray.kill(manager)
    _FAILED_ROLLBACK_OWNERS.pop(id(manager), None)


def create_managed_rollout(args: Any, pg: Any, data_source: Any, runtime_env: Any) -> tuple[Any, Any]:
    """Create a complete managed task, rolling back every partial startup."""
    from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

    from relax.core.node_group_affinity import require_control_plane_resource_on_node, with_control_plane_affinity
    from relax.distributed.ray.inference_handle import ManagedRolloutHandle
    from relax.distributed.ray.placement_group import _get_head_node_id
    from relax.distributed.ray.rollout_worker import RolloutWorker
    from relax.engine.sft.runtime import is_offline_mode
    from relax.utils.agentic_obs.startup import record_run_startup

    node = _get_head_node_id()
    require_control_plane_resource_on_node(args, node)
    options = with_control_plane_affinity(
        args,
        {
            "num_cpus": 1,
            "num_gpus": 0,
            "runtime_env": runtime_env,
            "scheduling_strategy": NodeAffinitySchedulingStrategy(node_id=node, soft=False),
        },
    )
    manager = getattr(args, "_rollout_engine_manager", None)
    owns_manager = manager is None
    startup_token = getattr(args, "_inference_startup_token", None)
    if owns_manager:
        startup_token = uuid4().hex
        args._inference_startup_token = startup_token
        manager = InferenceManager.options(**options).remote(startup_token=startup_token)
    worker = None
    try:
        address = ray.get(manager.start.remote(deepcopy(args), pg, startup_token=startup_token))
        worker_args = deepcopy(args)
        worker_args.sglang_router_ip = address["router_ip"]
        worker_args.sglang_router_port = address["router_port"]
        worker_args._inference_manager_handle = manager
        worker = RolloutWorker.options(**options).remote(worker_args, data_source, manager)
        handle = ManagedRolloutHandle(manager, worker, getattr(args, "_inference_phase_coordinator", None))
        if is_offline_mode(args):
            count = args.num_rollout_per_epoch
        else:
            count = ray.get(handle.get_num_rollout_per_epoch.remote())
            args.num_rollout_per_epoch = count
            if args.num_epoch is not None:
                epoch_count = count * args.num_epoch
                args.num_rollout = min(args.num_rollout, epoch_count) if args.num_rollout is not None else epoch_count
        evaluation_only = (
            args.num_rollout == 0
            and args.debug_rollout_only
            and args.eval_interval is not None
            and not args.skip_eval_before_train
        )
        if args.num_rollout is None or (args.num_rollout <= 0 and not evaluation_only):
            raise ValueError(f"Invalid resolved num_rollout: {args.num_rollout}")
        ray.get(worker.configure.remote(args.num_rollout, count))
        record_run_startup(args)
        if args.check_weight_update_equal:
            ray.get(handle.check_weights.remote(action="snapshot"))
            ray.get(handle.check_weights.remote(action="reset_tensors"))
        if args.offload_rollout:
            ray.get(handle.offload.remote())
        ray.get(manager.attach_workload.remote(worker, startup_token))
        return handle, count
    except BaseException:
        if not owns_manager:
            # Controller retains ownership of the task and its PGs.
            if worker is not None:
                try:
                    ray.get(worker.close.remote(), timeout=30)
                except Exception:
                    logger.warning("Could not close failed RolloutWorker before stopping it", exc_info=True)
                try:
                    ray.kill(worker)
                except Exception:
                    logger.exception("Could not stop failed RolloutWorker")
            raise
        try:
            ray.get(manager.dispose.remote(worker), timeout=60)
        except BaseException:
            _FAILED_ROLLBACK_OWNERS[id(manager)] = manager
            logger.exception("Managed rollout startup rollback could not be confirmed; retaining owner")
        else:
            if worker is not None:
                ray.kill(worker)
            ray.kill(manager)
        raise


def create_static_inference_manager(
    args: Any,
    role: str,
    binding: Any = None,
    *,
    runtime_env: Any = None,
    name: str | None = None,
    **pool_options: Any,
) -> Any:
    """Start one static model with rollback before returning its owner."""
    from relax.distributed.ray.inference_handle import InferenceRoleHandle

    task = pool_options.pop("task_manager", None) or getattr(args, "_rollout_engine_manager", None)
    if task is None:
        raise RuntimeError("Static pools must be registered in the task InferenceManager")
    model_id = pool_options.get("model_id", "default")
    ray.get(
        task.register_static.remote(
            role, deepcopy(args), binding, pool_options, startup_token=getattr(args, "_inference_startup_token", None)
        )
    )
    return InferenceRoleHandle(task, role, model_id)
