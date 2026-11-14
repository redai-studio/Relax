# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Task-local engine owner; blocking GPU RPCs never occupy the actor loop."""

import asyncio
import inspect
from copy import deepcopy
from typing import Any
from uuid import uuid4

import ray

from relax.engine.inference.discovery import new_manager_epoch
from relax.engine.inference.lifecycle import AdmissionLedger
from relax.engine.inference.placement import PlacementPlanner
from relax.engine.inference.types import LifecycleState, Role, RoleSnapshot, RoutingSpec
from relax.utils.logging_utils import get_logger


logger = get_logger(__name__)

# Explicit surface: private/workload methods cannot bypass the engine owner.
POOL_METHODS = frozenset(
    {
        "get_rollout_engines_and_lock",
        "get_weight_sync_lock",
        "get_router_address",
        "get_engines_info",
        "get_status",
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


@ray.remote(num_cpus=1, num_gpus=0, max_concurrency=4096)
class InferenceManager:
    def __init__(self) -> None:
        self.pool: Any = None
        self.placement = PlacementPlanner()
        self.ledger = AdmissionLedger()
        self.epoch = new_manager_epoch()
        self.revision = 0
        self._topology: Any = None
        self.state = LifecycleState.STARTING
        self.memory_ready = False
        self._transition = asyncio.Lock()
        self._updates: dict[str, str] = {}
        self._disposed = False
        self._closing = False
        self._cleanup_task: asyncio.Task | None = None
        self._active_calls: set[asyncio.Future] = set()
        self._recipients: set[str] = set()

    async def start(self, args: Any, binding: Any) -> dict[str, Any]:
        from relax.distributed.ray.inference_pool import RolloutPool, plan_rollout

        async with self._transition:
            if self._closing or self._disposed:
                raise RuntimeError("Inference manager is closed")
            if self.pool is None:
                await _blocking(plan_rollout, args, binding, self.placement)

                def create_pool() -> None:
                    # Publish ownership inside the operation, even if the RPC
                    # caller is cancelled before receiving the startup reply.
                    self.pool = RolloutPool(args, binding)

                await _blocking(create_pool)
                self._recipients = set(self.pool.synced_actor_ids)
                self.memory_ready = True
                self.state = LifecycleState.READY
            primary = next(iter(self.pool.backend.servers))
            return self.pool.backend.get_router_address(primary)

    async def snapshot(self, role: Role | str = Role.ROLLOUT) -> RoleSnapshot:
        if Role(role) is not Role.ROLLOUT or self.pool is None:
            raise RuntimeError("Inference role is not registered")
        ready = self.memory_ready and not self.ledger.fences
        models = self.pool.observe(ready=ready, state=self.state)
        topology = tuple(
            (
                model.model_id,
                model.router_url,
                tuple((item.engine_id, item.base_url, item.state) for item in model.replicas),
                tuple((kind, item.engine_id, item.base_url, item.state) for kind, item in model.pd_workers),
                model.issues,
                model.admission,
            )
            for model in models
        )
        if topology != self._topology:
            self.revision += 1
            self._topology = topology
        names = tuple(model.model_id for model in models)
        return RoleSnapshot(
            Role.ROLLOUT,
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
        deadline = asyncio.get_running_loop().time() + 120
        while True:
            if self._closing or self._disposed:
                raise RuntimeError("Inference manager is closing")
            if self.state is LifecycleState.DEAD:
                raise RuntimeError("Inference manager cannot restore engine residency")
            if asyncio.get_running_loop().time() >= deadline:
                raise TimeoutError("Inference model did not become ready within 120 seconds")
            if any(
                model.admission and (model_id is None or model.model_id == model_id)
                for model in (await self.snapshot(role)).models
            ):
                return
            await asyncio.sleep(0.05)

    async def complete_request(self, permit: str) -> None:
        await self.ledger.complete(permit)

    async def begin_weight_update(self, source: str) -> str:
        if self._closing or self._disposed or self.pool is None:
            raise RuntimeError("Inference manager is not running")
        token = uuid4().hex
        self._updates[token] = source
        await self.ledger.drain(token, timeout=120)
        return token

    async def end_weight_update(self, token: str) -> None:
        if token not in self._updates:
            raise ValueError("Unknown weight-update token")
        self._updates.pop(token)
        if self.pool is not None:
            self.pool.mark_synced(self._recipients)
        self.ledger.fences.discard(token)
        self.ledger.fences.discard("recovery")

    async def begin_rollout(self, rollout_id: int) -> None:
        if self.pool is None or self._closing or self._disposed:
            raise RuntimeError("Inference manager is not running")
        self.pool.backend.rollout_id = rollout_id
        await _blocking(self.pool.backend.health_monitoring_resume)

    async def inject_ci_fault(self) -> None:
        if self.pool is None or self._closing or self._disposed:
            raise RuntimeError("Inference manager is not running")
        await _blocking(self.pool.backend._try_ci_fault_injection)

    async def pool_call(self, method: str, *args: Any, **kwargs: Any) -> Any:
        releasing_weights = method == "set_weight_updating" and not (args[0] if args else kwargs["is_updating"])
        if self._closing and not releasing_weights and not method.startswith("cancel_"):
            raise RuntimeError("Inference manager is closing")
        completed = asyncio.get_running_loop().create_future()
        self._active_calls.add(completed)
        try:
            if method == "recover_rollout_engines":
                # Recovery cannot mutate engine residency during memory transitions.
                async with self._transition:
                    return await self._pool_call(method, *args, **kwargs)
            return await self._pool_call(method, *args, **kwargs)
        finally:
            self._active_calls.discard(completed)
            completed.set_result(None)

    async def _pool_call(self, method: str, *args: Any, **kwargs: Any) -> Any:
        if self.pool is None or self._disposed:
            raise RuntimeError("Inference manager is not running")
        backend = self.pool.backend
        if method in {"offload", "onload", "onload_weights", "onload_kv"}:
            async with self._transition:
                if method == "offload":
                    await self.ledger.drain("memory", timeout=120)
                    self.state = LifecycleState.DRAINING
                    try:
                        await _blocking(backend.offload)
                    except BaseException:
                        self.state = LifecycleState.DEAD
                        raise
                    self.memory_ready = False
                    self.state = LifecycleState.SLEEPING
                else:
                    # Close admission before even a partial memory transition.
                    await self.ledger.drain("memory", timeout=120)
                    self.state = LifecycleState.ONLOADING
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
                    await self.ledger.drain("fully_async", timeout=120)
                else:
                    await _blocking(backend.set_weight_updating, False)
                    self.pool.mark_synced(self._recipients)
                    self.ledger.fences.discard("fully_async")
                    self.ledger.fences.discard("recovery")
                return True
        if method not in POOL_METHODS:
            raise ValueError(f"Unsupported inference pool operation: {method}")
        if method == "recover_rollout_engines":
            await self.ledger.drain("recovery", timeout=120)
        result = await _blocking(getattr(backend, method), *args, **kwargs)
        if method in {"get_rollout_engines_and_lock", "recover_rollout_engines"}:
            self._recipients = {engine._actor_id.hex() for engine in result[0] if engine is not None}
            if method == "recover_rollout_engines" and self._recipients <= self.pool.synced_actor_ids:
                self.ledger.fences.discard("recovery")
        if method == "register_recovered_engines":
            self.pool.mark_synced(set(args[0] if args else kwargs["synced_actor_ids"]))
        return result

    async def dispose(self, worker: Any = None) -> None:
        self._closing = True
        self.ledger.fences.add("shutdown")
        if self._cleanup_task is None:
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
            try:
                if self.pool is not None:
                    # Abort residual backend work even if its HTTP caller disappeared.
                    engines = self.pool.backend.rollout_engines
                    if engines:
                        await _blocking(ray.get, [engine.abort_requests.remote() for engine in engines], timeout=10)
                if worker is not None:
                    await asyncio.wait_for(worker.close.remote(), timeout=30)
                await self.ledger.drain("shutdown", timeout=10)
            finally:
                try:
                    if self.pool is not None:
                        await _blocking(self.pool.dispose)
                        self.pool = None
                        self.state = LifecycleState.DEAD
                        self.memory_ready = False
                finally:
                    if worker is not None:
                        ray.kill(worker)


@ray.remote(num_cpus=0)
def dispose_managed_rollout(manager: Any, worker: Any) -> None:
    """Release the CPU owner after cleanup, including on global restarts."""
    try:
        ray.get(manager.dispose.remote(worker))
    finally:
        ray.kill(manager)


def create_managed_rollout(args: Any, pg: Any, data_source: Any, runtime_env: Any) -> tuple[Any, Any]:
    """Create a complete managed task, rolling back every partial startup."""
    from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

    from relax.core.node_group_affinity import require_control_plane_resource_on_node, with_control_plane_affinity
    from relax.distributed.ray.inference_handle import ManagedRolloutHandle
    from relax.distributed.ray.placement_group import _get_head_node_id
    from relax.distributed.ray.rollout_worker import RolloutWorker
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
    manager = InferenceManager.options(**options).remote()
    worker = None
    try:
        address = ray.get(manager.start.remote(deepcopy(args), pg))
        worker_args = deepcopy(args)
        worker_args.sglang_router_ip = address["router_ip"]
        worker_args.sglang_router_port = address["router_port"]
        worker_args._inference_manager_handle = manager
        worker = RolloutWorker.options(**options).remote(worker_args, data_source, manager)
        handle = ManagedRolloutHandle(manager, worker)
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
        return handle, count
    except BaseException:
        try:
            ray.get(manager.dispose.remote(worker), timeout=60)
        except Exception:
            logger.exception("Managed rollout startup rollback could not be confirmed")
        finally:
            if worker is not None:
                ray.kill(worker)
            ray.kill(manager)
        raise
