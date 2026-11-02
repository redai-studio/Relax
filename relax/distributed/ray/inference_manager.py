# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""The CPU owner of a role's inference resources.

Engine runtimes are in-process policy adapters, never independently scheduled
managers. All public mutation crosses this actor's lease validation boundary.
"""

from __future__ import annotations

import asyncio
import inspect
import time
import uuid
from dataclasses import dataclass, replace
from typing import Any

import ray

from relax.engine.inference.discovery import EngineState, RoleSnapshot, role_snapshot_from_payloads
from relax.engine.inference.holder import HolderIdentity, current_holder
from relax.engine.inference.ownership import ActivationLease, OwnershipError, ResourceBlockedError
from relax.engine.inference.phase_client import PhaseSession, await_control, current_phase, wait_control
from relax.utils.env import Envs
from relax.utils.logging_utils import get_logger


logger = get_logger(__name__)


_READ_METHODS = frozenset({"health_check", "is_onloaded", "get_status"})
_MEMORY_METHODS = frozenset({"onload", "offload", "onload_weights", "onload_kv", "shutdown", "dispose"})
_CONTROL_METHODS = frozenset(
    {
        "set_train_parallel_config",
        "set_weight_updating",
        "set_training_weight_updating",
        "health_monitoring_pause",
        "health_monitoring_resume",
        "cancel_scale_out",
        "cancel_all_scale_out_requests",
        "create_scale_out_request",
        "create_scale_in_request",
    }
)


async def _invoke(runtime: Any, method: str, *args: Any, **kwargs: Any) -> Any:
    function = getattr(runtime, method)

    def execute():
        if inspect.iscoroutinefunction(function):
            return asyncio.run(function(*args, **kwargs))
        return function(*args, **kwargs)

    return await asyncio.to_thread(execute)


@ray.remote(num_cpus=1, max_restarts=0)
class InferenceManager:
    def __init__(self, args: Any, role: str, pg: Any = None, owns_pg: bool = False) -> None:
        self.args, self.role, self.pg = args, role, pg
        self.authority = args._inference_authority
        self._owns_pg = owns_pg
        self._runtimes: dict[str, Any] = {}
        self._claims: dict[str, Any] = {}
        self._initializations: dict[str, asyncio.Task] = {}
        self._operations: dict[str, tuple[Any, asyncio.Task]] = {}
        self._mutations: set[asyncio.Task] = set()
        self._locks: dict[str, asyncio.Lock] = {}
        self._blocked: dict[str, str] = {}
        self._acceptance = asyncio.Lock()
        self._owned_resources = set()
        self._closing = False
        self._legacy_callers: set[HolderIdentity | None] = set()
        self._health_tasks: dict[str, asyncio.Task] = {}
        self._health_wakeups: dict[str, asyncio.Event] = {}

    def _own_mutation(self, coroutine: Any) -> asyncio.Task:
        """Keep accepted work alive independently of the RPC waiting for it."""
        task = asyncio.create_task(coroutine)
        self._mutations.add(task)

        def settled(done: asyncio.Task) -> None:
            self._mutations.discard(done)
            if not done.cancelled():
                done.exception()  # Observe failures even when the caller has left.

        task.add_done_callback(settled)
        return task

    async def add_model(self, name: str, args: Any, kwargs: dict[str, Any]) -> None:
        if self._closing:
            raise OwnershipError("InferenceManager is shutting down")
        if name not in self._initializations:
            self._initializations[name] = asyncio.create_task(self._initialize(name, args, kwargs))
        await asyncio.shield(self._initializations[name])

    async def claim_placement(self) -> None:
        await self.authority.register_manager.remote(self.role, ray.get_runtime_context().current_actor)
        if self.pg is not None:
            await self.authority.register_resource.remote(
                self.pg[0].id.hex(), f"inference:{self.role}" if self._owns_pg else "controller", self.pg[0]
            )

    async def _initialize(self, name: str, args: Any, kwargs: dict[str, Any]) -> None:
        claims = await self.authority.claims_for.remote(self.role)
        claims = tuple(c for c in claims if c.role == self.role and c.model == name)
        if not claims:
            raise OwnershipError(f"No placement claim for {self.role}/{name}")
        self._claims[name] = claims
        self._locks[name] = asyncio.Lock()
        async with PhaseSession(self.authority, self.role, f"initialize:{self.role}/{name}") as session:
            if self.pg is not None:
                pg = self.pg[0]
                await self.authority.register_resource.remote(
                    pg.id.hex(),
                    f"inference:{self.role}" if self._owns_pg else "controller",
                    pg,
                    borrowed=False,
                )

            def build() -> Any:
                if self.role == "rollout":
                    from relax.distributed.ray.rollout import RolloutEngineRuntime

                    cls = RolloutEngineRuntime
                elif self.role == "genrm":
                    from relax.distributed.ray.genrm import GenRMEngineRuntime

                    cls = GenRMEngineRuntime
                else:
                    from relax.distributed.ray.teacher_manager import TeacherEngineRuntime

                    cls = TeacherEngineRuntime
                runtime = cls.__new__(cls)
                self._runtimes[name] = runtime
                if self.role == "teacher":
                    cls.__init__(runtime, args, pg=self.pg, **kwargs)
                else:
                    cls.__init__(runtime, args, self.pg, **kwargs)
                return runtime

            handle = ray.get_runtime_context().current_actor
            for claim in claims:
                await self.authority.register_model.remote(self.role, name, handle, claim)
            await self.authority.reserve_activation.remote(session.lease, self.role, name)
            try:
                runtime = await asyncio.to_thread(build)
                runtime.owner_handle = ray.get_runtime_context().current_actor
                if self.role == "rollout" and getattr(args, "check_weight_update_equal", False):
                    await _invoke(runtime, "check_weights", action="snapshot")
                    await _invoke(runtime, "check_weights", action="reset_tensors")
                if getattr(args, "offload_rollout", False) and (
                    self.role != "teacher" or kwargs.get("shared_pg", False)
                ):
                    await self.apply(name, "offload", session.lease, "initialize:offload")
                elif self.role != "rollout":
                    # Rollout replacements also need training weight publication;
                    # its existing health/recovery path owns that sequencing.
                    self._health_wakeups[name] = asyncio.Event()
                    self._health_tasks[name] = asyncio.create_task(self._health_loop(name))
            except BaseException as error:
                # Keep partially constructed runtimes and their slots reachable.
                self._blocked[name] = f"Initialization did not complete: {error}"
                runtime = self._runtimes.get(name)
                if runtime is not None:
                    try:
                        await _invoke(runtime, "dispose" if self.role == "rollout" else "shutdown")
                    except BaseException:
                        pass  # PhaseSession quarantines the original claims.
                    else:
                        await self.authority.confirm_deactivation.remote(session.lease, self.role, name)
                        session._safe_to_release = True
                raise
            finally:
                await self._record_resources(name)

    def report_failure(self, name: str, endpoint: str | None = None) -> None:
        """A hint wakes a probe; it is never evidence to kill a generation."""
        wakeup = self._health_wakeups.get(name)
        if wakeup is not None and not self._closing:
            wakeup.set()

    async def _health_loop(self, name: str) -> None:
        wakeup = self._health_wakeups[name]
        attempts, next_attempt = 0, 0.0
        while not self._closing:
            try:
                await asyncio.wait_for(wakeup.wait(), timeout=max(0.01, Envs.RELAX_INFERENCE_HEALTH_INTERVAL_S))
            except TimeoutError:
                pass
            wakeup.clear()
            # Request bursts cannot bypass the backoff or launch a second task.
            if self._closing or time.monotonic() < next_attempt:
                continue
            try:
                async with self._locks[name]:
                    runtime = self._runtimes[name]
                    if self._closing or name in self._blocked or not runtime.is_onloaded():
                        continue
                    if await _invoke(runtime, "health_check"):
                        attempts = 0
                        continue
                    if not await self.authority.reserve_maintenance.remote(self.role, name):
                        continue
                    try:
                        await _invoke(runtime, "recover")
                    finally:
                        await self._record_resources(name)
                    if not await _invoke(runtime, "health_check"):
                        raise RuntimeError("Engine health is still unconfirmed after recovery")
                    attempts = 0
                    # Publish only the completed, healthy incarnation. This never
                    # opens a gate closed by a lifecycle operation.
                    self.get_model_snapshot(name)
            except Exception as error:
                attempts += 1
                delay = Envs.RELAX_INFERENCE_RECOVERY_BACKOFF_S * 2 ** (attempts - 1)
                if attempts >= max(1, Envs.RELAX_INFERENCE_RECOVERY_ATTEMPTS):
                    delay = max(delay, Envs.RELAX_INFERENCE_RECOVERY_COOLDOWN_S)
                    attempts = 0
                next_attempt = time.monotonic() + delay
                logger.warning("%s/%s recovery deferred for %.1fs: %s", self.role, name, delay, error)

    async def _record_resources(self, name: str) -> None:
        runtime = self._runtimes.get(name)
        pool = getattr(runtime, "_pool", None)
        owned = [pg[0] for pg, owns in getattr(pool, "placements", {}).values() if owns]
        owned.extend(getattr(runtime, "_owned_elastic_pgs", {}).values())
        for pg in owned:
            key = pg.id.hex()
            await self.authority.register_resource.remote(key, f"inference:{self.role}", pg)
            self._owned_resources.add(key)

    async def _validate(self, name: str, lease: ActivationLease | None, *, cleanup: bool = False) -> None:
        if self._closing and not cleanup:
            raise OwnershipError("InferenceManager is shutting down")
        if lease is None:
            raise OwnershipError("Lifecycle and resource mutations require an activation lease")
        await self.authority.check.remote(lease, self._claims[name], allow_blocked=cleanup)
        if name in self._blocked and not cleanup:
            raise ResourceBlockedError(self._blocked[name])

    async def apply(
        self,
        name: str,
        action: str,
        lease: ActivationLease,
        command_id: str,
        *,
        tags: list[str] | None = None,
        parent_command: str | None = None,
    ) -> Any:
        key = f"{lease.operation_id if lease else 'missing'}:{name}:{command_id}"
        signature = (lease, action, tags)
        async with self._acceptance:
            old = self._operations.get(key)
            if old is not None:
                if old[0] != signature:
                    raise OwnershipError("Conflicting retry of an inference mutation")
                task = old[1]
            else:
                cleanup = action in {"offload", "shutdown", "dispose"}
                await self._validate(name, lease, cleanup=cleanup)
                await self.authority.begin.remote(lease, key, parent_command)

                async def execute() -> Any:
                    try:
                        async with self._locks[name]:
                            await self._validate(name, lease, cleanup=cleanup)
                            runtime = self._runtimes[name]
                            method = "dispose" if action == "shutdown" and self.role == "rollout" else action
                            if method == "onload":
                                await self.authority.reserve_activation.remote(lease, self.role, name)
                            result = await _invoke(runtime, method, **({"tags": tags} if method == "onload" else {}))
                            if cleanup:
                                await self.authority.confirm_deactivation.remote(lease, self.role, name)
                                self._blocked.pop(name, None)
                            return result
                    except BaseException as error:
                        gate = getattr(self._runtimes.get(name), "_admission", None)
                        if gate is not None:
                            gate.close()
                        self._blocked[name] = f"{self.role}/{name} {action} was not confirmed: {error}"
                        await self.authority.block.remote(lease, self._blocked[name])
                        raise
                    finally:
                        try:
                            await self._record_resources(name)
                        finally:
                            await self.authority.finish.remote(lease, key)

                task = self._own_mutation(execute())
                self._operations[key] = (signature, task)
        return await asyncio.shield(task)

    async def legacy_apply(self, name: str, action: str, command_id: str, holder: HolderIdentity, *, tags=None) -> Any:
        if holder not in self._legacy_callers:
            self._legacy_callers.add(holder)
            logger.warning("Deprecated lifecycle call without an explicit lease from %s; use PhaseSession", holder)

        async def execute() -> Any:
            token = uuid.uuid4().hex
            lease, owned = await self.authority.resolve_legacy.remote(self.role, name, token, holder)
            try:
                return await self.apply(name, action, lease, command_id, tags=tags)
            finally:
                await self.authority.finish.remote(lease, token)
                if owned:
                    await self.authority.release.remote(lease)

        return await asyncio.shield(asyncio.create_task(execute()))

    async def call(self, name: str, method: str, args: tuple = (), kwargs: dict | None = None, lease=None) -> Any:
        kwargs = kwargs or {}
        if method.startswith("_") or method in _MEMORY_METHODS:
            raise OwnershipError("Use the leased lifecycle interface for this operation")
        read = method.startswith(("get_", "list_")) or method in _READ_METHODS or method in _CONTROL_METHODS
        if not read:
            if lease is None or method in {
                "recover",
                "recover_rollout_engines",
                "scale_out",
                "scale_in",
                "register_recovered_engines",
            }:
                return await self.maintenance(name, method, args, kwargs, lease=lease)
            await self._validate(name, lease)
        if read:
            if method == "health_check":
                # Probe results update slot observations. A probe started for
                # the old generation must settle before retirement/replacement.
                async def probe() -> Any:
                    async with self._locks[name]:
                        return await _invoke(self._runtimes[name], method, *args, **kwargs)

                async with self._acceptance:
                    if self._closing:
                        raise OwnershipError("InferenceManager is shutting down")
                    task = self._own_mutation(probe())
                return await asyncio.shield(task)
            return await _invoke(self._runtimes[name], method, *args, **kwargs)
        token = uuid.uuid4().hex

        async def execute() -> Any:
            await self.authority.begin.remote(lease, token)
            try:
                async with self._locks[name]:
                    await self._validate(name, lease)
                    return await _invoke(self._runtimes[name], method, *args, **kwargs)
            except BaseException as error:
                self._blocked[name] = f"{method} failed: {error}"
                self._runtimes[name]._admission.close()
                await self.authority.block.remote(lease, self._blocked[name])
                raise
            finally:
                try:
                    await self._record_resources(name)
                finally:
                    await self.authority.finish.remote(lease, token)

        async with self._acceptance:
            if self._closing:
                raise OwnershipError("InferenceManager is shutting down")
            task = self._own_mutation(execute())
        return await asyncio.shield(task)

    async def maintenance(
        self, name: str, method: str, args: tuple = (), kwargs: dict | None = None, *, lease=None
    ) -> Any:
        if method.startswith("_") or method in _MEMORY_METHODS:
            raise OwnershipError("Memory operations require the caller's explicit phase lease")

        async def execute() -> Any:
            token = uuid.uuid4().hex
            if lease is not None:
                await self._validate(name, lease)
                await self.authority.begin.remote(lease, token)
            async with self._locks[name]:
                try:
                    if lease is not None:
                        await self._validate(name, lease)
                    if method not in {"scale_in", "remove_engine"}:
                        allowed = await self.authority.reserve_maintenance.remote(
                            self.role, name, **({"lease": lease} if lease is not None else {})
                        )
                        if not allowed:
                            # Offloaded shared engines are rebuilt at the next onload.
                            if method in {"recover", "recover_rollout_engines", "register_recovered_engines"}:
                                return None
                            raise OwnershipError(f"{self.role}/{name} {method} deferred until its next onload")
                    try:
                        return await _invoke(self._runtimes[name], method, *args, **(kwargs or {}))
                    except BaseException as error:
                        self._blocked[name] = f"{method} was not confirmed: {error}"
                        self._runtimes[name]._admission.close()
                        if lease is not None:
                            await self.authority.block.remote(lease, self._blocked[name])
                        raise
                finally:
                    try:
                        await self._record_resources(name)
                    finally:
                        if lease is not None:
                            await self.authority.finish.remote(lease, token)

        async with self._acceptance:
            if self._closing:
                return None
            task = self._own_mutation(execute())
        return await asyncio.shield(task)

    async def acquire_inference(self, name: str, request_id: str, **selection: Any) -> Any:
        initialization = self._initializations.get(name)
        starting = initialization is not None and not initialization.done()
        if self._closing or name in self._blocked or starting:
            from relax.engine.inference.routing import ModelUnavailableError

            reason = "Model initialization is pending" if starting else "InferenceManager is shutting down"
            raise ModelUnavailableError(self._blocked.get(name, reason))
        return self._runtimes[name].acquire_inference(request_id, **selection)

    async def release_inference(self, name: str, permit: Any) -> None:
        self._runtimes[name].release_inference(permit)

    async def reconcile_dead_holder(self, name: str, lease: ActivationLease) -> bool:
        async def execute() -> bool:
            async with self._locks[name]:
                await self._validate(name, lease, cleanup=True)
                await _invoke(self._runtimes[name], "offload")
                await self.authority.confirm_deactivation.remote(lease, self.role, name)
                self._blocked.pop(name, None)
                return True

        async with self._acceptance:
            task = self._own_mutation(execute())
        return await asyncio.shield(task)

    def runtime_config(self, name: str) -> dict[str, Any]:
        runtime = self._runtimes[name]
        return {key: getattr(runtime.args, key, None) for key in ("sglang_router_ip", "sglang_router_port")}

    def get_model_snapshot(self, name: str) -> dict:
        runtime = self._runtimes.get(name)
        initialization = self._initializations.get(name)
        starting = runtime is None or (initialization is not None and not initialization.done())
        failed = initialization is not None and initialization.done() and initialization.exception() is not None
        if starting or failed:
            payload = {"state": "blocked" if failed else "starting", "engines": [], "topology_revision": 0}
            return (
                role_snapshot_from_payloads(self.role, {name: payload}).to_dict()
                if self.role == "rollout"
                else payload
            )
        snapshot = runtime.get_inference_snapshot()
        if name in self._blocked:
            if self.role == "rollout":
                current = RoleSnapshot.from_dict(snapshot)
                return replace(
                    current,
                    models=tuple(
                        replace(
                            m,
                            state=EngineState.BLOCKED,
                            engines=tuple(
                                replace(e, state=EngineState.BLOCKED, direct_eligible=False) for e in m.engines
                            ),
                        )
                        for m in current.models
                    ),
                ).to_dict()
            snapshot["state"] = "blocked"
            for engine in snapshot["engines"]:
                engine["state"] = "blocked"
        return snapshot

    def get_inference_snapshot(self) -> dict:
        if self.role == "rollout":
            return self.get_model_snapshot("__default__")
        return role_snapshot_from_payloads(
            self.role, {key: self.get_model_snapshot(key) for key in self._claims}
        ).to_dict()

    async def shutdown(self) -> None:
        async with self._acceptance:
            self._closing = True
        for wakeup in self._health_wakeups.values():
            wakeup.set()
        await await_control(
            asyncio.gather(*(asyncio.shield(task) for task in self._health_tasks.values()), return_exceptions=True),
            f"{self.role} health recovery before shutdown",
        )
        # Initialization can still be allocating after its caller was cancelled.
        # Keep the PG until every accepted initialization has settled.
        await await_control(
            asyncio.gather(*(asyncio.shield(task) for task in self._initializations.values()), return_exceptions=True),
            f"{self.role} accepted initializations before shutdown",
        )
        await await_control(
            asyncio.gather(*(asyncio.shield(task) for task in self._mutations), return_exceptions=True),
            f"{self.role} accepted mutations before shutdown",
        )
        if not self._runtimes:
            for resource in self._owned_resources:
                await self.authority.confirm_resource_release.remote(resource, f"inference:{self.role}")
            if self._owns_pg and self.pg is not None:
                from ray.util.placement_group import remove_placement_group

                remove_placement_group(self.pg[0])
                await self.authority.confirm_resource_release.remote(self.pg[0].id.hex(), f"inference:{self.role}")
                self.pg = None
            await self.authority.confirm_manager_shutdown.remote(self.role)
            return
        lease = await self.authority.acquire_for_shutdown.remote(self.role)
        try:
            for name in self._runtimes:
                await self.apply(name, "shutdown", lease, "shutdown")
            for resource in self._owned_resources:
                await self.authority.confirm_resource_release.remote(resource, f"inference:{self.role}")
            if self._owns_pg and self.pg is not None:
                from ray.util.placement_group import remove_placement_group

                remove_placement_group(self.pg[0])
                await self.authority.confirm_resource_release.remote(self.pg[0].id.hex(), f"inference:{self.role}")
                self.pg = None
            await self.authority.confirm_manager_shutdown.remote(self.role)
        finally:
            await self.authority.release.remote(lease)


@dataclass(frozen=True)
class ModelManager:
    """Serializable, model-scoped view of the common actor; owns no
    resources."""

    handle: Any
    model: str

    def __getattr__(self, method: str) -> Any:
        if method.startswith("__"):
            raise AttributeError(method)
        return _RemoteMethod(self, method)


@dataclass(frozen=True)
class _RemoteMethod:
    manager: ModelManager
    method: str

    def remote(self, *args: Any, **kwargs: Any) -> Any:
        handle, model, method = self.manager.handle, self.manager.model, self.method
        if method == "get_inference_snapshot":
            return handle.get_model_snapshot.remote(model)
        if method in {"acquire_inference", "release_inference", "report_failure"}:
            return getattr(handle, method).remote(model, *args, **kwargs)
        lease = kwargs.pop("lease", None)
        command = kwargs.pop("command_id", None)
        if lease is None:
            try:
                lease = current_phase().lease
            except OwnershipError:
                pass
        if method in _MEMORY_METHODS:
            tags = kwargs.get("tags", args[0] if args else None)
            if method in {"onload_weights", "onload_kv"}:
                tags = ["weights"] if method == "onload_weights" else ["kv_cache", "cuda_graph"]
                method = "onload"
            if lease is None:
                return handle.legacy_apply.remote(
                    model, method, command or uuid.uuid4().hex, current_holder(), tags=tags
                )
            return handle.apply.remote(model, method, lease, command or uuid.uuid4().hex, tags=tags)
        return handle.call.remote(model, method, args, kwargs, lease)


def create_model_manager(
    args: Any,
    role: str,
    model: str,
    pg: Any = None,
    *,
    runtime_env: dict | None = None,
    owns_pg: bool = False,
    **kwargs: Any,
) -> ModelManager:
    manager = ensure_inference_manager(args, role, pg, runtime_env=runtime_env, owns_pg=owns_pg)
    wait_control(manager.add_model.remote(model, args, kwargs), f"{role}/{model} initialization")
    return ModelManager(manager, model)


def ensure_inference_manager(args: Any, role: str, pg: Any = None, *, runtime_env=None, owns_pg=False) -> Any:
    from relax.core.node_group_affinity import with_control_plane_affinity

    options = with_control_plane_affinity(
        args,
        {
            "name": f"relax_inference_{role}",
            "get_if_exists": True,
            "runtime_env": runtime_env,
            "num_cpus": 1,
            "num_gpus": 0,
        },
    )
    if role == "rollout":
        from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

        from relax.distributed.ray.placement_group import _get_head_node_id

        options["scheduling_strategy"] = NodeAffinitySchedulingStrategy(_get_head_node_id(), soft=False)
    manager = InferenceManager.options(**options).remote(args, role, pg, owns_pg)
    wait_control(manager.claim_placement.remote(), f"{role} placement registration")
    return manager


@ray.remote(num_cpus=0)
class LegacyInferenceManager:
    """Named lookup compatibility.

    Old callers borrow a covering lease or receive a one-shot lease.
    """

    def __init__(self, manager: ModelManager) -> None:
        self.manager = manager

    def bind(self, manager: ModelManager) -> None:
        self.manager = manager

    async def onload(self, tags=None, *, lease=None, command_id=None):
        return await self.manager.onload.remote(tags=tags, lease=lease, command_id=command_id)

    async def offload(self, *, lease=None, command_id=None):
        return await self.manager.offload.remote(lease=lease, command_id=command_id)

    async def shutdown(self, *, lease=None, command_id=None):
        return await self.manager.shutdown.remote(lease=lease, command_id=command_id)

    async def get_inference_snapshot(self):
        return await self.manager.get_inference_snapshot.remote()

    async def get_engine_hosts_ports(self):
        return await self.manager.get_engine_hosts_ports.remote()

    async def get_urls(self):
        return await self.manager.get_urls.remote()

    async def health_check(self):
        return await self.manager.health_check.remote()


_legacy_facades: dict[str, Any] = {}


def register_legacy_name(name: str, manager: ModelManager) -> None:
    # A name is not a lifetime reference in Ray. Keep the facade alive with its
    # job owner, and rebind it when that owner installs a new resource epoch.
    facade = LegacyInferenceManager.options(name=name, get_if_exists=True).remote(manager)
    _legacy_facades[name] = facade
    wait_control(facade.bind.remote(manager), f"legacy facade {name} registration")
