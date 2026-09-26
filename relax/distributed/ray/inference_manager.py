# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""The task-scoped CPU control plane for every inference role.

One ``InferenceManager`` per training task owns the engine pools of Rollout,
GenRM and Teacher, their discovery state, request admission, lifecycle and
placement ledger. It runs as one CPU Ray actor on the head node; the engines it
creates live in that actor's process. Role Gateways read discovery and admit
requests through it.

The Manager's lifecycle operations are ``activate``, ``drain``, ``deactivate``
and ``shutdown``, all idempotent. When roles take turns on shared GPUs is
decided by the :class:`~relax.engine.inference.lifecycle.LifecycleCoordinator`
running in the same process; training and scoring code reach it through
``enter_phase``/``leave_phase`` and ``rollout_released``.

An engine pool is a :class:`~relax.distributed.ray.rollout.RolloutServer`: it
exposes ``onload``/``offload``/``recover``/``health_check``/``shutdown`` and
reports its state with ``observe``. The Manager serializes lifecycle calls per
model and is the only writer of the published snapshots.
"""

import asyncio
import time
from collections.abc import Sequence
from copy import deepcopy
from dataclasses import dataclass, field, replace
from threading import Condition, RLock
from typing import Any
from uuid import uuid4

import ray

from relax.core.node_group_affinity import with_control_plane_affinity
from relax.engine.inference.config import DeploymentSpec, InferenceModelSpec, InferenceRoleSpec
from relax.engine.inference.discovery import new_manager_epoch
from relax.engine.inference.lifecycle import SWITCH_DRAIN_TIMEOUT_S, LifecycleCoordinator
from relax.engine.inference.phase_plans import PHASE_GENERATE, reject_shared_co_resident
from relax.engine.inference.placement import (
    ModelPlacement,
    PlacementGroupView,
    PlacementOwner,
    PlacementPlanner,
    PlacementRequest,
    PlacementSlice,
)
from relax.engine.inference.types import (
    WORKLOAD_BY_ROLE,
    DeploymentMode,
    LifecycleState,
    ModelSnapshot,
    Role,
    RoleSnapshot,
    RoutingSpec,
)
from relax.utils.logging_utils import get_logger


logger = get_logger(__name__)

_LIFECYCLE_METHODS = frozenset({"activate", "deactivate", "recover", "health_check", "shutdown"})
_QUERY_METHODS = frozenset({"get_urls", "get_engine_hosts_ports"})


@dataclass(frozen=True)
class ModelHandle:
    """A picklable handle to one model on the manager.

    Shaped like a per-model actor handle, so training and service code call
    ``handle.activate.remote()`` / ``handle.deactivate.remote()``; every call
    runs serialized on the manager through :meth:`InferenceManager.call`.
    """

    manager: Any
    role: Role
    model_id: str

    def __getattr__(self, method: str) -> Any:
        if method.startswith("_"):
            raise AttributeError(method)
        return _RemoteCall(self, method)


@dataclass(frozen=True)
class _RemoteCall:
    handle: ModelHandle
    method: str

    def remote(self, *args: Any, **kwargs: Any) -> Any:
        return self.handle.manager.call.remote(self.handle.role, self.handle.model_id, self.method, *args, **kwargs)


@dataclass
class _Model:
    config: InferenceModelSpec
    pool: Any
    snapshot: ModelSnapshot
    lock: RLock = field(default_factory=RLock)
    # Whether the model may hold GPU memory. Lifecycle state cannot tell:
    # DEAD is published before a failed offload or a shutdown has released
    # anything, and a weights-only onload reports SLEEPING. Set before every
    # load and cleared only once an offload or shutdown has succeeded.
    resident: bool = True


class InferenceManager:
    """Owner of every inference model of one task; see the module docstring."""

    def __init__(self) -> None:
        self.manager_epoch = new_manager_epoch()
        self.placement = PlacementPlanner()
        self._lock = RLock()
        self._requests_done = Condition(self._lock)
        self._models: dict[Role, dict[str, _Model]] = {}
        self._specs: dict[Role, InferenceRoleSpec] = {}
        self._revision: dict[Role, int] = {}
        self._inflight: dict[str, tuple[Role, str]] = {}
        self._rollout_pool: Any = None
        self._rollout_started = False
        # Roles registered only with the models a failed ``create_role`` could
        # not stop; creating the role again stops them first.
        self._failed_starts: set[Role] = set()
        self.lifecycle = LifecycleCoordinator(self)

    # ------------------------------------------------------------------
    # Registration and discovery.
    # ------------------------------------------------------------------

    def register(self, role: Role | str, pools: dict[str, Any], *, observe: bool = True) -> tuple[str, ...]:
        """Register a role's started pools and route each model by its own ID;
        ``observe=False`` skips asking the pools for their state."""
        role = Role(role)
        with self._lock:
            if role in self._models:
                raise ValueError(f"Inference role is already registered: {role.value}")
            models = {
                model_id: _Model(deepcopy(pool.model_spec), pool, ModelSnapshot(model_id))
                for model_id, pool in pools.items()
            }
            names = tuple(pools)
            self._specs[role] = InferenceRoleSpec(
                role=role,
                workload=WORKLOAD_BY_ROLE[role],
                deployment=self._deployment(role),
                routing=RoutingSpec(
                    default_model=names[0] if len(names) == 1 else None,
                    route_key_to_model=tuple((name, name) for name in names),
                ),
                models=tuple(model.config for model in models.values()),
            )
            self._models[role] = models
            self._revision[role] = 1
        if observe:
            for model_id in names:
                self._observe(role, model_id)
        return names

    def roles(self) -> tuple[str, ...]:
        return tuple(role.value for role in self._models)

    def registered_roles(self) -> tuple[Role, ...]:
        with self._lock:
            return tuple(self._models)

    def role_spec(self, role: Role | str) -> InferenceRoleSpec:
        """A role's spec; its deployment follows the current placement
        ledger."""
        role = Role(role)
        with self._lock:
            if role not in self._specs:
                raise RuntimeError(f"Inference role is not registered: {role.value}")
            return replace(self._specs[role], deployment=self._deployment(role))

    def model_placement(self, role: Role | str, model_id: str) -> ModelPlacement:
        """Where one model's startup engine groups run."""
        prefix = f"{Role(role).value}/{model_id}/"
        slices = tuple(item for item in self.placement.allocations() if item.group_id.startswith(prefix))
        return ModelPlacement.from_slices(slices, shared_phase=PHASE_GENERATE)

    def model_ids(self, role: Role | str) -> tuple[str, ...]:
        return tuple(self._models.get(Role(role), {}))

    def model_spec(self, role: Role | str, model_id: str) -> InferenceModelSpec:
        return deepcopy(self._model(role, model_id).config)

    def snapshot(self, role: Role | str) -> RoleSnapshot:
        role = Role(role)
        with self._lock:
            if role not in self._models:
                raise RuntimeError(f"Inference role is not registered: {role.value}")
            spec = self._specs[role]
            models = tuple(model.snapshot for model in self._models[role].values())
            return RoleSnapshot(
                role=role,
                manager_epoch=self.manager_epoch,
                topology_revision=self._revision[role],
                phase=self.lifecycle.current_phase(PHASE_GENERATE),
                models=models,
                routing=spec.routing,
            )

    def publish(self, role: Role | str, model: ModelSnapshot) -> None:
        """Commit one model's observation after checking its invariants."""
        role = Role(role)
        with self._lock:
            entry = self._model(role, model.model_id)
            spec = entry.config
            # Direct eligibility is derived, never reported: only a READY
            # replica of a direct-routed model may take requests without the
            # Router, so sleeping, draining and restarting replicas drop out.
            model = replace(
                model,
                replicas=tuple(
                    replace(
                        replica,
                        direct_eligible=not spec.needs_router
                        and replica.state == LifecycleState.READY
                        and bool(replica.base_url),
                    )
                    for replica in model.replicas
                ),
                pd_workers=tuple((kind, replace(worker, direct_eligible=False)) for kind, worker in model.pd_workers),
            )
            if model.admission and model.state != LifecycleState.READY:
                raise ValueError("Only READY models may admit requests")
            if model.state == LifecycleState.READY:
                ready = [r for r in model.replicas if r.state == LifecycleState.READY and r.base_url]
                if spec.needs_router and not model.router_url:
                    raise ValueError("READY requires the model's Router")
                if not spec.needs_router and model.router_url:
                    raise ValueError("A direct-routed model cannot publish a Router")
                if not ready:
                    raise ValueError("READY requires a healthy logical replica")
                if spec.needs_weight_update and (
                    model.required_weight_version is None
                    or any(r.weight_version != model.required_weight_version for r in ready)
                ):
                    raise ValueError("READY policy replicas must have the required weight version")
            if _topology(model) != _topology(entry.snapshot) or _restarted(entry.snapshot, model):
                self._revision[role] += 1
            entry.snapshot = model

    def set_state(self, role: Role | str, model_id: str, state: LifecycleState) -> None:
        """Close admission and move the model and its replicas to ``state``."""
        if state == LifecycleState.READY:
            raise ValueError("READY is only published from an observation")
        with self._lock:
            current = self._model(role, model_id).snapshot
            self.publish(
                role,
                replace(
                    current,
                    state=state,
                    admission=False,
                    replicas=tuple(replace(replica, state=state) for replica in current.replicas),
                    pd_workers=tuple((kind, replace(worker, state=state)) for kind, worker in current.pd_workers),
                ),
            )

    def _model(self, role: Role | str, model_id: str) -> _Model:
        models = self._models.get(Role(role))
        if models is None or model_id not in models:
            raise KeyError(f"Unknown inference model: {Role(role).value}/{model_id}")
        return models[model_id]

    def _observe(self, role: Role, model_id: str) -> None:
        self.publish(role, self._model(role, model_id).pool.observe())

    # ------------------------------------------------------------------
    # Request admission.
    # ------------------------------------------------------------------

    def admit_request(self, role: Role | str, model_id: str, request_id: str | None = None) -> str:
        """Record an in-flight request on a READY model, or refuse it."""
        role = Role(role)
        with self._lock:
            model = self._model(role, model_id).snapshot
            if not model.admission or model.state != LifecycleState.READY:
                raise RuntimeError(f"Model {role.value}/{model_id} is not ready for inference")
            request_id = request_id or uuid4().hex
            self._inflight[request_id] = (role, model_id)
            return request_id

    def complete_request(self, request_id: str) -> None:
        """Forget a finished, failed or aborted request."""
        with self._requests_done:
            if self._inflight.pop(request_id, None) is not None:
                self._requests_done.notify_all()

    def drain(self, roles: Sequence[Role | str], timeout: float = SWITCH_DRAIN_TIMEOUT_S) -> None:
        """Close admission for ``roles`` and wait for their requests to end."""
        roles = {Role(role) for role in roles}
        for role in roles:
            for model_id in self.model_ids(role):
                self.set_state(role, model_id, LifecycleState.DRAINING)
        deadline = time.monotonic() + timeout
        with self._requests_done:
            while any(role in roles for role, _ in self._inflight.values()):
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(
                        f"Inference requests still in flight after {timeout}s: {sorted(self._inflight)}"
                    )
                self._requests_done.wait(remaining)

    # ------------------------------------------------------------------
    # Lifecycle.
    # ------------------------------------------------------------------

    def call(self, role: Role | str, model_id: str, method: str, /, *args: Any, **kwargs: Any) -> Any:
        """Run a lifecycle operation or a pool query on one model."""
        role = Role(role)
        if method in _LIFECYCLE_METHODS:
            return getattr(self, method)(role, model_id, *args, **kwargs)
        if method not in _QUERY_METHODS:
            raise ValueError(f"Unsupported pool method: {method}")
        return getattr(self._model(role, model_id).pool, method)(*args, **kwargs)

    def activate(self, role: Role | str, model_id: str | None = None, tags: list[str] | None = None) -> None:
        """Load a role's or one model's memory once no conflicting role holds
        its GPUs.

        ``tags`` loads part of the memory (weights, KV cache). Activating a
        model that is already READY does nothing. If a load fails, the models
        this call started loading are released again; models that were already
        resident are left alone.
        """
        role = Role(role)
        with self.lifecycle.lease([role]):
            model_ids = self.model_ids(role) if model_id is None else (model_id,)
            loading = [name for name in model_ids if not self._model(role, name).resident]
            try:
                if model_id is not None:
                    self._activate_model(role, model_id, tags)
                elif role is Role.ROLLOUT and self._rollout_pool is not None:
                    # The rollout pool tracks its own weight/KV status around the
                    # per-model loads.
                    self._rollout_pool.onload_local(tags)
                else:
                    for name in model_ids:
                        self._activate_model(role, name, tags)
            except Exception:
                for name in loading:
                    try:
                        self._deactivate_model(role, name)
                    except Exception as exc:
                        logger.warning(f"Failed to release {role.value}/{name} after a failed load: {exc}")
                raise

    def deactivate(self, role: Role | str, model_id: str | None = None) -> None:
        """Release a role's or one model's memory; a released model is left
        alone."""
        role = Role(role)
        try:
            if model_id is not None:
                self._deactivate_model(role, model_id)
            elif role is Role.ROLLOUT and self._rollout_pool is not None:
                self._rollout_pool.offload_local()
            else:
                for name in self.model_ids(role):
                    self._deactivate_model(role, name)
        finally:
            # A load waiting for these GPUs re-checks now.
            self.lifecycle.notify_released()

    def _activate_model(self, role: Role, model_id: str, tags: list[str] | None) -> None:
        entry = self._model(role, model_id)
        with entry.lock:
            if entry.resident and entry.snapshot.state == LifecycleState.READY:
                return
            entry.resident = True
            self.set_state(role, model_id, LifecycleState.ONLOADING)
            try:
                entry.pool.onload(tags)
            finally:
                self._observe(role, model_id)

    def _deactivate_model(self, role: Role, model_id: str) -> None:
        entry = self._model(role, model_id)
        with entry.lock:
            if not entry.resident:
                return
            self.set_state(role, model_id, LifecycleState.DRAINING)
            try:
                entry.pool.offload()
            except Exception:
                self.set_state(role, model_id, LifecycleState.DEAD)
                raise
            entry.resident = False
            self.set_state(role, model_id, LifecycleState.SLEEPING)

    def recover(self, role: Role | str, model_id: str) -> None:
        """Restart dead engines; they allocate memory, so this waits like a
        load."""
        role = Role(role)
        with self.lifecycle.lease([role]):
            entry = self._model(role, model_id)
            with entry.lock:
                entry.resident = True
                self.set_state(role, model_id, LifecycleState.STARTING)
                try:
                    entry.pool.recover()
                finally:
                    self._observe(role, model_id)

    def health_check(self, role: Role | str, model_id: str) -> bool:
        entry = self._model(role, model_id)
        with entry.lock:
            healthy = entry.pool.health_check()
            self._observe(Role(role), model_id)
            return healthy

    def shutdown(self, role: Role | str, model_id: str | None = None) -> None:
        """Stop a whole role, or one of its models; a stopped model is left
        alone."""
        role = Role(role)
        if model_id is None:
            self._shutdown_role(role)
            return
        entry = self._model(role, model_id)
        with entry.lock:
            if not entry.resident and entry.snapshot.state == LifecycleState.DEAD:
                return
            self.set_state(role, model_id, LifecycleState.DEAD)
            try:
                entry.pool.shutdown(self.placement)
                entry.resident = False
            finally:
                # Stopping engines reports a topology change; the model stays DEAD.
                self.set_state(role, model_id, LifecycleState.DEAD)
        self.lifecycle.notify_released()

    # ------------------------------------------------------------------
    # Lifecycle coordination; the coordinator runs in this process.
    # ------------------------------------------------------------------

    def enter_phase(self, phase_id: str, timeout: float = SWITCH_DRAIN_TIMEOUT_S) -> None:
        self.lifecycle.enter_phase(phase_id, timeout)

    def leave_phase(self, phase_id: str, timeout: float = SWITCH_DRAIN_TIMEOUT_S) -> None:
        self.lifecycle.leave_phase(phase_id, timeout)

    def rollout_released(self) -> bool:
        return self.lifecycle.rollout_released()

    def rollout_status(self) -> str | None:
        return None if self._rollout_pool is None else self._rollout_pool.get_status()

    def slices(self, role: Role) -> tuple[PlacementSlice, ...]:
        prefix = f"{Role(role).value}/"
        return tuple(item for item in self.placement.allocations() if item.group_id.startswith(prefix))

    def resident(self, role: Role) -> bool:
        with self._lock:
            models = tuple(self._models.get(Role(role), {}).values())
        return any(model.resident for model in models)

    def _deployment(self, role: Role) -> DeploymentSpec:
        """Derive how a role shares GPUs from where it was placed."""
        slices = self.slices(role)
        phases = {item.phase for item in slices}
        if len(phases) > 1:
            raise ValueError(f"Role {role.value} is placed in several phases: {sorted(phases)}")
        phase = phases.pop() if phases else PHASE_GENERATE
        if phase != PHASE_GENERATE:
            mode = DeploymentMode.DEFER
        elif any(item.owner is PlacementOwner.CONTROLLER for item in slices):
            mode = DeploymentMode.SPLIT
        else:
            mode = DeploymentMode.DECOUPLED
        return DeploymentSpec(mode=mode, phase=phase)

    # ------------------------------------------------------------------
    # Placement ledger.
    # ------------------------------------------------------------------

    def plan_placement(
        self, requests: Sequence[PlacementRequest], placement_group: PlacementGroupView, *, dry_run: bool = False
    ) -> tuple[PlacementSlice, ...]:
        return self.placement.plan(requests, placement_group, dry_run=dry_run)

    def allocations(self) -> tuple[PlacementSlice, ...]:
        return self.placement.allocations()

    # ------------------------------------------------------------------
    # Role creation and teardown; the engines live in this process.
    # ------------------------------------------------------------------

    @ray.method(concurrency_group="rollout")
    def create_rollout_role(self, args: Any, placement_group: Any) -> dict[str, Any]:
        """Create the rollout engines and return the primary Router address."""
        from relax.distributed.ray.rollout import RolloutEnginePool

        if Role.ROLLOUT in self._failed_starts:
            # Stop what a failed start left behind; raises while it still holds GPUs.
            self._shutdown_role(Role.ROLLOUT)
        if self._rollout_pool is None:
            # The pool registers its servers once they are started.
            self._rollout_started = False
            leftovers: dict[str, Any] = {}
            try:
                self._rollout_pool = RolloutEnginePool(
                    args, placement_group, inference_manager=self, startup_leftovers=leftovers
                )
            except Exception:
                self._keep_failed_start(Role.ROLLOUT, leftovers)
                raise
        return self._rollout_pool.get_primary_router_address()

    def begin_rollout(self) -> None:
        """Record the first generation before enabling engine health
        monitoring."""
        self._rollout_started = True
        self.rollout_operation("health_monitoring_resume")

    @ray.method(concurrency_group="rollout")
    def rollout_operation(self, method: str, /, *args: Any, **kwargs: Any) -> Any:
        """Run one public rollout engine-pool operation."""
        if self._rollout_pool is None:
            raise RuntimeError("The rollout engine pool has not been created on this manager")
        if method.startswith("_") or not callable(getattr(self._rollout_pool, method, None)):
            raise ValueError(f"Unsupported rollout pool method: {method}")
        if method == "recover_rollout_engines":
            kwargs["rollout_started"] = self._rollout_started
        result = getattr(self._rollout_pool, method)(*args, **kwargs)
        return asyncio.run(result) if asyncio.iscoroutine(result) else result

    def create_role(
        self, role: Role | str, models: Sequence[tuple[InferenceModelSpec, Any, dict[str, Any]]]
    ) -> tuple[str, ...]:
        """Start a static role's models here and register them.

        Each entry is ``(model_spec, engine_args, placement_kwargs)``; see
        :func:`relax.distributed.ray.rollout.start_servers`. The whole role's
        layout is checked against the task ledger before its first engine
        starts; a later failure closes whatever already started.
        """
        from relax.distributed.ray.rollout import placement_preview, start_servers

        role = Role(role)
        if role in self._failed_starts:
            # Stop what a failed start left behind; raises while it still holds GPUs.
            self._shutdown_role(role)
        if role in self._models:
            return self.model_ids(role)
        previews: dict[str, tuple[PlacementGroupView, list[PlacementRequest]]] = {}
        for config, engine_args, placement in models:
            view, requests = placement_preview(engine_args, [config], role=role, **placement)
            previews.setdefault(view.key, (view, []))[1].extend(requests)
        for view, requests in previews.values():
            self.placement.plan(requests, view, dry_run=True)
        pools: dict[str, Any] = {}
        # Engines of a model whose own startup failed and could not be stopped.
        leftovers: dict[str, Any] = {}
        try:
            for config, engine_args, placement in models:
                pools.update(
                    start_servers(
                        engine_args,
                        [config],
                        planner=self.placement,
                        role=role,
                        leftovers=leftovers,
                        **deepcopy(placement),
                    )
                )
        except Exception:
            # Stop every started model even if one fails to stop.
            for model_id, pool in pools.items():
                try:
                    pool.shutdown(self.placement)
                except Exception as exc:
                    leftovers[model_id] = pool
                    logger.error(f"Failed to stop {role.value}/{model_id} after a failed startup: {exc}")
            self._keep_failed_start(role, leftovers)
            raise
        return self.register(role, pools)

    def _keep_failed_start(self, role: Role, leftovers: dict[str, Any]) -> None:
        """Keep what a failed start could not stop.

        Such a model may still hold its GPUs: it stays registered, DEAD and
        resident, so conflicting loads keep waiting, and shutting down or
        creating the role again retries the stop.
        """
        if not leftovers:
            return
        self.register(role, leftovers, observe=False)
        for model_id in leftovers:
            self.set_state(role, model_id, LifecycleState.DEAD)
        self._failed_starts.add(role)

    def _shutdown_role(self, role: Role) -> None:
        if role not in self._models:
            return
        stranded = 0
        if role is Role.ROLLOUT and self._rollout_pool is not None:
            self._rollout_pool.stop_monitors()
            # Stranded scale-out replicas are not registered models and have no
            # later retry once the pool goes.
            stranded = asyncio.run(self._rollout_pool.retry_stranded_replicas(final=True))
            self._rollout_pool = None
        errors = {}
        for model_id in self.model_ids(role):
            try:
                self.shutdown(role, model_id)
            except Exception as exc:
                errors[model_id] = exc
        with self._lock:
            if errors:
                # A model that failed to stop may still hold its GPUs: keep it
                # registered so conflicting loads keep waiting, and a retry
                # shuts down only what is left.
                self._models[role] = {k: v for k, v in self._models[role].items() if k in errors}
            else:
                self._models.pop(role, None)
                self._specs.pop(role, None)
                self._revision.pop(role, None)
                self._failed_starts.discard(role)
        if not self._models:
            from relax.distributed.ray.rollout import stop_launched_routers

            stop_launched_routers()
        if errors:
            raise RuntimeError(f"Failed to shut down inference role {role.value}") from next(iter(errors.values()))
        if stranded:
            raise RuntimeError(
                f"{stranded} scale-out replicas of {role.value} could not be confirmed stopped; "
                "their placement groups were removed as a last resort"
            )

    def shutdown_all(self) -> None:
        errors = []
        for role in tuple(self._models):
            try:
                self.shutdown(role)
            except Exception as exc:
                errors.append(exc)
        if errors:
            raise RuntimeError("Failed to shut down one or more inference roles") from errors[0]


def _restarted(previous: ModelSnapshot, model: ModelSnapshot) -> bool:
    """Whether a replica came back READY after a restart.

    A restarted engine usually keeps its address, so the topology alone does
    not change; clients still have to drop what they cached for the old
    process.
    """
    restarting = {
        replica.engine_id
        for replica in (*previous.replicas, *(worker for _, worker in previous.pd_workers))
        if replica.state in (LifecycleState.STARTING, LifecycleState.DEAD)
    }
    return any(
        replica.engine_id in restarting and replica.state == LifecycleState.READY
        for replica in (*model.replicas, *(worker for _, worker in model.pd_workers))
    )


def _topology(model: ModelSnapshot) -> tuple:
    """The discovery fields that describe where a model can be reached."""
    return (
        model.router_url,
        tuple(sorted((replica.engine_id, replica.base_url) for replica in model.replicas)),
        tuple(sorted((worker.engine_id, worker.base_url) for _, worker in model.pd_workers)),
    )


# A drain blocks inside this actor until completions arrive as calls on this
# same actor, so it needs several execution slots; rollout engine operations get
# their own group because a scale-out runs for minutes.
_MANAGER_CONCURRENCY = 8
InferenceManagerActor = ray.remote(num_cpus=1, num_gpus=0, concurrency_groups={"rollout": 8})(InferenceManager)


def validate_task_layout(args: Any) -> None:
    """Plan every inference role of the task before its first engine starts.

    Each role is described exactly as it will start -- Teacher, then Rollout,
    then GenRM, in the placement groups the Controller gives them -- and
    planned into a scratch ledger over stand-in groups of the same size, so a
    layout error in a later role fails the task before an earlier role holds
    GPUs.
    """
    from relax.distributed.ray.rollout import placement_preview, rollout_role_models
    from relax.engine.inference.config_adapters import genrm_role_models
    from relax.utils.opd.opd_utils import is_managed_opd_teacher_colocate, is_managed_opd_teacher_enabled

    resource = getattr(args, "resource", None) or {}
    engines = not getattr(args, "debug_train_only", False)

    def stand_in(name: str, role: str) -> tuple[str, tuple[int, ...], tuple[int, ...]] | None:
        num_gpus = int(resource[role][1]) if role in resource else 0
        return (f"preflight/{name}", tuple(range(num_gpus)), tuple(range(num_gpus))) if num_gpus else None

    # Sync colocate hands the actor placement group to rollout and GenRM; the
    # colocated teacher builds that same group first.
    shared = None
    if (
        getattr(args, "colocate", False)
        and not getattr(args, "hybrid", False)
        and {"actor", "rollout"} <= set(resource)
    ):
        shared = stand_in("actor", "actor")
    # One entry per start_servers call: (role, models, engine args, placement).
    calls: list[tuple[Role, list[InferenceModelSpec], Any, dict[str, Any]]] = []
    if engines and is_managed_opd_teacher_enabled(args):
        from relax.utils.opd.opd_utils import managed_teacher_models

        teacher_pg = shared if is_managed_opd_teacher_colocate(args) else None
        for config, engine_args, placement in managed_teacher_models(args, teacher_pg):
            calls.append((Role.TEACHER, [config], engine_args, placement))
    # SFT starts a rollout only to predict.
    sft_without_rollout = (
        getattr(args, "loss_type", None) == "sft" and getattr(args, "sft_predict_interval", None) is None
    )
    if engines and "rollout" in resource and not sft_without_rollout:
        rollout_pg = shared or stand_in("rollout", "rollout")
        placement = {"pg": rollout_pg, "bundle_offset": 0 if rollout_pg is not None else None}
        calls.append((Role.ROLLOUT, rollout_role_models(args), args, placement))
    if getattr(args, "_genrm_instances_resolved", None) and "genrm" in resource:
        for config, engine_args, placement in genrm_role_models(args, shared or stand_in("genrm", "genrm")):
            calls.append((Role.GENRM, [config], engine_args, placement))

    ledger = PlacementPlanner()
    for role, models, engine_args, placement in calls:
        view, requests = placement_preview(engine_args, models, role=role, **placement)
        try:
            ledger.plan(requests, view)
        except ValueError as exc:
            raise ValueError(f"Invalid {role.value} placement; no inference engine was started: {exc}") from exc


def create_inference_manager(args: Any, runtime_env: dict[str, Any] | None = None) -> Any:
    """Create the task's inference manager actor, pinned to the head node.

    The Routers start in this actor's process and the rest of the job resolves
    a Router by the head node's address. An unsupported layout fails here,
    before any engine starts.
    """
    from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

    from relax.core.node_group_affinity import require_control_plane_resource_on_node
    from relax.distributed.ray.placement_group import _get_head_node_id

    reject_shared_co_resident(args)
    validate_task_layout(args)
    head_node_id = _get_head_node_id()
    require_control_plane_resource_on_node(args, head_node_id)
    return InferenceManagerActor.options(
        **with_control_plane_affinity(
            args,
            {
                "num_cpus": 1,
                "num_gpus": 0,
                "runtime_env": runtime_env,
                "max_concurrency": _MANAGER_CONCURRENCY,
                "scheduling_strategy": NodeAffinitySchedulingStrategy(node_id=head_node_id, soft=False),
            },
        )
    ).remote()
