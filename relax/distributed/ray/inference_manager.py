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
model and is the only writer of the published snapshots. Calls go one way,
Manager to pool: a pool never calls into the Manager, and reports changes to
its engines only through :class:`~relax.engine.inference.types.TopologyEvents`,
which records them for the Manager to act on.

Whether a model may serve is the Manager's to decide: its KV cache must be
loaded and no weight update of its role may be running. A pool observes its
engines with that verdict and never publishes READY on its own.

Handing shared GPUs over takes two separate proofs. An empty request ledger
proves that every caller has let go; it says nothing about the engines, which
may still run a request whose HTTP connection is gone or whose abort was only
acknowledged. ``offload`` supplies the second: it returns only once its
engines have confirmed they are idle and released their memory, and raises
otherwise. A model stays resident, and so keeps conflicting roles off its
GPUs, until ``offload`` or ``shutdown`` has succeeded.
"""

import asyncio
import functools
import time
from collections.abc import Awaitable, Callable, Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass, field, replace
from threading import Condition, Lock, RLock, Thread
from typing import Any
from uuid import uuid4

import ray

from relax.core.node_group_affinity import with_control_plane_affinity
from relax.engine.inference.config import DeploymentSpec, InferenceModelSpec, InferenceRoleSpec
from relax.engine.inference.discovery import new_manager_epoch
from relax.engine.inference.lifecycle import SWITCH_DRAIN_TIMEOUT_S, LifecycleCoordinator, LoopAwareCondition
from relax.engine.inference.phase_plans import PHASE_GENERATE, require_deferred_shared_genrm
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
    TopologyEvents,
)
from relax.utils.logging_utils import get_logger


logger = get_logger(__name__)

# Weight updates that close a role's admission until they end; see
# ``begin_weight_update``. A backend update replaces the weights in place; a
# fully-async update also takes the rollout pool's topology lease.
WEIGHT_UPDATE_BACKEND = "backend"
WEIGHT_UPDATE_FULLY_ASYNC = "fully_async"

# How long a shutdown lets admitted requests finish before aborting them and
# stopping the engines.
SHUTDOWN_DRAIN_TIMEOUT_S = 30.0

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


@dataclass(frozen=True)
class DeploymentManagerHandle:
    """The Manager handle one deployment of a service role runs with.

    It forwards every call to ``manager``; role starts also carry the
    deployment's ``start_token``, so the Manager refuses a start from an
    earlier deployment of the role; see :meth:`InferenceManager.open_starts`.
    """

    manager: Any
    start_token: str

    def __getattr__(self, method: str) -> Any:
        if method.startswith("_"):
            raise AttributeError(method)
        call = getattr(self.manager, method)
        if method in ("create_role", "create_rollout_role"):
            return _TokenedCall(call, self.start_token)
        return call


@dataclass(frozen=True)
class _TokenedCall:
    call: Any
    start_token: str

    def remote(self, *args: Any, **kwargs: Any) -> Any:
        return self.call.remote(*args, start_token=self.start_token, **kwargs)


# Marks a role whose starts a rollback closed; no start token matches it.
_STARTS_CLOSED = object()


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
    # The state the Manager holds a closed model in (DRAINING, SLEEPING or
    # DEAD), kept apart from what the pool reports. ``None`` leaves the state
    # to observations; otherwise ``publish`` pins state to it with admission
    # closed, and only a controlled activate or recover clears it.
    closed_state: LifecycleState | None = None
    # Whether a drain closed the model while it was READY. Until its offload
    # runs, such a model still holds everything it loaded, so reactivating it
    # only reopens admission instead of loading again.
    drained_ready: bool = False
    # Whether the loaded memory lets the model serve: false after a load
    # without the KV cache, until a load with it.
    memory_ready: bool = True
    # Bumped on every lifecycle transition; an observation taken before one is
    # stale and dropped.
    generation: int = 0


@dataclass(frozen=True)
class _Permit:
    """What one admission holds until its permit is returned.

    The permit token, not the caller's trace ID, keys the ledger: callers may
    reuse a trace ID for retries or across roles, and each admission must stay
    held until its own completion.
    """

    role: Role
    model_id: str
    trace_id: str

    @property
    def model(self) -> tuple[Role, str]:
        return self.role, self.model_id


class InferenceManager:
    """Owner of every inference model of one task; see the module docstring."""

    def __init__(self) -> None:
        self.manager_epoch = new_manager_epoch()
        self.placement = PlacementPlanner()
        self._lock = RLock()
        # Completions wake drains on threads and on the actor's event loops.
        self._requests_done = LoopAwareCondition(self._lock)
        self._models: dict[Role, dict[str, _Model]] = {}
        self._specs: dict[Role, InferenceRoleSpec] = {}
        self._revision: dict[Role, int] = {}
        # In-flight admissions by permit token.
        self._inflight: dict[str, _Permit] = {}
        self._rollout_pool: Any = None
        self._rollout_started = False
        # Roles registered only with the models a failed ``create_role`` could
        # not stop; creating the role again stops them first.
        self._failed_starts: set[Role] = set()
        # Role creations still running. Stopping a role waits for them: until
        # they register, their engines are invisible yet may hold GPUs.
        self._starting: dict[Role, int] = {}
        self._start_finished = Condition(self._lock)
        # The start token of each role's current deployment, or
        # ``_STARTS_CLOSED`` after a rollback; see ``open_starts``. A role
        # without an entry accepts every start.
        self._start_tokens: dict[Role, Any] = {}
        # Running weight updates per role; see ``begin_weight_update``.
        self._weight_updates: dict[Role, set[str]] = {}
        # Models whose engines a pool reported settled, awaiting an
        # observation by the refresher thread.
        self._stale: set[tuple[Role, str]] = set()
        self._stale_added = Condition(self._lock)
        self._refresher: Thread | None = None
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
            if entry.closed_state is not None:
                # Keep the observed topology; the Manager's closure wins.
                model = _with_state(model, entry.closed_state)
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
            self.publish(role, _with_state(self._model(role, model_id).snapshot, state))

    def _close(self, role: Role, model_id: str, state: LifecycleState) -> None:
        """Keep the model closed in ``state`` until a controlled activate."""
        with self._lock:
            entry = self._model(role, model_id)
            entry.closed_state = state
            entry.generation += 1
            self.set_state(role, model_id, state)

    def _reopen(self, role: Role, model_id: str, state: LifecycleState) -> None:
        """Start a controlled activate: lift the closure and move to ``state``;
        the next observation decides whether the model is READY."""
        with self._lock:
            entry = self._model(role, model_id)
            entry.closed_state = None
            entry.generation += 1
            self.set_state(role, model_id, state)

    def _model(self, role: Role | str, model_id: str) -> _Model:
        models = self._models.get(Role(role))
        if models is None or model_id not in models:
            raise KeyError(f"Unknown inference model: {Role(role).value}/{model_id}")
        return models[model_id]

    def refresh(self, role: Role | str, model_id: str) -> None:
        """Publish the pool's current observation of one model."""
        self._observe(Role(role), model_id)

    def _observe(self, role: Role, model_id: str) -> None:
        """Publish the pool's observation unless a lifecycle transition ran
        while it was taken.

        The pool observes with the Manager's verdict on whether the model may
        serve, and a READY observation is held back if that verdict changed
        meanwhile.
        """
        entry = self._model(role, model_id)
        fenced = self._pool_fenced(role)
        with self._lock:
            generation = entry.generation
            serving = self._may_serve(role, entry) and not fenced
        model = entry.pool.observe(serving=serving)
        fenced = self._pool_fenced(role)
        with self._lock:
            if entry.generation != generation:
                logger.info(f"Dropping a stale observation of {role.value}/{model_id}")
                return
            if model.state == LifecycleState.READY and (fenced or not self._may_serve(role, entry)):
                model = _with_state(model, LifecycleState.STARTING)
            self.publish(role, model)

    def _may_serve(self, role: Role, entry: _Model) -> bool:
        """The Manager's verdict on admission: KV cache loaded, no weight
        update of the role running.

        Call under the lock.
        """
        return entry.memory_ready and not self._weight_updates.get(role)

    def _pool_fenced(self, role: Role) -> bool:
        """Whether the rollout pool is replacing weights on its own (a scale-
        out syncing new engines).

        Call without the lock: the pool takes its own.
        """
        return role is Role.ROLLOUT and self._rollout_pool is not None and self._rollout_pool.weights_changing()

    # ------------------------------------------------------------------
    # Pool topology events; see TopologyEvents.
    # ------------------------------------------------------------------

    def topology_events(self, role: Role | str) -> TopologyEvents:
        """The events a pool of ``role`` reports to; they only record."""
        return _RoleEvents(self, Role(role))

    def _topology_changed(self, role: Role, model_id: str) -> None:
        """Close admission at once; an observation reopens it.

        Until then the published replicas predate the change, so the view is
        marked incomplete.
        """
        with self._lock:
            if model_id in self._models.get(role, {}):
                model = _with_state(self._model(role, model_id).snapshot, LifecycleState.STARTING)
                self.publish(role, replace(model, issues=((model_id, "topology_unsettled"),)))

    def _topology_settled(self, role: Role, model_id: str) -> None:
        """Queue an observation for the refresher thread.

        Until it is published, the replicas on view predate the change (a
        scale-out reports only this), so the view is marked incomplete;
        admission stays as it is.
        """
        with self._lock:
            if model_id not in self._models.get(role, {}):
                return
            snapshot = self._model(role, model_id).snapshot
            unsettled = (model_id, "topology_unsettled")
            if unsettled not in snapshot.issues:
                self.publish(role, replace(snapshot, issues=(*snapshot.issues, unsettled)))
            self._stale.add((role, model_id))
            self._stale_added.notify_all()
            if self._refresher is None:
                self._refresher = Thread(target=self._refresh_loop, name="inference-refresher", daemon=True)
                self._refresher.start()

    def _refresh_loop(self) -> None:
        while True:
            with self._lock:
                while not self._stale:
                    self._stale_added.wait()
            self._refresh_stale()

    def _refresh_stale(self) -> None:
        """Observe every model a pool reported settled."""
        with self._lock:
            stale, self._stale = self._stale, set()
        for role, model_id in sorted(stale):
            try:
                self._observe(role, model_id)
            except Exception as exc:
                logger.warning(f"Failed to observe {role.value}/{model_id} after a topology change: {exc!r}")

    # ------------------------------------------------------------------
    # Request admission.
    # ------------------------------------------------------------------

    def admit_request(self, role: Role | str, model_id: str, trace_id: str | None = None) -> str:
        """Record an in-flight request on a READY model and return its permit
        token, or refuse it.

        Every admission gets a new permit, even for a ``trace_id`` already in
        flight; ``trace_id`` only labels it in logs. The caller returns the
        permit to :meth:`complete_request`.
        """
        role = Role(role)
        with self._lock:
            model = self._model(role, model_id).snapshot
            if not model.admission or model.state != LifecycleState.READY:
                raise RuntimeError(f"Model {role.value}/{model_id} is not ready for inference")
            permit = uuid4().hex
            self._inflight[permit] = _Permit(role, model_id, trace_id or permit)
            return permit

    def complete_request(self, permit: str) -> None:
        """Return the permit of a finished, failed or aborted request.

        This only records that the caller let go; the engine may still run the
        request, which the release waits for. Idempotent: a permit already
        returned or forgotten by an abort is ignored, so a late completion
        never releases another admission.
        """
        with self._requests_done:
            if self._inflight.pop(permit, None) is not None:
                self._requests_done.notify_all()

    def drain(
        self, roles: Sequence[Role | str], timeout: float = SWITCH_DRAIN_TIMEOUT_S, force_abort: bool = False
    ) -> None:
        """Close admission for ``roles`` and wait for their requests to end;
        see :meth:`_drain_models`."""
        roles = [Role(role) for role in roles]
        models = [(role, model_id) for role in roles for model_id in self.model_ids(role)]
        self._drain_models(models, timeout, force_abort)

    def _drain_models(self, models: Sequence[tuple[Role, str]], timeout: float, force_abort: bool = False) -> None:
        """Close admission on the resident ``models`` and wait until every
        request they admitted has ended.

        Every release goes through here before its engines let go of memory, so
        a request admitted just before the release never reaches a sleeping
        engine. Returning only proves the callers are done; the release that
        follows confirms the engines are idle. Closing and waiting happen under
        the admission lock, which the wait gives up so ``complete_request`` can
        run. A model that is already closed keeps its state.

        Every ``timeout`` seconds that requests are still in flight, a warning
        is logged and the wait goes on. With ``force_abort`` the engines abort
        them instead and they are forgotten, so the release can go ahead; a
        failed abort raises and leaves the model DRAINING and resident, so
        conflicting roles keep waiting.
        """
        draining = set(models)
        with self._requests_done:
            self._close_draining(draining)
            started = time.monotonic()
            deadline = started + timeout
            while pending := self._pending(draining):
                remaining = deadline - time.monotonic()
                if remaining > 0:
                    self._requests_done.wait(remaining)
                    continue
                if self._overdue(draining, pending, started, force_abort):
                    break
                deadline = time.monotonic() + timeout
        if pending:
            self._abort_requests(draining, pending)

    async def drain_async(
        self,
        models: Sequence[tuple[Role, str]],
        timeout: float,
        force_abort: bool,
        run: Callable[..., Awaitable[Any]],
    ) -> None:
        """``_drain_models`` for an event loop: it waits for completions
        without holding a thread, and aborts through ``run``.

        A drain cancelled while it waits leaves the models closed.
        """
        draining = set(models)
        with self._requests_done:
            self._close_draining(draining)
        started = time.monotonic()
        deadline = started + timeout
        while True:
            if await self._requests_done.wait_for(lambda: not self._pending(draining), deadline):
                return
            with self._requests_done:
                pending = self._pending(draining)
                if pending and self._overdue(draining, pending, started, force_abort):
                    break
            deadline = time.monotonic() + timeout
        await run(self._abort_requests, draining, pending)

    def _close_draining(self, draining: set[tuple[Role, str]]) -> None:
        """Close admission on the resident ``draining`` models that are open;
        call under the lock."""
        for role, model_id in draining:
            entry = self._model(role, model_id)
            if entry.resident and entry.closed_state is None:
                entry.drained_ready = entry.snapshot.state == LifecycleState.READY
                self._close(role, model_id, LifecycleState.DRAINING)

    def _pending(self, draining: set[tuple[Role, str]]) -> list[str]:
        """The permits ``draining`` models still hold; call under the lock."""
        return sorted(permit for permit, admission in self._inflight.items() if admission.model in draining)

    def _overdue(self, draining: set[tuple[Role, str]], pending: list[str], started: float, force_abort: bool) -> bool:
        """Report requests still in flight after a drain timeout; True when
        they are to be aborted.

        Call under the lock.
        """
        names = sorted(f"{role.value}/{model_id}" for role, model_id in draining)
        waited = time.monotonic() - started
        requests = [f"trace={self._inflight[permit].trace_id} permit={permit}" for permit in pending]
        if force_abort:
            logger.warning(f"Aborting {len(pending)} inference requests of {names} after {waited:.0f}s: {requests}")
            return True
        logger.warning(
            f"{len(pending)} inference requests of {names} still in flight after {waited:.0f}s; "
            f"waiting for them: {requests}"
        )
        return False

    def _abort_requests(self, models: set[tuple[Role, str]], permits: Sequence[str]) -> None:
        """Abort everything the resident ``models`` run, then forget
        ``permits``.

        The engine calls run outside the admission lock, which admission and
        completions must never wait behind; the models are already closed, so
        nothing is admitted meanwhile. A request still on its way to an engine
        is caught by the release itself, which drains the engine before letting
        go of memory. A failed abort raises with ``permits`` still held.
        """
        with self._lock:
            pools = [
                self._model(role, model_id).pool
                for role, model_id in sorted(models)
                if self._model(role, model_id).resident
            ]
        for pool in pools:
            pool.abort_requests()
        with self._requests_done:
            for permit in permits:
                self._inflight.pop(permit, None)
            self._requests_done.notify_all()

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
                for name in model_ids:
                    self._activate_model(role, name, tags)
            except Exception:
                for name in loading:
                    try:
                        self._deactivate_model(role, name)
                    except Exception as exc:
                        logger.warning(f"Failed to release {role.value}/{name} after a failed load: {exc}")
                raise

    def deactivate(
        self,
        role: Role | str,
        model_id: str | None = None,
        timeout: float = SWITCH_DRAIN_TIMEOUT_S,
        force_abort: bool = False,
    ) -> None:
        """Release a role's or one model's memory; a released model is left
        alone.

        Admission closes and the admitted requests end, or with ``force_abort``
        are aborted after ``timeout``, before any engine releases memory; see
        :meth:`_drain_models`.
        """
        role = Role(role)
        try:
            model_ids = self.model_ids(role) if model_id is None else (model_id,)
            # Drain the whole role before its first model lets go.
            self._drain_models([(role, name) for name in model_ids], timeout, force_abort)
            if model_id is None and role is Role.ROLLOUT and self._rollout_pool is not None:
                # A restart must not race the release.
                self._rollout_pool.health_monitoring_pause()
            for name in model_ids:
                self._deactivate_model(role, name, timeout)
        finally:
            # A load waiting for these GPUs re-checks now.
            self.lifecycle.notify_released()

    def _activate_model(self, role: Role, model_id: str, tags: list[str] | None) -> None:
        entry = self._model(role, model_id)
        with entry.lock:
            if entry.resident and entry.snapshot.state == LifecycleState.READY:
                return
            if entry.resident and entry.closed_state == LifecycleState.DRAINING and entry.drained_ready:
                # Drained but never offloaded: its memory is still loaded.
                self._reopen(role, model_id, LifecycleState.ONLOADING)
                self._observe(role, model_id)
                return
            entry.resident = True
            # A weights-only load keeps the model closed until the KV cache
            # follows, after the weight sync.
            entry.memory_ready = _loads_kv_cache(tags)
            self._reopen(role, model_id, LifecycleState.ONLOADING)
            try:
                entry.pool.onload(tags)
            finally:
                self._observe(role, model_id)

    def _deactivate_model(self, role: Role, model_id: str, timeout: float = SWITCH_DRAIN_TIMEOUT_S) -> None:
        """Drain one model and release it once its engines confirm they are
        idle; an unconfirmed release leaves it DEAD and resident."""
        entry = self._model(role, model_id)
        with entry.lock:
            if not entry.resident:
                return
            self._drain_models([(role, model_id)], timeout)
            try:
                entry.pool.offload()
            except Exception:
                self._close(role, model_id, LifecycleState.DEAD)
                raise
            entry.resident = False
            self._close(role, model_id, LifecycleState.SLEEPING)

    def recover(self, role: Role | str, model_id: str) -> None:
        """Restart dead engines; they allocate memory, so this waits like a
        load."""
        role = Role(role)
        with self.lifecycle.lease([role]):
            entry = self._model(role, model_id)
            with entry.lock:
                entry.resident = True
                self._reopen(role, model_id, LifecycleState.STARTING)
                try:
                    entry.pool.recover()
                finally:
                    self._observe(role, model_id)

    def recover_rollout(self, model_name: str | None = None) -> Any:
        """Restart dead rollout engines, then return what the weight sync needs
        to reach them; see ``RolloutEnginePool.get_rollout_engines_and_lock``.

        Before the first ``generate`` pass a recovery would only race the
        initial bring-up, so it only pauses the health monitors.
        """
        return self._recover_rollout(self.rollout_recovery_target(model_name), model_name)

    def rollout_recovery_target(self, model_name: str | None = None) -> str | None:
        """The rollout model ``recover_rollout`` restarts engines of, or None
        before the first ``generate`` pass or for an unknown model."""
        names = self.model_ids(Role.ROLLOUT)
        name = model_name if model_name is not None else next(iter(names), None)
        return name if self._rollout_started and name in names else None

    def _recover_rollout(self, target: str | None, model_name: str | None) -> Any:
        pool = self._rollout_pool
        pool.health_monitoring_pause()
        if target is not None:
            self.recover(Role.ROLLOUT, target)
        return pool.get_rollout_engines_and_lock(model_name)

    def begin_weight_update(self, role: Role | str, source: str) -> bool:
        """Close a role's admission until ``end_weight_update(source)``.

        No model of the role publishes READY while any update runs, and
        observations taken before this call are dropped. A fully-async update
        first takes the rollout pool's topology lease; it returns False,
        changing nothing, while an elastic scale-in holds it.
        """
        role = Role(role)
        pool = self._rollout_pool if role is Role.ROLLOUT else None
        if source == WEIGHT_UPDATE_FULLY_ASYNC and pool is not None and not pool._set_weight_updating(True):
            return False
        with self._lock:
            self._weight_updates.setdefault(role, set()).add(source)
            for model_id in self.model_ids(role):
                self._model(role, model_id).generation += 1
                self.set_state(role, model_id, LifecycleState.STARTING)
        return True

    def end_weight_update(self, role: Role | str, source: str) -> None:
        """End one weight update and observe the role again; idempotent.

        Ending an update proves nothing about the weights: the observation
        publishes READY only with healthy, registered replicas of one weight
        version.
        """
        role = Role(role)
        with self._lock:
            self._weight_updates.get(role, set()).discard(source)
        if source == WEIGHT_UPDATE_FULLY_ASYNC and role is Role.ROLLOUT and self._rollout_pool is not None:
            self._rollout_pool._set_weight_updating(False)
        for model_id in self.model_ids(role):
            self._observe(role, model_id)

    def health_check(self, role: Role | str, model_id: str) -> bool:
        entry = self._model(role, model_id)
        with entry.lock:
            healthy = entry.pool.health_check()
            self._observe(Role(role), model_id)
            return healthy

    def shutdown(self, role: Role | str, model_id: str | None = None, close_starts: bool = False) -> None:
        """Stop a whole role, or one of its models; a stopped model is left
        alone. ``close_starts`` also refuses every later start of the role
        until :meth:`open_starts`.

        Admitted requests get a bounded grace period, then the engines abort
        them. Stopping proceeds even if the abort fails: a killed engine fails
        its requests, it never runs them on released memory, and teardown must
        not hang on a client that never completes.
        """
        role = Role(role)
        if model_id is None:
            with self._lock:
                if close_starts:
                    self._start_tokens[role] = _STARTS_CLOSED
                self._start_finished.wait_for(lambda: role not in self._starting)
            self._shutdown_role(role)
            return
        entry = self._model(role, model_id)
        with entry.lock:
            if not entry.resident and entry.snapshot.state == LifecycleState.DEAD:
                return
            try:
                self._drain_models([(role, model_id)], SHUTDOWN_DRAIN_TIMEOUT_S, force_abort=True)
            except Exception as exc:
                # Stopping the engines fails whatever they still run.
                logger.warning(f"Stopping {role.value}/{model_id} with requests still in flight: {exc}")
            self._close(role, model_id, LifecycleState.DEAD)
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

    def create_rollout_role(self, args: Any, placement_group: Any, start_token: str | None = None) -> dict[str, Any]:
        """Create the rollout engines and return the primary Router address."""
        with self._starting_role(Role.ROLLOUT, start_token):
            return self._create_rollout_role(args, placement_group)

    def _create_rollout_role(self, args: Any, placement_group: Any) -> dict[str, Any]:
        from relax.distributed.ray.rollout import RolloutEnginePool

        if Role.ROLLOUT in self._failed_starts:
            # Stop what a failed start left behind; raises while it still holds GPUs.
            self._shutdown_role(Role.ROLLOUT)
        if self._rollout_pool is None:
            # The pool registers its servers once they are started.
            self._rollout_started = False
            leftovers: dict[str, Any] = {}
            try:
                pool = RolloutEnginePool(
                    args,
                    placement_group,
                    planner=self.placement,
                    events=self.topology_events(Role.ROLLOUT),
                    startup_leftovers=leftovers,
                )
            except Exception:
                self._keep_failed_start(Role.ROLLOUT, leftovers)
                raise
            if pool.servers:
                self.register(Role.ROLLOUT, pool.servers)
            self._rollout_pool = pool
        return self._rollout_pool.get_primary_router_address()

    def begin_rollout(self) -> None:
        """Record the first generation before enabling engine health
        monitoring."""
        self._rollout_started = True
        self.rollout_operation("health_monitoring_resume")

    def rollout_operation(self, method: str, /, *args: Any, **kwargs: Any) -> Any:
        """Run one public rollout engine-pool operation, then observe every
        model whose engines it changed."""
        if self._rollout_pool is None:
            raise RuntimeError("The rollout engine pool has not been created on this manager")
        if method == "recover_rollout_engines":
            return self.recover_rollout(*args, **kwargs)
        if method.startswith("_") or not callable(getattr(self._rollout_pool, method, None)):
            raise ValueError(f"Unsupported rollout pool method: {method}")
        try:
            result = getattr(self._rollout_pool, method)(*args, **kwargs)
            return asyncio.run(result) if asyncio.iscoroutine(result) else result
        finally:
            # Callers see the topology the operation left behind.
            self._refresh_stale()

    def create_role(
        self,
        role: Role | str,
        models: Sequence[tuple[InferenceModelSpec, Any, dict[str, Any]]],
        start_token: str | None = None,
    ) -> tuple[str, ...]:
        """Start a static role's models here and register them.

        Each entry is ``(model_spec, engine_args, placement_kwargs)``; see
        :func:`relax.distributed.ray.rollout.start_servers`. The whole role's
        layout is checked against the task ledger before its first engine
        starts; a later failure closes whatever already started.
        """
        role = Role(role)
        with self._starting_role(role, start_token):
            return self._create_role(role, models)

    def _create_role(
        self, role: Role, models: Sequence[tuple[InferenceModelSpec, Any, dict[str, Any]]]
    ) -> tuple[str, ...]:
        from relax.distributed.ray.rollout import placement_preview, start_servers

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

    def open_starts(self, role: Role | str, start_token: str) -> None:
        """Accept starts of ``role`` from the deployment holding
        ``start_token`` only.

        A start sent by an earlier deployment, or after a ``close_starts``
        shutdown, can reach the Manager late; it is refused either way.
        """
        with self._lock:
            self._start_tokens[Role(role)] = start_token

    @contextmanager
    def _starting_role(self, role: Role, start_token: str | None) -> Iterator[None]:
        """Mark a creation of ``role`` as running until it returns."""
        with self._lock:
            if role in self._start_tokens and self._start_tokens[role] != start_token:
                raise RuntimeError(f"Start of inference role {role.value} is not from its current deployment")
            self._starting[role] = self._starting.get(role, 0) + 1
        try:
            yield
        finally:
            with self._lock:
                self._starting[role] -= 1
                if not self._starting[role]:
                    del self._starting[role]
                self._start_finished.notify_all()

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
            self._close(role, model_id, LifecycleState.DEAD)
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
        with self._lock:
            # A role still starting is stopped once it registers.
            roles = tuple(dict.fromkeys((*self._models, *self._starting)))
        for role in roles:
            try:
                self.shutdown(role)
            except Exception as exc:
                errors.append(exc)
        if errors:
            raise RuntimeError("Failed to shut down one or more inference roles") from errors[0]


@dataclass(frozen=True)
class _RoleEvents:
    """The :class:`TopologyEvents` of one role's pool; they only record."""

    manager: InferenceManager
    role: Role

    def changed(self, model_id: str) -> None:
        self.manager._topology_changed(self.role, model_id)

    def settled(self, model_id: str) -> None:
        self.manager._topology_settled(self.role, model_id)


def _loads_kv_cache(tags: Sequence[str] | None) -> bool:
    """Whether a load of ``tags`` leaves the model able to serve."""
    if tags is None:
        return True
    # SGLang's ``GPU_MEMORY_TYPE_KV_CACHE``; the CPU control plane does not import SGLang.
    return "kv_cache" in tags


def _with_state(model: ModelSnapshot, state: LifecycleState) -> ModelSnapshot:
    """``model`` with admission closed and it and its replicas in ``state``."""
    return replace(
        model,
        state=state,
        admission=False,
        replicas=tuple(replace(replica, state=state) for replica in model.replicas),
        pd_workers=tuple((kind, replace(worker, state=state)) for kind, worker in model.pd_workers),
    )


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


# The actor is async: each concurrency group is one event loop, and a slot is a
# coroutine, not a thread. Admission and completion never wait, so they run on
# their own loops: completions in particular must get in while a drain waits for
# them. A call that waits (for a GPU lease, a drain or another call on the same
# model) waits on its loop and takes a thread only for its work, which never
# waits for another call; so the thread pools bound throughput, never progress.
_MANAGER_CONCURRENCY = 4096
REQUEST_PERMIT_GROUP = "request_permit"
REQUEST_CONTROL_GROUP = "request_control"
ROLLOUT_GROUP = "rollout"
_CONCURRENCY_GROUPS = {REQUEST_PERMIT_GROUP: 1024, REQUEST_CONTROL_GROUP: 128, ROLLOUT_GROUP: 8}
# Threads that run blocking work: engine RPCs and memory transitions.
_LIFECYCLE_THREADS = 64
_ROLLOUT_THREADS = 8

_ModelKey = tuple[Role, str]


class _BlockingRunner:
    """Runs blocking Manager work on its own thread pool."""

    def __init__(self, name: str, threads: int) -> None:
        self.name = name
        self.threads = threads
        self._busy = 0
        self._busy_lock = Lock()
        self._executor = ThreadPoolExecutor(threads, thread_name_prefix=f"inference-{name}")

    async def run(
        self, fn: Callable[..., Any], /, *args: Any, releases: Sequence[Callable[[], None]] = (), **kwargs: Any
    ) -> Any:
        """Run ``fn`` on a thread, then ``releases`` on that thread.

        Once submitted the work runs to its end even if the caller is
        cancelled, and only then gives back what ``releases`` hold; if it
        cannot be submitted, they are given back at once.
        """

        def work() -> Any:
            try:
                return fn(*args, **kwargs)
            finally:
                for release in reversed(releases):
                    release()
                with self._busy_lock:
                    self._busy -= 1

        with self._busy_lock:
            self._busy += 1
            busy = self._busy
        if busy > self.threads:
            logger.warning(
                f"All {self.threads} {self.name} threads of the inference manager are busy; "
                f"{getattr(fn, '__name__', fn)} queues behind {busy - self.threads - 1} calls"
            )
        try:
            future = asyncio.get_running_loop().run_in_executor(self._executor, work)
        except BaseException:
            for release in reversed(releases):
                release()
            with self._busy_lock:
                self._busy -= 1
            raise
        return await asyncio.shield(future)


class _ModelGates:
    """Lets one blocking call at a time run per model; the others wait on an
    event loop instead of a thread."""

    def __init__(self) -> None:
        self._changed = LoopAwareCondition()
        self._busy: set[_ModelKey] = set()

    async def acquire(self, keys: frozenset[_ModelKey]) -> None:
        def claim() -> bool:
            if keys & self._busy:
                return False
            self._busy |= keys
            return True

        await self._changed.wait_for(claim)

    def release(self, keys: frozenset[_ModelKey]) -> None:
        with self._changed:
            self._busy -= keys
            self._changed.notify_all()


def _inline(name: str, group: str) -> Any:
    """An actor method running ``InferenceManager.<name>`` on ``group``'s event
    loop; only for calls that hold the admission lock briefly and never
    wait."""

    async def method(self: "_InferenceManagerActor", *args: Any, **kwargs: Any) -> Any:
        return getattr(self.manager, name)(*args, **kwargs)

    method.__name__ = method.__qualname__ = name
    return ray.method(concurrency_group=group)(method)


def _blocking(name: str, runner: str, group: str | None = None) -> Any:
    """An actor method running ``InferenceManager.<name>`` on a thread of
    ``runner``, in ``group`` or the default one; only for calls that wait for
    no other call."""

    async def method(self: "_InferenceManagerActor", *args: Any, **kwargs: Any) -> Any:
        return await getattr(self, runner).run(getattr(self.manager, name), *args, **kwargs)

    method.__name__ = method.__qualname__ = name
    return method if group is None else ray.method(concurrency_group=group)(method)


class _InferenceManagerActor:
    """The Ray actor around one :class:`InferenceManager`.

    It exposes the Manager's remote API under the same names and decides where
    each call runs and waits. The Manager stays synchronous, so its in-process
    callers and its own threads are unaffected.
    """

    def __init__(self) -> None:
        self.manager = InferenceManager()
        self._lifecycle = _BlockingRunner("lifecycle", _LIFECYCLE_THREADS)
        self._rollout = _BlockingRunner("rollout", _ROLLOUT_THREADS)
        self._gates = _ModelGates()

    # Gateway fast path.
    snapshot = _inline("snapshot", REQUEST_PERMIT_GROUP)
    admit_request = _inline("admit_request", REQUEST_PERMIT_GROUP)
    complete_request = _inline("complete_request", REQUEST_CONTROL_GROUP)
    rollout_released = _inline("rollout_released", REQUEST_CONTROL_GROUP)

    # Lifecycle. Each waits on the loop, then runs its work on a thread.
    async def activate(self, role: Role | str, model_id: str | None = None, tags: list[str] | None = None) -> None:
        role = Role(role)
        await self._leased(
            self._lifecycle, role, self._targets(role, model_id), self.manager.activate, role, model_id, tags
        )

    async def recover(self, role: Role | str, model_id: str) -> None:
        role = Role(role)
        await self._leased(self._lifecycle, role, self._targets(role, model_id), self.manager.recover, role, model_id)

    async def deactivate(
        self,
        role: Role | str,
        model_id: str | None = None,
        timeout: float = SWITCH_DRAIN_TIMEOUT_S,
        force_abort: bool = False,
    ) -> None:
        role = Role(role)
        targets = self._targets(role, model_id)
        try:
            # Drain the whole role before its first model lets go.
            await self.manager.drain_async(targets, timeout, force_abort, self._lifecycle.run)
        except BaseException:
            self.manager.lifecycle.notify_released()
            raise
        await self._gated(self._lifecycle, targets, self.manager.deactivate, role, model_id, timeout, force_abort)

    async def drain(
        self, roles: Sequence[Role | str], timeout: float = SWITCH_DRAIN_TIMEOUT_S, force_abort: bool = False
    ) -> None:
        targets = [key for role in roles for key in self._targets(Role(role), None)]
        await self.manager.drain_async(targets, timeout, force_abort, self._lifecycle.run)

    async def health_check(self, role: Role | str, model_id: str) -> bool:
        role = Role(role)
        return await self._gated(
            self._lifecycle, self._targets(role, model_id), self.manager.health_check, role, model_id
        )

    async def shutdown(self, role: Role | str, model_id: str | None = None, close_starts: bool = False) -> None:
        role = Role(role)
        targets = self._targets(role, model_id)
        try:
            await self.manager.drain_async(targets, SHUTDOWN_DRAIN_TIMEOUT_S, True, self._lifecycle.run)
        except Exception as exc:
            # Stopping the engines fails whatever they still run.
            logger.warning(f"Stopping {role.value} with requests still in flight: {exc}")
        await self._gated(self._lifecycle, targets, self.manager.shutdown, role, model_id, close_starts)

    async def call(self, role: Role | str, model_id: str, method: str, /, *args: Any, **kwargs: Any) -> Any:
        if method in _LIFECYCLE_METHODS:
            return await getattr(self, method)(role, model_id, *args, **kwargs)
        return await self._lifecycle.run(self.manager.call, role, model_id, method, *args, **kwargs)

    async def enter_phase(self, phase_id: str, timeout: float = SWITCH_DRAIN_TIMEOUT_S) -> None:
        await self.manager.lifecycle.enter_phase_async(phase_id, self._drain_roles, self._lifecycle.run, timeout)

    async def leave_phase(self, phase_id: str, timeout: float = SWITCH_DRAIN_TIMEOUT_S) -> None:
        await self.manager.lifecycle.leave_phase_async(phase_id, self._drain_roles, self._lifecycle.run, timeout)

    # Role creation, teardown and queries wait for no other call.
    shutdown_all = _blocking("shutdown_all", "_lifecycle")
    create_role = _blocking("create_role", "_lifecycle")
    open_starts = _blocking("open_starts", "_lifecycle")
    refresh = _blocking("refresh", "_lifecycle")
    roles = _blocking("roles", "_lifecycle")
    registered_roles = _blocking("registered_roles", "_lifecycle")
    role_spec = _blocking("role_spec", "_lifecycle")
    model_ids = _blocking("model_ids", "_lifecycle")
    model_spec = _blocking("model_spec", "_lifecycle")
    model_placement = _blocking("model_placement", "_lifecycle")
    slices = _blocking("slices", "_lifecycle")
    resident = _blocking("resident", "_lifecycle")
    plan_placement = _blocking("plan_placement", "_lifecycle")
    allocations = _blocking("allocations", "_lifecycle")

    # Rollout engine pool.
    create_rollout_role = _blocking("create_rollout_role", "_rollout", ROLLOUT_GROUP)
    begin_rollout = _blocking("begin_rollout", "_rollout", ROLLOUT_GROUP)
    begin_weight_update = _blocking("begin_weight_update", "_rollout", ROLLOUT_GROUP)
    end_weight_update = _blocking("end_weight_update", "_rollout", ROLLOUT_GROUP)

    @ray.method(concurrency_group=ROLLOUT_GROUP)
    async def rollout_operation(self, method: str, /, *args: Any, **kwargs: Any) -> Any:
        if method != "recover_rollout_engines":
            return await self._rollout.run(self.manager.rollout_operation, method, *args, **kwargs)
        # A recovery restarts engines, which allocate memory: it waits for the
        # lease like a load.
        model_name = args[0] if args else kwargs.get("model_name")
        target = self.manager.rollout_recovery_target(model_name)
        if target is None:
            return await self._rollout.run(self.manager._recover_rollout, None, model_name)
        targets = self._targets(Role.ROLLOUT, target)
        return await self._leased(
            self._rollout, Role.ROLLOUT, targets, self.manager._recover_rollout, target, model_name
        )

    # ------------------------------------------------------------------

    def _targets(self, role: Role, model_id: str | None) -> frozenset[_ModelKey]:
        names = self.manager.model_ids(role) if model_id is None else (model_id,)
        return frozenset((role, name) for name in names)

    async def _drain_roles(self, roles: Sequence[Role], timeout: float) -> None:
        targets = [key for role in roles for key in self._targets(role, None)]
        await self.manager.drain_async(targets, timeout, False, self._lifecycle.run)

    async def _gated(
        self, runner: _BlockingRunner, targets: frozenset[_ModelKey], fn: Callable[..., Any], *args: Any
    ) -> Any:
        """Run ``fn`` on a thread once no other call runs on ``targets``."""
        await self._gates.acquire(targets)
        return await runner.run(fn, *args, releases=[functools.partial(self._gates.release, targets)])

    async def _leased(
        self,
        runner: _BlockingRunner,
        role: Role,
        targets: frozenset[_ModelKey],
        fn: Callable[..., Any],
        *args: Any,
    ) -> Any:
        """Run the load ``fn`` on a thread under the GPU lease of ``role``.

        The lease is taken first and the model gates second, the order every
        load uses; a caller cancelled before the work is handed to the thread
        gives back what it took.
        """
        lifecycle = self.manager.lifecycle
        token = await lifecycle.acquire([role])
        releases = [functools.partial(lifecycle.release, token)]
        try:
            await self._gates.acquire(targets)
        except BaseException:
            lifecycle.release(token)
            raise
        releases.append(functools.partial(self._gates.release, targets))
        return await runner.run(lifecycle.owned(token, fn), *args, releases=releases)


InferenceManagerActor = ray.remote(num_cpus=1, num_gpus=0, concurrency_groups=_CONCURRENCY_GROUPS)(
    _InferenceManagerActor
)


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

    # Nothing is placed yet, so a stand-in carries the logical layout alone.
    def stand_in(name: str, role: str) -> PlacementGroupView | None:
        num_gpus = int(resource[role][1]) if role in resource else 0
        if not num_gpus:
            return None
        gpus = tuple(range(num_gpus))
        return PlacementGroupView(gpus, gpus, PlacementOwner.CONTROLLER, identity=f"preflight/{name}")

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

    require_deferred_shared_genrm(args)
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
