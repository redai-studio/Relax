# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Manager-owned request admission, shared by proxy and lookaside clients.

The condition protects admission and the start of a drain, not GPU operations.
Permits provide a bounded grace period. Only an engine idle/release receipt
lets the manager retire outstanding permits and reopen a new generation.
"""

from __future__ import annotations

import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from typing import Any, Callable

from relax.engine.inference.discovery import EngineSnapshot, EngineState, RoleSnapshot
from relax.engine.inference.holder import HolderIdentity, holder_alive
from relax.engine.inference.routing import (
    ModelUnavailableError,
    RouteTarget,
    RoutingState,
    select_model,
    select_target,
)
from relax.utils.env import Envs


ADMISSION_CONCURRENCY_GROUP = "inference_admission"
ADMISSION_DRAIN_TIMEOUT_S = 120.0


class AdmissionDrainError(RuntimeError):
    """The manager cannot prove that all request owners have stopped
    sending."""


@dataclass(frozen=True)
class RequestPermit:
    request_id: str
    generation: str
    target: RouteTarget
    holder: HolderIdentity | None = None
    token: str = field(default_factory=lambda: uuid.uuid4().hex)


class AdmissionGate:
    def __init__(self) -> None:
        self._condition = threading.Condition(threading.RLock())
        self._generation = uuid.uuid4().hex
        self._closed = False
        self._permits: dict[str, RequestPermit] = {}
        self._routing = RoutingState()

    def acquire(
        self,
        snapshot: Callable[[], RoleSnapshot],
        request_id: str,
        *,
        model: str | None = None,
        route_key: str | None = None,
        affinity_key: Any = None,
        direct: bool = False,
        holder: HolderIdentity | None = None,
    ) -> RequestPermit:
        with self._condition:
            current = snapshot()
            selected = select_model(current, model, route_key)
            if self._closed:
                raise ModelUnavailableError(f"Model {selected.name!r} is draining or sleeping.")
            if direct and not selected.diagnostic_workers:
                # SFT prediction intentionally balances regular heads directly.
                # PD workers must still be addressed through their router.
                selected = replace(
                    selected,
                    router_url=None,
                    engines=tuple(
                        replace(e, direct_eligible=e.state is EngineState.READY and e.base_url is not None)
                        for e in selected.engines
                    ),
                )
            target = select_target(
                replace(current, models=(selected,), default_model=selected.name),
                self._routing,
                model=selected.name,
                affinity_key=affinity_key,
            )
            permit = RequestPermit(request_id, self._generation, target, holder)
            self._permits[permit.token] = permit
            return permit

    def release(self, permit: RequestPermit) -> None:
        with self._condition:
            if self._permits.get(permit.token) == permit:
                del self._permits[permit.token]
                self._condition.notify_all()

    def close(self) -> None:
        with self._condition:
            self._closed = True

    def drain(self, abort: Callable[[], None], *, timeout: float | None = None) -> None:
        self.close()
        if timeout is None:
            timeout = Envs.RELAX_INFERENCE_ADMISSION_GRACE_S
        with self._condition:
            if self._condition.wait_for(lambda: not self._permits, timeout=timeout):
                return
            outstanding = list(self._permits.values())
        # Liveness queries cannot consume an unbounded extra drain budget.
        deadline = time.monotonic() + Envs.RELAX_INFERENCE_HOLDER_CHECK_S
        executor = ThreadPoolExecutor(max_workers=1)
        try:
            for permit in outstanding:
                if permit.holder is None:
                    continue
                future = executor.submit(holder_alive, permit.holder)
                try:
                    dead = future.result(timeout=max(0.0, deadline - time.monotonic())) is False
                except TimeoutError:
                    break
                if dead:
                    self.release(permit)
        finally:
            executor.shutdown(wait=False, cancel_futures=True)
        # No lock held across engine RPCs; owners must be able to return permits.
        # The engine's pause/abort/idle protocol remains mandatory in offload,
        # even when a dead holder's permit was reclaimed here.
        abort()

    def confirm_drained(self) -> None:
        """Called only after engines confirmed idle and released their
        memory."""
        with self._condition:
            if not self._closed:
                raise AdmissionDrainError("Engine drain receipt requires closed admission")
            self._permits.clear()
            self._condition.notify_all()

    def open(self) -> None:
        with self._condition:
            if not self._closed:
                return
            if self._permits:
                raise AdmissionDrainError("Cannot reopen inference while old request permits remain outstanding.")
            self._generation = uuid.uuid4().hex
            self._closed = False

    def view(self, snapshot: RoleSnapshot) -> RoleSnapshot:
        """Discovery remains diagnostic; acquiring a permit always reads live
        state."""
        with self._condition:
            if not self._closed:
                return snapshot

            def unavailable(engine: EngineSnapshot) -> EngineSnapshot:
                return (
                    replace(engine, state=EngineState.DRAINING, direct_eligible=False)
                    if (engine.state is EngineState.READY)
                    else engine
                )

            return replace(
                snapshot,
                models=tuple(
                    replace(
                        model,
                        state=EngineState.DRAINING if model.state is EngineState.READY else model.state,
                        engines=tuple(unavailable(e) for e in model.engines),
                        diagnostic_workers=tuple(unavailable(e) for e in model.diagnostic_workers),
                    )
                    for model in snapshot.models
                ),
            )
