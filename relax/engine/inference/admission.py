# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Manager-owned request admission, shared by proxy and lookaside clients.

The condition protects admission and the start of a drain, not GPU operations.
An unreturned permit is never expired: a client could still be about to send.
Only the owner can release it. An uncertain drain therefore fails closed.
"""

from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass, replace
from typing import Any, Callable

from relax.engine.inference.discovery import EngineSnapshot, EngineState, RoleSnapshot
from relax.engine.inference.routing import (
    ModelUnavailableError,
    RouteTarget,
    RoutingState,
    select_model,
    select_target,
)


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
    ) -> RequestPermit:
        with self._condition:
            if request_id in self._permits:
                return self._permits[request_id]
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
            permit = RequestPermit(request_id, self._generation, target)
            self._permits[request_id] = permit
            return permit

    def release(self, permit: RequestPermit) -> None:
        with self._condition:
            if self._permits.get(permit.request_id) == permit:
                del self._permits[permit.request_id]
                self._condition.notify_all()

    def close(self) -> None:
        with self._condition:
            self._closed = True

    def drain(self, abort: Callable[[], None], *, timeout: float = ADMISSION_DRAIN_TIMEOUT_S) -> None:
        self.close()
        with self._condition:
            if self._condition.wait_for(lambda: not self._permits, timeout=timeout):
                return
        # No lock held across engine RPCs; owners must be able to return permits.
        abort()
        with self._condition:
            if not self._condition.wait_for(lambda: not self._permits, timeout=timeout):
                raise AdmissionDrainError(
                    f"Cannot drain inference: {len(self._permits)} request owner(s) have not returned their permits."
                )

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
