# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Serialize shared-GPU handoffs without duplicating role residency state."""

import asyncio
from collections.abc import Awaitable, Callable, Mapping
from typing import Any


class PhaseCoordinator:
    """Coordinate role owners; each Manager retains its admission ledger.

    ``initial_phase`` must describe the startup layout already established by
    the caller. Roles in other phases must have completed their initial
    offload. A failed transition deliberately blocks subsequent handoffs until
    recovery reconstructs the coordinator from confirmed Manager state.
    """

    def __init__(
        self,
        phases: Mapping[str, tuple[Any, ...]],
        initial_phase: str,
        call: Callable[[Any, str], Awaitable[None]],
    ) -> None:
        if initial_phase not in phases:
            raise ValueError(f"Unknown initial phase: {initial_phase}")
        self._phases = dict(phases)
        self._phase = initial_phase
        self._call = call
        self._lock = asyncio.Lock()
        self._failure: BaseException | None = None

    async def switch(self, phase: str) -> None:
        if phase not in self._phases:
            raise ValueError(f"Unknown phase: {phase}")
        async with self._lock:
            if self._failure is not None:
                if phase == "startup" and not self._phases[phase]:
                    # Failed onload may have changed residency without advancing
                    # the phase. Teardown must retry every possible owner, while
                    # generation remains blocked until the task is rebuilt.
                    seen = set()
                    for name, owners in self._phases.items():
                        if name == "resident":
                            continue
                        for owner in owners:
                            if id(owner) not in seen:
                                seen.add(id(owner))
                                await self._call(owner, "offload")
                    return
                raise RuntimeError("Shared GPU handoff requires recovery") from self._failure
            if phase == self._phase:
                return
            transition = asyncio.create_task(self._switch(phase))
            cancelled = False
            while not transition.done():
                try:
                    await asyncio.shield(transition)
                except asyncio.CancelledError:
                    cancelled = True
            transition.result()
            if cancelled:
                raise asyncio.CancelledError

    async def _switch(self, phase: str) -> None:
        try:
            # Each offload closes admission, drains permits and confirms release.
            # Never activate any next-phase owner before all releases succeed.
            for owner in self._phases[self._phase]:
                await self._call(owner, "offload")
            for owner in self._phases[phase]:
                await self._call(owner, "onload")
        except BaseException as exc:
            self._failure = exc
            raise
        self._phase = phase


def create_phase_coordinator(phases: Mapping[str, tuple[Any, ...]], initial_phase: str) -> Any:
    """Configure the lifecycle inside an existing task actor; create no
    actor."""
    import ray

    from relax.distributed.ray.inference_handle import InferenceRoleHandle

    task = None
    keys = {}
    for phase, owners in phases.items():
        entries = []
        for owner in owners:
            candidate = owner.task if isinstance(owner, InferenceRoleHandle) else owner
            if task is not None and candidate != task:
                raise ValueError("All inference roles must belong to one task Manager")
            task = candidate
            entries.append(
                (owner.role, owner.model_id) if isinstance(owner, InferenceRoleHandle) else ("rollout", "default")
            )
        keys[phase] = tuple(entries)
    if task is None:
        raise ValueError("Task phases require an existing inference owner")
    ray.get(task.configure_phases.remote(keys, initial_phase))
    return task
