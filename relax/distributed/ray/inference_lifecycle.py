# Copyright (c) 2026 Relax Authors. All Rights Reserved.


from __future__ import annotations

import asyncio
import math
from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import ray

from relax.inference.specs import INFERENCE_ROLES


_COMMITTED_HISTORY_LIMIT = 1024
_OPERATION_HISTORY_LIMIT = 8192


class InferenceLifecycleError(RuntimeError):
    pass


class InferenceBatchError(RuntimeError):
    pass


@dataclass
class _Batch:
    state: str = "PENDING"
    error: str | None = None
    rollout_released: bool = False
    student_scoring_started: bool = False


def _consume_result(future: asyncio.Future) -> None:
    if not future.cancelled():
        future.exception()


class LifecycleCoordinatorState:
    def __init__(self, operation_timeout: float = 180.0) -> None:
        if not math.isfinite(operation_timeout) or operation_timeout <= 0:
            raise ValueError("operation_timeout must be finite and positive")
        self._operation_timeout = operation_timeout
        self._condition = asyncio.Condition()
        self._managers: dict[str, tuple[Any, ...]] = {}
        self._roles: dict[str, str] = {"rollout": "SLEEPING"}
        self._batches: dict[int | str, _Batch] = {}
        self._current: int | str | None = None
        self._failure: str | None = None
        self._operations: dict[str, tuple[tuple[int | str, str, str], asyncio.Task]] = {}
        self._committed: OrderedDict[int | str, None] = OrderedDict()
        self._retired_operations: OrderedDict[str, tuple[int | str, str, str]] = OrderedDict()

    def _remember_committed(self, rollout_id: int | str) -> None:
        self._committed[rollout_id] = None
        self._committed.move_to_end(rollout_id)
        while len(self._committed) > _COMMITTED_HISTORY_LIMIT:
            self._committed.popitem(last=False)

    def _evict_committed(self) -> None:
        for batch_id in [batch_id for batch_id, batch in self._batches.items() if batch.state == "COMMITTED"]:
            del self._batches[batch_id]
        retained = {}
        for operation_id, (fingerprint, task) in self._operations.items():
            if fingerprint[0] in self._batches or not task.done():
                retained[operation_id] = (fingerprint, task)
            else:
                self._retired_operations[operation_id] = fingerprint
        self._operations = retained
        while len(self._retired_operations) > _OPERATION_HISTORY_LIMIT:
            self._retired_operations.popitem(last=False)

    @staticmethod
    def _validate_batch_id(rollout_id: int | str) -> None:
        if isinstance(rollout_id, bool) or not isinstance(rollout_id, (int, str)):
            raise ValueError("rollout_id must be a nonnegative integer or a nonempty string")
        if isinstance(rollout_id, int) and rollout_id < 0:
            raise ValueError("rollout_id must be nonnegative")
        if isinstance(rollout_id, str) and (not rollout_id or rollout_id.strip() != rollout_id):
            raise ValueError("rollout_id must be nonempty without surrounding whitespace")

    def _current_batch(self, rollout_id: int | str, *, allow_failed: bool = False) -> _Batch:
        if rollout_id != self._current or rollout_id not in self._batches:
            raise InferenceLifecycleError(f"Batch {rollout_id!r} is not the current batch")
        batch = self._batches[rollout_id]
        if batch.state == "FAILED" and not allow_failed:
            raise InferenceBatchError(batch.error)
        if batch.state == "COMMITTED":
            raise InferenceLifecycleError(f"Batch {rollout_id!r} has already committed")
        return batch

    def _fail(self, rollout_id: int | str, message: str) -> None:
        batch = self._batches[rollout_id]
        if batch.state != "FAILED":
            batch.state, batch.error = "FAILED", message
        self._failure = batch.error
        self._condition.notify_all()

    async def register(self, role: str, managers: Sequence[Any]) -> None:
        if role not in INFERENCE_ROLES:
            raise ValueError(f"Unknown inference role: {role!r}")
        if not isinstance(managers, Sequence) or isinstance(managers, (str, bytes)) or not managers:
            raise ValueError("Each registered role requires at least one manager")
        handles = tuple(managers)
        for manager in handles:
            for method in ("onload", "offload"):
                if not callable(getattr(getattr(manager, method, None), "remote", None)):
                    raise ValueError(f"Manager must expose {method}.remote()")
        async with self._condition:
            if role in self._managers:
                if self._managers[role] != handles:
                    raise InferenceLifecycleError(f"Managers for {role} have already been registered")
                return
            if self._batches:
                raise InferenceLifecycleError("Register every role before the first batch")
            self._managers[role] = handles
            self._roles[role] = "SLEEPING"

    async def begin_batch(self, rollout_id: int | str) -> None:
        self._validate_batch_id(rollout_id)
        async with self._condition:
            if rollout_id in self._batches:
                batch = self._batches[rollout_id]
                if batch.state == "FAILED":
                    raise InferenceBatchError(batch.error)
                return
            if rollout_id in self._committed:
                return
            if self._failure is not None:
                raise InferenceBatchError(self._failure)
            if self._current is not None and self._batches[self._current].state != "COMMITTED":
                raise InferenceLifecycleError("Only one pending closed batch may own the shared GPUs")
            if any(state != "SLEEPING" for state in self._roles.values()):
                raise InferenceLifecycleError("Previous inference occupants have not released their leases")
            self._evict_committed()
            self._batches[rollout_id] = _Batch()
            self._current = rollout_id
            self._roles["rollout"] = "ACTIVE"
            self._condition.notify_all()

    async def activate(self, role: str, operation_id: str) -> None:
        await self._operate(role, operation_id, "onload")

    async def deactivate(self, role: str, operation_id: str) -> None:
        await self._operate(role, operation_id, "offload")

    async def _operate(self, role: str, operation_id: str, method: str) -> None:
        if not isinstance(operation_id, str) or not operation_id or operation_id.strip() != operation_id:
            raise ValueError("operation_id must be nonempty without surrounding whitespace")
        async with self._condition:
            if self._current is None:
                raise InferenceLifecycleError("Begin a batch before changing inference phases")
            fingerprint = (self._current, method, role)
            previous = self._operations.get(operation_id)
            retired = self._retired_operations.get(operation_id)
            if retired is not None and retired != fingerprint:
                raise InferenceLifecycleError(f"Operation ID {operation_id!r} has a different fingerprint")
            if previous is not None:
                if previous[0] != fingerprint:
                    raise InferenceLifecycleError(f"Operation ID {operation_id!r} has a different fingerprint")
                task = previous[1]
            else:
                self._current_batch(self._current, allow_failed=method == "offload")
                if role not in self._managers:
                    raise InferenceLifecycleError(f"No managers registered for {role!r}")
                state = self._roles[role]
                if state not in {"SLEEPING", "ACTIVE"}:
                    raise InferenceLifecycleError(f"Cannot {method} {role} in {state}; lease remains reserved")
                if method == "onload":
                    conflicts = [name for name, value in self._roles.items() if name != role and value != "SLEEPING"]
                    if conflicts:
                        raise InferenceLifecycleError(f"Cannot activate {role}; leases held by {', '.join(conflicts)}")
                target = "ACTIVE" if method == "onload" else "SLEEPING"
                already_done = state == target
                if not already_done:
                    self._roles[role] = "ONLOADING" if method == "onload" else "DRAINING"
                task = asyncio.create_task(self._transition(self._current, role, method, already_done))
                task.add_done_callback(_consume_result)
                self._operations[operation_id] = (fingerprint, task)
        await asyncio.shield(task)

    async def _transition(self, rollout_id: int | str, role: str, method: str, already_done: bool) -> None:
        if already_done:
            return

        async def invoke(manager: Any) -> None:
            result = await getattr(manager, method).remote()
            if result is False:
                raise InferenceLifecycleError(f"{role} manager rejected {method}")

        group = asyncio.gather(*(invoke(manager) for manager in self._managers[role]))
        group.add_done_callback(_consume_result)
        try:
            await asyncio.wait_for(asyncio.shield(group), timeout=self._operation_timeout)
        except (Exception, asyncio.CancelledError) as exc:
            detail = str(exc) or type(exc).__name__
            message = f"{role} {method} failed; lease retained: {detail}"
            async with self._condition:
                self._roles[role] = "BLOCKED"
                self._fail(rollout_id, message)
            raise InferenceLifecycleError(message) from exc
        async with self._condition:
            self._roles[role] = "ACTIVE" if method == "onload" else "SLEEPING"
            self._condition.notify_all()

    async def acknowledge_rollout_offloaded(self, rollout_id: int | str) -> None:
        self._validate_batch_id(rollout_id)
        async with self._condition:
            batch = self._current_batch(rollout_id, allow_failed=True)
            if batch.rollout_released:
                return
            if self._roles["rollout"] not in {"ACTIVE", "SLEEPING"}:
                raise InferenceLifecycleError("Cannot acknowledge an uncertain rollout transition")
            self._roles["rollout"] = "SLEEPING"
            batch.rollout_released = True
            self._condition.notify_all()

    async def begin_student_scoring(self, rollout_id: int | str) -> None:
        self._validate_batch_id(rollout_id)
        async with self._condition:
            batch = self._current_batch(rollout_id)
            if batch.student_scoring_started:
                return
            if not batch.rollout_released or any(state != "SLEEPING" for state in self._roles.values()):
                raise InferenceLifecycleError("Student scoring requires every inference role to be sleeping")
            batch.student_scoring_started = True
            batch.rollout_released = False
            self._roles["rollout"] = "ACTIVE"
            self._condition.notify_all()

    async def commit_batch(self, rollout_id: int | str) -> None:
        self._validate_batch_id(rollout_id)
        async with self._condition:
            batch = self._batches.get(rollout_id)
            if (batch is not None and batch.state == "COMMITTED") or rollout_id in self._committed:
                return
            batch = self._current_batch(rollout_id)
            occupants = [role for role, state in self._roles.items() if state != "SLEEPING"]
            if occupants:
                raise InferenceLifecycleError(f"Cannot commit before inference offload: {', '.join(occupants)}")
            batch.state = "COMMITTED"
            self._remember_committed(rollout_id)
            self._condition.notify_all()

    async def fail_batch(self, rollout_id: int | str, message: str) -> None:
        self._validate_batch_id(rollout_id)
        if not isinstance(message, str) or not message.strip():
            raise ValueError("Batch failure requires an error message")
        async with self._condition:
            self._current_batch(rollout_id, allow_failed=True)
            self._fail(rollout_id, message)

    async def wait_committed(self, rollout_id: int | str, timeout: float | None = None) -> None:
        self._validate_batch_id(rollout_id)
        if timeout is not None and (not math.isfinite(timeout) or timeout <= 0):
            raise ValueError("timeout must be finite and positive")

        async def wait() -> None:
            async with self._condition:
                while True:
                    batch = self._batches.get(rollout_id)
                    if (batch is not None and batch.state == "COMMITTED") or rollout_id in self._committed:
                        return
                    if self._failure is not None:
                        raise InferenceBatchError(self._failure)
                    await self._condition.wait()

        await asyncio.wait_for(wait(), timeout=timeout)

    async def get_state(self) -> dict[str, Any]:
        async with self._condition:
            batch = self._batches.get(self._current)
            return {
                "rollout_id": self._current,
                "batch_state": batch.state if batch is not None else None,
                "error": self._failure,
                "roles": dict(self._roles),
            }


InferenceLifecycleCoordinator = ray.remote(num_cpus=1, num_gpus=0)(LifecycleCoordinatorState)
