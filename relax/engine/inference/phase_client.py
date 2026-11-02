# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Clients of the job's authoritative lifecycle ledger.

The context follows custom rollout functions into their worker thread. It is
never inferred from a model's cached READY state or from a training step.
"""

from __future__ import annotations

import asyncio
import contextvars
import time
import uuid
from typing import Any, Awaitable, Iterable

import ray

from relax.engine.inference.client import _settle
from relax.engine.inference.holder import current_holder
from relax.engine.inference.ownership import ActivationLease, OwnershipError
from relax.utils.env import Envs


_CURRENT: contextvars.ContextVar[PhaseSession | None] = contextvars.ContextVar("inference_phase", default=None)


def wait_control(ref: Any, description: str, *, timeout: float | None = None) -> Any:
    try:
        return ray.get(ref, timeout=Envs.RELAX_INFERENCE_CONTROL_TIMEOUT_S if timeout is None else timeout)
    except ray.exceptions.GetTimeoutError as error:
        raise OwnershipError(f"Timed out waiting for {description}; resource release is unconfirmed") from error


async def await_control(call: Awaitable[Any], description: str, *, timeout: float | None = None) -> Any:
    try:
        return await asyncio.wait_for(call, Envs.RELAX_INFERENCE_CONTROL_TIMEOUT_S if timeout is None else timeout)
    except TimeoutError as error:
        raise OwnershipError(f"Timed out waiting for {description}; resource release is unconfirmed") from error


def current_phase() -> PhaseSession:
    session = _CURRENT.get()
    if session is None or session.closed:
        raise OwnershipError("This lifecycle operation requires an active inference phase lease")
    return session


async def control_call(awaitable: Awaitable[Any]) -> Any:
    """Cancellation waits for the control operation, then remains
    cancellation."""
    task = asyncio.ensure_future(awaitable)
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError as cancelled:
        try:
            await _settle(task)
        except Exception:
            raise cancelled from None
        raise


async def gather_owned(awaitables: Iterable[Awaitable[Any]]) -> list[Any]:
    """A failed or cancelled batch cannot leave sibling HTTP/retry tasks
    alive."""
    tasks = [asyncio.ensure_future(item) for item in awaitables]
    try:
        return await asyncio.gather(*tasks)
    except BaseException:
        for task in tasks:
            task.cancel()

        async def finish() -> None:
            await asyncio.gather(*tasks, return_exceptions=True)

        await _settle(asyncio.create_task(finish()))
        raise


class PhaseSession:
    def __init__(
        self,
        authority: Any,
        role: str,
        owner: str,
        *,
        operation_id: str | None = None,
        lease: ActivationLease | None = None,
    ) -> None:
        self.authority = authority
        self.role = role
        self.owner = owner
        self.operation_id = operation_id or uuid.uuid4().hex
        self.lease = lease
        self.closed = False
        self._sequence = 0
        self._token = None
        self._parent = None
        self._safe_to_release = False
        self.holder = current_holder()

    @classmethod
    def from_args(cls, args: Any, role: str, owner: str, **kwargs: Any) -> PhaseSession:
        authority = getattr(args, "_inference_authority", None)
        if authority is None:
            raise OwnershipError("Inference ownership was not initialized by the Controller")
        return cls(authority, role, owner, **kwargs)

    def __enter__(self) -> PhaseSession:
        parent = _CURRENT.get()
        if parent is not None and not parent.closed:
            self._parent = parent
            return parent
        if self.lease is None:
            claims = wait_control(self.authority.claims_for.remote(self.role), f"{self.owner} placement claims")
            self.acquire(claims)
        self._token = _CURRENT.set(self)
        return self

    def acquire(self, claims: Iterable[Any]) -> None:
        try:
            self.lease = wait_control(
                self.authority.acquire.remote(
                    self.owner,
                    self.operation_id,
                    claims,
                    role=self.role,
                    holder=self.holder,
                    expires_at=time.time() + Envs.RELAX_INFERENCE_LEASE_WAIT_S,
                ),
                f"{self.owner} lease grant",
                timeout=Envs.RELAX_INFERENCE_LEASE_WAIT_S + Envs.RELAX_INFERENCE_HOLDER_CHECK_S,
            )
        except BaseException:
            wait_control(
                self.authority.abandon.remote(self.operation_id, self.holder), f"{self.owner} abandoned grant"
            )
            raise

    def __exit__(self, kind: Any, error: Any, traceback: Any) -> None:
        if self._parent is not None:
            return
        try:
            if error is not None and not self.closed and not self._safe_to_release:
                self.block(f"{self.owner} failed before resource release was confirmed: {error}")
            self.close()
        finally:
            if self._token is not None:
                _CURRENT.reset(self._token)

    async def __aenter__(self) -> PhaseSession:
        parent = _CURRENT.get()
        if parent is not None and not parent.closed:
            self._parent = parent
            return parent

        async def acquire() -> ActivationLease:
            claims = await await_control(self.authority.claims_for.remote(self.role), f"{self.owner} placement claims")
            return await await_control(
                self.authority.acquire.remote(
                    self.owner,
                    self.operation_id,
                    claims,
                    role=self.role,
                    holder=self.holder,
                    expires_at=time.time() + Envs.RELAX_INFERENCE_LEASE_WAIT_S,
                ),
                f"{self.owner} lease grant",
                timeout=Envs.RELAX_INFERENCE_LEASE_WAIT_S + Envs.RELAX_INFERENCE_HOLDER_CHECK_S,
            )

        if self.lease is None:
            pending = asyncio.create_task(acquire())
            try:
                self.lease = await asyncio.shield(pending)
            except asyncio.CancelledError as cancelled:
                try:
                    self.lease = await _settle(pending)
                except Exception:
                    await await_control(
                        self.authority.abandon.remote(self.operation_id, self.holder), f"{self.owner} abandoned grant"
                    )
                    self.closed = True
                    raise cancelled from None
                await self.aclose()
                raise
            except Exception:
                await await_control(
                    self.authority.abandon.remote(self.operation_id, self.holder), f"{self.owner} abandoned grant"
                )
                self.closed = True
                raise
        self._token = _CURRENT.set(self)
        return self

    async def __aexit__(self, kind: Any, error: Any, traceback: Any) -> None:
        if self._parent is not None:
            return
        try:
            if error is not None and not self.closed and not self._safe_to_release:
                await self.ablock(f"{self.owner} failed before resource release was confirmed: {error}")
            await self.aclose()
        finally:
            if self._token is not None:
                _CURRENT.reset(self._token)

    def command_id(self, action: str) -> str:
        if self.closed or self.lease is None:
            raise OwnershipError("The phase session has already ended")
        self._sequence += 1
        return f"{self._sequence}:{action}"

    def switch(self, phase: str, stage: str | None = None) -> None:
        command = self.command_id(phase)
        wait_control(self.authority.command.remote(self.lease, command, phase, stage), f"{self.owner} {phase}")
        self._safe_to_release = phase in {"train", "leave_score"}

    async def aswitch(self, phase: str, stage: str | None = None) -> None:
        command = self.command_id(phase)

        async def switch() -> None:
            await await_control(
                self.authority.command.remote(self.lease, command, phase, stage), f"{self.owner} {phase}"
            )

        pending = asyncio.create_task(switch())
        try:
            await asyncio.shield(pending)
            self._safe_to_release = phase in {"train", "leave_score"}
        except asyncio.CancelledError:
            await _settle(pending)
            self._safe_to_release = phase in {"train", "leave_score"}
            raise

    def block(self, reason: str) -> None:
        if not self.closed:
            wait_control(self.authority.block.remote(self.lease, reason), f"{self.owner} quarantine")

    def suspect(self, reason: str) -> None:
        if not self.closed:
            wait_control(self.authority.suspect.remote(self.lease, reason), f"{self.owner} suspect receipt")

    async def ablock(self, reason: str) -> None:
        if not self.closed:

            async def block() -> None:
                await await_control(self.authority.block.remote(self.lease, reason), f"{self.owner} quarantine")

            await _settle(asyncio.create_task(block()))

    def close(self) -> None:
        if not self.closed and self.lease is not None:
            wait_control(self.authority.release.remote(self.lease), f"{self.owner} lease return")
            self.closed = True

    async def aclose(self) -> None:
        if not self.closed and self.lease is not None:

            async def release() -> None:
                await await_control(self.authority.release.remote(self.lease), f"{self.owner} lease return")

            await _settle(asyncio.create_task(release()))
            self.closed = True


class RemoteLifecycleCoordinator:
    """Compatibility vocabulary; every switch uses the current operation
    lease."""

    def __init__(self, *, warm_rollout_kv: bool = True) -> None:
        self.warm_rollout_kv = warm_rollout_kv

    def enter_train(self) -> None:
        current_phase().switch("train")

    def enter_generate(self, stage: Any = None) -> None:
        stage = getattr(stage, "value", stage)
        if stage == "rest" and not self.warm_rollout_kv:
            stage = "rest_without_kv"
        current_phase().switch("generate", stage)

    def enter_score(self) -> None:
        current_phase().switch("score")

    def leave_score(self) -> None:
        current_phase().switch("leave_score")
