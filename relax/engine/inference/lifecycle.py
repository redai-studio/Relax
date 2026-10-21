# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Phase order, drain, activation lease and barriers for shared GPUs.

The ``LifecycleCoordinator`` decides *when* inference roles hold their GPUs;
the ``InferenceManager`` it drives only knows *how* to activate, drain,
deactivate and shut down a role's engines. Roles placed on the same bundles in
different phases take turns, and the coordinator is the only place that orders
those turns:

- the activation lease lets one load run at a time, and only once no
  conflicting role is resident;
- a phase transition drains and deactivates the outgoing roles before the
  incoming ones are activated, and gives the GPUs back if activation fails;
- which roles belong to a phase comes from their deployment, not from callers;
- the rollout release barrier tells training when generation has let go.

It runs in the Manager's process, so the lease is a plain lock rather than a
token passed between actors. A thread takes it with ``lease``; the Manager's
async actor takes it with ``acquire`` on an event loop, so a load waiting for
its GPUs holds no thread, and hands it to the thread that runs the load with
``owned``.
"""

import asyncio
import functools
import threading
import time
from collections.abc import Awaitable, Callable, Iterator, Sequence
from contextlib import contextmanager
from typing import Any, Protocol

from relax.engine.inference.config import InferenceRoleSpec
from relax.engine.inference.phase_plans import PHASE_GENERATE
from relax.engine.inference.placement import PlacementSlice
from relax.engine.inference.types import Role
from relax.utils.logging_utils import get_logger


logger = get_logger(__name__)

# Generation has finished by the time a role is switched out, so nothing should
# still be in flight. A load waiting longer than this for its GPUs fails; a
# drain logs a warning every time this elapses, or aborts the requests when
# asked to.
SWITCH_DRAIN_TIMEOUT_S = 600.0

# Awaits the drain of the roles (``roles, timeout``); the async ``Manager.drain``.
_AsyncDrain = Callable[[Sequence[Role], float], Awaitable[None]]
# Runs a blocking call on a thread, then the ``releases`` callbacks on that
# thread once the call is done.
_BlockingRun = Callable[..., Awaitable[Any]]


class LoopAwareCondition:
    """A condition that threads and event loops both wait on.

    Threads use it like ``threading.Condition``. A coroutine awaits
    :meth:`wait_for` instead, which holds no thread while it waits;
    :meth:`notify_all`, from any thread, wakes both kinds of waiter.
    """

    def __init__(self, lock: Any = None) -> None:
        self._cond = threading.Condition(lock)
        self._waiters: set[tuple[asyncio.AbstractEventLoop, asyncio.Future]] = set()

    def __enter__(self) -> bool:
        return self._cond.__enter__()

    def __exit__(self, *exc: Any) -> None:
        self._cond.__exit__(*exc)

    def wait(self, timeout: float | None = None) -> bool:
        return self._cond.wait(timeout)

    def notify_all(self) -> None:
        """Wake every waiter; call with the lock held."""
        self._cond.notify_all()
        for loop, future in self._waiters:
            loop.call_soon_threadsafe(_wake, future)

    async def wait_for(self, predicate: Callable[[], bool], deadline: float | None = None) -> bool:
        """Wait until ``predicate``, checked under the lock, holds; False once
        ``deadline`` (``time.monotonic``) passes first.

        The check and whatever ``predicate`` claims when it holds happen under
        the lock, so a waiter that returns True already owns what it waited
        for.
        """
        loop = asyncio.get_running_loop()
        while True:
            with self._cond:
                if predicate():
                    return True
                remaining = None if deadline is None else deadline - time.monotonic()
                if remaining is not None and remaining <= 0:
                    return False
                waiter = (loop, loop.create_future())
                self._waiters.add(waiter)
            try:
                await asyncio.wait_for(waiter[1], remaining)
            except asyncio.TimeoutError:
                pass
            finally:
                with self._cond:
                    self._waiters.discard(waiter)


def _wake(future: asyncio.Future) -> None:
    if not future.done():
        future.set_result(None)


class LifecyclePort(Protocol):
    """What the coordinator needs from the Manager that owns the engines."""

    def registered_roles(self) -> tuple[Role, ...]: ...

    def role_spec(self, role: Role | str) -> InferenceRoleSpec: ...

    def slices(self, role: Role) -> tuple[PlacementSlice, ...]: ...

    def resident(self, role: Role) -> bool: ...

    def drain(
        self, roles: Sequence[Role | str], timeout: float = SWITCH_DRAIN_TIMEOUT_S, force_abort: bool = False
    ) -> None: ...

    def activate(self, role: Role | str, model_id: str | None = None, tags: list[str] | None = None) -> None: ...

    def deactivate(self, role: Role | str, model_id: str | None = None) -> None: ...


class LifecycleCoordinator:
    """Order the turns of roles that share GPUs; see the module docstring."""

    def __init__(self, manager: LifecyclePort) -> None:
        self._manager = manager
        # Loads (switch, activate, recover) run one at a time, under a lease
        # token, and only once no conflicting role is resident. A thread owns
        # the token it runs under, so a load nested in it runs directly. A
        # separate lock from the Manager's, because a waiting load must not
        # block admission or request completion.
        self._gpus_changed = LoopAwareCondition()
        self._gpu_owner: object | None = None
        self._local = threading.local()
        # Scoring phases entered and not yet left. Their roles are waited for,
        # never taken over, by another phase; generation has no such block.
        self._held_phases: set[str] = set()

    # ------------------------------------------------------------------
    # Activation lease.
    # ------------------------------------------------------------------

    @contextmanager
    def lease(
        self,
        roles: Sequence[Role],
        *,
        released: Sequence[Role] = (),
        release_conflicting: bool = False,
        timeout: float = SWITCH_DRAIN_TIMEOUT_S,
    ) -> Iterator[None]:
        """Own the GPUs while ``roles`` load.

        Waits until no other load runs and no role that conflicts with
        ``roles`` is resident, apart from ``released``, which the caller frees
        first. With ``release_conflicting`` the caller frees every conflicting
        role it finds once it holds the lease, so only other loads and roles
        inside a held scoring phase are waited for. A load nested in the owning
        thread runs directly.
        """
        if self._owns_lease():
            yield
            return
        token = object()
        deadline = time.monotonic() + timeout
        with self._gpus_changed:
            while not self._claim(token, roles, released, release_conflicting):
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise self._lease_timeout(roles, released, release_conflicting, timeout)
                self._gpus_changed.wait(remaining)
        try:
            with self.adopted(token):
                yield
        finally:
            self.release(token)

    async def acquire(
        self,
        roles: Sequence[Role],
        *,
        released: Sequence[Role] = (),
        release_conflicting: bool = False,
        timeout: float = SWITCH_DRAIN_TIMEOUT_S,
    ) -> object:
        """``lease`` for an event loop: wait without a thread and return the
        lease token.

        Run the load with :meth:`owned` and give the token back with
        :meth:`release`.
        """
        token = object()
        claimed = await self._gpus_changed.wait_for(
            lambda: self._claim(token, roles, released, release_conflicting), time.monotonic() + timeout
        )
        if not claimed:
            with self._gpus_changed:
                raise self._lease_timeout(roles, released, release_conflicting, timeout)
        return token

    def release(self, token: object) -> None:
        """Give the lease back; a token that no longer owns it is ignored."""
        with self._gpus_changed:
            if self._gpu_owner is token:
                self._gpu_owner = None
                self._gpus_changed.notify_all()

    @contextmanager
    def adopted(self, token: object) -> Iterator[None]:
        """Run the calling thread under ``token``, so loads it nests run
        directly."""
        previous = getattr(self._local, "token", None)
        self._local.token = token
        try:
            yield
        finally:
            self._local.token = previous

    def owned(self, token: object, fn: Callable[..., Any]) -> Callable[..., Any]:
        """``fn`` run under ``token`` on whichever thread calls it."""

        @functools.wraps(fn)
        def run(*args: Any, **kwargs: Any) -> Any:
            with self.adopted(token):
                return fn(*args, **kwargs)

        return run

    def _owns_lease(self) -> bool:
        token = getattr(self._local, "token", None)
        return token is not None and token is self._gpu_owner

    def _claim(
        self, token: object, roles: Sequence[Role], released: Sequence[Role], release_conflicting: bool
    ) -> bool:
        """Take the lease for ``token`` if it is free; call under the lock."""
        if self._gpu_owner is None and not self._lease_blockers(roles, released, release_conflicting):
            self._gpu_owner = token
            return True
        return False

    def _lease_blockers(
        self, roles: Sequence[Role], released: Sequence[Role], release_conflicting: bool
    ) -> list[Role]:
        blockers = self._blockers(roles, released)
        if release_conflicting:
            blockers = [role for role in blockers if self._phase(role) in self._held_phases]
        return blockers

    def _lease_timeout(
        self, roles: Sequence[Role], released: Sequence[Role], release_conflicting: bool, timeout: float
    ) -> TimeoutError:
        holders = [role.value for role in self._lease_blockers(roles, released, release_conflicting)]
        return TimeoutError(
            f"GPUs for {[role.value for role in roles]} still held by {holders or ['another load']} after {timeout}s"
        )

    def notify_released(self) -> None:
        """Wake loads waiting for GPUs after a role released its memory."""
        with self._gpus_changed:
            self._gpus_changed.notify_all()

    # ------------------------------------------------------------------
    # Phase order.
    # ------------------------------------------------------------------

    def enter_phase(self, phase_id: str, timeout: float = SWITCH_DRAIN_TIMEOUT_S) -> None:
        """Give the GPUs to the roles placed in ``phase_id``.

        Only roles that take turns with another role move; a role that keeps
        its own GPUs is left alone. Resident roles that share GPUs with an
        incoming role are drained and deactivated first; they are read once the
        lease is held, so a role another switch activated meanwhile is released
        too. A role inside another scoring phase that has not been left yet is
        waited for instead. A scoring phase is held until ``leave_phase``;
        generation is not.
        """
        self._switch(*self._enter_plan(phase_id), timeout)

    async def enter_phase_async(
        self, phase_id: str, drain: _AsyncDrain, run: _BlockingRun, timeout: float = SWITCH_DRAIN_TIMEOUT_S
    ) -> None:
        """``enter_phase`` for an event loop; see ``_switch_async``."""
        await self._switch_async(*self._enter_plan(phase_id), timeout, drain, run)

    def _enter_plan(self, phase_id: str) -> tuple[Callable[[], list[Role]], list[Role], str, dict[str, Any]]:
        registered = self._manager.registered_roles()
        incoming = [
            role for role in self._roles_in_phase(phase_id) if any(self.conflict(role, other) for other in registered)
        ]
        if not incoming:
            raise ValueError(f"No role takes turns in phase {phase_id!r}")

        def outgoing() -> list[Role]:
            # Read under the lease: a concurrent switch may have activated a
            # conflicting role while this one waited for it.
            return [
                other
                for other in self._manager.registered_roles()
                if other not in incoming
                and self._manager.resident(other)
                and any(self.conflict(other, role) for role in incoming)
            ]

        options = {"release_conflicting": True, "hold": phase_id if phase_id != PHASE_GENERATE else None}
        return outgoing, incoming, f"Entering phase {phase_id}", options

    def leave_phase(self, phase_id: str, timeout: float = SWITCH_DRAIN_TIMEOUT_S) -> None:
        """Drain and release the resident roles placed in ``phase_id``.

        Does not restore generation: the next weight sync activates it anyway.
        """
        try:
            self._switch(*self._leave_plan(phase_id), timeout)
        finally:
            self._end_hold(phase_id)

    async def leave_phase_async(
        self, phase_id: str, drain: _AsyncDrain, run: _BlockingRun, timeout: float = SWITCH_DRAIN_TIMEOUT_S
    ) -> None:
        """``leave_phase`` for an event loop; see ``_switch_async``."""
        try:
            await self._switch_async(*self._leave_plan(phase_id), timeout, drain, run)
        finally:
            self._end_hold(phase_id)

    def _leave_plan(self, phase_id: str) -> tuple[Callable[[], list[Role]], list[Role], str, dict[str, Any]]:
        def outgoing() -> list[Role]:
            # Read under the lease, so a role of the phase activated meanwhile
            # is released too.
            return [role for role in self._roles_in_phase(phase_id) if self._manager.resident(role)]

        return outgoing, [], f"Leaving phase {phase_id}", {}

    def _end_hold(self, phase_id: str) -> None:
        # Even a failed release ends the block; the next phase retries it.
        with self._gpus_changed:
            self._held_phases.discard(phase_id)
            self._gpus_changed.notify_all()

    def switch(
        self,
        deactivate: Sequence[Role | str],
        activate: Sequence[Role | str],
        timeout: float = SWITCH_DRAIN_TIMEOUT_S,
    ) -> None:
        """Hand shared GPUs from ``deactivate`` roles to ``activate`` roles.

        The outgoing roles stop admitting, finish their requests and release
        their memory before any incoming role is loaded, so two roles never
        hold the same GPUs at once. Like every load it runs alone, and waits
        while a role outside ``deactivate`` that conflicts with an incoming one
        is resident. Every step is idempotent.
        """
        deactivate = [Role(role) for role in deactivate]
        self._switch(lambda: deactivate, activate, "Switching", {"released": deactivate}, timeout)

    def _switch(
        self,
        outgoing: Callable[[], list[Role]],
        activate: Sequence[Role | str],
        reason: str,
        options: dict[str, Any],
        timeout: float,
    ) -> None:
        """``switch`` with the outgoing roles read once the lease is held.

        ``options`` holds the lease's ``released`` and ``release_conflicting``,
        and ``hold``, the phase marked held before the lease is given up.
        """
        activate = self._incoming(activate)
        hold = options.get("hold")
        lease_options = {key: value for key, value in options.items() if key != "hold"}
        with self.lease(activate, timeout=timeout, **lease_options):
            deactivate = outgoing()
            self._log_switch(reason, deactivate, activate)
            self._manager.drain(deactivate, timeout)
            self._hand_over(deactivate, activate, hold)

    async def _switch_async(
        self,
        outgoing: Callable[[], list[Role]],
        activate: Sequence[Role | str],
        reason: str,
        options: dict[str, Any],
        timeout: float,
        drain: _AsyncDrain,
        run: _BlockingRun,
    ) -> None:
        """``_switch`` for an event loop: the lease and the drain are awaited,
        and only the hand-over runs on a thread, through ``run``.

        ``run`` gives the lease back once that thread is done; until the hand-
        over is handed to it, this does. A switch cancelled while draining
        leaves the outgoing roles closed.
        """
        activate = self._incoming(activate)
        hold = options.get("hold")
        lease_options = {key: value for key, value in options.items() if key != "hold"}
        token = await self.acquire(activate, timeout=timeout, **lease_options)
        try:
            deactivate = outgoing()
            self._log_switch(reason, deactivate, activate)
            await drain(deactivate, timeout)
        except BaseException:
            self.release(token)
            raise
        await run(
            self.owned(token, self._hand_over),
            deactivate,
            activate,
            hold,
            releases=(functools.partial(self.release, token),),
        )

    def _incoming(self, activate: Sequence[Role | str]) -> list[Role]:
        activate = [Role(role) for role in activate]
        clashing = [
            (a.value, b.value) for i, a in enumerate(activate) for b in activate[i + 1 :] if self.conflict(a, b)
        ]
        if clashing:
            raise ValueError(f"Roles that share GPUs cannot be active together: {clashing}")
        return activate

    @staticmethod
    def _log_switch(reason: str, deactivate: Sequence[Role], activate: Sequence[Role]) -> None:
        logger.info(
            f"{reason}: release {[role.value for role in deactivate]}, activate {[role.value for role in activate]}"
        )

    def _hand_over(self, deactivate: Sequence[Role], activate: Sequence[Role], hold: str | None) -> None:
        """Release the drained outgoing roles, then load the incoming ones; run
        under the lease."""
        for role in deactivate:
            self._manager.deactivate(role)
        try:
            for role in activate:
                self._manager.activate(role)
        except Exception:
            # An incoming role that failed to load gives its GPUs back.
            for role in activate:
                try:
                    self._manager.deactivate(role)
                except Exception as exc:
                    logger.warning(f"Failed to release {role.value} after a failed switch: {exc}")
            raise
        if hold is not None:
            with self._gpus_changed:
                self._held_phases.add(hold)

    def current_phase(self, default: str) -> str:
        """The phase of the resident role that takes turns outside ``default``,
        if any."""
        for role in self._manager.registered_roles():
            phase = self._manager.role_spec(role).deployment.phase
            if phase != default and self._manager.resident(role):
                return phase
        return default

    # ------------------------------------------------------------------
    # Barriers.
    # ------------------------------------------------------------------

    def rollout_released(self) -> bool:
        """Whether generation has let go of its GPUs for training.

        Follows residency, not the rollout status: a model is resident from the
        start of a load (including a weights-only one) until its offload has
        succeeded, so a load in progress or a failed offload keeps the barrier
        closed.
        """
        return not self._manager.resident(Role.ROLLOUT)

    # ------------------------------------------------------------------
    # GPU sharing.
    # ------------------------------------------------------------------

    def conflict(self, role: Role, other: Role) -> bool:
        """Whether two roles are placed on the same GPUs in different phases,
        so they may only take turns."""
        return role is not other and any(
            mine.placement_group_key == theirs.placement_group_key
            and mine.phase != theirs.phase
            and mine.reserved_offset < theirs.reserved_offset + theirs.reserved_size
            and theirs.reserved_offset < mine.reserved_offset + mine.reserved_size
            for mine in self._manager.slices(role)
            for theirs in self._manager.slices(other)
        )

    def _phase(self, role: Role) -> str:
        return self._manager.role_spec(role).deployment.phase

    def _roles_in_phase(self, phase_id: str) -> list[Role]:
        return [
            role
            for role in self._manager.registered_roles()
            if self._manager.role_spec(role).deployment.phase == phase_id
        ]

    def _blockers(self, roles: Sequence[Role], released: Sequence[Role]) -> list[Role]:
        return [
            other
            for other in self._manager.registered_roles()
            if other not in released
            and other not in roles
            and any(self.conflict(role, other) for role in roles)
            and self._manager.resident(other)
        ]
