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
token passed between actors.
"""

import time
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from threading import Condition, get_ident
from typing import Protocol

from relax.engine.inference.config import InferenceRoleSpec
from relax.engine.inference.phase_plans import PHASE_GENERATE
from relax.engine.inference.placement import PlacementSlice
from relax.engine.inference.types import Role
from relax.utils.logging_utils import get_logger


logger = get_logger(__name__)

# Generation has finished by the time a role is switched out, so nothing should
# still be in flight; the bound turns a request that never completes into a
# failed switch instead of a hung training step.
SWITCH_DRAIN_TIMEOUT_S = 600.0


class LifecyclePort(Protocol):
    """What the coordinator needs from the Manager that owns the engines."""

    def registered_roles(self) -> tuple[Role, ...]: ...

    def role_spec(self, role: Role | str) -> InferenceRoleSpec: ...

    def slices(self, role: Role) -> tuple[PlacementSlice, ...]: ...

    def resident(self, role: Role) -> bool: ...

    def drain(self, roles: Sequence[Role | str], timeout: float = SWITCH_DRAIN_TIMEOUT_S) -> None: ...

    def activate(self, role: Role | str, model_id: str | None = None, tags: list[str] | None = None) -> None: ...

    def deactivate(self, role: Role | str, model_id: str | None = None) -> None: ...


class LifecycleCoordinator:
    """Order the turns of roles that share GPUs; see the module docstring."""

    def __init__(self, manager: LifecyclePort) -> None:
        self._manager = manager
        # Loads (switch, activate, recover) run one at a time, owned by the
        # thread running them, and only once no conflicting role is resident.
        # A separate lock from the Manager's, because a waiting load must not
        # block admission or request completion.
        self._gpus_changed = Condition()
        self._gpu_owner: int | None = None
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
        if self._gpu_owner == get_ident():
            yield
            return
        deadline = time.monotonic() + timeout
        with self._gpus_changed:
            while True:
                blockers = self._blockers(roles, released)
                if release_conflicting:
                    blockers = [role for role in blockers if self._phase(role) in self._held_phases]
                if self._gpu_owner is None and not blockers:
                    break
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    holders = [role.value for role in blockers] or ["another load"]
                    raise TimeoutError(
                        f"GPUs for {[role.value for role in roles]} still held by {holders} after {timeout}s"
                    )
                self._gpus_changed.wait(remaining)
            self._gpu_owner = get_ident()
        try:
            yield
        finally:
            with self._gpus_changed:
                self._gpu_owner = None
                self._gpus_changed.notify_all()

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

        self._switch(
            outgoing,
            incoming,
            timeout,
            f"Entering phase {phase_id}",
            release_conflicting=True,
            hold=phase_id if phase_id != PHASE_GENERATE else None,
        )

    def leave_phase(self, phase_id: str, timeout: float = SWITCH_DRAIN_TIMEOUT_S) -> None:
        """Drain and release the resident roles placed in ``phase_id``.

        Does not restore generation: the next weight sync activates it anyway.
        """

        def outgoing() -> list[Role]:
            # Read under the lease, so a role of the phase activated meanwhile
            # is released too.
            return [role for role in self._roles_in_phase(phase_id) if self._manager.resident(role)]

        try:
            self._switch(outgoing, [], timeout, f"Leaving phase {phase_id}")
        finally:
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
        self._switch(lambda: deactivate, activate, timeout, "Switching", released=deactivate)

    def _switch(
        self,
        outgoing: Callable[[], list[Role]],
        activate: Sequence[Role | str],
        timeout: float,
        reason: str,
        *,
        released: Sequence[Role] = (),
        release_conflicting: bool = False,
        hold: str | None = None,
    ) -> None:
        """``switch`` with the outgoing roles read once the lease is held;
        ``hold`` marks that phase held before the lease is given up."""
        activate = [Role(role) for role in activate]
        clashing = [
            (a.value, b.value) for i, a in enumerate(activate) for b in activate[i + 1 :] if self.conflict(a, b)
        ]
        if clashing:
            raise ValueError(f"Roles that share GPUs cannot be active together: {clashing}")
        with self.lease(activate, released=released, release_conflicting=release_conflicting, timeout=timeout):
            deactivate = outgoing()
            logger.info(
                f"{reason}: release {[role.value for role in deactivate]}, activate {[role.value for role in activate]}"
            )
            self._manager.drain(deactivate, timeout)
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
