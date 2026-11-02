# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Job-scoped ownership of inference resources.

Request admission is a manager-local operation. This authority handles the much
less frequent handoffs between workloads, training and maintenance. A lease has
no expiry: loss of a caller is not evidence that its GPU work has stopped.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass
from typing import Any, Iterable

import ray

from relax.distributed.ray.placement_planner import ACTOR_POOL, Phase, PlacementClaim, PlacementPlan, claims_overlap
from relax.engine.inference.holder import HolderIdentity, holder_alive
from relax.utils.env import Envs


_TRAINING_ROLES = frozenset({"actor", "critic", "reference", "actor_fwd"})


class OwnershipError(RuntimeError):
    """A resource operation cannot establish exclusive ownership."""


class StaleLeaseError(OwnershipError):
    """The operation belongs to an old job or an already completed lease."""


class ResourceBlockedError(OwnershipError):
    """A previous owner has not confirmed release of the resource."""


@dataclass(frozen=True)
class ActivationLease:
    epoch: str
    owner: str
    operation_id: str
    generation: int
    claims: tuple[PlacementClaim, ...]
    role: str | None = None
    holder: HolderIdentity | None = None


def model_key(role: str, model: str) -> str:
    return f"{role}/{model}"


class OwnershipLedger:
    """The state machine used by the CPU actor and transport-independent
    tests."""

    def __init__(self, plan: PlacementPlan) -> None:
        self.plan = plan
        self.epoch = uuid.uuid4().hex
        self._generation = 0
        self._condition = asyncio.Condition()
        self._active: dict[str, ActivationLease] = {}
        self._operations: dict[str, ActivationLease] = {}
        self._pending: dict[str, tuple[tuple, asyncio.Task]] = {}
        self._reconciliations: dict[str, asyncio.Task] = {}
        self._abandoned: dict[str, HolderIdentity | None] = {}
        self._blocked: dict[str, tuple[tuple[PlacementClaim, ...], str]] = {}
        self._resources: dict[str, dict[str, Any]] = {}
        self._models: dict[str, tuple[Any, str, PlacementClaim]] = {}
        self._managers: dict[str, Any] = {}
        self._resident: set[str] = set()
        self._commands: dict[tuple[str, str], tuple[Any, asyncio.Task]] = {}
        self._locks: dict[str, asyncio.Lock] = {}
        self._phase: dict[str, str] = {}
        self._closing: set[str] = set()
        self._inflight: dict[str, set[str]] = {}
        self._suspect: dict[str, tuple[ActivationLease, str]] = {}
        self._training_actors: dict[str, list[Any]] = {}
        self._restarting = False
        self._stopped_managers: set[str] = set()

    def register_training_actors(self, role: str, actors: list[Any]) -> None:
        if self._restarting:
            raise OwnershipError("Training registration is closed during restart")
        self._training_actors.setdefault(role, []).extend(actors)

    async def suspect(self, lease: ActivationLease, reason: str) -> None:
        self.check(lease, allow_blocked=True)
        async with self._condition:
            self._suspect[lease.operation_id] = (lease, reason)
            self._condition.notify_all()

    async def stop_training(self, timeout: float = 60.0) -> None:
        """Kill registered ranks and discharge only terminal actor receipts."""
        self._restarting = True
        async with self._condition:
            self._condition.notify_all()
        for role, actors in list(self._training_actors.items()):
            probes = [actor.termination_probe.remote() for actor in actors]
            for actor in actors:
                ray.kill(actor, no_restart=True)
            try:
                results = await asyncio.wait_for(asyncio.gather(*probes, return_exceptions=True), timeout=timeout)
            except TimeoutError as error:
                raise ResourceBlockedError(
                    f"Unconfirmed training {role} exit; inspect ranks before checkpoint restart"
                ) from error
            if not all(isinstance(result, ray.exceptions.RayActorError) for result in results):
                raise ResourceBlockedError(f"Unconfirmed training {role} exit; Ray must report every rank DEAD")
            del self._training_actors[role]
            async with self._condition:
                for operation, lease in list(self._operations.items()):
                    if lease.role == role:
                        self._active.pop(operation, None)
                        self._suspect.pop(operation, None)
                        self._blocked.pop(operation, None)
                self._condition.notify_all()
        if self._suspect:
            raise ResourceBlockedError("Training exit cannot be proved without every registered rank")

    async def acquire_for_shutdown(self, role: str) -> ActivationLease:
        claims = self.claims_for(role)
        if not self._restarting:
            return await self.acquire(f"shutdown:{role}", uuid.uuid4().hex, claims, role=role)
        if self._training_actors or self._suspect:
            raise ResourceBlockedError("Training exit is unconfirmed before inference shutdown")
        self._generation += 1
        lease = ActivationLease(self.epoch, f"shutdown:{role}", uuid.uuid4().hex, self._generation, claims, role)
        self._active[lease.operation_id] = self._operations[lease.operation_id] = lease
        self._locks[lease.operation_id] = asyncio.Lock()
        return lease

    def confirm_manager_shutdown(self, role: str) -> None:
        if any(self._models[key][2].role == role for key in self._resident):
            raise ResourceBlockedError(f"{role} still has unconfirmed resident models")
        self._stopped_managers.add(role)

    async def new_epoch(self) -> None:
        if self._training_actors or self._suspect or self._resident or set(self._managers) - self._stopped_managers:
            raise ResourceBlockedError("Cannot replace epoch before all resource owners confirm release")
        async with self._condition:
            for _, task in self._pending.values():
                task.cancel()
            self.epoch = uuid.uuid4().hex
            self._active.clear()
            self._operations.clear()
            self._pending.clear()
            self._reconciliations.clear()
            self._blocked.clear()
            self._commands.clear()
            self._locks.clear()
            self._inflight.clear()
            self._closing.clear()
            self._models.clear()
            self._managers.clear()
            self._phase.clear()
            self._stopped_managers.clear()
            self._restarting = False
            self._condition.notify_all()

    def managers(self) -> list[Any]:
        return list(self._managers.values())

    def register_manager(self, role: str, manager: Any) -> None:
        if role in self._managers and self._managers[role] != manager:
            raise OwnershipError(f"{role} already has a resource owner; lost owners cannot be silently replaced")
        self._managers[role] = manager

    def begin(self, lease: ActivationLease, token: str, parent_command: str | None = None) -> None:
        self.check(lease, allow_blocked=True)
        if lease.operation_id in self._closing and (lease.operation_id, parent_command) not in self._commands:
            raise StaleLeaseError("The activation lease is being returned")
        self._inflight.setdefault(lease.operation_id, set()).add(token)

    async def finish(self, lease: ActivationLease, token: str) -> None:
        async with self._condition:
            self._inflight.get(lease.operation_id, set()).discard(token)
            self._condition.notify_all()

    def claims_for(self, role: str) -> tuple[PlacementClaim, ...]:
        own = tuple(c for c in self.plan.claims if c.role == role)
        if role in _TRAINING_ROLES:
            shared = tuple(c for c in self.plan.claims if c.pool == ACTOR_POOL)
            return shared
        # A rollout step owns its deferred scoring window as well. Dedicated
        # scorers stay resident and do not participate in these handoffs.
        if role == "rollout":
            pools = {c.pool for c in own}
            return tuple(c for c in self.plan.claims if c.pool in pools)
        return own

    def _conflicts(self, role: str | None, other: str | None) -> bool:
        # Old explicit callers without role metadata retain conservative fencing.
        if role is None or other is None:
            return True
        return not (role in _TRAINING_ROLES and other in _TRAINING_ROLES)

    def _needs_handoff(self, role: str | None, claims: tuple[PlacementClaim, ...]) -> bool:
        if role is None:
            return True
        return any(
            claim.pool == ACTOR_POOL
            or any(other.role != role and claims_overlap(claim, other) for other in self.plan.claims)
            for claim in claims
        )

    async def acquire(
        self,
        owner: str,
        operation_id: str,
        claims: Iterable[PlacementClaim],
        *,
        role: str | None = None,
        holder: HolderIdentity | None = None,
        expires_at: float | None = None,
    ) -> ActivationLease:
        claims = tuple(claims)
        if operation_id in self._abandoned or (expires_at is not None and time.time() >= expires_at):
            raise StaleLeaseError("The ownership request was abandoned or expired before dispatch")
        if self._restarting:
            raise OwnershipError("New ownership is closed during restart")
        previous = self._operations.get(operation_id)
        if previous is not None:
            if (previous.owner, previous.claims, previous.role, previous.holder) != (owner, claims, role, holder):
                raise OwnershipError("An operation ID cannot be reused with different ownership arguments")
            return previous
        pending = self._pending.get(operation_id)
        if pending is not None:
            if pending[0] != (owner, claims, role, holder):
                raise OwnershipError("Conflicting retry of a pending ownership operation")
            return await asyncio.shield(pending[1])

        async def grant() -> ActivationLease:
            deadline = time.monotonic() + Envs.RELAX_INFERENCE_LEASE_WAIT_S
            while True:
                async with self._condition:
                    if operation_id in self._abandoned or (expires_at is not None and time.time() >= expires_at):
                        raise StaleLeaseError("The ownership request was abandoned or expired before grant")
                    if self._restarting:
                        raise OwnershipError("New ownership is closed during restart")
                    for suspect, reason in self._suspect.values():
                        if any(claims_overlap(a, b) for a in claims for b in suspect.claims):
                            raise ResourceBlockedError(f"Training release awaits all-rank death evidence: {reason}")
                    for blocked, reason in self._blocked.values():
                        if any(claims_overlap(a, b) for a in claims for b in blocked):
                            raise ResourceBlockedError(reason)
                    blockers = [
                        lease
                        for lease in self._active.values()
                        if self._needs_handoff(role, claims)
                        and self._conflicts(role, lease.role)
                        and any(claims_overlap(a, b) for a in lease.claims for b in claims)
                    ]
                    if not blockers:
                        self._generation += 1
                        lease = ActivationLease(
                            self.epoch, owner, operation_id, self._generation, claims, role, holder
                        )
                        self._active[operation_id] = self._operations[operation_id] = lease
                        self._locks[operation_id] = asyncio.Lock()
                        return lease
                    remaining = deadline - time.monotonic()
                    detail = "; ".join(
                        f"{lease.owner} ({lease.holder}, operation={lease.operation_id})" for lease in blockers
                    )
                    if remaining <= 0:
                        raise OwnershipError(f"Timed out acquiring {owner}; waiting for {detail}")
                    try:
                        await asyncio.wait_for(
                            self._condition.wait(), min(remaining, Envs.RELAX_INFERENCE_HOLDER_POLL_S)
                        )
                    except TimeoutError:
                        pass
                # Never await manager callbacks under the ledger condition.
                for lease in blockers:
                    if self._active.get(lease.operation_id) != lease or lease.holder is None:
                        continue
                    try:
                        alive = await asyncio.wait_for(
                            self._holder_alive(lease.holder), max(0.001, deadline - time.monotonic())
                        )
                        if alive is False:
                            task = self._reconciliations.get(lease.operation_id)
                            if task is None or (task.done() and (task.cancelled() or task.exception() is not None)):
                                task = asyncio.create_task(self._reconcile_dead_holder(lease))
                                self._reconciliations[lease.operation_id] = task
                            await asyncio.wait_for(asyncio.shield(task), max(0.001, deadline - time.monotonic()))
                    except TimeoutError as error:
                        raise OwnershipError(f"Timed out acquiring {owner}; waiting for {detail}") from error

        task = asyncio.create_task(grant())
        self._pending[operation_id] = ((owner, claims, role, holder), task)
        try:
            return await asyncio.shield(task)
        finally:
            if task.done():
                self._pending.pop(operation_id, None)

    async def abandon(self, operation_id: str, holder: HolderIdentity | None) -> None:
        """Fence a timed-out request, including grants whose reply was lost."""
        async with self._condition:
            lease = self._active.get(operation_id)
            if lease is not None and lease.holder != holder:
                raise OwnershipError("Cannot abandon another holder's lease")
            if self._inflight.get(operation_id) or any(op == operation_id for op, _ in self._commands):
                raise OwnershipError("Cannot abandon a lease with accepted GPU operations")
            self._abandoned[operation_id] = holder
            self._active.pop(operation_id, None)
            self._condition.notify_all()

    async def _holder_alive(self, holder: HolderIdentity) -> bool | None:
        try:
            return await asyncio.wait_for(asyncio.to_thread(holder_alive, holder), Envs.RELAX_INFERENCE_HOLDER_CHECK_S)
        except TimeoutError:
            return None

    async def _reconcile_dead_holder(self, lease: ActivationLease) -> None:
        if lease.role in _TRAINING_ROLES:
            await self.suspect(lease, f"Holder {lease.holder} died; all training ranks must exit before restart")
            return
        self._closing.add(lease.operation_id)
        await self._await_inflight(lease)
        if lease.epoch != self.epoch:
            return
        covered = {
            key: record
            for key, record in self._models.items()
            if any(
                c.pool == record[2].pool and c.start <= record[2].start and c.stop >= record[2].stop
                for c in lease.claims
            )
            and (key in self._resident or lease.role is None or record[2].role == lease.role)
        }
        if not covered:
            await self.block(
                lease,
                f"Dead holder {lease.holder}: no manager can confirm residency; inspect nodes before checkpoint restart",
            )
            return
        for key, (manager, name, claim) in covered.items():
            try:
                confirmed = await asyncio.wait_for(
                    manager.reconcile_dead_holder.remote(name, lease), Envs.RELAX_INFERENCE_CONTROL_TIMEOUT_S
                )
                if confirmed is not True:
                    raise ResourceBlockedError("Manager did not confirm release")
            except Exception as error:
                if lease.epoch != self.epoch:
                    return
                self._blocked[f"{lease.operation_id}:{key}"] = ((claim,), f"Dead holder {lease.holder}: {error}")
            else:
                if lease.epoch != self.epoch:
                    return
                self._resident.discard(key)
        await self.release(lease)

    def check(
        self, lease: ActivationLease, claims: Iterable[PlacementClaim] = (), *, allow_blocked: bool = False
    ) -> None:
        if lease.epoch != self.epoch or self._active.get(lease.operation_id) != lease:
            raise StaleLeaseError("The activation lease is no longer current")
        for claim in claims:
            if not any(
                own.pool == claim.pool and own.start <= claim.start and own.stop >= claim.stop for own in lease.claims
            ):
                raise OwnershipError(f"Lease does not cover {claim.role}/{claim.model}")
        if lease.operation_id in self._blocked and not allow_blocked:
            raise ResourceBlockedError(self._blocked[lease.operation_id][1])

    async def release(self, lease: ActivationLease) -> None:
        if self._operations.get(lease.operation_id) == lease and lease.operation_id not in self._active:
            return
        self._closing.add(lease.operation_id)
        await self._await_inflight(lease)
        async with self._condition:
            if self._operations.get(lease.operation_id) == lease and lease.operation_id not in self._active:
                return
            if self._active.get(lease.operation_id) != lease:
                raise StaleLeaseError("Cannot return another generation's lease")
            del self._active[lease.operation_id]
            self._condition.notify_all()

    async def _await_inflight(self, lease: ActivationLease) -> None:
        # Stop new commands, then settle accepted commands and direct mutations.
        pending = [task for (operation, _), (_, task) in self._commands.items() if operation == lease.operation_id]

        async def settle() -> None:
            if pending:
                await asyncio.gather(*(asyncio.shield(task) for task in pending), return_exceptions=True)
            async with self._condition:
                while self._inflight.get(lease.operation_id):
                    await self._condition.wait()

        try:
            await asyncio.wait_for(settle(), Envs.RELAX_INFERENCE_CONTROL_TIMEOUT_S)
        except TimeoutError as error:
            raise OwnershipError(
                f"Timed out returning lease held by {lease.owner} ({lease.holder}); "
                f"waiting for commands {sorted(self._inflight.get(lease.operation_id, ())) or 'phase transition'}"
            ) from error

    async def block(self, lease: ActivationLease, reason: str) -> None:
        if self._active.get(lease.operation_id) != lease:
            raise StaleLeaseError("Cannot quarantine resources with an old lease")
        async with self._condition:
            self._blocked[lease.operation_id] = (lease.claims, reason)
            self._condition.notify_all()

    def register_model(self, role: str, model: str, manager: Any, claim: PlacementClaim) -> None:
        key = f"{model_key(role, model)}@{claim.pool}:{claim.start}"
        if key in self._models:
            raise OwnershipError(f"Model ownership is already registered: {key}")
        self._models[key] = (manager, model, claim)

    def reserve_activation(self, lease: ActivationLease, role: str, model: str) -> None:
        keys = [key for key, (_, name, claim) in self._models.items() if claim.role == role and name == model]
        self.check(lease, [self._models[key][2] for key in keys])
        for key in keys:
            for resident in self._resident.difference(keys):
                if claims_overlap(self._models[key][2], self._models[resident][2]):
                    raise OwnershipError(f"{resident} must confirm release before activating {role}/{model}")
        # Reserve before dispatch, so concurrent activations cannot both pass.
        self._resident.update(keys)

    def reserve_maintenance(self, role: str, model: str) -> bool:
        """Serialize allocation with cross-role residency, not workload leases.

        Once reserved, training handoff goes through the manager's model lock
        and waits for this maintenance before offloading the model.
        """
        keys = [key for key, (_, name, claim) in self._models.items() if claim.role == role and name == model]
        claims = [self._models[key][2] for key in keys]
        for blocked, _ in self._blocked.values():
            if any(claims_overlap(a, b) for a in claims for b in blocked):
                return False
        for lease in self._active.values():
            if lease.role in _TRAINING_ROLES and any(claims_overlap(a, b) for a in claims for b in lease.claims):
                return False
        for key in self._resident.difference(keys):
            other = self._models[key][2]
            if other.role != role and any(claims_overlap(claim, other) for claim in claims):
                return False
        self._resident.update(keys)
        return True

    def confirm_deactivation(self, lease: ActivationLease, role: str, model: str) -> None:
        keys = [key for key, (_, name, claim) in self._models.items() if claim.role == role and name == model]
        self.check(lease, [self._models[key][2] for key in keys], allow_blocked=True)
        self._resident.difference_update(keys)

    def register_resource(self, resource_id: str, owner: str, handle: Any, borrowed: bool = False) -> None:
        previous = self._resources.get(resource_id)
        if previous is not None:
            if previous["owner"] != owner:
                raise OwnershipError(f"Resource {resource_id} already belongs to {previous['owner']}")
            return
        self._resources[resource_id] = {"owner": owner, "handle": handle, "borrowed": borrowed, "released": False}

    def confirm_resource_release(self, resource_id: str, owner: str) -> None:
        record = self._resources[resource_id]
        if record["owner"] != owner or record["borrowed"]:
            raise OwnershipError(f"{owner} does not own resource {resource_id}")
        record["released"] = True

    def resources_for(self, owner: str) -> dict[str, Any]:
        return {
            key: record["handle"]
            for key, record in self._resources.items()
            if record["owner"] == owner and not record["released"] and not record["borrowed"]
        }

    async def command(self, lease: ActivationLease, command_id: str, phase: str, stage: str | None = None) -> None:
        if lease.epoch != self.epoch:
            raise StaleLeaseError("The command belongs to a previous resource epoch")
        key = (lease.operation_id, command_id)
        signature = (lease, phase, stage)
        previous = self._commands.get(key)
        if previous is not None:
            if previous[0] != signature:
                raise OwnershipError("A command ID cannot be reused for another transition")
            return await asyncio.shield(previous[1])
        self.check(lease, allow_blocked=phase in {"train", "leave_score"})
        if lease.operation_id in self._closing:
            raise StaleLeaseError("The activation lease is being returned")
        task = asyncio.create_task(self._transition(lease, command_id, phase, stage))
        self._commands[key] = (signature, task)
        await asyncio.shield(task)

    async def _apply(self, keys: list[str], lease: ActivationLease, command: str, action: str, tags=None) -> None:
        calls = []
        seen = set()
        for key in keys:
            manager, model, _ = self._models[key]
            identity = (str(manager), model)
            if identity in seen:
                continue
            seen.add(identity)
            calls.append(
                manager.apply.remote(
                    model,
                    action,
                    lease,
                    f"{command}:{key}:{action}",
                    tags=tags,
                    parent_command=command.removesuffix(":compensate"),
                )
            )
        results = await asyncio.gather(*calls, return_exceptions=True)
        errors = [r for r in results if isinstance(r, BaseException)]
        if errors:
            raise OwnershipError("; ".join(f"{type(e).__name__}: {e}" for e in errors))

    async def _transition(self, lease: ActivationLease, command: str, phase: str, stage: str | None) -> None:
        async with self._locks[lease.operation_id]:
            self.check(lease, allow_blocked=phase in {"train", "leave_score"})
            covered = {
                key: claim
                for key, (_, _, claim) in self._models.items()
                if any(c.pool == claim.pool and c.start <= claim.start and c.stop >= claim.stop for c in lease.claims)
            }
            if phase == "train":
                yielding, residents = list(covered), []
            elif phase == "score":
                residents = [key for key, c in covered.items() if Phase.SCORE in c.phases]
                yielding = [
                    key
                    for key, c in covered.items()
                    if key not in residents and any(claims_overlap(c, covered[r]) for r in residents)
                ]
            elif phase == "leave_score":
                residents = []
                yielding = [
                    key for key, c in covered.items() if Phase.SCORE in c.phases and Phase.GENERATE not in c.phases
                ]
            elif phase == "generate":
                residents = [key for key, c in covered.items() if Phase.GENERATE in c.phases]
                yielding = [
                    key
                    for key, c in covered.items()
                    if key not in residents and any(claims_overlap(c, covered[r]) for r in residents)
                ]
                if stage == "weights":
                    residents = [key for key in residents if covered[key].role != "genrm"]
                elif stage == "rest":
                    residents = [key for key in residents if covered[key].role != "teacher"]
                elif stage == "rest_without_kv":
                    residents = [key for key in residents if covered[key].role not in {"teacher", "rollout"}]
            else:
                raise ValueError(f"Unknown inference phase: {phase}")
            yielded = False
            try:
                await self._apply(yielding, lease, command, "offload")
                yielded = True
                for key in residents:
                    role = covered[key].role
                    tags = None
                    if role == "rollout" and stage is not None:
                        tags = ["weights"] if stage == "weights" else ["kv_cache", "cuda_graph"]
                    await self._apply([key], lease, command, "onload", tags)
                for key in covered:
                    self._phase[key] = phase
                if phase == "train":
                    self._blocked.pop(lease.operation_id, None)
            except BaseException as error:
                # All dispatched work has settled before compensation starts.
                # Even a failed activate may have allocated memory.
                try:
                    await self._apply(residents, lease, f"{command}:compensate", "offload")
                except BaseException as cleanup:
                    await self.block(lease, f"Phase {phase} failed; cleanup unconfirmed: {cleanup}")
                else:
                    if yielded:
                        self._blocked.pop(lease.operation_id, None)
                if not yielded:
                    await self.block(lease, f"Phase {phase} failed during resource handoff: {error}")
                raise

    def snapshot(self) -> dict[str, Any]:
        return {
            "epoch": self.epoch,
            "generation": self._generation,
            "phase": dict(self._phase),
            "resident": sorted(self._resident),
            "owners": {
                key: {"owner": value.owner, "generation": value.generation} for key, value in self._active.items()
            },
            "blocked": {key: reason for key, (_, reason) in self._blocked.items()},
            "suspect": {key: reason for key, (_, reason) in self._suspect.items()},
            "resources": {
                key: {k: v for k, v in value.items() if k != "handle"} for key, value in self._resources.items()
            },
        }


# A job owns this actor. Losing it must not silently create an empty ledger.
LifecycleAuthority = ray.remote(num_cpus=0, max_restarts=0)(OwnershipLedger)
