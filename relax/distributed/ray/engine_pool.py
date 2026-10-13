# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""One pool of engine replicas, whatever role they serve.

``EnginePool`` owns the mechanics that used to exist twice -- once in
``MultiEngineManager`` (GenRM judges, OPD teachers) and once in ``EngineGroup``
(rollout): creating engine actors on a placement group, firing ``init``,
switching GPU memory, detecting and retiring dead engines, and tearing
everything down. What differs per role is injected through ``EnginePoolSpec``.

Slots are the unit of scheduling (one Ray actor each); a ``None`` slot is dead
and needs a rebuild. A logical engine may span ``nodes_per_engine`` consecutive
slots, of which only the head serves HTTP, so lifecycle state is kept per head
slot. The slot list is shared by reference with the pool's owner, which may
read it and clear entries in place.

Two ways to bring engines up:

- ``create`` + ``fire_init`` return without waiting; the caller allocates
  addresses in between and owns failure handling (rollout).
- ``start`` waits for every ``init`` and rolls the whole call back on failure
  (GenRM, teachers).
"""

from __future__ import annotations

import dataclasses
from typing import Any, Callable, Iterable, Optional

import ray
import requests

from relax.engine.inference.discovery import EngineState
from relax.utils.logging_utils import get_logger


logger = get_logger(__name__)

# An engine process can die on its own (e.g. SGLang's scheduler watchdog
# SIGQUITs the server after a CUDA-level hang). The next call into it then
# raises one of these. Everything else is a real bug and must propagate.
#   - ConnectionError / TimeoutError: raised by the engine's
#     release_memory_occupation when a drain loop hits its dead-server
#     fast-fail or its deadline.
#   - requests.exceptions.{ConnectionError,Timeout}: raised by _make_request,
#     i.e. the resume_memory_occupation path. These are OSError subclasses but
#     NOT builtin ConnectionError/TimeoutError, so they must be listed
#     explicitly -- otherwise an engine that died during the offloaded window
#     (only observable at onload) escalates to a global restart.
#   - RayActorError: the Ray actor itself is gone.
# ray.get re-raises as a class inheriting from BOTH RayTaskError and the
# original cause (ray/exceptions.py::as_instanceof_cause), so isinstance works.
ENGINE_DEAD_EXCEPTIONS = (
    ConnectionError,
    TimeoutError,
    requests.exceptions.ConnectionError,
    requests.exceptions.Timeout,
    ray.exceptions.RayActorError,
)


def is_engine_dead(exc: BaseException) -> bool:
    return isinstance(exc, ENGINE_DEAD_EXCEPTIONS)


# The only edges a live engine's state moves along; any state can also fall to
# DEAD, and a rebuilt slot starts over at STARTING.
_NEXT_STATE = {
    EngineState.STARTING: EngineState.READY,
    EngineState.READY: EngineState.DRAINING,
    EngineState.DRAINING: EngineState.SLEEPING,
    EngineState.SLEEPING: EngineState.ONLOADING,
    EngineState.ONLOADING: EngineState.READY,
}


@dataclasses.dataclass(frozen=True)
class EnginePoolSpec:
    """The role-specific pieces of running a pool of engines."""

    # () -> the Ray actor class engines are created from.
    actor_class: Callable[[], Any]
    # (slot) -> (pg_tuple, owns_pg, gpu_index). ``pg_tuple`` is
    # ``(pg, reordered_bundle_indices, reordered_gpu_ids)``; ``owns_pg`` marks a
    # placement group created for this slot, which the pool must remove again;
    # ``gpu_index`` is the slot's first index into the two reordered lists.
    resolve_placement: Callable[[int], tuple[tuple, bool, int]]
    # (slot, placement_group, bundle_index) -> keyword arguments of ``actor.options``.
    actor_options: Callable[[int, Any, int], dict]
    # (slot, base_gpu_id) -> (args, kwargs) of the engine constructor.
    ctor: Callable[[int, int], tuple[tuple, dict]]
    # (slot, address) -> keyword arguments of ``engine.init``.
    init_kwargs: Callable[[int, dict], dict]
    # ([(slot, engine)]) -> {slot: address}. Only ``start`` needs it.
    allocate_addresses: Optional[Callable[[list[tuple[int, Any]]], dict[int, dict]]] = None
    # Engine methods called, in order, before an engine actor is killed.
    teardown_calls: tuple[str, ...] = ("shutdown",)
    teardown_timeout_s: float = 60.0
    # Upper bound on waiting for ``init`` in ``start``; ``None`` waits forever.
    start_timeout_s: Optional[float] = None
    log_prefix: str = ""


class EnginePool:
    def __init__(self, spec: EnginePoolSpec, slots: list, *, nodes_per_engine: int = 1) -> None:
        self.spec = spec
        self.slots = slots
        self.nodes_per_engine = max(1, nodes_per_engine)
        # Per-slot (pg_tuple, owns_pg), so only placement groups created for
        # this pool are ever removed by it.
        self.placements: dict[int, tuple] = {}
        self.addresses: dict[int, dict] = {}
        # How many times each slot has been built.
        self.incarnations: dict[int, int] = {}
        self._states: dict[int, EngineState] = {}
        # Whether the last completed memory switch left the engines on GPU.
        # Engines come up holding GPU memory.
        self._active = True

    # ------------------------------------------------------------------
    # Slots and state.
    # ------------------------------------------------------------------

    def head_slots(self) -> range:
        return range(0, len(self.slots), self.nodes_per_engine)

    def is_active(self) -> bool:
        return self._active

    def state(self, head_slot: int) -> EngineState:
        if self.slots[head_slot] is None:
            return EngineState.DEAD
        return self._states.get(head_slot, EngineState.READY if self._active else EngineState.SLEEPING)

    def _move(self, head_slot: int, target: EngineState) -> None:
        """Take a live engine's state to ``target`` along the allowed edges."""
        state = self.state(head_slot)
        if state is EngineState.DEAD:
            return
        for _ in range(len(_NEXT_STATE)):
            if state is target:
                break
            state = _NEXT_STATE[state]
        self._states[head_slot] = target

    def _move_live(self, sources: tuple[EngineState, ...], target: EngineState, *, skip: Iterable[int] = ()) -> None:
        skipped = set(skip)
        for head in self.head_slots():
            if head not in skipped and self.state(head) in sources:
                self._move(head, target)

    def settle(self, resident: EngineState) -> None:
        """Align states with what the owner knows the last completed switch
        left the engines in (``READY`` or ``SLEEPING``).

        For owners that switch memory through the non-blocking ``*_handles``
        calls: the pool sees a switch begin but not end. An engine still on
        its way *away* from ``resident`` is left alone -- that switch is in
        flight. STARTING engines wait for an explicit ``mark_initialized``.
        """
        in_flight = EngineState.DRAINING if resident is EngineState.READY else EngineState.ONLOADING
        for head in self.head_slots():
            if self.state(head) not in (in_flight, EngineState.STARTING, EngineState.DEAD):
                self._move(head, resident)
        self._active = resident is EngineState.READY

    def mark_initialized(self, slots: Iterable[int]) -> None:
        """Mark head slots ready after their engines finish initialization."""
        for slot in slots:
            if slot % self.nodes_per_engine == 0:
                self._move(slot, EngineState.READY)

    def _log(self, message: str) -> str:
        return f"{self.spec.log_prefix} {message}" if self.spec.log_prefix else message

    # ------------------------------------------------------------------
    # Bring-up.
    # ------------------------------------------------------------------

    def create(self, slots: Optional[Iterable[int]] = None) -> list[tuple[int, Any]]:
        """Create an engine actor in every empty slot (of ``slots``, default
        all) without initializing it.

        Returns the new ``(slot, engine)`` pairs.
        """
        created: list[tuple[int, Any]] = []
        self._create_into(slots, created)
        return created

    def _create_into(self, slots: Optional[Iterable[int]], created: list[tuple[int, Any]]) -> None:
        # Appends as it goes so a caller can roll back a call that fails midway.
        actor_class = self.spec.actor_class()
        for slot in range(len(self.slots)) if slots is None else slots:
            if self.slots[slot] is not None:
                continue

            pg_tuple, owns_pg, gpu_index = self.spec.resolve_placement(slot)
            self.placements[slot] = (pg_tuple, owns_pg)
            pg, reordered_bundle_indices, reordered_gpu_ids = pg_tuple
            base_gpu_id = int(reordered_gpu_ids[gpu_index])

            options = self.spec.actor_options(slot, pg, reordered_bundle_indices[gpu_index])
            ctor_args, ctor_kwargs = self.spec.ctor(slot, base_gpu_id)
            engine = actor_class.options(**options).remote(*ctor_args, **ctor_kwargs)

            created.append((slot, engine))
            self.slots[slot] = engine
            self.incarnations[slot] = self.incarnations.get(slot, 0) + 1
            if slot % self.nodes_per_engine == 0:
                self._states[slot] = EngineState.STARTING

    def fire_init(self, engines: list[tuple[int, Any]], addresses: dict[int, dict]) -> list:
        """Record each new engine's address and fire its ``init`` without
        waiting.

        Returns the init handles.
        """
        for slot, _ in engines:
            self.addresses[slot] = addresses[slot]
        return [engine.init.remote(**self.spec.init_kwargs(slot, addresses[slot])) for slot, engine in engines]

    def start(self, slots: Optional[Iterable[int]] = None) -> list[int]:
        """Bring up every empty slot (of ``slots``) and wait until its engine
        is initialized. Returns the slots started.

        All or nothing: if anything fails, every engine this call created is
        killed, its slot emptied and its own placement group removed, and the
        error is raised. Engines that were already running are not touched.
        """
        if self.spec.allocate_addresses is None:
            raise RuntimeError("EnginePool.start needs a spec with allocate_addresses.")
        created: list[tuple[int, Any]] = []
        known_placements = set(self.placements)
        try:
            self._create_into(slots, created)
            if not created:
                return []
            addresses = self.spec.allocate_addresses(created)
            init_handles = self.fire_init(created, addresses)
            # Bounded: a bundle on a dead node never schedules, and an unbounded
            # ray.get would block the caller forever.
            ray.get(init_handles, timeout=self.spec.start_timeout_s)
        except Exception:
            for slot, engine in created:
                try:
                    ray.kill(engine)
                except Exception:
                    pass
                self.slots[slot] = None
            # Includes a slot whose placement was resolved but whose actor was never created.
            resolved = (set(self.placements) - known_placements) | {slot for slot, _ in created}
            for slot in resolved:
                self._remove_owned_pg(slot)
            raise

        self.mark_initialized(slot for slot, _ in created)
        return [slot for slot, _ in created]

    def _remove_owned_pg(self, slot: int) -> None:
        placement = self.placements.pop(slot, None)
        if placement is None:
            return
        pg_tuple, owns_pg = placement
        if not owns_pg:
            return
        try:
            from ray.util.placement_group import remove_placement_group

            remove_placement_group(pg_tuple[0])
        except Exception as exc:
            logger.warning(self._log(f"remove placement group for rank={slot} failed: {exc}"))

    # ------------------------------------------------------------------
    # GPU memory switching.
    # ------------------------------------------------------------------

    def release_handles(self) -> list:
        """Fire ``release_memory_occupation`` on every live head engine without
        waiting.

        Returns the handles.
        """
        handles = []
        for head in self.head_slots():
            engine = self.slots[head]
            if engine is None:
                continue
            self._move(head, EngineState.DRAINING)
            handles.append(engine.release_memory_occupation.remote())
        return handles

    def resume_handles(self, tags: Optional[list[str]] = None) -> list:
        """Fire ``resume_memory_occupation`` on every live head engine without
        waiting.

        Returns the handles.
        """
        handles = []
        for head in self.head_slots():
            engine = self.slots[head]
            if engine is None:
                continue
            self._move(head, EngineState.ONLOADING)
            handles.append(engine.resume_memory_occupation.remote(tags=tags))
        return handles

    def drain(self) -> None:
        """Take ready engines out of routing ahead of ``deactivate``.

        Marks them DRAINING, which discovery reports as unavailable. The
        engine-side drain (stop admitting, abort what is in flight) is part of
        the engine's release call. Idempotent.
        """
        self._move_live((EngineState.READY,), EngineState.DRAINING)

    def deactivate(self) -> list[int]:
        """Release GPU memory on every live engine and wait. Idempotent.

        Engines found dead are retired but not rebuilt: this typically runs
        while other ranks wait on a barrier, so the rebuild is left to the next
        ``activate``. Returns the head slots retired.
        """
        if not self._active:
            logger.info(self._log("engines already offloaded; skipping"))
            return []
        logger.info(self._log("engines offload started"))
        self.drain()
        dead = self.call_all("release_memory_occupation")
        self.retire(dead)
        self._move_live((EngineState.DRAINING,), EngineState.SLEEPING)
        # Unconditional: the surviving engines did release, so the pool must
        # not claim to still be active just because one engine died.
        self._active = False
        logger.info(self._log(f"engines offload completed (retired {len(dead)} dead)"))
        return dead

    def activate(self, tags: Optional[list[str]] = None) -> set[int]:
        """Bring every engine back onto GPU and wait. Idempotent without
        ``tags``. Returns the slots rebuilt.

        Also the recovery point for engines that died since the last switch: a
        freshly built engine comes up holding GPU memory, which is exactly the
        state this call wants.
        """
        rebuilt = self.recover()
        if self._active and tags is None:
            logger.info(self._log("engines already onloaded; skipping"))
            return rebuilt
        logger.info(self._log(f"engines onload started with tags={tags}"))
        # Engines rebuilt just above already hold GPU memory -- resuming them
        # again would be a double-resume, so only touch the ones that survived.
        # The same goes for an engine rebuilt while the pool was inactive.
        resident = set(rebuilt)
        if tags is None:
            resident |= {head for head in self.head_slots() if self.state(head) is EngineState.READY}
        self._move_live((EngineState.SLEEPING,), EngineState.ONLOADING, skip=resident)
        dead = self.call_all("resume_memory_occupation", skip=resident, tags=tags)
        if dead:
            # An engine that died while offloaded is only discovered here
            # (deactivate() short-circuits when already inactive), so it missed
            # the recover() above. Rebuild now rather than leaving the pool a
            # man down for the whole next phase.
            self.retire(dead)
            rebuilt |= self.recover()
        self._move_live((EngineState.ONLOADING,), EngineState.READY)
        self._active = True
        logger.info(self._log("engines onload completed"))
        return rebuilt

    # ------------------------------------------------------------------
    # Failure handling.
    # ------------------------------------------------------------------

    def call_all(self, method: str, *, skip: Optional[set] = None, **kwargs) -> list[int]:
        """Call ``method`` on every live head engine and wait; return the head
        slots whose engine turned out to be dead.

        Per-handle ray.get rather than one ray.get over the list: the batched
        form aborts on the first failure and loses which engine raised.
        """
        skipped = skip or set()
        handles = {}
        for head in self.head_slots():
            engine = self.slots[head]
            if engine is None or head in skipped:
                continue
            handles[head] = getattr(engine, method).remote(**kwargs)

        dead = []
        for head, handle in handles.items():
            try:
                ray.get(handle)
            except Exception as exc:
                if not is_engine_dead(exc):
                    raise
                logger.warning(self._log(f"engine rank={head} died during {method}: {exc}"))
                self._states[head] = EngineState.DEAD
                dead.append(head)
        return dead

    def failed_health_checks(
        self, slots: Iterable[int], *, timeout: Optional[float] = None, get_timeout: Optional[float] = None
    ) -> set[int]:
        """Probe the engines in ``slots``; return the slots whose probe failed.

        ``timeout`` is passed to the engine's own probe, ``get_timeout`` bounds
        waiting for the answer. Empty slots are skipped.
        """
        call_kwargs = {} if timeout is None else {"timeout": timeout}
        get_kwargs = {} if get_timeout is None else {"timeout": get_timeout}
        failed = set()
        for slot in slots:
            engine = self.slots[slot]
            if engine is None:
                continue
            try:
                ray.get(engine.health_generate.remote(**call_kwargs), **get_kwargs)
            except Exception as exc:
                logger.warning(self._log(f"engine rank={slot} health check failed: {exc}"))
                failed.add(slot)
        return failed

    def teardown(self, slots: Iterable[int]) -> None:
        """Shut down and kill the engine in each slot, empty the slot and
        remove a placement group created for it."""
        for slot in slots:
            engine = self.slots[slot]
            if engine is not None:
                for method in self.spec.teardown_calls:
                    # The engine's own shutdown kills the SGLang process tree. It
                    # must run before ray.kill or the scheduler subprocesses are
                    # orphaned and keep holding GPU memory.
                    try:
                        ray.get(getattr(engine, method).remote(), timeout=self.spec.teardown_timeout_s)
                    except Exception as exc:
                        logger.warning(self._log(f"engine rank={slot} {method} failed (killing anyway): {exc}"))
                try:
                    ray.kill(engine)
                except Exception as exc:
                    logger.warning(self._log(f"engine rank={slot} ray.kill failed: {exc}"))
                logger.info(self._log(f"engine rank={slot} retired"))
            self.slots[slot] = None
            self._remove_owned_pg(slot)

    def retire(self, head_slots: Iterable[int]) -> None:
        """Tear down the given logical engines, all their slots, so ``recover``
        rebuilds them."""
        for head in head_slots:
            self.teardown(range(head, head + self.nodes_per_engine))

    def recover(self) -> set[int]:
        """Rebuild every empty slot. Returns the slots rebuilt.

        Degrades rather than escalates: running on fewer engines beats a global
        restart. Only a pool with no engine left raises.
        """
        dead = [slot for slot, engine in enumerate(self.slots) if engine is None]
        if not dead:
            return set()

        logger.info(self._log(f"recovering {len(dead)} engine(s): ranks={dead}"))
        try:
            self.start(dead)
        except Exception as exc:
            logger.exception(self._log(f"engine rebuild failed for ranks={dead}: {exc}"))

        rebuilt = {slot for slot in dead if self.slots[slot] is not None}
        still_dead = [slot for slot in dead if slot not in rebuilt]
        if still_dead:
            if all(engine is None for engine in self.slots):
                raise RuntimeError(f"All engines are dead and could not be rebuilt (ranks={still_dead})")
            logger.error(self._log(f"engines still dead after recovery, continuing degraded: ranks={still_dead}"))
        if rebuilt:
            logger.info(self._log(f"recovered engine ranks={sorted(rebuilt)}"))
        return rebuilt

    def shutdown(self) -> None:
        """Tear down every engine and remove every placement group created for
        this pool.

        Idempotent.
        """
        self.teardown(range(len(self.slots)))
        logger.info(self._log("shutdown complete."))
