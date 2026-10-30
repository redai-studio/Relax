# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Ordering of GPU memory switches between models that share bundles.

A colocated run moves through phases -- train, generate, score -- and the
inference models that share a placement group with training, and with each
other, must not hold GPU memory at the same time. ``LifecycleCoordinator`` is
where that order is written down: on entering a phase, whatever has to yield
is deactivated and confirmed first, and only then are the phase's residents
activated. Who resides in which phase, and who shares bundles with whom, comes
from the placement plan.

The coordinator is stateless sequential logic over idempotent operations. Each
process that drives a switch -- rank 0 of the training actor, the rollout
manager for deferred scoring -- builds its own from the same plan, and a switch
that failed halfway can simply be issued again.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Iterable, Mapping, Sequence

from relax.distributed.ray.placement_planner import (
    DEFAULT_MODEL_KEY,
    GENRM_ROLE,
    ROLLOUT_ROLE,
    TEACHER_ROLE,
    Phase,
    PlacementClaim,
    PlacementPlan,
    claims_overlap,
    plan_placement,
)


# Starts a switch and returns the Ray refs to wait for; an empty list when the
# switch already completed synchronously.
Switch = Callable[[], list]


class GenerateStage(str, Enum):
    """Entering generation is split around weight sync."""

    # Before weight sync: what the sync itself needs on GPU.
    WEIGHTS = "weights"
    # After weight sync: everything whose memory would collide with the sync's
    # temporary buffers.
    REST = "rest"


@dataclass(frozen=True)
class LifecycleParticipant:
    """One model, as the switches the coordinator may ask of it.

    ``generate_stages`` says what the model does in each stage of entering
    generation; a stage it is absent from leaves it untouched.
    """

    role: str
    model: str
    deactivate: Switch
    activate: Switch
    generate_stages: Mapping[GenerateStage, Switch] = field(default_factory=dict)


def rollout_participant(rollout_manager: Any, *, warm_kv: bool = True) -> LifecycleParticipant:
    """Rollout, driven through its manager's Ray handle.

    Its weights come back before weight sync (the sync writes into them) and
    its KV cache after. ``warm_kv=False`` leaves the cache to whoever generates
    next.
    """
    stages: dict[GenerateStage, Switch] = {GenerateStage.WEIGHTS: lambda: [rollout_manager.onload_weights.remote()]}
    if warm_kv:
        stages[GenerateStage.REST] = lambda: [rollout_manager.onload_kv.remote()]
    return LifecycleParticipant(
        ROLLOUT_ROLE,
        DEFAULT_MODEL_KEY,
        deactivate=lambda: [rollout_manager.offload.remote()],
        activate=lambda: [rollout_manager.onload.remote()],
        generate_stages=stages,
    )


def local_rollout_participant(rollout_manager: Any) -> LifecycleParticipant:
    """Rollout, driven from inside its own manager process.

    A remote call to itself would deadlock there, so the switches run the
    manager's synchronous bodies and leave nothing to wait for.
    """

    def offload() -> list:
        rollout_manager._offload_local()
        return []

    def onload() -> list:
        # The manager's onload body resumes its engines unconditionally, so
        # only run it for a rollout that was actually released.
        if rollout_manager.status == "offload":
            rollout_manager._onload_local()
        return []

    return LifecycleParticipant(ROLLOUT_ROLE, DEFAULT_MODEL_KEY, deactivate=offload, activate=onload)


def manager_participant(role: str, model: str, manager: Any) -> LifecycleParticipant:
    """A GenRM or teacher model, driven through its manager's Ray handle.

    A teacher is woken together with the rollout weights. A GenRM waits until
    weight sync is over: it takes no part in the sync, and in a shared pool its
    static memory would collide with the sync's temporary buffers.
    """

    def activate() -> list:
        return [manager.onload.remote()]

    stage = GenerateStage.WEIGHTS if role == TEACHER_ROLE else GenerateStage.REST
    return LifecycleParticipant(
        role,
        model,
        deactivate=lambda: [manager.offload.remote()],
        activate=activate,
        generate_stages={stage: activate},
    )


def _ray_get(refs: list) -> Any:
    import ray

    return ray.get(refs)


class LifecycleCoordinator:
    def __init__(
        self,
        plan: PlacementPlan,
        participants: Iterable[LifecycleParticipant],
        *,
        wait: Callable[[list], Any] = _ray_get,
    ) -> None:
        """
        Args:
            plan: Placement plan of the run; decides residency and overlap.
            participants: The models this coordinator can switch, in the order
                their switches are issued within one step.
            wait: Blocks until the given Ray refs resolve, raising if any
                switch failed.
        """
        self._plan = plan
        self._participants = tuple(participants)
        self._wait = wait

    # ------------------------------------------------------------------
    # What the plan says about a participant.
    # ------------------------------------------------------------------

    def _claims(self, participant: LifecycleParticipant) -> list[PlacementClaim]:
        return [
            claim for claim in self._plan.claims if claim.role == participant.role and claim.model == participant.model
        ]

    def _resides(self, participant: LifecycleParticipant, phase: Phase) -> bool:
        claims = self._claims(participant)
        if not claims:
            # Not planned: treat it as woken with rollout, which is what every
            # inference model was before phases existed.
            return phase is Phase.GENERATE
        return any(phase in claim.phases for claim in claims)

    def _shares_bundles(self, a: LifecycleParticipant, b: LifecycleParticipant) -> bool:
        return any(claims_overlap(x, y) for x in self._claims(a) for y in self._claims(b))

    def _run(self, switches: Iterable[Switch]) -> None:
        """Issue the switches together and wait until every one is
        confirmed."""
        refs: list = []
        for switch in switches:
            refs.extend(switch())
        if refs:
            self._wait(refs)

    # ------------------------------------------------------------------
    # Phase entries.
    # ------------------------------------------------------------------

    def enter_train(self) -> None:
        """Release every scorer before training reclaims its GPUs.

        Returns once the releases are confirmed. Rollout is not touched: it
        releases itself after each generation, behind its own barrier.
        """
        self._run(p.deactivate for p in self._participants if p.role != ROLLOUT_ROLE)

    def enter_generate(self, stage: GenerateStage | None = None) -> None:
        """Wake the models that hold GPU memory while rollout generates.

        With ``stage``, only that part of the entry; without, everything in one
        step. A model that only lives in the score phase stays asleep.
        """
        residents = [p for p in self._participants if self._resides(p, Phase.GENERATE)]
        if stage is None:
            self._run(p.activate for p in residents)
        else:
            self._run(p.generate_stages[stage] for p in residents if stage in p.generate_stages)

    def enter_score(self) -> None:
        """Hand the bundles over to the models that score a finished batch.

        Every model that shares bundles with a scorer, and does not itself live
        in the score phase, is deactivated first. Only when that is confirmed
        are the scorers activated; if it fails, nothing is.
        """
        residents = [p for p in self._participants if self._resides(p, Phase.SCORE)]
        yielding = [
            p
            for p in self._participants
            if p not in residents and any(self._shares_bundles(p, resident) for resident in residents)
        ]
        self._run(p.deactivate for p in yielding)
        self._run(p.activate for p in residents)

    def leave_score(self) -> None:
        """Put the models that only live in the score phase back to sleep."""
        self._run(
            p.deactivate
            for p in self._participants
            if self._resides(p, Phase.SCORE) and not self._resides(p, Phase.GENERATE)
        )


def _teacher_models(args: Any, teacher_manager: Any) -> dict[str, Any]:
    if teacher_manager is None:
        return {}
    if isinstance(teacher_manager, list):
        # MOPD: one manager per data source, in --opd-teacher-routes order.
        return dict(zip(json.loads(args.opd_teacher_routes), teacher_manager, strict=True))
    return {DEFAULT_MODEL_KEY: teacher_manager}


def build_train_coordinator(
    args: Any,
    *,
    rollout_manager: Any = None,
    genrm_managers: Sequence[Any] | None = None,
    teacher_manager: Any = None,
    warm_rollout_kv: bool = True,
) -> LifecycleCoordinator:
    """Coordinator for the switches the training actor drives.

    Args:
        rollout_manager: Rollout manager handle.
        genrm_managers: GenRM manager handles, one per instance, in
            ``args._genrm_instances_resolved`` order.
        teacher_manager: A teacher manager handle, or a list of them for MOPD.
        warm_rollout_kv: Whether rollout's KV cache is restored after weight
            sync, ready for a generation that follows right away.
    """
    participants: list[LifecycleParticipant] = []
    if rollout_manager is not None:
        participants.append(rollout_participant(rollout_manager, warm_kv=warm_rollout_kv))
    if genrm_managers:
        keys = list(getattr(args, "_genrm_instances_resolved", None) or {})
        for key, manager in zip(keys, genrm_managers, strict=True):
            participants.append(manager_participant(GENRM_ROLE, key, manager))
    for key, manager in _teacher_models(args, teacher_manager).items():
        participants.append(manager_participant(TEACHER_ROLE, key, manager))
    # The layout was validated when the run started; here only residency matters.
    return LifecycleCoordinator(plan_placement(args, validate=False), participants)
