# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import asyncio
from types import SimpleNamespace

import pytest

from relax.engine.inference.phase_plans import PHASE_GENERATE, PHASE_GENRM
from relax.engine.inference.phases import PhaseCoordinator
from relax.engine.rollout.scoring_phase import async_scoring_phase


@pytest.mark.asyncio
async def test_scoring_phase_restores_generation_after_scoring_failure():
    transitions = []

    async def call(owner, method):
        transitions.append((owner, method))

    coordinator = PhaseCoordinator({PHASE_GENERATE: ("student",), PHASE_GENRM: ("judge",)}, PHASE_GENERATE, call)
    args = SimpleNamespace(
        _genrm_instances_resolved={"judge": {}},
        defer_reward_to_post_process=True,
        _genrm_colocate_with_rollout=True,
        _inference_phase_coordinator=SimpleNamespace(switch=SimpleNamespace(remote=coordinator.switch)),
    )
    with pytest.raises(ValueError, match="scoring failed"):
        async with async_scoring_phase(args, PHASE_GENRM):
            raise ValueError("scoring failed")
    assert transitions == [("student", "offload"), ("judge", "onload"), ("judge", "offload"), ("student", "onload")]


@pytest.mark.asyncio
async def test_scoring_phase_cancelled_entry_waits_and_releases_scorer():
    entered = asyncio.Event()
    release = asyncio.Event()
    transitions = []

    async def call(owner, method):
        if owner == "judge" and method == "onload":
            entered.set()
            await release.wait()
        transitions.append((owner, method))

    coordinator = PhaseCoordinator({PHASE_GENERATE: ("student",), PHASE_GENRM: ("judge",)}, PHASE_GENERATE, call)
    args = SimpleNamespace(
        _genrm_instances_resolved={"judge": {}},
        defer_reward_to_post_process=True,
        _genrm_colocate_with_rollout=True,
        _inference_phase_coordinator=SimpleNamespace(switch=SimpleNamespace(remote=coordinator.switch)),
    )

    async def score():
        async with async_scoring_phase(args, PHASE_GENRM):
            pytest.fail("Cancelled entry must not execute scoring")

    task = asyncio.create_task(score())
    await entered.wait()
    task.cancel()
    await asyncio.sleep(0)
    assert not task.done()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert transitions[-2:] == [("judge", "offload"), ("student", "onload")]


@pytest.mark.asyncio
async def test_manual_shared_genrm_scores_directly_without_deferred_session():
    from relax.engine.rollout.deferred_opd import DeferredOpdSession

    events = []

    async def call(owner, method):
        events.append((owner, method))

    coordinator = PhaseCoordinator({"inference": ("student",), "genrm": ("judge",)}, "inference", call)
    args = SimpleNamespace(
        use_opd=False,
        _genrm_instances_resolved={"judge": {}},
        _genrm_colocate_with_rollout=True,
        defer_reward_to_post_process=False,
        _inference_phase_coordinator=SimpleNamespace(switch=SimpleNamespace(remote=coordinator.switch)),
    )
    assert DeferredOpdSession.maybe_create(args, 0, None, None, publish=None) is None
    async with async_scoring_phase(args, "genrm") as activated:
        assert activated
        events.append(("judge", "request"))
    assert events == [
        ("student", "offload"),
        ("judge", "onload"),
        ("judge", "request"),
        ("judge", "offload"),
        ("student", "onload"),
    ]
