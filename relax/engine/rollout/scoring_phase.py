# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Run a deferred scoring stage on GPUs shared with generation.

Deferred scoring needs the scorer resident and the generation engines asleep on
a shared slice. The task's LifecycleCoordinator performs that transition: it
closes admission, drains in-flight requests and releases memory before loading
the roles placed in the next phase. Without a deferred layout these helpers do
nothing.
"""

import asyncio
from contextlib import asynccontextmanager, contextmanager
from typing import Any, AsyncIterator, Iterator

import ray

from relax.engine.inference.phase_plans import (
    PHASE_GENERATE,
    PHASE_GENRM,
    PHASE_TEACHER,
    deferred_genrm_enabled,
    deferred_opd_enabled,
)
from relax.utils.logging_utils import get_logger


logger = get_logger(__name__)

# Generation has finished before scoring starts, so the drain should be empty.
SCORING_DRAIN_TIMEOUT_S = 600.0

_SCORER_PHASES = {
    PHASE_GENRM: deferred_genrm_enabled,
    PHASE_TEACHER: deferred_opd_enabled,
}


def _task_inference_manager() -> Any:
    """Find the task InferenceManager from inside the rollout actor process."""
    from relax.distributed.ray.rollout_worker import get_local_inference_manager

    return get_local_inference_manager()


def _transition(action: str, phase_id: str) -> None:
    """Ask the coordinator to ``enter`` or ``leave`` ``phase_id``."""
    manager = _task_inference_manager()
    if manager is None:
        raise RuntimeError("Deferred scoring requires the task InferenceManager")
    ray.get(getattr(manager, f"{action}_phase").remote(phase_id, SCORING_DRAIN_TIMEOUT_S))


async def _transition_to_completion(action: str, phase_id: str) -> None:
    """Run a transition on a worker thread and wait for it even when cancelled.

    The thread keeps switching after the awaiting task is cancelled, so the
    caller must not move on as if the GPUs had not changed hands.
    """
    transition = asyncio.ensure_future(asyncio.to_thread(_transition, action, phase_id))
    cancelled = False
    while not transition.done():
        try:
            await asyncio.shield(transition)
        except asyncio.CancelledError:
            cancelled = True
    if cancelled:
        if not transition.cancelled():
            transition.exception()
        raise asyncio.CancelledError
    transition.result()


@contextmanager
def scoring_phase(args: Any, phase_id: str) -> Iterator[bool]:
    """Hold the GPUs for ``phase_id``'s scorer while the block runs.

    Yields whether a switch happened. Leaving puts the scorer back to sleep but
    does not restore generation: the next weight sync onloads it anyway.
    """
    if not _SCORER_PHASES[phase_id](args):
        yield False
        return
    _transition("enter", phase_id)
    logger.info(f"Entered scoring phase {phase_id}")
    try:
        yield True
    finally:
        _transition("leave", phase_id)
        logger.info(f"Left scoring phase {phase_id}")


@asynccontextmanager
async def async_scoring_phase(args: Any, phase_id: str) -> AsyncIterator[bool]:
    """``scoring_phase`` for async callers; switches run on a worker thread."""
    if not _SCORER_PHASES[phase_id](args):
        yield False
        return
    try:
        await _transition_to_completion("enter", phase_id)
    except asyncio.CancelledError:
        # The scorer may already hold the GPUs; release it before leaving.
        await _transition_to_completion("leave", phase_id)
        raise
    try:
        yield True
    finally:
        await _transition_to_completion("leave", phase_id)


async def async_reactivate_generation(args: Any) -> None:
    """Bring the student back after the teacher stage, for a second pass."""
    if deferred_opd_enabled(args):
        await _transition_to_completion("enter", PHASE_GENERATE)


__all__ = ["SCORING_DRAIN_TIMEOUT_S", "async_reactivate_generation", "async_scoring_phase", "scoring_phase"]
