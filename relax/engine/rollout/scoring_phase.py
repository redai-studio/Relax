# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Run workload scoring through the task's shared GPU coordinator."""

import asyncio
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

from relax.engine.inference.phase_plans import (
    PHASE_GENERATE,
    PHASE_GENRM,
    PHASE_TEACHER,
    deferred_opd_enabled,
    genrm_shares_rollout_bundles,
)


async def _switch(args: Any, phase: str) -> None:
    coordinator = getattr(args, "_inference_phase_coordinator", None)
    if coordinator is None:
        raise RuntimeError("Shared scoring requires the task phase coordinator")
    task = asyncio.ensure_future(coordinator.switch.remote(phase))
    cancelled = False
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            cancelled = True
    task.result()
    if cancelled:
        raise asyncio.CancelledError


@asynccontextmanager
async def async_scoring_phase(args: Any, phase: str) -> AsyncIterator[bool]:
    enabled = {PHASE_GENRM: genrm_shares_rollout_bundles, PHASE_TEACHER: deferred_opd_enabled}[phase](args)
    if not enabled:
        yield False
        return
    try:
        await _switch(args, phase)
    except asyncio.CancelledError:
        await _switch(args, PHASE_GENERATE)
        raise
    try:
        yield True
    finally:
        await _switch(args, PHASE_GENERATE)
