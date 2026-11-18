# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""The metrics service's straggler endpoints: the control page, reading and
setting the requested level, and the primary rank's poll.

The state lives in this module because the metrics service runs as a single
replica; a second replica would answer with a level of its own.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from relax.utils.straggler.analysis.control import CONTROL_PAGE, ControlState


router = APIRouter()
state = ControlState()


class StragglerLevelRequest(BaseModel):
    level: int


class StragglerPollRequest(BaseModel):
    level: int
    rollout: int
    health: str


@router.get("/straggler")
async def straggler_page() -> HTMLResponse:
    """Page to switch the straggler profiler's level while training runs."""
    return HTMLResponse(CONTROL_PAGE)


@router.get("/straggler/control")
async def straggler_state() -> dict[str, Any]:
    return state.state()


@router.post("/straggler/control")
async def request_straggler_level(request: StragglerLevelRequest) -> dict[str, Any]:
    try:
        return state.request(request.level)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/straggler/poll")
async def poll_straggler_level(request: StragglerPollRequest) -> dict[str, Any]:
    """The primary rank reports the level in effect and reads the requested
    one."""
    return state.poll(request.model_dump())
