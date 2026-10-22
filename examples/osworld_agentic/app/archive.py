# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""OSWorld observations and evaluator semantics mapped onto the public
archive."""

from __future__ import annotations

import base64
import functools
import hashlib
import json
import math
import os
import time
import uuid
from typing import Any, Callable

from relax.agentic.archive.client import Recorder
from relax.agentic.archive.schema import encode
from relax.utils.logging_utils import get_logger


logger = get_logger(__name__)


def fail_open(function: Callable) -> Callable:
    @functools.wraps(function)
    def wrapped(self: OSWorldCapture, *args: Any, **kwargs: Any) -> Any:
        if not self.recorder.trajectory_id:
            return None
        try:
            return function(self, *args, **kwargs)
        except Exception:
            logger.warning("OSWorld archive capture failed; training continues", exc_info=True)
            self.recorder.abort()
            return None

    return wrapped


def create_capture(task: dict, config: dict, metadata: dict) -> OSWorldCapture | None:
    if os.environ.get("CUA_ARCHIVE_ENABLE") != "1":
        return None
    try:
        task_version = hashlib.sha256(encode(task)).hexdigest()
        recorder = Recorder(
            os.environ["CUA_ARCHIVE_URL"],
            {
                "run_id": os.environ["OSWORLD_RUN_TAG"],
                "session_id": os.environ["RELAX_SESSION_ID"],
                "task_source": "osworld",
                "task_id": task.get("id"),
                "task_version": task_version,
                "instruction": task.get("instruction"),
                "task_config": task,
                "model": os.environ.get("OPENAI_MODEL", "unknown"),
                "policy_version": metadata.get("policy_version", "unknown"),
                "agent_config": config,
                "environment_version": "unknown",
                "started_at": time.time(),
            },
            os.environ.get("CUA_ARCHIVE_TOKEN", ""),
        )
        return OSWorldCapture(recorder)
    except Exception:
        logger.warning("OSWorld archive unavailable; training continues", exc_info=True)
        return None


class OSWorldCapture:
    def __init__(self, recorder: Recorder) -> None:
        self.recorder = recorder
        self.step = 0
        self.observation_id: str | None = None
        self.step_calls: list[str] = []
        self.selected_call: str | None = None
        self.action_order = 0

    @fail_open
    def observe(self, obs: Any, step: int) -> str:
        identity = uuid.uuid4().hex
        valid = bool(getattr(obs, "screenshot_valid", False))
        self.recorder.event(
            "observation",
            observation_id=identity,
            time=time.time(),
            step=step,
            screenshot="data:image/png;base64," + base64.b64encode(obs.screenshot_bytes).decode() if valid else None,
            valid=valid,
            fields=getattr(obs, "archive_fields", {}),
        )
        self.observation_id = identity
        return identity

    @fail_open
    def before_predict(self, step: int) -> None:
        self.step, self.step_calls, self.selected_call = step, [], None

    @fail_open
    def model_start(self, payload: dict, model: str, retry: int) -> str:
        identity = uuid.uuid4().hex
        self.recorder.event(
            "model_call",
            call_id=identity,
            time=time.time(),
            step=self.step,
            transport_retry=retry,
            parse_retry=len(self.step_calls),
            request=payload,
            model=model,
            policy_version="unknown",
            observation_ids=[self.observation_id],
            parse_status="not_parsed",
            selected=False,
        )
        return identity

    @fail_open
    def model_result(self, call_id: str, body: str | None, status: int | None, error: str | None = None) -> None:
        message = None
        if status == 200:
            try:
                message = json.loads(body)["choices"][0]["message"]
            except (ValueError, KeyError, IndexError, TypeError):
                pass
        self.recorder.event(
            "call_result",
            call_id=call_id,
            response={"body": body, "http_status": status, "error": error},
            assistant_message=message,
            ended_at=time.time(),
        )
        if message and message.get("content"):
            self.step_calls.append(call_id)

    @fail_open
    def parsed(self, records: list[dict]) -> None:
        if len(records) != len(self.step_calls):
            self.recorder.event("incomplete", reason="model calls cannot be matched to parser records")
            return
        for call_id, record in zip(self.step_calls, records, strict=True):
            selected = record["accepted_for_execution"]
            self.recorder.event(
                "call_parse",
                call_id=call_id,
                parse_retry=record["parse_attempt"],
                selected=selected,
                parse_status="parsed" if selected else "not_parsed",
            )
            if selected:
                self.selected_call = call_id

    @fail_open
    def action_start(self, action: str, *, terminal: bool = False) -> str:
        from .env_osworld import parse_mcp_action

        identity = uuid.uuid4().hex
        mcp = parse_mcp_action(action)
        payload = {"tool_name": mcp[0], "parameters": mcp[1], "raw": action} if mcp else {"code": action}
        self.recorder.event(
            "action",
            action_id=identity,
            call_id=self.selected_call or (self.step_calls[-1] if self.step_calls else None),
            order=self.action_order,
            step=self.step,
            time=time.time(),
            type="terminal" if terminal else "mcp" if mcp else "pyautogui",
            payload=payload,
            origin="model" if self.selected_call else "harness",
            before_observation_id=self.observation_id,
        )
        self.action_order += 1
        return identity

    @fail_open
    def action_result(self, action_id: str, obs: Any) -> None:
        after = self.observe(obs, self.step)
        self.recorder.event(
            "action_result",
            action_id=action_id,
            after_observation_id=after,
            ended_at=time.time(),
            status="unknown" if obs.exec_ok else "failure",
            execution_evidence="rpc_completed",
            result=obs.exec_result,
            error=None if obs.exec_ok else obs.exec_result.get("error", "reported failure"),
        )

    @fail_open
    def finish(self, raw: Any, error: str | None, reason: str) -> None:
        if reason in {"predict_error", "context_length_exceeded", "discarded_by_pipeline"}:
            self.recorder.event("incomplete", reason=reason)
        # Only this adapter interprets the evaluator's scalar convention.
        score = float(raw) if raw is not None else None
        valid = score is not None and math.isfinite(score) and error is None
        status = "unknown"
        if valid:
            status = (
                "success" if score == 1 else "failure" if score == 0 else "partial" if 0 < score < 1 else "unknown"
            )
        self.recorder.finish(
            {
                "status": status,
                "raw_score": raw if valid else None,
                "evaluation_error": error,
                "basis": "OSWorld archive: 1=success, 0=failure, (0,1)=partial; other/missing/error=unknown",
                "final_observation_id": self.observation_id,
                "ended_at": time.time(),
                "termination_reason": reason,
                "result_files": [],
                "result_files_availability": "not exposed by the adapter",
            }
        )

    def abort(self) -> None:
        self.recorder.abort()
