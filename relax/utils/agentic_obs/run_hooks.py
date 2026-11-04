# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Driver-owned Run outcomes, independent of collection freshness."""

from __future__ import annotations

import threading
from argparse import Namespace
from functools import wraps
from typing import Any, Callable, Literal

from relax.utils.agentic_obs.config import agentic_obs_enabled, resolve_agentic_obs_config, write_manifest
from relax.utils.agentic_obs.records import ExportStageRecord
from relax.utils.agentic_obs.writers import JsonlAppender
from relax.utils.logging_utils import get_logger


logger = get_logger(__name__)
_FAILURE_STAGE_ATTRIBUTE = "_relax_run_failure_stage"


def annotate_failure_stage(stage: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Tag a backend exception without replacing its type or traceback.

    RayTaskError transports the original exception as ``cause``. The driver can
    recover the stage from it even after a service wraps the remote error. This
    decorator is for backend methods whose owner carries ``args``.
    """

    def decorate(function: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(function)
        def call(owner: Any, *args: Any, **kwargs: Any) -> Any:
            try:
                return function(owner, *args, **kwargs)
            except BaseException as error:
                if agentic_obs_enabled(getattr(owner, "args", None)):
                    try:
                        setattr(error, _FAILURE_STAGE_ATTRIBUTE, stage)
                    except Exception:  # Some exception types disallow extra attributes.
                        pass
                raise

        return call

    return decorate


def _failure_details(error: BaseException, operation: str) -> tuple[str, str]:
    pending = [error]
    seen: set[int] = set()
    cause = error
    marked = False
    while pending and len(seen) < 64:
        current = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        marked_stage = getattr(current, _FAILURE_STAGE_ATTRIBUTE, None)
        if isinstance(marked_stage, str):
            operation = marked_stage
            cause = current
            marked = True
        elif not marked:
            cause = current
        for linked in (current.__context__, current.__cause__, getattr(current, "cause", None)):
            if isinstance(linked, BaseException):
                pending.append(linked)
    return operation, f"{type(cause).__name__}: {cause}"


class RunRecorder:
    """Write one start and the first known outcome for a training driver.

    Capture the outcome before cleanup; flush it after the shutdown watchdog is
    armed and before remote teardown. Synchronous writes avoid relying on an
    atexit drain that ``os._exit`` would skip.
    """

    def __init__(self, args: Namespace) -> None:
        self._args = args
        self._config = resolve_agentic_obs_config(args)
        self._lock = threading.RLock()  # A signal may interrupt the main thread.
        self._operation = "startup"
        self._started = False
        self._flushed = False
        self._outcome: ExportStageRecord | None = None
        self._start = (
            ExportStageRecord(
                None, "execution", "start", "", run_id=self._config.run_id, scope="run", operation="startup"
            )
            if self._config is not None
            else None
        )

    def start(self) -> None:
        if self._config is None:
            return
        with self._lock:
            if self._started:
                return
            self._started = True
        write_manifest(self._args, self._config)
        self._append(self._start)

    def set_phase(self, operation: str) -> None:
        if self._config is None:
            return
        with self._lock:
            if self._outcome is None:
                self._operation = operation

    def record_outcome(
        self,
        status: Literal["completed", "failed", "cancelled"],
        *,
        error: BaseException | None = None,
        reason_code: str | None = None,
        reason: str | None = None,
    ) -> None:
        if self._config is None:
            return
        with self._lock:
            if self._outcome is not None:
                return
            operation = self._operation
        if error is not None:
            try:
                operation, reason = _failure_details(error, operation)
            except Exception:
                reason = type(error).__name__
        outcome = ExportStageRecord(
            None,
            "execution",
            status,
            "",
            run_id=self._config.run_id,
            scope="run",
            stage_id=self._start.stage_id,
            event_seq=2,
            is_terminal=True,
            operation=operation,
            reason_code=reason_code or (f"run_{status}" if status != "completed" else None),
            reason=reason,
            category=None if status == "completed" else "training",
        )
        with self._lock:
            # A signal can record an outcome while error formatting runs.
            if self._outcome is None:
                self._outcome = outcome

    def flush(self) -> None:
        if self._config is None:
            return
        with self._lock:
            if self._outcome is None or self._flushed:
                return
            self._flushed = True
            outcome = self._outcome
        self.start()
        self._append(outcome)

    def _append(self, record: ExportStageRecord) -> None:
        writer = None
        try:
            writer = JsonlAppender(self._config.stages_dir / "run.jsonl")
            writer.append(record)
        except Exception as error:
            logger.warning("agentic_obs: run event write failed: %s", error)
        finally:
            if writer is not None:
                try:
                    writer.close()
                except Exception as error:
                    logger.warning("agentic_obs: run event close failed: %s", error)
