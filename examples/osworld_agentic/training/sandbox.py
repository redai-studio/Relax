# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Retry failed OSWorld preparation without drawing a replacement task."""

from __future__ import annotations

import asyncio
from copy import deepcopy
from dataclasses import replace
from functools import wraps
from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from relax.agentic.pipeline import GroupInput


_INSTALL_MARKER = "_osworld_fixed_task_prepare_retry_attempts"


def install_prepare_retry(runtime_domain: Any, max_attempts: int = 3) -> None:
    """Install once on a resident runtime; count the original call as attempt
    one.

    RuntimeDomain.prepare_group must finish cleaning up a failed attempt before
    returning None. Exceptions and cancellation retain the runtime's semantics.
    Only an explicitly accepted launch may retry; no reward is examined here.
    """
    if isinstance(max_attempts, bool) or not isinstance(max_attempts, int) or max_attempts < 1:
        raise ValueError("max_attempts must be a positive integer")
    installed = getattr(runtime_domain, _INSTALL_MARKER, None)
    if installed is not None:
        if installed != max_attempts:
            raise ValueError(f"Preparation retry is already installed with max_attempts={installed}")
        return

    original_prepare = runtime_domain.prepare_group

    @wraps(original_prepare)
    async def prepare_with_retry(
        group_input: GroupInput,
        launch_decision: asyncio.Future[bool] | None = None,
    ) -> Any:
        # The runtime writes session IDs into samples. Preserve their input state
        # before the first call, while passing that call the original objects.
        template = deepcopy(group_input)
        current_input = group_input
        for attempt in range(1, max_attempts + 1):
            stream = await original_prepare(current_input, launch_decision=launch_decision)
            if stream is not None:
                return stream
            # Do not wait for a decision owned by the caller after preparation
            # has ended: that would block the caller from observing completion.
            if launch_decision is None or not launch_decision.done():
                return None
            if launch_decision.result() is not True:
                return None
            task_ids = sorted({str(sample.metadata.get("task_id")) for sample in template.samples})
            if attempt == max_attempts:
                raise RuntimeError(
                    f"Fixed OSWorld preparation exhausted {max_attempts} attempts for "
                    f"tasks={task_ids!r}, group={template.group_id!r}; refusing to replace the task."
                )
            # A distinct runtime group and fresh samples isolate late messages
            # from the cleaned-up attempt. Logical sample/group indices survive.
            current_input = replace(
                template,
                group_id=f"{template.group_id}:retry_{attempt + 1}",
                samples=deepcopy(template.samples),
            )
            from relax.utils.logging_utils import get_logger

            get_logger(__name__).warning(
                "Fixed OSWorld prepare retry task_id=%s group_id=%s retry_group_id=%s attempt=%s/%s",
                ",".join(task_ids),
                template.group_id,
                current_input.group_id,
                attempt + 1,
                max_attempts,
            )

    runtime_domain.prepare_group = prepare_with_retry
    setattr(runtime_domain, _INSTALL_MARKER, max_attempts)
