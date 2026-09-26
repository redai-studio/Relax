# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Per-request inference permit for rollout request scheduling.

Concurrency control is expressed as one permit == one in-flight model request
(one HTTP request to the rollout engine, per turn for multi-turn rollouts).
Env/tool execution and CPU-side encoding must run *outside* the permit so that
a long multi-turn session does not hold a slot while it is not talking to the
engine. This mirrors the agentic path's acquire/release_sglang_request_permit
(relax/agentic/session/service.py).

This module intentionally depends only on the standard library so that the
permit logic can be unit-tested on CPU without constructing the heavy
``GenerateState`` singleton (which loads tokenizers/processors).
"""

import asyncio
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager


class GenerationAborted(Exception):
    """Raised when a permit is acquired after rollout abort has been signalled.

    Used as internal control flow inside the rollout dispatch layer; it is
    converted back into an ``ABORTED`` sample before it can escape to
    ``asyncio.gather`` (which is not created with ``return_exceptions=True``).
    """


class InferencePermitManager:
    """Bounds the number of concurrent in-flight model requests.

    NOTE: the underlying semaphore is NOT reentrant. A caller must not hold the
    session-level lock (``GenerateState.semaphore``, an alias of the same
    object) while acquiring a permit -- doing so deadlocks. See the
    ``manages_inference_permit`` opt-in contract in
    docs/{zh,en}/guide/customize-training.md.
    """

    def __init__(self, capacity: int) -> None:
        if capacity < 1:
            raise ValueError(
                f"InferencePermitManager capacity must be >= 1, got {capacity}. "
                "It is derived from sglang_server_concurrency * rollout_num_gpus // "
                "rollout_num_gpus_per_engine; a value < 1 would block every request forever."
            )
        self.capacity = capacity
        # Public: GenerateState exposes it as ``self.semaphore`` (backward-compat
        # alias) and the session-lock dispatch path acquires it directly.
        self.semaphore = asyncio.Semaphore(capacity)

    @asynccontextmanager
    async def permit(self, abort_check: Callable[[], bool] | None = None) -> AsyncIterator[None]:
        """Acquire one permit for the duration of the ``async with`` body.

        The permit is released on success, exception, or cancellation (the
        ``async with self.semaphore`` guarantees ``release()`` runs in all
        cases). If ``abort_check`` is provided and returns True right after the
        permit is acquired, ``GenerationAborted`` is raised (and the permit is
        released as the exception unwinds).
        """
        async with self.semaphore:
            if abort_check is not None and abort_check():
                raise GenerationAborted()
            yield
