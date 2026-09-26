# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""GPU ownership coordination utilities for colocated deployments.

In colocated deployments, multiple training and inference roles share the same
GPUs. The SGLang ``torch_memory_saver`` static pool, Megatron weights, and
optimizer state all occupy GPU memory, so waking multiple roles simultaneously
causes contention. This module consolidates "wait for a peer to release its
GPUs" signals into two barriers:

- :class:`RolloutOffloadBarrier` — waits for rollout (SGLang) to finish offloading
- :class:`PeerStepBarrier` — waits for a group of training peers to finish a target round

Both barriers are stateless, lightweight polling wrappers around existing
service handles or the rollout manager. They add no overhead when unused in
fully asynchronous or hybrid deployments.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Optional

import ray


class RolloutOffloadBarrier:
    """Poll the rollout manager until SGLang has offloaded."""

    def __init__(
        self,
        rollout_manager: Any,
        poll_interval: float = 1.0,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self._rollout_manager = rollout_manager
        self._poll_interval = poll_interval
        self._logger = logger

    async def wait_offloaded(self) -> None:
        while ray.get(self._rollout_manager.get_status.remote()) == "onload":
            await asyncio.sleep(self._poll_interval)

    def wait_offloaded_sync(self) -> None:
        while ray.get(self._rollout_manager.get_status.remote()) == "onload":
            time.sleep(self._poll_interval)


class PeerStepBarrier:
    """Wait for colocated peers sharing GPUs to finish a target round.

    Each peer only needs to expose ``get_step()``, which is already provided by
    the component base class. A peer whose ``get_step()`` is strictly greater
    than ``round_id`` has completed all computation for that round, including
    backward, the optimizer step, and sleep, and has released its GPUs.

    Args:
        peers: Mapping from role names to Ray Serve deployment handles.
        poll_interval: Polling interval in seconds.
        log_every: Number of polling attempts between waiting log messages.
        logger: Logger to use. If omitted, waiting is silent.
    """

    def __init__(
        self,
        peers: dict[str, Any],
        poll_interval: float = 1.0,
        log_every: int = 30,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self._peers = dict(peers)
        self._poll_interval = poll_interval
        self._log_every = max(1, log_every)
        self._logger = logger

    def is_empty(self) -> bool:
        return not self._peers

    async def wait_completed_round(self, round_id: int) -> None:
        wait_count = 0
        while True:
            pending = await self.list_pending_async(round_id)
            if not pending:
                return
            self._maybe_log(pending, round_id, wait_count)
            wait_count += 1
            await asyncio.sleep(self._poll_interval)

    def wait_completed_round_sync(self, round_id: int) -> None:
        wait_count = 0
        while True:
            pending = self.list_pending_sync(round_id)
            if not pending:
                return
            self._maybe_log(pending, round_id, wait_count)
            wait_count += 1
            time.sleep(self._poll_interval)

    async def list_pending_async(self, round_id: int) -> list[tuple[str, int]]:
        """Non-blocking peek: peers whose ``get_step()`` is still ``<=
        round_id``."""
        pending: list[tuple[str, int]] = []
        for role, handle in self._peers.items():
            step = await handle.get_step.remote()
            if step <= round_id:
                pending.append((role, step))
        return pending

    def list_pending_sync(self, round_id: int) -> list[tuple[str, int]]:
        pending: list[tuple[str, int]] = []
        for role, handle in self._peers.items():
            step = handle.get_step.remote().result()
            if step <= round_id:
                pending.append((role, step))
        return pending

    def _maybe_log(self, pending: list[tuple[str, int]], round_id: int, wait_count: int) -> None:
        if self._logger is None or wait_count % self._log_every != 0:
            return
        desc = ", ".join(f"{r}(step={s})" for r, s in pending)
        self._logger.info(f"Waiting on peers to finish round {round_id}: {desc}")
