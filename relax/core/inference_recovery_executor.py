# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Execute local inference recovery while Controller owns supervision."""

from __future__ import annotations

import asyncio
import time
from argparse import Namespace
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import ray
import transfer_queue as tq

from relax.distributed.coordination import RolloutOffloadBarrier
from relax.utils.async_utils import run
from relax.utils.logging_utils import get_logger


if TYPE_CHECKING:
    from relax.core.service import Service


logger = get_logger(__name__)


@dataclass(frozen=True)
class InferenceRecoveryResult:
    """Completed recovery steps awaiting Controller's task publication."""

    step: int
    attempt: int
    generation: int
    started: float


class InferenceRecoveryExecutor:
    """Use one generation's services without owning Controller's locks or
    threads."""

    def __init__(
        self,
        config: Namespace,
        services: Mapping[str, Service],
        state: Any,
        timeout: float,
        check_current: Callable[[], None],
    ) -> None:
        self.config = config
        self.services = services
        self.state = state
        self.timeout = timeout
        self._check_current = check_current

    def recover(self, role: str) -> InferenceRecoveryResult:
        """Run on the tracked recovery worker, including the entire rebuild."""
        self._check_current()
        state = self.state
        attempt = ray.get(state.request_recovery.remote(str(role)))
        started = time.monotonic()
        deadline = started + self.timeout
        actor = self.services["actor"]
        while True:
            self._check_current()
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("Old operations did not finish; require global restart")
            status = ray.get(state.snapshot.remote(), timeout=remaining)
            if status["unsafe"] or status["sync"] or set(status["failures"]) != {str(role)}:
                raise RuntimeError("Training or inference state changed during recovery")
            if not status["active"]:
                break
            # Release a healthy Fwd already waiting for the original Actor's weights.
            if "actor" not in status["active"] and status["fwd_ready"] and str(role) == "rollout":
                if not ray.get(state.begin_sync.remote(maintenance=True), timeout=remaining):
                    raise RuntimeError("Cannot release the healthy forward receiver")
                remaining = max(0, deadline - time.monotonic())
                run(asyncio.wait_for(actor.handle.update_weights_fully_async.remote(actor_fwd_only=True), remaining))
                ray.get(state.end_sync.remote(), timeout=max(0, deadline - time.monotonic()))
            time.sleep(0.05)

        step = run(self._prepare(role))
        self._check_current()
        # Cancellation of the waiter must not detach an untracked rebuild thread.
        self.services[role].rebuild_for_recovery(self._check_current)
        self._check_current()
        generation = run(self._synchronize(role, step))
        return InferenceRecoveryResult(step, attempt, generation, started)

    async def _prepare(self, role: str) -> int:
        self._check_current()
        rollout = self.services["rollout"]
        # The old loop tail can still exit after generation has drained.
        if str(role) == "actor_fwd" and await rollout.get_step() >= self.config.num_rollout - 1:
            raise RuntimeError("Rollout reached its final generation; require global restart")
        step = await self.services["actor"].get_step()
        client = tq.get_client()
        from relax.utils.tq.recovery_storage import require_strict_clear_contract, strict_clear_partition

        require_strict_clear_contract(client)
        for partition_id in await client.async_get_partition_list():
            if partition_id.startswith("train_") and int(partition_id.rsplit("_", 1)[1]) >= step:
                meta = await client._get_partition_meta(partition_id)
                await strict_clear_partition(client, partition_id, list(meta.global_indexes))
        await rollout.data_source.discard_pending.remote()
        manager = await rollout.get_rollout_manager()
        await manager.reset_data_window.remote()
        return step

    async def _synchronize(self, role: str, step: int) -> int:
        self._check_current()
        state = self.state
        actor = self.services["actor"]
        rollout = self.services["rollout"]
        failed = self.services[role]
        for component_role in ("rollout", "actor_fwd", "advantages"):
            if component_role in self.services:
                await self.services[component_role].set_step(step)
        if str(role) == "rollout":
            manager = await rollout.get_rollout_manager()
            await actor.handle.rebind_rollout_manager.remote(manager)
            await rollout.handle.resume_data_source.remote()
            if not getattr(self.config, "fully_async", False) and not getattr(self.config, "hybrid", False):
                await actor.set_barriers(rollout=RolloutOffloadBarrier(manager, logger=logger))
        if not ray.get(state.begin_sync.remote(maintenance=True)):
            raise RuntimeError("Cannot reserve replacement weight synchronization")
        if self.config.fully_async or getattr(self.config, "hybrid", False):
            sync_options = dict(rollout_only=str(role) == "rollout", actor_fwd_only=str(role) == "actor_fwd")
            if str(role) == "rollout":
                engines, *_ = await manager.get_rollout_engines_and_lock.remote()
                if not engines or any(engine is None for engine in engines):
                    raise RuntimeError("Replacement Rollout engines are incomplete; require global restart")
                sync_options["expected_rollout_actor_ids"] = [engine._actor_id.hex() for engine in engines]
            await actor.handle.reset_inference_transport.remote(str(role))
            sends = [actor.handle.update_weights_fully_async.remote(**sync_options)]
            if str(role) == "actor_fwd":
                sends.append(failed.handle.recv_weight_fully_async.remote())
            await asyncio.gather(*sends)
        else:
            await actor.handle.recovery_update_weights_colocate.remote()
        ray.get(state.end_sync.remote())
        # Every old writer is terminal and every replacement weight RPC succeeded.
        self._check_current()
        return ray.get(state.resume.remote(step))
