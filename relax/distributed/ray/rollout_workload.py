# Copyright (c) 2026 Relax Authors. All Rights Reserved.


from __future__ import annotations

import asyncio
from typing import Any

from relax.engine.rollout.base_types import call_rollout_fn
from relax.inference.defer import run_deferred_rollout


class RolloutWorkload:
    def __init__(self, facade: Any) -> None:
        self._facade = facade

    async def generate(self, rollout_id: int) -> None:

        manager = self._facade
        manager.rollout_id = rollout_id
        manager.health_monitoring_resume()
        if manager.args.ci_test and manager.args.use_fault_tolerance and rollout_id >= 2:
            manager._try_ci_fault_injection()
        if getattr(manager.args, "inference_defer_roles", None):
            await asyncio.to_thread(run_deferred_rollout, manager, rollout_id)
            return
        output = await asyncio.to_thread(
            call_rollout_fn,
            manager.generate_rollout,
            manager.args,
            rollout_id,
            manager.data_source,
            manager.data_system_client,
            evaluation=False,
        )
        if manager.args.partial_rollout and manager.args.use_dynamic_global_batch_size:
            manager._dynamic_global_batch_size = len(
                {sample.index for sample_group in output.samples for sample in sample_group}
            )

    async def evaluate(self, rollout_id: int) -> Any:

        manager = self._facade
        manager.health_monitoring_resume()
        return await asyncio.to_thread(
            call_rollout_fn,
            manager.eval_generate_rollout,
            manager.args,
            rollout_id,
            manager.data_source,
            manager.data_system_client,
            evaluation=True,
        )

    async def save(self, rollout_id: int) -> None:
        await self._facade.data_source.save.remote(rollout_id)

    async def load(self, rollout_id: int | None = None) -> None:
        await self._facade.data_source.load.remote(rollout_id)
