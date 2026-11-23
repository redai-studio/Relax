# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""CPU workload host.

GPU handles and Router processes belong to the Manager.
"""

import asyncio
from typing import Any

import ray

from relax.engine.rollout.workload import RolloutWorkload
from relax.utils.http_utils import init_http_client
from relax.utils.tq.lifecycle import attach_tq_client, detach_tq_client
from relax.utils.tracking_utils import init_tracking


class _InferencePort:
    def __init__(self, args: Any, manager: Any) -> None:
        self.args = args
        self.manager = manager

    async def resume_health_monitoring(self) -> None:
        await self.manager.pool_call.remote("health_monitoring_resume")

    async def inject_ci_fault(self) -> None:
        await self.manager.inject_ci_fault.remote()

    async def onload_kv(self) -> None:
        await self.manager.pool_call.remote("onload_kv")

    def router_base_url(self, model_name: str = "default") -> str:
        return f"http://{self.args.sglang_router_ip}:{self.args.sglang_router_port}"


@ray.remote(num_cpus=1, num_gpus=0)
class RolloutWorker:
    def __init__(self, args: Any, data_source: Any, manager: Any) -> None:
        self.args = args
        self.manager = manager
        self._closed = False
        self._active: set[asyncio.Task] = set()
        init_tracking(args, primary=False)
        init_http_client(args)
        client = attach_tq_client(args.tq_config, role="rollout_worker")
        try:
            self.workload = RolloutWorkload(args, data_source, client, _InferencePort(args, manager))
        except BaseException:
            detach_tq_client()
            raise

    def configure(self, num_rollout: int, num_rollout_per_epoch: int) -> None:
        self.args.num_rollout = num_rollout
        self.args.num_rollout_per_epoch = num_rollout_per_epoch

    async def _run(self, method: str, *args: Any, **kwargs: Any) -> Any:
        if self._closed:
            raise RuntimeError("Rollout worker is closed")
        task = asyncio.create_task(getattr(self.workload, method)(*args, **kwargs))
        self._active.add(task)
        try:
            # A caller cancellation must not hide a still-running generation
            # thread from close(), which owns the TransferQueue attachment.
            return await asyncio.shield(task)
        finally:
            if task.done():
                self._active.discard(task)
            else:
                task.add_done_callback(self._active.discard)

    async def generate(self, rollout_id: int) -> None:
        await self.manager.begin_rollout.remote(rollout_id)
        await self._run("generate", rollout_id)

    async def eval(self, rollout_id: int) -> None:
        await self.manager.begin_rollout.remote(rollout_id)
        await self._run("eval", rollout_id)

    async def save(self, rollout_id: int) -> None:
        await self._run("save", rollout_id)

    async def load(self, rollout_id: int | None = None) -> None:
        await self._run("load", rollout_id)

    def get_num_rollout_per_epoch(self) -> int:
        return self.workload.get_num_rollout_per_epoch()

    def get_dynamic_global_batch_size(self) -> int:
        return self.workload.get_dynamic_global_batch_size()

    def set_train_parallel_config(self, config: dict) -> None:
        self.workload.set_train_parallel_config(config)

    def reload_function_by_name(self, module_name: str) -> dict:
        return self.workload.reload_function_by_name(module_name)

    def reload_all_functions(self) -> dict:
        return self.workload.reload_all_functions()

    def reload_module(self, module_name: str, module_path: str | None = None) -> dict:
        return self.workload.reload_module(module_name, module_path)

    def get_loaded_modules(self) -> dict:
        return self.workload.get_loaded_modules()

    def get_loaded_functions_info(self) -> dict:
        return self.workload.get_loaded_functions_info()

    async def close(self) -> None:
        self._closed = True
        if self._active:
            await asyncio.gather(*tuple(self._active), return_exceptions=True)
        detach_tq_client()
