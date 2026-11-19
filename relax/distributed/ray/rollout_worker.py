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


@ray.remote(num_cpus=1, num_gpus=0, concurrency_groups={"reload": 1})
class RolloutWorker:
    def __init__(self, args: Any, data_source: Any, manager: Any) -> None:
        self.args = args
        self.manager = manager
        self._closed = False
        self._startup_ready = False
        self._generation_running = False
        self._generation_failed = False
        self._generation_completed = False
        self._training_ready = False
        self._phase_lock = asyncio.Lock()
        self._active: set[asyncio.Task] = set()
        init_tracking(args, primary=False)
        init_http_client(args)
        client = attach_tq_client(args.tq_config, role="rollout_worker")
        self._agentic_started = False
        try:
            if getattr(args, "use_agentic_rollout", False):
                from relax.agentic.rollout import bind_agentic_router, init_agentic_resident_pipeline

                init_agentic_resident_pipeline(args, data_source, client)
                self._agentic_started = True
                bind_agentic_router(args.sglang_router_ip, args.sglang_router_port)
            self.workload = RolloutWorkload(args, data_source, client, _InferencePort(args, manager))
        except BaseException:
            if getattr(self, "_agentic_started", False):
                from relax.agentic.rollout import shutdown_agentic_resident_pipeline

                shutdown_agentic_resident_pipeline()
                self._agentic_started = False
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

    async def _activate_generation(self) -> None:
        async def activate() -> None:
            coordinator = getattr(self.args, "_inference_phase_coordinator", None)
            if coordinator is not None:
                from relax.engine.inference.phase_plans import PHASE_GENERATE

                await coordinator.switch.remote(PHASE_GENERATE)
            await self.manager.wait_ready.remote("rollout")

        # Only fully-async startup has this total budget, including queued RPCs.
        # Later weight updates and scoring phases retain their own drain budgets.
        if getattr(self.args, "fully_async", False) and not getattr(self, "_startup_ready", False):
            await asyncio.wait_for(activate(), timeout=120)
        else:
            await activate()
        self._startup_ready = True

    async def generate(self, rollout_id: int) -> None:
        shared = getattr(self.args, "_inference_phase_coordinator", None) is not None
        if shared:
            async with self._phase_lock:
                if self._closed:
                    raise RuntimeError("Rollout worker is closed")
                if self._generation_failed:
                    raise RuntimeError("Shared generation attempt failed; recovery requires a new RolloutWorker")
                if self._generation_running:
                    raise RuntimeError("Shared generation is already running")
                self._generation_running = True
                self._generation_completed = False
                self._training_ready = False
        try:
            await self._activate_generation()
            await self.manager.begin_rollout.remote(rollout_id)
            await self._run("generate", rollout_id)
            if shared:
                self._generation_completed = True
        except BaseException:
            if shared:
                # Cancellation may leave shielded work or an unknown TQ write
                # alive. This owner must never begin another attempt.
                self._generation_failed = True
            raise
        finally:
            if shared:
                self._generation_running = False

    async def offload(self) -> None:
        async with self._phase_lock:
            if self._generation_running or self._generation_failed:
                raise RuntimeError("Cannot release shared rollout GPUs before generation and scoring complete")
            await self.args._inference_phase_coordinator.switch.remote("startup")
            self._training_ready = self._generation_completed

    def get_status(self) -> str:
        return "offload" if self._training_ready else "onload"

    async def eval(self, rollout_id: int) -> None:
        if getattr(self.args, "_inference_phase_coordinator", None) is not None:
            if self._generation_failed:
                raise RuntimeError("Shared generation attempt failed; recovery requires a new RolloutWorker")
            if self._generation_running:
                raise RuntimeError("Shared generation is already running")
        await self._activate_generation()
        await self.manager.begin_rollout.remote(rollout_id)
        await self._run("eval", rollout_id)

    async def drain_eval_judges(self) -> None:
        await self._run("drain_eval_judges")

    async def run_predict(self, train_step: int) -> None:
        await self._activate_generation()
        await self._run("run_predict", train_step)

    async def generate_predict(
        self, prompts: list[str], multimodal_inputs_list: list[dict | None] | None = None
    ) -> Any:
        await self._activate_generation()
        return await self._run("generate_predict", prompts, multimodal_inputs_list)

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

    @ray.method(concurrency_group="reload")
    def reload_function_by_name(self, module_name: str) -> dict:
        return self.workload.reload_function_by_name(module_name)

    @ray.method(concurrency_group="reload")
    def reload_all_functions(self) -> dict:
        return self.workload.reload_all_functions()

    @ray.method(concurrency_group="reload")
    def reload_module(self, module_name: str, module_path: str | None = None) -> dict:
        return self.workload.reload_module(module_name, module_path)

    def get_loaded_modules(self) -> dict:
        return self.workload.get_loaded_modules()

    def get_loaded_functions_info(self) -> dict:
        return self.workload.get_loaded_functions_info()

    async def close(self) -> None:
        self._closed = True
        try:
            if self._active:
                await asyncio.gather(*tuple(self._active), return_exceptions=True)
            try:
                await self.workload.close()
            finally:
                if getattr(self, "_agentic_started", False):
                    from relax.agentic.rollout import shutdown_agentic_resident_pipeline

                    await asyncio.to_thread(shutdown_agentic_resident_pipeline)
                    self._agentic_started = False
        finally:
            detach_tq_client()
