# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""CPU Ray host for generation, evaluation and rollout data flow."""

from typing import Any

import ray
import transfer_queue as tq

from relax.engine.rollout.workload import RolloutWorkload
from relax.utils.http_utils import init_http_client
from relax.utils.tracking_utils import init_tracking


_LOCAL_INFERENCE_MANAGER: Any = None


def get_local_inference_manager() -> Any:
    """Return the task inference handle inside the rollout worker process."""
    import sys

    manager = getattr(sys.modules[__name__], "_LOCAL_INFERENCE_MANAGER", None)
    if manager is None:
        raise RuntimeError("Inference manager is unavailable outside an initialized RolloutWorker")
    return manager


class _InferencePort:
    """Connect workload operations directly to the task inference owner."""

    def __init__(self, args: Any, manager: Any) -> None:
        self.args = args
        self.manager = manager

    async def resume_health_monitoring(self) -> None:
        await self.manager.begin_rollout.remote()

    async def inject_ci_fault(self) -> None:
        await self.manager.rollout_operation.remote("inject_ci_fault")

    async def onload_kv(self) -> None:
        from sglang.srt.constants import GPU_MEMORY_TYPE_CUDA_GRAPH, GPU_MEMORY_TYPE_KV_CACHE

        from relax.engine.inference.types import Role

        await self.manager.activate.remote(Role.ROLLOUT, tags=[GPU_MEMORY_TYPE_KV_CACHE, GPU_MEMORY_TYPE_CUDA_GRAPH])

    def router_base_url(self, model_name: str = "default") -> str:
        return f"http://{self.args.sglang_router_ip}:{self.args.sglang_router_port}"


@ray.remote
class RolloutWorker:
    """Own the rollout workload; engine operations go to InferenceManager."""

    def __init__(self, args: Any, data_source: Any, inference_manager_handle: Any) -> None:
        import sys

        if inference_manager_handle is None:
            raise ValueError("RolloutWorker requires the task inference manager handle")
        self.args = args
        init_tracking(args, primary=False)
        init_http_client(args)
        tq.init(args.tq_config)
        self.workload = RolloutWorkload(
            args, data_source, tq.get_client(), _InferencePort(args, inference_manager_handle)
        )
        # Ray/cloudpickle may reconstruct class globals outside the module namespace.
        sys.modules[__name__]._LOCAL_INFERENCE_MANAGER = inference_manager_handle
        if args.use_agentic_rollout:
            from relax.agentic.rollout import init_agentic_resident_pipeline

            init_agentic_resident_pipeline(args, data_source, self.workload.data_system_client)

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

    def get_dynamic_global_batch_size(self) -> int:
        return self.workload.get_dynamic_global_batch_size()

    def get_num_rollout_per_epoch(self) -> int:
        return self.workload.get_num_rollout_per_epoch()

    async def generate(self, rollout_id: int) -> None:
        await self.workload.generate(rollout_id)

    async def eval(self, rollout_id: int) -> None:
        await self.workload.eval(rollout_id)

    async def run_predict(self, train_step: int) -> None:
        await self.workload.run_predict(train_step)

    async def generate_predict(
        self,
        prompts: list[str],
        multimodal_inputs_list: list[dict | None] | None = None,
    ) -> list[str]:
        return await self.workload.generate_predict(prompts, multimodal_inputs_list)

    async def save(self, rollout_id: int) -> None:
        await self.workload.save(rollout_id)

    async def load(self, rollout_id: int | None = None) -> None:
        await self.workload.load(rollout_id)

    def set_train_parallel_config(self, config: dict) -> None:
        self.workload.set_train_parallel_config(config)
