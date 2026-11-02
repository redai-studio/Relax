# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Rollout data flow, isolated from the actor that owns inference resources."""

from __future__ import annotations

import asyncio
import sys
from contextlib import asynccontextmanager
from typing import Any

import ray

from relax.engine.inference.client import _settle
from relax.engine.inference.ownership import OwnershipError
from relax.engine.inference.phase_client import PhaseSession, control_call, current_phase, gather_owned
from relax.engine.rollout.base_types import call_rollout_fn
from relax.utils.http_utils import init_http_client, post
from relax.utils.logging_utils import get_logger
from relax.utils.misc import load_function
from relax.utils.reload_utils import ReloadableMixin
from relax.utils.s3_model_loader import prepare_model_maybe_update_args
from relax.utils.tq.lifecycle import attach_tq_client, detach_tq_client
from relax.utils.tracking_utils import init_tracking
from relax.utils.training.train_dump_utils import save_debug_rollout_data


logger = get_logger(__name__)
_LOCAL_WORKLOAD = None


def get_local_rollout_workload():
    if _LOCAL_WORKLOAD is None:
        raise RuntimeError("No RolloutWorkload is active in this process")
    return _LOCAL_WORKLOAD


@ray.remote(num_cpus=1, max_restarts=0)
class RolloutWorkload(ReloadableMixin):
    def __init__(self, args, manager, data_source=None):
        sys.modules[__name__]._LOCAL_WORKLOAD = self
        self.args = args
        self.inference_manager = manager
        self.data_source = data_source
        self._dynamic_global_batch_size = None
        self.rollout_id = -1
        self._tokenizer = None
        self._pending_operations = set()
        self._closing = False
        init_tracking(args, primary=False)
        init_http_client(args)
        for key, value in ray.get(manager.handle.runtime_config.remote(manager.model)).items():
            setattr(args, key, value)
        self.data_system_client = attach_tq_client(args.tq_config, role="rollout_worker")
        self.generate_rollout = load_function(args.rollout_function_path)
        self.eval_generate_rollout = load_function(args.eval_function_path)
        if args.use_agentic_rollout:
            from relax.agentic.rollout import bind_agentic_router, init_agentic_resident_pipeline

            init_agentic_resident_pipeline(args, data_source, self.data_system_client)
            bind_agentic_router(args.sglang_router_ip, args.sglang_router_port)

    def get_inference_manager(self):
        return self.inference_manager

    async def dispose(self):
        self._closing = True
        for task in tuple(self._pending_operations):
            try:
                await _settle(task)
            except BaseException:
                pass
        detach_tq_client()
        self.data_system_client = None

    @asynccontextmanager
    async def _phase(self, owner: str, *, offload_after: bool = False):
        if self._closing:
            raise RuntimeError("RolloutWorkload is shutting down")
        task = asyncio.current_task()
        self._pending_operations.add(task)
        try:
            async with PhaseSession.from_args(self.args, "rollout", owner) as phase:
                try:
                    await phase.aswitch("generate")
                    yield phase
                finally:
                    if offload_after and self.args.offload_rollout:
                        await phase.aswitch("train")
        finally:
            self._pending_operations.discard(task)

    async def _run_phase(self, fn, *args, evaluation=False):
        async with self._phase(f"rollout:{self.rollout_id}:{evaluation}", offload_after=not evaluation):
            if getattr(self.args, "ci_test", False) and getattr(self.args, "use_fault_tolerance", False):
                await self.inference_manager.inject_fault.remote()
            # The worker thread owns custom callbacks, deferred scoring and
            # publication. It must settle before the phase can release GPUs.
            return await control_call(
                asyncio.to_thread(
                    call_rollout_fn,
                    fn,
                    self.args,
                    *args,
                    self.data_source,
                    self.data_system_client,
                    evaluation=evaluation,
                )
            )

    def _offload_local(self, *, lease=None):
        lease = lease or current_phase().lease
        ray.get(self.inference_manager.offload.remote(lease=lease))

    def _onload_local(self, tags=None, *, lease=None):
        lease = lease or current_phase().lease
        ray.get(self.inference_manager.onload.remote(tags=tags, lease=lease))

    async def onload_kv(self):
        await self.inference_manager.onload_kv.remote()

    async def health_monitoring_resume(self):
        await self.inference_manager.health_monitoring_resume.remote()

    def get_dynamic_global_batch_size(self):
        """Return the actual sample count from the last rollout step.

        Used by training side to compute correct batch_size for TQ fetch when
        use_dynamic_global_batch_size is enabled.
        """
        assert self._dynamic_global_batch_size is not None, (
            "get_dynamic_global_batch_size called before first generate()"
        )
        return self._dynamic_global_batch_size

    def get_num_rollout_per_epoch(self):
        assert self.args.rollout_global_dataset
        return ray.get(self.data_source.lengths.remote()) // self.args.rollout_batch_size

    async def generate(self, rollout_id):
        self.rollout_id = rollout_id
        output = await self._run_phase(self.generate_rollout, rollout_id)
        if self.args.partial_rollout and self.args.use_dynamic_global_batch_size:
            self._dynamic_global_batch_size = len(
                {sample.index for sample_group in output.samples for sample in sample_group}
            )

    async def eval(self, rollout_id):
        # TODO: add fault tolerance to eval
        result = await self._run_phase(self.eval_generate_rollout, rollout_id, evaluation=True)
        data = result.data
        self._save_debug_rollout_data(data, rollout_id=rollout_id, evaluation=True)
        from relax.distributed.ray.rollout import _log_eval_rollout_data

        _log_eval_rollout_data(rollout_id, self.args, data, result.metrics)

    async def run_predict(self, train_step: int) -> None:
        """Periodic SFT predict pass entry point.

        Ensures KV/cuda-graph is onloaded (no-op if already on), then delegates
        to ``run_predict_loop`` which renders the eval set, batches calls to
        ``self.generate_predict``, and writes
        ``<args.save>/predict/predictions_step_<train_step>.jsonl``.

        Like eval, keeps the synced weights resident for the next generation.
        The next training step drives the resource transition.
        """
        from relax.engine.sft.predict.loop import run_predict_loop

        async with self._phase(f"predict:{train_step}"):
            await run_predict_loop(self, self.args, train_step)

    async def generate_predict(
        self,
        prompts: list[str],
        multimodal_inputs_list: list[dict | None] | None = None,
    ) -> list[str]:
        """Generate completions for ``prompts`` concurrently.

        POSTs all prompts at once in round-robin order directly to engine
        workers, bypassing the router. Predict prompts share a long fixed
        prefix (``<|vision_start|><|image_pad|><|vision_end|>...``); cache-
        aware routing would pin every request to the engine that first
        cached the prefix, defeating multi-engine throughput.

        ``multimodal_inputs_list`` is a parallel list of dicts (or ``None``)
        carrying images/videos/audio for each prompt; encoded inline and
        merged into the payload, mirroring the RL ``generate()`` path.
        """
        from relax.engine.rollout.sglang_rollout import _encode_multimodal_inputs

        try:
            current_phase()
        except OwnershipError:
            # Direct callers receive the same ownership window as /predict.
            async with self._phase("predict:batch"):
                return await self.generate_predict(prompts, multimodal_inputs_list)

        router_base = f"http://{self.args.sglang_router_ip}:{self.args.sglang_router_port}"
        # Reuse the shared --eval-* sampling args (no SFT-predict-specific
        # flags). Defaults preserve the original SFT predict behaviour
        # (greedy, max_new_tokens=512) when --eval-* is not provided.
        eval_temperature = getattr(self.args, "eval_temperature", None)
        eval_max_response_len = getattr(self.args, "eval_max_response_len", None)
        eval_top_p = getattr(self.args, "eval_top_p", None)
        sampling_params = {
            "temperature": 0.0 if eval_temperature is None else eval_temperature,
            "max_new_tokens": 512 if eval_max_response_len is None else eval_max_response_len,
            "top_p": 1.0 if eval_top_p is None else eval_top_p,
        }
        if multimodal_inputs_list is None:
            multimodal_inputs_list = [None] * len(prompts)

        async def _one(idx: int, prompt: str, mm: dict | None) -> str:
            url = f"{router_base}/generate"
            payload: dict[str, Any] = {"text": prompt, "sampling_params": sampling_params}
            if mm:
                encoded_mm, _ = await _encode_multimodal_inputs(mm)
                payload.update(encoded_mm)
            output = await post(url, payload, inference={"role": "rollout", "direct": True})
            if isinstance(output, dict) and "text" in output:
                return output["text"]
            if isinstance(output, str):
                return output
            return str(output)

        return await gather_owned(
            _one(i, p, m) for i, (p, m) in enumerate(zip(prompts, multimodal_inputs_list, strict=True))
        )

    async def save(self, rollout_id):
        await self.data_source.save.remote(rollout_id)

    async def load(self, rollout_id=None):
        try:
            await self.data_source.load.remote(rollout_id)
        except Exception as e:
            logger.warning(f"Failed to load data source: {e}")

    @property
    def tokenizer(self):
        """Lazy-initialized tokenizer for debug data saving."""
        if self._tokenizer is None:
            try:
                from relax.utils.data.processing_utils import load_tokenizer

                prepare_model_maybe_update_args(self.args, completeness="metadata")
                self._tokenizer = load_tokenizer(self.args.hf_checkpoint, trust_remote_code=True)
                logger.info(f"Loaded tokenizer from {self.args.hf_checkpoint}")
            except Exception as e:
                logger.warning(f"Failed to load tokenizer: {e}")
        return self._tokenizer

    def _save_debug_rollout_data(self, data, rollout_id, evaluation: bool):
        """Save debug rollout data using shared utility function."""
        save_debug_rollout_data(self.args, data, rollout_id, evaluation, tokenizer=self.tokenizer)

    async def get_rollout_engines_and_lock(self, *args, **kwargs):
        return await self.inference_manager.get_rollout_engines_and_lock.remote(*args, **kwargs)

    async def get_weight_sync_lock(self, *args, **kwargs):
        return await self.inference_manager.get_weight_sync_lock.remote(*args, **kwargs)

    async def offload(self, *args, **kwargs):
        return await self.inference_manager.offload.remote(*args, **kwargs)

    async def onload(self, *args, **kwargs):
        return await self.inference_manager.onload.remote(*args, **kwargs)

    async def onload_weights(self, *args, **kwargs):
        return await self.inference_manager.onload_weights.remote(*args, **kwargs)

    async def get_status(self, *args, **kwargs):
        return await self.inference_manager.get_status.remote(*args, **kwargs)

    async def recover_rollout_engines(self, *args, **kwargs):
        return await self.inference_manager.recover_rollout_engines.remote(*args, **kwargs)

    async def register_recovered_engines(self, *args, **kwargs):
        return await self.inference_manager.register_recovered_engines.remote(*args, **kwargs)

    async def clear_num_new_engines(self, *args, **kwargs):
        return await self.inference_manager.clear_num_new_engines.remote(*args, **kwargs)

    async def health_monitoring_pause(self, *args, **kwargs):
        return await self.inference_manager.health_monitoring_pause.remote(*args, **kwargs)

    async def check_weights(self, *args, **kwargs):
        return await self.inference_manager.check_weights.remote(*args, **kwargs)

    async def set_force_unhealthy(self, *args, **kwargs):
        return await self.inference_manager.set_force_unhealthy.remote(*args, **kwargs)

    async def set_train_parallel_config(self, *args, **kwargs):
        return await self.inference_manager.set_train_parallel_config.remote(*args, **kwargs)

    async def create_scale_out_request(self, *args, **kwargs):
        return await self.inference_manager.create_scale_out_request.remote(*args, **kwargs)

    async def execute_scale_out(self, *args, **kwargs):
        return await self.inference_manager.execute_scale_out.remote(*args, **kwargs)

    async def get_scale_out_status(self, *args, **kwargs):
        return await self.inference_manager.get_scale_out_status.remote(*args, **kwargs)

    async def cancel_scale_out(self, *args, **kwargs):
        return await self.inference_manager.cancel_scale_out.remote(*args, **kwargs)

    async def get_router_address(self, *args, **kwargs):
        return await self.inference_manager.get_router_address.remote(*args, **kwargs)

    async def get_engines_info(self, *args, **kwargs):
        return await self.inference_manager.get_engines_info.remote(*args, **kwargs)

    async def acquire_inference(self, *args, **kwargs):
        return await self.inference_manager.acquire_inference.remote(*args, **kwargs)

    async def release_inference(self, *args, **kwargs):
        return await self.inference_manager.release_inference.remote(*args, **kwargs)

    async def get_inference_snapshot(self, *args, **kwargs):
        return await self.inference_manager.get_inference_snapshot.remote(*args, **kwargs)

    async def list_all_scale_out_requests(self, *args, **kwargs):
        return await self.inference_manager.list_all_scale_out_requests.remote(*args, **kwargs)

    async def cancel_all_scale_out_requests(self, *args, **kwargs):
        return await self.inference_manager.cancel_all_scale_out_requests.remote(*args, **kwargs)

    async def set_weight_updating(self, *args, **kwargs):
        return await self.inference_manager.set_weight_updating.remote(*args, **kwargs)

    async def sync_weights_for_scaled_out_engines(self, *args, **kwargs):
        return await self.inference_manager.sync_weights_for_scaled_out_engines.remote(*args, **kwargs)

    async def create_scale_in_request(self, *args, **kwargs):
        return await self.inference_manager.create_scale_in_request.remote(*args, **kwargs)

    async def execute_scale_in(self, *args, **kwargs):
        return await self.inference_manager.execute_scale_in.remote(*args, **kwargs)

    async def get_scale_in_status(self, *args, **kwargs):
        return await self.inference_manager.get_scale_in_status.remote(*args, **kwargs)

    async def list_all_scale_in_requests(self, *args, **kwargs):
        return await self.inference_manager.list_all_scale_in_requests.remote(*args, **kwargs)
