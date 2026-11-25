# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Cancellation while real multimodal executor work still owns CUDA IPC output.

The HF preprocessing boundary is controlled with thread events. Production
executor scheduling, item finalization, pool allocation, tokenizer
registration, and request-failure cleanup are all real. Run this against
main/candidate using PYTHONPATH; an old failure must be active_lease_count ==
1, not an import error.
"""

import asyncio
import threading
from types import SimpleNamespace
from typing import Any

import pytest


torch = pytest.importorskip("torch")

pytestmark = pytest.mark.usefixtures("sglang_cuda")


def test_cancel_during_thread_preprocessing_reclaims_late_ipc_lease(manager_pool: tuple[Any, Any]) -> None:
    from sglang.srt.managers.io_struct import GenerateReqInput
    from sglang.srt.managers.schedule_batch import (
        Modality,
        MultimodalDataItem,
        MultimodalProcessorOutput,
    )
    from sglang.srt.managers.tokenizer_manager import ReqState
    from sglang.srt.multimodal.processors.base_processor import (
        BaseMultimodalProcessor,
        BaseMultiModalProcessorOutput,
        MultimodalSpecialTokens,
    )
    from sglang.srt.multimodal.processors.executor import MultimodalProcessorExecutor
    from sglang.srt.observability.req_time_stats import APIServerReqTimeStats

    started = threading.Event()
    release_hf = threading.Event()
    finished = threading.Event()
    results: list[Any] = []
    worker_errors: list[str] = []

    class ControlledHFProcessor(BaseMultimodalProcessor):
        """Only the expensive HF result is substituted; ownership stays
        real."""

        def __init__(self, pool: Any) -> None:
            self._processor = SimpleNamespace(tokenizer=SimpleNamespace())
            self._tokenizer = self._processor.tokenizer
            self.cudaipc_mmfeature_pool = pool
            self.use_cuda_ipc = True
            self.use_ipc_pool_handle_cache = True
            self.precompute_hash_before_cpu_transfer = False
            self.mm_processor_executor = MultimodalProcessorExecutor(lambda: self._processor, max_workers=2)

        def _process_and_collect_mm_items(self, **kwargs: Any) -> tuple[list[Any], Any, dict[str, Any]]:
            started.set()
            if not release_hf.wait(10):
                raise TimeoutError("test did not release the controlled HF boundary")
            torch.cuda.set_device(0)
            item = MultimodalDataItem(
                modality=Modality.IMAGE,
                feature=torch.arange(64, device="cuda", dtype=torch.float32),
                offsets=[(0, 0)],
            )
            return [item], torch.tensor([1], dtype=torch.long), {}

        def process_and_combine_mm_data(self, *args: Any, **kwargs: Any) -> Any:
            # Observe completion without replacing finalization or transport.
            try:
                result = super().process_and_combine_mm_data(*args, **kwargs)
                results.append(result)
                return result
            except BaseException as exc:
                worker_errors.append(repr(exc))
                raise
            finally:
                finished.set()

        async def process_mm_data_async(self, **kwargs: Any) -> Any:
            base = BaseMultiModalProcessorOutput(input_text="unused", images=[object()])
            items, ids, _ = await self.process_and_combine_mm_data_async(
                base, MultimodalSpecialTokens(image_token_id=1)
            )
            return MultimodalProcessorOutput(input_ids=ids.tolist(), mm_items=items)

    manager, pool = manager_pool
    processor = ControlledHFProcessor(pool)
    manager.mm_processor = processor
    manager.model_config = SimpleNamespace(hf_config=SimpleNamespace(architectures=[]))
    manager.max_req_input_len = 1024

    async def scenario() -> None:
        rid = "cancel-during-hf-thread"
        request = GenerateReqInput(rid=rid, input_ids=[1], image_data=[object()], video_data=None, audio_data=None)
        state = ReqState([], False, asyncio.Event(), request, APIServerReqTimeStats())
        manager.rid_to_state[rid] = state
        request_task = asyncio.create_task(manager._tokenize_one_request(request))
        try:
            entered = await asyncio.to_thread(started.wait, 10)
            if not entered and request_task.done():
                # Surface actual setup errors instead of calling them a leak.
                await request_task
            assert entered, "actual worker never reached the controlled HF boundary"
            assert pool.active_lease_count == 0
            request_task.cancel()
            # Let cancellation reach the awaited executor boundary before the
            # thread creates the actual CUDA IPC allocation.
            await asyncio.sleep(0)
            release_hf.set()
            with pytest.raises(asyncio.CancelledError):
                await request_task
            manager._release_req_states_on_failure([rid])
            assert await asyncio.to_thread(finished.wait, 10)
            assert not worker_errors, worker_errors
            assert len(results) == 1, "the real worker did not finish its output"
            assert len(results[0][0]) == 1
            # A shielded task's cleanup callback runs on the event loop.
            deadline = asyncio.get_running_loop().time() + 2
            while True:
                torch.cuda.synchronize()
                with pool._pool._lock:
                    pool._pool._recycle_ready_leases_locked()
                if pool.active_lease_count == 0:
                    break
                if asyncio.get_running_loop().time() >= deadline:
                    break
                await asyncio.sleep(0.01)
            assert rid not in manager.rid_to_state
            assert pool.active_lease_count == 0, (
                "late thread output retained a real CUDA IPC producer lease after tokenizer cancellation"
            )
        finally:
            release_hf.set()
            if not request_task.done():
                request_task.cancel()
                await asyncio.gather(request_task, return_exceptions=True)

    try:
        asyncio.run(scenario())
    finally:
        release_hf.set()
        processor.mm_processor_executor.shutdown()
        # Teardown occurs after assertions. Clean the deliberately exposed old
        # implementation's leaked slice without changing the observed result.
        for items, _, _ in results:
            for item in items:
                proxy = item.feature
                if hasattr(proxy, "pool_handle"):
                    try:
                        pool.cancel_proxy(proxy)
                    except Exception:
                        pass
