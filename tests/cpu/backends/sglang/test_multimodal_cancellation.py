# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Cancel real tokenizer/executor work before it creates multimodal output."""

import asyncio
import threading
from types import SimpleNamespace
from typing import Any

import pytest


executor_module = pytest.importorskip("sglang.srt.multimodal.processors.executor")


def test_tokenizer_cancel_during_media_load_does_not_start_preprocessing() -> None:
    from sglang.srt.managers.io_struct import GenerateReqInput
    from sglang.srt.managers.tokenizer_manager import TokenizerManager
    from sglang.srt.runtime_context import publish, reset_context
    from sglang.srt.server_args import ServerArgs

    publish(ServerArgs(model_path="dummy"), role="tokenizer")
    executor = executor_module.MultimodalProcessorExecutor(SimpleNamespace, max_workers=1)
    preprocess_calls: list[Any] = []

    def preprocess(*, processor: Any) -> None:
        preprocess_calls.append(processor)
        return None

    async def scenario() -> None:
        loading, release_media, finished = asyncio.Event(), asyncio.Event(), asyncio.Event()

        async def process_mm_data_async(**kwargs: Any) -> None:
            try:
                # Only media I/O is controlled; tokenizer cancellation and
                # subsequent executor scheduling use their production paths.
                loading.set()
                await release_media.wait()
                return await executor.run(preprocess)
            finally:
                finished.set()

        manager = TokenizerManager.__new__(TokenizerManager)
        manager.mm_processor = SimpleNamespace(
            process_mm_data_async=process_mm_data_async,
            prefer_tokenized_input=False,
        )
        manager.rid_to_state = {}
        manager.model_config = SimpleNamespace(hf_config=SimpleNamespace(architectures=[]))
        manager.max_req_input_len = 1024
        request = GenerateReqInput(rid="cancel-media", input_ids=[1], image_data=[object()])
        task = asyncio.create_task(manager._tokenize_one_request(request))
        try:
            await asyncio.wait_for(loading.wait(), 2)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(task, 2)
            release_media.set()
            await asyncio.wait_for(finished.wait(), 2)
            assert not preprocess_calls, "cancelled media loading started new preprocessing"
        finally:
            release_media.set()
            if not task.done():
                task.cancel()
            await asyncio.wait_for(asyncio.gather(task, return_exceptions=True), 2)
            await asyncio.wait_for(finished.wait(), 2)

    try:
        asyncio.run(scenario())
    finally:
        executor.shutdown()
        reset_context()


@pytest.mark.parametrize("stage,cleanup", [("queued", False), ("queued", True), ("running", True)])
def test_processor_executor_cancellation_discards_only_running_results(stage: str, cleanup: bool) -> None:
    executor = executor_module.MultimodalProcessorExecutor(SimpleNamespace, max_workers=1)
    worker_started, release_worker, processing_started = threading.Event(), threading.Event(), threading.Event()
    output = object()
    discarded: list[object] = []

    def occupy_worker(*, processor: Any) -> None:
        worker_started.set()
        if not release_worker.wait(10):
            raise TimeoutError("test did not release the occupied worker")

    def process(*, processor: Any) -> object:
        processing_started.set()
        if not release_worker.wait(10):
            raise TimeoutError("test did not release preprocessing")
        return output

    async def scenario() -> None:
        submitted, cleaned = asyncio.Event(), asyncio.Event()

        def discard(result: object) -> None:
            discarded.append(result)
            cleaned.set()

        async def run_processing() -> object:
            submitted.set()
            return await executor.run(process, on_cancelled_result=discard if cleanup else None)

        blocker = None
        task = None
        try:
            if stage == "queued":
                blocker = asyncio.create_task(executor.run(occupy_worker))
                assert await asyncio.to_thread(worker_started.wait, 2)
            task = asyncio.create_task(run_processing())
            # run_processing has no suspension before the actual executor
            # submission, so waking here observes a submitted job.
            await asyncio.wait_for(submitted.wait(), 2)
            if stage == "running":
                assert await asyncio.to_thread(processing_started.wait, 2)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(task, 2)
            release_worker.set()
            if blocker is not None:
                await asyncio.wait_for(blocker, 2)
            # Drain the real FIFO queue before checking a cancelled queued job.
            await asyncio.wait_for(executor.run(lambda *, processor: None), 2)
            if stage == "queued":
                assert not processing_started.is_set(), "cancelled queued preprocessing still ran"
                assert discarded == []
            else:
                await asyncio.wait_for(cleaned.wait(), 2)
                assert discarded == [output], "late worker output was not returned exactly once"
        finally:
            release_worker.set()
            pending = [candidate for candidate in (task, blocker) if candidate is not None]
            for candidate in pending:
                if not candidate.done():
                    candidate.cancel()
            await asyncio.wait_for(asyncio.gather(*pending, return_exceptions=True), 2)

    try:
        asyncio.run(scenario())
    finally:
        release_worker.set()
        executor.shutdown()
