# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Optional real-CUDA fixtures; collection does not import SGLang."""

from collections.abc import Iterator
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import pytest

from .multimodal_test_utils import qwen_processor_context, require_cuda_devices


@pytest.fixture
def sglang_cuda() -> None:
    require_cuda_devices()


@pytest.fixture
def qwen_processor(sglang_cuda: None) -> Iterator[Any]:
    with qwen_processor_context() as processor:
        yield processor


@pytest.fixture
def cached_qwen_processor(sglang_cuda: None) -> Iterator[Any]:
    with qwen_processor_context(
        cache_size_mb=16,
        mm_process_config={"image": {"size": {"shortest_edge": 65536, "longest_edge": 262144}}},
    ) as processor:
        from sglang.srt.multimodal.cache import build_processor_fingerprint

        assert processor.mm_preprocess_cache.enabled
        assert processor.processor_fingerprint == build_processor_fingerprint(processor, processor.hf_config)
        yield processor


@pytest.fixture
def manager_pool(sglang_cuda: None) -> Iterator[tuple[Any, Any]]:
    from sglang.srt.managers.tokenizer_manager import TokenizerManager
    from sglang.srt.multimodal.transport.cuda_ipc import MmItemMemoryPool
    from sglang.srt.runtime_context import publish, reset_context
    from sglang.srt.server_args import ServerArgs

    publish(ServerArgs(model_path="dummy"), role="tokenizer")
    pool = MmItemMemoryPool(8 * 1024**2, 3600, 0, 1, max_inflight_slices=8)
    manager = TokenizerManager.__new__(TokenizerManager)
    manager.mm_processor = SimpleNamespace(cudaipc_mmfeature_pool=pool)
    manager.rid_to_state = {}
    manager.encoder_dispatch_ready = {}
    manager._lora_draining_requests = {}
    manager._lora_release_tasks = set()
    manager.child_rid_to_logical_rid = {}
    manager.logical_rid_to_child_rids = {}
    # The unused VMM transport is an external boundary. CUDA IPC remains real.
    manager.cuda_vmm_feature_transport = SimpleNamespace(
        prepare_for_dispatch_async=AsyncMock(return_value=[]),
        cancel_for_dispatch=lambda _: None,
    )
    try:
        yield manager, pool
    finally:
        pool.shutdown()
        reset_context()
