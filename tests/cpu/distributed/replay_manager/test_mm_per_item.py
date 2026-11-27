# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Run the installed SGLang per-image scheduler with CPU vision outputs.

The native HIP branch, cache, chunk slicing and Relax publication are real.
Only the vision encoder, scheduler process and response transport are isolated.
"""

import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

from relax.distributed.replay_manager.backend import mm_sglang, sglang
from relax.distributed.replay_manager.mm_embedding import read_mm_embedding, release_media_entries
from tests.cpu.distributed.replay_manager.test_mm_sglang import capture, finish, item  # noqa: F401


@pytest.mark.parametrize("case", ["cold-tensor", "cold-list", "cache-hit", "eviction"])
def test_hip_per_item_capture_publishes_complete_embeddings_before_chunking(monkeypatch, capture, case):  # noqa: F811
    schedule = pytest.importorskip("sglang.srt.managers.mm_schedule")
    context = pytest.importorskip("sglang.srt.runtime_context")
    runtime, store, _ = capture
    # Restore every native function that install_capture_patch replaces.
    for name in (
        "_batch_encode_per_image_misses",
        "_get_chunked_embedding_by_item",
        "_get_precomputed_embedding",
        "_get_chunked_embedding_full",
        "get_embedding_chunk",
    ):
        monkeypatch.setattr(schedule, name, getattr(schedule, name))
    monkeypatch.setattr(schedule, "_is_hip", True)
    monkeypatch.setattr(schedule, "embedding_cache", schedule.MultiModalStaticCache(max_size=4 * 2560 * 2))
    if case.startswith("cold"):
        schedule.embedding_cache.max_size = 0
    monkeypatch.setattr(context, "get_parallel", lambda: SimpleNamespace(attn_tp_rank=0, tp_size=1))
    monkeypatch.setattr(schedule, "get_parallel", context.get_parallel)

    images = [item(12)] + ([item(13, 4)] if case == "eviction" else [])
    expected = {}
    for index, image in enumerate(images):
        image.feature = torch.zeros(1)
        image.model_specific_data = {}
        image.acknowledge_deferred_cuda_ipc_feature = lambda _: None
        expected[image.hash] = (torch.arange(4 * 2560).reshape(4, 2560) + index * 20000).bfloat16()
    if case in {"cache-hit", "eviction"}:
        schedule.embedding_cache.set(images[0].hash, schedule.EmbeddingResult(embedding=expected[images[0].hash]))

    encoded = []

    class Qwen3_5MoeForConditionalGeneration:
        def get_image_feature(self, items):
            encoded.extend(image.hash for image in items)
            values = [expected[image.hash] for image in items]
            return values if case == "cold-list" else torch.cat(values)

    encoder = Qwen3_5MoeForConditionalGeneration()
    original_move = schedule._move_items_to_device

    def move(items, device, embedding_func):
        # Wrapping capture must preserve the encoder's native H2D policy.
        assert schedule._can_skip_pre_embed_feature_move(embedding_func)
        return original_move(items, device, embedding_func)

    monkeypatch.setattr(schedule, "_move_items_to_device", move)
    offsets = [image.offsets[0] for image in images]
    prefix, length = (2, 4) if case == "eviction" else (1, 2)
    utilities = ModuleType("sglang.srt.managers.mm_utils")

    def general(input_ids, forward_batch):
        return schedule._get_chunked_prefill_embedding(
            encoder.get_image_feature, images, [0, len(images)], [prefix], [length], [offsets], input_ids
        )

    utilities.general_mm_embed_routine = general
    scheduler_module = ModuleType("sglang.srt.managers.scheduler")
    scheduler_module.Scheduler = type(
        "Scheduler",
        (),
        {
            "run_batch": lambda *_: None,
            "flush_cache": lambda *_: True,
            "get_output_streamer_class": lambda *_: type("Streamer", (), {}),
        },
    )
    requests = ModuleType("sglang.srt.managers.schedule_batch")
    requests.Req = type("Req", (), {"set_finish_with_abort": lambda *_: None})
    for module in (utilities, scheduler_module, requests):
        monkeypatch.setitem(sys.modules, module.__name__, module)
        monkeypatch.setattr(
            sys.modules["sglang.srt.managers"], module.__name__.rsplit(".", 1)[1], module, raising=False
        )
    monkeypatch.setattr(sglang, "_install_response_patch", lambda *_: None)
    monkeypatch.setattr(sglang, "_install_abort_patch", lambda: None)
    monkeypatch.setattr(mm_sglang, "model_hidden_size", lambda: 2560)
    monkeypatch.setattr(mm_sglang, "MMCapture", lambda *a, **kw: runtime)
    monkeypatch.setenv("RELAX_REPLAY_ENGINE_PUBLISHER", '[0, "engine"]')
    mm_sglang.install_capture_patch(SimpleNamespace(store=lambda: store))

    rid = mm_sglang.REQUEST_PREFIX + case
    chunk, _ = utilities.general_mm_embed_routine(
        torch.zeros(length, dtype=torch.int64),
        SimpleNamespace(rids=[rid], mm_inputs=[SimpleNamespace(mm_items=images)]),
    )
    entries = finish(runtime, rid)
    try:
        torch.testing.assert_close(chunk, torch.cat(list(expected.values()))[prefix : prefix + length])
        assert len(entries) == len(images)
        for image, entry in zip(images, entries, strict=True):
            restored = read_mm_embedding(store, entry)
            torch.testing.assert_close(restored.view(torch.uint8), expected[image.hash].view(torch.uint8))
        assert encoded == {"cold-tensor": [12], "cold-list": [12], "cache-hit": [], "eviction": [13]}[case]
        assert mm_sglang._CAPTURE.get() is None
    finally:
        release_media_entries(store, entries)
