# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""MM capture contracts without importing SGLang or initializing CUDA."""

from types import SimpleNamespace

import pytest
import torch

from relax.distributed.replay_manager.backend.mm_sglang import (
    META_FIELD,
    REQUEST_PREFIX,
    MMCapture,
    accept_mm_response,
)
from relax.distributed.replay_manager.backend.sglang import SGLangProducer
from relax.distributed.replay_manager.config import ReplayConfig
from relax.distributed.replay_manager.core import ReplayInventory, ReplayManager
from relax.distributed.replay_manager.mm_embedding import read_mm_embedding, release_media_entries
from relax.distributed.replay_manager.storage import TQReplayStorage


@pytest.fixture
def capture():
    bodies = {}

    class Backend:
        def write_many(self, keys, values):
            bodies.update((key, value.copy()) for key, value in zip(keys, values, strict=True))

        def read(self, keys):
            return [bodies[key] for key in keys]

        read_encoded = read

        def delete(self, keys):
            for key in keys:
                bodies.pop(key, None)

    backend = Backend()
    config = ReplayConfig(delivery="reference", run_id="a", chunk_rows=2, publication_bytes=1 << 20)
    inventory = ReplayInventory(config, backend.delete)
    store = TQReplayStorage(config, inventory=inventory)
    store._backend = backend
    runtime = MMCapture(store)
    yield runtime, store, inventory
    runtime.flush()
    runtime._worker.shutdown(wait=True)
    if store._writer is not None:
        store._writer.close()


def item(media_hash=12, start=0):
    return SimpleNamespace(
        hash=media_hash,
        modality=SimpleNamespace(name="IMAGE"),
        image_grid_thw=torch.tensor([[1, 4, 4]]),
        offsets=[(start, start + 3)],
    )


def finish(runtime, rid):
    runtime.ready(rid)
    runtime._worker.submit(lambda: None).result(timeout=10)
    return runtime.finish(rid)


def test_mm_capture_hash_deduplicates_bits_and_preserves_independent_response_holds(capture):
    runtime, store, inventory = capture
    image = item()
    first, second = REQUEST_PREFIX + "first", REQUEST_PREFIX + "second"
    runtime.bind(first, SimpleNamespace(mm_items=[image]))
    runtime.bind(second, SimpleNamespace(mm_items=[image]))
    expected = torch.arange(4 * 2560, dtype=torch.float32).reshape(4, 2560).bfloat16()
    runtime.capture(image, expected)
    a, b = finish(runtime, first), finish(runtime, second)
    assert a[0]["ref"] == b[0]["ref"]
    assert a[0]["lease"] != b[0]["lease"]
    torch.testing.assert_close(read_mm_embedding(store, a[0]).view(torch.uint8), expected.view(torch.uint8))
    release_media_entries(store, a)
    assert len(inventory.state["objects"]) == 1
    release_media_entries(store, b)
    # Full prefix-cache hits must remain replayable after original response release.
    third = REQUEST_PREFIX + "cache-hit"
    runtime.bind(third, SimpleNamespace(mm_items=[image]))
    c = finish(runtime, third)
    assert c[0]["ref"] == a[0]["ref"]
    release_media_entries(store, c)
    runtime.flush()
    assert not inventory.state["objects"]


def test_mm_capture_keeps_prompt_order_and_waits_for_complete_chunk(capture):
    runtime, store, _ = capture
    first, second = item(1, 8), item(2, 0)
    rid = REQUEST_PREFIX + "packed"
    runtime.bind(rid, SimpleNamespace(mm_items=[first, second]))
    with pytest.raises(ValueError, match="complete BF16"):
        runtime.capture(first, torch.zeros(2, 2560, dtype=torch.bfloat16))
    runtime.capture(first, torch.ones(4, 2560, dtype=torch.bfloat16))
    runtime.capture(second, torch.zeros(4, 2560, dtype=torch.bfloat16))
    entries = finish(runtime, rid)
    assert torch.count_nonzero(read_mm_embedding(store, entries[0])) == 0
    assert torch.all(read_mm_embedding(store, entries[1]) == 1)
    release_media_entries(store, entries)


def test_mm_abort_drains_response_holds_before_cache_flush(capture):
    runtime, _, inventory = capture
    rid, image = REQUEST_PREFIX + "aborted", item()
    runtime.bind(rid, SimpleNamespace(mm_items=[image]))
    runtime.capture(image, torch.zeros(4, 2560, dtype=torch.bfloat16))
    runtime.abort(rid)
    runtime.flush()
    runtime._worker.submit(lambda: None).result(timeout=10)
    assert not runtime.requests and not inventory.state["objects"]


def test_mm_request_marker_is_absent_in_evaluation():
    manager = ReplayManager(config=ReplayConfig(delivery="reference", run_id="a"))
    producer = SGLangProducer(manager, frozenset({"mm_embedding"}))
    assert producer.request_options()["rid"].startswith(REQUEST_PREFIX)
    assert producer.request_options(evaluation=True) == {}


def test_mm_ingress_checks_grid_and_sends_only_metadata(capture):
    runtime, store, _ = capture
    rid, image = REQUEST_PREFIX + "ingress", item()
    runtime.bind(rid, SimpleNamespace(mm_items=[image]))
    runtime.capture(image, torch.ones(4, 2560, dtype=torch.bfloat16))
    entries = finish(runtime, rid)
    sample = SimpleNamespace(
        tokens=[1, 2],
        multimodal_train_inputs={"pixel_values": torch.ones(16, 3), "image_grid_thw": torch.tensor([[1, 4, 4]])},
    )
    result = accept_mm_response(sample, {META_FIELD: entries})
    assert result == {"version": 1, "media": entries}
    assert set(sample.multimodal_train_inputs) == {"image_grid_thw"}
    sample.multimodal_train_inputs["image_grid_thw"] = torch.tensor([[1, 8, 4]])
    with pytest.raises(ValueError, match="grids differ"):
        accept_mm_response(sample, {META_FIELD: entries})
    release_media_entries(store, entries)


def test_mm_capture_chunks_media_larger_than_publication_budget(capture):
    runtime, store, _ = capture
    # The byte budget limits a transfer chunk, not the complete image.
    store.config = ReplayConfig(delivery="reference", run_id="a", chunk_rows=2, publication_bytes=16384)
    store.publication_bytes = store.config.publication_bytes
    image = item()
    image.image_grid_thw = torch.tensor([[1, 8, 8]])
    image.offsets = [(0, 15)]
    rid = REQUEST_PREFIX + "large"
    runtime.bind(rid, SimpleNamespace(mm_items=[image]))
    expected = torch.ones(16, 2560, dtype=torch.bfloat16)
    runtime.capture(image, expected)
    entries = finish(runtime, rid)
    torch.testing.assert_close(read_mm_embedding(store, entries[0]), expected)
    release_media_entries(store, entries)


def test_mm_sglang_hooks_defer_runtime_and_emit_parked_completion_once(monkeypatch, capture, request):
    import sys
    from concurrent.futures import Future
    from types import ModuleType

    from relax.distributed.replay_manager.backend import mm_sglang
    from relax.distributed.replay_manager.backend.mm_sglang import install_capture_patch

    modules = {}
    for name in (
        "sglang",
        "sglang.srt",
        "sglang.srt.managers",
        "sglang.srt.runtime_context",
        "sglang.srt.managers.mm_schedule",
        "sglang.srt.managers.mm_utils",
        "sglang.srt.managers.schedule_batch",
        "sglang.srt.managers.scheduler",
    ):
        modules[name] = ModuleType(name)
        monkeypatch.setitem(sys.modules, name, modules[name])
        parent, _, child = name.rpartition(".")
        if parent:
            setattr(modules[parent], child, modules[name])

    parallel_ready = False

    def not_initialized():
        if parallel_ready:
            return SimpleNamespace(attn_tp_rank=0)
        raise AssertionError("Parallel runtime was read during hook installation")

    modules["sglang.srt.runtime_context"].get_parallel = not_initialized
    schedule = modules["sglang.srt.managers.mm_schedule"]
    for name in (
        "_batch_encode_per_image_misses",
        "_get_precomputed_embedding",
        "_get_chunked_embedding_full",
        "get_embedding_chunk",
    ):
        setattr(schedule, name, lambda *args, **kwargs: None)
    modules["sglang.srt.managers.mm_utils"].general_mm_embed_routine = lambda input_ids, forward_batch: None

    class Req:
        def set_finish_with_abort(self):
            return None

    class Streamer:
        def __init__(self):
            self.emitted = []

        def stream_output(self, reqs, return_logprob, skip_req=None):
            # Native SGLang admits a terminal request once, marks it emitted,
            # then calls the customized-info hook on the admitted requests.
            admitted = [req for req in reqs if req is not skip_req and not req.finished_output]
            for req in admitted:
                req.finished_output = True
            if admitted:
                self.emitted.append(self.build_additional_customized_info(admitted))

        def build_additional_customized_info(self, reqs):
            assert all(req.finished_output for req in reqs)
            return {}

    class Scheduler:
        def run_batch(self, batch):
            return None

        def get_output_streamer_class(self):
            return Streamer

        def process_input_requests(self):
            return None

        def maybe_sleep_on_idle(self):
            return None

        def abort_request(self, request):
            return None

        def flush_cache(self):
            self.native_flushes = getattr(self, "native_flushes", 0) + 1
            return True

    modules["sglang.srt.managers.schedule_batch"].Req = Req
    modules["sglang.srt.managers.scheduler"].Scheduler = Scheduler
    manager = SimpleNamespace(store=not_initialized)
    install_capture_patch(manager)
    # Streamer selection also occurs while Scheduler is being constructed.
    assert issubclass(Scheduler().get_output_streamer_class(), Streamer)
    scheduler = Scheduler()
    scheduler._relax_mm_pending = {"finished-mm": object()}
    assert scheduler.flush_cache() is False
    assert getattr(scheduler, "native_flushes", 0) == 0
    scheduler._relax_mm_pending.clear()
    scheduler._relax_replay_pending = {"finished-r3": object()}
    assert scheduler.flush_cache() is False
    assert getattr(scheduler, "native_flushes", 0) == 0
    scheduler._relax_replay_pending.clear()
    assert scheduler.flush_cache() is True
    assert scheduler.native_flushes == 1

    runtime, store, _ = capture
    monkeypatch.setattr(mm_sglang, "MMCapture", lambda _: runtime)
    manager.store = lambda: store
    parallel_ready = True
    scheduler.max_running_requests = 2
    scheduler.output_streamer = scheduler.get_output_streamer_class()()
    rid = REQUEST_PREFIX + "overlap-completion"
    runtime.bind(rid, SimpleNamespace(mm_items=[item()]))
    publication = Future()
    request.addfinalizer(publication.cancel)
    runtime.requests[rid][0].future = publication
    req = SimpleNamespace(rid=rid, finished_output=False, finished=lambda: True)

    scheduler.output_streamer.stream_output([req], False)
    assert rid in scheduler._relax_mm_pending
    assert not scheduler.output_streamer.emitted
    entry = {"completed": rid}
    publication.set_result(entry)
    # The overlap scheduler can output the same completed request before its
    # next process-input/drain call. This must retire the parked output hold.
    scheduler.output_streamer.stream_output([req], False)
    assert scheduler.output_streamer.emitted == [{META_FIELD: [[entry]]}]
    assert rid not in runtime.requests and not scheduler._relax_mm_pending
    scheduler.process_input_requests()
    scheduler.output_streamer.stream_output([req], False)
    # Drain must also tolerate an already-emitted entry in its snapshot, e.g.
    # when an inner output/drain invocation emitted another parked request.
    scheduler._relax_mm_pending[rid] = (req, False)
    scheduler.process_input_requests()
    assert not scheduler._relax_mm_pending
    assert scheduler.output_streamer.emitted == [{META_FIELD: [[entry]]}]


def test_mm_video_frame_spans_and_repeated_images_keep_media_occurrences(capture):
    runtime, store, _ = capture
    image = item()
    image.offsets = [(0, 3), (20, 23)]
    video = SimpleNamespace(
        hash=25,
        modality=SimpleNamespace(name="VIDEO"),
        video_grid_thw=torch.tensor([[2, 4, 4]]),
        offsets=[(8, 11), (14, 17)],
    )
    rid = REQUEST_PREFIX + "video"
    runtime.bind(rid, SimpleNamespace(mm_items=[video, image]))
    runtime.capture(image, torch.ones(4, 2560, dtype=torch.bfloat16))
    runtime.capture(video, torch.zeros(8, 2560, dtype=torch.bfloat16))
    entries = finish(runtime, rid)
    assert [entry["modality"] for entry in entries] == ["image", "video", "image"]
    assert entries[0]["ref"] == entries[2]["ref"]
    assert entries[0]["lease"] != entries[2]["lease"]
    assert [read_mm_embedding(store, entry).shape[0] for entry in entries] == [4, 8, 4]
    release_media_entries(store, entries)
