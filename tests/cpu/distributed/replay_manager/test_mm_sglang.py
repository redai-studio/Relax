# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""MM capture contracts without importing SGLang or initializing CUDA."""

import json
from types import SimpleNamespace

import httpx
import numpy as np
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
    class Backend:
        blocks = {}

        def write_many(self, keys, values):
            self.blocks.update((key, value.copy()) for key, value in zip(keys, values, strict=True))

        def read(self, keys):
            return [self.blocks[key] for key in keys]

        read_encoded = read

        def delete(self, keys):
            for key in keys:
                self.blocks.pop(key, None)

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


@pytest.mark.parametrize("release_failure", [None, "before-ack", "after-ack"])
def test_mm_abort_drains_response_holds_before_cache_flush(capture, release_failure):
    runtime, _, inventory = capture
    release = runtime.release
    attempts = 0

    def release_with_transient_error(store, entries):
        nonlocal attempts
        attempts += 1
        if attempts == 1 and release_failure:
            if release_failure == "after-ack":
                release(store, entries)
            raise ConnectionError("injected lease release connection failure")
        release(store, entries)

    runtime.release = release_with_transient_error
    rid, image = REQUEST_PREFIX + "aborted", item()
    runtime.bind(rid, SimpleNamespace(mm_items=[image]))
    runtime.capture(image, torch.zeros(4, 2560, dtype=torch.bfloat16))
    runtime.abort(rid)
    # Observe the first asynchronous release attempt before flushing. A flush
    # must retry unacknowledged leases even though the request is already gone.
    runtime._worker.submit(lambda: None).result(timeout=10)
    runtime.flush()
    runtime._worker.submit(lambda: None).result(timeout=10)
    assert not runtime.requests and not inventory.state["objects"]
    assert inventory._used_bytes == 0


def test_mm_flush_retains_failed_cache_lease_for_next_flush(capture):
    runtime, store, inventory = capture
    rid, image = REQUEST_PREFIX + "cache-release-failure", item()
    runtime.bind(rid, SimpleNamespace(mm_items=[image]))
    runtime.capture(image, torch.zeros(4, 2560, dtype=torch.bfloat16))
    release_media_entries(store, finish(runtime, rid))
    release = runtime.release

    def unavailable(*_args):
        raise ConnectionError("registry unavailable during cache release")

    runtime.release = unavailable
    try:
        with pytest.raises(ConnectionError, match="registry unavailable"):
            runtime.flush()
    finally:
        runtime.release = release
    assert not runtime.requests and not runtime.cache
    assert inventory._used_bytes == 20480
    runtime.flush()
    assert not inventory.state["objects"]
    assert inventory._used_bytes == 0


def test_mm_close_retries_aborted_release_without_submitting_after_shutdown(capture):
    _, store, inventory = capture
    runtime = MMCapture(store)
    release = runtime.release
    attempts = 0

    def transient_failure(store, entries):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise ConnectionError("first abort release failed")
        release(store, entries)

    runtime.release = transient_failure
    try:
        rid = REQUEST_PREFIX + "close-after-release-failure"
        runtime.bind(rid, SimpleNamespace(mm_items=[item()]))
        runtime.capture(item(), torch.zeros(4, 2560, dtype=torch.bfloat16))
        runtime.abort(rid)
        runtime._worker.submit(lambda: None).result(timeout=10)
        runtime.close()
        assert not inventory.state["objects"]
        assert inventory._used_bytes == 0
    finally:
        runtime.release = release
        runtime.close()


@pytest.mark.parametrize("delivery", ["success", "response-lost", "late-publication"])
async def test_builtin_mm_generation_transfers_only_delivered_response_leases(capture, monkeypatch, delivery):
    from relax.engine.rollout import sglang_rollout
    from relax.utils import http_utils

    runtime, store, inventory = capture
    manager = ReplayManager(config=store.config, store_factory=lambda: store)
    monkeypatch.setattr(sglang_rollout, "replay_data_manager", manager)
    payload = manager.register_producer({"mm_embedding"}).request_options()
    rid, image = payload["rid"], item()
    expected = torch.ones(4, 2560, dtype=torch.bfloat16)
    calls = []

    def publish():
        runtime.bind(rid, SimpleNamespace(mm_items=[image]))
        runtime.capture(image, expected)
        return finish(runtime, rid)

    def worker(request):
        calls.append(request.url.path)
        if delivery == "late-publication":
            raise httpx.ReadError("response lost before capture completes")
        entries = publish()
        assert len(entries) == 1
        if delivery == "response-lost":
            raise httpx.ReadError("response lost after capture completes")
        return httpx.Response(200, json={"meta_info": {META_FIELD: entries}})

    args = SimpleNamespace(fully_async=True)
    async with httpx.AsyncClient(transport=httpx.MockTransport(worker)) as client:
        monkeypatch.setattr(http_utils, "_http_client", client)
        monkeypatch.setattr(http_utils, "_distributed_post_enabled", False)
        result = sglang_rollout._post_generation(args, None, "http://worker/generate", payload, None, False)
        if delivery == "success":
            entries = (await result)["meta_info"][META_FIELD]
            runtime.flush()
            torch.testing.assert_close(read_mm_embedding(store, entries[0]), expected)
            release_media_entries(store, entries)
        else:
            with pytest.raises(httpx.ReadError):
                await result
            if delivery == "late-publication":
                assert publish() == []
            runtime.flush()
    assert calls == ["/generate"]
    assert not inventory.state["objects"]


@pytest.mark.parametrize("r3", [False, True], ids=["mm-only", "r3-mm"])
@pytest.mark.parametrize("delivery", ["success", "response-lost", "late-publication"])
async def test_agentic_mm_response_lease_follows_wire_request(capture, monkeypatch, r3, delivery):
    from relax.agentic.pipeline import runtime as agentic
    from relax.distributed.replay_manager.backend import sglang
    from tests.cpu.test_agentic_rollout import _backend_adapter

    media, store, inventory = capture
    manager = ReplayManager(config=store.config, store_factory=lambda: store)
    monkeypatch.setattr(agentic, "replay_data_manager", manager)
    monkeypatch.setattr(sglang, "replay_data_manager", manager)
    monkeypatch.setenv("RELAX_REPLAY_REFERENCES", "1")
    adapter = _backend_adapter(lifecycle_enabled=False)
    adapter._args.use_rollout_mm_embedding_replay = True
    adapter._args.use_rollout_routing_replay = r3
    expected = torch.ones(4, 2560, dtype=torch.bfloat16)
    calls = []

    def publish(payload):
        assert payload["image_data"] == ["image"]
        rid = payload["rid"]
        image = item()
        media.bind(rid, SimpleNamespace(mm_items=[image]))
        media.capture(image, expected)
        meta = {META_FIELD: finish(media, rid), "finish_reason": {"type": "stop"}}
        if payload.get("return_routed_experts"):
            req = SimpleNamespace(origin_input_ids=payload["input_ids"], output_ids=[2], req_pool_idx=0, **payload)
            processor = SimpleNamespace(req_to_token_pool=SimpleNamespace(req_to_token=torch.tensor([[0]])))
            capturer = SimpleNamespace(host_cache=SimpleNamespace(buffer=torch.tensor([[[3]]], dtype=torch.int32)))
            route = sglang._advance_publication(processor, req, 1, {"moe": lambda: capturer}, final=True)["moe"]
            meta["routed_experts"] = sglang._wire_ref(route)
        return meta

    def worker(request):
        payload = json.loads(request.content)
        calls.append((request.url.path, payload))
        if request.url.path == "/abort_request":
            return httpx.Response(200, json={})
        if delivery == "late-publication":
            raise httpx.ReadError("response lost before capture completes")
        meta = publish(payload)
        if delivery == "response-lost":
            raise httpx.ReadError("response lost after capture completes")
        return httpx.Response(200, json={"output_ids": [2], "meta_info": meta})

    async with httpx.AsyncClient(transport=httpx.MockTransport(worker)) as client:
        adapter._managed_http = client
        result = adapter.generate(
            input_ids=[1],
            image_data=["image"],
            sampling_params={},
            session_id="s",
            request_id="r",
            engine_url="http://worker",
            lora_path="v",
        )
        if delivery == "success":
            meta = (await result).meta_info
            (entry,) = meta[META_FIELD]
            media.flush()
            torch.testing.assert_close(read_mm_embedding(store, entry), expected)
            release_media_entries(store, [entry])
            if r3:
                producer = manager.register_producer({"moe", "mm_embedding"})
                (route,) = producer.response_refs(meta)
                await producer.adopt_response([route], "session")
                store.discard([route])
        else:
            with pytest.raises(httpx.ReadError):
                await result
            await adapter.abort_request("r")
            (generate_url, payload), (abort_url, abort) = calls
            assert (generate_url, abort_url) == ("/generate", "/abort_request")
            assert abort["rid"] == payload["rid"]
            if delivery == "late-publication":
                assert publish(payload)[META_FIELD] == []
            media.flush()
    inventory.retire_partition("train_0")
    assert not inventory.snapshot()["objects"] and not store._backend.blocks


def test_mm_request_marker_is_absent_in_evaluation():
    manager = ReplayManager(config=ReplayConfig(delivery="reference", run_id="a"))
    producer = SGLangProducer(manager, frozenset({"mm_embedding"}))
    assert producer.request_options()["rid"].startswith(REQUEST_PREFIX)
    assert producer.request_options(evaluation=True) == {}


@pytest.mark.parametrize("use_reference_policy", [False, True])
def test_mm_ingress_keeps_pixels_only_for_reference_policy(capture, use_reference_policy):
    runtime, store, _ = capture
    rid, image = REQUEST_PREFIX + "ingress", item()
    runtime.bind(rid, SimpleNamespace(mm_items=[image]))
    runtime.capture(image, torch.ones(4, 2560, dtype=torch.bfloat16))
    entries = finish(runtime, rid)
    sample = SimpleNamespace(
        tokens=[1, 2],
        multimodal_train_inputs={"pixel_values": torch.ones(16, 3), "image_grid_thw": torch.tensor([[1, 4, 4]])},
    )
    manager = ReplayManager(config=store.config)
    producer = SGLangProducer(manager, frozenset({"mm_embedding"}))
    producer.accept_response(sample, {META_FIELD: entries}, SimpleNamespace(use_kl_loss=use_reference_policy))
    assert sample.rollout_replay_refs["mm_embedding"] == {"version": 1, "media": entries}
    assert ("pixel_values" in sample.multimodal_train_inputs) == use_reference_policy
    torch.testing.assert_close(sample.multimodal_train_inputs["image_grid_thw"], torch.tensor([[1, 4, 4]]))
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


@pytest.mark.parametrize("first", ["media-only", "media", "moe"])
@pytest.mark.parametrize("request_prefix", [REQUEST_PREFIX, "relax-agentic-r3-mm-"])
def test_mm_sglang_hooks_wait_for_all_references_and_emit_once(monkeypatch, capture, request, first, request_prefix):
    import sys
    from concurrent.futures import Future
    from types import ModuleType

    from relax.distributed.replay_manager.backend import mm_sglang, sglang
    from relax.distributed.replay_manager.ref_data import ReplayRef

    modules = {}
    for name in (
        "sglang",
        "sglang.srt",
        "sglang.srt.managers",
        "sglang.srt.runtime_context",
        "sglang.srt.state_capturer",
        "sglang.srt.state_capturer.routed_experts",
        "sglang.srt.managers.io_struct",
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
    capturer_module = modules["sglang.srt.state_capturer.routed_experts"]
    capturer_module.RoutedExpertsCapturer = type("Capturer", (), {"on_forward_end": lambda self: None})
    schedule = modules["sglang.srt.managers.mm_schedule"]
    for name in (
        "_batch_encode_per_image_misses",
        "_get_precomputed_embedding",
        "_get_chunked_embedding_full",
        "_get_chunked_embedding_by_item",
        "get_embedding_chunk",
    ):
        setattr(schedule, name, lambda *args, **kwargs: None)
    modules["sglang.srt.managers.mm_utils"].general_mm_embed_routine = lambda input_ids, forward_batch: (
        mm_sglang._CAPTURE.get()
    )

    class Req:
        def set_finish_with_abort(self):
            return None

        def reset_for_retract(self):
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
        max_running_requests = 2

        def _maybe_collect_routed_experts(self, req):
            raise AssertionError("Native capture ran for an explicit publication")

        def process_batch_result_prefill(self, batch, result):
            return None

        process_batch_result_decode = process_batch_result_prefill

        def run_batch(self, batch):
            # Breakable prefill graphs retain media but omit request IDs.
            forward_batch = SimpleNamespace(rids=None, mm_inputs=[req.multimodal_inputs for req in batch.reqs])
            return modules["sglang.srt.managers.mm_utils"].general_mm_embed_routine(None, forward_batch)

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
    modules["sglang.srt.managers.io_struct"].AbortReq = SimpleNamespace
    scheduler_module = modules["sglang.srt.managers.scheduler"]
    scheduler_module.Scheduler = Scheduler
    scheduler_module.get_global_experts_capturer = lambda: None
    monkeypatch.setattr(sglang, "replay_data_manager", sglang.replay_data_manager)
    monkeypatch.setattr(sglang, "_scheduler_contract", lambda: (scheduler_module.__name__, "Scheduler", "_"))
    manager = SimpleNamespace(uses_references=True, store=not_initialized)
    monkeypatch.setenv("RELAX_REPLAY_ENGINE_PUBLISHER", '[0, "engine"]')
    kinds = {"mm_embedding"} if first == "media-only" else {"mm_embedding", "moe"}
    sglang.apply_sglang_patch(manager, kinds=frozenset(kinds))
    scheduler = Scheduler()
    # SGLang chooses and constructs its streamer before parallel startup.
    scheduler.output_streamer = scheduler.get_output_streamer_class()()
    assert scheduler.flush_cache() is True

    runtime, store, _ = capture
    monkeypatch.setattr(mm_sglang, "model_hidden_size", lambda: 2560)
    monkeypatch.setattr(mm_sglang, "MMCapture", lambda *_, **_kwargs: runtime)
    manager.store = lambda: store
    parallel_ready = True
    ref = ReplayRef(2, "a", "a", "0" * 32, "moe", 4, 3, 2, 2, "0" * 64)
    emit = scheduler.output_streamer.stream_output

    def pending_request(suffix):
        req = SimpleNamespace(
            rid=request_prefix + suffix,
            multimodal_inputs=SimpleNamespace(mm_items=[item()]),
            finished_output=False,
            finished=lambda: True,
        )
        assert scheduler.run_batch(SimpleNamespace(reqs=[req])) is runtime
        assert mm_sglang._CAPTURE.get() is None
        evaluation = SimpleNamespace(rid="evaluation", multimodal_inputs=req.multimodal_inputs)
        assert scheduler.run_batch(SimpleNamespace(reqs=[evaluation])) is None
        assert mm_sglang._CAPTURE.get() is None
        media = Future()
        request.addfinalizer(media.cancel)
        runtime.requests[req.rid][0].future = media
        moe = None
        if first != "media-only":
            moe = Future()
            req._relax_replay_finish = {"moe": moe}
        emit([req], False)
        return req, media, moe

    req, media, moe = pending_request("overlap-completion")
    entry = {"completed": req.rid}
    if first == "media":
        media.set_result(entry)
    elif first == "moe":
        moe.set_result(ref)
    emit([req], False)
    scheduler.process_input_requests()
    assert not scheduler.output_streamer.emitted
    assert scheduler.flush_cache() is False
    assert scheduler.native_flushes == 1

    if not media.done():
        media.set_result(entry)
    if moe is not None and not moe.done():
        moe.set_result(ref)
    scheduler.process_input_requests()
    emit([req], False)
    assert scheduler.output_streamer.emitted == [{META_FIELD: [[entry]]}]
    if moe is not None:
        assert req.routed_experts == ref.to_dict()
    assert not runtime.requests

    cancelled, media, moe = pending_request("cancelled")
    releases, aborted = [], []
    scheduler.waiting_queue = []
    scheduler.ipc_channels = SimpleNamespace(
        send_to_tokenizer=SimpleNamespace(send_output=lambda message, req: aborted.append((message.rid, req)))
    )
    if moe is not None:
        cancelled._relax_replay_streams = {"moe": SimpleNamespace(abort=lambda: releases.append("moe"))}
    scheduler.running_batch = SimpleNamespace(reqs=[cancelled])
    scheduler.abort_request(SimpleNamespace(rid=cancelled.rid, abort_all=False))
    scheduler.abort_request(SimpleNamespace(rid="", abort_all=True))
    # Native abort ignores completed requests. Their tokens may already be in
    # the tokenizer, so replacing this terminal response with a bare AbortReq
    # would expose those tokens without their replay manifests.
    assert not media.cancelled() and cancelled.rid in runtime.requests
    assert not aborted and not releases
    assert scheduler.flush_cache() is False
    completed_entry = {"completed": cancelled.rid}
    media.set_result(completed_entry)
    if moe is not None:
        moe.set_result(ref)
    scheduler.process_input_requests()
    emit([cancelled], False)
    assert scheduler.output_streamer.emitted == [{META_FIELD: [[entry]]}, {META_FIELD: [[completed_entry]]}]
    if moe is not None:
        assert cancelled.routed_experts == ref.to_dict()
    assert not runtime.requests and scheduler.flush_cache() is True

    if first in {"moe", "media"}:
        failed, media_future, moe_future = pending_request("failed")
        publication = store.begin_publication(kind="moe", layers=1, topk=1)
        publication.append(1, lambda _: np.zeros((1, 1), dtype=np.int32))
        route = publication.finish([10, 11])
        failed._relax_replay_streams = {"moe": publication}
        if first == "moe":
            moe_future.set_result(route)
            media_future.set_exception(RuntimeError("media failed"))
        else:
            media_entry = runtime.publisher.publish("a" * 64, np.zeros((4, 5120), dtype=np.uint8), [1, 4, 4])
            media_future.set_result(media_entry)
            moe_future.set_exception(RuntimeError("route failed"))
        with pytest.raises(RuntimeError, match="failed"):
            scheduler.process_input_requests()
        runtime._worker.submit(lambda: None).result(timeout=5)
        assert not runtime.requests and not capture[2].state["objects"]
        assert scheduler.output_streamer.emitted == [{META_FIELD: [[entry]]}, {META_FIELD: [[completed_entry]]}]


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


@pytest.mark.parametrize(
    "kinds,disposition",
    [
        ({"mm_embedding"}, "stop"),
        ({"mm_embedding", "moe"}, "stop"),
        ({"mm_embedding"}, "error"),
        ({"mm_embedding"}, "waiting"),
        ({"mm_embedding"}, "native"),
    ],
)
def test_scheduler_pause_preserves_mm_until_response_but_hard_abort_releases(capture, monkeypatch, kinds, disposition):
    import sys
    from types import ModuleType

    from relax.agentic.session.service import _last_token_is_stop_token
    from relax.distributed.replay_manager.backend import sglang
    from relax.distributed.replay_manager.producer import prepare_replay_export

    class FINISH_ABORT:
        def __init__(self, status_code):
            self.status_code = status_code

    class Scheduler:
        max_running_requests = 1

        def abort_request(self, notice):
            if self.waiting_queue:
                self.waiting_queue.clear()
            else:
                request.to_finish = FINISH_ABORT(503 if disposition == "error" else None)

        def stream_output(self, reqs, return_logprob, skip_req=None):
            assert reqs == [request]
            self.entries = getattr(request, "_relax_mm_output", [])
            self.tokens = list(request.output_ids)

        def process_input_requests(self):
            pass

        def maybe_sleep_on_idle(self):
            pass

    module = ModuleType("sglang.srt.managers.scheduler")
    module.Scheduler = Scheduler
    monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setenv("RELAX_REPLAY_REFERENCES", "1")
    sglang._install_abort_patch()
    sglang._install_response_patch(lambda _: True)
    runtime, store, inventory = capture
    producer = ReplayManager(config=store.config, store_factory=lambda: store).register_producer(kinds)
    payload = producer.request_options(request_id=None if disposition == "native" else "pause")
    rid, image = payload["rid"], item()
    inventory.open_response_attempt(rid)
    runtime.bind(rid, SimpleNamespace(mm_items=[image]))
    expected = torch.arange(4 * 2560, dtype=torch.float32).reshape(4, 2560).bfloat16()
    runtime.capture(image, expected)
    request = SimpleNamespace(
        **payload, _relax_replay_media=runtime, to_finish=None, finished_reason=None, output_ids=[]
    )
    request.finished = lambda: request.finished_reason is not None
    scheduler = Scheduler()
    waiting = disposition == "waiting"
    scheduler.waiting_queue = [request] if waiting else []
    scheduler.running_batch = SimpleNamespace(reqs=[] if waiting else [request])
    scheduler.abort_request(SimpleNamespace(rid=rid, abort_all=False))
    if disposition == "stop":
        # SGLang emits the pending abort even when its last decoded token is EOS.
        request.output_ids.append(2)
        request.finished_reason, request.to_finish = request.to_finish, None
        runtime._worker.submit(lambda: None).result(timeout=10)
        scheduler.stream_output([request], True)
        scheduler.process_input_requests()
        entries = scheduler.entries
        assert _last_token_is_stop_token(
            token_ids=scheduler.tokens, tokenizer=SimpleNamespace(eos_token_id=2), sampling_params={}
        )
        sample = SimpleNamespace(
            tokens=[1, *scheduler.tokens],
            multimodal_train_inputs={"image_grid_thw": image.image_grid_thw},
            rollout_replay_refs={"mm_embedding": producer.response_media({META_FIELD: entries}, required=True)},
        )
        prepare_replay_export(sample, SimpleNamespace())
        inventory.finish_response_attempt(rid, [entry["lease"] for entry in entries])
        runtime.flush()
        runtime._worker.submit(lambda: None).result(timeout=10)
        # The emitted response owns a lease independently of the flushed cache.
        torch.testing.assert_close(read_mm_embedding(store, entries[0]).view(torch.uint8), expected.view(torch.uint8))
        release_media_entries(store, entries)
        resumed = rid + "-resumed"
        inventory.open_response_attempt(resumed)
        runtime.bind(resumed, SimpleNamespace(mm_items=[image]))
        runtime.capture(image, expected)
        resumed_entries = finish(runtime, resumed)
        inventory.finish_response_attempt(resumed, [entry["lease"] for entry in resumed_entries])
        torch.testing.assert_close(read_mm_embedding(store, resumed_entries[0]), expected)
        release_media_entries(store, resumed_entries)
    else:
        inventory.finish_response_attempt(rid)
        assert request._relax_replay_media is None
        assert rid not in runtime.requests
    runtime.flush()
    runtime._worker.submit(lambda: None).result(timeout=10)
    assert not inventory.state["objects"] and not store._backend.blocks
