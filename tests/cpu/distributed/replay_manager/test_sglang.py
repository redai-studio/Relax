# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import sys
from concurrent.futures import Future
from contextlib import contextmanager
from dataclasses import dataclass
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from relax.distributed.replay_manager.backend.sglang import (
    _install_owner_capture_patch,
    _install_response_patch,
    _install_tq_topology_cache,
)
from relax.distributed.replay_manager.ref_data import ReplayRef


def test_reference_non_owner_keeps_capture_but_skips_routing_snapshot(monkeypatch) -> None:
    collective_inputs = []

    class Capturer:
        def capture(self, routes: np.ndarray) -> None:
            # Stand in for the mandatory capture-side TP all-gather.
            collective_inputs.append(routes.copy())
            self.routes = routes.copy()

        def on_forward_end(self, *, no_copy_to_cpu: bool) -> dict:
            return {"topk": self.routes.copy(), "async": no_copy_to_cpu}

    module = ModuleType("sglang.srt.state_capturer.routed_experts")
    module.RoutedExpertsCapturer = Capturer
    monkeypatch.setitem(sys.modules, module.__name__, module)
    is_owner = False
    _install_owner_capture_patch(lambda: is_owner)
    capturer = Capturer()
    routes = np.array([[0, 255], [17, 8]], dtype=np.int32)
    capturer.capture(routes)
    assert capturer.on_forward_end(no_copy_to_cpu=True) is None
    is_owner = True
    capturer.capture(routes)
    result = capturer.on_forward_end(no_copy_to_cpu=True)
    np.testing.assert_array_equal(result["topk"], routes)
    assert result["async"] is True
    assert len(collective_inputs) == 2
    for captured in collective_inputs:
        np.testing.assert_array_equal(captured, routes)


def test_multimodal_r3_hashes_the_original_placeholder_tokens(monkeypatch):
    import torch

    from relax.distributed.replay_manager.backend import sglang
    from relax.distributed.replay_manager.config import ReplayConfig
    from relax.distributed.replay_manager.core import ReplayInventory, ReplayManager
    from relax.distributed.replay_manager.storage import TQReplayStorage
    from tests.cpu.distributed.replay_manager.test_reference import MemoryBody

    config = ReplayConfig(delivery="reference", run_id="a", chunk_rows=2)
    body = MemoryBody()
    store = TQReplayStorage(config, inventory=ReplayInventory(config, body.delete))
    store._backend = body
    manager = ReplayManager(config=config, store_factory=lambda: store)
    monkeypatch.setattr(sglang, "replay_data_manager", manager)
    prompt = [10, 11, 12]
    request = SimpleNamespace(
        rid="multimodal",
        origin_input_ids=[10, 300, 12],
        origin_input_ids_unpadded=prompt,
        output_ids=[13],
        req_pool_idx=0,
        return_routed_experts=True,
    )
    processor = SimpleNamespace(req_to_token_pool=SimpleNamespace(req_to_token=torch.arange(3).reshape(1, 3)))
    capturer = SimpleNamespace(host_cache=SimpleNamespace(buffer=torch.arange(6, dtype=torch.int32).reshape(3, 1, 2)))
    try:
        refs = sglang._advance_publication(processor, request, 3, {"moe": lambda: capturer}, final=True)
        sample = SimpleNamespace(tokens=[*prompt, 13])
        manager.register_producer({"moe"}).accept_response(sample, {"routed_experts": refs["moe"]}, SimpleNamespace())
        assert sample.rollout_replay_refs == refs
    finally:
        store.close_run()


@pytest.mark.parametrize(
    "delivery,close_fault",
    [
        ("success", None),
        ("connection-refused", None),
        ("response-lost", None),
        ("late-publication", None),
        ("late-publication", "before-dispatch"),
        ("late-publication", "lost-ack"),
        ("late-publication", "outage"),
    ],
)
async def test_builtin_r3_owns_routes_until_http_handoff(monkeypatch, delivery, close_fault):
    import httpx
    import torch

    from relax.distributed.replay_manager.backend import sglang
    from relax.distributed.replay_manager.config import ReplayConfig
    from relax.distributed.replay_manager.core import ReplayInventory, ReplayManager
    from relax.distributed.replay_manager.storage import TQReplayStorage
    from tests.cpu.distributed.replay_manager.test_reference import MemoryBody

    config = ReplayConfig(delivery="reference", run_id="a")
    body = MemoryBody()
    inventory = ReplayInventory(config, body.delete)
    store = TQReplayStorage(config, inventory=inventory)
    store._backend = body
    manager = ReplayManager(config=config, store_factory=lambda: store)
    monkeypatch.setattr(sglang, "replay_data_manager", manager)
    producer = manager.register_producer({"moe"})
    payload = producer.request_options()
    request = SimpleNamespace(origin_input_ids=[1], output_ids=[2], req_pool_idx=0, **payload)
    processor = SimpleNamespace(req_to_token_pool=SimpleNamespace(req_to_token=torch.tensor([[0]])))
    capturer = SimpleNamespace(host_cache=SimpleNamespace(buffer=torch.tensor([[[3]]], dtype=torch.int32)))

    def publish():
        return sglang._advance_publication(processor, request, 1, {"moe": lambda: capturer}, final=True)["moe"]

    if close_fault:
        invoke = store.invoke_async
        close_calls = 0

        async def faulty_close(method, *args):
            nonlocal close_calls
            if method == "finish_response_attempt" and args == (request.rid,):
                close_calls += 1
                if close_fault == "outage" or close_calls == 1:
                    if close_fault == "lost-ack":
                        await invoke(method, *args)
                    raise ConnectionError("lost registry connection")
            return await invoke(method, *args)

        monkeypatch.setattr(store, "invoke_async", faulty_close)

    attempts = []

    async def send(url, sent, **kwargs):
        attempts.append(sent["rid"])
        if delivery == "connection-refused" and len(attempts) == 1:
            raise httpx.ConnectError("connection refused")
        if delivery == "late-publication":
            raise httpx.ReadError("connection lost before sealing")
        route = publish()
        if delivery == "response-lost":
            raise httpx.ReadError("connection lost after sealing")
        return {"meta_info": {"routed_experts": route}}

    try:
        post = sglang.post_generation_request(
            manager, send, "/generate", payload, headers=None, fail_fast_no_workers=False
        )
        if delivery in {"success", "connection-refused"}:
            meta = (await post)["meta_info"]
            refs = producer.response_refs(meta)
            await producer.adopt_response(refs, "receiver")
            assert body.blocks
            store.discard(refs)
        else:
            with pytest.raises(httpx.ReadError):
                await post
            if delivery == "late-publication":
                publish()
                if close_fault == "outage":
                    # The late publisher is held until a new request reaches the recovered registry.
                    assert inventory.snapshot()["objects"] and body.blocks
                    monkeypatch.setattr(store, "invoke_async", invoke)

                    async def next_send(*args, **kwargs):
                        assert not inventory.snapshot()["objects"] and not body.blocks
                        return {"meta_info": {}}

                    await sglang.post_generation_request(
                        manager,
                        next_send,
                        "/generate",
                        producer.request_options(),
                        headers=None,
                        fail_fast_no_workers=False,
                    )
        assert attempts == [request.rid] * (2 if delivery == "connection-refused" else 1)
        assert not inventory.snapshot()["objects"] and not body.blocks
    finally:
        store.close_run()


@pytest.mark.parametrize("component_output", [False, True], ids=["legacy", "component"])
def test_reference_response_waits_for_publication_and_drains_on_empty_input(monkeypatch, component_output):
    # Match SGLang's independent, slotted output component. It intentionally
    # has no scheduler fields, and the scheduler has no stream_output method.
    @dataclass(slots=True)
    class OutputStreamer:
        emitted: list

        def stream_output(self, reqs, return_logprob, skip_req=None):
            self.emitted.extend(req for req in reqs if req is not skip_req)

    class Scheduler:
        max_running_requests = 4

        def get_output_streamer_class(self):
            return OutputStreamer

        def process_input_requests(self, reqs):
            return "processed"

        def maybe_sleep_on_idle(self):
            return "idle"

    if not component_output:
        Scheduler.stream_output = lambda self, reqs, logprob, skip=None: self.emitted.extend(
            req for req in reqs if req is not skip
        )
    module = ModuleType("sglang.srt.managers.scheduler")
    module.Scheduler = Scheduler
    monkeypatch.setitem(sys.modules, module.__name__, module)
    assert _install_response_patch(lambda scheduler: True)
    scheduler = Scheduler()
    scheduler.emitted = []
    if component_output:
        scheduler.output_streamer = scheduler.get_output_streamer_class()(scheduler.emitted)
        emit = scheduler.output_streamer.stream_output
    else:
        emit = scheduler.stream_output

    publication = Future()
    req = SimpleNamespace(_relax_replay_finish={"moe": publication}, _relax_replay_streams={"moe": object()})
    skipped = SimpleNamespace()
    emit([req, skipped], True, skipped)
    assert not scheduler.emitted
    ref = ReplayRef(2, "a", "a", "0" * 32, "moe", 4, 3, 2, 2, "0" * 64)
    publication.set_result(ref)
    assert scheduler.process_input_requests([]) == "processed"
    assert scheduler.emitted == [req]
    assert req.routed_experts == ref.to_dict()
    assert req._relax_replay_finish is None and req._relax_replay_streams == {}
    assert scheduler.maybe_sleep_on_idle() == "idle"
    assert not scheduler._relax_replay_pending


@pytest.mark.parametrize("abort_mode", ["running", "suffix", "waiting", "error"])
def test_agentic_reference_abort_seals_only_active_moe_routes(monkeypatch, abort_mode):
    import torch

    from relax.distributed.replay_manager.backend import sglang
    from relax.distributed.replay_manager.config import ReplayConfig
    from relax.distributed.replay_manager.core import ReplayInventory, ReplayManager
    from relax.distributed.replay_manager.ref_data import token_digest
    from relax.distributed.replay_manager.storage import TQReplayStorage
    from tests.cpu.distributed.replay_manager.test_reference import MemoryBody

    class FINISH_ABORT:
        def __init__(self, status_code=None):
            self.status_code = status_code

    class Req:
        rid = sglang.AGENTIC_R3_REQUEST_PREFIX + "test"
        origin_input_ids = [10, 11, 12]
        return_routed_experts = True
        is_retracted = False
        req_pool_idx = 0
        finished_reason = to_finish = None

        def __init__(self):
            self.output_ids = []

        def finished(self):
            return self.finished_reason is not None

        def reset_for_retract(self):
            self.is_retracted = True

        def set_finish_with_abort(self):
            self.finished_reason = FINISH_ABORT()

    class Scheduler:
        max_running_requests = 4
        tp_rank = 0

        def abort_request(self, notice):
            if self.waiting_queue:
                self.waiting_queue.pop()
            else:
                self.running_batch.reqs[0].to_finish = FINISH_ABORT(503 if abort_mode == "error" else None)

        def maybe_collect_routed_experts(self, req):
            raise AssertionError("The native collector cannot seal a reference")

        def process_batch_result_decode(self, batch, result):
            for req in batch.reqs:
                req.output_ids.append(13)
                if req.to_finish is not None:
                    req.finished_reason, req.to_finish = req.to_finish, None
                    self.maybe_collect_routed_experts(req)
                    self.stream_output([req], True)

        process_batch_result_prefill = process_batch_result_decode

        def stream_output(self, reqs, return_logprob, skip_req=None):
            self.emitted.extend(reqs)

        def process_input_requests(self, reqs):
            pass

        def maybe_sleep_on_idle(self):
            pass

    config = ReplayConfig(delivery="reference", run_id="a", chunk_rows=2, publication_bytes=4096)
    body = MemoryBody()
    inventory = ReplayInventory(config, body.delete)
    store = TQReplayStorage(config, inventory=inventory)
    store._backend = body
    manager = ReplayManager(config=config, store_factory=lambda: store)
    monkeypatch.setattr(sglang, "replay_data_manager", manager)
    monkeypatch.setattr(sglang, "_scheduler_contract", lambda: ("sglang.srt.managers.scheduler", "Scheduler", ""))
    module = ModuleType("sglang.srt.managers.scheduler")
    module.Scheduler = Scheduler
    values = torch.arange(8, dtype=torch.int32).reshape(4, 1, 2)
    module.get_global_experts_capturer = lambda: SimpleNamespace(host_cache=SimpleNamespace(buffer=values))
    monkeypatch.setitem(sys.modules, module.__name__, module)
    schedule_batch = ModuleType("sglang.srt.managers.schedule_batch")
    schedule_batch.Req = Req
    monkeypatch.setitem(sys.modules, schedule_batch.__name__, schedule_batch)
    req = Req()
    inventory.open_response_attempt(req.rid)
    req.routed_experts_start_len = 1 if abort_mode == "suffix" else 0
    scheduler = Scheduler()
    scheduler.emitted = []
    active = abort_mode != "waiting"
    scheduler.waiting_queue = [] if active else [req]
    scheduler.running_batch = SimpleNamespace(reqs=[req] if active else [], seq_lens_cpu=[3])
    scheduler.req_to_token_pool = SimpleNamespace(req_to_token=torch.arange(4).reshape(1, 4))

    try:
        sglang.install_scheduler_patch(manager, kinds=frozenset({"moe"}))
        # Leave one progressive chunk in each stream before the abort.
        sglang._advance_publication(
            scheduler,
            req,
            2,
            {"moe": module.get_global_experts_capturer},
            final=False,
        )
        scheduler.abort_request(SimpleNamespace(rid=req.rid, abort_all=False))
        if not active:
            assert not req._relax_replay_streams and not inventory.state["objects"]
            return

        scheduler.process_batch_result_decode(scheduler.running_batch, SimpleNamespace())
        if abort_mode == "error":
            assert scheduler.emitted == [req]
            assert not hasattr(req, "routed_experts") and not inventory.state["objects"]
            return
        if scheduler._relax_replay_pending:
            req._relax_replay_finish["moe"].result(timeout=5)
            scheduler.process_input_requests([])
        assert scheduler.emitted == [req]
        ref = ReplayRef.from_dict(req.routed_experts)
        inventory.finish_response_attempt(req.rid, [ref.record])
        start = req.routed_experts_start_len
        assert ref.rows == 3 - start and ref.tokens == token_digest([10, 11, 12, 13][start:])
        with store.reader([ref]) as reader:
            rows = np.arange(ref.rows)
            output = np.zeros((ref.rows, 2), dtype=np.int32)
            reader.read_into(0, 0, rows, output, rows)
            np.testing.assert_array_equal(output, values[start:3, 0].numpy())
        assert len(inventory.state["objects"]) == 1
        store.discard([ref])
        assert not body.blocks
    finally:
        store._writer.close()


def test_reference_tq_topology_cache_preserves_thread_restore_and_fallback(monkeypatch) -> None:
    import psutil
    import torch
    import transfer_queue
    import transfer_queue.storage
    import transfer_queue.utils

    queries, threads, fallback = [], [], []

    def physical_cores(*, logical: bool) -> int:
        queries.append(logical)
        return 16

    @contextmanager
    def original(target_num_threads=None, info=""):
        fallback.append((target_num_threads, info))
        yield

    client = ModuleType("transfer_queue.client")
    client.limit_pytorch_auto_parallel_threads = original
    common = ModuleType("transfer_queue.utils.common")
    common.DEFAULT_TORCH_NUM_THREADS = 24
    storage = ModuleType("transfer_queue.storage.simple_storage")
    storage.limit_pytorch_auto_parallel_threads = original
    monkeypatch.setitem(sys.modules, storage.__name__, storage)
    monkeypatch.setattr(transfer_queue.storage, "simple_storage", storage)
    monkeypatch.setitem(sys.modules, client.__name__, client)
    monkeypatch.setitem(sys.modules, common.__name__, common)
    monkeypatch.setattr(transfer_queue, "client", client)
    monkeypatch.setattr(transfer_queue.utils, "common", common)
    monkeypatch.setattr(psutil, "cpu_count", physical_cores)
    monkeypatch.setattr(torch, "set_num_threads", threads.append)
    _install_tq_topology_cache()
    _install_tq_topology_cache()
    for module in (client, storage):
        with module.limit_pytorch_auto_parallel_threads(8):
            pass
    with pytest.raises(RuntimeError, match="body failed"):
        with client.limit_pytorch_auto_parallel_threads(8):
            raise RuntimeError("body failed")
    for target in (None, 32):
        with client.limit_pytorch_auto_parallel_threads(target, info="fallback"):
            pass
    assert queries == [False]
    assert threads == [8, 24, 8, 24, 8, 24]
    assert fallback == [(None, "fallback"), (32, "fallback")]
