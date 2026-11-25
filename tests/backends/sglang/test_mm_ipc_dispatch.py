# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import asyncio
import gc
import multiprocessing
import pickle
import time
from array import array
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
import zmq
import zmq.asyncio


torch = pytest.importorskip("torch")


def _consume(connection):
    from sglang.srt.managers.io_struct import _USE_PICKLE_IPC, msgpack_decode
    from sglang.srt.multimodal.transport.cuda_ipc import _pool_handle_cache_clear

    with connection:
        while True:
            frame = connection.recv()
            if frame is None:
                return
            obj = pickle.loads(frame) if _USE_PICKLE_IPC else msgpack_decode(frame)
            proxy = obj.mm_inputs.mm_items[0].feature
            result = proxy.reconstruct_on_target_device(0).cpu().tolist()
            del obj, proxy
            _pool_handle_cache_clear()
            connection.send(result)


@pytest.fixture
def tokenizer_pool(monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("requires a CUDA IPC memory pool")
    from sglang.srt.managers import tokenizer_manager as tokenizer
    from sglang.srt.multimodal.transport.cuda_ipc import MmItemMemoryPool

    pool = MmItemMemoryPool(1 << 20, 0.01, 0, 1, max_inflight_slices=4)
    pool.track_tokenizer_dispatch()
    manager = tokenizer.TokenizerManager.__new__(tokenizer.TokenizerManager)
    manager.mm_processor = SimpleNamespace(cudaipc_mmfeature_pool=pool)
    manager.cuda_vmm_feature_transport = SimpleNamespace(
        prepare_for_dispatch_async=AsyncMock(return_value=[]), cancel_for_dispatch=Mock()
    )
    manager.rid_to_state = {}
    manager.encoder_dispatch_ready = {}
    manager.tokenizer_ipc_name = None
    # Only the tokenizer-to-scheduler boundary is substituted. Send orchestration,
    # pool allocation, rollback and stream-ordered consumer ACKs are real.
    monkeypatch.setattr(tokenizer, "wrap_shm_features", lambda obj: obj)
    try:
        yield manager, pool
    finally:
        pool.shutdown()


def _request(pool, number):
    from sglang.srt.managers.io_struct import TokenizedGenerateReqInput
    from sglang.srt.managers.schedule_batch import Modality, MultimodalDataItem, MultimodalProcessorOutput
    from sglang.srt.observability.req_time_stats import APIServerReqTimeStats
    from sglang.srt.sampling.sampling_params import SamplingParams

    value = torch.full((4,), number, device="cuda")
    proxy = pool.wrap_tensor(value, use_pool_handle_cache=True)
    assert proxy is not None
    item = MultimodalDataItem(modality=Modality.IMAGE, feature=proxy, precomputed_embeddings=proxy)
    return TokenizedGenerateReqInput(
        rid=str(number),
        input_text=None,
        input_ids=array("i", [1]),
        input_embeds=None,
        mm_inputs=MultimodalProcessorOutput(mm_items=[item]),
        token_type_ids=None,
        sampling_params=SamplingParams(),
        return_logprob=False,
        logprob_start_len=-1,
        top_logprobs_num=0,
        token_ids_logprob=None,
        stream=False,
        time_stats=APIServerReqTimeStats(),
    )


def _wait_leases(pool, expected):
    deadline = time.monotonic() + 5
    while pool.active_lease_count != expected and time.monotonic() < deadline:
        time.sleep(0.01)
    assert pool.active_lease_count == expected


@pytest.mark.parametrize("batch", [False, True])
@pytest.mark.parametrize("failure", ["cancel", "not_queued"])
def test_undispatched_slices_are_reusable(tokenizer_pool, batch, failure):
    manager, pool = tokenizer_pool
    for attempt in range(5):
        requests = [_request(pool, i) for i in range(2 if batch else 1)]

        async def exercise():
            context = zmq.asyncio.Context()
            with context.socket(zmq.PUSH) as push:
                if failure == "cancel":
                    manager._raise_if_request_aborted = Mock(side_effect=asyncio.CancelledError())
                    expected_error = asyncio.CancelledError
                else:
                    # An async PUSH with no peer returns a Future failed with EAGAIN.
                    manager.send_to_scheduler = SimpleNamespace(send=lambda frame: push.send(frame, zmq.DONTWAIT))
                    expected_error = zmq.Again
                with pytest.raises(expected_error):
                    await (manager._send_batch_request(requests) if batch else manager._send_one_request(requests[0]))
            context.term()

        asyncio.run(exercise())
        # Hold the request objects alive. The failure path must explicitly return the
        # leases, without waiting for their owners to be garbage-collected.
        _wait_leases(pool, 0)
        proxies = [request.mm_inputs.mm_items[0].feature for request in requests]
        pool.finish_tokenizer_dispatch(proxies, transferred=False)
        assert manager.cuda_vmm_feature_transport.cancel_for_dispatch.call_count == attempt + 1


@pytest.mark.parametrize("batch", [False, True])
@pytest.mark.parametrize("pickle_wire", [False, True])
def test_wire_encoding_failure_reclaims_before_send(tokenizer_pool, monkeypatch, batch, pickle_wire):
    from sglang.srt.managers import io_struct

    manager, pool = tokenizer_pool
    monkeypatch.setattr(io_struct, "_USE_PICKLE_IPC", pickle_wire)
    manager.send_to_scheduler = Mock()
    requests = [_request(pool, i) for i in range(2 if batch else 1)]
    # Neither msgspec nor the pickle protocol can encode a local lambda.
    requests[-1].input_text = lambda: None
    with pytest.raises((AttributeError, TypeError, pickle.PicklingError)):
        asyncio.run(manager._send_batch_request(requests) if batch else manager._send_one_request(requests[0]))
    manager.send_to_scheduler.send.assert_not_called()
    _wait_leases(pool, 0)


def test_real_async_socket_cancels_before_queueing(tokenizer_pool):
    manager, pool = tokenizer_pool

    async def exercise():
        ctx = zmq.asyncio.Context()
        with ctx.socket(zmq.PUSH) as push:
            push.bind("inproc://blocked")
            manager.send_to_scheduler = push
            task = asyncio.create_task(manager._send_one_request(_request(pool, 8)))
            # A real PUSH without a peer has to wait for a writable socket.
            with pytest.raises(asyncio.TimeoutError):
                await asyncio.wait_for(asyncio.shield(task), 0.05)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            with ctx.socket(zmq.PULL) as pull:
                pull.connect("inproc://blocked")
                assert not await pull.poll(timeout=50)
        ctx.term()

    asyncio.run(exercise())
    _wait_leases(pool, 0)


def test_failed_handler_only_reclaims_its_own_request_generation(tokenizer_pool):
    from sglang.srt.managers.tokenizer_manager import ReqState

    manager, pool = tokenizer_pool
    request = _request(pool, 1)
    state = ReqState([], False, asyncio.Event(), request, Mock())
    _, state.pending_mm_ipc = manager._pending_mm_ipc((request.mm_inputs,))
    manager.rid_to_state[request.rid] = state
    manager._lora_draining_requests = {}
    manager.child_rid_to_logical_rid = {}

    manager._release_req_states_on_failure([request.rid], {object()})
    assert pool.active_lease_count == 1
    manager._release_req_states_on_failure([request.rid], {state.lifecycle_id})
    manager._release_req_states_on_failure([request.rid], {state.lifecycle_id})
    _wait_leases(pool, 0)


def test_abandoned_preprocessing_output(tokenizer_pool):
    _, pool = tokenizer_pool
    request = _request(pool, 1)
    assert pool.active_lease_count == 1
    del request
    gc.collect()
    _wait_leases(pool, 0)


def test_delivered_and_ambiguous_slices_wait_for_consumer(tokenizer_pool):
    from sglang.srt.managers.tokenizer_manager import ReqState

    manager, pool = tokenizer_pool
    ctx = multiprocessing.get_context("spawn")
    parent, child = ctx.Pipe()
    process = ctx.Process(target=_consume, args=(child,))
    process.start()
    child.close()
    try:
        # The first item in a sequential batch succeeded. The next send can
        # fail after acceptance. Neither lease can be reclaimed by the producer.
        sent = []

        async def exercise():
            ctx = zmq.asyncio.Context()
            with ctx.socket(zmq.PUSH) as push, ctx.socket(zmq.PULL) as pull:
                push.bind("inproc://scheduler")
                pull.connect("inproc://scheduler")
                manager.send_to_scheduler = push
                first = _request(pool, 1)
                sent.append(first.mm_inputs.mm_items[0].feature)
                await manager._send_one_request(first)
                frames = [await asyncio.wait_for(pull.recv(), 2)]

                async def ambiguous(frame):
                    await push.send(frame)
                    raise RuntimeError("unknown delivery")

                manager.send_to_scheduler = SimpleNamespace(send=ambiguous)
                second = _request(pool, 2)
                sent.append(second.mm_inputs.mm_items[0].feature)
                with pytest.raises(RuntimeError, match="unknown delivery"):
                    await manager._send_one_request(second)
                frames.append(await asyncio.wait_for(pull.recv(), 2))

            # Cancel the waiter after a blocked, single-frame PyZMQ send has
            # completed but before the coroutine has observed the result.
            class CancelAfterAcceptance(zmq.asyncio.Socket):
                def send(self, frame):
                    task = asyncio.current_task()
                    future = super().send(frame)
                    assert not future.done()
                    future.add_done_callback(lambda completed: task.cancel())
                    asyncio.get_running_loop().call_soon(pull.connect, "inproc://late-cancel")
                    return future

            with ctx.socket(zmq.PUSH, socket_class=CancelAfterAcceptance) as push, ctx.socket(zmq.PULL) as pull:
                push.bind("inproc://late-cancel")
                manager.send_to_scheduler = push
                request = _request(pool, 3)
                state = ReqState([], False, asyncio.Event(), request, Mock())
                _, state.pending_mm_ipc = manager._pending_mm_ipc((request.mm_inputs,))
                manager.rid_to_state[request.rid] = state
                dispatch_ready = asyncio.Event()
                manager.encoder_dispatch_ready[request.rid] = dispatch_ready
                sent.append(request.mm_inputs.mm_items[0].feature)
                task = asyncio.create_task(manager._send_one_request(request))
                with pytest.raises(asyncio.CancelledError):
                    await asyncio.wait_for(task, 2)
                frames.append(await asyncio.wait_for(pull.recv(), 2))
                assert state.dispatched
                manager.abort_request = Mock()
                manager._release_req_states_on_failure([request.rid], {state.lifecycle_id})
                manager.abort_request.assert_called_once_with(request.rid)
                assert manager.rid_to_state[request.rid] is state
                assert dispatch_ready.is_set()
            ctx.term()
            return frames

        frames = asyncio.run(exercise())
        pool.finish_tokenizer_dispatch(sent, transferred=False)
        assert pool.active_lease_count == 3
        for number, frame in enumerate(frames, 1):
            parent.send(frame)
            assert parent.poll(60)
            assert parent.recv() == [number] * 4
        _wait_leases(pool, 0)
    finally:
        parent.send(None)
        parent.close()
        process.join(10)
        if process.is_alive():
            process.terminate()
            process.join(10)
    assert process.exitcode == 0
