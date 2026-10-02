# Copyright (c) 2026 Relax Authors. All Rights Reserved.


"""Native ownership protocol tests, without claiming CUDA/server integration.

Import the complete patched module. Events/registry/queues are controlled here;
the actual scheduler hooks still require the pinned SGLang runtime and GPU
tests.
"""

import asyncio
import importlib.util
import os
import pickle
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.backends.sglang.conftest import request_state


@pytest.fixture
def native(native_abort_type):
    root = os.environ.get("RELAX_SGLANG_SOURCE")
    if not root:
        pytest.skip("requires patched native source: RELAX_SGLANG_SOURCE")
    path = Path(root) / "python/sglang/srt/lora/version_control.py"
    spec = importlib.util.spec_from_file_location("native_lora_execution", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class Event:
    def __init__(self):
        self.done = False
        self.stream = None

    def record(self, *, stream):
        self.stream = stream

    def query(self):
        return self.done


class Registry:
    def __init__(self):
        self.refs = 0
        self.releases = 0
        self.entered = asyncio.Event()
        self.allow_acquire = asyncio.Event()
        self.allow_acquire.set()
        self.released = asyncio.Event()

    async def acquire_exact(self, ref):
        self.entered.set()
        await self.allow_acquire.wait()
        self.refs += 1

    async def release(self, uid):
        assert self.refs == 1
        self.refs -= 1
        self.releases += 1
        self.released.set()


def identity(native, **changes):
    return replace(native.LoRAExecutionIdentity("cohort", "boot", "uid-A", "owner", "sid", "rid"), **changes)


def execution_control(native, registry, send, cohort, boot, limit, **topology):
    control = native.LoRAVersionControl(registry, send, cohort, boot, limit, **topology)
    control.versions["uid-A"] = native.TokenizerVersion(
        native.NativeLoadIdentity(boot, "uid-A"),
        SimpleNamespace(lora_id="uid-A"),
        "digest",
        state="READY",
        registered=True,
    )
    return control


def scheduler():
    replies, cancellations = [], []
    engine = SimpleNamespace(
        lora_execution_owner=("cohort", "boot"),
        lora_execution_limit=16,
        lora_execution_accepting=True,
        lora_versions={},
        lora_version_names={},
        lora_retirements={},
        lora_sessions={},
        lora_session_closures={},
        ps=SimpleNamespace(pp_rank=0, pp_size=1, tp_rank=0, tp_size=1, attn_tp_rank=0, attn_cp_rank=0, attn_dp_rank=0),
        ipc_channels=SimpleNamespace(send_to_tokenizer=SimpleNamespace(send_output=replies.append)),
        waiting_queue=[],
        running_batch=None,
        last_batch=None,
        result_queue=[],
        chunked_req=None,
        _pending_chunked_abort_req=None,
        device_module=SimpleNamespace(Event=Event),
        forward_stream=object(),
        tp_worker=SimpleNamespace(
            model_runner=SimpleNamespace(lora_manager=SimpleNamespace(_validate_managed_batch_ids=lambda ids: None))
        ),
        abort_request=lambda command, *, exact: cancellations.extend(
            req for req in engine.waiting_queue if req.rid == command.rid
        ),
        output_streamer=SimpleNamespace(on_terminal=None),
    )
    return engine, replies, cancellations


def admitted(native, engine, attempt):
    version = native.NativeVersion(
        native.NativeLoadIdentity(*attempt.instance_key),
        SimpleNamespace(lora_id=attempt.native_lora_id),
        "digest",
        state="READY",
    )
    engine.lora_versions.setdefault(attempt.native_lora_id, version)
    req = SimpleNamespace(
        rid=attempt.rid,
        lora_id=attempt.native_lora_id,
        lora_engine_boot_id=attempt.engine_boot_id,
        lora_request_kind=attempt.kind,
        lora_session_id=attempt.session_id,
        http_worker_ipc=None,
    )
    native.accept_execution(engine, req)
    return req


def reply(native, attempt, **changes):
    return native.LoRAExecutionReply(attempt.rid, attempt.engine_boot_id, "REQUEST_FINISHED", **changes)


@pytest.mark.parametrize(
    "queue",
    [
        "waiting_queue",
        "running_batch",
        "last_batch",
        "result_queue",
        "cur_batch_for_debug",
        "chunked_req",
        "_pending_chunked_abort_req",
        "mbs",
        "running_mbs",
        "last_mbs",
    ],
)
def test_native_completion_reuses_actual_request_owners(native, queue):
    engine, replies, _ = scheduler()
    req = admitted(native, engine, identity(native))
    batch = SimpleNamespace(reqs=[req])
    if queue == "waiting_queue":
        value, empty = [req], []
    elif queue == "result_queue":
        value, empty = [(batch, None)], []
    elif queue in {"mbs", "running_mbs", "last_mbs"}:
        value, empty = [None, batch], []
    elif queue in {"chunked_req", "_pending_chunked_abort_req"}:
        value, empty = req, None
    else:
        value, empty = batch, None
    setattr(engine, queue, value)
    assert req in native.native_requests(engine) and not replies
    setattr(engine, queue, empty)
    assert req not in native.native_requests(engine) and not replies
    native.finish_native_request(engine, req)
    native.finish_native_request(engine, req)
    assert len(replies) == 1 and replies[0].state == "REQUEST_FINISHED"


@pytest.mark.parametrize("rank", [1, 2, 3])
def test_nonleader_has_no_request_ledger_or_completion_ack(native, rank):
    engine, replies, _ = scheduler()
    engine.ps.tp_rank = rank
    engine.ps.attn_tp_rank = rank
    admitted(native, engine, identity(native))
    assert not hasattr(engine, "lora_requests")
    native.control_execution(engine, native.LoRAExecutionControl(identity(native), "cancel"))
    assert not replies


def test_logical_completion_does_not_discard_instance_gpu_dependency(native):
    engine, replies, _ = scheduler()
    req = admitted(native, engine, identity(native))
    versions = native.begin_execution_batch(engine, SimpleNamespace(reqs=[req]))
    native.finish_execution_batch(engine, versions)
    version = versions[0]
    assert not version.last_use_event.query()
    native.finish_native_request(engine, req)
    assert replies[-1].state == "REQUEST_FINISHED"
    assert not version.last_use_event.query()  # Only retirement may consume this dependency.


def test_later_batch_replaces_instance_event_on_same_stream(native):
    engine, _, _ = scheduler()
    req = admitted(native, engine, identity(native))
    batch = SimpleNamespace(reqs=[req, req])
    versions = native.begin_execution_batch(engine, batch)
    assert len(versions) == 1
    native.finish_execution_batch(engine, versions)
    first = versions[0].last_use_event
    first.done = True
    versions = native.begin_execution_batch(engine, batch)
    assert versions[0].launching
    native.finish_execution_batch(engine, versions)
    assert versions[0].last_use_event is not first and not versions[0].last_use_event.query()


@pytest.mark.parametrize("overlap", [True, False])
def test_instance_completion_records_actual_launch_stream(native, overlap):
    engine, _, _ = scheduler()
    req = admitted(native, engine, identity(native))
    engine.enable_overlap = overlap
    actual_stream = object()
    engine.device_module.current_stream = lambda: actual_stream
    versions = native.begin_execution_batch(engine, SimpleNamespace(reqs=[req]))
    native.finish_execution_batch(engine, versions)
    assert versions[0].last_use_event.stream is (engine.forward_stream if overlap else actual_stream)


def test_uncertain_event_record_keeps_instance_launching(native):
    engine, _, _ = scheduler()
    req = admitted(native, engine, identity(native))
    versions = native.begin_execution_batch(engine, SimpleNamespace(reqs=[req]))

    def fail(**_):
        raise RuntimeError("event failed")

    engine.device_module.Event = lambda: SimpleNamespace(record=fail)
    with pytest.raises(RuntimeError, match="event failed"):
        native.finish_execution_batch(engine, versions)
    assert versions[0].launching and versions[0].last_use_event is None


def test_lost_instance_fails_engine_instead_of_repairing_one_rank_batch(native):
    engine, _, _ = scheduler()
    req = admitted(native, engine, identity(native))

    def fail(ref):
        raise ValueError("missing pinned slot")

    engine.tp_worker.model_runner.lora_manager._validate_managed_batch_ids = fail
    with pytest.raises(ValueError, match="missing pinned slot"):
        native.begin_execution_batch(engine, SimpleNamespace(reqs=[req]))


def test_cancel_uses_exact_native_request_and_status_absence_is_not_completion(native):
    engine, replies, cancelled = scheduler()
    req = admitted(native, engine, identity(native, rid="rid-long"))
    engine.waiting_queue = [req]
    native.control_execution(engine, native.LoRAExecutionControl(identity(native), "status"))
    assert replies[-1].state == "UNKNOWN" and not cancelled
    native.control_execution(engine, native.LoRAExecutionControl(identity(native), "cancel"))
    assert replies[-1].state == "REQUEST_FINISHED" and not cancelled
    native.control_execution(engine, native.LoRAExecutionControl(identity(native, rid="rid-long"), "cancel"))
    assert cancelled == [req]


@pytest.mark.parametrize("delivery_first", [True, False])
async def test_output_and_native_completion_are_independent(native, delivery_first):
    registry = Registry()
    control = execution_control(native, registry, lambda _: None, "cohort", "boot", 16)
    attempt = identity(native)
    await control.acquire(attempt, SimpleNamespace(lora_id="uid-A"), request_state())
    control.begin_submit(attempt)
    if delivery_first:
        control.delivered(attempt)
    control.observe(reply(native, attempt, used_gpu=True))
    await registry.released.wait()
    if not delivery_first:
        control.delivered(attempt)
    control.observe(reply(native, attempt))
    await asyncio.sleep(0)
    assert registry.releases == 1 and control.outcome(attempt).used_gpu
    assert not control.requests


async def test_tokenizer_cancel_fences_delayed_dispatch_and_rid_reuse(native):
    registry = Registry()
    control = execution_control(native, registry, lambda _: None, "cohort", "boot", 16)
    attempt = identity(native)
    control.cancel(attempt)  # Before generate/acquire.
    with pytest.raises(ValueError):
        await control.acquire(attempt, SimpleNamespace(lora_id="uid-A"), request_state())
    assert registry.refs == 0
    with pytest.raises(ValueError):
        control.begin_submit(attempt)


@pytest.mark.parametrize("pp_size,dp_rank", [(1, 0), (1, 1), (2, 1)])
async def test_only_native_output_leader_finishes_logical_request(native, pp_size, dp_rank):
    registry = Registry()
    boots = tuple(f"worker-{r}" for r in range(4 * pp_size))
    control = execution_control(
        native, registry, lambda _: None, "cohort", "boot", 32, worker_boots=boots, tp_size=4, dp_size=2
    )
    attempt = identity(native, dp_rank=dp_rank)
    await control.acquire(attempt, SimpleNamespace(lora_id="uid-A"), request_state())
    control.begin_submit(attempt)
    leader = dp_rank * 2
    for rank in set(range(len(boots))) - {leader}:
        assert not control.observe(reply(native, attempt, sender_rank=rank, worker_boot_id=boots[rank]))
    assert not control.observe(reply(native, attempt, sender_rank=leader, worker_boot_id="old"))
    assert not control.observe(
        reply(native, replace(attempt, rid="other"), sender_rank=leader, worker_boot_id=boots[leader])
    )
    assert registry.refs == 1
    message = reply(native, attempt, sender_rank=leader, worker_boot_id=boots[leader])
    control.observe(message)
    await registry.released.wait()
    control.observe(message)
    assert registry.releases == 1


def test_native_protocol_roundtrip_uses_rid_not_per_rank_request_identity(native):
    message = reply(native, identity(native), used_gpu=True)
    assert pickle.loads(pickle.dumps(message)) == message
    assert not hasattr(message, "identity")


async def test_native_http_cancel_during_acquire_cannot_lose_increment(native):
    registry = Registry()
    registry.allow_acquire.clear()
    control = execution_control(native, registry, lambda message: None, "cohort", "boot", 16)
    attempt = identity(native)
    waiter = asyncio.create_task(control.acquire(attempt, SimpleNamespace(lora_id="uid-A"), request_state()))
    await registry.entered.wait()
    waiter.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiter
    registry.allow_acquire.set()
    await asyncio.wait_for(registry.released.wait(), timeout=1)
    assert registry.refs == 0 and registry.releases == 1
    with pytest.raises(ValueError, match="ATTEMPT_NOT_ADMITTED"):
        control.begin_submit(attempt)


def bootstrap_scheduler():
    engine, _, _ = scheduler()
    del engine.lora_execution_owner
    engine.server_args = SimpleNamespace(
        dp_size=1,
        tokenizer_worker_num=1,
        disable_cuda_graph=True,
        disable_prefill_cuda_graph=True,
        enable_lora_overlap_loading=False,
        enable_strict_thinking=False,
    )
    engine.device = "cuda"
    engine.enable_lora = engine.enable_overlap = True
    engine.ps = SimpleNamespace(
        tp_size=1, pp_size=1, tp_rank=0, pp_rank=0, attn_tp_rank=0, attn_cp_rank=0, attn_dp_rank=0
    )
    engine.disaggregation_mode = SimpleNamespace(value="null")
    engine.spec_algorithm = SimpleNamespace(is_none=lambda: True, is_ngram=lambda: False)
    engine.enable_hierarchical_cache = engine.enable_hisparse = engine.enable_unified_memory = False
    engine.dllm_config = None
    engine.model_config = SimpleNamespace(
        is_multimodal=False, hf_config=SimpleNamespace(model_type="qwen3"), quantization=None
    )
    engine.is_fully_idle = lambda: True
    return engine


@pytest.mark.parametrize("ranks", [1, 2, 4])
def test_commit_fences_restarted_worker_before_installing_owner(native, ranks):
    boots = tuple(f"worker-{rank}" for rank in range(ranks))
    commit = native.LoRAExecutionInit("cohort", "tokenizer", "commit", 16, worker_boots=boots)
    engines = []
    for rank in range(ranks):
        engine = bootstrap_scheduler()
        engine.ps.tp_size, engine.ps.tp_rank = ranks, rank
        engine._lora_scheduler_boot_id = boots[rank]
        boot = native.validate_execution_commit(engine, commit)
        native.configure_execution_tracking(engine, "cohort", boot, 16)
        engines.append(engine)
    assert len({engine.lora_execution_owner for engine in engines}) == 1

    restarted = bootstrap_scheduler()
    restarted.ps.tp_size, restarted.ps.tp_rank = ranks, ranks - 1
    restarted._lora_scheduler_boot_id = "replacement-process"
    with pytest.raises(ValueError, match="ENGINE_EPOCH_MISMATCH"):
        native.validate_execution_commit(restarted, commit)
    assert not hasattr(restarted, "lora_execution_owner")
    stale = identity(native, engine_boot_id=engines[0].lora_execution_owner[1])
    with pytest.raises(ValueError, match="ENGINE_EPOCH_MISMATCH"):
        native.accept_execution(restarted, SimpleNamespace(lora_engine_boot_id=stale.engine_boot_id))


async def test_session_close_waits_for_each_rank_including_no_request_rank(native):
    sent = asyncio.Queue()
    control = native.LoRAVersionControl(Registry(), sent.put_nowait, "cohort", "boot", 16, worker_boots=("w0", "w1"))
    session = native.LoRASessionIdentity("cohort", "boot", "owner", "sid")
    closing = asyncio.create_task(control.close_session(session))
    command = await sent.get()
    first = native.LoRASessionReply(session, command.control_id, "SESSION_DRAINED", worker_boot_id="w0")
    assert control.observe_session(first)
    assert not control.observe_session(first)
    assert not control.observe_session(replace(first, sender_rank=1))
    await asyncio.sleep(0)
    assert not closing.done()
    assert control.observe_session(replace(first, sender_rank=1, worker_boot_id="w1"))
    assert await closing == "SESSION_DRAINED"


@pytest.mark.parametrize(
    "field,value",
    [
        ("enable_strict_thinking", True),
        ("tokenizer_worker_num", 2),
    ],
)
def test_native_bootstrap_rejects_uncovered_execution_profiles(native, field, value):
    engine = bootstrap_scheduler()
    setattr(engine.server_args, field, value)
    with pytest.raises(ValueError, match="Unsupported"):
        native.configure_execution_tracking(engine, "cohort", "boot", 16)
    assert not hasattr(engine, "lora_execution_owner")


def test_native_old_boot_control_is_rejected_after_scheduler_restart(native):
    engine, _, _ = scheduler()
    del engine.lora_execution_owner
    with pytest.raises(ValueError, match="ENGINE_EPOCH_MISMATCH"):
        native.control_execution(engine, native.LoRAExecutionControl(identity(native), "cancel"))


@pytest.mark.parametrize("prefill_graph", [False, True])
def test_native_bootstrap_preserves_graph_execution(native, prefill_graph):
    engine = bootstrap_scheduler()
    engine.server_args.disable_cuda_graph = False
    engine.server_args.disable_prefill_cuda_graph = not prefill_graph
    native.configure_execution_tracking(engine, "cohort", "boot", 16)
    req = admitted(native, engine, identity(native))
    launched = native.begin_execution_batch(engine, SimpleNamespace(reqs=[req]))
    native.finish_execution_batch(engine, launched)  # After graph replay.
    native.finish_native_request(engine, req)
    assert not hasattr(engine, "lora_requests")
    assert launched[0].last_use_event.stream is engine.forward_stream
    assert not launched[0].last_use_event.query()


@pytest.mark.parametrize("ngram", [True, False])
def test_native_speculative_guard_preserves_upstream_ngram_support(native, ngram):
    engine = bootstrap_scheduler()
    engine.spec_algorithm = SimpleNamespace(is_none=lambda: False, is_ngram=lambda: ngram)
    if ngram:
        native.configure_execution_tracking(engine, "cohort", "boot", 16)
        assert engine.lora_execution_owner == ("cohort", "boot")
    else:
        with pytest.raises(ValueError, match="Unsupported"):
            native.configure_execution_tracking(engine, "cohort", "boot", 16)


@pytest.mark.parametrize("pp_rank,tp_rank", [(0, 0), (0, 1), (1, 0), (1, 1)])
def test_pipeline_boot_handshake_has_unique_stage_and_tensor_rank(native, pp_rank, tp_rank):
    engine = bootstrap_scheduler()
    engine.ps = SimpleNamespace(tp_size=2, pp_size=2, tp_rank=tp_rank, pp_rank=pp_rank)
    boots = ("p0t0", "p0t1", "p1t0", "p1t1")
    rank = pp_rank * 2 + tp_rank
    engine._lora_scheduler_boot_id = boots[rank]
    command = native.LoRAExecutionInit("cohort", "tokenizer", "control", 16, worker_boots=boots)
    assert native.execution_rank(engine) == rank
    assert native.validate_execution_commit(engine, command) == native.execution_boot_id("tokenizer", boots)
    engine._lora_scheduler_boot_id = boots[(rank + 1) % 4]
    with pytest.raises(ValueError, match="ENGINE_EPOCH_MISMATCH"):
        native.validate_execution_commit(engine, command)


@pytest.mark.parametrize("modules,allowed", [({"qkv_proj", "o_proj"}, True), ({"qkv_proj", "down_proj"}, False)])
def test_attention_dp_profile_checks_actual_wrapped_modules(native, modules, allowed):
    engine = bootstrap_scheduler()
    engine.server_args.dp_size = 2
    engine.server_args.enable_dp_attention = True
    engine.server_args.lora_backend = "triton"
    engine.ps.tp_size = 4
    engine.tp_worker.model_runner.lora_manager.target_modules = modules
    if allowed:
        native.configure_execution_tracking(engine, "cohort", "boot", 16)
    else:
        with pytest.raises(ValueError, match="attention-only"):
            native.configure_execution_tracking(engine, "cohort", "boot", 16)


def test_dcp_cannot_gather_different_attention_dp_requests(native):
    engine = bootstrap_scheduler()
    engine.ps.tp_size = 4
    engine.server_args.dp_size = 2
    engine.server_args.enable_dp_attention = True
    engine.server_args.dcp_size = 4
    with pytest.raises(ValueError, match="DCP groups must stay"):
        native.configure_execution_tracking(engine, "cohort", "boot", 16)


@pytest.fixture
def communicator():
    root = os.environ.get("RELAX_SGLANG_SOURCE")
    if not root:
        pytest.skip("requires patched native source: RELAX_SGLANG_SOURCE")
    path = Path(root) / "python/sglang/srt/managers/communicator.py"
    spec = importlib.util.spec_from_file_location("native_owned_communicator", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.FanOutCommunicator


def message(control="load-1", *, boot="boot", phase="load", rank=0):
    return SimpleNamespace(control=control, boot=boot, phase=phase, operation="B-1", uid="native-B-1", rank=rank)


def key(obj):
    return obj.control, obj.boot, obj.operation, obj.uid, obj.phase


async def test_owned_rpc_installs_waiter_before_synchronous_send(communicator):
    channel = communicator(lambda obj: channel.handle_recv(obj), 1)
    result = await asyncio.wait_for(channel(message()), timeout=1)
    assert result[0].control == "load-1"


async def test_owned_rpc_http_cancel_keeps_native_channel_until_matching_ack(communicator):
    sent = []
    first_sent, second_sent = asyncio.Event(), asyncio.Event()

    def send(obj):
        sent.append(obj.control)
        (first_sent if obj.control == "load-1" else second_sent).set()

    channel = communicator(send, 1, mode="owned", correlation_key=key, response_rank=lambda obj: obj.rank)
    first = asyncio.create_task(channel(message()))
    await first_sent.wait()
    first.cancel()
    with pytest.raises(asyncio.CancelledError):
        await first
    second = asyncio.create_task(channel(message("load-2")))
    await asyncio.sleep(0)
    assert sent == ["load-1"]
    channel.handle_recv(message())
    await asyncio.wait_for(second_sent.wait(), timeout=1)
    # A duplicate old ACK must not consume the next RPC's reply budget.
    channel.handle_recv(message())
    channel.handle_recv(message("load-2", boot="old-boot"))
    channel.handle_recv(message("load-2", phase="retire"))
    correct = message("load-2")
    channel.handle_recv(correct)
    assert await asyncio.wait_for(second, timeout=1) == [correct]


async def test_owned_rpc_deduplicates_sender_before_counting_fanout(communicator):
    sent = asyncio.Event()
    channel = communicator(
        lambda obj: sent.set(), 2, mode="owned", correlation_key=key, response_rank=lambda obj: obj.rank
    )
    waiter = asyncio.create_task(channel(message()))
    await sent.wait()
    first, second = message(rank=0), message(rank=1)
    channel.handle_recv(first)
    channel.handle_recv(message(rank=0))
    channel.handle_recv(message(rank=9))
    channel.handle_recv(message(rank=-1))
    channel.handle_recv(second)
    assert await asyncio.wait_for(waiter, timeout=1) == [first, second]


async def test_owned_rpc_ambiguous_send_does_not_release_channel(communicator):
    sent = asyncio.Event()

    def fail_after_possible_send(obj):
        sent.set()
        raise OSError("lost send confirmation")

    channel = communicator(
        fail_after_possible_send, 1, mode="owned", correlation_key=key, response_rank=lambda obj: obj.rank
    )
    waiter = asyncio.create_task(channel(message()))
    await sent.wait()
    await asyncio.sleep(0)
    assert not waiter.done()
    reply = message()
    channel.handle_recv(reply)
    assert await asyncio.wait_for(waiter, timeout=1) == [reply]
