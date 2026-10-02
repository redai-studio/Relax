# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Run production tokenizer/scheduler methods in the patched native
environment.

Only transport and GPU completion are controlled; no extracted method bodies.
This is hook coverage, not a substitute for two live engines or CUDA evidence.
"""

import asyncio
from contextlib import nullcontext
from types import SimpleNamespace

import pytest


@pytest.fixture
def native():
    tokenizer = pytest.importorskip(
        "sglang.srt.managers.tokenizer_manager", reason="requires complete patched SGLang runtime"
    )
    scheduler = pytest.importorskip("sglang.srt.managers.scheduler", reason="requires complete patched SGLang runtime")
    from sglang.srt.lora import version_control

    assert hasattr(scheduler.Scheduler, "handle_lora_execution_control")
    return tokenizer, scheduler, version_control


class Registry:
    def __init__(self):
        self.refs = 0
        self.released = asyncio.Event()

    async def acquire_exact(self, ref):
        self.refs += 1

    async def release(self, uid):
        assert self.refs == 1
        self.refs -= 1
        self.released.set()


async def tokenizer_owner(native):
    tokenizer, _, protocol = native
    instance = tokenizer.TokenizerManager.__new__(tokenizer.TokenizerManager)
    registry, messages = Registry(), []
    attempt = protocol.LoRAExecutionIdentity("cohort", "boot", "uid-A", "owner", "sid", "rid")
    control = protocol.LoRAVersionControl(registry, messages.append, "cohort", "boot", 8)
    control.versions["uid-A"] = protocol.TokenizerVersion(
        protocol.NativeLoadIdentity(*attempt.instance_key),
        SimpleNamespace(lora_id="uid-A"),
        "digest",
        state="READY",
        registered=True,
    )
    obj = SimpleNamespace(rid="rid", _lora_execution=attempt, stream=False, return_logprob=False)
    state = tokenizer.ReqState(
        [], False, asyncio.Event(), obj, SimpleNamespace(set_finished_time=lambda: None, get_e2e_latency=lambda: 0)
    )
    instance.rid_to_state = control.requests = {"rid": state}
    await control.acquire(attempt, SimpleNamespace(lora_id="uid-A"), state)
    control.begin_submit(attempt)
    instance.lora_version_control = control
    instance.child_rid_to_logical_rid = {}
    instance.logical_rid_to_child_rids = {}
    return instance, registry, messages, attempt, state


@pytest.mark.parametrize("managed", [False, True])
def test_pp_loop_reaches_native_batch_selection(native, managed):
    _, scheduler, _ = native
    engine = scheduler.Scheduler.__new__(scheduler.Scheduler)
    engine.init_pp_loop_state = lambda: None
    engine.pp_loop_size = 1
    engine.running_mbs = engine.last_mbs = [None]
    engine.ps = SimpleNamespace(pp_size=2)
    engine.pp_group = SimpleNamespace(is_last_rank=True)
    engine.request_receiver = SimpleNamespace(recv_requests=lambda: [])
    engine.process_input_requests = lambda requests: None
    if managed:
        engine.lora_execution_owner = ("cohort", "boot")

    class BatchSelectionReached(Exception):
        pass

    def select_batch(**kwargs):
        raise BatchSelectionReached

    engine.get_next_batch_to_run = select_batch
    # Run the real PP entry through message handling, without a GPU forward.
    # A stale call to a removed helper fails before this native boundary.
    with pytest.raises(BatchSelectionReached):
        engine.event_loop_pp()


def test_forward_residency_checks_slots_without_rechecking_static_configuration(native):
    from sglang.srt.lora.lora_manager import LoRAManager

    manager = LoRAManager.__new__(LoRAManager)
    ref = SimpleNamespace(lora_id="uid", pinned=True)
    config = SimpleNamespace(r=8, lora_alpha=8)
    manager.lora_refs = {"uid": ref}
    manager.configs = {"uid": config}
    manager.loras = {"uid": SimpleNamespace(config=config, scaling=1)}
    manager.retiring_loras = manager.pending_lora_load_events = {}
    manager.dtype = "fixture-dtype"

    def static_check(config):
        raise AssertionError("static configuration checked")

    manager.memory_pool = SimpleNamespace(
        uid_to_buffer_id={"uid": 0},
        buffer_id_to_uid=["uid"],
        retiring_loras=set(),
        clearing_loras={},
        dtype=manager.dtype,
        can_support=static_check,
    )
    manager._validate_managed_batch_ids({"uid"})
    with pytest.raises(AssertionError, match="static configuration checked"):
        manager.validate_managed_residency(ref)  # Prepare still performs the full check.
    manager.memory_pool.clearing_loras["uid"] = object()
    with pytest.raises(ValueError, match="ADAPTER_NOT_READY"):
        manager._validate_managed_batch_ids({"uid"})


async def test_native_discard_after_ambiguous_ipc_send_keeps_execution_owner(native):
    instance, registry, messages, attempt, state = await tokenizer_owner(native)
    instance._discard_pending_req_states(state.obj, {"rid": state.lifecycle_id})
    assert instance.rid_to_state["rid"] is state
    assert state.consumer_detached
    assert messages[-1].identity == attempt and messages[-1].action == "cancel"
    assert registry.refs == 1
    instance.lora_version_control.observe(
        native[2].LoRAExecutionReply(attempt.rid, attempt.engine_boot_id, "REQUEST_FINISHED")
    )
    await registry.released.wait()
    assert instance.lora_version_control.outcome(attempt).delivery == "ABANDONED"


async def test_native_scheduler_rejection_wakes_real_tokenizer_consumer(native, monkeypatch):
    tokenizer, scheduler, protocol = native
    instance, registry, messages, attempt, state = await tokenizer_owner(native)
    engine = scheduler.Scheduler.__new__(scheduler.Scheduler)
    engine.lora_execution_owner = ("cohort", "boot")
    # Rejection occurs before Req creation or any GPU submission.
    reply = engine.handle_generate_request(
        SimpleNamespace(
            lora_engine_boot_id=attempt.engine_boot_id,
            rid=attempt.rid,
            sampling_params=SimpleNamespace(regex=".*"),
            session_params=None,
            http_worker_ipc=None,
        )
    )
    assert reply.error == "UNSUPPORTED_EXECUTION_REQUEST"
    incoming = asyncio.Queue()

    async def recv(socket):
        return await incoming.get()

    monkeypatch.setattr(tokenizer, "async_sock_recv", recv)
    instance.recv_from_detokenizer = object()
    instance.soft_watchdog = SimpleNamespace(disable=nullcontext, feed=lambda: None)
    instance.config_value = lambda name: "legacy-version"
    loop = asyncio.create_task(instance.handle_loop())
    try:
        await incoming.put(reply)
        await asyncio.wait_for(state.event.wait(), timeout=1)
        assert state.out_list[-1]["meta_info"]["finish_reason"]["message"] == reply.error
        # The leader rejected this submission before Req creation.
        await asyncio.wait_for(registry.released.wait(), timeout=1)
        assert registry.refs == 0
    finally:
        loop.cancel()
        await asyncio.gather(loop, return_exceptions=True)


async def test_native_bootstrap_rejects_msgpack_before_sending_any_message(native, monkeypatch):
    tokenizer, _, _ = native
    instance = tokenizer.TokenizerManager.__new__(tokenizer.TokenizerManager)
    # No other fields are initialized: the profile guard must run first.
    from sglang.srt.managers import tokenizer_control_mixin

    monkeypatch.setattr(tokenizer_control_mixin, "_USE_PICKLE_IPC", False)
    with pytest.raises(ValueError, match="requires pickle IPC"):
        await instance.start_lora_version_control("cohort", 8)


@pytest.mark.parametrize(
    "result",
    [
        {"output_ids": [1], "meta_info": {"finish_reason": {"type": "length"}}},
        {"output_ids": [], "meta_info": {"finish_reason": {"type": "length"}}},
        {"output_ids": [1], "meta_info": {"finish_reason": {"type": "abort"}}},
    ],
)
async def test_native_warmup_builds_exact_lora_generation_and_requires_success(native, result):
    tokenizer, _, protocol = native
    manager = tokenizer.TokenizerManager.__new__(tokenizer.TokenizerManager)
    identity = protocol.LoRAExecutionIdentity("cohort", "boot", "uid", "prepare", "session", "rid")
    ref = SimpleNamespace(lora_id="uid", lora_name="version-name", pinned=True)
    sent = []

    async def generate(obj):
        sent.append(obj)
        yield result

    manager.generate_request = generate
    valid = bool(result["output_ids"]) and result["meta_info"]["finish_reason"]["type"] == "length"
    if valid:
        await manager._generate_lora_control_token(identity, ref)
    else:
        with pytest.raises(RuntimeError, match="INTERNAL_GENERATION_FAILED"):
            await manager._generate_lora_control_token(identity, ref)
    obj = sent[0]
    assert obj.lora_path == ref.lora_name and obj._lora_execution_ref is ref
    assert obj._lora_execution == identity and obj.rid == identity.rid
    assert obj.sampling_params["max_new_tokens"] == 1
    assert obj.stream is False and obj.log_metrics is False


@pytest.mark.parametrize("outcome", ["success", "failure", "handshake_pending"])
def test_native_managed_health_handler_uses_owned_probe_during_bootstrap(monkeypatch, outcome):
    server = pytest.importorskip(
        "sglang.srt.entrypoints.http_server", reason="requires complete patched SGLang runtime"
    )

    async def scenario():
        calls = []

        async def generate(identity, ref):
            raise AssertionError("HTTP handler must delegate probe ownership")

        async def probe(callback, timeout):
            assert callback is generate and timeout == server.HEALTH_CHECK_TIMEOUT
            calls.append("probe")
            if outcome == "failure":
                raise RuntimeError("own probe incomplete; unrelated traffic is irrelevant")

        manager = SimpleNamespace(
            gracefully_exit=False,
            server_status=server.ServerStatus.Starting,
            _lora_execution_starting=True,
            lora_version_control=None if outcome == "handshake_pending" else SimpleNamespace(),
            health_lora_publication=probe,
            _generate_lora_control_token=generate,
            last_receive_tstamp=float("inf"),  # Must never substitute for the actual probe result.
        )
        monkeypatch.setattr(server, "_global_state", SimpleNamespace(tokenizer_manager=manager))
        response = await server.health_generate(SimpleNamespace(url=SimpleNamespace(path="/health")))
        assert response.status_code == (200 if outcome == "success" else 503)
        assert calls == ([] if outcome == "handshake_pending" else ["probe"])
        if outcome == "success":
            assert manager.server_status == server.ServerStatus.Up

    asyncio.run(scenario())


async def test_native_base_probe_uses_private_execution_without_adapter(native):
    tokenizer, _, protocol = native
    manager = tokenizer.TokenizerManager.__new__(tokenizer.TokenizerManager)
    identity = protocol.LoRAExecutionIdentity(
        "cohort", "boot", None, "native-health", "session", "relax-probe-rid", kind="probe"
    )
    sent = []

    async def generate(obj):
        sent.append(obj)
        yield {"output_ids": [1], "meta_info": {"finish_reason": {"type": "length"}}}

    manager.generate_request = generate
    await manager._generate_lora_control_token(identity, None)
    obj = sent[0]
    assert obj.lora_path is None and obj._lora_execution_ref is None
    assert obj._lora_execution == identity
    assert obj.rid == identity.rid and obj.input_ids == [0]
    assert obj.log_metrics is False


@pytest.mark.parametrize("mode", ["bootstrap", "running"])
async def test_native_mutations_reject_before_local_state_or_ipc(native, mode):
    tokenizer, _, protocol = native
    manager = tokenizer.TokenizerManager.__new__(tokenizer.TokenizerManager)
    if mode == "bootstrap":
        manager._lora_execution_starting = True
    else:
        manager.lora_version_control = object()
    manager.is_pause = False
    # No loop, locks, registry or IPC objects exist: rejection must precede
    # all their side effects, even for malformed legacy control payloads.
    calls = {
        "flush_cache": (),
        "clear_hicache_storage": (),
        "attach_hicache_storage": ("unused",),
        "detach_hicache_storage": (),
        "init_weights_update_group": (None,),
        "destroy_weights_update_group": (None,),
        "init_weights_send_group_for_remote_instance": (None,),
        "send_weights_to_remote_instance": (None,),
        "load_lora_adapter": (None,),
        "load_lora_adapter_from_tensors": (None,),
        "update_lora_from_distributed": (None,),
        "unload_lora_adapter": (None,),
        "update_weights_from_distributed": (None,),
        "update_weights_from_tensor": (None,),
        "update_weights_from_ipc": (None,),
        "update_weights_from_disk": (None,),
        "release_memory_occupation": (None,),
        "resume_memory_occupation": (None,),
        "post_process_weights": (None,),
        "slow_down": (None,),
        "set_internal_state": (None,),
        "open_session": (None,),
        "close_session": (None,),
        "pause_generation": (None,),
        "continue_generation": (None,),
        "scale_elastic_ep": (None,),
    }
    for name, args in calls.items():
        with pytest.raises(protocol.ManagedLoRAOperationError) as error:
            await getattr(manager, name)(*args)
        assert error.value.code == "MANAGED_OPERATION_FORBIDDEN"
        assert error.value.operation == name
    with pytest.raises(protocol.ManagedLoRAOperationError):
        manager.abort_request(abort_all=True)
    assert manager.is_pause is False


def test_native_scheduler_rejects_late_mutations_without_invoking_handlers(native):
    _, scheduler, _ = native
    from sglang.srt.managers import io_struct as io

    engine = scheduler.Scheduler.__new__(scheduler.Scheduler)
    engine.lora_execution_owner = ("cohort", "boot")
    messages = [
        io.FlushCacheReqInput(),
        io.UpdateWeightFromDiskReqInput(model_path="unused"),
        io.UpdateWeightsFromTensorReqInput(serialized_named_tensors=[]),
        io.UpdateWeightsFromDistributedReqInput(names=[], dtypes=[], shapes=[]),
        io.UpdateWeightsFromIPCReqInput(zmq_handles={}),
        io.PostProcessWeightsReqInput(),
        io.ReleaseMemoryOccupationReqInput(),
        io.ResumeMemoryOccupationReqInput(),
        io.InitWeightsUpdateGroupReqInput(master_address="unused", master_port=1, rank_offset=0, world_size=1),
        io.DestroyWeightsUpdateGroupReqInput(),
        io.InitWeightsSendGroupForRemoteInstanceReqInput(
            master_address="unused", ports="1", group_rank=0, world_size=1
        ),
        io.SendWeightsToRemoteInstanceReqInput(master_address="unused", ports="1"),
        io.SlowDownReqInput(forward_sleep_time=1),
        io.ClearHiCacheReqInput(),
        io.AttachHiCacheStorageReqInput(hicache_storage_backend="unused"),
        io.DetachHiCacheStorageReqInput(),
        io.ScaleElasticEPReqInput(new_ep_size=2),
    ]
    for message in messages:
        response = engine._dispatch_input_request(message)
        assert response.success is False
        assert "MANAGED_OPERATION_FORBIDDEN" in response.message
    assert engine._dispatch_input_request(io.SetInternalStateReq(server_args={})).updated is False
    assert engine.handle_rpc_request(io.RpcReqInput(method="flush_cache")).success is False
    assert engine.flush_cache() is False
    engine.continue_generation(io.ContinueGenerationReqInput(torch_empty_cache=True))
    assert engine.open_session(io.OpenSessionReqInput(capacity_of_str_len=1, session_id="legacy")).success is False

    # Ordinary mode retains dispatch, without requiring a GPU for this check.
    del engine.lora_execution_owner
    observed = []
    engine._request_dispatcher = lambda message: observed.append(message)
    for message in messages:
        engine._dispatch_input_request(message)
    assert observed == messages
    assert engine._lora_legacy_control_used is True


async def test_native_exact_abort_retains_owner_after_consumer_detach(native):
    instance, registry, messages, attempt, state = await tokenizer_owner(native)
    instance.lora_version_control.abandon(attempt)
    assert instance.rid_to_state[attempt.rid] is state
    instance.abort_request(attempt.rid)
    assert messages[-1].identity == attempt
    assert messages[-1].action == "cancel"
    assert registry.refs == 1
    with pytest.raises(native[2].ManagedLoRAOperationError):
        instance.abort_request("unknown-prefix")
    instance.lora_version_control.observe(
        native[2].LoRAExecutionReply(attempt.rid, attempt.engine_boot_id, "REQUEST_FINISHED")
    )
    await registry.released.wait()
    assert registry.refs == 0


def test_native_managed_http_error_has_stable_code_and_conflict_status(native):
    import json

    from sglang.srt.entrypoints import http_server

    error = native[2].ManagedLoRAOperationError("pause_generation")
    response = http_server._create_error_response(error)
    assert response.status_code == 409
    assert json.loads(response.body)["error"] == {
        "code": "MANAGED_OPERATION_FORBIDDEN",
        "message": "MANAGED_OPERATION_FORBIDDEN: pause_generation",
        "operation": "pause_generation",
    }


async def test_native_bootstrap_cannot_adopt_already_admitted_legacy_control(native, monkeypatch):
    tokenizer, _, _ = native
    from sglang.srt.managers import tokenizer_control_mixin

    monkeypatch.setattr(tokenizer_control_mixin, "_USE_PICKLE_IPC", True)
    manager = tokenizer.TokenizerManager.__new__(tokenizer.TokenizerManager)
    manager.auto_create_handle_loop = lambda: None
    manager.is_pause = False
    entered, finish = asyncio.Event(), asyncio.Event()

    async def flush(_):
        entered.set()
        await finish.wait()
        return [SimpleNamespace(success=True)]

    manager.flush_cache_communicator = flush
    old = asyncio.create_task(manager.flush_cache())
    await entered.wait()
    with pytest.raises(ValueError, match="MANAGED_BOOTSTRAP_REQUIRES_FRESH_ENGINE"):
        await manager.start_lora_version_control("cohort", 8)
    finish.set()
    await old
    # Finishing or cancelling an old operation never makes this a fresh
    # process again. No attempted adoption of uncertain legacy state.
    with pytest.raises(ValueError, match="MANAGED_BOOTSTRAP_REQUIRES_FRESH_ENGINE"):
        await manager.start_lora_version_control("cohort", 8)
    assert not getattr(manager, "_lora_execution_starting", False)


async def test_public_binding_keeps_identity_and_rejects_reserved_probe_ids(native, tmp_path):
    import copy

    from sglang.srt.lora.lora_registry import LoRARef
    from sglang.srt.managers.io_struct import GenerateReqInput

    tokenizer, _, protocol = native
    manager = tokenizer.TokenizerManager.__new__(tokenizer.TokenizerManager)
    control = protocol.LoRAVersionControl(Registry(), lambda message: None, "cohort", "boot", 32)
    ref = LoRARef(lora_id="uid", lora_name="uid", lora_path=str(tmp_path), pinned=True)
    record = protocol.TokenizerVersion(
        protocol.NativeLoadIdentity("boot", "uid"),
        ref,
        "a" * 64,
        state="READY",
    )
    manager.lora_version_control = control
    manager._lora_artifact_root = tmp_path
    control.versions["uid"] = record
    envelope = {
        "cohort_id": "cohort",
        "engine_boot_id": "boot",
        "native_lora_id": "uid",
        "digest": "a" * 64,
        "owner_epoch": "owner",
        "session_id": "s",
    }
    obj = GenerateReqInput(
        input_ids=[1, 2],
        rid="r",
        sampling_params={"max_new_tokens": 1},
        lora_path=ref.lora_name,
        lora_binding=envelope,
        token_ids_logprob=[3],
    )
    manager.bind_lora_publication_request(obj)
    assert not hasattr(obj, "_lora_execution_fingerprint")
    for field, value in (("session_id", "relax-health:1"), ("owner_epoch", "native-health")):
        changed = copy.deepcopy(obj)
        changed.lora_binding[field] = value
        with pytest.raises(ValueError, match="RESERVED_SESSION_ID"):
            manager.bind_lora_publication_request(changed)
    changed = copy.deepcopy(obj)
    changed.rid = "relax-probe-old"
    with pytest.raises(ValueError, match="RESERVED_REQUEST_ID"):
        manager.bind_lora_publication_request(changed)
    assert obj._lora_adapter_metadata["native_lora_id"] == "uid"
    obj.session_params = {"id": "foreign"}
    with pytest.raises(ValueError, match="UNSUPPORTED_MANAGED_REQUEST"):
        manager.bind_lora_publication_request(obj)


@pytest.mark.parametrize(
    "method,action", [("release_memory_occupation", "suspend"), ("resume_memory_occupation", "resume")]
)
async def test_native_memory_http_preserves_owner_and_retry_sequence(native, method, action):
    tokenizer, _, _ = native
    manager = tokenizer.TokenizerManager.__new__(tokenizer.TokenizerManager)
    calls = []

    async def change(received_action, tags, *, sequence):
        calls.append((received_action, tags, sequence))
        return {"memory_state": "SUSPENDED"}

    manager.lora_version_control = SimpleNamespace(owner=("cohort", "boot"))
    manager._change_lora_memory = change
    request = SimpleNamespace(lora_memory_owner=("cohort", "old-boot"), lora_memory_sequence=7, tags=["weights"])
    with pytest.raises(ValueError, match="ENGINE_EPOCH_MISMATCH"):
        await getattr(manager, method)(request)
    assert not calls
    request.lora_memory_owner = ("cohort", "boot")
    await getattr(manager, method)(request)
    await getattr(manager, method)(request)
    assert calls == [(action, ["weights"], 7), (action, ["weights"], 7)]


def test_native_dp_idle_clears_static_graph_adapter_metadata(native):
    import torch
    from sglang.srt.lora.backend.triton_backend import TritonLoRABackend

    backend = TritonLoRABackend.__new__(TritonLoRABackend)
    for name in ("cuda_graph_batch_info", "cuda_graph_sgemm_batch_info", "prefill_cuda_graph_batch_info"):
        setattr(
            backend,
            name,
            SimpleNamespace(
                lora_ranks=torch.full((3,), 8),
                scalings=torch.ones(3),
                weight_indices=torch.ones(4),
                seg_lens=torch.ones(4),
                seg_indptr=torch.arange(5),
            ),
        )
    backend.reset_batch_state()
    for name in ("cuda_graph_batch_info", "cuda_graph_sgemm_batch_info", "prefill_cuda_graph_batch_info"):
        info = getattr(backend, name)
        assert torch.count_nonzero(info.lora_ranks) == 0
    # Decode segmentation remains valid; rank zero makes the captured kernels no-op.
    assert torch.all(backend.cuda_graph_batch_info.seg_lens == 1)


@pytest.mark.parametrize("pending", ["send_proxy_work", "send_output_work", "last_rank_comm_queue", "peer_rank"])
def test_native_pp_memory_waits_for_communication_and_all_stage_tp_ranks(native, monkeypatch, pending):
    _, module, _ = native
    engine = module.Scheduler.__new__(module.Scheduler)
    command = object()
    engine._lora_pending_memory = command
    engine.is_fully_idle = lambda: True
    engine.lora_versions = {}
    engine.waiting_queue = []
    engine.running_batch = engine.last_batch = None
    engine.last_mbs = [None]
    engine.last_rank_comm_queue = []
    engine.pp_outputs = None
    engine.send_proxy_work = []
    engine.send_output_work = []
    engine._pp_tensor_dict_inbox = {}
    peer_ready = False
    engine._lora_memory_consensus = lambda ready: [ready, peer_ready if pending == "peer_rank" else ready]
    if pending != "peer_rank":
        getattr(engine, pending).append(object())
    calls = []
    monkeypatch.setattr(module, "control_memory", lambda scheduler, cmd: calls.append(cmd) or "ack")
    monkeypatch.setattr(module, "send_managed_reply", lambda scheduler, reply: calls.append(reply))
    engine._poll_lora_memory()
    assert not calls and engine._lora_pending_memory is command
    peer_ready = True
    if pending != "peer_rank":
        getattr(engine, pending).clear()
    engine._poll_lora_memory()
    engine._poll_lora_memory()
    assert calls == [command, "ack"] and engine._lora_pending_memory is None


@pytest.mark.parametrize("cancel_first", [True, False])
async def test_native_dispatch_and_cancel_share_ordered_input_channel(native, monkeypatch, cancel_first):
    module, _, protocol = native
    instance, registry, _, attempt, state = await tokenizer_owner(native)
    control = instance.lora_version_control
    # Reset fixture to acquired-but-not-dispatched, as after tokenization.
    control.requests[attempt.rid].dispatched = False
    sent = []
    instance._dispatch_to_scheduler = control.send = sent.append
    monkeypatch.setattr(module, "wrap_shm_features", lambda obj: obj)
    tokenized = SimpleNamespace(
        rid=attempt.rid,
        lora_engine_boot_id=attempt.engine_boot_id,
        wrap_pickle_fields=lambda: None,
        time_stats=SimpleNamespace(
            set_api_server_dispatch_time=lambda: None, set_api_server_dispatch_finish_time=lambda: None
        ),
    )
    if cancel_first:
        control.cancel(attempt)
        with pytest.raises(ValueError, match="ATTEMPT_NOT_ADMITTED"):
            instance._send_one_request(tokenized)
        assert not sent
    else:
        instance._send_one_request(tokenized)
        control.cancel(attempt)
        assert sent[0] is tokenized and isinstance(sent[1], protocol.LoRAExecutionControl)
        assert sent[1].identity == attempt
        assert registry.refs == 1


def test_native_unmanaged_embedding_dispatch_needs_no_lora_fields(native, monkeypatch):
    module, _, _ = native
    manager = module.TokenizerManager.__new__(module.TokenizerManager)
    sent = []
    manager._dispatch_to_scheduler = sent.append
    manager.rid_to_state = {}
    monkeypatch.setattr(module, "wrap_shm_features", lambda obj: obj)
    obj = SimpleNamespace(
        rid="embedding",
        wrap_pickle_fields=lambda: None,
        time_stats=SimpleNamespace(
            set_api_server_dispatch_time=lambda: None, set_api_server_dispatch_finish_time=lambda: None
        ),
    )
    manager._send_one_request(obj)
    assert sent == [obj]


@pytest.mark.parametrize("mode", ["nonoverlap", "pp", "overlap"])
@pytest.mark.parametrize("managed", [False, True])
def test_native_abort_handles_optional_overlap_queue(native, monkeypatch, mode, managed):
    """Execute Scheduler.abort_request with each event loop's actual queue
    shape."""
    _, scheduler, _ = native
    engine = scheduler.Scheduler.__new__(scheduler.Scheduler)
    engine.ps = SimpleNamespace(pp_size=2 if mode == "pp" else 1)
    engine.chunked_req = None
    engine.waiting_queue = []
    engine.dllm_config = None
    engine.disaggregation_mode = scheduler.DisaggregationMode.NULL
    grammar_calls = []
    engine.grammar_manager = SimpleNamespace(
        abort_requests=lambda command, **kwargs: grammar_calls.append((command, kwargs))
    )
    if managed:
        engine.lora_execution_owner = ("cohort", "boot")

    class Request:
        def __init__(self, rid, finished=False):
            self.rid = rid
            self.to_finish = None
            self.is_finished = finished

        def finished(self):
            return self.is_finished

    target = Request("attempt")
    neighbor = Request("attempt-other")
    completed = Request("attempt", finished=True)
    batch = SimpleNamespace(reqs=[target, neighbor, completed])
    if mode == "pp":
        engine.running_mbs = [None]
        engine.mbs = [batch]
        engine.last_mbs = [batch]  # The same Req may have several native owners.
    else:
        engine.running_batch = batch if mode == "nonoverlap" else None
        engine.last_batch = None
    if mode == "overlap":
        # Only a pending overlap result owns these requests.
        engine.result_queue = [(batch, object())]
    else:
        assert not hasattr(engine, "result_queue")

    def premature_finish(*args):
        pytest.fail("abort must not release an in-flight native request")

    monkeypatch.setattr(scheduler, "finish_native_request", premature_finish)
    command = scheduler.AbortReq(rid="attempt")
    scheduler.Scheduler.abort_request(engine, command, exact=managed)

    assert isinstance(target.to_finish, scheduler.FINISH_ABORT)
    # Managed cancellation is exact; unmanaged legacy cancellation keeps prefix semantics.
    assert (neighbor.to_finish is None) == managed
    assert completed.to_finish is None
    assert grammar_calls == [(command, {"exact": managed})]
    if mode == "overlap":
        assert engine.result_queue[0][0] is batch
