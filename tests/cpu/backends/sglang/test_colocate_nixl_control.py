# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import json
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

from relax.backends.sglang import shard_p2p_weight_loader as loader
from relax.backends.sglang.weight_loader import EXPERT_SLICE_LOADER
from relax.distributed.checkpoint_service.backends.shard_p2p.remote import common


@pytest.mark.parametrize("enabled,backend", [(False, "triton"), (True, "auto")])
def test_async_probe_preserves_expert_slice_capability(monkeypatch, enabled, backend):
    from relax.backends.megatron.weight_update.expert_layout import DirectExpertLayout
    from relax.utils import misc

    model = torch.nn.Linear(2, 2, dtype=torch.bfloat16)
    if backend == "auto":
        from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
        from sglang.srt.layers.moe.utils import MoeRunnerBackend
        from sglang.srt.layers.quantization.unquant import UnquantizedFusedMoEMethod

        layer = FusedMoE.__new__(FusedMoE)
        torch.nn.Module.__init__(layer)
        method = UnquantizedFusedMoEMethod.__new__(UnquantizedFusedMoEMethod)
        torch.nn.Module.__init__(method)
        method.runner = SimpleNamespace(runner_backend=MoeRunnerBackend.TRITON)
        layer.quant_method = method
        model.add_module("experts", layer)
    model._relax_native_weight_guard = SimpleNamespace(models=[model])
    model._relax_nixl_probe_agent = object()
    args = SimpleNamespace(
        model_path="fixture",
        tp_size=8,
        pp_size=1,
        ep_size=1,
        moe_dp_size=1,
        enable_lora=False,
        speculative_algorithm=None,
        moe_runner_backend=backend,
        disable_shared_experts_fusion=True,
        custom_weight_loader=[EXPERT_SLICE_LOADER] if enabled else [],
    )
    monkeypatch.setattr(
        misc, "get_hf_config", lambda _: SimpleNamespace(num_experts=256, hidden_size=3072, moe_intermediate_size=1024)
    )
    monkeypatch.setattr(torch.version, "cuda", "12.9")
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda group: 1)
    monkeypatch.setattr(
        torch.distributed, "all_gather_object", lambda result, value, *, group: result.__setitem__(0, value)
    )
    command = dict(op="probe_nixl", geometry=dict(tp_size=args.tp_size), tp_rank=0, persistent_targets=True)
    loader._control_tp(model, [(json.dumps(command), torch.empty(0))], args, 0, None)
    reply = json.loads(model._relax_weight_sync_reply)["shard_p2p_nixl_capabilities"][0]
    assert reply["layout"]["moe_runner_backend"] == "triton"
    assert DirectExpertLayout.from_receiver(reply["layout"]).supports_expert_slices is enabled


@pytest.mark.parametrize("transport_error", [False, True])
def test_native_worker_entry_agreement_requires_completed_collective(monkeypatch, transport_error):
    import torch.distributed as dist

    from relax.backends.sglang.p2p_weight_lease import _worker_control_finished
    from relax.distributed.checkpoint_service.backends.shard_p2p.lifecycle import NativeWeightLease

    group = object()
    guard = SimpleNamespace(
        manager=SimpleNamespace(tp_cpu_group=group), lease=NativeWeightLease(), control_entry_seen=False
    )
    guard.models = [SimpleNamespace(_relax_native_weight_guard=guard)]
    send_timeout = RuntimeError("first TP send timed out")

    def gather(out, record, *, group):
        if transport_error:
            raise send_timeout
        out[:] = [record, (record[0], None)]

    gather_object = Mock(side_effect=gather)
    monkeypatch.setattr(dist, "get_world_size", lambda group: 2)
    monkeypatch.setattr(dist, "all_gather_object", gather_object)
    with pytest.raises(
        RuntimeError, match="first TP send timed out" if transport_error else "TP agreement failed"
    ) as exc:
        _worker_control_finished(guard, ("colocate-shard", {}), "native worker could not decode the request")
    assert guard.lease.failed
    if transport_error:
        assert exc.value is send_timeout
        gather_object.assert_called_once()
    else:
        first, second = [call.args[1] for call in gather_object.call_args_list]
        assert second[0] == (first[0], "native-worker-complete")
        assert "could not decode" in second[1]


@pytest.mark.parametrize("failure", [None, "transport", "model", "device"])
def test_receiver_startup_reuses_transport_and_preserves_failure_boundary(monkeypatch, failure):
    import sys
    from threading import Event, get_ident

    ready, main_thread, agent = Event(), get_ident(), object()
    create = Mock()

    def create_agent():
        create()
        ready.set()
        if failure == "transport":
            raise RuntimeError("transport startup")
        return agent

    class Runner:
        def initialize(self):
            assert failure == "device" or ready.wait(5)
            if failure == "model":
                raise RuntimeError("model startup")
            self.model = torch.nn.Linear(2, 2)
            self.model._prewarm_mhc_kernels = lambda: self.threads.append(get_ident())

    monkeypatch.setitem(sys.modules, "sglang.srt.model_executor.model_runner", SimpleNamespace(ModelRunner=Runner))
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 3)
    set_device = Mock()
    monkeypatch.setattr(torch.cuda, "set_device", set_device)
    monkeypatch.setattr(common, "create_agent", create_agent)
    loader.install_receiver_initialization()
    installed = Runner.initialize
    loader.install_receiver_initialization()
    assert Runner.initialize is installed
    runner = Runner()
    runner.device = "cpu" if failure == "device" else "cuda"
    runner.threads = []
    if failure == "model":
        with pytest.raises(RuntimeError, match="model startup"):
            runner.initialize()
        return
    runner.initialize()
    if failure == "device":
        create.assert_not_called()
        assert runner.threads == []
        return
    assert runner.threads == [main_thread]
    if failure == "transport":
        for _ in range(2):
            with pytest.raises(RuntimeError, match="transport startup"):
                loader._probe_nixl_agent(runner.model)
    else:
        assert loader._probe_nixl_agent(runner.model) is loader._receiver_nixl_agent(runner.model) is agent
        assert "_relax_nixl_probe_future" not in vars(runner.model)
    create.assert_called_once()
    set_device.assert_called_once_with(3)
    assert "_relax_expert_ipc" not in vars(runner.model)


@pytest.mark.parametrize("enabled", [False, True])
def test_consistency_policy_accepts_the_selected_numerical_audit(monkeypatch, enabled):
    monkeypatch.setenv("RELAX_P2P_WEIGHT_CONSISTENCY_CHECK", str(int(enabled)))
    common.validate_consistency_check(dict(consistency_check=enabled, audit_native=enabled), audit_key="audit_native")


def test_cold_plan_registers_only_remote_span_and_reuses_it(monkeypatch):
    monkeypatch.setenv("RELAX_COLOCATE_PINNED_NIXL_TARGET", "1")
    original_empty = torch.empty
    monkeypatch.setattr(torch, "empty", lambda *a, **kw: original_empty(*a, **(kw | {"pin_memory": False})))
    agent = SimpleNamespace(get_reg_descs=lambda values: values, register_memory=Mock(), deregister_memory=Mock())
    state = dict(
        nixl_agent=agent,
        delivery=dict(routes=dict(nixl=["r"])),
        writes=[dict(id="r", param="w", dst_offset=8, rows=2, dst_stride=8, row_bytes=4)],
    )
    loader._prefetch_pinned_targets(state, {"w": torch.empty(12, dtype=torch.bfloat16)})
    pinned = loader._finish_pinned_prefetch(state)
    assert pinned.sizes == {"w": 24} and pinned.spans == {"w": (8, 20)}
    assert pinned.buffer.numel() == 12
    assert loader._finish_pinned_prefetch(state) is pinned
    agent.register_memory.assert_called_once()
    assert "pinned_future" not in state and "pinned_pool" not in state
    pinned.registration.release()
    agent.deregister_memory.assert_called_once()


def test_tp_replica_registers_only_the_assigned_remote_output_chunk(monkeypatch):
    monkeypatch.setenv("RELAX_COLOCATE_PINNED_NIXL_TARGET", "1")
    original_empty = torch.empty
    monkeypatch.setattr(torch, "empty", lambda *a, **kw: original_empty(*a, **(kw | {"pin_memory": False})))
    agent = SimpleNamespace(get_reg_descs=lambda values: values, register_memory=Mock(), deregister_memory=Mock())
    state = dict(
        nixl_agent=agent,
        tp_rank=1,
        delivery=dict(routes=dict(nixl=["r"])),
        writes=[
            dict(
                id="r",
                source="source",
                source_rank=0,
                param="w",
                src_offset=0,
                src_stride=8,
                dst_offset=0,
                rows=4,
                dst_stride=8,
                row_bytes=8,
            )
        ],
        tp_replicas=dict(members=[{}, {}], layouts=dict(w=dict(nbytes=32)), parameters=["w"]),
    )
    loader._prefetch_pinned_targets(state, {"w": torch.empty(4, 4, dtype=torch.bfloat16)})
    pinned = loader._finish_pinned_prefetch(state)
    assert pinned.sizes == {"w": 32} and pinned.spans == {"w": (16, 32)}
    assert pinned.buffer.numel() == 16
    pinned.registration.release()
    agent.deregister_memory.assert_called_once()


@pytest.mark.parametrize("enabled,remote", [(False, True), (True, False)])
def test_unused_cold_plan_does_not_start_allocation(monkeypatch, enabled, remote):
    monkeypatch.setenv("RELAX_COLOCATE_PINNED_NIXL_TARGET", str(int(enabled)))
    state = dict(nixl_agent=object(), delivery=dict(routes=dict(nixl=["remote"] if remote else [])))
    loader._prefetch_pinned_targets(state, {})
    assert "pinned_future" not in state and "pinned_pool" not in state


def test_resident_gpu_targets_do_not_allocate_pinned_memory(monkeypatch):
    monkeypatch.setenv("RELAX_COLOCATE_PINNED_NIXL_TARGET", "1")
    from relax.distributed.checkpoint_service.backends.shard_p2p.remote import transport

    writer = Mock()
    monkeypatch.setattr(transport, "PersistentPinnedTargets", writer)
    state = dict(
        persistent_targets=True,
        nixl_agent=object(),
        delivery=dict(routes=dict(nixl=["remote"])),
        writes=[dict(id="remote", param="weight", dst_offset=0, rows=2, dst_stride=4, row_bytes=4)],
    )
    loader._prefetch_pinned_targets(state, {"weight": torch.empty(2, 2, dtype=torch.bfloat16)})
    assert "pinned_future" not in state and "pinned_targets" not in state
    writer.assert_not_called()


@pytest.mark.parametrize("busy,release_fails", [(False, False), (True, False), (False, True)])
def test_persistent_registration_retires_only_after_completion(busy, release_fails):
    registration = Mock()
    if release_fails:
        registration.release.side_effect = RuntimeError("deregister")
    model = torch.nn.Linear(2, 2)
    model._relax_expert_ipc = dict(
        persistent_targets=True,
        failed=False,
        epoch=object() if busy else None,
        persistent_registration=registration,
        persistent_parameters={"weight": model.weight},
        persistent_storage_ids={"weight": model.weight.untyped_storage()._cdata},
    )
    if busy or release_fails:
        with pytest.raises(RuntimeError, match="pending" if busy else "deregister"):
            loader.retire_persistent_targets(model)
        if busy:
            registration.release.assert_not_called()
        assert model._relax_expert_ipc["persistent_registration"] is registration
    else:
        loader.retire_persistent_targets(model)
        loader.retire_persistent_targets(model)
        registration.release.assert_called_once()
        assert "persistent_parameters" not in model._relax_expert_ipc


@pytest.mark.parametrize("fails,prepared", [(False, False), (True, False), (False, True), (True, True)])
def test_prefetched_resident_registration_is_drained_before_retirement(fails, prepared):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Event

    from relax.distributed.checkpoint_service.backends.shard_p2p.remote import transport

    entered, proceed, retiring = Event(), Event(), Event()
    model = torch.nn.Linear(2, 2, bias=False)

    def register(value):
        entered.set()
        assert proceed.wait(5)

    descriptor = object()
    agent = SimpleNamespace(
        get_reg_descs=lambda values: descriptor,
        register_memory=Mock(side_effect=register),
        deregister_memory=Mock(side_effect=RuntimeError("deregister") if fails else None),
    )
    state = dict(
        persistent_targets=True,
        failed=False,
        epoch=None,
        nixl_agent=agent,
        delivery=dict(routes=dict(nixl=["remote"])),
        writes=[dict(id="remote", param="weight")],
    )
    before = len(transport._FAILED_REGISTRATIONS)
    try:
        if prepared:
            model._relax_nixl_probe_agent = agent
            model._relax_expert_delivery = dict(delivery=state["delivery"])
            command = dict(
                persistent_targets=True,
                writes=state["writes"],
                geometry=dict(expert_enabled=False, ordinary_targets=dict(weight=(2, 2))),
            )
            loader._prepare_persistent_targets(model, command, model._relax_expert_delivery)
            state = model._relax_persistent_target_prefetch["state"]
        else:
            model._relax_expert_ipc = state
            loader._prefetch_persistent_targets(state, dict(model.named_parameters()))
        registration = state["persistent_registration"]
        assert entered.wait(5) and "persistent_agent_metadata" not in state
        wait_for_registration = registration._registration_future
        wait = wait_for_registration.result

        def wait_at_registration(*args, **kwargs):
            retiring.set()
            return wait(*args, **kwargs)

        with ThreadPoolExecutor(max_workers=1) as pool:
            with patch.object(wait_for_registration, "result", side_effect=wait_at_registration):
                future = pool.submit(loader.retire_persistent_targets, model)
                assert retiring.wait(5) and not future.done()
                agent.deregister_memory.assert_not_called()
                proceed.set()
            if fails:
                with pytest.raises(RuntimeError, match="deregister"):
                    future.result(timeout=5)
                assert state["persistent_registration"] is registration and registration.failed
            else:
                future.result(timeout=5)
                assert registration.released
                assert not hasattr(model, "_relax_persistent_target_prefetch")
                assert prepared or "persistent_registration" not in state
        agent.register_memory.assert_called_once_with(descriptor)
        agent.deregister_memory.assert_called_once_with(descriptor)
    finally:
        proceed.set()
        del transport._FAILED_REGISTRATIONS[before:]


@pytest.mark.parametrize("replace_parameter", [False, True])
def test_prepared_resident_registration_adopts_only_original_storage_without_waiting(replace_parameter):
    from threading import Event

    entered, proceed = Event(), Event()
    model = torch.nn.Linear(2, 2, bias=False)

    def register(value):
        entered.set()
        assert proceed.wait(5)

    agent = SimpleNamespace(
        get_reg_descs=lambda values: values,
        register_memory=Mock(side_effect=register),
        deregister_memory=Mock(),
    )
    model._relax_nixl_probe_agent = agent
    binding = model._relax_expert_delivery = dict(delivery=dict(routes=dict(nixl=["remote"])))
    command = dict(
        persistent_targets=True,
        writes=[dict(id="remote", param="weight")],
        geometry=dict(expert_enabled=False, ordinary_targets=dict(weight=(2, 2))),
    )
    try:
        loader._prepare_persistent_targets(model, command, binding)
        pending = model._relax_persistent_target_prefetch
        registration = pending["state"]["persistent_registration"]
        assert entered.wait(5)
        state = dict(persistent_targets=True, nixl_agent=agent, failed=False, epoch=None)
        if replace_parameter:
            model.weight = torch.nn.Parameter(model.weight.detach().clone())
            with pytest.raises(RuntimeError, match="accepted native seed"):
                loader._adopt_persistent_targets(model, state, dict(model.named_parameters()))
            assert model._relax_persistent_target_prefetch is pending
        else:
            assert loader._adopt_persistent_targets(model, state, dict(model.named_parameters()))
            assert state["persistent_registration"] is registration
            assert not hasattr(model, "_relax_persistent_target_prefetch")
            model._relax_expert_ipc = state
        assert not registration._registration_future.done()
    finally:
        proceed.set()
        loader.retire_persistent_targets(model)
    agent.register_memory.assert_called_once()
    agent.deregister_memory.assert_called_once()


def test_replica_completion_requires_matching_receipts_and_gpu_completion():
    from relax.backends.sglang.replica_weight_sync import validate_replica_completion
    from relax.distributed.checkpoint_service.backends.shard_p2p.plan import fingerprint

    state = dict(replicas={}, epoch=dict(replica_ready=dict(receipt_hash=fingerprint([]))))
    with pytest.raises(RuntimeError, match="GPU collective"):
        validate_replica_completion(state, [])
    state["epoch"]["replica_complete"] = "completed"
    validate_replica_completion(state, [])
    with pytest.raises(RuntimeError, match="GPU collective"):
        validate_replica_completion(state, [dict(version=2)])
