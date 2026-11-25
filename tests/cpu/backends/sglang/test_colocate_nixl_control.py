# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from relax.backends.sglang import shard_p2p_weight_loader as loader
from relax.distributed.checkpoint_service.backends.shard_p2p.remote import common


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
