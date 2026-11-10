# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Real scheduler/worker dispatch and bounded two-rank failure agreement.

CPU registrations stand in for NIXL allocations here; GPU DONE/fences and the
native receiver's layout/receipt rules have separate coverage.
"""

import json
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from relax.backends.sglang import p2p_weight_lease as protocol
from relax.distributed.checkpoint_service.backends.shard_p2p.remote.common import (
    NATIVE_LEASE_PROTOCOL,
    REMOTE_SHARD_LOADER,
    layout_of,
    serialize_control,
)
from relax.distributed.checkpoint_service.backends.shard_p2p.remote.transport import RegisteredMemory


def _manager(rank):
    from sglang.srt.managers.scheduler_components.weight_updater import SchedulerWeightUpdaterManager
    from sglang.srt.managers.tp_worker import BaseTpWorker
    from sglang.srt.model_executor.model_runner_components.weight_updater import WeightUpdater

    class Manager(SchedulerWeightUpdaterManager):
        pass

    class Worker:
        update_weights_from_tensor = BaseTpWorker.update_weights_from_tensor
        _deserialize_own_rank = BaseTpWorker._deserialize_own_rank

        def __setattr__(self, name, value):
            if name == "update_weights_from_tensor" and getattr(self, "reject_wrapper", False):
                raise RuntimeError("worker wrapper installation failed")
            object.__setattr__(self, name, value)

    model = torch.nn.Linear(2, 2, bias=False)
    runner = SimpleNamespace(model=model, server_args=SimpleNamespace(weight_cache_mode="off"))
    runner.weight_updater = WeightUpdater(
        tp_rank=rank,
        device="cpu",
        gpu_id=0,
        model_config=None,
        custom_weight_loaders={REMOTE_SHARD_LOADER: None},
        get_model=lambda: model,
        update_model_fields=lambda **kwargs: None,
        recapture_cuda_graph=lambda: None,
        get_model_runner=lambda: runner,
    )
    worker = Worker()
    worker.model_runner, worker.ps = runner, SimpleNamespace(tp_rank=rank)
    __import__(
        "relax.backends.sglang.weight_sync_guard", fromlist=["install_native_weight_guard"]
    ).install_native_weight_guard(Manager, remote=True)
    manager = Manager(
        tp_worker=worker,
        draft_worker=None,
        tp_cpu_group=dist.group.WORLD,
        memory_saver_adapter=SimpleNamespace(),
        flush_cache=lambda **kwargs: True,
        is_fully_idle=lambda: True,
    )
    return manager, model


def _request(command):
    from sglang.srt.managers.io_struct import UpdateWeightsFromTensorReqInput

    blob = serialize_control(command, torch.empty(0))
    return UpdateWeightsFromTensorReqInput(
        serialized_named_tensors=[blob, blob], load_format=REMOTE_SHARD_LOADER, flush_cache=False
    )


def _cpu_control(model, weights):
    def apply(command):
        if command["op"] == "capability":
            from relax.backends.sglang.p2p_weight_loader import _apply_control

            args = SimpleNamespace(
                tp_size=2,
                pp_size=1,
                dp_size=1,
                ep_size=1,
                quantization=None,
                enable_lora=False,
                speculative_algorithm=None,
                enable_dp_attention=False,
            )
            # CPU storage proves dispatch/lease agreement; the GPU probe
            # independently proves the actual native model capability.
            capability = dict(
                checks={"cpu_fixture": model.config.model_type == "fixture-valid"},
                native_class=type(model).__qualname__,
                adapter="cpu-fixture",
                rejection=None,
            )
            with patch("relax.backends.sglang.shard_p2p_reload.receiver_capability", return_value=capability):
                return _apply_control(model, command, args, SimpleNamespace(tp_rank=dist.get_rank(dist.group.WORLD)))
        if command["op"] == "prepare":
            from relax.backends.sglang.shard_p2p_reload import Qwen35ReloadAdapter

            params = dict(model.named_parameters())
            layouts = {name: layout_of(value) for name, value in params.items()}
            owner = protocol.P2PWeightLease(model, command, "generation", params, layouts)
            agent = SimpleNamespace(get_reg_descs=lambda values: tuple(values), register_memory=lambda values: None)
            registration = RegisteredMemory(agent, params)
            registration.register(batched=False)
            state = dict(
                registration=registration,
                native_reload=Qwen35ReloadAdapter(type(model)).bind(model),
                active=True,
                version=command["version"],
                directory=command["directory"],
                key=f"0-{dist.get_rank(dist.group.WORLD)}",
            )
            model._relax_p2p_state = state
            owner.bind(state)
        elif command["op"] == "plan":
            model._relax_p2p_state["plan"] = "checked CPU fixture"
        elif command["op"] == "commit":
            return {"version": command["version"], "cpu_fixture": True}
        else:
            raise ValueError(command["op"])

    protocol.run_control(model, weights, REMOTE_SHARD_LOADER, apply)


def _native_worker(rank, rendezvous, directory):
    from sglang.srt.model_executor.model_runner_components import weight_updater

    # The real sender imports Megatron's safe storage loader; the real receiver
    # installs SG's CUDA reducer. This process must tolerate both together.
    __import__("megatron.core")
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2, timeout=timedelta(seconds=30))
    cases = []
    try:
        with patch.object(weight_updater, "dynamic_import", return_value=_cpu_control):
            for fault in (
                None,
                "deserialize",
                "worker_false",
                "install",
                "flush",
                "metrics",
                "device",
                "publish",
                "busy",
                "poison",
            ):
                manager, model = _manager(rank)
                root = Path(directory) / str(fault)
                root.mkdir(mode=0o700, exist_ok=True)
                if fault is None:
                    for family, supported in (("fixture-valid", True), ("fixture-unsupported", False)):
                        model.config = SimpleNamespace(model_type=family)
                        capability = dict(
                            op="capability", engine="0", version=0, directory=str(root), challenge=family
                        )
                        assert manager.update_weights_from_tensor(_request(capability)).success
                        actual = json.loads((root / f"{rank}.json").read_text())
                        assert actual["supported"] is supported and actual["challenge"] == family
                        assert not model._relax_native_weight_guard.lease.failed
                        assert model._relax_native_weight_guard.lease.active is None
                command = dict(
                    op="prepare",
                    engine="0",
                    version=2,
                    directory=str(root),
                    audit=False,
                    native_weight_lease_protocol=NATIVE_LEASE_PROTOCOL,
                    weight_update_session=False,
                )
                request = _request(command)
                worker = manager.tp_worker
                if rank == 0:
                    if fault == "deserialize":
                        worker._deserialize_own_rank = lambda payload: (_ for _ in ()).throw(
                            ValueError("decode failed")
                        )
                    elif fault == "worker_false":
                        worker.update_weights_from_tensor = lambda request: (False, "native cache rejected")
                    elif fault == "install":
                        worker.reject_wrapper = True
                    elif fault == "flush":
                        request.flush_cache = True
                    elif fault == "metrics":
                        manager.metrics_collector = SimpleNamespace(
                            observe_weight_load=lambda *args: (_ for _ in ()).throw(RuntimeError("metrics failed"))
                        )
                original_unwrap = weight_updater._unwrap_tensor

                def unwrap(tensor, tp_rank, device):
                    return original_unwrap(tensor, tp_rank, "meta" if fault == "device" and rank == 0 else device)

                with patch.object(weight_updater, "_unwrap_tensor", side_effect=unwrap):
                    failure = fault in ("deserialize", "worker_false", "install", "flush", "metrics", "device")
                    if failure:
                        with pytest.raises(RuntimeError):
                            manager.update_weights_from_tensor(request)
                    else:
                        assert manager.update_weights_from_tensor(request).success
                guard = worker._relax_native_weight_guard
                if not failure:
                    owner = guard.remote_owner
                    assert owner is not None and guard.lease.active == owner.identity
                    if fault == "poison":
                        audit_directory = root / "audit"
                        audit_directory.mkdir(mode=0o700, exist_ok=True)
                        model._relax_broadcast_audit = dict(
                            active=True,
                            deferred=True,
                            poison_complete=False,
                            version=2,
                            directory=str(audit_directory),
                            p2p_directory=str(root),
                        )
                        if rank == 0:
                            guard.lease.pending.clear()
                        poison = _request(
                            dict(command, op="poison", directory=str(audit_directory), p2p_directory=str(root))
                        )
                        poison.load_format = protocol.AUDIT_LOADER
                        with pytest.raises(RuntimeError, match="allocation lease changed"):
                            manager.update_weights_from_tensor(poison)
                    elif fault == "busy":
                        with pytest.raises(RuntimeError, match="lifetime guard rejected"):
                            manager.release_memory_occupation(SimpleNamespace())
                    else:
                        command["op"] = "plan"
                        assert manager.update_weights_from_tensor(_request(command)).success
                        command["op"] = "commit"
                        original_write = protocol.atomic_json

                        def publish(path, value):
                            if fault == "publish" and rank == 0:
                                raise OSError("verified write failed")
                            return original_write(path, value)

                        with patch.object(protocol, "atomic_json", side_effect=publish):
                            if fault == "publish":
                                with pytest.raises(RuntimeError, match="verified write failed"):
                                    manager.update_weights_from_tensor(_request(command))
                            else:
                                assert manager.update_weights_from_tensor(_request(command)).success
                                assert guard.remote_owner is None and guard.lease.active is None
                                assert model._relax_p2p_state["completed_version"] == 2
                if fault is not None:
                    assert guard.lease.failed
                    with pytest.raises(RuntimeError, match="lifetime guard rejected"):
                        manager.release_memory_occupation(SimpleNamespace())
                    if fault in ("publish", "busy", "poison"):
                        assert guard.remote_owner is not None and model._relax_p2p_state["active"]
                # A fresh real collective detects sequence mismatches immediately.
                healthy = torch.ones(1)
                dist.all_reduce(healthy, group=dist.group.WORLD)
                assert healthy.item() == 2
                cases.append(str(fault))
            from relax.distributed.checkpoint_service.backends.shard_p2p.remote.moe_qualification import (
                qualify_before_native_reference,
            )

            for supported in (True, False):
                backend = SimpleNamespace(
                    args=SimpleNamespace(
                        use_shard_p2p_weight_sync=True,
                        true_on_policy_mode=True,
                        tensor_model_parallel_size=1,
                        expert_model_parallel_size=2,
                        expert_tensor_parallel_size=1,
                        rollout_num_gpus_per_engine=1,
                        context_parallel_size=1,
                        pipeline_model_parallel_size=1,
                        num_experts=2,
                        update_weight_buffer_size=1024,
                    ),
                    _lora_enabled=False,
                    _use_bridge=True,
                    quantization_config=None,
                )
                if rank == 0:
                    backend.rollout_topology = {
                        "0": dict(
                            metadata=dict(
                                weight_update_session=False,
                                native_weight_lease_protocol=NATIVE_LEASE_PROTOCOL if supported else None,
                            )
                        )
                    }
                assert qualify_before_native_reference(backend, dist.group.WORLD) is supported
                if supported:
                    backend.args.true_on_policy_mode = False
                    assert qualify_before_native_reference(backend, dist.group.WORLD) is False
                    backend.args.true_on_policy_mode = True
                    backend.args.pipeline_model_parallel_size = 2
                    with patch(
                        "relax.distributed.checkpoint_service.backends.shard_p2p.remote.moe_qualification.socket.gethostname",
                        return_value=f"node-{rank}",
                    ):
                        assert qualify_before_native_reference(backend, dist.group.WORLD) is False
                    backend.args.pipeline_model_parallel_size = 1
                    with patch(
                        "relax.distributed.checkpoint_service.backends.shard_p2p.remote.moe_qualification.socket.gethostname",
                        return_value=f"node-{rank}",
                    ):
                        assert qualify_before_native_reference(backend, dist.group.WORLD) is False
                    backend.args.use_shard_p2p_weight_sync = False
                    assert qualify_before_native_reference(backend, dist.group.WORLD) is False
                    backend.args.use_shard_p2p_weight_sync = rank == 0
                    with pytest.raises(RuntimeError, match="disagree"):
                        qualify_before_native_reference(backend, dist.group.WORLD)
                    healthy = torch.ones(1)
                    dist.all_reduce(healthy, group=dist.group.WORLD)
                    assert healthy.item() == 2
        (Path(directory) / f"rank-{rank}.json").write_text(json.dumps(cases))
    finally:
        dist.destroy_process_group()


def test_actual_native_worker_and_two_rank_failures_do_not_strand_scheduler_barrier(tmp_path):
    if not protocol.native_worker_dispatch_supported():
        from relax.backends.sglang.sglang_engine import SGLangEngine

        engine = SimpleNamespace(
            args=SimpleNamespace(use_shard_p2p_weight_sync=True, rollout_external=False),
            _make_request=lambda *args, **kwargs: pytest.fail("Unsupported native dispatch sent a control RPC"),
        )
        assert SGLangEngine._remote_weight_capabilities(engine) == {}
        return
    mp.spawn(_native_worker, args=(f"file://{tmp_path}/rdzv", str(tmp_path)), nprocs=2, join=True)
    rows = [json.loads(path.read_text()) for path in tmp_path.glob("rank-*.json")]
    assert len(rows) == 2 and rows[0] == rows[1] and len(rows[0]) == 10


def test_control_decode_distinguishes_serialized_cpu_and_native_device_conversion():
    from sglang.srt.model_executor.model_runner_components.weight_updater import _unwrap_tensor

    payload = [(json.dumps(dict(engine=0, version=1, directory="/tmp/session")), torch.empty(0))]
    command = protocol.decode_control(payload, REMOTE_SHARD_LOADER)
    assert command["engine"] == "0"
    native = [(payload[0][0], _unwrap_tensor(payload[0][1], tp_rank=0, device="meta"))]
    assert protocol.decode_control(native, REMOTE_SHARD_LOADER, device=torch.device("meta")) == command
    with pytest.raises(ValueError, match="checked worker device"):
        protocol.decode_control(native, REMOTE_SHARD_LOADER)
    with pytest.raises(ValueError, match="empty CPU storage"):
        serialize_control({}, torch.ones(4)[:0])
