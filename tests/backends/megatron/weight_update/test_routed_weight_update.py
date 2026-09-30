# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Exercise routed IPC error handling without connecting to Ray."""

from unittest.mock import Mock, patch

import pytest


@pytest.fixture
def module():
    from relax.backends.megatron.weight_update import update_weight_from_tensor

    return update_weight_from_tensor


@pytest.mark.parametrize("failure", [False, True, "exception"])
def test_routed_ipc_wait_checks_receiver_results(module, failure):
    updater = object.__new__(module.UpdateWeightFromTensor)
    updater._route_experts = True
    group = object()
    get = (
        Mock(side_effect=RuntimeError("RPC failed"))
        if failure == "exception"
        else Mock(return_value=[{"success": not failure}])
    )

    def collect(errors, local, group):
        errors[:] = [local, None]

    with (
        patch.object(module.ray, "get", get),
        patch.object(module, "get_gloo_group", return_value=group),
        patch.object(module.dist, "all_reduce") as reduce,
        patch.object(module.dist, "get_world_size", return_value=2),
        patch.object(module.dist, "all_gather_object", side_effect=collect),
    ):
        if failure:
            with pytest.raises(RuntimeError, match="Routed weight load failed"):
                updater._wait_for_ipc([object()])
        else:
            updater._wait_for_ipc([object()])
        assert reduce.call_args.kwargs["group"] is group
        assert reduce.call_args.args[0].device.type == "cpu"


def test_routed_ipc_empty_rank_propagates_peer_failure(module):
    updater = object.__new__(module.UpdateWeightFromTensor)
    updater._route_experts = True
    with (
        patch.object(module.ray, "get") as get,
        patch.object(module, "get_gloo_group"),
        patch.object(module.dist, "all_reduce", side_effect=lambda tensor, **kw: tensor.fill_(1)),
        patch.object(module.dist, "get_world_size", return_value=2),
        patch.object(
            module.dist, "all_gather_object", side_effect=lambda errors, local, **kw: errors.__setitem__(1, "bad load")
        ),
    ):
        with pytest.raises(RuntimeError, match="bad load"):
            updater._wait_for_ipc([])
        get.assert_not_called()


def test_legacy_ipc_wait_does_not_add_collectives(module):
    updater = object.__new__(module.UpdateWeightFromTensor)
    updater._route_experts = False
    with patch.object(module.ray, "get") as get, patch.object(module.dist, "all_reduce") as reduce:
        updater._wait_for_ipc([])
        get.assert_not_called()
        updater._wait_for_ipc(["ref"])
        get.assert_called_once_with(["ref"])
        reduce.assert_not_called()


@pytest.mark.parametrize("fail_load", [False, True, "conversion", "dispatch"])
def test_routed_update_publishes_version_only_after_all_loads(module, fail_load):
    from types import SimpleNamespace

    events = []
    engine = Mock()
    for method in (
        "pause_generation",
        "flush_cache",
        "post_process_weights",
        "update_weight_version",
        "continue_generation",
    ):
        getattr(engine, method).remote.side_effect = lambda _method=method, **kw: (
            events.append((_method, kw)) or {"success": True}
        )
    updater = object.__new__(module.UpdateWeightFromTensor)
    updater._route_experts = True
    updater.lora_enabled = updater.lora_merge_mode = False
    updater.rollout_engines, updater.distributed_rollout_engines = [engine], []
    updater.weight_version = 0
    updater.weights_getter = lambda: {}
    completed_reads = []

    def chunks(weights):
        yield []
        if fail_load == "conversion":
            assert completed_reads == [True]
            raise RuntimeError("next conversion failed")
        yield []

    def get(refs):
        if isinstance(refs, list) and any(result.get("payload") for result in refs):
            completed_reads.append(True)
        return refs

    updater._hf_weight_iterator = SimpleNamespace(get_hf_weight_chunks=chunks)

    def send(tensors, *, dispatch_errors=None):
        events.append(("send", {}))
        if fail_load == "dispatch":
            dispatch_errors.append("dispatch failed")
        return [{"success": fail_load is not True, "payload": True}], None

    updater._send_hf_params = send
    with (
        patch.object(module.ray, "get", side_effect=get),
        patch.object(module.dist, "get_rank", return_value=0),
        patch.object(module.dist, "all_reduce"),
        patch.object(module.dist, "barrier"),
        patch.object(module.dist, "get_world_size", return_value=1),
        patch.object(module.dist, "all_gather_object", side_effect=lambda out, local, **kw: out.__setitem__(0, local)),
        patch.object(module, "get_gloo_group"),
        patch.object(module.device_utils, "maybe_backend_barrier_on_weight_chunk"),
    ):
        if fail_load:
            with pytest.raises(
                RuntimeError,
                match="next conversion failed" if fail_load == "conversion" else "Routed weight load failed",
            ):
                updater.update_weights()
            assert sum(name == "send" for name, _ in events) == 1
            assert not engine.update_weight_version.remote.called
            assert not engine.continue_generation.remote.called
            if fail_load == "dispatch":
                assert completed_reads == [True]
        else:
            updater.update_weights()
            assert [name for name, _ in events] == [
                "pause_generation",
                "flush_cache",
                "post_process_weights",
                "send",
                "send",
                "post_process_weights",
                "update_weight_version",
                "continue_generation",
            ]
            assert events[2][1]["restore_weights_before_load"] is True
            assert events[5][1]["post_process_quantization"] is True
            engine.update_weight_version.remote.assert_called_once_with(weight_version="1")


@pytest.mark.parametrize("version", [None, 3])
def test_routed_ipc_payload_omits_uncommitted_version(module, version):
    import torch

    engine = Mock()
    with (
        patch.object(module, "make_current_torch_device", return_value=torch.device("cpu")),
        patch.object(module.MultiprocessingSerializer, "serialize", return_value="test-bucket"),
        patch.object(module.dist, "get_rank", return_value=0),
        patch.object(module.dist, "get_world_size", return_value=1),
        patch.object(
            module.dist,
            "gather_object",
            side_effect=lambda value, object_gather_list, **kw: object_gather_list.__setitem__(0, value),
        ),
    ):
        module._send_to_colocated_engine(
            [("weight", torch.ones(4))],
            ipc_engine=engine,
            ipc_gather_src=0,
            ipc_gather_group=object(),
            weight_version=version,
        )
    kwargs = engine.update_weights_from_tensor.remote.call_args.kwargs
    assert kwargs["serialized_named_tensors"] == ["test-bucket"]
    assert ("weight_version" in kwargs) is (version is not None)
    if version is not None:
        assert kwargs["weight_version"] == "3"


def test_routed_ipc_exception_drains_all_readers_before_raising(module):
    updater = object.__new__(module.UpdateWeightFromTensor)
    updater._route_experts = True
    calls = []

    def get(refs):
        calls.append(refs)
        if isinstance(refs, list) or refs == "failed":
            raise RuntimeError("reader failed")
        return {"success": True}

    with (
        patch.object(module.ray, "get", side_effect=get),
        patch.object(module, "get_gloo_group"),
        patch.object(module.dist, "all_reduce"),
        patch.object(module.dist, "get_world_size", return_value=1),
        patch.object(module.dist, "all_gather_object", side_effect=lambda out, local, **kw: out.__setitem__(0, local)),
    ):
        with pytest.raises(RuntimeError, match="reader failed"):
            updater._wait_for_ipc(["failed", "still-reading"])
    assert calls == [["failed", "still-reading"], "failed", "still-reading"]


@pytest.mark.parametrize("successful_dispatches", [0, 1])
def test_routed_ipc_dispatch_failure_preserves_submitted_readers(module, successful_dispatches):
    import torch

    engine = Mock()
    engine.update_weights_from_tensor.remote.side_effect = ["reader"] * successful_dispatches + [
        RuntimeError("dispatch failed")
    ]
    errors = []
    with (
        patch.object(module, "make_current_torch_device", return_value=torch.device("cpu")),
        patch.object(module.FlattenedTensorBucket, "supports_multi_dtypes", False, create=True),
        patch.object(module.MultiprocessingSerializer, "serialize", return_value="test-bucket"),
        patch.object(module.dist, "get_rank", return_value=0),
        patch.object(module.dist, "get_world_size", return_value=1),
        patch.object(
            module.dist,
            "gather_object",
            side_effect=lambda value, object_gather_list, **kw: object_gather_list.__setitem__(0, value),
        ),
    ):
        refs, buffers = module._send_to_colocated_engine(
            [("packed", torch.ones(4, dtype=torch.uint8)), ("scale", torch.ones(4))],
            ipc_engine=engine,
            ipc_gather_src=0,
            ipc_gather_group=object(),
            weight_version=None,
            dispatch_errors=errors,
        )
    assert refs == ["reader"] * successful_dispatches
    assert len(buffers) == 2
    assert errors == ["dispatch failed"]
