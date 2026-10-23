# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""An idle transport reset must preserve the unrelated training/runtime
state."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from relax.distributed.checkpoint_service.backends import device_direct
from relax.distributed.checkpoint_service.backends.device_direct import DeviceDirectBackend
from relax.distributed.checkpoint_service.client.engine import CheckpointEngineClient


def _backend() -> DeviceDirectBackend:
    backend = object.__new__(DeviceDirectBackend)
    backend.role_info = SimpleNamespace(role_name="actor")
    backend._model_update_groups = object()
    backend._model_update_groups_for_actor_fwd_ref = object()
    backend.rollout_engines = {0: object(), 1: object()}
    backend.rollout_topology = {"0": {"ip": "old"}}
    backend._rollout_topology_signature = frozenset({"old"})
    backend._lora_adapter_mode = False
    backend.model = object()
    backend.weight_version = 12
    return backend


@pytest.mark.parametrize("role", ["rollout", "actor_fwd"])
def test_reset_destroys_only_the_requested_transport(monkeypatch, role):
    backend = _backend()
    before = vars(backend).copy()
    destroy, kill = Mock(), Mock()
    monkeypatch.setattr(device_direct.dist, "destroy_process_group", destroy)
    monkeypatch.setattr(device_direct.ray, "kill", kill)

    backend.reset_inference_transport(role)

    assert backend.model is before["model"]
    assert backend.weight_version == 12
    if role == "rollout":
        destroy.assert_called_once_with(before["_model_update_groups"])
        assert backend._model_update_groups is None
        assert backend._model_update_groups_for_actor_fwd_ref is before["_model_update_groups_for_actor_fwd_ref"]
        assert kill.call_count == 2
        assert all(call.kwargs == {"no_restart": True} for call in kill.call_args_list)
        assert backend.rollout_engines == backend.rollout_topology == {}
        assert backend._rollout_topology_signature is None
    else:
        destroy.assert_called_once_with(before["_model_update_groups_for_actor_fwd_ref"])
        assert backend._model_update_groups_for_actor_fwd_ref is None
        assert backend._model_update_groups is before["_model_update_groups"]
        assert backend.rollout_engines is before["rollout_engines"]
        assert backend.rollout_topology is before["rollout_topology"]
        assert backend._rollout_topology_signature is before["_rollout_topology_signature"]
        kill.assert_not_called()


@pytest.mark.parametrize("role", ["rollout", "actor_fwd"])
def test_reset_without_sender_group_never_destroys_default_group(monkeypatch, role):
    backend = _backend()
    backend._model_update_groups = backend._model_update_groups_for_actor_fwd_ref = None
    backend.rollout_engines = {}
    destroy = Mock()
    monkeypatch.setattr(device_direct.dist, "destroy_process_group", destroy)
    backend.reset_inference_transport(role)
    destroy.assert_not_called()


@pytest.mark.parametrize("role", ["rollout", "actor_fwd"])
def test_destroy_failure_propagates_without_clearing_group(monkeypatch, role):
    backend = _backend()
    attribute = "_model_update_groups" if role == "rollout" else "_model_update_groups_for_actor_fwd_ref"
    group = getattr(backend, attribute)
    monkeypatch.setattr(device_direct.dist, "destroy_process_group", Mock(side_effect=RuntimeError("destroy failed")))
    with pytest.raises(RuntimeError, match="destroy failed"):
        backend.reset_inference_transport(role)
    assert getattr(backend, attribute) is group


def test_rollout_proxy_failure_is_not_reported_as_clean_reset(monkeypatch):
    backend = _backend()
    monkeypatch.setattr(device_direct.dist, "destroy_process_group", Mock())
    monkeypatch.setattr(device_direct.ray, "kill", Mock(side_effect=RuntimeError("proxy cleanup failed")))
    with pytest.raises(RuntimeError, match="proxy cleanup failed"):
        backend.reset_inference_transport("rollout")
    assert backend.rollout_engines
    assert backend._rollout_topology_signature is not None


def test_reset_allows_original_actor_fwd_group_initializer_to_rebuild(monkeypatch):
    backend = _backend()
    backend._megatron = SimpleNamespace(
        mpu=SimpleNamespace(
            get_data_parallel_rank=lambda **_: 0,
            get_tensor_model_parallel_rank=lambda: 0,
            get_pipeline_model_parallel_rank=lambda: 0,
        )
    )
    backend.backend_type = "gloo"
    init = Mock(return_value=object())
    monkeypatch.setattr(device_direct.dist, "destroy_process_group", Mock())
    monkeypatch.setattr(device_direct, "init_process_group", init)
    topology = {"world_size": 3, "pp_groups": {"update_actor_pp_0": "tcp://127.0.0.1:12345"}}

    backend.init_process_groups_for_actor_fwd_ref(topology)
    init.assert_not_called()
    backend.reset_inference_transport("actor_fwd")
    backend.init_process_groups_for_actor_fwd_ref(topology)
    assert backend._model_update_groups_for_actor_fwd_ref is init.return_value
    assert init.call_args.kwargs["world_size"] == 3
    assert init.call_args.kwargs["group_name"] == "update_actor_pp_0"


@pytest.mark.parametrize("role,rank,clear_count", [("actor_fwd", 0, 1), ("actor_fwd", 1, 0), ("rollout", 0, 0)])
def test_client_clears_metadata_before_reset_only_on_forward_rank_zero(role, rank, clear_count):
    client = object.__new__(CheckpointEngineClient)
    client.role_info = SimpleNamespace(role_name="actor", rank=rank)
    client.coordinator_url = "http://coordinator"
    events = []
    client._http_client = SimpleNamespace(get=AsyncMock(return_value=Mock()))
    client._http_client.get.return_value.raise_for_status.side_effect = lambda: events.append("cleared")
    client._backend = SimpleNamespace(reset_inference_transport=lambda _: events.append("reset"))

    asyncio.run(client.reset_inference_transport(role))

    assert events == (["cleared"] if clear_count else []) + ["reset"]
    assert client._http_client.get.await_count == clear_count
    if clear_count:
        client._http_client.get.assert_awaited_once_with("http://coordinator/clear_weight_meta")


def test_metadata_clear_failure_does_not_proceed_to_reset():
    client = object.__new__(CheckpointEngineClient)
    client.role_info = SimpleNamespace(role_name="actor", rank=0)
    client.coordinator_url = "http://coordinator"
    client._http_client = SimpleNamespace(get=AsyncMock(side_effect=RuntimeError("clear failed")))
    client._backend = Mock()
    with pytest.raises(RuntimeError, match="clear failed"):
        asyncio.run(client.reset_inference_transport("actor_fwd"))
    client._backend.reset_inference_transport.assert_not_called()


@pytest.mark.parametrize("role", ["actor", "reference", "unknown"])
def test_reset_rejects_other_transports_before_mutation(role):
    backend = _backend()
    before = vars(backend).copy()
    with pytest.raises(ValueError, match="Unsupported inference transport"):
        backend.reset_inference_transport(role)
    assert vars(backend) == before


def test_receiver_cannot_reset_sender_transport():
    backend = _backend()
    backend.role_info.role_name = "actor_fwd"
    with pytest.raises(RuntimeError, match="Only the Actor sender"):
        backend.reset_inference_transport("actor_fwd")


def test_rollout_adapter_mode_rejects_recovery_without_losing_sync_cache():
    backend = _backend()
    backend._lora_adapter_mode = True
    before = vars(backend).copy()
    with pytest.raises(NotImplementedError, match="LoRA adapter"):
        backend.reset_inference_transport("rollout")
    assert vars(backend) == before
