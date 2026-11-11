# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from relax.distributed.checkpoint_service.backends import device_direct
from relax.distributed.checkpoint_service.client import engine as engine_module


@pytest.mark.parametrize("rank", [0, 1])
@pytest.mark.parametrize("fail_fetch", [False, True])
def test_rollout_update_shares_one_topology_snapshot(monkeypatch, rank, fail_fetch):
    group = object()
    data = {"nodes": {"rollout": {"0": {"metadata": {"actor_id": "A"}}}}}
    response = Mock()
    response.json.return_value = data
    client = engine_module.CheckpointEngineClient(
        args=SimpleNamespace(), coordinator_url="http://dcs", role="actor", rank=rank
    )
    client._http_client = SimpleNamespace(get=AsyncMock(return_value=response))
    if fail_fetch:
        client._http_client.get.side_effect = RuntimeError("coordinator unavailable")
    backend = Mock(_lora_adapter_mode=False, _is_pp_src_rank=True, rollout_topology=data["nodes"]["rollout"])
    client._backend = backend
    monkeypatch.setattr(engine_module, "get_gloo_group", lambda: group)
    monkeypatch.setattr(engine_module.dist, "get_rank", lambda **kw: rank)
    monkeypatch.setattr(engine_module.dist, "get_world_size", lambda **kw: 2)

    def broadcast(status, src, group):
        assert src == 0
        if rank == 0:
            assert status[0] == (None, "coordinator unavailable") if fail_fetch else status[0] == (data, None)
        else:
            assert status == [None]
            status[0] = (None, "coordinator unavailable") if fail_fetch else (data, None)

    monkeypatch.setattr(engine_module.dist, "broadcast_object_list", broadcast)
    monkeypatch.setattr(
        engine_module.dist, "all_gather_object", lambda out, ids, **kw: out.__setitem__(slice(None), [{"A"}, {"A"}])
    )
    if fail_fetch:
        with pytest.raises(RuntimeError, match="Failed to fetch rollout topology: coordinator unavailable"):
            asyncio.run(client.update_weights_for_rollout(rollout_only=True))
        backend.update_weights_for_rollout.assert_not_called()
    else:
        assert asyncio.run(client.update_weights_for_rollout(rollout_only=True)) == (["A"] if rank == 0 else [])
        backend.init_process_group_for_rollout.assert_called_once_with(data)
    assert client._http_client.get.await_count == (1 if rank == 0 else 0)


@pytest.fixture
def rollout_backend(monkeypatch):
    backend = object.__new__(device_direct.DeviceDirectBackend)
    backend.weight_version = 0
    backend._use_bridge = backend._lora_merge_mode = backend._lora_adapter_mode = False
    backend._group_name = "pp0"
    backend._is_pp_src_rank = True
    backend.args = SimpleNamespace(num_experts=4)
    backend.quantization_config = None
    backend.rollout_engines = {0: Mock()}
    events = []
    backend._named_params_and_buffers = lambda **kw: iter(())
    backend._run_expert_pass = lambda **kw: events.append("push")
    backend._batch_request = lambda endpoint, payload=None: events.append((endpoint, payload)) or [endpoint]
    backend.rollout_engines[0].flush_cache.remote.side_effect = lambda: events.append("flush")
    monkeypatch.setattr(device_direct.dist, "get_rank", lambda *a, **kw: 0)
    monkeypatch.setattr(device_direct.dist, "barrier", lambda **kw: None)
    monkeypatch.setattr(device_direct.dist, "broadcast_object_list", lambda *a, **kw: None)
    monkeypatch.setattr(device_direct.dist, "all_reduce", lambda *a, **kw: None)
    monkeypatch.setattr(device_direct, "get_gloo_group", lambda: object())
    awaited = []
    monkeypatch.setattr(device_direct.ray, "get", lambda refs: awaited.append(refs))
    monkeypatch.setattr(device_direct.device_module, "empty_cache", lambda: None)
    return backend, events, awaited


@pytest.mark.parametrize("quant_method", [None, "compressed-tensors"])
@pytest.mark.parametrize("actor_fwd_only", [False, True])
def test_dcs_quantized_rollout_restores_and_repacks_before_resume(rollout_backend, quant_method, actor_fwd_only):
    backend, events, awaited = rollout_backend
    backend.quantization_config = {"quant_method": quant_method} if quant_method else None
    backend.update_weights_for_rollout(rollout_only=True, actor_fwd_only=actor_fwd_only)
    if actor_fwd_only:
        assert events == ["push"]
    else:
        expected = [("/pause_generation", None), "flush"]
        if quant_method:
            expected.append(
                ("/post_process_weights", {"restore_weights_before_load": True, "post_process_quantization": False})
            )
        expected.append("push")
        if quant_method:
            expected.append(
                ("/post_process_weights", {"restore_weights_before_load": False, "post_process_quantization": True})
            )
        expected.append(("/continue_generation", None))
        assert events == expected
        assert awaited[-1] == ["/continue_generation"]


def test_quantized_adapter_delta_does_not_restore_or_repack_frozen_base(rollout_backend):
    backend, events, _ = rollout_backend
    backend.quantization_config = {"quant_method": "compressed-tensors"}
    backend._lora_adapter_mode = True
    backend._lora_sync = SimpleNamespace(base_sync_done=False, adapter_loaded=False)
    backend._push_lora_adapter_distributed = lambda **kw: events.append("adapter")
    backend.update_weights_for_rollout(rollout_only=True)
    assert len([e for e in events if isinstance(e, tuple) and e[0] == "/post_process_weights"]) == 2
    assert backend._lora_sync.base_sync_done
    events.clear()
    backend.update_weights_for_rollout(rollout_only=True)
    assert events == [("/pause_generation", None), "flush", "adapter", ("/continue_generation", None)]


@pytest.mark.parametrize("rank", [0, 1])
@pytest.mark.parametrize("endpoint", ["/post_process_weights", "/continue_generation"])
def test_receiver_control_failure_reaches_every_training_rank(rollout_backend, monkeypatch, rank, endpoint):
    backend, _, _ = rollout_backend
    monkeypatch.setattr(device_direct.dist, "get_rank", lambda **kw: rank)
    monkeypatch.setattr(device_direct.ray, "get", Mock(side_effect=RuntimeError("receiver unavailable")))

    def broadcast(status, **kwargs):
        if rank == 0:
            assert status == ["receiver unavailable"]
        else:
            assert status == [None]
            status[0] = "receiver unavailable"

    monkeypatch.setattr(device_direct.dist, "broadcast_object_list", broadcast)
    with pytest.raises(RuntimeError, match=f"Rollout {endpoint} failed: receiver unavailable"):
        backend._rollout_control_request(endpoint)


@pytest.mark.parametrize("generate", [False, True])
def test_rollout_health_can_probe_without_generating(monkeypatch, generate):
    cls = device_direct.RolloutEngine.__ray_metadata__.modified_class
    engine = cls.__new__(cls)
    engine.base_url = "http://engine"
    request = Mock()
    monkeypatch.setattr(device_direct.requests, "get", request)
    assert engine.health(timeout=7, generate=generate)
    request.assert_called_once_with(f"http://engine/{'health_generate' if generate else 'get_server_info'}", timeout=7)
