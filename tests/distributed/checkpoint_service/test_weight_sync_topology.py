# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

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
