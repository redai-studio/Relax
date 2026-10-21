# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""A rollout weight sync confirms only replicas every PP source reached."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest


try:
    from relax.distributed.checkpoint_service.client import engine as client_module

    HAS_DEPS = True
except ImportError:
    HAS_DEPS = False


pytestmark = pytest.mark.skipif(not HAS_DEPS, reason="Missing DCS client dependencies")


def _client(rank: int, recipients: set[str] | None, lora: bool = False):
    client = object.__new__(client_module.CheckpointEngineClient)
    client.role_info = SimpleNamespace(rank=rank)
    client.coordinator_url = "http://dcs.test"
    response = MagicMock()
    response.json.return_value = {}
    client._http_client = SimpleNamespace(get=AsyncMock(return_value=response))
    topology = {index: {"metadata": {"actor_id": actor}} for index, actor in enumerate(sorted(recipients or ()))}
    client._backend = SimpleNamespace(
        _lora_adapter_mode=lora,
        _is_pp_src_rank=recipients is not None,
        rollout_topology=topology,
        update_weights_for_rollout=MagicMock(),
        init_process_group_for_rollout=MagicMock(),
    )
    return client


@pytest.mark.parametrize("rank", [0, 1])
def test_rollout_sync_returns_recipients_common_to_every_pp_source(monkeypatch, rank) -> None:
    # Two PP sources pruned independently; rank 1 is not a PP source.
    gathered = [{"seed", "recovered"}, None, {"seed"}]
    monkeypatch.setattr(client_module, "get_gloo_group", lambda: "gloo")
    monkeypatch.setattr(client_module.dist, "get_world_size", lambda group: len(gathered))

    def all_gather(output, value, group):
        assert group == "gloo" and value == gathered[rank]
        output[:] = gathered

    monkeypatch.setattr(client_module.dist, "all_gather_object", all_gather)

    result = asyncio.run(_client(rank, gathered[rank]).update_weights_for_rollout())

    assert result == (["seed"] if rank == 0 else [])


@pytest.mark.parametrize("lora,actor_fwd_only", [(True, False), (False, True)])
def test_rollout_sync_without_full_confirmation_publishes_nothing(monkeypatch, lora, actor_fwd_only) -> None:
    gather = MagicMock()
    monkeypatch.setattr(client_module.dist, "all_gather_object", gather)

    result = asyncio.run(_client(0, {"seed"}, lora=lora).update_weights_for_rollout(actor_fwd_only=actor_fwd_only))

    assert result == []
    gather.assert_not_called()
