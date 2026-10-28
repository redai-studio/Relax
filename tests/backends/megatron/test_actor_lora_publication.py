# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
import torch

from tests.engine.lora_versions.test_training import ThreadRanks, assert_rank_errors


pytest.importorskip("megatron.core")
from relax.backends.megatron import actor as actor_module
from relax.distributed.checkpoint_service.client.engine import CheckpointEngineClient


@pytest.mark.parametrize("failure", ["none", "acquire", "publish", "release"])
def test_managed_weight_sync_releases_topology_lease_on_all_exit_paths(monkeypatch, failure):
    ranks = ThreadRanks(monkeypatch)
    monkeypatch.setattr(actor_module, "get_gloo_group", lambda: ranks)
    monkeypatch.setattr(actor_module, "print_memory", lambda *args, **kwargs: None)
    monkeypatch.setattr(actor_module, "run", asyncio.run)
    monkeypatch.setattr(actor_module.ray, "get", lambda value, **kwargs: value)
    manager = SimpleNamespace(set_weight_updating=SimpleNamespace(remote=Mock(return_value=True)))
    lock = SimpleNamespace(
        acquire=SimpleNamespace(remote=Mock(return_value=True)),
        release=SimpleNamespace(remote=Mock()),
    )
    if failure == "acquire":
        lock.acquire.remote.side_effect = RuntimeError("acquire failed")
    elif failure == "release":
        lock.release.remote.side_effect = RuntimeError("release failed")
    actors = []
    for _rank in range(2):
        actor = object.__new__(actor_module.MegatronTrainRayActor)
        actor.args = SimpleNamespace(rollout_http_timeout=10)
        actor._managed_lora_publication = True
        actor.rollout_manager = manager
        actor._weight_sync_lock = lock
        actor.checkpoint_engine_client = SimpleNamespace(
            update_weights_for_rollout=AsyncMock(return_value=[]),
            init_process_groups_for_actor_fwd_ref=AsyncMock(),
        )
        if failure == "publish":
            actor.checkpoint_engine_client.update_weights_for_rollout.side_effect = RuntimeError("publish failed")
        actors.append(actor)
    results = ranks.run(
        lambda rank: actor_module.MegatronTrainRayActor.update_weights_fully_async.__wrapped__(
            actors[rank], 1, rollout_only=True
        )
    )
    if failure == "none":
        assert results == [None, None]
    else:
        assert_rank_errors(results, f"{failure} failed")
    assert [call.args for call in manager.set_weight_updating.remote.call_args_list] == [(True,), (False,)]
    assert lock.release.remote.call_count == (0 if failure == "acquire" else 1)


@pytest.mark.parametrize("actor_fwd_absent", [False, True])
def test_managed_health_check_never_enters_legacy_rollout_pause(monkeypatch, actor_fwd_absent):
    ranks = ThreadRanks(monkeypatch)
    monkeypatch.setattr(actor_module, "get_gloo_group", lambda: ranks)

    def reduce(tensor, *, op, group):
        values = ranks.exchange("reduce", tensor.tolist())
        tensor.copy_(torch.tensor(values).max(dim=0).values)

    monkeypatch.setattr(actor_module.dist, "all_reduce", reduce)
    get_url = Mock(return_value="http://actor-fwd")
    request = Mock(return_value=Mock())
    monkeypatch.setattr(actor_module, "get_serve_url", get_url)
    monkeypatch.setattr(actor_module.requests, "get", request)
    actors = []
    for _rank in range(2):
        actor = object.__new__(actor_module.MegatronTrainRayActor)
        actor._managed_lora_publication = True
        actor.args = SimpleNamespace(true_on_policy_mode=actor_fwd_absent, hybrid=False, rollout_http_timeout=10)
        actors.append(actor)
    assert (
        ranks.run(lambda rank: actor_module.MegatronTrainRayActor._check_services_health.__wrapped__(actors[rank]))
        == [(actor_fwd_absent, False)] * 2
    )
    if actor_fwd_absent:
        get_url.assert_not_called()
        request.assert_not_called()
    else:
        get_url.assert_called_once_with("actor_fwd")
        request.assert_called_once_with("http://actor-fwd/get_step", timeout=10)


@pytest.mark.parametrize("failure", ["lookup", "group_init"])
def test_single_rank_actor_fwd_initialization_failure_exits_all_trainers(monkeypatch, failure):
    ranks = ThreadRanks(monkeypatch)
    monkeypatch.setattr(actor_module, "get_gloo_group", lambda: ranks)
    monkeypatch.setattr(actor_module, "print_memory", lambda *args, **kwargs: None)
    monkeypatch.setattr(actor_module, "run", asyncio.run)
    monkeypatch.setattr(actor_module.ray, "get", lambda value, **kwargs: value)
    manager = SimpleNamespace(set_weight_updating=SimpleNamespace(remote=Mock(return_value=True)))
    lock = SimpleNamespace(
        acquire=SimpleNamespace(remote=Mock(return_value=True)), release=SimpleNamespace(remote=Mock())
    )
    actors = []
    for rank in range(2):
        client = object.__new__(CheckpointEngineClient)
        client.args = SimpleNamespace(ref_update_interval=None)
        client.role_info = SimpleNamespace(role_name="actor", rank=rank)
        client.coordinator_url = "http://coordinator"
        client._backend = SimpleNamespace(
            managed_lora_publication_enabled=True, init_process_groups_for_actor_fwd_ref=Mock()
        )
        client._http_client = SimpleNamespace(get=AsyncMock(return_value=Mock()))
        client.update_weights_for_rollout = AsyncMock(return_value=[])
        if rank == 1:
            if failure == "lookup":
                client._http_client.get.side_effect = RuntimeError("one rank failed")
            else:
                client._backend.init_process_groups_for_actor_fwd_ref.side_effect = RuntimeError("one rank failed")
        actor = object.__new__(actor_module.MegatronTrainRayActor)
        actor._managed_lora_publication = True
        actor.rollout_manager, actor._weight_sync_lock = manager, lock
        actor.checkpoint_engine_client = client
        actors.append(actor)
    assert_rank_errors(
        ranks.run(
            lambda rank: actor_module.MegatronTrainRayActor.update_weights_fully_async.__wrapped__(actors[rank], 1)
        ),
        "one rank failed",
    )
    assert all(actor.checkpoint_engine_client.update_weights_for_rollout.call_count == 0 for actor in actors)
    lock.release.remote.assert_called_once()
    assert [call.args for call in manager.set_weight_updating.remote.call_args_list] == [(True,), (False,)]


@pytest.mark.parametrize("managed", [False, True])
def test_manager_assignment_configures_publication_on_every_training_rank(monkeypatch, managed):
    monkeypatch.setattr(actor_module.ray, "get", lambda value, **kwargs: value)
    config = object() if managed else None
    lock = object()
    manager = SimpleNamespace(
        set_train_parallel_config=SimpleNamespace(remote=Mock()),
        get_weight_sync_lock=SimpleNamespace(remote=Mock(return_value=lock)),
        get_lora_training_config=SimpleNamespace(remote=Mock(return_value=config)),
    )
    for rank in (0, 1):
        actor = object.__new__(actor_module.MegatronTrainRayActor)
        actor.args = SimpleNamespace(
            rank=rank,
            debug_rollout_only=False,
            fully_async=True,
            hybrid=False,
            colocate=False,
            offload_rollout=False,
            lora_adapter_mode=True,
        )
        actor.train_parallel_config = {}
        backend = SimpleNamespace(configure_lora_publication=Mock())
        actor.checkpoint_engine_client = SimpleNamespace(backend=backend)
        actor.set_rollout_manager(manager)
        assert actor._managed_lora_publication is managed
        assert actor._weight_sync_lock is lock
        if managed:
            backend.configure_lora_publication.assert_called_once_with(config, manager)
        else:
            backend.configure_lora_publication.assert_not_called()
    manager.set_train_parallel_config.remote.assert_called_once()
