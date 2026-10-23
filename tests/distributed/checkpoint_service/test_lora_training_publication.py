# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
import torch

from tests.engine.lora_versions.test_training import ThreadRanks, assert_rank_errors


pytest.importorskip("megatron.core")

from relax.distributed.checkpoint_service.backends import device_direct
from relax.distributed.checkpoint_service.client.engine import CheckpointEngineClient
from relax.engine.lora_versions.training import LoRATrainingConfig


@pytest.fixture
def backends(monkeypatch):
    ranks = ThreadRanks(monkeypatch)
    monkeypatch.setattr(device_direct, "get_gloo_group", lambda: ranks)
    monkeypatch.setattr(device_direct.ray, "get", lambda value, **kwargs: value)
    monkeypatch.setattr(device_direct.device_module, "empty_cache", lambda: None)
    workers = []
    for _rank in range(2):
        events = []
        backend = object.__new__(device_direct.DeviceDirectBackend)
        backend.args = SimpleNamespace(num_experts=0, rollout_http_timeout=10)
        backend.weight_version = 0
        backend._use_bridge = backend._lora_merge_mode = backend._is_pp_src_rank = False
        backend._lora_adapter_mode = True
        backend._lora_sync = SimpleNamespace(base_sync_done=False, adapter_loaded=False)
        backend._managed_lora_publisher = SimpleNamespace(
            publish=Mock(
                side_effect=lambda params, events=events: events.append("publish") or {"status": "published"}
            ),
            config=SimpleNamespace(engine_actor_ids=("engine",)),
        )
        backend._megatron = SimpleNamespace(named_params_and_buffers=lambda *args, **kwargs: [])
        backend.model = []
        backend.rollout_engines = {
            0: SimpleNamespace(
                flush_cache=SimpleNamespace(remote=Mock(side_effect=lambda e=events: e.append("flush")))
            )
        }
        backend._batch_request = Mock(side_effect=lambda path, events=events: events.append(path))
        backend._group_name = "test"
        backend._run_expert_pass = Mock(side_effect=lambda **kwargs: None)
        backend.coordinator_url = "http://coordinator"
        backend.http_client = SimpleNamespace(
            post=Mock(side_effect=lambda *a, events=events, **kw: events.append("actor-fwd-end") or Mock())
        )
        backend.events = events
        workers.append(backend)
    return ranks, workers


def test_first_sync_resumes_before_probe_and_later_sync_never_pauses(backends):
    ranks, workers = backends
    assert ranks.run(lambda rank: workers[rank].update_weights_for_rollout(rollout_only=True)) == [None, None]
    assert workers[0].events == ["/pause_generation", "flush", "/continue_generation", "publish"]
    assert workers[1].events == ["publish"]
    assert all(worker._lora_sync.base_sync_done and worker._lora_sync.adapter_loaded for worker in workers)
    for worker in workers:
        worker.events.clear()
        worker._run_expert_pass.reset_mock()
    assert ranks.run(lambda rank: workers[rank].update_weights_for_rollout(rollout_only=True)) == [None, None]
    assert all(worker.events == ["publish"] for worker in workers)
    assert all(worker._run_expert_pass.call_count == 0 for worker in workers)


@pytest.mark.parametrize("actor_fwd_only", [False, True])
def test_later_actor_fwd_updates_keep_completion_marker_without_pausing_rollout(backends, actor_fwd_only):
    ranks, workers = backends
    for worker in workers:
        worker._lora_sync.base_sync_done = True
    assert ranks.run(lambda rank: workers[rank].update_weights_for_rollout(actor_fwd_only=actor_fwd_only)) == [
        None,
        None,
    ]
    assert workers[0].events == (["actor-fwd-end"] if actor_fwd_only else ["actor-fwd-end", "publish"])
    assert workers[1].events == ([] if actor_fwd_only else ["publish"])
    assert all(worker._run_expert_pass.call_count == 1 for worker in workers)


@pytest.mark.parametrize("phase", ["pause", "flush", "resume", "actor_fwd_end"])
def test_rank_zero_control_failure_does_not_strand_other_ranks(backends, phase):
    ranks, workers = backends
    first = workers[0]
    if phase == "flush":
        first.rollout_engines[0].flush_cache.remote.side_effect = RuntimeError("control failed")
    elif phase == "actor_fwd_end":
        first.http_client.post.side_effect = RuntimeError("control failed")
    else:
        failed_path = "/pause_generation" if phase == "pause" else "/continue_generation"

        def request(path):
            if path == failed_path:
                raise RuntimeError("control failed")

        first._batch_request.side_effect = request
    results = ranks.run(lambda rank: workers[rank].update_weights_for_rollout(rollout_only=phase != "actor_fwd_end"))
    assert_rank_errors(results, "control failed")
    assert all(worker._managed_lora_publisher.publish.call_count == 0 for worker in workers)


def test_failed_first_publication_does_not_cause_base_resynchronization(backends):
    ranks, workers = backends
    for worker in workers:
        worker._managed_lora_publisher.publish.side_effect = RuntimeError("lost publication reply")
    assert_rank_errors(
        ranks.run(lambda rank: workers[rank].update_weights_for_rollout(rollout_only=True)), "lost publication"
    )
    assert all(worker._lora_sync.base_sync_done for worker in workers)
    for worker in workers:
        worker.events.clear()
    assert_rank_errors(
        ranks.run(lambda rank: workers[rank].update_weights_for_rollout(rollout_only=True)), "lost publication"
    )
    assert all(not worker.events for worker in workers)


def test_training_configuration_requires_frozen_base(backends, tmp_path):
    _ranks, workers = backends
    backend = workers[0]
    backend._managed_lora_publisher = None
    base = torch.nn.Parameter(torch.ones(1))
    adapter = torch.nn.Parameter(torch.ones(1))
    backend.model = [SimpleNamespace(named_parameters=lambda: [("base.weight", base), ("q.lora_A.weight", adapter)])]
    config = LoRATrainingConfig(str(tmp_path), "base", (1,))
    with pytest.raises(ValueError, match="frozen"):
        backend.configure_lora_publication(config, object())
    base.requires_grad_(False)
    backend.configure_lora_publication(config, object())
    assert backend.managed_lora_publication_enabled


@pytest.mark.parametrize("failure", ["none", "lookup", "pruned"])
def test_client_agrees_on_incomplete_initial_pool_and_skips_later_topology(backends, failure):
    ranks, workers = backends
    clients = []
    topology = {"nodes": {"rollout": {"0": {"metadata": {"actor_id": "engine"}}}}}
    for rank, backend in enumerate(workers):
        backend._is_pp_src_rank = True
        backend.rollout_topology = topology["nodes"]["rollout"]
        backend.init_process_group_for_rollout = Mock()
        backend.update_weights_for_rollout = Mock()
        client = object.__new__(CheckpointEngineClient)
        client._backend = backend
        client.coordinator_url = "http://coordinator"
        response = Mock()
        response.json.return_value = topology
        client._http_client = SimpleNamespace(get=AsyncMock(return_value=response))
        if rank == 1 and failure == "lookup":
            client._http_client.get.side_effect = RuntimeError("lookup failed")
        elif rank == 1 and failure == "pruned":
            backend.rollout_topology = {}
        clients.append(client)
    results = ranks.run(lambda rank: asyncio.run(clients[rank].update_weights_for_rollout(rollout_only=True)))
    if failure != "none":
        assert_rank_errors(results, "LoRA base")
        assert all(worker.update_weights_for_rollout.call_count == 0 for worker in workers)
        return
    assert results == [[], []]
    for client in clients:
        client._backend._lora_sync.base_sync_done = True
        client._http_client.get.reset_mock()
        client._backend.init_process_group_for_rollout.reset_mock()
    assert ranks.run(lambda rank: asyncio.run(clients[rank].update_weights_for_rollout(rollout_only=True))) == [[], []]
    assert all(client._http_client.get.call_count == 0 for client in clients)
    assert all(client._backend.init_process_group_for_rollout.call_count == 0 for client in clients)
