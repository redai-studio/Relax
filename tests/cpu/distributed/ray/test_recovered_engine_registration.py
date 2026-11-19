# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Recovery publication uses successful weight recipients, not live rank/URL
alone."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from conftest import (
    HAS_DEPS,
    AwaitableValue,
    create_test_manager,
    make_engine_group,
    make_mock_engine,
    make_rollout_server,
)
from ray.core.generated.gcs_pb2 import ActorTableData

from relax.distributed.checkpoint_service.client import engine as client_module
from relax.distributed.checkpoint_service.client.engine import CheckpointEngineClient


@pytest.fixture(autouse=True)
def single_rank_collective(monkeypatch):
    monkeypatch.setattr(client_module, "get_gloo_group", lambda: "gloo")
    monkeypatch.setattr(client_module.dist, "get_rank", lambda **kwargs: 0)
    monkeypatch.setattr(client_module.dist, "get_world_size", lambda **kwargs: 1)
    monkeypatch.setattr(client_module.dist, "broadcast_object_list", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        client_module.dist, "all_gather_object", MagicMock(side_effect=AssertionError("duplicate recipient gather"))
    )


def make_recovery():
    rollout = pytest.importorskip("relax.distributed.ray.rollout", exc_type=ModuleNotFoundError)
    if not HAS_DEPS:
        pytest.skip("Requires rollout runtime dependencies for shared Ray test helpers")
    engine = make_mock_engine(url="http://engine-b:30000")
    engine._actor_id.hex.return_value = "actor-b"
    engine._get_local_state.return_value = ActorTableData.ALIVE
    group = make_engine_group(engines=[engine], rank_offset=4, is_scaled_out=True)
    group.pg = object()
    record = rollout._EngineInitRecord({"url": "http://engine-b:30000", "pid": 42, "node_id": "node"})
    record.pending_router_registration = True
    group.engine_init_records = {0: record}
    manager = create_test_manager(servers={"default": make_rollout_server(engine_groups=[group])})
    manager._training_weight_updating = True
    return manager, group, engine, record


def test_recovered_engine_registers_once_after_successful_sync(patch_ray_get):
    manager, _, engine, record = make_recovery()
    manager.register_recovered_engines(["actor-b"])
    manager.register_recovered_engines(["actor-b"])
    engine.register_to_router.remote.assert_called_once()
    assert not record.pending_router_registration


@pytest.mark.parametrize("reason", ["not_synced", "replacement", "draining", "evicted", "no_lease", "pending"])
def test_recovery_registration_rejects_unsafe_candidates(reason, patch_ray_get):
    manager, group, engine, record = make_recovery()
    from relax.distributed.ray import rollout

    ids = ["actor-b"]
    if reason == "not_synced":
        ids = []
    elif reason == "replacement":
        engine._actor_id.hex.return_value = "actor-c"
    elif reason == "draining":
        group.lifecycle_status = rollout.EngineGroupLifecycle.DRAINING
    elif reason == "evicted":
        engine.is_evicted.remote.return_value = AwaitableValue(True)
    elif reason == "no_lease":
        manager._training_weight_updating = False
    else:
        record.value = object()  # No successful init metadata is available.
    manager.register_recovered_engines(ids)
    engine.register_to_router.remote.assert_not_called()
    assert record.pending_router_registration


@pytest.mark.parametrize("failure", [False, RuntimeError("router unavailable"), TimeoutError("unconfirmed")])
def test_recovery_router_failure_remains_retryable(failure, patch_ray_get):
    manager, _, engine, record = make_recovery()
    if isinstance(failure, Exception):
        engine.register_to_router.remote.side_effect = failure
    else:
        engine.register_to_router.remote.return_value = AwaitableValue(failure)
    manager.register_recovered_engines(["actor-b"])
    assert record.pending_router_registration
    engine.register_to_router.remote.side_effect = None
    engine.register_to_router.remote.return_value = AwaitableValue(True)
    manager.register_recovered_engines(["actor-b"])
    assert not record.pending_router_registration


def make_client(*, rank=0, fail=False, recipients=None):
    client = object.__new__(CheckpointEngineClient)
    client.role_info = SimpleNamespace(rank=rank)
    client.coordinator_url = "http://coordinator"
    response = MagicMock()
    response.json.return_value = {"nodes": {"rollout": {"4": {}, "5": {}}}}
    client._http_client = SimpleNamespace(get=AsyncMock(return_value=response))
    backend = MagicMock(spec=["init_process_group_for_rollout", "update_weights_for_rollout"])
    backend.update_weights_for_rollout.return_value = ["actor-b"] if recipients is None else recipients
    if fail:
        backend.update_weights_for_rollout.side_effect = RuntimeError("weight transfer failed")
    client._backend = backend
    return client


@pytest.mark.parametrize(
    "recipients,rollout_only,actor_fwd_only",
    [(["actor-b"], False, False), (["actor-shared"], True, False), ([], False, False), ([], False, True)],
)
def test_client_returns_backend_confirmed_recipients(recipients, rollout_only, actor_fwd_only):
    # PP agreement and adapter skips are covered by test_weight_sync_lifecycle;
    # the client must preserve the backend's confirmed receiver snapshot.
    client = make_client(recipients=recipients)
    expected = list(recipients)
    result = asyncio.run(client.update_weights_for_rollout(rollout_only=rollout_only, actor_fwd_only=actor_fwd_only))
    assert result is recipients
    assert result == expected
    client._backend.update_weights_for_rollout.assert_called_once_with(rollout_only, actor_fwd_only)


def test_failed_weight_update_does_not_return_publishable_ids():
    client = make_client(fail=True)
    with pytest.raises(RuntimeError, match="weight transfer failed"):
        asyncio.run(client.update_weights_for_rollout())


def test_same_endpoint_replacement_invalidates_weight_group_cache():
    from relax.distributed.checkpoint_service.backends.device_direct import DeviceDirectBackend

    old = {"4": {"ip": "same-host", "port": 15000, "metadata": {"actor_id": "actor-a", "num_gpus_per_engine": 1}}}
    new = {"4": {"ip": "same-host", "port": 15000, "metadata": {"actor_id": "actor-b", "num_gpus_per_engine": 1}}}
    assert DeviceDirectBackend._rollout_topology_signature_of(
        old
    ) != DeviceDirectBackend._rollout_topology_signature_of(new)


@pytest.mark.parametrize("fail,actor_fwd_only", [(False, False), (True, False), (False, True)])
def test_actor_publishes_only_after_successful_rollout_update(monkeypatch, fail, actor_fwd_only):
    actor_module = pytest.importorskip("relax.backends.megatron.actor", exc_type=ModuleNotFoundError)
    events = []
    actor = object.__new__(actor_module.MegatronTrainRayActor)
    actor.checkpoint_engine_client = make_client(fail=fail)
    actor.rollout_manager = MagicMock()
    actor.rollout_manager.register_recovered_engines.remote.side_effect = lambda ids: events.append(("publish", ids))
    actor._weight_sync_lock = None
    monkeypatch.setattr(actor_module, "get_gloo_group", lambda: "gloo")
    monkeypatch.setattr(actor_module.dist, "barrier", lambda **kwargs: None)
    monkeypatch.setattr(actor_module.dist, "get_rank", lambda **kwargs: 0)
    monkeypatch.setattr(actor_module, "print_memory", lambda *args, **kwargs: None)
    monkeypatch.setattr(actor_module, "run", asyncio.run)
    monkeypatch.setattr(actor_module.ray, "get", lambda ref, **kwargs: ref)
    actor.args = SimpleNamespace(rollout_http_timeout=120)
    # rollout_only bypasses actor_fwd setup but still performs real client update flow.
    if fail:
        with pytest.raises(RuntimeError, match="weight transfer failed"):
            actor.update_weights_fully_async(0, rollout_only=True)
    else:
        actor.update_weights_fully_async(0, rollout_only=True, actor_fwd_only=actor_fwd_only)
    assert events == ([] if fail or actor_fwd_only else [("publish", ["actor-b"])])


def test_dcs_registration_carries_exact_engine_generation(monkeypatch):
    sglang_engine = pytest.importorskip("relax.backends.sglang.sglang_engine", exc_type=ModuleNotFoundError)
    engine = object.__new__(sglang_engine.SGLangEngine)
    engine.node_rank = 0
    engine.rank = 4
    engine.num_gpus_per_engine = 1
    engine.server_host = "engine-b"
    engine.server_port = 30000
    engine._sync_nextn_draft = False
    engine.args = SimpleNamespace(fully_async=True, coordinator_url="http://coordinator")
    create_client = AsyncMock()
    monkeypatch.setattr(sglang_engine, "create_client", create_client)
    monkeypatch.setattr(sglang_engine, "run", asyncio.run)
    monkeypatch.setattr(
        sglang_engine.ray, "get_runtime_context", lambda: SimpleNamespace(get_actor_id=lambda: "actor-b")
    )
    engine.register_dcs()
    assert create_client.call_args.kwargs["metadata"]["actor_id"] == "actor-b"
    assert create_client.call_args.kwargs["metadata"]["sync_nextn_draft"] is False


@pytest.mark.asyncio
async def test_initial_scaleout_success_clears_pending_without_changing_recovery_flag():
    manager, group, engine, record = make_recovery()
    from relax.distributed.ray import rollout

    group.skip_router_registration = True
    server = make_rollout_server(engine_groups=[])
    request = rollout.ScaleOutRequest(request_id="scale", status=rollout.ScaleOutStatus.CREATING)
    manager._health_check_engines = AsyncMock(return_value=True)
    manager._sync_weights_from_seed_engine = AsyncMock(return_value=True)
    result = await manager._finalize_engine_group_registration(
        request=request, srv=server, engines=[engine], engine_group=group
    )
    assert result.success
    assert not record.pending_router_registration
    assert group.skip_router_registration
