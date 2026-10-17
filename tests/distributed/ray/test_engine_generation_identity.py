# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""DCS identifies each rollout engine generation by its Ray actor id."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest


def test_same_endpoint_replacement_invalidates_weight_group_cache():
    from relax.distributed.checkpoint_service.backends.device_direct import DeviceDirectBackend

    old = {"4": {"ip": "same-host", "port": 15000, "metadata": {"actor_id": "actor-a", "num_gpus_per_engine": 1}}}
    new = {"4": {"ip": "same-host", "port": 15000, "metadata": {"actor_id": "actor-b", "num_gpus_per_engine": 1}}}
    assert DeviceDirectBackend._rollout_topology_signature_of(
        old
    ) != DeviceDirectBackend._rollout_topology_signature_of(new)


def test_dcs_registration_carries_exact_engine_generation(monkeypatch):
    sglang_engine = pytest.importorskip("relax.backends.sglang.sglang_engine", exc_type=ModuleNotFoundError)
    engine = object.__new__(sglang_engine.SGLangEngine)
    engine.node_rank = 0
    engine.rank = 4
    engine.num_gpus_per_engine = 1
    engine.server_host = "engine-b"
    engine.server_port = 30000
    engine.args = SimpleNamespace(fully_async=True, coordinator_url="http://coordinator")
    create_client = AsyncMock()
    monkeypatch.setattr(sglang_engine, "create_client", create_client)
    monkeypatch.setattr(sglang_engine, "run", asyncio.run)
    monkeypatch.setattr(
        sglang_engine.ray, "get_runtime_context", lambda: SimpleNamespace(get_actor_id=lambda: "actor-b")
    )
    engine.register_dcs()
    assert create_client.call_args.kwargs["metadata"]["actor_id"] == "actor-b"
