# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""The unified inference endpoints the Rollout deployment gains, and the legacy
``/engines`` shape it keeps."""

from __future__ import annotations

import json

import httpx
import pytest
from fastapi import HTTPException

from relax.components import rollout as rollout_module
from relax.engine.inference.discovery import EngineState, RoleSnapshot, build_model_snapshot
from relax.engine.inference.gateway import InferenceGateway


Rollout = rollout_module.Rollout.func_or_class

LEGACY_ENGINES = {
    "models": {"default": {"router_ip": "192.0.2.1", "router_port": 3000, "engine_groups": [], "total_engines": 0}},
    "total_engines": 0,
}
SNAPSHOT = RoleSnapshot(
    role="rollout",
    topology_revision=3,
    default_model="default",
    models=(
        build_model_snapshot(
            "default", [(0, "http://192.0.2.2:15000", EngineState.READY)], router_url="http://192.0.2.1:3000"
        ),
    ),
)


class _Awaitable:
    def __init__(self, value):
        self._value = value

    def __await__(self):
        yield
        return self._value


class _RemoteMethod:
    def __init__(self, value):
        self.calls: list[tuple] = []
        self._value = value

    def remote(self, *args, **kwargs):
        self.calls.append(args)
        return _Awaitable(self._value)


def _make_rollout():
    rollout = object.__new__(Rollout)
    rollout._logger_instance = None
    rollout._sglang_base_url = None
    rollout._gateway = InferenceGateway("rollout", SNAPSHOT.to_dict, upstream_name="SGLang router")
    rollout.rollout_manager = type("_Manager", (), {})()
    rollout.rollout_manager.get_engines_info = _RemoteMethod(LEGACY_ENGINES)
    rollout.rollout_manager.get_router_address = _RemoteMethod({"router_ip": "192.0.2.1", "router_port": 3000})
    return rollout


async def test_rollout_engines_default_schema_is_unchanged():
    rollout = _make_rollout()

    assert await rollout.get_engines() == LEGACY_ENGINES
    assert await rollout.get_engines(model_name="default") == LEGACY_ENGINES
    assert rollout.rollout_manager.get_engines_info.calls == [(None,), ("default",)]


async def test_rollout_engines_schema_v2_returns_snapshot():
    rollout = _make_rollout()

    result = await rollout.get_engines(schema_version=2)

    assert result == SNAPSHOT.to_dict()
    assert (result["schema_version"], result["role"], result["topology_revision"]) == (2, "rollout", 3)
    assert rollout.rollout_manager.get_engines_info.calls == []


async def test_rollout_engines_rejects_unknown_schema_version():
    rollout = _make_rollout()

    with pytest.raises(HTTPException) as excinfo:
        await rollout.get_engines(schema_version=3)

    assert excinfo.value.status_code == 400
    assert rollout.rollout_manager.get_engines_info.calls == []


async def test_rollout_health_reports_model_states():
    health = await _make_rollout().health()

    assert health["role"] == "rollout"
    assert health["models"] == {"default": "ready"}


async def test_rollout_generate_forwards_native_payload_to_router(monkeypatch):
    seen = {}
    real_async_client = httpx.AsyncClient

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(url=str(request.url), payload=json.loads(request.content))
        return httpx.Response(200, json={"text": "done"})

    monkeypatch.setattr(
        httpx, "AsyncClient", lambda **kwargs: real_async_client(transport=httpx.MockTransport(handler))
    )

    class _Request:
        async def json(self):
            return {"text": "hi", "sampling_params": {"max_new_tokens": 4}}

    assert await _make_rollout().generate(_Request()) == {"text": "done"}
    assert seen == {
        "url": "http://192.0.2.1:3000/generate",
        "payload": {"text": "hi", "sampling_params": {"max_new_tokens": 4}},
    }


def _registered_paths(app) -> set[str]:
    """Paths of every route, looking inside the router ``@serve.ingress`` re-
    includes the class-based routes through."""
    paths: set[str] = set()
    for route in app.routes:
        included = getattr(route, "original_router", None)
        for item in included.routes if included is not None else [route]:
            if getattr(item, "path", None):
                paths.add(item.path)
    return paths


def test_rollout_registers_unified_routes_alongside_legacy_ones():
    paths = _registered_paths(rollout_module.app)

    assert {"/engines", "/health", "/v1/models", "/generate", "/chat/completions", "/v1/chat/completions"} <= paths
    # Legacy control routes are still there.
    assert {"/scale_out", "/scale_in", "/get_step", "/evaluate"} <= paths
