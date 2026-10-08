# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""The standalone gateway deployment that fronts the managed OPD teacher."""

from __future__ import annotations

import json
from types import SimpleNamespace

import httpx
import pytest
from fastapi import HTTPException

from relax.components import inference_gateway as gateway_module


Service = gateway_module.InferenceGatewayService.func_or_class


class _Manager:
    def __init__(self, name: str, state: str = "ready"):
        self.calls: list[str] = []
        self._payload = {
            "topology_revision": 3,
            "state": state,
            "router_url": None,
            "engines": [{"index": 0, "base_url": f"http://{name}:1", "state": state}],
        }
        for method in ("get_inference_snapshot", "onload"):
            setattr(self, method, SimpleNamespace(remote=lambda method=method: self._call(method)))

    def _call(self, method: str):
        self.calls.append(method)
        return self._payload


class _Request:
    def __init__(self, payload):
        self._payload = payload
        self.headers: dict[str, str] = {}

    async def json(self):
        return self._payload

    async def body(self) -> bytes:
        return json.dumps(self._payload).encode()


def _make_service(monkeypatch, managers, default_model=None):
    # Manager calls return their value directly, so ``ray.get`` is the identity.
    monkeypatch.setitem(Service._fetch_inference_snapshot.__globals__, "ray", SimpleNamespace(get=lambda value: value))
    service = object.__new__(Service)
    # ``@serve.ingress`` wraps the class in one whose ``__init__`` is async and
    # also boots the ASGI app; the first base is the class as written.
    Service.__bases__[0].__init__(service, "teacher", managers, default_model)
    return service


@pytest.fixture
def upstream(monkeypatch):
    seen: list[httpx.Request] = []
    real_async_client = httpx.AsyncClient

    def dispatch(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json={"text": "ok"})

    monkeypatch.setattr(
        httpx, "AsyncClient", lambda **kwargs: real_async_client(transport=httpx.MockTransport(dispatch))
    )
    return seen


async def test_inference_gateway_service_engines_reports_teacher_topology(monkeypatch):
    service = _make_service(monkeypatch, {"__default__": _Manager("teacher")}, "__default__")

    snapshot = await service.get_engines()

    assert (snapshot["schema_version"], snapshot["role"], snapshot["topology_revision"]) == (2, "teacher", 3)
    assert snapshot["routing"]["default_model"] == "__default__"
    assert list(snapshot["models"]) == ["__default__"]


async def test_inference_gateway_service_generate_forwards_to_a_ready_replica(monkeypatch, upstream):
    service = _make_service(monkeypatch, {"math": _Manager("math"), "code": _Manager("code")})

    response = await service.generate(_Request({"model": "code", "input_ids": [1, 2]}))

    assert response == {"text": "ok"}
    assert [(str(request.url), json.loads(request.content)) for request in upstream] == [
        ("http://code:1/generate", {"input_ids": [1, 2]})
    ]


async def test_inference_gateway_service_unknown_model_is_a_400_listing_models(monkeypatch, upstream):
    service = _make_service(monkeypatch, {"math": _Manager("math"), "code": _Manager("code")})

    with pytest.raises(HTTPException) as excinfo:
        await service.generate(_Request({"model": "physics", "input_ids": [1]}))

    assert excinfo.value.status_code == 400
    assert "['math', 'code']" in excinfo.value.detail
    assert upstream == []


async def test_inference_gateway_service_sleeping_teacher_is_healthy_but_unavailable(monkeypatch, upstream):
    manager = _Manager("teacher", state="sleeping")
    service = _make_service(monkeypatch, {"__default__": manager}, "__default__")

    health = await service.health()
    with pytest.raises(HTTPException) as excinfo:
        await service.chat_completions(_Request({"messages": []}))

    assert health["status"] == "ok"
    assert health["models"] == {"__default__": "sleeping"}
    assert excinfo.value.status_code == 503
    assert "Retry-After" in excinfo.value.headers
    assert upstream == []
    assert "onload" not in manager.calls


async def test_inference_gateway_service_rejects_non_object_payload(monkeypatch):
    service = _make_service(monkeypatch, {"__default__": _Manager("teacher")}, "__default__")

    with pytest.raises(HTTPException) as excinfo:
        await service.generate(_Request([1, 2]))

    assert excinfo.value.status_code == 400


def test_inference_gateway_service_registers_the_unified_routes():
    paths: set[str] = set()
    for route in gateway_module.app.routes:
        # ``@serve.ingress`` re-includes the class-based routes through a router.
        included = getattr(route, "original_router", None)
        paths.update(item.path for item in (included.routes if included is not None else [route]))

    assert {"/engines", "/health", "/v1/models", "/generate", "/chat/completions", "/v1/chat/completions"} <= paths


def test_inference_gateway_service_requests_no_gpu():
    options = gateway_module.InferenceGatewayService.ray_actor_options or {}

    assert not options.get("num_gpus")
