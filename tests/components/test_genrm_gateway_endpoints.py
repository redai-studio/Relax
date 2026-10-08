# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""The unified inference endpoints the GenRM deployment gains, and the
``messages``-based ``/generate`` contract it keeps."""

from __future__ import annotations

import json
from types import SimpleNamespace

import httpx
import pytest
from fastapi import HTTPException
from fastapi.exceptions import RequestValidationError

from relax.components import genrm as genrm_module
from relax.engine.inference.gateway import InferenceGateway


GenRM = genrm_module.GenRM.func_or_class

# RFC 5737 TEST-NET-1: the gitleaks hook rejects private-range literals.
ENGINE = ("192.0.2.1", 16001)


class _Manager:
    """Stands in for one instance's GenRMManager handle."""

    def __init__(self, name: str, state: str = "ready"):
        self.name = name
        self.state = state
        self.calls: list[str] = []
        for method in ("get_inference_snapshot", "health_check", "onload", "offload"):
            setattr(self, method, SimpleNamespace(remote=lambda method=method: self._call(method)))

    def _call(self, method: str):
        self.calls.append(method)
        if method == "health_check":
            return True
        if method != "get_inference_snapshot":
            return None
        return {
            "topology_revision": 1,
            "state": self.state,
            "router_url": None,
            "engines": [{"index": 0, "base_url": f"http://{self.name}:1", "state": self.state}],
        }


class _Request:
    def __init__(self, payload):
        self._payload = payload
        self.headers: dict[str, str] = {}

    async def json(self):
        return self._payload

    async def body(self) -> bytes:
        return json.dumps(self._payload).encode()


class _Tokenizer:
    def __init__(self):
        self.calls: list[tuple] = []

    def apply_chat_template(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        return [11, 12, 13]


class _EngineClient:
    """Stands in for the replica's engine ``httpx.AsyncClient``."""

    def __init__(self, text: str):
        self.posts: list[tuple[str, dict]] = []
        self._text = text

    async def post(self, url, json):
        self.posts.append((url, json))
        return SimpleNamespace(raise_for_status=lambda: None, json=lambda: {"text": self._text})


def _make_genrm(monkeypatch, *instances: str, state: str = "ready"):
    """A shell GenRM deployment over fake managers, without a cluster."""
    # Manager calls return their value directly, so ``ray.get`` is the identity.
    monkeypatch.setitem(
        GenRM._fetch_inference_snapshot.__globals__, "ray", SimpleNamespace(get=lambda value, timeout=None: value)
    )

    genrm = object.__new__(GenRM)
    genrm._logger_instance = None
    genrm.role = "genrm"
    genrm.genrm_managers = {name: _Manager(name, state) for name in instances}
    genrm.instance_specs = {
        name: {
            "model_path": f"/models/{name}",
            "num_gpus": 2,
            "num_gpus_per_engine": 1,
            "sampling_config": {"temperature": 0.1, "max_response_len": 64},
        }
        for name in instances
    }
    genrm._engine_caches = {name: genrm_module._EngineCacheState() for name in instances}
    for cache in genrm._engine_caches.values():
        cache.refresh([ENGINE])
    genrm.tokenizers = {name: _Tokenizer() for name in instances}
    genrm._http_client = _EngineClient(" A is better \n")
    genrm._gateway = InferenceGateway("genrm", genrm._fetch_inference_snapshot, upstream_name="GenRM engine")
    return genrm


@pytest.fixture
def upstream(monkeypatch):
    """Route the gateway's proxy client into a recording fake engine."""
    seen: list[httpx.Request] = []
    real_async_client = httpx.AsyncClient

    def dispatch(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json={"text": "raw"})

    monkeypatch.setattr(
        httpx, "AsyncClient", lambda **kwargs: real_async_client(transport=httpx.MockTransport(dispatch))
    )
    return seen


# ----------------------------------------------------------------------
# The legacy contract.
# ----------------------------------------------------------------------


async def test_genrm_generate_with_messages_keeps_legacy_response(monkeypatch, upstream):
    genrm = _make_genrm(monkeypatch, "quality", "safety")
    messages = [{"role": "user", "content": "which is better?"}]

    response = await genrm.generate_endpoint(
        _Request({"messages": messages, "route_key": "safety", "sampling_params": {"temperature": 0.7}})
    )

    assert response == genrm_module.GenerateResponse(response="A is better")
    # Rendered with the selected instance's chat template...
    rendered, template_kwargs = genrm.tokenizers["safety"].calls[0]
    assert [message.model_dump() for message in rendered] == messages
    assert template_kwargs == {"tokenize": True, "add_generation_prompt": True}
    assert genrm.tokenizers["quality"].calls == []
    # ...and sent with request-level sampling params merged over the instance defaults.
    assert genrm._http_client.posts == [
        (
            f"http://{ENGINE[0]}:{ENGINE[1]}/generate",
            {
                "input_ids": [11, 12, 13],
                "sampling_params": {"temperature": 0.7, "top_p": 1.0, "top_k": -1, "max_new_tokens": 64},
            },
        )
    ]
    # The legacy path picks its own engine: the gateway is not involved.
    assert upstream == []
    assert all(manager.calls == [] for manager in genrm.genrm_managers.values())


async def test_genrm_generate_with_invalid_messages_is_a_validation_error(monkeypatch):
    genrm = _make_genrm(monkeypatch, "__default__")

    with pytest.raises(RequestValidationError):
        await genrm.generate_endpoint(_Request({"messages": "not a list"}))


async def test_genrm_generate_without_messages_forwards_native_payload(monkeypatch, upstream):
    genrm = _make_genrm(monkeypatch, "quality", "safety")

    response = await genrm.generate_endpoint(_Request({"route_key": "safety", "input_ids": [1, 2]}))

    assert response == {"text": "raw"}
    assert [(str(request.url), json.loads(request.content)) for request in upstream] == [
        ("http://safety:1/generate", {"input_ids": [1, 2]})
    ]
    assert genrm._http_client.posts == []


async def test_genrm_health_and_metrics_keep_their_shape(monkeypatch):
    monkeypatch.setitem(GenRM.health.__globals__, "ray", SimpleNamespace(get=lambda value, timeout=None: value))
    single = _make_genrm(monkeypatch, "__default__")
    multi = _make_genrm(monkeypatch, "quality", "safety")

    assert await single.health() == {"status": "healthy", "service": "genrm"}
    assert await single.metrics() == {
        "service": "genrm",
        "model_path": "/models/__default__",
        "num_gpus": 2,
        "num_engines": 2,
    }
    assert await multi.health() == {
        "status": "healthy",
        "service": "genrm",
        "instances": {"quality": {"status": "healthy"}, "safety": {"status": "healthy"}},
    }


# ----------------------------------------------------------------------
# The unified endpoints.
# ----------------------------------------------------------------------


async def test_genrm_models_lists_instances(monkeypatch):
    genrm = _make_genrm(monkeypatch, "quality", "safety")

    listing = await genrm.list_models()

    assert [item["id"] for item in listing["data"]] == ["quality", "safety"]


async def test_genrm_engines_returns_role_snapshot(monkeypatch):
    snapshot = await _make_genrm(monkeypatch, "quality", "safety").get_engines()

    assert (snapshot["schema_version"], snapshot["role"], snapshot["topology_revision"]) == (2, "genrm", 2)
    assert list(snapshot["models"]) == ["quality", "safety"]


async def test_genrm_request_while_sleeping_returns_503(monkeypatch, upstream):
    genrm = _make_genrm(monkeypatch, "__default__", state="sleeping")

    with pytest.raises(HTTPException) as excinfo:
        await genrm.chat_completions(_Request({"messages": [{"role": "user", "content": "hi"}]}))

    assert excinfo.value.status_code == 503
    assert "Retry-After" in excinfo.value.headers
    # Nothing was forwarded, and the request did not wake the model up.
    assert upstream == []
    assert genrm.genrm_managers["__default__"].calls == ["get_inference_snapshot"]


@pytest.mark.parametrize(
    ("instances", "route_key"),
    [
        (("__default__",), None),
        (("quality",), None),
        (("quality", "safety"), "safety"),
        (("__default__", "safety"), None),
        (("__default__", "safety"), "safety"),
        (("quality", "safety"), None),
        (("quality", "safety"), "missing"),
    ],
)
async def test_genrm_gateway_and_legacy_path_select_same_instance(monkeypatch, instances, route_key):
    genrm = _make_genrm(monkeypatch, *instances)

    try:
        legacy = genrm._resolve_instance_key(route_key)
    except RuntimeError:
        legacy = None
    try:
        unified = (await genrm._gateway.resolve(route_key=route_key)).model
    except HTTPException as exc:
        assert exc.status_code == 400
        unified = None

    assert unified == legacy


def test_genrm_registers_unified_routes_alongside_legacy_ones():
    paths: set[str] = set()
    for route in genrm_module.app.routes:
        # ``@serve.ingress`` re-includes the class-based routes through a router.
        included = getattr(route, "original_router", None)
        paths.update(item.path for item in (included.routes if included is not None else [route]))

    assert {"/engines", "/v1/models", "/generate", "/chat/completions", "/v1/chat/completions"} <= paths
    assert {"/health", "/metrics"} <= paths
