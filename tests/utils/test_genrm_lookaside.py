# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import json
from types import SimpleNamespace

import httpx

from relax.engine.inference.discovery import EngineState, RoleSnapshot, build_model_snapshot
from relax.utils import genrm_client, http_utils
from tests.engine.inference.helpers import AdmissionManager


async def test_managed_genrm_preserves_template_sampling_and_response(monkeypatch):
    owner = AdmissionManager(
        RoleSnapshot("genrm", 1, models=(build_model_snapshot("judge", [(0, "http://engine:1", EngineState.READY)]),))
    )
    spec = {
        "model_path": "/models/judge",
        "sampling_config": {
            "temperature": 0.1,
            "max_response_len": 64,
            "chat_template_kwargs": {"enable_thinking": False},
        },
    }
    control = owner.client("genrm", specs={"judge": spec})
    monkeypatch.setattr(genrm_client, "get_request_client", lambda role: control)
    monkeypatch.setattr(http_utils, "get_request_client", lambda role: control)
    monkeypatch.setattr(genrm_client, "get_serve_url", lambda role: "http://gateway:1/genrm")
    seen, templates = [], []

    def tokenize(messages, **kwargs):
        templates.append((messages, kwargs))
        return {"input_ids": [1, 2]}

    def dispatch(request):
        seen.append(request)
        assert owner.released == []
        return httpx.Response(200, json={"text": " yes \n"})

    client = genrm_client.GenRMClient()
    await client._async_client.aclose()
    client._tokenizers["judge"] = SimpleNamespace(apply_chat_template=tokenize)
    async with httpx.AsyncClient(transport=httpx.MockTransport(dispatch)) as http:
        client._async_client = http
        assert await client.generate([{"role": "user", "content": "hi"}], {"top_p": 0.8}, "judge") == "yes"
    client._sync_client.close()
    assert str(seen[0].url) == "http://engine:1/generate"
    assert json.loads(seen[0].content) == {
        "input_ids": [1, 2],
        "sampling_params": {"temperature": 0.1, "top_p": 0.8, "top_k": -1, "max_new_tokens": 64},
    }
    assert templates[0][1] == {"tokenize": True, "add_generation_prompt": True, "enable_thinking": False}
    assert owner.acquired == owner.released


async def test_explicit_genrm_service_keeps_messages_http_contract(monkeypatch):
    monkeypatch.setattr(
        genrm_client, "get_request_client", lambda role: (_ for _ in ()).throw(AssertionError("external service"))
    )
    seen = []

    def dispatch(request):
        seen.append(request)
        return httpx.Response(200, json={"response": "external answer"})

    client = genrm_client.GenRMClient("http://external:1/genrm")
    await client._async_client.aclose()
    async with httpx.AsyncClient(transport=httpx.MockTransport(dispatch)) as http:
        client._async_client = http
        assert await client.generate([], route_key="judge") == "external answer"
    client._sync_client.close()
    assert str(seen[0].url) == "http://external:1/genrm/generate"
    assert json.loads(seen[0].content) == {"messages": [], "route_key": "judge"}


def test_explicit_genrm_url_does_not_reuse_managed_singleton(monkeypatch):
    monkeypatch.setattr(genrm_client, "_genrm_client", None)
    monkeypatch.setattr(genrm_client, "_external_clients", {})
    monkeypatch.setattr(genrm_client, "GenRMClient", lambda **kwargs: SimpleNamespace(**kwargs))
    managed = genrm_client.get_genrm_client()
    external = genrm_client.get_genrm_client("http://external:1")
    assert external is not managed
    assert genrm_client.get_genrm_client() is managed
    assert genrm_client.get_genrm_client("http://external:1") is external
