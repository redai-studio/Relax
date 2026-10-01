# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import asyncio
import json
from types import SimpleNamespace

import httpx
import pytest
from fastapi import FastAPI, Request

from relax.components.inference_gateway import InferenceGateway


@pytest.mark.parametrize("name,route", [("__default__", None), ("math", "math")])
async def test_legacy_genrm_http_messages_and_response(name, route):
    seen = []
    tokenized = []
    messages = [{"role": "user", "content": "Judge this answer"}]

    class Tokenizer:
        def apply_chat_template(self, value, **kwargs):
            tokenized.append((value, kwargs))
            return [1, 2]

    def manager(key):
        snapshot = {
            "topology_revision": 1,
            "phase": "inference",
            "models": {
                "default": {
                    "state": "ready",
                    "router_url": None,
                    "engines": [{"state": "ready", "direct_eligible": True, "base_url": f"http://{key}-head"}],
                }
            },
        }
        return SimpleNamespace(
            get_inference_snapshot=SimpleNamespace(remote=lambda: asyncio.sleep(0, result=snapshot))
        )

    keys = [name] if route is None else [name, "code"]
    gateway = InferenceGateway(
        "genrm",
        {key: manager(key) for key in keys},
        tokenizers={key: Tokenizer() for key in keys},
        instance_specs={
            key: {
                "sampling_config": {
                    "temperature": 0.7,
                    "top_p": 0.88,
                    "top_k": 5,
                    "max_response_len": 13,
                    "chat_template_kwargs": {"enable_thinking": False},
                }
            }
            for key in keys
        },
    )
    await gateway._client.aclose()

    def backend(request):
        seen.append((request.url.host, json.loads(request.content)))
        return httpx.Response(200, json={"text": " answer \n"})

    gateway._client = httpx.AsyncClient(transport=httpx.MockTransport(backend))
    app = FastAPI()

    @app.post("/generate")
    async def generate(request: Request):
        return await gateway.forward(request, "generate")

    payload = {"messages": messages, "sampling_params": {"temperature": 0.1, "max_new_tokens": 7}}
    if route is not None:
        payload["route_key"] = route
    try:
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://gateway") as client:
            response = await client.post("/generate", json=payload)
            assert response.status_code == 200
            assert response.json() == {"response": "answer"}
            assert seen == [
                (
                    f"{name}-head",
                    {
                        "input_ids": [1, 2],
                        "sampling_params": {
                            "temperature": 0.1,
                            "top_p": 0.88,
                            "top_k": 5,
                            "max_new_tokens": 7,
                        },
                    },
                )
            ]
            assert tokenized == [
                (
                    messages,
                    {
                        "tokenize": True,
                        "add_generation_prompt": True,
                        "enable_thinking": False,
                    },
                )
            ]
            rejected = await client.post("/generate", json={**payload, "route_key": "missing"})
            assert rejected.status_code == 400
            assert len(seen) == 1
    finally:
        await gateway.aclose()
