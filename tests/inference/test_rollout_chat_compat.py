# Copyright (c) 2026 Relax Authors. All Rights Reserved.


import ast
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest


def _load_chat_dispatch():
    source_path = Path(__file__).parents[2] / "relax/components/rollout.py"
    module = ast.parse(source_path.read_text())
    predicate = next(
        node for node in module.body if isinstance(node, ast.FunctionDef) and node.name == "use_legacy_rollout_chat"
    )
    rollout_class = next(node for node in module.body if isinstance(node, ast.ClassDef) and node.name == "Rollout")
    handler = next(
        node
        for node in rollout_class.body
        if isinstance(node, ast.AsyncFunctionDef) and node.name == "_handle_chat_completions"
    )

    class ChatRequest:
        @staticmethod
        def model_validate_json(body):
            return SimpleNamespace(stream=json.loads(body)["stream"])

    namespace = {
        "Namespace": object,
        "Request": object,
        "ChatCompletionRequest": ChatRequest,
        "HTTPException": RuntimeError,
    }
    exec(compile(ast.Module(body=[predicate, handler], type_ignores=[]), str(source_path), "exec"), namespace)
    return namespace["_handle_chat_completions"]


class _Request:
    def __init__(self, stream):
        self._body = json.dumps({"model": "policy", "messages": [], "stream": stream}).encode()
        self.headers = {"x-request-id": "request-1"}

    async def body(self):
        return self._body


def _rollout(*, rollout_external=False, debug_rollout_only=False):
    gateway = SimpleNamespace(handle_chat=AsyncMock(return_value="gateway"))
    return SimpleNamespace(
        config=SimpleNamespace(rollout_external=rollout_external, debug_rollout_only=debug_rollout_only),
        _get_inference_gateway=lambda: gateway,
        _get_proxy_client=lambda: "client",
        _get_sglang_url=AsyncMock(return_value="http://router.example/v1/chat/completions"),
        _non_stream_chat_completions=AsyncMock(return_value="router-non-stream"),
        _stream_chat_completions=AsyncMock(return_value="router-stream"),
        gateway=gateway,
    )


@pytest.mark.parametrize("mode", ["external", "debug"])
@pytest.mark.parametrize("stream", [False, True])
async def test_legacy_rollout_modes_keep_router_chat_proxy(mode, stream):
    dispatch = _load_chat_dispatch()
    instance = _rollout(rollout_external=mode == "external", debug_rollout_only=mode == "debug")

    result = await dispatch(instance, _Request(stream))

    assert result == ("router-stream" if stream else "router-non-stream")
    instance._get_sglang_url.assert_awaited_once_with("/v1/chat/completions")
    instance.gateway.handle_chat.assert_not_awaited()


async def test_managed_rollout_chat_still_uses_discovery_gateway():
    dispatch = _load_chat_dispatch()
    instance = _rollout()

    assert await dispatch(instance, _Request(False)) == "gateway"

    instance.gateway.handle_chat.assert_awaited_once()
    instance._get_sglang_url.assert_not_awaited()


def test_both_chat_aliases_share_compatibility_dispatch():
    source = (Path(__file__).parents[2] / "relax/components/rollout.py").read_text()
    module = ast.parse(source)
    methods = {
        node.name: ast.unparse(node)
        for node in ast.walk(module)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name in {"chat_completions", "chat_completions_alias"}
    }

    assert set(methods) == {"chat_completions", "chat_completions_alias"}
    assert all("self._handle_chat_completions(request)" in method for method in methods.values())
