# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from __future__ import annotations

import asyncio
import json
import socket
import sys
from collections.abc import Iterator
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, Mock

import httpx
import openai
import pytest
import yaml


EXAMPLE_DIR = Path(__file__).resolve().parents[3] / "examples" / "deepeyes_v2_agentic"
sys.path.insert(0, str(EXAMPLE_DIR))

from app import agent, env_deepeyes_v2, search_http, search_utils  # noqa: E402
from app.prompt import UNIFIED_SYSTEM_PROMPT  # noqa: E402
from app.search_config import SEARCH_CONFIG_ENV  # noqa: E402


@pytest.mark.asyncio
async def test_agent_retries_search_after_error_and_receives_service_evidence(monkeypatch: pytest.MonkeyPatch) -> None:
    search_text = '<tool_call>{"name":"search","arguments":{"query":"query","size":2}}</tool_call>'
    responses = iter([search_text, search_text, "<answer>Completed answer.</answer>"])
    requests = []

    async def create(**kwargs: Any) -> SimpleNamespace:
        requests.append(deepcopy(kwargs["messages"]))
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=next(responses)), finish_reason="stop")]
        )

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    monkeypatch.setattr(openai, "AsyncOpenAI", Mock(return_value=client))
    monkeypatch.setenv("OPENAI_API_KEY", "test-model-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://model.example.test/v1")
    executor = Mock()
    environment = env_deepeyes_v2.DeepEyesV2Env(data_index="search", sandbox_executor=executor, image=None)
    close = AsyncMock(wraps=environment.close)
    monkeypatch.setattr(environment, "close", close)
    monkeypatch.setattr(agent, "_build_executor", Mock(return_value=executor))
    monkeypatch.setattr(agent, "DeepEyesV2Env", Mock(return_value=environment))
    search = Mock(
        side_effect=[
            "Error",
            {
                "elapsed_time": 0.1,
                "data": [
                    {"title": "Document", "link": "", "snippet": "Service evidence.", "date": None},
                    {"title": "Source", "link": "https://example.test", "snippet": "Details.", "date": "2026-09-20"},
                ],
            },
        ]
    )
    monkeypatch.setattr(env_deepeyes_v2, "search", search)
    messages = [{"role": "user", "content": "Find evidence."}]

    output = await agent.run_session(messages, {})

    assert len(requests) == 3
    assert requests[1][-1]["role"] == "tool"
    assert requests[1][-1]["content"].startswith("Error:")
    assert requests[2][:-2] == requests[1]
    observation = requests[2][-1]
    assert observation["role"] == "tool"
    assert "1. Document\nService evidence." in observation["content"]
    assert "[Source](https://example.test)\nDate published: 2026-09-20" in observation["content"]
    assert output["metadata"]["final_answer"] == "Completed answer."
    assert output["metadata"]["stop_reason"] == "env_done"
    assert output["metadata"]["last_error"] == "search_failed"
    assert search.call_count == 2
    search.assert_called_with("query", size=2)
    close.assert_awaited_once_with()
    assert executor.mock_calls == []


@pytest.fixture
def no_network(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    denied = Mock(side_effect=AssertionError("Unexpected network access"))
    async_denied = AsyncMock(side_effect=AssertionError("Unexpected async network access"))
    for target, name in (
        (socket.socket, "connect"),
        (socket.socket, "connect_ex"),
        (socket, "create_connection"),
        (socket, "getaddrinfo"),
        (httpx.HTTPTransport, "handle_request"),
    ):
        monkeypatch.setattr(target, name, denied)
    monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", async_denied)
    yield
    denied.assert_not_called()
    async_denied.assert_not_called()


@pytest.mark.usefixtures("no_network")
@pytest.mark.parametrize("backend", ["mock", "retriever", "external"])
def test_agent_main_completes_search_without_network(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, backend: str
) -> None:
    monkeypatch.delenv(SEARCH_CONFIG_ENV, raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "test-model-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://model.example.test/v1")
    monkeypatch.setenv("OPENAI_MODEL", "test-model")
    if backend != "mock":
        template = "brave" if backend == "external" else "retriever"
        config = yaml.safe_load((EXAMPLE_DIR / f"search_config.{template}.yaml").read_text(encoding="utf-8"))
        config.update(endpoint=f"https://{backend}.example.test/search", retry_delay_s=0.0, retry_max_delay_s=0.0)
        config_path = tmp_path / "search.yaml"
        config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
        monkeypatch.setenv(SEARCH_CONFIG_ENV, str(config_path))
        monkeypatch.setenv("BRAVE_SEARCH_API_KEY", "test-search-key")

    query = "DeepEyes V2"
    search_text = "<tool_call>" + json.dumps({"name": "search", "arguments": {"query": query}}) + "</tool_call>"
    messages = [
        {"role": "system", "content": UNIFIED_SYSTEM_PROMPT},
        {"role": "user", "content": "Search for DeepEyes V2 and answer."},
    ]
    input_path, output_path = tmp_path / "input.json", tmp_path / "output.json"
    input_path.write_text(json.dumps({"messages": messages, "metadata": {"data_index": "search"}}), encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["agent", "--input-json", str(input_path), "--output-json", str(output_path)])
    requests: list[dict[str, Any]] = []
    events: list[str] = []

    def model_response(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST" and request.url.path == "/v1/chat/completions"
        requests.append(json.loads(request.content))
        events.append("model")
        assert len(requests) <= 2
        text = search_text if len(requests) == 1 else "<answer>Completed answer.</answer>"
        return httpx.Response(
            200,
            json={
                "id": "test-completion",
                "object": "chat.completion",
                "created": 0,
                "model": "test-model",
                "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
            },
        )

    search_responses = [
        httpx.Response(503, stream=httpx.ByteStream(b"")) for _ in range(0 if backend == "mock" else 3)
    ]
    search_handler = Mock(side_effect=search_responses)
    search_clients: list[httpx.Client] = []

    def create_search_client(config: search_http.HttpSearchConfig) -> httpx.Client:
        assert backend != "mock"
        client = httpx.Client(transport=httpx.MockTransport(search_handler), timeout=config.timeout_s, trust_env=False)
        search_clients.append(client)
        return client

    def search(query: str, size: int | None = None) -> Any:
        events.append("search")
        return search_utils.search(query, size=size)

    search_call = Mock(side_effect=search)
    monkeypatch.setattr(search_http, "_create_client", create_search_client)
    monkeypatch.setattr(env_deepeyes_v2, "search", search_call)
    executor = Mock(spec=agent.SandboxExecutor)
    environment = env_deepeyes_v2.DeepEyesV2Env(data_index="search", sandbox_executor=executor, image=None)
    real_close = environment.close

    async def close_environment() -> None:
        await real_close()
        events.append("close")

    close = AsyncMock(side_effect=close_environment)
    monkeypatch.setattr(environment, "close", close)
    monkeypatch.setattr(agent, "_build_executor", Mock(return_value=executor))
    monkeypatch.setattr(agent, "DeepEyesV2Env", Mock(return_value=environment))
    client = openai.AsyncOpenAI(
        api_key="test-model-key",
        base_url="https://model.example.test/v1",
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(model_response), trust_env=False),
        max_retries=0,
    )
    monkeypatch.setattr(openai, "AsyncOpenAI", Mock(return_value=client))
    try:
        agent.main()
    finally:
        asyncio.run(client.close())

    assert len(requests) == 2
    assert requests[0]["messages"] == messages
    assert requests[1]["messages"][:-2] == messages
    assert requests[1]["messages"][-2] == {"role": "assistant", "content": search_text}
    observation = requests[1]["messages"][-1]
    assert observation["role"] == "tool"
    if backend == "mock":
        assert "found 5 results" in observation["content"] and "[mock]" in observation["content"]
        assert query in observation["content"]
    else:
        assert observation["content"].startswith("Error:")
    search_call.assert_called_once_with(query, size=None)
    assert search_handler.call_count == (0 if backend == "mock" else 3)
    assert len(search_clients) == (0 if backend == "mock" else 1)
    assert all(response.is_closed for response in search_responses)
    assert all(search_client.is_closed for search_client in search_clients)
    close.assert_awaited_once_with()
    assert executor.mock_calls == []
    assert events == ["model", "search", "model", "close"]
    metadata = json.loads(output_path.read_text(encoding="utf-8"))["metadata"]
    assert metadata["final_answer"] == "Completed answer."
    assert metadata["stop_reason"] == "env_done"
    assert metadata["last_error"] == (None if backend == "mock" else "search_failed")
    assert metadata["branch_counts"] == {"answer": 1, "code": 0, "tool_call": 1, "format_error": 0}
