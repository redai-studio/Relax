# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from __future__ import annotations

import sys
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, Mock

import openai
import pytest


EXAMPLE_DIR = Path(__file__).resolve().parents[3] / "examples" / "deepeyes_v2_agentic"
sys.path.insert(0, str(EXAMPLE_DIR))

from app import agent, env_deepeyes_v2  # noqa: E402


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
