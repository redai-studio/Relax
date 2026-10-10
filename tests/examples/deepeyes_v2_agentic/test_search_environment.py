# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any
from unittest.mock import Mock

import pytest


EXAMPLE_DIR = Path(__file__).resolve().parents[3] / "examples" / "deepeyes_v2_agentic"
sys.path.insert(0, str(EXAMPLE_DIR))

from app import env_deepeyes_v2 as env_module  # noqa: E402


@pytest.mark.asyncio
@pytest.mark.parametrize("result", [{"elapsed_time": 0.0, "data": []}, "Error", RuntimeError("private marker")])
async def test_search_observation_remains_recoverable(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, result: Any
) -> None:
    search = Mock(side_effect=result) if isinstance(result, Exception) else Mock(return_value=result)
    monkeypatch.setattr(env_module, "search", search)
    monkeypatch.setattr(env_module.logger, "handlers", [caplog.handler])
    executor = Mock()
    environment = env_module.DeepEyesV2Env(data_index="search", sandbox_executor=executor, image=None)

    observation = await environment.exec_tool('<tool_call>{"name":"search","arguments":" query "}</tool_call>')

    assert observation.done is False
    assert observation.images == []
    assert observation.error == (None if isinstance(result, dict) else "search_failed")
    if isinstance(result, dict):
        assert "found 0 results" in observation.body_text
    else:
        assert observation.body_text.startswith("Error:")
    assert "private marker" not in observation.body_text + caplog.text
    search.assert_called_once_with("query", size=None)
    assert executor.mock_calls == []
    await environment.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "arguments", [{"query": " "}, {"query": "query", "size": True}, {"query": "query", "extra": 1}]
)
async def test_search_invalid_arguments_stop_before_backend(monkeypatch: pytest.MonkeyPatch, arguments: Any) -> None:
    search = Mock()
    monkeypatch.setattr(env_module, "search", search)
    environment = env_module.DeepEyesV2Env(data_index="search", sandbox_executor=Mock(), image=None)

    observation = await environment.exec_tool(
        f"<tool_call>{json.dumps({'name': 'search', 'arguments': arguments})}</tool_call>"
    )

    assert observation.error == "invalid_search_args"
    assert observation.done is False
    search.assert_not_called()
    await environment.close()
