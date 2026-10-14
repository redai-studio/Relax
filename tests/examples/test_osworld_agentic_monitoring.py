# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import ast
import asyncio
import json
import os
from pathlib import Path
from types import SimpleNamespace


def _load_adapter_functions():
    # Exercise the actual adapter functions without loading the optional VM SDK.
    path = Path(__file__).resolve().parents[2] / "examples/osworld_agentic/app/agent.py"
    tree = ast.parse(path.read_text())
    names = {"_predict_one_step", "write_session_output"}
    nodes = [
        node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in names
    ]
    module = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), *nodes],
        type_ignores=[],
    )
    namespace = {"asyncio": asyncio, "json": json, "os": os, "Path": Path}
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    return namespace


def test_prediction_records_distinguish_retry_and_executed_action():
    functions = _load_adapter_functions()
    agent = SimpleNamespace(turn_records=[], max_steps=30)

    def predict(*args, **kwargs):
        agent.turn_records.extend([{}, {}])
        return "valid", ["click()"], {"action_type": "gui"}

    agent.predict = predict
    obs = SimpleNamespace(screenshot_bytes=b"", cur_app=None, app_info=None, tool_list=[])
    asyncio.run(functions["_predict_one_step"](agent, "task", obs, 4))
    assert agent.turn_records == [
        {"turn_index": 4, "parse_attempt": 0, "accepted_for_execution": False},
        {"turn_index": 4, "parse_attempt": 1, "accepted_for_execution": True},
    ]


def test_forced_step_cap_is_not_reported_as_executed_model_action():
    functions = _load_adapter_functions()
    agent = SimpleNamespace(turn_records=[], max_steps=30)

    def predict(*args, **kwargs):
        agent.turn_records.append({})
        return "click", ["FAIL"], {"action_type": "gui"}

    agent.predict = predict
    obs = SimpleNamespace(screenshot_bytes=b"", cur_app=None, app_info=None, tool_list=[])
    result = asyncio.run(functions["_predict_one_step"](agent, "task", obs, 29))
    assert result[2]["forced_max_turns"] is True
    assert agent.turn_records[0]["accepted_for_execution"] is False


def test_export_preserves_step_identity_and_session_metrics(tmp_path, monkeypatch):
    functions = _load_adapter_functions()
    monkeypatch.setenv("RELAX_SESSION_ID", "session-1")
    monkeypatch.setenv("OSWORLD_REWARD_KEY", "score")
    records = [
        {"messages": [], "turn_index": 3, "parse_attempt": i, "accepted_for_execution": i == 1} for i in range(2)
    ]
    output = tmp_path / "output.jsonl"
    functions["write_session_output"](
        output,
        {
            "_turn_records": records,
            "reward": 1,
            "metadata": {"agent_turns": 4, "_session_metrics": {"osworld/environment_steps": 4}},
        },
    )
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert [row["metadata"]["turn_index"] for row in rows] == [3, 3]
    assert [row["metadata"]["export_index"] for row in rows] == [0, 1]
    assert rows[1]["metadata"]["session_id"] == "session-1"
    assert rows[1]["metadata"]["_session_metrics"] == {"osworld/environment_steps": 4}
    assert rows[0]["reward"] == {"score": 1.0}
    assert "reward" not in rows[1]


def test_voluntary_failure_at_last_step_is_not_forced_cap():
    functions = _load_adapter_functions()
    agent = SimpleNamespace(turn_records=[], max_steps=30)

    def predict(*args, **kwargs):
        agent.turn_records.append({})
        return "terminate", ["FAIL"], {"action_type": "gui", "parsed_action": {"action": "FAIL"}}

    agent.predict = predict
    obs = SimpleNamespace(screenshot_bytes=b"", cur_app=None, app_info=None, tool_list=[])
    result = asyncio.run(functions["_predict_one_step"](agent, "task", obs, 29))
    assert result[2]["forced_max_turns"] is False
    assert agent.turn_records[0]["accepted_for_execution"] is True
