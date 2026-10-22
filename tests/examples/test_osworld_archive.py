# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import asyncio
import base64
import importlib.util
import json
import sys
from io import BytesIO
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
from PIL import Image

from examples.osworld_agentic.app import archive
from relax.agentic.archive.client import Recorder
from relax.agentic.archive.export import export_archive
from relax.agentic.archive.store import ArchiveStore


ROOT = Path(__file__).resolve().parents[2]


def load_env(monkeypatch):
    stub = ModuleType("desktop_env.desktop_env")
    stub.DesktopEnv = object
    monkeypatch.setitem(sys.modules, "desktop_env", ModuleType("desktop_env"))
    monkeypatch.setitem(sys.modules, "desktop_env.desktop_env", stub)
    name = "examples.osworld_agentic.app.env_osworld"
    spec = importlib.util.spec_from_file_location(name, ROOT / "examples/osworld_agentic/app/env_osworld.py")
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module


def image(size=32):
    buf = BytesIO()
    Image.new("RGB", (size, size), "blue").save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture
def capture(tmp_path, monkeypatch):
    store = ArchiveStore(tmp_path / "library", 1_000_000)

    def rpc(self, operation, payload):
        tid = payload.get("trajectory_id")
        if operation == "begin":
            return {"ok": True, "trajectory_id": store.begin(payload["metadata"])}
        if operation == "event":
            store.append(tid, payload["event"])
        elif operation == "finish":
            return {"ok": True, **store.finish(tid, payload["outcome"])}
        else:
            store.abort(tid)
        return {"ok": True}

    monkeypatch.setattr(Recorder, "_rpc", rpc)
    monkeypatch.setenv("CUA_ARCHIVE_ENABLE", "1")
    monkeypatch.setenv("CUA_ARCHIVE_URL", "http://mock-only")
    monkeypatch.setenv("OSWORLD_RUN_TAG", "run")
    monkeypatch.setenv("RELAX_SESSION_ID", "session")
    cap = archive.create_capture({"id": "task", "instruction": "Open settings"}, {"max_turns": 30}, {})
    yield cap, store
    store.close()


def test_osworld_archive_preserves_retries_multiple_actions_terminal_and_resized_inputs(
    capture, monkeypatch, tmp_path
):
    module = load_env(monkeypatch)
    cap, store = capture
    obs = module.OSWorldEnv._pack_obs(
        {"screenshot": image(), "cur_app": "browser"}, done=False, exec_ok=True, exec_result={}, capture=True
    )
    cap.observe(obs, 0)
    cap.before_predict(0)
    scaled = "data:image/png;base64," + base64.b64encode(image(16)).decode()
    request = {
        "messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": scaled}}]}],
        "tools": [],
        "chat_template_kwargs": {"enable_thinking": True},
    }
    for content in ("bad parse", "click twice"):
        cid = cap.model_start(request, "model", 0)
        cap.model_result(cid, json.dumps({"choices": [{"message": {"role": "assistant", "content": content}}]}), 200)
    cap.parsed(
        [{"parse_attempt": 0, "accepted_for_execution": False}, {"parse_attempt": 1, "accepted_for_execution": True}]
    )
    for action in ("pyautogui.click(1, 2)", 'MCP:click:{"target":"settings"}', "DONE"):
        aid = cap.action_start(action, terminal=action == "DONE")
        cap.action_result(aid, obs)
    cap.finish(1.0, None, "agent_done")
    [trajectory] = store.retained()
    assert len(trajectory["model_calls"]) == 2
    assert trajectory["model_calls"][0]["selected"] is False
    assert trajectory["model_calls"][0]["action_ids"] == []
    assert len(trajectory["model_calls"][1]["action_ids"]) == 3
    assert [a["type"] for a in trajectory["actions"]] == ["pyautogui", "mcp", "terminal"]
    assert len(list((store.root / "assets").iterdir())) == 2  # Original and actual scaled input.
    assert trajectory["actions"][0]["status"] == "unknown"
    assert trajectory["actions"][0]["execution_evidence"] == "rpc_completed"
    manifest = export_archive(store, tmp_path / "export")
    assert manifest["sft_samples"] == 0
    assert manifest["sft_skipped"]["retry_or_correction"] == 2


def test_osworld_archive_evaluator_keeps_reward_behavior_and_captures_error(monkeypatch):
    module = load_env(monkeypatch)
    env = object.__new__(module.OSWorldEnv)
    env._health = None
    env._archive_enabled = True
    env._env = SimpleNamespace(evaluate=lambda: 0.75)
    assert env._evaluate_sync() == 0.75
    assert env.archive_evaluation == {"raw": 0.75, "error": None}

    def fail():
        raise RuntimeError("evaluator failed")

    env._env.evaluate = fail
    assert env._evaluate_sync() == 0.0
    assert "evaluator failed" in env.archive_evaluation["error"]
    placeholder = module.OSWorldEnv._pack_obs({}, done=False, exec_ok=True, exec_result={}, capture=True)
    assert placeholder.screenshot_bytes and not placeholder.screenshot_valid


def test_osworld_archive_disabled_has_no_recorder_or_io(monkeypatch):
    monkeypatch.delenv("CUA_ARCHIVE_ENABLE", raising=False)
    monkeypatch.setattr(archive, "Recorder", lambda *args: pytest.fail("unexpected archive I/O"))
    assert archive.create_capture({}, {}, {}) is None


def test_osworld_actual_model_hook_records_transport_retry_without_changing_payload(capture, monkeypatch):
    from copy import deepcopy

    import httpx

    cap, store = capture
    module = load_env(monkeypatch)
    obs = module.OSWorldEnv._pack_obs({"screenshot": image()}, done=False, exec_ok=True, exec_result={}, capture=True)
    cap.observe(obs, 0)
    cap.before_predict(0)
    external = ModuleType("mm_agents.hybrid_agent_local")
    external.HybridAgentLocal = object
    monkeypatch.setitem(sys.modules, "mm_agents", ModuleType("mm_agents"))
    monkeypatch.setitem(sys.modules, "mm_agents.hybrid_agent_local", external)
    name = "examples.osworld_agentic.app.hybrid_agent"
    spec = importlib.util.spec_from_file_location(name, ROOT / "examples/osworld_agentic/app/hybrid_agent.py")
    hybrid = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(hybrid)
    agent = object.__new__(hybrid.RelaxHybridAgent)
    agent.archive = cap
    agent._base_url, agent._api_key = "http://mock-model", "session"
    agent._max_llm_call_retries, agent._stop_token_ids = 2, []
    agent._llm_call_timeout_s, agent._backoff_seconds = 1, 0
    agent.transport_retries, agent.model_request_s = 0, 0
    agent._input_toks_per_step, agent._output_toks_per_step, agent.turn_records = [], [], []
    payload = {"messages": [{"role": "user", "content": "task"}], "tools": [], "chat_template_kwargs": {}}
    original = deepcopy(payload)
    requests = []

    def post(*args, **kwargs):
        requests.append(deepcopy(kwargs["json"]))
        if len(requests) == 1:
            raise httpx.ConnectError("temporary")
        return httpx.Response(
            200, json={"choices": [{"message": {"role": "assistant", "content": "DONE"}, "finish_reason": "stop"}]}
        )

    monkeypatch.setattr(hybrid.httpx, "post", post)
    assert agent.call_llm(payload, "model") == "DONE"
    assert payload == original and requests == [original, original]
    assert len(agent.turn_records) == 1 and agent.transport_retries == 1
    cap.parsed([{"parse_attempt": 0, "accepted_for_execution": True}])
    aid = cap.action_start("DONE", terminal=True)
    cap.action_result(aid, obs)
    cap.finish(1, None, "agent_done")
    [trajectory] = store.retained()
    assert [c["transport_retry"] for c in trajectory["model_calls"]] == [0, 1]
    assert trajectory["model_calls"][0]["response"]["error"] == "ConnectError: temporary"


def test_osworld_capture_error_does_not_escape(capture, monkeypatch):
    cap, store = capture
    monkeypatch.setattr(store, "append", lambda *a: (_ for _ in ()).throw(OSError("disk full")))
    cap.observe(SimpleNamespace(screenshot_valid=True, screenshot_bytes=image()), 0)
    assert cap.recorder.trajectory_id is None
    assert not store.retained()


@pytest.mark.parametrize("score", [0, 0.5, 0.75, float("nan"), 2])
def test_osworld_partial_unknown_and_failure_not_retained(capture, score):
    cap, store = capture
    cap.finish(score, None, "agent_done")
    assert store.retained() == []


def test_osworld_real_session_hooks_preserve_training_outputs(capture, monkeypatch):
    import ast
    import logging
    import signal

    cap, store = capture
    env_module = load_env(monkeypatch)
    obs = env_module.OSWorldEnv._pack_obs(
        {"screenshot": image()}, done=False, exec_ok=True, exec_result={}, capture=True
    )
    executions = []

    class Env:
        def __init__(self, **kwargs):
            self.archive_evaluation = {"raw": 1, "error": None}

        async def reset(self):
            return obs

        async def exec_pyautogui(self, code):
            executions.append(code)
            return obs

        async def evaluate(self):
            return 1.0

        async def close(self):
            executions.append("close")

    class Agent:
        def __init__(self, **kwargs):
            self.turn_records = []
            self.max_steps = 3
            self.model_request_s = self.transport_retries = 0

        def reset(self, *args):
            pass

        def predict(self, instruction, observations, step_idx):
            payload = {"messages": [{"role": "user", "content": instruction}]}
            active = getattr(self, "archive", None)
            if active:
                cid = active.model_start(payload, "model", 0)
                active.model_result(
                    cid, json.dumps({"choices": [{"message": {"role": "assistant", "content": "act"}}]}), 200
                )
            self.turn_records.append({"messages": [*payload["messages"], {"role": "assistant", "content": "act"}]})
            return "act", ["DONE"] if step_idx else ["click()", "scroll()"], {"action_type": "gui"}

        def record_exec_result(self, *args):
            pass

    app = ModuleType("app")
    app.__path__ = []
    monkeypatch.setitem(sys.modules, "app", app)
    monkeypatch.setitem(sys.modules, "app.archive", archive)
    monkeypatch.setattr(archive, "create_capture", lambda *args: cap)
    path = ROOT / "examples/osworld_agentic/app/agent.py"
    tree = ast.parse(path.read_text())
    nodes = [
        n for n in tree.body if isinstance(n, ast.AsyncFunctionDef) and n.name in {"run_session", "_predict_one_step"}
    ]
    module = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), *nodes],
        type_ignores=[],
    )
    ns = dict(
        asyncio=asyncio,
        signal=signal,
        logging=logging,
        os=__import__("os"),
        time=SimpleNamespace(monotonic=lambda: 1),
        _load_config=lambda: {"max_turns": 3},
        _extract_task_config=lambda s: {"id": "task", "instruction": "task"},
        shotdump=SimpleNamespace(
            init=lambda *a: None, shot=lambda *a: None, turn=lambda *a: None, finish=lambda *a: None
        ),
        OSWorldEnv=Env,
        RelaxHybridAgent=Agent,
        parse_mcp_action=env_module.parse_mcp_action,
        SessionDiscardedError=type("Discarded", (Exception,), {}),
        ContextLengthExceededError=type("ContextError", (Exception,), {}),
        logger=logging.getLogger("test"),
        emit=lambda *a, **k: None,
    )
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), ns)
    with_capture = asyncio.run(ns["run_session"]({}, {}))
    monkeypatch.delenv("CUA_ARCHIVE_ENABLE")
    without_capture = asyncio.run(ns["run_session"]({}, {}))
    assert with_capture == without_capture
    assert executions == ["click()", "scroll()", "DONE", "close"] * 2
    [trajectory] = store.retained()
    assert len(trajectory["model_calls"]) == 2
    assert len(trajectory["actions"]) == 3
    assert trajectory["actions"][-1]["type"] == "terminal"
    assert trajectory["outcome"]["final_observation_id"] == trajectory["actions"][-1]["after_observation_id"]
