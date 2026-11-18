# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Real example entrypoints and SIGTERM with resource/model boundaries
replaced.

No containers, model server or GPUs are started. Existing tests cover each
adapter; these regressions cover unwinding the actual application finally path.
"""

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[3]
EXAMPLES = {
    "deep": "deepeyes_v2_agentic",
    "webshop": "on_policy_distillation/agentic_opd/webshop",
}

BOOTSTRAP = r"""
import asyncio, json, os, sys, time, types
from pathlib import Path
kind, mode = sys.argv[1:3]
base = Path(sys.argv[3])
sys.path.insert(0, str(Path(sys.argv[4])))
def close_resource():
    (base / "closed").touch()
class Env:
    turn = 0
    def __init__(self, **kwargs): pass
    def reset(self, *args):
        if mode == "term":
            (base / "ready").touch()
            while True: time.sleep(0.01)
        return ("task", []) if kind == "webshop" else "task"
    def close(self): close_resource()
if kind == "deep":
    async def close(self): close_resource()
    Env.close = close
async def complete(**kwargs):
    if mode == "term":
        (base / "ready").touch()
        while True: await asyncio.sleep(0.01)
    return types.SimpleNamespace(choices=[types.SimpleNamespace(
        message=types.SimpleNamespace(content="answer"), finish_reason="length")],
        usage=types.SimpleNamespace(completion_tokens=1))
class Model:
    def __init__(self, **kwargs):
        self.chat = types.SimpleNamespace(completions=types.SimpleNamespace(create=complete))
import openai
openai.AsyncOpenAI = Model
if kind == "deep":
    module = types.ModuleType("app.env_deepeyes_v2")
    module.DeepEyesV2Env = Env
    module.ToolObs = object
    module.encode_image_data_uri = lambda *args: ""
    module.extract_answer = lambda *args: None
    module.extract_tool_call = lambda *args: None
    sys.modules[module.__name__] = module
    module = types.ModuleType("app.sandboxes")
    module.SandboxExecutor = object
    module.get_sandbox_backend = lambda *args, **kwargs: None
    sys.modules[module.__name__] = module
import app.agent as agent
if kind == "deep": agent._build_executor = lambda *args: None
if kind == "webshop":
    agent.WebshopClient = Env
    agent.prompt.extract_task = lambda *args: "task"
    agent.prompt.format_obs = lambda *args: "task"
    agent.prompt.build_prompt = lambda **kwargs: "task"
sys.argv = ["agent", "--input-json", str(base / "input.json"), "--output-json", str(base / "output.json")]
agent.main()
"""


def await_file(path, process):
    deadline = time.monotonic() + 10
    while not path.exists() and process.poll() is None and time.monotonic() < deadline:
        time.sleep(0.01)
    assert path.exists(), process.communicate(timeout=2)


@pytest.mark.parametrize("kind", EXAMPLES)
@pytest.mark.parametrize("mode", ["normal", "term"])
def test_example_cleans_resources_on_completion_and_root_term(tmp_path, kind, mode):
    (tmp_path / "input.json").write_text(
        json.dumps(
            {
                "messages": [{"role": "user", "content": "task"}],
                "metadata": {"goal_idx": 0, "game_file": "task.z8"},
            }
        )
    )
    env = {
        **os.environ,
        "OPENAI_API_KEY": "test",
        "OPENAI_BASE_URL": "http://unused",
        "WEBSHOP_URL": "http://unused",
        "RELAX_SESSION_ID": "test",
    }
    process = subprocess.Popen(
        [sys.executable, "-c", BOOTSTRAP, kind, mode, str(tmp_path), str(ROOT / "examples" / EXAMPLES[kind])],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        if mode == "term":
            await_file(tmp_path / "ready", process)
            process.send_signal(signal.SIGTERM)
        _, stderr = process.communicate(timeout=10)
        assert process.returncode == (143 if mode == "term" else 0), stderr
        assert (tmp_path / "closed").exists()
        assert (tmp_path / "output.json").exists() == (mode != "term")
    finally:
        if process.poll() is None:
            process.kill()
            process.communicate(timeout=3)


def test_codex_cancel_allows_sandbox_to_handle_term(tmp_path):
    child = f"""import signal,time
from pathlib import Path
def stop(*_):
    Path({str(tmp_path / "closed")!r}).touch()
    raise SystemExit(0)
signal.signal(signal.SIGTERM,stop)
Path({str(tmp_path / "ready")!r}).touch()
while True: time.sleep(0.01)
"""
    bootstrap = f"""import importlib.util,signal,sys
spec=importlib.util.spec_from_file_location("codex_agent",{str(ROOT / "examples/codex/r2e_gym/agent.py")!r})
agent=importlib.util.module_from_spec(spec)
spec.loader.exec_module(agent)
signal.signal(signal.SIGTERM,agent._handle_sigterm)
with open({str(tmp_path / "sandbox.log")!r},"w") as log:
    agent._run_sandbox([sys.executable,"-c",{child!r}],input="task",log_file=log,timeout=15)
"""
    process = subprocess.Popen([sys.executable, "-c", bootstrap], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        await_file(tmp_path / "ready", process)
        process.send_signal(signal.SIGTERM)
        _, stderr = process.communicate(timeout=10)
        assert process.returncode == 143, stderr
        assert (tmp_path / "closed").exists()
    finally:
        if process.poll() is None:
            process.kill()
            process.communicate(timeout=3)
