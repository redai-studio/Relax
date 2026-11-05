# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""CPU coverage for portable ReTool launch, isolation and job ownership."""

import asyncio
import json
import os
import platform
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from examples.retool import landlock_service, train
from examples.retool.landlock_sandbox import run_isolated
from examples.retool.sandbox import SandboxClient


ROOT = Path(__file__).resolve().parents[2]


def test_retool_sandbox_python_limits_highs_without_rewriting_payload(tmp_path):
    # Exercise the real import/launcher boundary; the recording module replaces
    # the pinned native extension, not the policy that injects its options.
    package = tmp_path / "scipy/optimize/_highs"
    package.mkdir(parents=True)
    for directory in [package, package.parent, package.parent.parent]:
        (directory / "__init__.py").touch()
    (package / "_highs_wrapper.py").write_text("def _highs_wrapper(*args):\n    return args[-1]\n")
    payload = tmp_path / "payload.py"
    source = (
        "from scipy.optimize._highs._highs_wrapper import _highs_wrapper\n"
        "import json, sys\n"
        "print(json.dumps(_highs_wrapper(None, {'threads': 16, 'presolve': True})))\n"
        "assert __name__ == '__main__' and sys.argv[1] == 'argument'\n"
    )
    payload.write_text(source)
    result = subprocess.run(
        [sys.executable, str(ROOT / "examples/retool/sandbox_python.py"), str(payload), "argument"],
        env=dict(os.environ, PYTHONPATH=str(tmp_path)),
        capture_output=True,
        text=True,
        check=True,
    )
    assert json.loads(result.stdout) == {"threads": 1, "presolve": True}
    assert payload.read_text() == source


@pytest.mark.parametrize("exit_code", [0, 7])
def test_retool_job_owns_service_through_training_hard_exit(tmp_path, monkeypatch, exit_code):
    monkeypatch.delenv("SANDBOX_FUSION_URL", raising=False)
    monkeypatch.setattr(train, "tool_service", landlock_service.local_service)
    monkeypatch.setenv("RELAX_PROPAGATE_ENV_VARS", "CUSTOM_ENV")
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setattr(landlock_service, "validate_runtime", lambda root: None)
    monkeypatch.setattr(
        landlock_service,
        "execute",
        lambda code, root: dict(status="Success", run_result=dict(status="Finished", stdout="2", stderr="")),
    )
    captured = {}
    original_popen = subprocess.Popen

    def popen(command, *, env):
        captured.update(env=env, command=command)
        # Exercise a real hard-exiting training process after an HTTP request.
        program = """
import json, os, urllib.request
request = urllib.request.Request(os.environ['SANDBOX_FUSION_URL'],
    data=json.dumps(dict(language='python', code='print(2)', run_timeout=30, memory_limit_MB=1024)).encode())
opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
with opener.open(request, timeout=5) as response:
    assert json.load(response)['run_result']['stdout'] == '2'
os._exit(int(os.environ['TEST_EXIT_CODE']))
"""
        return original_popen([sys.executable, "-c", program], env=dict(env, TEST_EXIT_CODE=str(exit_code)))

    monkeypatch.setattr(subprocess, "Popen", popen)
    monkeypatch.setattr(sys, "argv", ["retool.train", "--num-rollout", "200"])
    with pytest.raises(SystemExit) as error:
        train.main()
    assert error.value.code == exit_code
    assert captured["command"][1:] == ["-m", "relax.entrypoints.train", "--num-rollout", "200"]
    env = captured["env"]
    assert set(env["RELAX_PROPAGATE_ENV_VARS"].split(",")) == {"CUSTOM_ENV", "SANDBOX_FUSION_URL", "DATA_DIR"}
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with pytest.raises(urllib.error.URLError):
        opener.open(env["SANDBOX_FUSION_URL"], timeout=1)


def test_retool_job_closes_training_child_on_interrupt(monkeypatch):
    monkeypatch.setenv("SANDBOX_FUSION_URL", "http://external-sandbox/run_code")
    calls = []

    class Process:
        def wait(self, timeout=None):
            if timeout is None:
                raise SystemExit(143)
            calls.append(("wait", timeout))

        def poll(self):
            return None

        def terminate(self):
            calls.append("terminate")

    monkeypatch.setattr(subprocess, "Popen", lambda *args, **kwargs: Process())
    with pytest.raises(SystemExit) as error:
        train.main()
    assert error.value.code == 143
    assert calls == ["terminate", ("wait", 60)]


def test_retool_guard_failure_stops_before_training(monkeypatch):
    monkeypatch.delenv("SANDBOX_FUSION_URL", raising=False)
    monkeypatch.setattr(train, "tool_service", landlock_service.local_service)

    def unsupported(root):
        raise RuntimeError("Landlock unavailable")

    monkeypatch.setattr(landlock_service, "validate_runtime", unsupported)
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: pytest.fail("Training started before guard validation"))
    with pytest.raises(RuntimeError, match="Landlock unavailable"):
        train.main()


def test_retool_service_keeps_infrastructure_failure_as_http_error(monkeypatch):
    monkeypatch.setattr(landlock_service, "validate_runtime", lambda root: None)

    def broken(code, root):
        raise RuntimeError("No guard attestation")

    monkeypatch.setattr(landlock_service, "execute", broken)
    with landlock_service.local_service() as url:
        with pytest.raises(urllib.error.HTTPError) as error:
            asyncio.run(SandboxClient(url).execute("print(2)"))
        assert error.value.code == 500


@pytest.mark.skipif(os.getuid() != 0 or platform.machine() != "x86_64", reason="Requires Linux x86_64 root driver")
def test_retool_real_landlock_service_is_stateless(tmp_path):
    # /tmp is traversable by uid65534; pytest's private parent directories are not.
    import tempfile

    with tempfile.TemporaryDirectory(dir="/tmp") as directory:
        root = Path(directory)
        root.chmod(0o711)
        probe = run_isolated("print(2)", root)
    if probe["guard"] is None and "Landlock ABI: Function not implemented" in probe["stderr"]:
        pytest.skip("Current kernel does not implement Landlock")
    landlock_service.check_guard(probe)
    with landlock_service.local_service() as url:
        client = SandboxClient(url)
        first = asyncio.run(client.execute("x = 123; print(x)"))
        second = asyncio.run(client.execute("print(x)"))
        denied = asyncio.run(client.execute("import socket; socket.socket()"))
    assert first == dict(status="ok", stdout="123\n", stderr="")
    assert second["status"] == denied["status"] == "code_error"


@pytest.mark.parametrize("mode", ["sft", "grpo"])
def test_retool_scripts_use_local_entrypoint_and_stable_checkpoint_paths(tmp_path, mode):
    tools = tmp_path / "bin"
    tools.mkdir()
    record = tmp_path / "ray.jsonl"
    # Run the actual local.sh control flow while replacing all process/cluster
    # operations, so this regression cannot touch a developer's running Ray.
    for name in ["pkill", "sleep", "nvidia-smi"]:
        path = tools / name
        path.write_text("#!/bin/sh\nexit 0\n")
        path.chmod(0o755)
    ray = tools / "ray"
    ray.write_text(
        f"#!{sys.executable}\nimport json, os, sys\n"
        "with open(os.environ['TEST_RAY_RECORD'], 'a') as f:\n"
        "    f.write(json.dumps(sys.argv[1:]) + '\\n')\n"
    )
    ray.chmod(0o755)
    date = tools / "date"
    date.write_text('#!/bin/sh\nprintf "%s\\n" "$TEST_NOW"\n')
    date.chmod(0o755)
    model, data, save = [tmp_path / name for name in ("model with spaces", "data", "save")]
    model.mkdir()
    (model / "config.json").write_text("{}")
    (data / "sft").mkdir(parents=True)
    (data / "train.jsonl").write_text("{}\n")
    (data / "sft/rows.jsonl").write_text("{}\n")
    env = {key: value for key, value in os.environ.items() if not key.startswith(("RAY_", "RELAX_", "SANDBOX_"))}
    env.update(
        PATH=f"{tools}:{os.environ['PATH']}",
        MODEL_DIR=str(model),
        DATA_DIR=str(data),
        SAVE_DIR=str(save),
        TEST_RAY_RECORD=str(record),
        EXP_NAME=f"retool-test-{mode}",
    )
    for now in ["first", "second"]:
        result = subprocess.run(
            ["bash", str(ROOT / f"examples/retool/scripts/run-qwen25-7B-{mode}-8xgpu.sh")],
            env=dict(env, TEST_NOW=now),
            cwd=tmp_path,
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0, result.stdout + result.stderr
    commands = [json.loads(line) for line in record.read_text().splitlines()]
    assert sum(command[0] == "start" for command in commands) == 2
    submitted = [command for command in commands if command[:2] == ["job", "submit"]]
    assert len(submitted) == 2
    for command in submitted:
        assert "--no-wait" not in command
        assert command[command.index("--load") + 1] == str(save / f"retool-test-{mode}")
        assert command[command.index("--save") + 1] == str(save / f"retool-test-{mode}")
        assert command[command.index("--hf-checkpoint") + 1] == str(model)
        runtime = json.loads(next(item.split("=", 1)[1] for item in command if item.startswith("--runtime-env-json=")))
        assert runtime["env_vars"]["DATA_DIR"] == str(data)
    assert submitted[0][submitted[0].index("--tb-experiment-name") + 1].endswith("first")
    assert submitted[1][submitted[1].index("--tb-experiment-name") + 1].endswith("second")
