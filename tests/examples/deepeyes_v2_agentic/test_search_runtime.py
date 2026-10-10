# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml


REPO_ROOT = Path(__file__).resolve().parents[3]
EXAMPLE_DIR = REPO_ROOT / "examples" / "deepeyes_v2_agentic"
FIXTURES_DIR = Path(__file__).parent / "runtime_fixtures"
sys.path.insert(0, str(EXAMPLE_DIR))

from app import search_runtime  # noqa: E402
from app.search_config import SEARCH_CONFIG_ENV, SearchError  # noqa: E402


AUTH_ENV = "SEARCH_RUNTIME_TEST_SEARCH_AUTH"
SECRET = 'test-auth-"quoted"-\\value=token'


def write_config(tmp_path: Path, auth_name: str = AUTH_ENV) -> Path:
    payload = yaml.safe_load((EXAMPLE_DIR / "search_config.brave.yaml").read_text(encoding="utf-8"))
    payload["auth"]["env"] = auth_name
    path = tmp_path / "搜索 'config'.yaml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    return path


@pytest.fixture(autouse=True)
def clean_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (SEARCH_CONFIG_ENV, AUTH_ENV, "RUNTIME_ENV_JSON", "DEEPEYES_V2_BASE_RUNTIME_ENV_JSON"):
        monkeypatch.delenv(name, raising=False)


def test_runtime_default_mock_and_environment_merge(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    assert search_runtime.prepare_search_environment() == {
        SEARCH_CONFIG_ENV: str(EXAMPLE_DIR / "search_config.mock.yaml")
    }
    path = write_config(tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv(SEARCH_CONFIG_ENV, path.name)
    monkeypatch.setenv(AUTH_ENV, SECRET)
    monkeypatch.setenv("BRAVE_SEARCH_API_KEY", "unrelated-token")
    monkeypatch.setenv("DEEPEYES_V2_BASE_RUNTIME_ENV_JSON", json.dumps({"pip": ["base"], "env_vars": {"A": "base"}}))
    monkeypatch.setenv(
        "RUNTIME_ENV_JSON", json.dumps({"pip": ["current"], "env_vars": {"A": "current", "B": "value"}})
    )

    runtime = search_runtime.build_runtime_environment("standard")

    assert runtime["pip"] == ["current"]
    assert runtime["env_vars"]["A"] == "current"
    assert runtime["env_vars"]["B"] == "value"
    assert runtime["env_vars"][SEARCH_CONFIG_ENV] == str(path)
    assert runtime["env_vars"][AUTH_ENV] == SECRET
    assert "BRAVE_SEARCH_API_KEY" not in runtime["env_vars"]


@pytest.mark.parametrize("auth_name", [AUTH_ENV, "OPENAI_API_KEY", "RELAX_SESSION_ID"])
def test_runtime_rejects_missing_or_conflicting_auth(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, auth_name: str
) -> None:
    monkeypatch.setenv(SEARCH_CONFIG_ENV, str(write_config(tmp_path, auth_name)))
    if auth_name != AUTH_ENV:
        monkeypatch.setenv(auth_name, SECRET)
    reason = "missing_search_auth" if auth_name == AUTH_ENV else "invalid_search_auth_env"
    with pytest.raises(SearchError, match=f"^{reason}$"):
        search_runtime.prepare_search_environment()


def stage_runtime(tmp_path: Path) -> tuple[Path, dict[str, str]]:
    example = tmp_path / "repo/examples/deepeyes_v2_agentic"
    for folder in ("app", "scripts", "apptainer_env"):
        (example / folder).mkdir(parents=True)
    for name in (
        "app/__init__.py",
        "app/prompt.py",
        "app/search_config.py",
        "app/search_runtime.py",
        "search_config.mock.yaml",
        "run_agent_app.sh",
        "run_deepeyes_v2_agentic.sh",
        "run_deepeyes_v2_agentic_klx.sh",
        "scripts/run_single_session.py",
    ):
        shutil.copyfile(EXAMPLE_DIR / name, example / name)
    entrypoint = tmp_path / "repo/scripts/entrypoint"
    entrypoint.mkdir(parents=True)
    shutil.copyfile(REPO_ROOT / "scripts/entrypoint/runtime-env-klx.sh", entrypoint / "runtime-env-klx.sh")
    shutil.copyfile(FIXTURES_DIR / "agent_probe.py", example / "app/agent.py")
    shutil.copyfile(FIXTURES_DIR / "environment.sh", example / "env.sh")
    for name in ("sglang_judge_service.sh", "sglang_judge_service_klx.sh"):
        shutil.copyfile(FIXTURES_DIR / "judge.sh", example / name)
    shutil.copyfile(FIXTURES_DIR / "model.sh", tmp_path / "qwen36-35B-A3B.sh")
    commands = tmp_path / "commands"
    commands.mkdir()
    for name in ("find", "ray"):
        shutil.copyfile(FIXTURES_DIR / f"{name}.sh", commands / name)
        (commands / name).chmod(0o755)
    python = tmp_path / "app_env/.venv/bin/python"
    python.parent.mkdir(parents=True)
    shutil.copyfile(FIXTURES_DIR / "python.sh", python)
    python.chmod(0o755)
    (tmp_path / "sif").mkdir()
    (tmp_path / "sif/deepeyes_v2_kernel.sif").touch()
    (tmp_path / "input.json").write_text("{}", encoding="utf-8")
    config = write_config(tmp_path)
    environment = {
        "PATH": os.pathsep.join((str(commands), os.environ["PATH"])),
        "TMPDIR": str(tmp_path),
        "PYTHONPATH": str(REPO_ROOT),
        "PYTHONDONTWRITEBYTECODE": "1",
        "SEARCH_RUNTIME_TEST_PYTHON": sys.executable,
        "SEARCH_RUNTIME_TEST_FIXTURES": str(FIXTURES_DIR),
        "SEARCH_RUNTIME_TEST_TOKEN": SECRET,
        "SEARCH_RUNTIME_TEST_SEARCH_AUTH": SECRET,
        "SEARCH_RUNTIME_TEST_UNRELATED_TOKEN": "unrelated-token",
        "SEARCH_RUNTIME_TEST_JUDGE_TOKEN": "judge-token",
        "SEARCH_RUNTIME_TEST_CONFIG_SOURCE": str(config),
        SEARCH_CONFIG_ENV: str(config),
        "SEARCH_RUNTIME_TEST_RUNTIME_REPORT": str(tmp_path / "runtime.json"),
        "SEARCH_RUNTIME_TEST_EFFECTS_FILE": str(tmp_path / "effects.txt"),
        "SEARCH_RUNTIME_TEST_AGENT_INPUT": str(tmp_path / "input.json"),
        "SEARCH_RUNTIME_TEST_AGENT_OUTPUT": str(tmp_path / "output.json"),
        "RELAX_ENTRYPOINT_MODE": "fixture",
        "MODEL_CONFIG_DIR": str(tmp_path),
        "MODEL_DIR": str(tmp_path / "models"),
        "DATA_DIR": str(tmp_path),
        "SAVE_DIR": str(tmp_path / "save"),
        "WORKDIR": str(tmp_path / "work"),
        "CONDA_PREFIX": str(tmp_path / "conda"),
        "DEEPEYES_V2_APP_ENV_ROOT": str(tmp_path / "app_env"),
        "CPU_THREADS_PER_ACTOR": "6",
        "WANDB_API_KEY": "wandb-token",
        "OPENAI_BASE_URL": "https://model.example.test/v1",
        "OPENAI_API_KEY": "model-token",
        "RUNTIME_ENV_JSON": json.dumps({"pip": ["fixture-package"], "env_vars": {"BASE": "preserved"}}),
    }
    return example, environment


def run_entry(command: list[str], tmp_path: Path, environment: dict[str, str]) -> subprocess.CompletedProcess[str]:
    process = subprocess.run(command, cwd=tmp_path, env=environment, capture_output=True, text=True, timeout=30)
    logs = process.stdout + process.stderr
    for path in tmp_path.rglob("*.log"):
        logs += path.read_text(encoding="utf-8")
    assert all(marker not in logs for marker in ("test-auth", "unrelated-token", "judge-token", "wandb-token"))
    return process


def read_output(tmp_path: Path) -> dict[str, Any]:
    return json.loads((tmp_path / "output.json").read_text(encoding="utf-8"))["metadata"]


@pytest.mark.parametrize("profile,backend", [("standard", "external"), ("klx", "external"), ("standard", "mock")])
def test_training_entry_passes_configuration_without_logging_secrets(
    tmp_path: Path, profile: str, backend: str
) -> None:
    example, environment = stage_runtime(tmp_path)
    if backend == "mock":
        environment.pop("SEARCH_RUNTIME_TEST_CONFIG_SOURCE")
        environment.pop("RUNTIME_ENV_JSON")
    script = "run_deepeyes_v2_agentic_klx.sh" if profile == "klx" else "run_deepeyes_v2_agentic.sh"

    process = run_entry(["bash", "-x", str(example / script)], tmp_path, environment)

    assert process.returncode == 0, process.stderr
    runtime = json.loads((tmp_path / "runtime.json").read_text(encoding="utf-8"))
    values = runtime["env_vars"]
    expected_path = (
        environment[SEARCH_CONFIG_ENV] if backend == "external" else str(example / "search_config.mock.yaml")
    )
    assert values[SEARCH_CONFIG_ENV] == expected_path
    assert "BRAVE_SEARCH_API_KEY" not in values
    assert values["DEEPEYES_JUDGE_API_KEY"] == "judge-token"
    if backend == "external":
        assert values[AUTH_ENV] == SECRET
        assert runtime["pip"] == ["fixture-package"]
        assert values["BASE"] == "preserved"
    else:
        assert AUTH_ENV not in values
    if profile == "klx":
        assert values["CUDA_DEVICE_MAX_CONNECTIONS"] == "1"
        assert values["OMP_NUM_THREADS"] == "6"
        assert values["WANDB_API_KEY"] == "wandb-token"
    assert read_output(tmp_path) == {"backend": backend, "config_path": expected_path, "auth": values.get(AUTH_ENV)}


@pytest.mark.parametrize("profile", ["standard", "klx"])
def test_training_entry_rejects_missing_auth_before_services(tmp_path: Path, profile: str) -> None:
    example, environment = stage_runtime(tmp_path)
    environment["SEARCH_RUNTIME_TEST_TOKEN"] = ""
    script = "run_deepeyes_v2_agentic_klx.sh" if profile == "klx" else "run_deepeyes_v2_agentic.sh"

    process = run_entry(["bash", "-x", str(example / script)], tmp_path, environment)

    assert process.returncode == 1
    assert "search runtime configuration is invalid" in process.stderr
    assert (tmp_path / "effects.txt").read_text(encoding="utf-8").splitlines() == ["find", "find"]
    assert not (tmp_path / "runtime.json").exists()


def test_single_session_passes_relative_configuration_to_agent(tmp_path: Path) -> None:
    example, environment = stage_runtime(tmp_path)
    environment[SEARCH_CONFIG_ENV] = Path(environment[SEARCH_CONFIG_ENV]).name
    command = [
        sys.executable,
        str(example / "scripts/run_single_session.py"),
        "--synthetic",
        "--input-json",
        environment["SEARCH_RUNTIME_TEST_AGENT_INPUT"],
        "--output-json",
        environment["SEARCH_RUNTIME_TEST_AGENT_OUTPUT"],
    ]

    process = run_entry(command, tmp_path, environment)

    assert process.returncode == 0, process.stderr
    assert read_output(tmp_path) == {
        "backend": "external",
        "config_path": environment["SEARCH_RUNTIME_TEST_CONFIG_SOURCE"],
        "auth": SECRET,
    }
    assert not (tmp_path / "input.json").exists()
