# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


SMOKE_SCRIPT = Path(__file__).resolve().parents[3] / "examples/deepeyes_v2_agentic/scripts/smoke_search_offline.py"


@pytest.mark.parametrize("scenario", ["mock", "retriever-error", "external-error"])
def test_offline_smoke_completes_without_network(tmp_path: Path, scenario: str) -> None:
    output_dir = tmp_path / "smoke results"
    environment = {
        **os.environ,
        "TMPDIR": str(tmp_path),
        "DEEPEYES_V2_SEARCH_CONFIG_PATH": str(tmp_path / "missing.yaml"),
        "OPENAI_API_KEY": "parent-private-marker",
        "BRAVE_SEARCH_API_KEY": "parent-private-marker",
    }
    command = [sys.executable, str(SMOKE_SCRIPT), "--output-dir", str(output_dir)]
    if scenario != "mock":
        command.extend(["--scenario", scenario])

    process = subprocess.run(command, cwd=tmp_path, env=environment, capture_output=True, text=True, timeout=30)

    assert process.returncode == 0, process.stderr
    report = json.loads(process.stdout)
    assert report == json.loads((output_dir / "report.json").read_text(encoding="utf-8"))
    assert report["scenario"] == scenario
    assert report["model_requests"] == 2
    assert report["search_calls"] == 1
    assert report["search_http_requests"] == (0 if scenario == "mock" else 3)
    assert report["network_attempts"] == 0
    assert report["environment_closed"] and report["search_clients_closed"] and report["model_clients_closed"]
    assert report["stop_reason"] == "env_done"
    assert report["last_error"] == (None if scenario == "mock" else "search_failed")
    assert report["final_answer"]
    requests = json.loads((output_dir / "model_requests.json").read_text(encoding="utf-8"))
    assert requests[1]["messages"][:-2] == requests[0]["messages"]
    assert requests[1]["messages"][-1] == {"role": "tool", "content": report["observation"]}
    if scenario == "mock":
        assert "[mock]" in report["observation"]
    else:
        assert report["observation"].startswith("Error:")
    assert "parent-private-marker" not in process.stdout + process.stderr
