# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Opt-in real HTTP/Serve/Ray smoke, isolated from other tests' runtimes."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.skipif(os.environ.get("RUN_AGENTIC_HTTP_RAY_SMOKE") != "1", reason="Opt-in local HTTP/Ray CPU smoke")
def test_session_export_over_http_and_ray(tmp_path: Path) -> None:
    import psutil

    root = Path(__file__).resolve().parents[2]
    output = tmp_path / "smoke.json"
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "scripts.benchmarks.smoke_session_export_http_ray",
            "--json-out",
            str(output),
        ],
        cwd=root,
        env={
            **os.environ,
            "RAY_ADDRESS": "local",
            "PYTHONPATH": str(root),
            "PYTHONNOUSERSITE": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "OMP_NUM_THREADS": "1",
        },
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        stdout, _ = process.communicate(timeout=240)
    except subprocess.TimeoutExpired:
        # Kill only descendants of this smoke driver, never another cluster.
        descendants = psutil.Process(process.pid).children(recursive=True)
        process.kill()
        for child in descendants:
            try:
                child.kill()
            except psutil.NoSuchProcess:
                pass
        psutil.wait_procs(descendants, timeout=10)
        stdout, _ = process.communicate()
        pytest.fail(f"HTTP/Ray smoke timed out:\n{stdout}")
    assert process.returncode == 0, stdout
    result = json.loads(output.read_text())
    assert result["num_gpus"] == result["runtime_processes_remaining"] == 0
    assert result["training_content_equal"]
    assert [row["mode"] for row in result["runs"]] == ["inline", "thread"]
    for row in result["runs"]:
        assert row["completed_exports"] == 4
        assert row["tokens_per_export"] == 131072
        assert row["short_requests"]["count"] == row["health_requests"]["count"] == 40
        assert all(value == 0 for value in row["cleanup"].values())
    assert result["runs"][1]["lifecycle"] == [
        "sse_disconnect_then_drop",
        "ray_export_cancel",
        "materialization_error_then_success",
        "drop_active_and_queued",
        "cancelled_shutdown_active_and_queued",
        "http_during_gated_export",
    ]
