# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""The production HTTP/Ray boundaries need a private opt-in runtime."""

import json
import os
import sys
from pathlib import Path

import pytest


@pytest.mark.skipif(sys.platform != "linux", reason="Linux process-tree supervisor")
@pytest.mark.skipif(
    os.environ.get("RELAX_RUN_HTTP_RAY_SMOKE") != "1", reason="Opt-in private CPU HTTP/Ray integration"
)
def test_session_export_http_ray_lifecycle(tmp_path: Path) -> None:
    from scripts.benchmarks.smoke_session_export_http_ray import run_smoke

    result = run_smoke()
    (tmp_path / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    assert result["gpu_count"] == 0
    assert result["owned_processes_reaped"] and result["private_directory_removed"]
    assert result["training_exports_checked"] == 2
    assert result["request_permits"]["grants"] == result["request_permits"]["releases"] >= 8
    assert result["request_permits"]["active_permits"] == 0
    assert result["scenarios"] == [
        "http_chat_and_training_roundtrip",
        "sse_disconnect_then_explicit_drop",
        "delivered_ray_cancel_drains_export",
        "assembly_error_then_recovery",
        "drop_active_and_queued_http_responsive",
        "shutdown_active_and_queued_http_responsive",
    ]
