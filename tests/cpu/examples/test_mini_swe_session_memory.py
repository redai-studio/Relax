# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Bounded real-kernel fault tests; opt in on a delegated cgroup-v1 host.

Mocks cannot prove fork inheritance, descendant OOM detection, or isolation
from a healthy sibling. These tests use at most 256 MiB of requested test
allocation.
"""

import importlib
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest


EXAMPLE = Path(__file__).resolve().parents[3] / "examples/mini_swe_agent"
sys.path.insert(0, str(EXAMPLE))
memory = importlib.import_module("session_memory")


def test_memory_configuration_fails_closed():
    memory.validate_memory_limit(0, supervised=False)
    for limit, supervised in ((-1, True), (128, False)):
        with pytest.raises(ValueError):
            memory.validate_memory_limit(limit, supervised=supervised)


@pytest.mark.skipif(
    os.environ.get("RUN_SESSION_CGROUP_TESTS") != "1",
    reason="Explicit delegated cgroup-v1 host required; no GPU needed",
)
@pytest.mark.parametrize("scenario", ["oom", "cancel", "truncated", "nonzero"])
def test_private_memory_group_covers_orphans_and_overrides_stale_success(tmp_path, scenario):
    oom = scenario == "oom"
    memory.validate_memory_limit(128, supervised=True)
    sibling_group = memory.SessionMemory.create(128)
    sibling = subprocess.Popen(sibling_group.wrap([sys.executable, "-c", "import time;time.sleep(60)"]))
    cli = tmp_path / "apptainer"
    cli.write_text("#!/bin/sh\necho 'INSTANCE PID'\n")
    cli.chmod(0o755)
    spec, report, child_file = tmp_path / "spec.json", tmp_path / "report.json", tmp_path / "child.json"
    spec.write_text(
        json.dumps(
            {
                "apptainer_args": [str(cli), []],
                "instance_name": "mswe-test",
                "record": {"session_id": "client"},
                "release_nonce": "nonce",
                "memory_limit_mib": 128,
            }
        )
    )
    hog = (
        "import os,time,signal;from pathlib import Path;"
        "signal.signal(signal.SIGTERM,signal.SIG_IGN);"
        f"Path({str(child_file)!r}).write_text(str(os.getpid()));"
        + ("data=[bytearray(1024*1024) for _ in range(256)];" if oom else "")
        + "time.sleep(60)"
    )
    # The intermediate process exits, making the detached grandchild an orphan.
    bridge = f"import subprocess,sys;subprocess.Popen([sys.executable,'-c',{hog!r}],start_new_session=True)"
    worker = (
        "import json,subprocess,sys,time;from pathlib import Path;"
        f"Path({str(report.with_suffix('.worker.json'))!r}).write_text(json.dumps({{'info':{{}},'error':''}}));"
        f"subprocess.run([sys.executable,'-c',{bridge!r}],check=True);"
        + (
            f"Path({str(report.with_suffix('.worker.json'))!r}).write_text('{{');time.sleep(0.3);sys.exit(7)"
            if scenario == "truncated"
            else "time.sleep(0.3);sys.exit(7)"
            if scenario == "nonzero"
            else "time.sleep(60)"
        )
    )
    runner = (
        "import sys;from pathlib import Path;"
        f"sys.path.insert(0,{str(EXAMPLE)!r});from session_process import supervise;"
        f"supervise(Path({str(spec)!r}),Path({str(report)!r}),worker_command=[sys.executable,'-c',{worker!r}])"
    )
    supervisor = subprocess.Popen([sys.executable, "-c", runner])
    try:
        deadline = time.monotonic() + 10
        while not child_file.exists() and time.monotonic() < deadline:
            time.sleep(0.02)
        assert child_file.exists()
        if scenario == "cancel":
            supervisor.send_signal(signal.SIGTERM)
        assert supervisor.wait(timeout=20) == 0
        result = json.loads(report.read_text())
        assert result["resources_released"] and result["children_reaped"]
        assert result["memory"]["released"]
        assert not Path(result["memory"]["cgroup"]).exists()
        assert not Path(f"/proc/{child_file.read_text()}").exists()
        assert bool(result["memory"]["oom_kills"]) is oom
        assert ("SessionMemoryLimitExceeded" in result["error"]) is oom
        if scenario == "truncated":
            assert "valid result" in result["error"]
        if scenario == "nonzero":
            assert "code 7" in result["error"]
        assert sibling.poll() is None
        assert sibling_group.snapshot()["oom_kills"] == 0
    finally:
        if supervisor.poll() is None:
            supervisor.terminate()
            supervisor.wait(timeout=40)
        sibling.terminate()
        sibling.wait(timeout=5)
        assert sibling_group.remove_if_empty()
