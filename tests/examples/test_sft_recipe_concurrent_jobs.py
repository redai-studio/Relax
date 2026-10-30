# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Check concurrent-job protection through both recipe entrypoints."""

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
RECIPE = "examples/models/qwen3.8-flash-next/scripts/run-qwen38-qsa-sft-256k-32xgpu-blackwell.sh"
ENTRYPOINT = "scripts/entrypoint/ray-job.sh"


@pytest.fixture
def recipe_env(tmp_path: Path) -> dict[str, str]:
    for relative in (RECIPE, ENTRYPOINT, "scripts/models/qwen38-flash-next.sh", "scripts/entrypoint/kernel-cache.sh"):
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, target)

    bindir = tmp_path / "bin"
    bindir.mkdir()
    ray = bindir / "ray"
    ray.write_text(
        r"""#!/usr/bin/env python3
import json
import os
import sys

args = sys.argv[1:]
with open(os.environ["MOCK_RAY_TRACE"], "a") as trace:
    trace.write(json.dumps(args) + "\n")
if args[:2] == ["job", "list"]:
    status = os.environ["MOCK_RAY_JOB_STATUS"]
    if status == "unavailable":
        sys.exit("Ray Jobs endpoint unavailable")
    if status == "running":
        print("JobDetails(status='RUNNING', submission_id='existing-training')")
    else:
        print("[]")
elif args[:2] == ["list", "nodes"]:
    print('[{"state":"ALIVE","resources_total":{"GPU":8},"is_head_node":true,"node_ip":"127.0.0.1"}]')
elif args[:2] != ["job", "submit"]:
    sys.exit("Unexpected Ray action: " + repr(args))
"""
    )
    ray.chmod(0o755)
    for name, body in (
        ("jq", '#!/bin/sh\ncat >/dev/null\nprintf "127.0.0.1\\n"\n'),
        ("nvidia-smi", "#!/bin/sh\nexit 1\n"),
    ):
        target = bindir / name
        target.write_text(body)
        target.chmod(0o755)

    env = {key: value for key, value in os.environ.items() if key in ("PATH", "HOME", "LANG", "TMPDIR")}
    env.update(
        PATH=str(bindir) + os.pathsep + env["PATH"],
        MOCK_RAY_TRACE=str(tmp_path / "ray-trace.jsonl"),
        MODEL_DIR=str(tmp_path / "model"),
        DATA_DIR=str(tmp_path / "data"),
        SAVE_DIR=str(tmp_path / "output"),
        MEGATRON=str(tmp_path / "megatron"),
        RELAX_HYBRID_EP_COMMIT="image-pinned",
        RELAX_SKIP_CLUSTER_CLEANUP="1",
        RAY_JOBS_ADDRESS="http://127.0.0.1:18265",
    )
    return env


@pytest.mark.parametrize("via_entrypoint", [False, True], ids=["direct", "ray-job"])
@pytest.mark.parametrize("job_status", ["running", "unavailable", "idle"])
def test_sft_recipe_checks_jobs_before_submission(
    tmp_path: Path, recipe_env: dict[str, str], via_entrypoint: bool, job_status: str
) -> None:
    recipe_env["MOCK_RAY_JOB_STATUS"] = job_status
    command = ["bash", str(tmp_path / RECIPE)]
    if via_entrypoint:
        command.insert(1, str(tmp_path / ENTRYPOINT))
    result = subprocess.run(command, cwd=tmp_path, env=recipe_env, capture_output=True, text=True, timeout=30)
    trace = Path(recipe_env["MOCK_RAY_TRACE"])
    assert trace.exists(), result.stderr
    calls = [json.loads(line) for line in trace.read_text().splitlines()]
    job_calls = [call for call in calls if call[0] == "job"]
    assert job_calls and job_calls[0][:2] == ["job", "list"], result.stderr
    for call in job_calls:
        assert call[call.index("--address") + 1] == recipe_env["RAY_JOBS_ADDRESS"]
    submissions = [call for call in job_calls if call[:2] == ["job", "submit"]]
    if job_status == "idle":
        assert result.returncode == 0, result.stderr
        assert len(submissions) == 1
        assert "--no-wait" not in submissions[0]
        assert sum(call[:2] == ["list", "nodes"] for call in calls) == 1
    else:
        assert result.returncode == (2 if job_status == "running" else 1), result.stderr
        assert not submissions
        expected = "refusing concurrent training" if job_status == "running" else "Ray Jobs endpoint unavailable"
        assert expected in result.stderr


def test_sft_recipe_dry_run_does_not_contact_ray(tmp_path: Path, recipe_env: dict[str, str]) -> None:
    recipe_env.update(DRY_RUN="1", MOCK_RAY_JOB_STATUS="running")
    result = subprocess.run(
        ["bash", str(tmp_path / RECIPE)],
        cwd=tmp_path,
        env=recipe_env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert "--seq-length" in json.loads(result.stdout)["argv"]
    assert not Path(recipe_env["MOCK_RAY_TRACE"]).exists()
