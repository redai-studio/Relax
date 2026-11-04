# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Check operator-managed exclusive-cluster launches through both recipe
entrypoints."""

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[3]
RECIPES = (
    "examples/models/qwen3.8-flash-next/scripts/run-qwen38-qsa-sft-256k-32xgpu-blackwell.sh",
    "examples/models/qwen3.8-flash-next/scripts/run-qwen38-qsa-sft-32xgpu-blackwell.sh",
)
ENTRYPOINT = "scripts/entrypoint/ray-job.sh"


@pytest.fixture(params=RECIPES, ids=["256k", "standard"])
def recipe(request: pytest.FixtureRequest) -> str:
    return request.param


@pytest.fixture
def recipe_env(tmp_path: Path, recipe: str) -> dict[str, str]:
    for relative in (recipe, ENTRYPOINT, "scripts/models/qwen38-flash-next.sh", "scripts/entrypoint/kernel-cache.sh"):
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
if args[:2] == ["list", "nodes"]:
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
def test_sft_recipe_submits_without_querying_other_jobs(
    tmp_path: Path, recipe_env: dict[str, str], recipe: str, via_entrypoint: bool
) -> None:
    command = ["bash", str(tmp_path / recipe)]
    if via_entrypoint:
        command.insert(1, str(tmp_path / ENTRYPOINT))
    result = subprocess.run(command, cwd=tmp_path, env=recipe_env, capture_output=True, text=True, timeout=30)
    trace = Path(recipe_env["MOCK_RAY_TRACE"])
    assert trace.exists(), result.stderr
    calls = [json.loads(line) for line in trace.read_text().splitlines()]
    job_calls = [call for call in calls if call[0] == "job"]
    assert job_calls and all(call[:2] == ["job", "submit"] for call in job_calls), result.stderr
    for call in job_calls:
        if "--address" in call:
            address = call[call.index("--address") + 1]
        else:
            address = next(arg.split("=", 1)[1] for arg in call if arg.startswith("--address="))
        assert address == recipe_env["RAY_JOBS_ADDRESS"]
    submissions = [call for call in job_calls if call[:2] == ["job", "submit"]]
    assert result.returncode == 0, result.stderr
    assert len(submissions) == 1
    assert "--no-wait" not in submissions[0]
    assert sum(call[:2] == ["list", "nodes"] for call in calls) == 1


def test_sft_recipe_dry_run_does_not_contact_ray(tmp_path: Path, recipe_env: dict[str, str], recipe: str) -> None:
    recipe_env.update(DRY_RUN="1")
    result = subprocess.run(
        ["bash", str(tmp_path / recipe)],
        cwd=tmp_path,
        env=recipe_env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert "--seq-length" in json.loads(result.stdout)["argv"]
    assert not Path(recipe_env["MOCK_RAY_TRACE"]).exists()
