# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).parents[2]
SCRIPT = ROOT / "scripts/training/hpc/run-unified-inference-3role.sh"


def _run_script(
    tmp_path: Path,
    *,
    layout: str,
    actor_gpus: int,
    selective: bool = False,
    role_gpus: int | None = None,
    rollout_batch_size: int | None = None,
    n_samples_per_prompt: int | None = None,
    global_batch_size: int | None = None,
) -> subprocess.CompletedProcess:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir(parents=True)
    captured = tmp_path / "python-args.txt"
    fake_python = fake_bin / "python"
    fake_python.write_text('#!/usr/bin/env bash\nprintf \'%s\\n\' "$@" > "$FAKE_ARGS_FILE"\n')
    fake_python.chmod(0o755)

    env = os.environ.copy()
    env.update(
        {
            "MODEL_CONFIG_DIR": str(ROOT / "scripts/models"),
            "MODEL_DIR": str(tmp_path / "model"),
            "INFERENCE_TRAIN_OUTPUT": str(tmp_path / "output"),
            "RAY_DASHBOARD": "http://127.0.0.1:8265",
            "RUNTIME_ENV_JSON": "{}",
            "INFERENCE_LAYOUT": layout,
            "INFERENCE_ACTOR_NODES": "1",
            "INFERENCE_ACTOR_GPUS_PER_NODE": str(actor_gpus),
            "INFERENCE_VALIDATE_ARGS_ONLY": "1",
            "FAKE_ARGS_FILE": str(captured),
            "PATH": f"{fake_bin}:{env['PATH']}",
        }
    )
    if selective:
        env["INFERENCE_SELECTIVE_OFFLOAD"] = "1"
    else:
        env.pop("INFERENCE_SELECTIVE_OFFLOAD", None)
    for name, value in (
        ("INFERENCE_ROLLOUT_GPUS", role_gpus),
        ("INFERENCE_TEACHER_GPUS", role_gpus),
        ("INFERENCE_GENRM_GPUS", role_gpus),
        ("INFERENCE_ROLLOUT_BATCH_SIZE", rollout_batch_size),
        ("INFERENCE_N_SAMPLES_PER_PROMPT", n_samples_per_prompt),
        ("INFERENCE_GLOBAL_BATCH_SIZE", global_batch_size),
    ):
        if value is not None:
            env[name] = str(value)

    result = subprocess.run(["bash", str(SCRIPT)], cwd=ROOT, env=env, capture_output=True, text=True, check=False)
    if captured.exists():
        result.args_captured = captured.read_text().splitlines()  # type: ignore[attr-defined]
    else:
        result.args_captured = []  # type: ignore[attr-defined]
    return result


def test_three_role_decoupled_has_no_empty_layout_expansion(tmp_path):
    result = _run_script(tmp_path, layout="decoupled", actor_gpus=2)

    assert result.returncode == 0, result.stderr
    assert "--colocate" not in result.args_captured
    assert "--offload-train" not in result.args_captured
    assert "--offload-rollout" not in result.args_captured
    assert "--fully-async" in result.args_captured
    assert "--selective-offload" not in result.args_captured
    assert "--opd-teacher-defer" not in result.args_captured
    assert "--defer-reward-to-post-process" not in result.args_captured
    assert "--rm-type" in result.args_captured
    resource = json.loads(result.args_captured[result.args_captured.index("--resource") + 1])
    assert resource["advantages"] == [1, 0]
    assert sum(spec[1] for spec in resource.values()) == 8


def test_three_role_decoupled_rejects_batch_without_actor_forward_producer(tmp_path):
    result = _run_script(tmp_path, layout="decoupled", actor_gpus=1, global_batch_size=8)

    assert result.returncode == 2
    assert "global batch size must equal" in result.stderr


def test_three_role_split_defaults_to_tms_and_allows_selective_opt_in(tmp_path):
    default = _run_script(tmp_path / "default", layout="split", actor_gpus=6)
    selective = _run_script(tmp_path / "selective", layout="split", actor_gpus=6, selective=True)

    assert default.returncode == 0, default.stderr
    assert selective.returncode == 0, selective.stderr
    for flag in ("--colocate", "--offload-train", "--offload-rollout"):
        assert flag in default.args_captured
        assert flag in selective.args_captured
    assert "--selective-offload" not in default.args_captured
    assert "--selective-offload" in selective.args_captured
    assert "--opd-teacher-defer" in default.args_captured
    assert "--defer-reward-to-post-process" in default.args_captured
    assert "--rm-type" in default.args_captured
    assert default.args_captured[default.args_captured.index("--rm-type") + 1] == "dummy"
    resource = json.loads(default.args_captured[default.args_captured.index("--resource") + 1])
    assert "advantages" not in resource


def test_three_role_split_rejects_overlapping_actor_budget(tmp_path):
    result = _run_script(tmp_path, layout="split", actor_gpus=2)

    assert result.returncode == 2
    assert "requires actor GPUs=6" in result.stderr


def test_three_role_split_can_match_three_way_actor_data_parallelism(tmp_path):
    result = _run_script(
        tmp_path,
        layout="split",
        actor_gpus=3,
        role_gpus=1,
        rollout_batch_size=3,
        n_samples_per_prompt=2,
        global_batch_size=6,
    )

    assert result.returncode == 0, result.stderr
    assert "--rollout-batch-size" in result.args_captured
    assert result.args_captured[result.args_captured.index("--rollout-batch-size") + 1] == "3"
    assert result.args_captured[result.args_captured.index("--n-samples-per-prompt") + 1] == "2"
    assert result.args_captured[result.args_captured.index("--global-batch-size") + 1] == "6"


def test_three_role_acceptance_wait_mode_uses_blocking_ray_submission():
    script = SCRIPT.read_text()

    assert 'if [[ "${INFERENCE_WAIT_FOR_JOB:-0}" == 1 ]]; then' in script
    assert 'RAY_SUBMIT_ARGS+=(--runtime-env-json="$RUNTIME_ENV_JSON")' in script
    assert 'RAY_SUBMIT_ARGS+=(--no-wait --runtime-env-json="$RUNTIME_ENV_JSON")' in script


def test_three_role_acceptance_wait_mode_omits_no_wait_flag(tmp_path):
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    captured = tmp_path / "ray-args.txt"
    fake_ray = fake_bin / "ray"
    fake_ray.write_text('#!/usr/bin/env bash\nprintf \'%s\\n\' "$@" > "$RAY_ARGS_FILE"\n')
    fake_ray.chmod(0o755)

    env = os.environ.copy()
    env.update(
        {
            "MODEL_CONFIG_DIR": str(ROOT / "scripts/models"),
            "MODEL_DIR": str(tmp_path / "model"),
            "INFERENCE_TRAIN_OUTPUT": str(tmp_path / "output"),
            "RAY_DASHBOARD": "http://127.0.0.1:8265",
            "RUNTIME_ENV_JSON": "{}",
            "INFERENCE_LAYOUT": "decoupled",
            "INFERENCE_ACTOR_NODES": "1",
            "INFERENCE_ACTOR_GPUS_PER_NODE": "1",
            "INFERENCE_WAIT_FOR_JOB": "1",
            "RAY_ARGS_FILE": str(captured),
            "PATH": f"{fake_bin}:{env['PATH']}",
        }
    )
    result = subprocess.run(["bash", str(SCRIPT)], cwd=ROOT, env=env, capture_output=True, text=True, check=False)

    assert result.returncode == 0, result.stderr
    assert "--no-wait" not in captured.read_text().splitlines()
