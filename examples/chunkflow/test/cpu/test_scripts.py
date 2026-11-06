# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Preview the real recipes without sourcing a cluster entrypoint."""

from __future__ import annotations

import json
import os
import shlex
import subprocess
from pathlib import Path

import pytest

from examples.chunkflow.arguments import add_arguments, validate_args


ROOT = Path(__file__).resolve().parents[4]


@pytest.mark.parametrize(
    "script,model,experts",
    [
        ("run-qwen3-4B-sft-8xgpu.sh", "Qwen3-4B", "1"),
        ("run-qwen36-35B-A3B-sft-8xgpu.sh", "Qwen3.6-35B-A3B", "8"),
        ("run-qwen38-flash-next-sft-32xgpu.sh", "Qwen3.8-Flash-Next", "8"),
    ],
)
def test_recipe_preview_has_recoverable_paths_and_supported_configuration(tmp_path, script, model, experts):
    from argparse import ArgumentParser

    env = {
        **os.environ,
        "DRY_RUN": "1",
        "MODEL_DIR": str(tmp_path / "models with spaces"),
        "DATA_DIR": str(tmp_path / "data"),
        "SAVE_DIR": str(tmp_path / "checkpoints"),
        "EXP_NAME": "resume-test",
        "CHUNK_SIZE": "1024",
        "KEEP_LAST_K": "2",
    }
    command = ["bash", str(ROOT / "examples/chunkflow/scripts" / script)]
    outputs = [subprocess.check_output(command, env=env, text=True) for _ in range(2)]
    for output in outputs:
        argv = shlex.split(output)

        def value(option):
            # argparse uses the last value when model defaults are overridden.
            return argv[len(argv) - 1 - argv[::-1].index(option) + 1]

        assert argv[:3] == ["python3", "-m", "relax.entrypoints.train"]
        assert value("--hf-checkpoint") == str(tmp_path / "models with spaces" / model)
        assert value("--load") == value("--save") == str(tmp_path / "checkpoints/resume-test")
        assert value("--context-parallel-size") == value("--pipeline-model-parallel-size") == "1"
        assert value("--expert-model-parallel-size") == experts
        expected_gpus = 32 if model == "Qwen3.8-Flash-Next" else 8
        assert json.loads(value("--resource"))["actor"] == [1, expected_gpus]
        assert "--recompute-granularity" not in argv
        args, _ = add_arguments(ArgumentParser()).parse_known_args(argv[3:])
        validate_args(args)
        assert args.use_chunkflow and args.chunkflow_chunk_size == 1024 and args.chunkflow_keep_last_k == 2
        if experts != "1":
            assert (ROOT / value("--custom-config-path")).is_file()
            assert value("--moe-router-load-balancing-type") == "none"


def test_submission_forwards_model_settings_to_workers(tmp_path):
    """A local shell export must survive Ray's independent worker
    environment."""
    import json
    import sys

    # Exercise the submission helper with a recording Ray CLI and an isolated
    # installer check; no cluster entrypoint or network connection is invoked.
    setup = tmp_path / "examples/chunkflow/setup.py"
    setup.parent.mkdir(parents=True)
    setup.write_text("import sys\nassert sys.argv[-1] == '--check'\n")
    binary = tmp_path / "bin"
    binary.mkdir()
    (binary / "python3").symlink_to(sys.executable)
    ray = binary / "ray"
    ray.write_text(
        f"#!{sys.executable}\nimport json, sys\nfrom pathlib import Path\n"
        f"Path({str(tmp_path / 'submission.json')!r}).write_text(json.dumps(sys.argv[1:]))\n"
    )
    ray.chmod(0o755)
    env = {
        **os.environ,
        "PATH": f"{binary}:{os.environ['PATH']}",
        "RELAX_ROOT": str(tmp_path),
        "SCRIPT_DIR": str(tmp_path),
        "RELAX_ENTRYPOINT_MODE": "test",
        "DRY_RUN": "0",
        "RAY_NO_WAIT": "",
        "RAY_JOBS_ADDRESS": "http://unused.invalid",
        "EXP_NAME": "test",
        "now": "test",
        "RUNTIME_ENV_JSON": json.dumps({"env_vars": {"EXISTING": "preserved"}}),
        "MCORE_QSA_SPARSE_BACKEND": "flex",
        "RELAX_QWEN38_PLE_STREAMED": "0",
        "RELAX_CHUNKFLOW_KV_GRAD_FP32": "1",
        "QWEN38_PLE_CPU_OFFLOAD": "1",
    }
    common = ROOT / "examples/chunkflow/scripts/_common.sh"
    subprocess.run(
        ["bash", "-c", 'set -eo pipefail; source "$1"; submit_training --use-chunkflow', "test", str(common)],
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    argv = json.loads((tmp_path / "submission.json").read_text())
    assert "--no-wait" not in argv
    runtime = json.loads(next(arg.split("=", 1)[1] for arg in argv if arg.startswith("--runtime-env-json=")))
    assert runtime["env_vars"]["EXISTING"] == "preserved"
    for key in (
        "MCORE_QSA_SPARSE_BACKEND",
        "RELAX_QWEN38_PLE_STREAMED",
        "RELAX_CHUNKFLOW_KV_GRAD_FP32",
        "QWEN38_PLE_CPU_OFFLOAD",
    ):
        assert runtime["env_vars"][key] == env[key]
