# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Check recipes and standard entrypoints without starting Ray or touching
GPUs."""

from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from examples.chunkflow.arguments import add_arguments, validate_args


ROOT = Path(__file__).resolve().parents[4]


@pytest.mark.parametrize(
    "options,error",
    [
        ({"calculate_per_token_loss": False}, "--calculate-per-token-loss"),
        ({"fully_async": True, "use_dynamic_batch_size": True}, "streaming iterators"),
        ({"fully_async": False, "use_dynamic_batch_size": True}, None),
        ({"fully_async": True, "use_dynamic_batch_size": False}, None),
        ({"use_chunkflow": False, "calculate_per_token_loss": False}, None),
        ({"use_chunkflow": False, "fully_async": True, "use_dynamic_batch_size": True}, None),
    ],
)
def test_validation_rejects_unsupported_loss_and_streaming_contracts(options, error):
    args = SimpleNamespace(
        **{"use_chunkflow": True, "chunkflow_chunk_size": 128, "calculate_per_token_loss": True, **options}
    )
    if error is None:
        validate_args(args)
    else:
        with pytest.raises(ValueError, match=error):
            validate_args(args)


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
    env.pop("PROMPT_DATA", None)
    command = ["bash", str(ROOT / "examples/chunkflow/scripts" / script)]
    outputs = [subprocess.check_output(command, env=env, text=True) for _ in range(2)]
    prompt_data = str(tmp_path / "data" / "LongAlign-10k" / "long.jsonl")
    outputs.append(subprocess.check_output(command, env={**env, "PROMPT_DATA": prompt_data}, text=True))
    for index, output in enumerate(outputs):
        argv = shlex.split(output)

        def value(option):
            # argparse uses the last value when model defaults are overridden.
            return argv[len(argv) - 1 - argv[::-1].index(option) + 1]

        assert argv[:3] == ["python3", "-m", "relax.entrypoints.train"]
        assert value("--hf-checkpoint") == str(tmp_path / "models with spaces" / model)
        assert value("--load") == value("--save") == str(tmp_path / "checkpoints/resume-test")
        assert value("--prompt-data") == (prompt_data if index == 2 else str(tmp_path / "data" / "train.jsonl"))
        assert value("--context-parallel-size") == value("--pipeline-model-parallel-size") == "1"
        assert value("--expert-model-parallel-size") == experts
        expected_gpus = 32 if model == "Qwen3.8-Flash-Next" else 8
        assert json.loads(value("--resource"))["actor"] == [1, expected_gpus]
        assert "--recompute-granularity" not in argv
        parser = ArgumentParser()
        parser.add_argument("--calculate-per-token-loss", action="store_true")
        args, _ = add_arguments(parser).parse_known_args(argv[3:])
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
        ["bash", "-c", 'set -eo pipefail; source "$1"; submit_training local --use-chunkflow', "test", str(common)],
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


@pytest.fixture
def launch_recipe(tmp_path):
    """Run real shell entrypoints with isolated, recording process/GPU/Ray
    commands.

    This covers accidental local cleanup, duplicate multi-node submission and
    wrong Jobs addresses in seconds on CPU, without a Ray or GPU dependency.
    """
    for directory in ("examples/chunkflow/scripts", "scripts/models", "scripts/entrypoint"):
        shutil.copytree(ROOT / directory, tmp_path / directory, ignore=shutil.ignore_patterns("log"))
    setup = tmp_path / "examples/chunkflow/setup.py"
    setup.write_text(
        "import os, sys\nassert sys.argv[-1] == '--check'\nsys.exit(int(os.getenv('TEST_SETUP_FAIL', '0')))\n"
    )
    binary = tmp_path / "bin"
    binary.mkdir()
    for name in ("python", "python3"):
        (binary / name).symlink_to(sys.executable)
    command = binary / "record-command"
    command.write_text(
        f"#!{sys.executable}\n"
        """import json, os, sys
from pathlib import Path

name, args = Path(sys.argv[0]).name, sys.argv[1:]
with open(os.environ['TEST_COMMANDS'], 'a') as stream:
    stream.write(json.dumps([name, *args]) + '\\n')
if name == 'ray':
    if args[0] == 'status':
        print('0.0/32.0 GPU')
        sys.exit(int(os.getenv('TEST_RAY_UNREACHABLE', '0')))
    if args[:2] == ['list', 'nodes']:
        print('[{"is_head_node": true, "node_ip": "192.0.2.1"}]')
    if args[:2] == ['job', 'submit']:
        sys.exit(int(os.getenv('TEST_SUBMIT_FAIL', '0')))
elif name == 'nvidia-smi':
    print('NV4' if args == ['topo', '-m'] else 'GPU 0')
elif name == 'jq':
    json.load(sys.stdin)
    print('192.0.2.1')
elif name == 'timeout':
    # The SPMD worker's TCP readiness probe must never open a real socket.
    if args[1] != 'bash':
        os.execvp(args[1], args[1:])
elif name == 'grep':
    if args[0] == '-oP':
        # BSD grep lacks -P; emulate the entrypoint's GPU resource extraction.
        import re
        for match in re.findall(args[1], sys.stdin.read()):
            print(match)
    else:
        os.execv('/usr/bin/grep', ['grep', *args])
"""
    )
    command.chmod(0o755)
    for name in ("ray", "pkill", "sleep", "nvidia-smi", "sysctl", "timeout", "jq", "grep"):
        (binary / name).symlink_to(command)
    env = {
        **os.environ,
        "PATH": f"{binary}:{os.environ['PATH']}",
        "RELAX": str(tmp_path),
        "DRY_RUN": "0",
        "RELAX_KERNEL_CACHE_DIR": "",
        "RELAX_SKIP_CLUSTER_CLEANUP": "1",
        "RAY_NO_WAIT": "",
        "MODEL_DIR": str(tmp_path / "models with spaces"),
        "DATA_DIR": str(tmp_path / "data"),
        "SAVE_DIR": str(tmp_path / "checkpoints"),
        "TEST_COMMANDS": str(tmp_path / "commands.jsonl"),
    }
    for key in (
        "RELAX_ENTRYPOINT_MODE",
        "RAY_ADDRESS",
        "RAY_JOBS_ADDRESS",
        "MASTER_ADDR",
        "POD_NAME",
        "HOST_IP",
        "WORLD_SIZE",
        "NUM_GPUS",
        "CUDA_VISIBLE_DEVICES",
        "RUNTIME_ENV_JSON",
        "WORKING_DIR",
        "_RAY_JOB_RUN_SCRIPT",
        "_RELAX_KERNEL_CACHE_HELPER_SOURCED",
    ):
        env.pop(key, None)

    def run(script, **overrides):
        result = subprocess.run(
            ["bash", str(tmp_path / "examples/chunkflow/scripts" / script)],
            env={**env, **overrides},
            capture_output=True,
            text=True,
            timeout=15,
        )
        log = Path(env["TEST_COMMANDS"])
        calls = [json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []
        submissions = [call[1:] for call in calls if call[:3] == ["ray", "job", "submit"]]
        return result, calls, submissions

    return run


@pytest.mark.parametrize("script", ["run-qwen3-4B-sft-8xgpu.sh", "run-qwen36-35B-A3B-sft-8xgpu.sh"])
def test_launch_single_node_starts_ray_and_submits_locally(launch_recipe, script):
    result, calls, submissions = launch_recipe(script)
    assert result.returncode == 0, result.stderr
    starts = [call for call in calls if call[:2] == ["ray", "start"]]
    assert len(starts) == len(submissions) == 1
    assert "--head" in starts[0] and starts[0][starts[0].index("--num-gpus") + 1] == "8"
    assert "--address=http://127.0.0.1:8265" in submissions[0]
    assert "--no-wait" not in submissions[0]
    assert json.loads(submissions[0][submissions[0].index("--resource") + 1])["actor"] == [1, 8]


@pytest.mark.parametrize("script", ["run-qwen3-4B-sft-8xgpu.sh", "run-qwen38-flash-next-sft-32xgpu.sh"])
def test_launch_existing_cluster_never_restarts_ray(launch_recipe, script):
    result, calls, submissions = launch_recipe(
        script,
        RAY_ADDRESS="192.0.2.1:6379",
        RAY_JOBS_ADDRESS="http://192.0.2.1:8265",
    )
    assert result.returncode == 0, result.stderr
    assert len(submissions) == 1
    assert "--address=http://192.0.2.1:8265" in submissions[0]
    assert not any(call[0] == "pkill" or call[:2] in (["ray", "start"], ["ray", "stop"]) for call in calls)


@pytest.mark.parametrize("worker", [False, True])
def test_launch_multinode_only_head_submits(launch_recipe, worker):
    result, calls, submissions = launch_recipe(
        "run-qwen38-flash-next-sft-32xgpu.sh",
        MASTER_ADDR="test-head",
        POD_NAME="test-worker" if worker else "test-head",
        HOST_IP="192.0.2.2" if worker else "192.0.2.1",
    )
    assert result.returncode == 0, result.stderr
    starts = [call for call in calls if call[:2] == ["ray", "start"]]
    assert len(starts) == 1
    assert starts[0][starts[0].index("--num-gpus") + 1] == "8"
    if worker:
        assert "--address=test-head:6379" in starts[0]
        assert ["sleep", "inf"] in calls and not submissions
    else:
        assert "--head" in starts[0] and len(submissions) == 1
        assert "--address=http://127.0.0.1:8265" in submissions[0]
        assert json.loads(submissions[0][submissions[0].index("--resource") + 1])["actor"] == [1, 32]
        runtime = json.loads(
            next(arg.split("=", 1)[1] for arg in submissions[0] if arg.startswith("--runtime-env-json="))
        )
        assert runtime["env_vars"]["MASTER_ADDR"] == "192.0.2.1"
        assert runtime["env_vars"]["MCORE_QSA_SPARSE_BACKEND"] == "flex"


@pytest.mark.parametrize(
    "overrides,error",
    [
        ({}, "MASTER_ADDR"),
        ({"MASTER_ADDR": "head", "POD_NAME": "head", "HOST_IP": "192.0.2.1", "WORLD_SIZE": "2"}, "= 32"),
        ({"MASTER_ADDR": "head", "POD_NAME": "head", "HOST_IP": "192.0.2.1", "NUM_GPUS": "zero"}, "= 32"),
        ({"RAY_ADDRESS": "192.0.2.1:6379"}, "RAY_JOBS_ADDRESS"),
        (
            {
                "RAY_ADDRESS": "192.0.2.1:6379",
                "RAY_JOBS_ADDRESS": "http://192.0.2.1:8265",
                "TEST_RAY_UNREACHABLE": "1",
            },
            "Cannot reach",
        ),
    ],
)
def test_launch_invalid_cluster_fails_before_cleanup(launch_recipe, overrides, error):
    result, calls, submissions = launch_recipe("run-qwen38-flash-next-sft-32xgpu.sh", **overrides)
    assert result.returncode != 0 and error in result.stderr
    assert not submissions
    assert not any(call[0] == "pkill" or call[:2] in (["ray", "start"], ["ray", "stop"]) for call in calls)


def test_launch_missing_patch_fails_before_starting_ray(launch_recipe):
    result, calls, submissions = launch_recipe("run-qwen3-4B-sft-8xgpu.sh", TEST_SETUP_FAIL="1")
    assert result.returncode != 0 and not calls and not submissions


def test_launch_submission_failure_is_not_hidden_by_tee(launch_recipe):
    result, _, submissions = launch_recipe("run-qwen3-4B-sft-8xgpu.sh", TEST_SUBMIT_FAIL="7")
    assert result.returncode == 7 and len(submissions) == 1
