# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[3]
EXAMPLE = ROOT / "examples/mini_swe_agent"
BLACKWELL_SCRIPT = "run_mini_swe_agent_pd_blackwell.sh"


@pytest.fixture
def submission(tmp_path):
    version = subprocess.check_output(["bash", "-c", 'echo "${BASH_VERSINFO[0]} ${BASH_VERSINFO[1]}"'], text=True)
    if tuple(map(int, version.split())) < (5, 1):
        pytest.skip("The Mini SWE PD launcher requires Bash 5.1+")
    repo = tmp_path / "repo"
    example = repo / "examples/mini_swe_agent"
    example.mkdir(parents=True)
    for name in (
        "run_mini_swe_agent_pd.sh",
        "run_mini_swe_agent_hicache.sh",
        BLACKWELL_SCRIPT,
        "pd-hicache-cpu-1p1d-tp2-blackwell.yaml",
    ):
        shutil.copy2(EXAMPLE / name, example / name)
    models = repo / "scripts/models"
    models.mkdir(parents=True)
    (models / "qwen35-35B-A3B.sh").write_text("MODEL_ARGS=()\n")
    checkpoint = tmp_path / "models/Qwen3.5-35B-A3B"
    checkpoint.mkdir(parents=True)
    (checkpoint / "config.json").write_text("{}")
    binary = tmp_path / "bin"
    binary.mkdir()
    (binary / "ray").write_text(
        f"#!{sys.executable}\nimport json, os, sys\nfrom pathlib import Path\n"
        'Path(os.environ["CAPTURE"]).write_text(json.dumps(sys.argv[1:]))\n'
    )
    (binary / "ray").chmod(0o755)
    os.symlink(sys.executable, binary / "python")
    # Package discovery must work without importing Torch or initializing CUDA.
    packages = tmp_path / "runtime-packages"
    for module in ("torch", "nvidia/nvshmem"):
        (packages / module / "lib").mkdir(parents=True)
    (packages / "torch/__init__.py").write_text(
        'raise AssertionError("Torch must not be imported during submission")\n'
    )
    env = os.environ.copy()
    for key in (
        "HICACHE_DEPLOYMENT",
        "RAY_NO_WAIT",
        "GLOBAL_BATCH_SIZE",
        "OVER_SAMPLING_BATCH_SIZE",
        "NUM_ITERS_PER_TRAIN_UPDATE",
        "NUM_ROLLOUT",
        "MAX_CONTEXT_LEN",
        "MAX_RESPONSE_LEN",
        "MAX_STALENESS",
        "MAX_TOKENS_PER_GPU",
        "LOG_PROBS_MAX_TOKENS_PER_GPU",
        "ENABLE_CLEARML",
        "AGENT_SERVER_MAX_TURNS",
        "SGLANG_MAX_TOTAL_TOKENS",
        "SGLANG_MEM_FRACTION_STATIC",
        "SGLANG_RUST_BUILD_MODE",
        "MC_GID_INDEX",
    ):
        env.pop(key, None)
    env.update(
        PATH=f"{binary}:{os.environ['PATH']}",
        PYTHONPATH=str(packages),
        CAPTURE=str(tmp_path / "captured.json"),
        RELAX_ENTRYPOINT_MODE="ray-job",
        RUNTIME_ENV_JSON=json.dumps(
            {
                "worker_process_setup_hook": "existing.hook",
                "env_vars": {
                    "PYTHONPATH": "isolated-sglang",
                    "LD_LIBRARY_PATH": f"/custom::{packages}/nvidia/nvshmem/lib:/custom:",
                    "KEEP": "yes",
                },
            }
        ),
        MINI_SWE_AGENT_VENV=str(tmp_path),
        MODEL_DIR=str(tmp_path / "models"),
        R2E_DATA_PATH=str(tmp_path / "data"),
        R2E_SIF_DIR=str(tmp_path / "sif"),
        SAVE_DIR=str(tmp_path / "save"),
        EXP_ROOT=str(tmp_path / "exp"),
        RUN_ID="test-hicache",
        ROLLOUT_BATCH_SIZE="1",
        AGENT_SERVER_TRAIN_CONCURRENCY="8",
        ENABLE_EVAL="0",
    )
    return example, env


@pytest.fixture
def blackwell_submission(submission):
    example, env = submission
    # Exercise the recipe defaults, without the common PD fixture's tiny batch
    # or a checkpoint directory masking an accidental SAVE_DIR requirement.
    for key in ("ROLLOUT_BATCH_SIZE", "AGENT_SERVER_TRAIN_CONCURRENCY", "SAVE_DIR"):
        env.pop(key)
    return example, env


def _submit(example, env, script):
    result = subprocess.run(["bash", str(example / script)], env=env, capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stderr
    return json.loads(Path(env["CAPTURE"]).read_text())


def _last_value(args, flag):
    return [args[index + 1] for index, item in enumerate(args) if item == flag][-1]


@pytest.mark.parametrize("deployment", [None, "regular", "pd"])
def test_hicache_submission_selects_topology_and_preserves_runtime(submission, deployment):
    example, env = submission
    if deployment is not None:
        env["HICACHE_DEPLOYMENT"] = deployment
    args = _submit(example, env, "run_mini_swe_agent_hicache.sh")
    config = "pd-hicache-cpu-2p2d-tp2.yaml" if deployment == "pd" else "hicache-cpu-4x-tp2.yaml"
    assert _last_value(args, "--sglang-config") == str(example / config)
    runtime = json.loads(_last_value(args, "--runtime-env-json"))
    assert runtime["env_vars"]["PYTHONPATH"].endswith("isolated-sglang")
    assert args[args.index("--") + 1 : args.index("--") + 4] == [
        "bash",
        str(example / "run_mini_swe_agent_pd.sh"),
        "--ray-driver",
    ]


def test_hicache_submission_rejects_unknown_deployment(submission):
    example, env = submission
    result = subprocess.run(
        ["bash", str(example / "run_mini_swe_agent_hicache.sh")],
        env={**env, "HICACHE_DEPLOYMENT": "unknown"},
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode != 0 and "regular or pd" in result.stderr
    assert not Path(env["CAPTURE"]).exists()


def test_common_pd_enables_capacity_without_changing_sync_batch_or_checkpoint(submission):
    example, env = submission
    env.update(ROLLOUT_BATCH_SIZE="2", AGENT_SERVER_TRAIN_CONCURRENCY="24")
    args = _submit(example, env, "run_mini_swe_agent_pd.sh")
    assert json.loads(_last_value(args, "--resource")) == {"actor": [1, 8], "rollout": [1, 8]}
    assert "--colocate" in args and "--fully-async" not in args
    assert _last_value(args, "--max-staleness") == "0"
    assert _last_value(args, "--rollout-batch-size") == _last_value(args, "--agentic-concurrency") == "2"
    assert _last_value(args, "--global-batch-size") == "16"
    assert _last_value(args, "--agentic-session-capacity-per-node") == "24"
    assert _last_value(args, "--agent-resource-owner-path") == "relax.agentic.runner.resources.ReceiptResourceOwner"
    assert "--over-sampling-batch-size" not in args
    assert _last_value(args, "--save") == str(
        Path(env["SAVE_DIR"]) / "Qwen3.5-35B-A3B-R2E-Gym-MiniSWEAgent-PD-Checkpoint"
    )
    assert _last_value(args, "--save-interval") == "20"


def test_blackwell_defaults_use_four_gpu_async_roles_and_session_capacity(blackwell_submission):
    example, env = blackwell_submission
    args = _submit(example, env, BLACKWELL_SCRIPT)
    assert json.loads(_last_value(args, "--resource")) == {"actor": [1, 4], "rollout": [1, 4], "advantages": [1, 0]}
    assert "--fully-async" in args and "--colocate" not in args
    assert "--no-wait" not in args
    assert "--use-clearml" in args
    expected = {
        "--global-batch-size": "256",
        "--rollout-batch-size": "32",
        "--over-sampling-batch-size": "48",
        "--n-samples-per-prompt": "8",
        "--agentic-concurrency": "48",
        "--agentic-session-capacity-per-node": "128",
        "--num-rollout": "20",
        "--num-steps-per-rollout": "1",
        "--num-iters-per-train-update": "8",
        "--tensor-model-parallel-size": "4",
        "--expert-model-parallel-size": "4",
        "--context-parallel-size": "1",
        "--seq-length": "131072",
        "--rollout-max-context-len": "131072",
        "--sglang-context-length": "131072",
        "--max-tokens-per-gpu": "131072",
        "--log-probs-max-tokens-per-gpu": "131072",
        "--recompute-granularity": "full",
        "--recompute-method": "block",
        "--recompute-num-layers": "32",
        "--optimizer-offload-fraction": "1.0",
        "--sglang-mem-fraction-static": "0.90",
        "--sglang-disaggregation-transfer-backend": "mooncake",
        "--sglang-disaggregation-ib-device": "mlx5_gdr_4,mlx5_gdr_5,mlx5_gdr_6,mlx5_gdr_7",
        "--sglang-router-prefill-policy": "consistent_hashing",
        "--sglang-router-decode-policy": "consistent_hashing",
        "--agent-resource-owner-path": "relax.agentic.runner.resources.ReceiptResourceOwner",
    }
    for flag, value in expected.items():
        assert _last_value(args, flag) == value, flag
    assert "--sequence-parallel" in args
    assert "--optimizer-cpu-offload" in args
    assert "--sglang-router-disable-circuit-breaker" in args
    assert "--sglang-router-disable-health-check" not in args
    assert all(flag not in args for flag in ("--save", "--save-interval", "--save-hf"))
    assert all(
        flag not in args
        for flag in (
            "--overlap-cpu-optimizer-d2h-h2d",
            "--rollout-result-dir",
            "--dump-details",
            "--sglang-max-total-tokens",
            "--sglang-max-mamba-cache-size",
        )
    )
    train_env = json.loads(_last_value(args, "--train-env-vars"))
    assert train_env["CUDA_DEVICE_MAX_CONNECTIONS"] == "1"
    assert "NVTE_USE_CUTLASS_GROUPED_GEMM" not in train_env
    assert train_env["PYTORCH_ALLOC_CONF"] == "expandable_segments:True"
    assert args[args.index("--") + 1 : args.index("--") + 4] == [
        "bash",
        str(example / "run_mini_swe_agent_pd.sh"),
        "--ray-driver",
    ]
    runtime_config = json.loads(_last_value(args, "--runtime-env-json"))
    assert runtime_config["worker_process_setup_hook"] == "existing.hook"
    runtime = runtime_config["env_vars"]
    assert runtime["KEEP"] == "yes"
    assert runtime["LD_LIBRARY_PATH"].split(":") == [
        str(Path(env["PYTHONPATH"]) / "nvidia/nvshmem/lib"),
        str(Path(env["PYTHONPATH"]) / "torch/lib"),
        "/custom",
    ]
    assert runtime["AGENT_SERVER_TRAIN_CONCURRENCY"] == "128"
    assert runtime["AGENT_SERVER_MAX_TURNS"] == "100"
    assert runtime["AGENT_SERVER_RESUMABLE_GENERATION"] == "1"
    assert runtime["SGLANG_RUST_BUILD_MODE"] == "never"
    assert runtime["MC_GID_INDEX"] == "3"
    assert runtime["LITELLM_LOCAL_MODEL_COST_MAP"] == "True"
    assert runtime["PYTHONPATH"].endswith("isolated-sglang")
    config = example / "pd-hicache-cpu-1p1d-tp2-blackwell.yaml"
    assert _last_value(args, "--sglang-config") == str(config)
    model = yaml.safe_load(config.read_text())["sglang"][0]
    assert model["num_gpus_per_engine"] == 2
    groups = model["engine_groups"]
    assert [group["worker_type"] for group in groups] == ["prefill", "decode"]
    assert [group["num_gpus"] for group in groups] == [2, 2]
    for group in groups:
        overrides = group["overrides"]
        assert overrides["enable_hierarchical_cache"] == (group["worker_type"] == "prefill")
        assert "max_total_tokens" not in overrides
        assert "max_mamba_cache_size" not in overrides
    assert groups[1]["overrides"]["disaggregation_decode_enable_radix_cache"] is False


def test_blackwell_overrides_allow_incremental_capacity_not_divisible_by_group(blackwell_submission):
    example, env = blackwell_submission
    env.update(
        RAY_NO_WAIT="1",
        GLOBAL_BATCH_SIZE="128",
        AGENT_SERVER_TRAIN_CONCURRENCY="63",
        MAX_TOKENS_PER_GPU="262144",
        SGLANG_MEM_FRACTION_STATIC="0.92",
    )
    args = _submit(example, env, BLACKWELL_SCRIPT)
    assert "--no-wait" in args
    assert _last_value(args, "--global-batch-size") == "128"
    assert _last_value(args, "--rollout-batch-size") == "16"
    assert _last_value(args, "--over-sampling-batch-size") == _last_value(args, "--agentic-concurrency") == "24"
    assert _last_value(args, "--agentic-session-capacity-per-node") == "63"
    assert _last_value(args, "--max-tokens-per-gpu") == "262144"
    assert _last_value(args, "--log-probs-max-tokens-per-gpu") == "262144"
    assert _last_value(args, "--sglang-mem-fraction-static") == "0.92"
    runtime = json.loads(_last_value(args, "--runtime-env-json"))["env_vars"]
    assert runtime["AGENT_SERVER_TRAIN_CONCURRENCY"] == "63"


def test_blackwell_missing_runtime_library_fails_before_submission(blackwell_submission):
    example, env = blackwell_submission
    (Path(env["PYTHONPATH"]) / "torch/lib").rmdir()
    result = subprocess.run(
        ["bash", str(example / BLACKWELL_SCRIPT)], env=env, capture_output=True, text=True, timeout=15
    )
    assert result.returncode != 0 and "Missing torch runtime libraries" in result.stderr
    assert not Path(env["CAPTURE"]).exists()


@pytest.mark.parametrize(
    "override,error",
    [
        ({"GLOBAL_BATCH_SIZE": "130"}, "complete eight-member Groups"),
        ({"AGENT_SERVER_TRAIN_CONCURRENCY": "7"}, "fit one complete eight-member Group"),
        ({"OVER_SAMPLING_BATCH_SIZE": "31"}, "at least ROLLOUT_BATCH_SIZE"),
        ({"MAX_TOKENS_PER_GPU": "16384"}, "token budgets must fit"),
        ({"LOG_PROBS_MAX_TOKENS_PER_GPU": "16384"}, "token budgets must fit"),
        ({"MAX_TOKENS_PER_GPU": "131073"}, "Token budgets must be divisible"),
    ],
)
def test_blackwell_rejects_invalid_batches_capacity_and_token_budgets(blackwell_submission, override, error):
    example, env = blackwell_submission
    result = subprocess.run(
        ["bash", str(example / BLACKWELL_SCRIPT)],
        env={**env, **override},
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode != 0 and error in result.stderr
    assert not Path(env["CAPTURE"]).exists()


@pytest.mark.parametrize(
    ("filename", "roles", "gpus", "engine_gpus", "attention_dp"),
    [
        ("hicache-cpu-4x-tp2.yaml", ["regular"], [8], 2, 1),
        ("pd-hicache-cpu-2p2d-tp2.yaml", ["prefill", "decode"], [4, 4], 2, 1),
        ("pd-hicache-cpu-1p1d-dp2-tp2-ep4.yaml", ["prefill", "decode"], [4, 4], 4, 2),
    ],
)
def test_cpu_cache_examples_support_both_topologies(filename, roles, gpus, engine_gpus, attention_dp):
    model = yaml.safe_load((EXAMPLE / filename).read_text())["sglang"][0]
    assert model["num_gpus_per_engine"] == engine_gpus
    groups = model["engine_groups"]
    assert [group["worker_type"] for group in groups] == roles
    assert [group["num_gpus"] for group in groups] == gpus
    for group in groups:
        overrides = group["overrides"]
        assert overrides["hicache_storage_backend"] is None
        assert overrides["enable_hierarchical_cache"] == (group["worker_type"] != "decode")
        if attention_dp > 1:
            assert overrides["dp_size"] == attention_dp
            assert overrides["enable_dp_attention"] is True
            assert overrides["ep_size"] == 4
            assert engine_gpus // attention_dp == 2
        if group["worker_type"] != "decode":
            assert overrides["mamba_radix_cache_strategy"] == "extra_buffer"
            assert overrides["mamba_track_interval"] % overrides["page_size"] == 0
            assert overrides["hicache_write_policy"] == "write_through"
            assert overrides["chunked_prefill_size"] // attention_dp == 8192
            assert overrides["max_mamba_cache_size"] // attention_dp == 666
