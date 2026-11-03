# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import json
import os
import subprocess
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[3] / "scripts/training/diffusion/run-qwen-image-sft-condition-cache.sh"


def preview(tmp_path, extra=(), **overrides):
    env = {
        **os.environ,
        "DRY_RUN": "1",
        "MODEL_PATH": "/models/with spaces/Qwen-Image",
        "DATA_ROOT": "/data/coco",
        "METADATA_PATH": "/data/caption subset.json",
        "DATASET_SIZE": "24",
        "SAVE_DIR": str(tmp_path / "save"),
        "RELAX_CONDITION_CACHE_RUN_DIR": str(tmp_path / "events"),
        "RAY_DASHBOARD_ADDRESS": "http://ray-head:8265",
        "RUNTIME_ENV_JSON": '{"env_vars":{"EXISTING":"yes","PATH":"/worker/bin"}}',
        "DRIVER_PYTHONPATH": "/imports/with spaces",
        "USE_CLEARML": "0",
        "SAVE_CHECKPOINTS": "0",
        "RELAX_CONDITION_CACHE": "0",
        "RELAX_CONDITION_PREFETCH_DEPTH": "0",
        "IMAGE_PREFETCH_DEPTH": "0",
        "GPUS": "8",
        "GPUS_PER_NODE": "8",
        "GLOBAL_BATCH_SIZE": "8",
        "MICRO_BATCH_SIZE": "1",
        "CACHE_VAE_POSTERIOR": "0",
    }
    env.update(overrides)
    return subprocess.run(["bash", str(SCRIPT), *extra], env=env, capture_output=True, text=True)


@pytest.mark.parametrize("cache", ["0", "1"])
def test_qwen_image_preview_uses_t2i_adapter_and_bounded_cache_without_side_effects(tmp_path, cache):
    result = preview(tmp_path, RELAX_CONDITION_CACHE=cache, RELAX_CONDITION_CACHE_DIR="/shared/cache dir")
    assert result.returncode == 0, result.stderr
    args = json.loads(result.stdout)
    config = json.loads(args[args.index("--t2i-sft-config") + 1])
    assert config == {
        "data_root": "/data/coco",
        "dataset_size": 24,
        "height": 512,
        "width": 512,
        "max_sequence_length": 512,
        "text_encoder_placement": "sharded",
        "cache_vae_posterior": False,
        "gradient_checkpointing": True,
        "deterministic_attention": False,
        "caption_dropout": 0.0,
        "random_crop": False,
        "image_prefetch_depth": 0,
        "image_prefetch_workers": 2,
    }
    assert args[args.index("--generation-task") + 1] == "t2i"
    assert args[args.index("--model-adapter-path") + 1] == "relax.models.qwen_image.sft_adapter.QwenImageSFTAdapter"
    assert args[args.index("--model-path") + 1] == "/models/with spaces/Qwen-Image"
    assert args[args.index("--prompt-data") + 1] == "/data/caption subset.json"
    assert args[args.index("--fsdp-param-dtype") + 1] == "bf16"
    assert args[args.index("--fsdp-master-dtype") + 1] == "fp32"
    assert args[args.index("--fsdp-reduce-dtype") + 1] == "fp32"
    runtime = json.loads(args[args.index("--runtime-env-json") + 1])["env_vars"]
    assert runtime["RELAX_CONDITION_CACHE"] == cache
    assert runtime["RELAX_CONDITION_CACHE_MAX_BYTES"] == "1073741824"
    assert runtime["RELAX_CONDITION_CACHE_DIR"] == "/shared/cache dir"
    assert runtime["RELAX_CONDITION_CACHE_RUN_DIR"] == str(tmp_path / "events")
    assert runtime["EXISTING"] == "yes" and runtime["PATH"] == "/worker/bin"
    assert "--no-wait" in args and "--save" not in args
    assert not (tmp_path / "events").exists() and not (tmp_path / "save").exists()


def test_qwen_image_preview_preserves_effective_batch_with_accumulation_and_exact_verification(tmp_path):
    result = preview(
        tmp_path,
        GLOBAL_BATCH_SIZE="32",
        MICRO_BATCH_SIZE="1",
        RELAX_CONDITION_VERIFY="1",
        RELAX_CONDITION_VERIFY_RTOL="0",
        RELAX_CONDITION_VERIFY_SNAPSHOT="1",
        LR_WARMUP_ITERS="0",
    )
    assert result.returncode == 0, result.stderr
    args = json.loads(result.stdout)
    assert args[args.index("--global-batch-size") + 1] == "32"
    assert args[args.index("--micro-batch-size") + 1] == "1"
    assert args[args.index("--lr-warmup-iters") + 1] == "0"
    env = json.loads(args[args.index("--runtime-env-json") + 1])["env_vars"]
    assert env["RELAX_CONDITION_VERIFY_RTOL"] == "0"
    assert env["RELAX_CONDITION_VERIFY_SNAPSHOT"] == "1"


@pytest.mark.parametrize(
    "overrides",
    [
        {"GLOBAL_BATCH_SIZE": "12"},
        {"MICRO_BATCH_SIZE": "2", "GLOBAL_BATCH_SIZE": "8"},
        {"MICRO_BATCH_SIZE": "2", "GLOBAL_BATCH_SIZE": "32"},
        {"RELAX_CONDITION_CACHE": "yes"},
        {"RELAX_CONDITION_CACHE": "1", "RELAX_CONDITION_CACHE_MAX_BYTES": "0"},
        {"RELAX_CONDITION_VERIFY_RTOL": "nan"},
        {"CAPTION_DROPOUT": "1.1"},
        {"HEIGHT": "511"},
        {"MAX_SEQUENCE_LENGTH": "513"},
        {"TEXT_ENCODER_PLACEMENT": "offload"},
        {"CACHE_VAE_POSTERIOR": "true"},
        {"IMAGE_PREFETCH_DEPTH": "1"},
        {"RELAX_CONDITION_PREFETCH_DEPTH": "1"},
    ],
)
def test_qwen_image_preview_rejects_invalid_configuration_before_submission(tmp_path, overrides):
    result = preview(tmp_path, **overrides)
    assert result.returncode != 0
    assert not (tmp_path / "events").exists()


def test_qwen_image_preview_rejects_hidden_batch_override(tmp_path):
    result = preview(tmp_path, extra=("--global-batch-size=4",))
    assert result.returncode != 0
    assert "GLOBAL_BATCH_SIZE/MICRO_BATCH_SIZE" in result.stderr


def test_qwen_image_preview_declares_replicated_encoder_for_both_cache_modes(tmp_path):
    for enabled in ("0", "1"):
        result = preview(tmp_path, RELAX_CONDITION_CACHE=enabled, TEXT_ENCODER_PLACEMENT="replicated")
        assert result.returncode == 0, result.stderr
        args = json.loads(result.stdout)
        config = json.loads(args[args.index("--t2i-sft-config") + 1])
        assert config["text_encoder_placement"] == "replicated"


@pytest.mark.parametrize("placement", ["sharded", "replicated"])
def test_qwen_image_preview_uses_same_posterior_bundle_policy_for_online_and_cache(tmp_path, placement):
    configs = []
    for enabled in ("0", "1"):
        result = preview(
            tmp_path, RELAX_CONDITION_CACHE=enabled, TEXT_ENCODER_PLACEMENT=placement, CACHE_VAE_POSTERIOR="1"
        )
        assert result.returncode == 0, result.stderr
        args = json.loads(result.stdout)
        configs.append(json.loads(args[args.index("--t2i-sft-config") + 1]))
    assert configs[0] == configs[1]
    assert configs[0]["cache_vae_posterior"] is True
