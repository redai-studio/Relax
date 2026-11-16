# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Independent ART recipe; config/preflight never initialize Ray or train."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path
from typing import Any, Mapping


def _required(env: Mapping[str, str], key: str) -> str:
    value = env.get(key, "")
    if not value:
        raise ValueError(f"Set {key} explicitly; ART does not reuse NFT or cluster defaults.")
    return value


def build_argv(env: Mapping[str, str]) -> list[str]:
    """Build one argv shared by preflight and submission, with no side
    effects."""
    model, prompt, artifacts, save, reward = (
        _required(env, key) for key in ("MODEL_PATH", "PROMPT_SET", "ARTIFACT_ROOT", "SAVE_DIR", "REWARD_ENDPOINT")
    )
    for path in (model, prompt, artifacts, save):
        if not Path(path).is_absolute():
            raise ValueError("Model, data and output paths must be absolute worker-visible paths.")
    if not reward.rstrip("/").endswith("/v1"):
        raise ValueError("REWARD_ENDPOINT must end in /v1.")
    groups, candidates, resolution, steps = (
        int(env.get(key, default))
        for key, default in (("N_PROMPTS", "8"), ("N_SAMPLES", "8"), ("RESOLUTION", "1024"), ("STEPS", "20"))
    )
    if groups < 1 or candidates < 2 or candidates % 2 or groups * candidates % 16:
        raise ValueError("Eight-rank ART requires even K>=2 and N_PROMPTS*K divisible by 16.")
    if resolution < 1 or steps < 1:
        raise ValueError("RESOLUTION and STEPS must be positive.")
    overlap = env.get("ART_GROUP_OVERLAP", "0")
    if overlap not in ("0", "1"):
        raise ValueError("ART_GROUP_OVERLAP must be 0 or 1.")
    pending = int(env.get("ART_OVERLAP_PENDING_GROUPS", "2"))
    if not 1 <= pending <= 3:
        raise ValueError("ART_OVERLAP_PENDING_GROUPS must be between 1 and 3.")
    prefetch = env.get("ART_GENERATION_PREFETCH", "0")
    if prefetch not in ("0", "1"):
        raise ValueError("ART_GENERATION_PREFETCH must be 0 or 1.")
    if prefetch == "1" and (overlap != "1" or pending < 2 or candidates != 8):
        raise ValueError("ART_GENERATION_PREFETCH requires group overlap, >=2 pending groups and N_SAMPLES=8.")
    sampling = {
        "algorithm": "art",
        "storage_dtype": "float16",
        "trajectory_storage": "ray",
        "wave_groups": 1,
        "overlap_group_processing": overlap == "1",
        "height": resolution,
        "width": resolution,
        "output_size_from_condition": True,
        "fixed_input_image_pixels": resolution * resolution,
        "max_input_image_side_length": int(env.get("MAX_INPUT_IMAGE_SIDE_LENGTH", "0")),
        "num_inference_steps": steps,
        "guidance_scale": 1.0,
        "image_guidance_scale": 1.0,
        "think_mode": False,
        "shift": float(env.get("SHIFT", "3.0")),
        "t_eps": float(env.get("T_EPS", "0.02")),
        "sde_type": "ode",
        "eta": 0.0,
        "art": {
            "train_step_indices": json.loads(_required(env, "ART_TRAIN_STEP_INDICES")),
            "guidance_gain": float(env.get("ART_GUIDANCE_GAIN", "1.0")),
            "max_cache_bytes": int(_required(env, "ART_MAX_CACHE_BYTES")),
            "max_replay_bytes": int(_required(env, "ART_MAX_REPLAY_BYTES")),
        },
    }
    if prefetch == "1":
        sampling["generation_prefetch"] = True
    if overlap == "1":
        sampling["overlap_pending_groups"] = pending
    argv = [
        "--resource",
        '{"actor":[1,8],"rollout":[1,8]}',
        "--actor-num-nodes",
        "1",
        "--actor-num-gpus-per-node",
        "8",
        "--colocate",
        "--train-backend",
        "fsdp",
        "--generation-task",
        "i2i",
        "--model-path",
        model,
        "--model-adapter-path",
        "relax.models.sensenova.adapter.SenseNovaU1ImageEditAdapter",
        "--rollout-engine-class-path",
        "relax.backends.sglang.diffusion_engine.SGLangNativeGenerationEngine",
        "--rollout-function-path",
        "relax.engine.rollout.native_generation.generate_rollout",
        "--custom-convert-samples-to-train-data-path",
        "relax.engine.rollout.native_generation.convert_samples_to_train_data",
        "--custom-reward-post-process-path",
        "relax.engine.rewards.generative.post_process",
        "--prompt-data",
        prompt,
        "--input-key",
        "prompt",
        "--artifact-root",
        artifacts,
        "--artifact-retention-rollouts",
        "1",
        "--num-rollout",
        env.get("NUM_ROLLOUT", "2"),
        "--rollout-batch-size",
        str(groups),
        "--n-samples-per-prompt",
        str(candidates),
        "--global-batch-size",
        str(groups * candidates),
        "--micro-batch-size",
        "1",
        "--sampling-config",
        json.dumps(sampling),
        "--seed",
        env.get("SEED", "42"),
        "--generation-seed",
        env.get("SEED", "42"),
        "--rollout-seed",
        env.get("ROLLOUT_SEED", "42"),
        "--advantage-estimator",
        "grpo",
        "--generative-kl-coef",
        "0",
        "--generative-reference-policy",
        "fixed",
        "--num-updates-per-batch",
        "2",
        "--lr",
        env.get("LR", "1e-5"),
        "--lr-warmup-iters",
        env.get("LR_WARMUP_ITERS", "0"),
        "--adam-eps",
        env.get("ADAM_EPS", "1e-8"),
        "--weight-decay",
        env.get("WEIGHT_DECAY", "1e-4"),
        "--fsdp-trainable-mode",
        "full",
        "--fsdp-trainable-attr",
        "model",
        "--fsdp-param-dtype",
        "bf16",
        "--fsdp-master-dtype",
        "fp32",
        "--fsdp-reduce-dtype",
        "fp32",
        "--fsdp-activation-checkpointing",
        "--fsdp-load-wave-size",
        "2",
        "--offload-train",
        "--offload-rollout",
        "--reward-runtime",
        "cpu",
        "--reward-scorer-path",
        "relax.engine.rewards.textpecker_reward.TextPeckerRewardScorer",
        "--reward-endpoint",
        reward,
        "--reward-model-path",
        env.get("REWARD_MODEL_NAME", "TextPecker"),
        "--reward-required-components",
        '["textpecker_lpips"]',
        "--reward-component-weights",
        '{"textpecker_lpips":1.0}',
        "--weight-sync-wire-dtype",
        "bf16",
        "--weight-sync-bucket-size-mb",
        env.get("WEIGHT_SYNC_BUCKET_MB", "512"),
        "--rollout-num-gpus-per-engine",
        "1",
        "--sglang-config",
        env.get("SGLANG_CONFIG_PATH", "scripts/training/diffusion/configs/sglang-sensenova-art-8xgpu.yaml"),
        "--save",
        save,
        "--save-interval",
        env.get("SAVE_INTERVAL", "50"),
        "--max-actor-ckpt-to-keep",
        "1",
        "--disable-s3-model-download",
    ]
    if env.get("LOAD_DIR"):
        argv.extend(["--load", env["LOAD_DIR"]])
    return argv


def check_output_paths(args: Any, min_free_bytes: int) -> dict[str, int]:
    """Never reuse an existing run or put outputs inside protected inputs."""
    if min_free_bytes <= 0:
        raise ValueError("Set a positive ART_MIN_FREE_BYTES disk headroom requirement.")
    if any(
        not Path(value).is_absolute() for value in (args.save, args.artifact_root, args.model_path, args.prompt_data)
    ):
        raise ValueError("Parsed input/output paths must remain absolute.")
    outputs = [Path(args.save).resolve(), Path(args.artifact_root).resolve()]
    protected = [
        Path(__file__).resolve().parents[3],
        Path(args.model_path).resolve(),
        Path(args.prompt_data).resolve(),
    ]
    if args.load:
        protected.append(Path(args.load).resolve())

    def overlaps(left: Path, right: Path) -> bool:
        return left.is_relative_to(right) or right.is_relative_to(left)

    if overlaps(*outputs) or any(overlaps(out, source) for out in outputs for source in protected):
        raise ValueError("SAVE_DIR/ARTIFACT_ROOT must be separate new directories, outside code, inputs and LOAD_DIR.")
    free = []
    for output in outputs:
        if output.exists() and (not output.is_dir() or any(output.iterdir())):
            raise ValueError(f"Output directory is not empty; refusing to overwrite or rotate an old run: {output}")
        target = output
        while not target.exists():
            target = target.parent
        free.append(shutil.disk_usage(target).free)
    if min(free) < min_free_bytes:
        raise ValueError("Output filesystem free space is below ART_MIN_FREE_BYTES; no cleanup is performed.")
    return {"minimum_output_free_bytes": min(free), "required_free_bytes": min_free_bytes}


def parse_training_argv(argv: list[str]) -> Any:
    """Use the REAL framework parser; no import of the training entrypoint."""
    from relax.utils.arguments import parse_args

    previous = sys.argv
    try:
        sys.argv = ["sensenova-art-preflight", *argv]
        return parse_args()
    finally:
        sys.argv = previous


def check_resume(args: Any) -> int | None:
    """Require model AND data state from the same completed acquisition."""
    if not args.load:
        return None
    import torch

    from relax.backends.fsdp.checkpoint import find_latest_committed, iter_dir_name

    iteration = find_latest_committed(args.load, "i2i")
    if iteration is None:
        raise ValueError("LOAD_DIR has no committed FSDP I2I checkpoint.")
    root = Path(args.load) / "i2i" / iter_dir_name(iteration)
    state = json.loads((root / "trainer_state.json").read_text())
    contract = json.loads((root / "adapter_contract.json").read_text())
    if state["rollout_id"] != iteration or not contract.get("art") or contract.get("nft"):
        raise ValueError("Resume requires a completed ART acquisition, not NFT or a mismatched cursor.")
    if not (root / "fsdp/.metadata").is_file() or not list((root / "fsdp").glob("*.distcp")):
        raise ValueError("Committed model checkpoint has no DCP metadata/shards.")
    for rank in range(args.actor_num_nodes * args.actor_num_gpus_per_node):
        rng = torch.load(root / f"rng_rank{rank}.pt", map_location="cpu", weights_only=False)
        if not {"torch", "cuda", "python", "numpy"} <= set(rng):
            raise ValueError(f"Incomplete RNG checkpoint for rank {rank}.")
    cursor = Path(args.load) / "dataset" / f"global_dataset_state_dict_{iteration}.pt"
    data = torch.load(cursor, map_location="cpu", weights_only=False)
    required = {"sample_offset", "epoch_id", "sample_group_index", "sample_index", "metadata", "use_streaming"}
    if not required <= set(data) or data["use_streaming"]:
        raise ValueError("ART recipe requires a complete eager-dataset cursor.")
    if data["sample_index"] != data["sample_group_index"] * args.n_samples_per_prompt:
        raise ValueError("Dataset candidate/group identity does not match the ART K contract.")
    if args.num_rollout <= iteration + 1:
        raise ValueError(
            "NUM_ROLLOUT must exceed the restored next rollout; otherwise resume performs no new training."
        )
    if getattr(args, "_start_rollout_id_explicit", False) and args.start_rollout_id != iteration + 1:
        raise ValueError("Explicit start_rollout_id disagrees with the restored complete-acquisition cursor.")
    return iteration


def check_inputs(args: Any) -> dict[str, Any]:
    from PIL import Image

    from relax.engine.rollout.art_cache import art_cache_limits
    from relax.engine.rollout.data_source import RolloutDataSource
    from relax.models.sensenova.adapter import SenseNovaU1ImageEditAdapter

    model = Path(args.model_path)
    if not (model / "config.json").is_file() or not list(model.glob("model*.safetensors")):
        raise ValueError("MODEL_PATH must contain the SenseNova config and safetensors weights.")
    if args.hf_checkpoint or args.use_streaming_dataset or not args.rollout_global_dataset:
        raise ValueError("This recipe requires native eager prompts, with no LLM tokenizer/model loading.")
    iteration = check_resume(args)
    source = RolloutDataSource(args)
    if len(source) < args.rollout_batch_size:
        raise ValueError("Prompt manifest cannot fill a complete acquisition.")
    if iteration is not None:
        source.load(iteration)
    groups = source.get_samples(args.rollout_batch_size)
    adapter, dense_bytes = SenseNovaU1ImageEditAdapter(), 0
    for group in groups:
        sample = group[0]
        metadata = sample.metadata
        images, gt, targets = metadata.get("images"), metadata.get("gt_path"), metadata.get("target_texts")
        if (
            not isinstance(sample.prompt, str)
            or not sample.prompt.strip()
            or not isinstance(images, list)
            or not images
            or any(not isinstance(path, str) or not path for path in images)
            or not isinstance(gt, str)
            or not gt
            or not isinstance(targets, list)
            or not targets
            or any(not isinstance(text, str) or not text.strip() for text in targets)
        ):
            raise ValueError("Each prompt requires ordered conditions, separate GT and explicit target_texts.")
        if Path(gt).resolve() in {Path(path).resolve() for path in images}:
            raise ValueError("GT must not be used as a condition image.")
        for path in [*images, gt]:
            if not Path(path).is_absolute():
                raise ValueError("Image and GT paths must be absolute worker-visible paths.")
            with Image.open(path) as handle:
                handle.verify()
        request = adapter.build_rollout_request(sample, args.sampling_config, args.generation_seed)
        dense_bytes += (
            len(group)
            * 3
            * request["height"]
            * request["width"]
            * (6 * len(args.sampling_config["art"]["train_step_indices"]) + 2)
        )
    cache_limit, _ = art_cache_limits(args.sampling_config)
    if dense_bytes > cache_limit:
        raise ValueError("Actual acquisition geometry exceeds the ART dense cache lower-bound budget.")
    return {
        "manifest_sha256": hashlib.sha256(Path(args.prompt_data).read_bytes()).hexdigest(),
        "next_rollout": 0 if iteration is None else iteration + 1,
        "dense_cache_lower_bound_bytes": dense_bytes,
        "dataset_cursor_gpu_resume_validated": False,
    }


def check_runtime() -> None:
    """Verify installed ART capture and offline reward assets without CUDA."""
    from dataclasses import fields

    import lpips  # noqa: F401
    import torch
    from sglang.multimodal_gen.runtime.models.sensenova_u1.rollout import ART_ROLLOUT_PROTOCOL_VERSION
    from sglang.multimodal_gen.runtime.post_training.rl_dataclasses import RolloutDitTrajectory

    if ART_ROLLOUT_PROTOCOL_VERSION != 5 or "old_velocities" not in {
        field.name for field in fields(RolloutDitTrajectory)
    }:
        raise ValueError("Installed SGLang lacks the ART v5 FP32 old-velocity cache protocol.")
    if not (Path(torch.hub.get_dir()) / "checkpoints/alexnet-owt-7be5be79.pth").is_file():
        raise ValueError("Provision the offline AlexNet weights for real LPIPS before starting ART.")
    if torch.cuda.is_initialized() or torch.distributed.is_initialized():
        raise RuntimeError("Preflight unexpectedly initialized CUDA/distributed; refusing to continue.")


def _http_json(url: str) -> Any:
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(url, timeout=10) as response:
        return json.load(response)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("config", "preflight", "submit"), default="preflight")
    options, extra = parser.parse_known_args()
    argv = build_argv(os.environ) + (extra[1:] if extra[:1] == ["--"] else extra)
    if options.mode == "config":
        print(json.dumps({"training_argv": argv, "submitted": False, "full_parser_checked": False}, indent=2))
        return
    if options.mode == "submit":
        if os.environ.get("ART_EXCLUSIVE_RAY") != "1" or os.environ.get("RELAX_ISOLATED_JOB") != "1":
            raise ValueError(
                "Submit requires ART_EXCLUSIVE_RAY=1 and RELAX_ISOLATED_JOB=1 via scripts/entrypoint/ray-job.sh; "
                "train.py exit shuts down the cluster's Serve apps."
            )
    # Preflight runs with zero visible GPUs. The submitted job receives the
    # caller-supplied runtime_env, never this temporary driver-only setting.
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    args = parse_training_argv(argv)
    from relax.backends.fsdp.arguments import validate_generative_config

    validate_generative_config(args)
    if args.sampling_config.get("algorithm") != "art" or args.load != (os.environ.get("LOAD_DIR") or None):
        raise ValueError("The real parser changed the ART algorithm or LOAD_DIR; refusing an unintended cold start.")
    if args.actor_num_nodes != 1 or args.actor_num_gpus_per_node != 8 or not args.colocate:
        raise ValueError("This independent recipe supports exactly one eight-GPU colocated actor node.")
    report = check_output_paths(args, int(_required(os.environ, "ART_MIN_FREE_BYTES")))
    report.update(check_inputs(args))
    check_runtime()
    import ray

    if ray.is_initialized():
        raise RuntimeError("Preflight unexpectedly initialized Ray.")
    report.update(submitted=False, full_parser_checked=True, worker_environment_checked=False)
    if options.mode == "preflight":
        print(json.dumps(report, indent=2))
        return
    _required(os.environ, "RAY_ADDRESS")
    dashboard = _required(os.environ, "RAY_DASHBOARD_ADDRESS").rstrip("/")
    runtime = _required(os.environ, "RUNTIME_ENV_JSON")
    json.loads(runtime)
    if _http_json(dashboard + "/api/serve/applications/").get("applications"):
        raise ValueError("Ray has existing Serve apps; no cleanup or submission is permitted.")
    models = _http_json(args.reward_endpoint.rstrip("/") + "/models")
    if args.reward_model_path not in {row["id"] for row in models["data"]}:
        raise ValueError("The requested reward served-model name is unavailable.")
    subprocess.run(
        [
            "ray",
            "job",
            "submit",
            "--no-wait",
            f"--address={dashboard}",
            f"--runtime-env-json={runtime}",
            "--working-dir",
            os.environ.get("WORKING_DIR", "."),
            "--",
            "python3",
            "-m",
            "relax.entrypoints.train",
            *argv,
        ],
        check=True,
    )


if __name__ == "__main__":
    main()
