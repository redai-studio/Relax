#!/usr/bin/env bash
# Copyright (c) 2026 Relax Authors. All Rights Reserved.
# Run through scripts/entrypoint/ray-job.sh on an isolated eight-GPU Ray cluster.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
cd "${SCRIPT_DIR}/../../.."
for arg in "$@"; do
    if [[ "$arg" == --mode || "$arg" == --mode=* ]]; then
        echo "--mode has been removed; this script submits training directly." >&2
        exit 2
    fi
done
: "${MODEL_PATH:?Set MODEL_PATH to the worker-visible SenseNova model directory}"
: "${PROMPT_SET:?Set PROMPT_SET to the prompt JSONL}"
: "${SAVE_DIR:?Set SAVE_DIR to the checkpoint output directory}"
: "${ARTIFACT_ROOT:?Set ARTIFACT_ROOT to the trajectory output directory}"
: "${REWARD_ENDPOINT:?Set REWARD_ENDPOINT to the OpenAI-compatible reward /v1 endpoint}"
: "${ART_TRAIN_STEP_INDICES:?Set ART_TRAIN_STEP_INDICES to the selected ODE input indices}"
: "${ART_MAX_CACHE_BYTES:?Set ART_MAX_CACHE_BYTES to the acquisition tensor budget}"
: "${ART_MAX_REPLAY_BYTES:?Set ART_MAX_REPLAY_BYTES to the replay tensor budget}"
N_PROMPTS="${N_PROMPTS:-8}"
N_SAMPLES="${N_SAMPLES:-8}"
export RAY_NO_WAIT="${RAY_NO_WAIT-}"
if [[ "${ART_GROUP_OVERLAP:-0}" != 0 || "${ART_GENERATION_PREFETCH:-0}" != 0 ]]; then
    echo "Group overlap and generation prefetch require the separate ART performance branch." >&2
    exit 1
fi

SAMPLING_CONFIG=$("${PYTHON:-python3}" - "$N_PROMPTS" "$N_SAMPLES" "${RESOLUTION:-1024}" "${STEPS:-20}" \
    "$ART_TRAIN_STEP_INDICES" "$ART_MAX_CACHE_BYTES" "$ART_MAX_REPLAY_BYTES" \
    "${ART_GUIDANCE_GAIN:-1.0}" "${SHIFT:-3.0}" "${T_EPS:-0.02}" "${MAX_INPUT_IMAGE_SIDE_LENGTH:-0}" <<'PY'
import json
import sys

groups, candidates, resolution, steps = map(int, sys.argv[1:5])
if groups < 1 or candidates < 2 or candidates % 2 or groups * candidates % 16:
    raise ValueError("Eight-rank ART requires even K>=2 and N_PROMPTS*K divisible by 16.")
if resolution < 1 or steps < 1:
    raise ValueError("RESOLUTION and STEPS must be positive.")
sampling = {
    "algorithm": "art", "storage_dtype": "float16", "trajectory_storage": "ray", "wave_groups": 1,
    "height": resolution, "width": resolution, "output_size_from_condition": True,
    "fixed_input_image_pixels": resolution * resolution, "max_input_image_side_length": int(sys.argv[11]),
    "num_inference_steps": steps, "guidance_scale": 1.0, "image_guidance_scale": 1.0, "think_mode": False,
    "shift": float(sys.argv[9]), "t_eps": float(sys.argv[10]), "sde_type": "ode", "eta": 0.0,
    "art": {"train_step_indices": json.loads(sys.argv[5]), "max_cache_bytes": int(sys.argv[6]),
            "max_replay_bytes": int(sys.argv[7]), "guidance_gain": float(sys.argv[8])},
}
print(json.dumps(sampling))
PY
)
TRAIN_ARGS=(
    --resource '{"actor":[1,8],"rollout":[1,8]}'
    --actor-num-nodes 1
    --actor-num-gpus-per-node 8
    --colocate
    --train-backend fsdp
    --generation-task i2i
    --model-path "$MODEL_PATH"
    --model-adapter-path relax.models.sensenova.adapter.SenseNovaU1ImageEditAdapter
    --rollout-engine-class-path relax.backends.sglang.diffusion_engine.SGLangNativeGenerationEngine
    --rollout-function-path relax.engine.rollout.native_generation.generate_rollout
    --custom-convert-samples-to-train-data-path relax.engine.rollout.native_generation.convert_samples_to_train_data
    --custom-reward-post-process-path relax.engine.rewards.generative.post_process
    --prompt-data "$PROMPT_SET"
    --input-key prompt
    --artifact-root "$ARTIFACT_ROOT"
    --artifact-retention-rollouts 1
    --num-rollout "${NUM_ROLLOUT:-2}"
    --rollout-batch-size "$N_PROMPTS"
    --n-samples-per-prompt "$N_SAMPLES"
    --global-batch-size "$((N_PROMPTS * N_SAMPLES))"
    --micro-batch-size 1
    --sampling-config "$SAMPLING_CONFIG"
    --seed "${SEED:-42}"
    --generation-seed "${SEED:-42}"
    --rollout-seed "${ROLLOUT_SEED:-42}"
    --advantage-estimator grpo
    --num-updates-per-batch 2
    --lr "${LR:-1e-5}"
    --lr-warmup-iters "${LR_WARMUP_ITERS:-0}"
    --adam-eps "${ADAM_EPS:-1e-8}"
    --weight-decay "${WEIGHT_DECAY:-1e-4}"
    --fsdp-trainable-mode full
    --fsdp-trainable-attr model
    --fsdp-param-dtype bf16
    --fsdp-master-dtype fp32
    --fsdp-reduce-dtype fp32
    --fsdp-activation-checkpointing
    --fsdp-load-wave-size 2
    --offload-train
    --offload-rollout
    --reward-runtime cpu
    --reward-scorer-path relax.engine.rewards.textpecker_reward.TextPeckerRewardScorer
    --reward-endpoint "$REWARD_ENDPOINT"
    --reward-model-path "${REWARD_MODEL_NAME:-TextPecker}"
    --reward-required-components '["textpecker_lpips"]'
    --reward-component-weights '{"textpecker_lpips":1.0}'
    --weight-sync-wire-dtype bf16
    --weight-sync-bucket-size-mb "${WEIGHT_SYNC_BUCKET_MB:-512}"
    --rollout-num-gpus-per-engine 1
    --sglang-config "${SGLANG_CONFIG_PATH:-scripts/training/diffusion/configs/sglang-sensenova-art-8xgpu.yaml}"
    --save "$SAVE_DIR"
    --save-interval "${SAVE_INTERVAL:-50}"
    --max-actor-ckpt-to-keep 1
    --disable-s3-model-download
)
if [[ -n "${LOAD_DIR:-}" ]]; then TRAIN_ARGS+=(--load "$LOAD_DIR"); fi
SUBMIT_ARGS=(
    --address "${RAY_DASHBOARD_ADDRESS:-http://127.0.0.1:8265}"
    --runtime-env-json "${RUNTIME_ENV_JSON:-"{}"}"
)
if [[ -n "$RAY_NO_WAIT" ]]; then SUBMIT_ARGS+=(--no-wait); fi
if [[ -n "${WORKING_DIR:-}" ]]; then SUBMIT_ARGS+=(--working-dir "$WORKING_DIR"); fi
ray job submit "${SUBMIT_ARGS[@]}" -- \
    python3 -m relax.entrypoints.train \
    "${TRAIN_ARGS[@]}" \
    "$@"
