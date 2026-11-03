#!/usr/bin/env bash
# Copyright (c) 2026 Relax Authors. All Rights Reserved.
# Compare online and cached Qwen-Image LoRA SFT with the same effective batch.
# Invoke through scripts/entrypoint/ray-job.sh on an existing Ray cluster.
# MODEL_PATH=/models/Qwen-Image DATA_ROOT=/data/COCO2017 METADATA_PATH=/data/train.json \
# DATASET_SIZE=128 SAVE_DIR=/runs RELAX_CONDITION_CACHE_RUN_DIR=/runs/online-r01 \
# RELAX_SKIP_CLUSTER_CLEANUP=1 RELAX_CONDITION_CACHE=0 bash scripts/entrypoint/ray-job.sh \
# scripts/training/diffusion/run-qwen-image-sft-condition-cache.sh
# Repeat with a fresh run directory and RELAX_CONDITION_CACHE=1. DRY_RUN=1 prints
# the submission without sourcing entrypoints, creating directories or using Ray.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
RELAX_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
export RELAX_SKIP_CLUSTER_CLEANUP="${RELAX_SKIP_CLUSTER_CLEANUP:-1}"
if [[ "${DRY_RUN:-0}" != 1 && -z "${RELAX_ENTRYPOINT_MODE:-}" ]]; then
    # shellcheck source=../../entrypoint/ray-job.sh
    source "${RELAX_ROOT}/scripts/entrypoint/ray-job.sh"
fi

MODEL_PATH="${MODEL_PATH:-${MODEL_DIR:-}}"
DATA_ROOT="${DATA_ROOT:-${DATA_DIR:-}}"
METADATA_PATH="${METADATA_PATH:-${DATA_ROOT}/train.json}"
: "${MODEL_PATH:?Set MODEL_PATH or MODEL_DIR to the Qwen-Image model directory}"
: "${DATA_ROOT:?Set DATA_ROOT or DATA_DIR to the image root}"
: "${DATASET_SIZE:?Set DATASET_SIZE to the exact metadata row count}"
: "${SAVE_DIR:?Set SAVE_DIR to the experiment checkpoint root}"
PROJECT_NAME="${PROJECT_NAME:-Relax/sft/condition-cache}"
EXP_NAME="${EXP_NAME:-qwen-image-sft-condition-cache}"
for argument in "$@"; do
    case "$argument" in
        --global-batch-size|--global-batch-size=*|--rollout-batch-size|--rollout-batch-size=*|\
        --micro-batch-size|--micro-batch-size=*|--n-samples-per-prompt|--n-samples-per-prompt=*|\
        --fsdp-trainable-mode|--fsdp-trainable-mode=*)
            echo "Set batch sizes through GLOBAL_BATCH_SIZE/MICRO_BATCH_SIZE; this recipe fixes LoRA and cannot override $argument" >&2
            exit 1 ;;
    esac
done
export RELAX_CONDITION_CACHE="${RELAX_CONDITION_CACHE:-0}"
export RELAX_CONDITION_PROFILE="${RELAX_CONDITION_PROFILE:-1}"
export RELAX_CONDITION_VERIFY="${RELAX_CONDITION_VERIFY:-0}"
export RELAX_CONDITION_VERIFY_SNAPSHOT="${RELAX_CONDITION_VERIFY_SNAPSHOT:-0}"
export RELAX_CONDITION_VERIFY_RTOL="${RELAX_CONDITION_VERIFY_RTOL:-0.01}"
export RELAX_CONDITION_CACHE_MAX_BYTES="${RELAX_CONDITION_CACHE_MAX_BYTES:-1073741824}"
export RELAX_CONDITION_CACHE_DIR="${RELAX_CONDITION_CACHE_DIR:-}"
export RELAX_CONDITION_CACHE_RUN_DIR="${RELAX_CONDITION_CACHE_RUN_DIR:-${SAVE_DIR}/${EXP_NAME}/metrics}"
export RELAX_CONDITION_PREFETCH_DEPTH="${RELAX_CONDITION_PREFETCH_DEPTH:-0}"
export RELAX_CONDITION_PREFETCH_MAX_BYTES="${RELAX_CONDITION_PREFETCH_MAX_BYTES:-536870912}"
export RAY_NO_WAIT="${RAY_NO_WAIT:-1}"
GPUS="${GPUS:-8}"
GPUS_PER_NODE="${GPUS_PER_NODE:-$GPUS}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-8}"
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-1}"
STEPS="${STEPS:-64}"
SEED="${SEED:-42}"
HEIGHT="${HEIGHT:-512}"
WIDTH="${WIDTH:-512}"
MAX_SEQUENCE_LENGTH="${MAX_SEQUENCE_LENGTH:-512}"
TEXT_ENCODER_PLACEMENT="${TEXT_ENCODER_PLACEMENT:-sharded}"
# Joint text/posterior reuse requires repeated (augmented image, caption) pairs.
# Keep this identical for online/cache: it also keeps the frozen VAE resident.
CACHE_VAE_POSTERIOR="${CACHE_VAE_POSTERIOR:-0}"
CAPTION_DROPOUT="${CAPTION_DROPOUT:-0}"
RANDOM_CROP="${RANDOM_CROP:-0}"
IMAGE_PREFETCH_DEPTH="${IMAGE_PREFETCH_DEPTH:-0}"
IMAGE_PREFETCH_WORKERS="${IMAGE_PREFETCH_WORKERS:-2}"
export DATA_ROOT DATASET_SIZE GPUS GPUS_PER_NODE GLOBAL_BATCH_SIZE MICRO_BATCH_SIZE STEPS SEED
export HEIGHT WIDTH MAX_SEQUENCE_LENGTH CAPTION_DROPOUT RANDOM_CROP
export TEXT_ENCODER_PLACEMENT CACHE_VAE_POSTERIOR
export IMAGE_PREFETCH_DEPTH IMAGE_PREFETCH_WORKERS
export DETERMINISTIC_ATTENTION="${DETERMINISTIC_ATTENTION:-0}"
export GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-1}"

T2I_CONFIG=$(python3 - <<'PYCONFIG'
import json
import math
import os

for name in ("RELAX_CONDITION_CACHE", "RELAX_CONDITION_PROFILE", "RELAX_CONDITION_VERIFY", "RELAX_CONDITION_VERIFY_SNAPSHOT", "DETERMINISTIC_ATTENTION", "GRADIENT_CHECKPOINTING", "RANDOM_CROP", "CACHE_VAE_POSTERIOR"):
    if os.environ[name] not in ("0", "1"):
        raise SystemExit(f"{name} must be 0 or 1")
positive = ("DATASET_SIZE", "GPUS", "GPUS_PER_NODE", "GLOBAL_BATCH_SIZE", "MICRO_BATCH_SIZE", "STEPS", "HEIGHT", "WIDTH", "MAX_SEQUENCE_LENGTH", "IMAGE_PREFETCH_WORKERS")
for name in positive:
    if not os.environ[name].isdecimal() or int(os.environ[name]) <= 0:
        raise SystemExit(f"{name} must be a positive integer")
for name in ("RELAX_CONDITION_CACHE_MAX_BYTES", "IMAGE_PREFETCH_DEPTH", "RELAX_CONDITION_PREFETCH_DEPTH", "RELAX_CONDITION_PREFETCH_MAX_BYTES"):
    if not os.environ[name].isdecimal():
        raise SystemExit(f"{name} must be a nonnegative integer")
gpus, per_node = int(os.environ["GPUS"]), int(os.environ["GPUS_PER_NODE"])
batch, micro = int(os.environ["GLOBAL_BATCH_SIZE"]), int(os.environ["MICRO_BATCH_SIZE"])
if micro != 1:
    raise SystemExit("Qwen-Image SFT currently requires MICRO_BATCH_SIZE=1; use GLOBAL_BATCH_SIZE for accumulation")
if gpus % per_node or batch % (gpus * micro):
    raise SystemExit("GPUS must be divisible by GPUS_PER_NODE; GLOBAL_BATCH_SIZE by GPUS * MICRO_BATCH_SIZE")
if int(os.environ["HEIGHT"]) % 16 or int(os.environ["WIDTH"]) % 16:
    raise SystemExit("HEIGHT and WIDTH must be multiples of 16")
if int(os.environ["MAX_SEQUENCE_LENGTH"]) > 512:
    raise SystemExit("MAX_SEQUENCE_LENGTH must be at most 512")
if os.environ["TEXT_ENCODER_PLACEMENT"] not in ("sharded", "replicated"):
    raise SystemExit("TEXT_ENCODER_PLACEMENT must be sharded or replicated")
if os.environ["RELAX_CONDITION_CACHE"] == "1" and int(os.environ["RELAX_CONDITION_CACHE_MAX_BYTES"]) == 0:
    raise SystemExit("An enabled cache requires a positive byte budget")
if int(os.environ["RELAX_CONDITION_PREFETCH_DEPTH"]) != 0 or int(os.environ["IMAGE_PREFETCH_DEPTH"]) != 0:
    raise SystemExit("Qwen-Image SFT currently requires condition and image prefetch depth 0")
try:
    tolerance = float(os.environ["RELAX_CONDITION_VERIFY_RTOL"])
    dropout = float(os.environ["CAPTION_DROPOUT"])
except ValueError:
    raise SystemExit("Verification tolerance and caption dropout must be numeric") from None
if not math.isfinite(tolerance) or not 0 <= tolerance <= 0.01:
    raise SystemExit("RELAX_CONDITION_VERIFY_RTOL must be finite and between 0 and 0.01")
if not math.isfinite(dropout) or not 0 <= dropout <= 1:
    raise SystemExit("CAPTION_DROPOUT must be finite and between 0 and 1")
config = {
    "data_root": os.environ["DATA_ROOT"],
    "dataset_size": int(os.environ["DATASET_SIZE"]),
    "height": int(os.environ["HEIGHT"]),
    "width": int(os.environ["WIDTH"]),
    "max_sequence_length": int(os.environ["MAX_SEQUENCE_LENGTH"]),
    "text_encoder_placement": os.environ["TEXT_ENCODER_PLACEMENT"],
    "cache_vae_posterior": os.environ["CACHE_VAE_POSTERIOR"] == "1",
    "gradient_checkpointing": os.environ["GRADIENT_CHECKPOINTING"] == "1",
    "deterministic_attention": os.environ["DETERMINISTIC_ATTENTION"] == "1",
    "caption_dropout": dropout,
    "random_crop": os.environ["RANDOM_CROP"] == "1",
    "image_prefetch_depth": int(os.environ["IMAGE_PREFETCH_DEPTH"]),
    "image_prefetch_workers": int(os.environ["IMAGE_PREFETCH_WORKERS"]),
}
print(json.dumps(config))
PYCONFIG
)
RUNTIME_ENV_JSON=$(python3 - <<'PY'
import json
import os

runtime = json.loads(os.environ.get("RUNTIME_ENV_JSON", "{}"))
env = runtime.setdefault("env_vars", {})
env.setdefault("PATH", os.environ["PATH"])
for key in ("RELAX_CONDITION_CACHE", "RELAX_CONDITION_PROFILE", "RELAX_CONDITION_CACHE_MAX_BYTES",
            "RELAX_CONDITION_CACHE_DIR", "RELAX_CONDITION_CACHE_RUN_DIR", "RELAX_CONDITION_VERIFY",
            "RELAX_CONDITION_VERIFY_RTOL", "RELAX_CONDITION_VERIFY_SNAPSHOT", "RELAX_CONDITION_PREFETCH_DEPTH", "RELAX_CONDITION_PREFETCH_MAX_BYTES"):
    env[key] = os.environ[key]
print(json.dumps(runtime))
PY
)
SUBMIT_ARGS=(--address "${RAY_DASHBOARD_ADDRESS:?Set RAY_DASHBOARD_ADDRESS to the Ray Jobs HTTP endpoint}")
if [[ -n "$RAY_NO_WAIT" ]]; then SUBMIT_ARGS+=(--no-wait); fi
if [[ -n "${JOB_ID:-}" ]]; then SUBMIT_ARGS+=(--submission-id "$JOB_ID"); fi
if [[ -n "${WORKING_DIR:-}" ]]; then SUBMIT_ARGS+=(--working-dir "$WORKING_DIR"); fi
DRIVER_ENV=(env)
if [[ -n "${DRIVER_PYTHONPATH:-}" ]]; then DRIVER_ENV=(env "PYTHONPATH=$DRIVER_PYTHONPATH"); fi
case "${SAVE_CHECKPOINTS:-1}" in
    0|1) ;;
    *) echo "SAVE_CHECKPOINTS must be 0 or 1" >&2; exit 1 ;;
esac
TRACKING_ARGS=(--use-metrics-service)
case "${USE_CLEARML:-1}" in
    1) TRACKING_ARGS+=(--use-clearml --tb-project-name "$PROJECT_NAME"
        --tb-experiment-name "${EXP_NAME}-$(date +%Y%m%d-%H%M%S)") ;;
    0) ;;
    *) echo "USE_CLEARML must be 0 or 1" >&2; exit 1 ;;
esac
COMMAND=(ray job submit "${SUBMIT_ARGS[@]}" --runtime-env-json "$RUNTIME_ENV_JSON" --
    "${DRIVER_ENV[@]}" python3 -m relax.entrypoints.train
    --train-backend fsdp --loss-type sft --generation-task t2i
    --model-path "$MODEL_PATH" --prompt-data "$METADATA_PATH" --t2i-sft-config "$T2I_CONFIG"
    --model-adapter-path relax.models.qwen_image.sft_adapter.QwenImageSFTAdapter
    --resource "{\"actor\": [1, $GPUS], \"sft\": [1, 0]}"
    --actor-num-nodes "$((GPUS / GPUS_PER_NODE))" --actor-num-gpus-per-node "$GPUS_PER_NODE"
    --num-gpus-per-node "$GPUS_PER_NODE" --fsdp-trainable-mode lora
    --fsdp-param-dtype bf16 --fsdp-reduce-dtype fp32 --fsdp-master-dtype fp32
    --lora-rank "${LORA_RANK:-32}" --lora-alpha "${LORA_RANK:-32}"
    --micro-batch-size "$MICRO_BATCH_SIZE" --global-batch-size "$GLOBAL_BATCH_SIZE" --rollout-batch-size "$GLOBAL_BATCH_SIZE"
    --n-samples-per-prompt 1 --num-rollout "$STEPS" --seed "$SEED"
    --lr "${LR:-1e-4}" --lr-warmup-iters "${LR_WARMUP_ITERS:-200}" --weight-decay 0.01
    --adam-beta1 0.9 --adam-beta2 0.999 --adam-eps 1e-8 --clip-grad 1.0
    "${TRACKING_ARGS[@]}")
if [[ "${SAVE_CHECKPOINTS:-1}" == 1 ]]; then
    COMMAND+=(--save "${SAVE_DIR}/${EXP_NAME}" --save-interval "${SAVE_INTERVAL:-64}" --max-actor-ckpt-to-keep 2)
fi
if [[ -n "${LOAD_DIR:-}" ]]; then COMMAND+=(--load "$LOAD_DIR"); fi
COMMAND+=("$@")
if [[ "${DRY_RUN:-0}" == 1 ]]; then
    python3 - "${COMMAND[@]}" <<'PY'
import json
import sys
print(json.dumps(sys.argv[1:]))
PY
    exit 0
fi
if compgen -G "${RELAX_CONDITION_CACHE_RUN_DIR}/events-rank*.jsonl" >/dev/null; then
    echo "Run directory already contains metrics; select a fresh RELAX_CONDITION_CACHE_RUN_DIR" >&2
    exit 1
fi
mkdir -p "$RELAX_CONDITION_CACHE_RUN_DIR"
"${COMMAND[@]}" 2>&1 | tee "${RELAX_CONDITION_CACHE_RUN_DIR}/submission.log"
