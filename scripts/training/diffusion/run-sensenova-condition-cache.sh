#!/usr/bin/env bash
# Copyright (c) 2026 Relax Authors. All Rights Reserved.
# Compare online and cached SenseNova LoRA SFT with the same effective batch.
# Invoke through scripts/entrypoint/ray-job.sh on an existing Ray cluster.
# MODEL_PATH=/models/SenseNova DATA_ROOT=/data/MagicBrush METADATA_PATH=/data/train.json \
# DATASET_SIZE=128 SAVE_DIR=/runs RELAX_CONDITION_CACHE_RUN_DIR=/runs/online-r01 \
# RELAX_CONDITION_CACHE=0 bash scripts/entrypoint/ray-job.sh \
# scripts/training/diffusion/run-sensenova-condition-cache.sh
# Repeat with a fresh run directory and RELAX_CONDITION_CACHE=1. DRY_RUN=1 prints
# the submission without sourcing entrypoints, creating directories or using Ray.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
RELAX_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
if [[ "${DRY_RUN:-0}" != 1 && -z "${RELAX_ENTRYPOINT_MODE:-}" ]]; then
    # shellcheck source=../../entrypoint/ray-job.sh
    source "${RELAX_ROOT}/scripts/entrypoint/ray-job.sh"
fi

MODEL_PATH="${MODEL_PATH:-${MODEL_DIR:-}}"
DATA_ROOT="${DATA_ROOT:-${DATA_DIR:-}}"
METADATA_PATH="${METADATA_PATH:-${DATA_ROOT}/train.json}"
: "${MODEL_PATH:?Set MODEL_PATH or MODEL_DIR to the SenseNova model directory}"
: "${DATA_ROOT:?Set DATA_ROOT or DATA_DIR to the image root}"
: "${DATASET_SIZE:?Set DATASET_SIZE to the exact metadata row count}"
: "${SAVE_DIR:?Set SAVE_DIR to the experiment checkpoint root}"
PROJECT_NAME="${PROJECT_NAME:-Relax/sft/condition-cache}"
EXP_NAME="${EXP_NAME:-sensenova-condition-cache}"
for argument in "$@"; do
    case "$argument" in
        --global-batch-size|--global-batch-size=*|--rollout-batch-size|--rollout-batch-size=*|\
        --micro-batch-size|--micro-batch-size=*|--n-samples-per-prompt|--n-samples-per-prompt=*|\
        --fsdp-trainable-mode|--fsdp-trainable-mode=*)
            echo "This comparison recipe fixes LoRA and effective batch 8; cannot override $argument" >&2
            exit 1 ;;
    esac
done
export RELAX_CONDITION_CACHE="${RELAX_CONDITION_CACHE:-0}"
export RELAX_CONDITION_PROFILE="${RELAX_CONDITION_PROFILE:-1}"
export RELAX_CONDITION_VERIFY="${RELAX_CONDITION_VERIFY:-0}"
export RELAX_CONDITION_VERIFY_SNAPSHOT="${RELAX_CONDITION_VERIFY_SNAPSHOT:-0}"
export RELAX_CONDITION_VERIFY_RTOL="${RELAX_CONDITION_VERIFY_RTOL:-0.01}"
export RELAX_CONDITION_CACHE_MAX_BYTES="${RELAX_CONDITION_CACHE_MAX_BYTES:-8589934592}"
export RELAX_CONDITION_CACHE_DIR="${RELAX_CONDITION_CACHE_DIR:-}"
export RELAX_CONDITION_CACHE_RUN_DIR="${RELAX_CONDITION_CACHE_RUN_DIR:-${SAVE_DIR}/${EXP_NAME}/metrics}"
export RELAX_CONDITION_PREFETCH_DEPTH="${RELAX_CONDITION_PREFETCH_DEPTH:-0}"
export RELAX_CONDITION_PREFETCH_MAX_BYTES="${RELAX_CONDITION_PREFETCH_MAX_BYTES:-536870912}"
export RAY_NO_WAIT="${RAY_NO_WAIT-}"
GPUS="${GPUS:-8}"
GPUS_PER_NODE="${GPUS_PER_NODE:-$GPUS}"
GLOBAL_BATCH_SIZE=8
STEPS="${STEPS:-64}"
SEED="${SEED:-42}"
MAX_PIXELS="${MAX_PIXELS:-1048576}"
CONDITIONING_IMAGES="${CONDITIONING_IMAGES:-1}"
IMAGE_PREFETCH_DEPTH="${IMAGE_PREFETCH_DEPTH:-0}"
IMAGE_PREFETCH_WORKERS="${IMAGE_PREFETCH_WORKERS:-2}"
export DATA_ROOT DATASET_SIZE GPUS GPUS_PER_NODE STEPS SEED MAX_PIXELS CONDITIONING_IMAGES
export IMAGE_PREFETCH_DEPTH IMAGE_PREFETCH_WORKERS
export ATTENTION_IMPLEMENTATION="${ATTENTION_IMPLEMENTATION:-flash_attention_2}"
export DETERMINISTIC_ATTENTION="${DETERMINISTIC_ATTENTION:-0}"
export PREFIX_GRADIENT_CHECKPOINTING="${PREFIX_GRADIENT_CHECKPOINTING:-1}"
export FSDP_BRANCH_GROUPS="${FSDP_BRANCH_GROUPS:-0}"

I2I_CONFIG=$(python3 - <<'PY'
import json
import math
import os

for name in ("RELAX_CONDITION_CACHE", "RELAX_CONDITION_PROFILE", "RELAX_CONDITION_VERIFY", "RELAX_CONDITION_VERIFY_SNAPSHOT", "DETERMINISTIC_ATTENTION", "PREFIX_GRADIENT_CHECKPOINTING", "FSDP_BRANCH_GROUPS"):
    if os.environ[name] not in ("0", "1"):
        raise SystemExit(f"{name} must be 0 or 1")
positive = ("DATASET_SIZE", "GPUS", "GPUS_PER_NODE", "STEPS", "MAX_PIXELS", "CONDITIONING_IMAGES", "IMAGE_PREFETCH_WORKERS")
for name in positive:
    if not os.environ[name].isdecimal() or int(os.environ[name]) <= 0:
        raise SystemExit(f"{name} must be a positive integer")
for name in ("RELAX_CONDITION_CACHE_MAX_BYTES", "IMAGE_PREFETCH_DEPTH", "RELAX_CONDITION_PREFETCH_DEPTH", "RELAX_CONDITION_PREFETCH_MAX_BYTES"):
    if not os.environ[name].isdecimal():
        raise SystemExit(f"{name} must be a nonnegative integer")
gpus, per_node = int(os.environ["GPUS"]), int(os.environ["GPUS_PER_NODE"])
if 8 % gpus or gpus % per_node:
    raise SystemExit("GPUS must divide the fixed effective batch 8 and be divisible by GPUS_PER_NODE")
if os.environ["RELAX_CONDITION_CACHE"] == "1" and int(os.environ["RELAX_CONDITION_CACHE_MAX_BYTES"]) == 0:
    raise SystemExit("An enabled cache requires a positive byte budget")
if int(os.environ["RELAX_CONDITION_PREFETCH_DEPTH"]) > 0:
    if int(os.environ["RELAX_CONDITION_PREFETCH_MAX_BYTES"]) == 0:
        raise SystemExit("Conditioning prefetch requires a positive byte budget")
    if int(os.environ["IMAGE_PREFETCH_DEPTH"]) == 0:
        raise SystemExit("Conditioning prefetch requires IMAGE_PREFETCH_DEPTH > 0")
if int(os.environ["CONDITIONING_IMAGES"]) not in (1, 2):
    raise SystemExit("CONDITIONING_IMAGES must be 1 or 2")
try:
    tolerance = float(os.environ["RELAX_CONDITION_VERIFY_RTOL"])
except ValueError:
    raise SystemExit("RELAX_CONDITION_VERIFY_RTOL must be finite and between 0 and 0.01") from None
if not math.isfinite(tolerance) or not 0 <= tolerance <= 0.01:
    raise SystemExit("RELAX_CONDITION_VERIFY_RTOL must be finite and between 0 and 0.01")
config = {
    "data_root": os.environ["DATA_ROOT"],
    "dataset_size": int(os.environ["DATASET_SIZE"]),
    "max_pixels": int(os.environ["MAX_PIXELS"]),
    "conditioning_images": int(os.environ["CONDITIONING_IMAGES"]),
    "attention_implementation": os.environ["ATTENTION_IMPLEMENTATION"],
    "deterministic_attention": os.environ["DETERMINISTIC_ATTENTION"] == "1",
    "prefix_gradient_checkpointing": os.environ["PREFIX_GRADIENT_CHECKPOINTING"] == "1",
    "image_prefetch_depth": int(os.environ["IMAGE_PREFETCH_DEPTH"]),
    "image_prefetch_workers": int(os.environ["IMAGE_PREFETCH_WORKERS"]),
}
if os.environ["FSDP_BRANCH_GROUPS"] == "1":
    config["fsdp_branch_groups"] = True
print(json.dumps(config))
PY
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
    --train-backend fsdp --loss-type sft --generation-task i2i
    --model-path "$MODEL_PATH" --prompt-data "$METADATA_PATH" --i2i-sft-config "$I2I_CONFIG"
    --resource "{\"actor\": [1, $GPUS], \"sft\": [1, 0]}"
    --actor-num-nodes "$((GPUS / GPUS_PER_NODE))" --actor-num-gpus-per-node "$GPUS_PER_NODE"
    --num-gpus-per-node "$GPUS_PER_NODE" --fsdp-trainable-mode lora --fsdp-master-dtype fp32
    --lora-rank "${LORA_RANK:-32}" --lora-alpha "${LORA_RANK:-32}"
    --micro-batch-size 1 --global-batch-size "$GLOBAL_BATCH_SIZE" --rollout-batch-size "$GLOBAL_BATCH_SIZE"
    --n-samples-per-prompt 1 --num-rollout "$STEPS" --seed "$SEED"
    --lr "${LR:-1e-4}" --lr-warmup-iters 200 --weight-decay 0.01
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
