#!/bin/bash
# Copyright (c) 2026 Relax Authors. All Rights Reserved.
# Kimi K3 Think multimodal LoRA SFT on 16 x 8 B300; derived from official K3 recipes.

set -eo pipefail
export RAY_NO_WAIT="${RAY_NO_WAIT-}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
RELAX_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." &>/dev/null && pwd)"
MODEL_CONFIG_DIR="${MODEL_CONFIG_DIR:-${RELAX_ROOT}/scripts/models}"
source "${MODEL_CONFIG_DIR}/kimi-k3.sh"

# Reuse the prepared dataset; isolate checkpoints from the previous completed run.
# Set RELAX_SKIP_CLUSTER_CLEANUP=1 before invoking ray-job.sh, not here:
# entrypoint cleanup happens before this training script is executed.
MODEL_DIR="${MODEL_DIR:-/mnt/tidalfs-awsatl01-ren/dataset/pai/MODEL_CACHE/kimi_k3}"
DATA_DIR="${DATA_DIR:-/mnt/tidalfs-awsatl01-ren/dataset/pai/xianqing/runs/kimi-k3-lora-sft-20261025-004009-c61d3b9b/data}"
SAVE_DIR="${SAVE_DIR:-/mnt/tidalfs-awsatl01-ren/dataset/pai/xianqing/runs/kimi-k3-lora-sft-20261025-004009-c61d3b9b/checkpoints-high-fp32}"

HF_CHECKPOINT="${HF_CHECKPOINT:-${MODEL_DIR:-}}"
: "${HF_CHECKPOINT:?Set MODEL_DIR or HF_CHECKPOINT to the shared HF model directory}"
LLAVA_DATA_DIR="${LLAVA_DATA_DIR:-${DATA_DIR:-}}"
: "${LLAVA_DATA_DIR:?Set DATA_DIR or LLAVA_DATA_DIR to the prepared dataset directory}"
PROMPT_DATA="${PROMPT_DATA:-${LLAVA_DATA_DIR}/train.jsonl}"
PROJECT_NAME="${PROJECT_NAME:-Relax/sft/kimi-k3-think-lora}"
EXP_NAME="${EXP_NAME:-kimi-k3-think-lora-b300-gpu128}"
if [ "${DRY_RUN:-0}" != 1 ] && [ ! -f "${LLAVA_DATA_DIR}/READY.json" ]; then
    echo "Data preparation is incomplete: ${LLAVA_DATA_DIR}/READY.json is missing." >&2
    exit 1
fi
NUM_STEPS="${NUM_STEPS-}"
MAX_TOKENS_PER_GPU="${MAX_TOKENS_PER_GPU:-32768}"
FREEZE_VISION_TOWER="${FREEZE_VISION_TOWER:-1}"
now=$(date "+%Y-%m-%d-%H-%M-%S")
export KIMI_SUBMISSION_ID="${KIMI_SUBMISSION_ID:-kimi-k3-think-lora-gpu128-${now}-$$}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"
export RAY_USAGE_STATS_ENABLED="${RAY_USAGE_STATS_ENABLED:-0}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-/tmp/kimi-k3-extensions-${KIMI_SUBMISSION_ID}}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/tmp/kimi-k3-triton-${KIMI_SUBMISSION_ID}}"
# Persist compiler artifacts on shared storage; compile on node-local NVMe.
export RELAX_KERNEL_CACHE_DIR="${RELAX_KERNEL_CACHE_DIR:-${RELAX_ROOT}/.cache/relax-kernels}"
export RELAX_KERNEL_CACHE_LOCAL_DIR="${RELAX_KERNEL_CACHE_LOCAL_DIR:-/data/temp/relax-kernel-cache}"

# The source HellaSwag script enabled text-only rendering. Clear any inherited
# setting so SFT uses Kimi K3's multimodal chat renderer for training.
unset RELAX_SFT_RAW_TEXT_CONCAT

# All-to-all does not use NVSHMEM.
export NVSHMEM_REMOTE_TRANSPORT="${NVSHMEM_REMOTE_TRANSPORT:-libfabric}"

CKPT_ARGS=(
    --hf-checkpoint "${HF_CHECKPOINT}"
    --trust-remote-code
    --ref-load "${HF_CHECKPOINT}"
    --megatron-to-hf-mode bridge
    --num-epoch "${NUM_EPOCH:-2}"
    --save-interval "${SAVE_INTERVAL:-50}"
    --max-actor-ckpt-to-keep 3
)
# Leave epoch-based training unchanged unless an explicit step limit is set.
if [ -n "${NUM_STEPS}" ]; then
    CKPT_ARGS+=(--num-rollout "${NUM_STEPS}")
fi
# Enable checkpoint saving with a stable shared directory.
if [ -n "${SAVE_DIR:-}" ]; then
    CKPT_ARGS+=(--save "${SAVE_DIR}/${EXP_NAME}" --load "${SAVE_DIR}/${EXP_NAME}"
        --save-interval "${SAVE_INTERVAL:-50}")
fi
# Allow an explicit checkpoint to override the experiment's default resume path.
if [ -n "${LOAD_DIR:-}" ]; then
    CKPT_ARGS+=(--load "${LOAD_DIR}")
fi

# LoRA-only checkpoints retain adapter optimizer and RNG state for resume.

SFT_ARGS=(
    --loss-type sft
    # Match Swift last_round+ignore_think_prefix: learn empty-think closing tags.
    --sft-loss-last-turn-only
    # K3 preserves historical reasoning natively; default to high thinking effort.
    --apply-chat-template-kwargs '{"thinking":true,"thinking_effort":"high"}'
    --prompt-data "${PROMPT_DATA}"
    --input-key messages
    --tool-key tools
    --multimodal-keys '{"image":"images"}'
    --image-max-token-num 1024
    --global-batch-size "${GLOBAL_BATCH_SIZE:-64}"
    # dataloader.shuffle: true in the recipe.
    --rollout-shuffle
    --use-dynamic-batch-size
    --max-tokens-per-gpu "${MAX_TOKENS_PER_GPU}"
    --data-pad-size-multiplier 4096
    --sft-oversize-strategy skip
    --sft-invalid-multimodal-strategy skip
    --balance-data
    --per-rank-fetch
    --sft-prefetch-num-workers 16
    --sft-prefetch-buffer-size 256
    --sft-tq-timeout-minutes 60
)

# MoE token dispatcher: plain all-to-all.
DISPATCHER_ARGS=(
    --moe-token-dispatcher-type alltoall
)

# Trainable vision uses per-layer activation checkpointing; vision DP requires a frozen tower.
VISION_ARGS=()
if [ "${FREEZE_VISION_TOWER}" = 1 ]; then
    VISION_ARGS=(--vision-dp-when-tp --freeze-vision-model)
fi

PERF_ARGS=(
    --tensor-model-parallel-size 4
    --sequence-parallel
    "${VISION_ARGS[@]}"
    --pipeline-model-parallel-size 4
    --decoder-first-pipeline-num-layers 23
    --decoder-last-pipeline-num-layers 22
    # TP4 x PP4 x CP4 with 128 GPUs gives DP2.
    --context-parallel-size "${CP_SIZE:-4}"
    --calculate-per-token-loss
    # EP16 keeps 896/16 = 56 experts/rank; 32 GPUs per PP stage give EDP2.
    --expert-model-parallel-size "${EP_SIZE:-16}"
    --expert-tensor-parallel-size 1

    --recompute-granularity full
    --recompute-method uniform
    --recompute-num-layers 1

    --optimizer-cpu-offload
    --overlap-cpu-optimizer-d2h-h2d
    --use-precision-aware-optimizer
    --main-grads-dtype fp32
    --optimizer-offload-fraction "${OPTIMIZER_OFFLOAD_FRACTION:-1.0}"

    --disable-weights-backuper
    --cross-entropy-loss-fusion
    --sft-chunked-logits
    --sft-logits-chunk-size "${SFT_LOGITS_CHUNK_SIZE:-1024}"

    "${DISPATCHER_ARGS[@]}"
)

# Restore the original Kimi learning-rate schedule.
OPTIMIZER_ARGS=(
    --optimizer adam
    --lr "${LR:-1e-5}"
    --lr-decay-style cosine
    --lr-warmup-fraction 0.05
    --min-lr 1e-6
    --weight-decay "${WEIGHT_DECAY:-0.1}"
    --adam-beta1 0.9
    --adam-beta2 0.95
    --clip-grad 1.0
    # Requested auxiliary-loss settings. TODO(agent): K3's Bridge currently
    # protects both fields in ARCHITECTURE_KEYS and retains none/0 at runtime.
    --moe-router-load-balancing-type aux_loss
    --moe-aux-loss-coeff 1e-6
)

METRICS_ARGS=(
    --use-tensorboard
    --use-clearml
    --tb-project-name "${PROJECT_NAME}"
    --tb-experiment-name "${EXP_NAME}-${now}"
)

MISC_ARGS=(
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --attention-backend fused
    # rng.seed: 1234 in the recipe.
    --seed "${SEED:-1234}"
)

# Enable the validated expert LoRA, SiTU and AttnRes optimizations.
TRAIN_ENV_VARS="{\"FLA_TILELANG\":\"${FLA_TILELANG:-0}\",\"PYTORCH_CUDA_ALLOC_CONF\":\"expandable_segments:True\",\"RELAX_KIMI_K3_EXPERT_SPLIT\":\"${RELAX_KIMI_K3_EXPERT_SPLIT:-1}\",\"RELAX_KIMI_K3_FUSED_SITU\":\"${RELAX_KIMI_K3_FUSED_SITU:-1}\",\"RELAX_KIMI_K3_FUSED_ATTN_RES\":\"${RELAX_KIMI_K3_FUSED_ATTN_RES:-1}\"}"

TRAIN_ARGS=(
    --train-env-vars "${TRAIN_ENV_VARS}"
    --resource '{"sft": [1, 0], "actor": [1, 128]}'
    --sft-max-in-flight-steps 4
    --num-data-storage-units 16
    "${MODEL_ARGS[@]}"
    "${CKPT_ARGS[@]}"
    "${SFT_ARGS[@]}"
    "${OPTIMIZER_ARGS[@]}"
    "${METRICS_ARGS[@]}"
    "${PERF_ARGS[@]}"
    "${MISC_ARGS[@]}"
    --lora-rank "${LORA_RANK:-16}"
    --lora-alpha "${LORA_ALPHA:-32}"
    --lora-dropout 0.0
    --lora-scope all
    --lora-merge-mode
    --save-lora-only
    --lora-target-modules
    q_proj k_proj v_proj o_proj f_a_proj f_b_proj b_proj g_proj
    q_a_proj q_b_proj kv_a_proj_with_mqa kv_b_proj
    linear_fc1 linear_fc2 fc1_latent_proj fc2_latent_proj
    "$@"
)

if [ "${DRY_RUN:-0}" = 1 ]; then
    printf '%q ' python3 -m relax.entrypoints.train "${TRAIN_ARGS[@]}"
    printf '\n'
    exit 0
fi

if [ -z "${RELAX_ENTRYPOINT_MODE:-}" ]; then
    source "${RELAX_ROOT}/scripts/entrypoint/ray-job.sh"
elif [ -z "${RELAX_KERNEL_CACHE_SESSION_ID:-}" ]; then
    # The outer entrypoint ran before this script supplied its cache defaults.
    source "${RELAX_ROOT}/scripts/entrypoint/kernel-cache.sh"
    relax_kernel_cache_configure "${BASH_SOURCE[0]}" "$@"
    relax_kernel_cache_start_agents cluster
    relax_kernel_cache_inject_runtime_env
fi

# Keep the raw-text rendering mode consistent in Ray workers.
if [ -n "${RUNTIME_ENV_JSON:-}" ]; then
    RUNTIME_ENV_JSON=$(python3 - <<'PY'
import json
import os

runtime = json.loads(os.environ["RUNTIME_ENV_JSON"])
env = runtime.setdefault("env_vars", {})
env.pop("RELAX_SFT_RAW_TEXT_CONCAT", None)
propagate = env.get("RELAX_PROPAGATE_ENV_VARS", "").split(",")
env["RELAX_PROPAGATE_ENV_VARS"] = ",".join(
    dict.fromkeys(name for name in propagate if name and name != "RELAX_SFT_RAW_TEXT_CONCAT")
)
print(json.dumps(runtime))
PY
    )
    export RUNTIME_ENV_JSON
fi

RUNTIME_ENV_JSON=$(python3 - <<'PYENV'
import json, os
runtime = json.loads(os.environ["RUNTIME_ENV_JSON"])
env = runtime.setdefault("env_vars", {})
env["RELAX_LORA_SHARE_EXPERT_ADAPTERS"] = "false"
# ray-job.sh builds runtime_env before exec'ing us, so local exports alone
# would not update Ray workers (including SFT image-processing workers).
worker_env_names = (
    "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
    "RAY_USAGE_STATS_ENABLED", "NVSHMEM_REMOTE_TRANSPORT",
    "TORCH_EXTENSIONS_DIR", "TRITON_CACHE_DIR",
)
env.update({name: os.environ[name] for name in worker_env_names})
# Credentials/endpoints are supplied externally; never store their values here.
clearml_env_names = (
    "CLEARML_API_HOST", "CLEARML_WEB_HOST", "CLEARML_FILES_HOST",
    "CLEARML_API_ACCESS_KEY", "CLEARML_API_SECRET_KEY",
    "CLEARML_CONFIG_FILE", "CLEARML_TAGS",
)
env.update({name: os.environ[name] for name in clearml_env_names if name in os.environ})
propagate = env.get("RELAX_PROPAGATE_ENV_VARS", "").split(",")
env["RELAX_PROPAGATE_ENV_VARS"] = ",".join(dict.fromkeys(
    name for name in [*propagate, "RELAX_LORA_SHARE_EXPERT_ADAPTERS", *worker_env_names,
                     *(name for name in clearml_env_names if name in env)] if name
))
print(json.dumps(runtime))
PYENV
)
export RUNTIME_ENV_JSON

mkdir -p "${RELAX_ROOT}/log"
ray job submit ${RAY_NO_WAIT:+--no-wait} --submission-id="${KIMI_SUBMISSION_ID:?}" --address="${RAY_DASHBOARD_ADDRESS:-http://127.0.0.1:8265}" \
    ${WORKING_DIR:+--working-dir "${WORKING_DIR}"} \
    --runtime-env-json="${RUNTIME_ENV_JSON}" \
    -- python3 -m relax.entrypoints.train "${TRAIN_ARGS[@]}" \
    2>&1 | tee "${RELAX_ROOT}/log/${EXP_NAME}-${now}.log"
