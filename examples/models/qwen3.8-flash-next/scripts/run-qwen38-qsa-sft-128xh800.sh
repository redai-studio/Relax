#!/usr/bin/env bash
# Copyright (c) 2026 Relax Authors. All Rights Reserved.
# Qwen3.8-Flash-Next 48-layer multimodal SFT on 16 nodes with 8 H800 GPUs each.
# Use a Relax Hopper image with the Qwen3.8 Megatron/Bridge patches, TE and DeepEP.
# Run on the head of an existing Ray cluster with model, data, images and output
# on paths accessible to all workers.
#
# MODEL_DIR=/path/to/models MODEL_NAME=Qwen3.8-Flash-Next \
# DATA_DIR=/path/to/train.jsonl SAVE_DIR=/path/to/output \
#   bash examples/models/qwen3.8-flash-next/scripts/run-qwen38-qsa-sft-128xh800.sh
#
# MODEL_DIR can also point directly to the HF checkpoint when MODEL_NAME is unset.
# With DATA_DIR set to a directory, DATA_FILE defaults to train.jsonl. Rows use
# messages, optional tools and images. Image paths in the JSONL must be readable.
# Defaults to 15 updates from HF, with a Python-stack trace on step 11.
# Set PROFILE_STEP=0 to disable profiling and USE_CLEARML=0 for local logging.
# Set SAVE_INTERVAL to enable checkpoint save/resume under SAVE_DIR/EXP_NAME.
# Keep SAVE_DIR and EXP_NAME fixed when resuming. DRY_RUN=1 previews the launch.
# Set RELAX_QWEN38_RETAIN_QSA_ATTENTION=0 to use regular activation recomputation.
# Set RELAX_SKIP_CLUSTER_CLEANUP=1 when reusing a Ray cluster with active workloads.

set -eo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
RELAX_ROOT="$(cd -- "${SCRIPT_DIR}/../../../.." && pwd)"
export RELAX="${RELAX_ROOT}"
export RAY_NO_WAIT="${RAY_NO_WAIT-}"

: "${MODEL_DIR:?Set MODEL_DIR to the HF checkpoint or its parent directory}"
: "${DATA_DIR:?Set DATA_DIR to the JSONL file or its parent directory}"
: "${SAVE_DIR:?Set SAVE_DIR to a shared run output directory}"
MODEL_PATH="${MODEL_DIR}${MODEL_NAME:+/${MODEL_NAME}}"
DATA_PATH="${DATA_DIR}"
if [[ "${DATA_DIR}" != *.jsonl ]]; then
    DATA_PATH="${DATA_DIR}/${DATA_FILE:-train.jsonl}"
fi

if [[ "${DRY_RUN:-0}" != 1 && -z "${MEGATRON:-}" ]]; then
    MEGATRON=$(python3 -c 'import importlib.util; from pathlib import Path; print(Path(importlib.util.find_spec("megatron.core").origin).resolve().parents[2])')
fi
export MEGATRON
if [[ "${DRY_RUN:-0}" != 1 && -z "${RELAX_ENTRYPOINT_MODE:-}" ]]; then
    exec bash "${RELAX_ROOT}/scripts/entrypoint/ray-job.sh" "${BASH_SOURCE[0]}" "$@"
fi

export PYTHONPATH="${RELAX_ROOT}${MEGATRON:+:${MEGATRON}}${PYTHONPATH:+:${PYTHONPATH}}"
source "${RELAX_ROOT}/scripts/models/qwen38-flash-next.sh"

PROJECT_NAME="${PROJECT_NAME:-Relax/sft/qwen38}"
EXP_NAME="${EXP_NAME:-qwen38-qsa-sft-128xh800}"
now=$(date '+%Y%m%d-%H%M%S')
SUBMISSION_ID="${SUBMISSION_ID:-${EXP_NAME}-${now}-$$}"
NUM_STEPS="${NUM_STEPS:-15}"
if [[ ! "${NUM_STEPS}" =~ ^[1-9][0-9]*$ ]]; then
    echo 'NUM_STEPS must be a positive integer.' >&2
    exit 2
fi
PROFILE_STEP="${PROFILE_STEP:-$((NUM_STEPS >= 11 ? 11 : 0))}"
if [[ ! "${PROFILE_STEP}" =~ ^(0|[1-9][0-9]*)$ ]] || (( PROFILE_STEP > NUM_STEPS )); then
    echo 'PROFILE_STEP must be 0 or an update between 1 and NUM_STEPS.' >&2
    exit 2
fi

ENV_ARGS=(
    "RELAX=${RELAX_ROOT}"
    "MEGATRON=${MEGATRON:-}"
    "PYTHONPATH=${PYTHONPATH}"
    "LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}"
    "TENSORBOARD_DIR=${TENSORBOARD_DIR:-${SAVE_DIR}/${EXP_NAME}/tensorboard}"
    "CLEARML_CACHE_DIR=${CLEARML_CACHE_DIR:-${SAVE_DIR}/${EXP_NAME}/clearml}"
    "TORCHINDUCTOR_CACHE_DIR=${TORCHINDUCTOR_CACHE_DIR:-${TMPDIR:-/tmp}/relax-qwen38/inductor}"
    "TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-${TMPDIR:-/tmp}/relax-qwen38/triton}"
    "CUDA_CACHE_PATH=${CUDA_CACHE_PATH:-${TMPDIR:-/tmp}/relax-qwen38/cuda}"
    PYTHONNOUSERSITE=1
    TOKENIZERS_PARALLELISM=false
    CUDA_DEVICE_MAX_CONNECTIONS=1
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,garbage_collection_threshold:0.8
    TORCHINDUCTOR_COMPILE_THREADS=1
    NCCL_NVLS_ENABLE=0
    NVSHMEM_DISABLE_NCCL=1
    EP_DISABLE_GIN=1
    INDEXER_ROPE_NEOX_STYLE=0
    NVTE_FLASH_ATTN_V2=1
    NVTE_FLASH_ATTN_V3=0
    NVTE_FLASH_ATTN_V4=0
    NVTE_USE_CUTLASS_GROUPED_GEMM=0
    NVTE_CUTLASS_GROUPED_GEMM_WARN_FALLBACK=1
    RAY_OVERRIDE_JOB_RUNTIME_ENV=1
    QWEN38_PLE_CPU_OFFLOAD=1
    MCORE_QSA_SPARSE_BACKEND=flex
    RELAX_QWEN38_PLE_STREAMED=0
    RELAX_QWEN38_PLE_FUSED_CONV=1
    RELAX_QWEN38_HC_FUSED_MIX=1
    RELAX_QWEN38_HC_FUSED_COMBINE=1
    RELAX_QWEN38_FLA_CONV_CACHE_REUSE=1
    RELAX_QWEN38_QSA_COMMUNITY_BACKEND=
    RELAX_QWEN38_QSA_FUSED_ATTENTION=triton_v2
    RELAX_QWEN38_QSA_FUSED_SELECTOR=1
    RELAX_QWEN38_QSA_SELECTOR_TILE_MIB=384
    RELAX_QWEN38_QSA_SPARSE_BLOCKMASK=1
    RELAX_QWEN38_QSA_LOCAL_QUERY=1
    RELAX_QWEN38_QSA_ROUTED_QKV=1
    RELAX_QWEN38_QSA_RECOMPUTE_REPLAY=1
    RELAX_QWEN38_QSA_CPU_LAYOUT=0
    RELAX_QWEN38_QSA_TP_SELECTOR=0
    "RELAX_QWEN38_RETAIN_QSA_ATTENTION=${RELAX_QWEN38_RETAIN_QSA_ATTENTION:-1}"
    "RELAX_QWEN38_QSA_RETENTION_POLICY=${RELAX_QWEN38_QSA_RETENTION_POLICY:-last_stage_l48}"
    RELAX_SFT_FUSED_LINEAR_CE=0
    RELAX_SFT_SELECTIVE_LOGITS=1
    RELAX_SFT_SELECTIVE_MULTIMODAL_LOGITS=1
    RELAX_SFT_ASSISTANT_MEDIA_TAGS_AS_TEXT=1
    RELAX_LOGPROB_CE_IMPL=te
    RELAX_MTP_CE_IMPL=te
)
TRAIN_ENV_JSON=$(python3 - "${ENV_ARGS[@]}" <<'PYENV'
import json
import os
import sys

env = dict(item.split("=", 1) for item in sys.argv[1:])
runtime = json.loads(os.environ.get("RUNTIME_ENV_JSON") or "{}")
env["LD_LIBRARY_PATH"] = runtime.get("env_vars", {}).get("LD_LIBRARY_PATH", env["LD_LIBRARY_PATH"])
print(json.dumps(env))
PYENV
)
RUNTIME_ENV_JSON=$(python3 - "${TRAIN_ENV_JSON}" <<'PYRUNTIME'
import json
import os
import sys

runtime = json.loads(os.environ.get("RUNTIME_ENV_JSON") or "{}")
env = runtime.setdefault("env_vars", {})
training = json.loads(sys.argv[1])
env.update(training)
forwarded = [
    *env.get("RELAX_PROPAGATE_ENV_VARS", "").split(","),
    *os.environ.get("RELAX_PROPAGATE_ENV_VARS", "").split(","),
    *training,
]
env["RELAX_PROPAGATE_ENV_VARS"] = ",".join(dict.fromkeys(name for name in forwarded if name))
print(json.dumps(runtime))
PYRUNTIME
)

CKPT_ARGS=(
    --hf-checkpoint "${MODEL_PATH}"
    --megatron-to-hf-mode bridge
)
if [[ -n "${SAVE_INTERVAL:-}" ]]; then
    CKPT_ARGS+=(
        --load "${SAVE_DIR}/${EXP_NAME}/checkpoints"
        --save "${SAVE_DIR}/${EXP_NAME}/checkpoints"
        --save-interval "${SAVE_INTERVAL}"
        --max-actor-ckpt-to-keep 2
    )
fi

SFT_ARGS=(
    --loss-type sft
    --prompt-data "${DATA_PATH}"
    --num-rollout "${NUM_STEPS}"
    --eval-size "${EVAL_SIZE:-512}"
    --eval-interval "${EVAL_INTERVAL:-100}"
    --input-key messages
    --tool-key tools
    --apply-chat-template
    --apply-chat-template-kwargs '{"add_generation_prompt":false,"enable_thinking":false,"preserve_thinking":true,"add_non_thinking_prefix":true}'
    --multimodal-keys '{"image":"images"}'
    --image-max-token-num "${IMAGE_MAX_TOKEN_NUM:-256}"
    --sft-invalid-multimodal-strategy error
    --sft-oversize-strategy skip
    --calculate-per-token-loss
    --rollout-max-context-len 32768
    --seq-length 32768
    --global-batch-size 512
    --micro-batch-size 1
    --use-dynamic-batch-size
    --max-tokens-per-gpu 65536
    --data-pad-size-multiplier 4096
    --per-rank-fetch
    --sft-prefetch-num-workers 16
    --sft-train-data-prefetch
    --freeze-vision-model
    --freeze-vision-projection
    --freeze-params-name-list 'self_attention\.indexer\.' '(^|\.)per_layer_embedding\.ple_embedding\.ngram_embedding\.weight$'
    --enable-mtp-training
    --mtp-num-layers 1
    --mtp-loss-scaling-factor 0.1
    --mtp-detach-paths none
)
PARALLEL_ARGS=(
    --tensor-model-parallel-size 4
    --pipeline-model-parallel-size 8
    --context-parallel-size 1
    --expert-model-parallel-size 8
    --expert-tensor-parallel-size 1
    --pipeline-model-parallel-layout 'Ettttt|ttttttt|tttttt|tttttt|tttttt|ttttttt|ttttttt|ttttmL'
    --sequence-parallel
    --moe-token-dispatcher-type flex
    --moe-flex-dispatcher-backend deepep
    --moe-flex-dispatcher-num-sms 24
    --moe-shared-expert-overlap
    --moe-grouped-gemm
    --colocate
    --no-offload-train
    --sft-chunked-logits
    --sft-logits-chunk-size 3072
    --recompute-granularity full
    --recompute-method uniform
    --recompute-num-layers 1
    --no-rope-fusion
    --attention-backend flash
)
OPTIMIZER_ARGS=(
    --optimizer adam
    --lr "${LR:-1e-5}"
    --min-lr "${MIN_LR:-1e-6}"
    --lr-decay-style cosine
    --lr-decay-iters "${LR_DECAY_ITERS:-${NUM_STEPS}}"
    --lr-warmup-fraction 0.05
    --weight-decay 0.1
    --adam-beta1 0.9
    --adam-beta2 0.98
    --clip-grad 1.0
    --chunked-optimizer-state-offload
    --optimizer-state-offload-fraction 1.0
    --optimizer-state-offload-chunk-size-mb 512
    --overlap-grad-reduce
    --overlap-param-gather
    --use-precision-aware-optimizer
    --use-distributed-optimizer
    --no-store-param-remainders
    --dist-ckpt-optim-fully-reshardable
    --moe-router-load-balancing-type global_aux_loss
    --moe-aux-loss-coeff 1e-6
)
TRACKING_ARGS=(
    --use-metrics-service
    --use-tensorboard
    --tb-project-name "${PROJECT_NAME}"
    --tb-experiment-name "${EXP_NAME}-${now}"
)
if [[ "${USE_CLEARML:-1}" == 1 ]]; then
    TRACKING_ARGS+=(--use-clearml)
fi
if (( PROFILE_STEP > 0 )); then
    # Profiler steps are zero-based.
    profile_index=$((PROFILE_STEP - 1))
    TRACKING_ARGS+=(
        --use-pytorch-profiler
        --profile-target train_overall
        --profile-step-start "${profile_index}"
        --profile-step-end "${profile_index}"
        --profile-with-stack
        --profile-ranks 0 16 32 48 64 80 96 112
        --profile-output-dir "${PROFILE_DIR:-${SAVE_DIR}/${EXP_NAME}/traces}"
    )
fi

TRAIN_ARGS=(
    --resource '{"sft":[1,0],"actor":[1,128]}'
    --actor-num-nodes 16
    --actor-num-gpus-per-node 8
    --sft-max-in-flight-steps 2
    --num-data-storage-units 2
    "${MODEL_ARGS[@]}"
    "${CKPT_ARGS[@]}"
    "${SFT_ARGS[@]}"
    "${PARALLEL_ARGS[@]}"
    "${OPTIMIZER_ARGS[@]}"
    "${TRACKING_ARGS[@]}"
    --seed "${SEED:-1234}"
    --bf16
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --accumulate-allreduce-grads-in-fp32
    --attention-softmax-in-fp32
    --train-env-vars "${TRAIN_ENV_JSON}"
    "$@"
)

if [[ "${DRY_RUN:-0}" == 1 ]]; then
    python3 - "${RUNTIME_ENV_JSON}" python3 -m relax.entrypoints.train "${TRAIN_ARGS[@]}" <<'PYDRY'
import json
import sys

print(json.dumps({"runtime_env": json.loads(sys.argv[1]), "argv": sys.argv[2:]}, indent=2))
PYDRY
    exit 0
fi

cd "${RELAX_ROOT}"
mkdir -p "${SAVE_DIR}/${EXP_NAME}/logs"
RAY_JOB_ADDRESS_ARGS=()
if [[ -n "${RAY_JOBS_ADDRESS:-}" ]]; then
    RAY_JOB_ADDRESS_ARGS=(--address "${RAY_JOBS_ADDRESS}")
fi
ray job submit ${RAY_NO_WAIT:+--no-wait} "${RAY_JOB_ADDRESS_ARGS[@]}" \
    --submission-id "${SUBMISSION_ID}" \
    ${WORKING_DIR:+--working-dir "${WORKING_DIR}"} \
    --runtime-env-json="${RUNTIME_ENV_JSON}" \
    -- python3 -m relax.entrypoints.train "${TRAIN_ARGS[@]}" \
    2>&1 | tee "${SAVE_DIR}/${EXP_NAME}/logs/${EXP_NAME}-${now}.log"
