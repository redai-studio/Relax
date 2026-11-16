#!/usr/bin/env bash
# Copyright (c) 2026 Relax Authors. All Rights Reserved.
# Qwen3.8 multimodal SFT: 1 node x 8 Blackwell GPUs, TP2/PP1/DP4/CP1/EP8.
# Full 48-layer model with one MTP layer; GPU memory feasibility is not yet measured.
# MOE_BACKEND=hybridep (default) or mok; use the same patched Megatron for A/B.
# MoK must be installed or present on PYTHONPATH (propagated to Ray workers).
# MOK_BACKEND_CONFIG optionally supplies the MCore/MoK backend JSON options.
# Both modes pad every packed microbatch to MAX_TOKENS_PER_GPU for equal EP shapes.
# Defaults: 8K context/token budget, global batch 32, full recompute every two layers.
# Override SEQ_LENGTH, MAX_TOKENS_PER_GPU and GLOBAL_BATCH_SIZE for other workloads.
# Run this script directly; it delegates setup to scripts/entrypoint/ray-job.sh.
# DRY_RUN=1 prints JSON without Ray setup, cleanup, network calls or GPU imports.
# MODEL_DIR and DATA_DIR are the exact HF checkpoint and prompt-data paths.
# Multimodal recipe: 50 steps and evaluation every 100 steps.
# Keep SAVE_DIR/EXP_NAME stable to resume; save every 50 steps and retain 2.
# SAVE_CHECKPOINTS=0 disables checkpoint writes, including the final step.
# Usage: MODEL_DIR=... DATA_DIR=... SAVE_DIR=... MEGATRON=... bash <this-script>
# All dependency paths must exist at the same location on every worker node.
# Set RELAX_SKIP_NONFINITE_GRAD=0 for fail-fast diagnostics.
set -eo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
RELAX_ROOT="$(cd -- "${SCRIPT_DIR}/../../../.." && pwd)"
export RELAX="${RELAX_ROOT}"
source "${RELAX_ROOT}/scripts/models/qwen38-flash-next.sh"
: "${MODEL_DIR:?Set MODEL_DIR to the HF model directory}"
: "${DATA_DIR:?Set DATA_DIR to the prompt-data directory or file}"
: "${SAVE_DIR:?Set SAVE_DIR to a persistent checkpoint root}"
: "${MEGATRON:?Set MEGATRON to the compatible Megatron-LM checkout}"
# Use the unified image's installed Bridge, Python dependencies and HybridEP.
# Do not prepend the pre-refactor /opt/qwen38-flash-next runtime.
export MOE_BACKEND="${MOE_BACKEND:-hybridep}"
case "${MOE_BACKEND}" in
    hybridep) : "${RELAX_HYBRID_EP_COMMIT:?Use the unified Blackwell image with HybridEP installed}" ;;
    mok) ;;
    *) echo "MOE_BACKEND must be hybridep or mok" >&2; exit 2 ;;
esac
# CP1/TP2: padding to the full packed-token budget gives every EP rank the same
# SP-local token count. MoK requires at least 512 local tokens, aligned to 256.
MAX_TOKENS_PER_GPU="${MAX_TOKENS_PER_GPU:-8192}"
if [[ ! "${MAX_TOKENS_PER_GPU}" =~ ^[1-9][0-9]*$ ]] \
    || (( MAX_TOKENS_PER_GPU < 1024 || MAX_TOKENS_PER_GPU % 512 != 0 )); then
    echo "MAX_TOKENS_PER_GPU must be at least 1024 and divisible by 512" >&2
    exit 2
fi
DATA_PAD_SIZE_MULTIPLIER="${DATA_PAD_SIZE_MULTIPLIER:-$((MAX_TOKENS_PER_GPU / 2))}"
if [[ "${DATA_PAD_SIZE_MULTIPLIER}" != "$((MAX_TOKENS_PER_GPU / 2))" ]]; then
    echo "DATA_PAD_SIZE_MULTIPLIER must equal MAX_TOKENS_PER_GPU / 2 for this A/B recipe" >&2
    exit 2
fi
export RAY_NO_WAIT="${RAY_NO_WAIT-}"
export RELAX_SKIP_CLUSTER_CLEANUP="${RELAX_SKIP_CLUSTER_CLEANUP:-1}"
export EXP_NAME="${EXP_NAME:-qwen38-qsa-mm-8xb300-${MOE_BACKEND}}" # gitleaks:allow -- stable experiment/checkpoint identity
export PROJECT_NAME="${PROJECT_NAME:-Relax/sft/qwen38-qsa-8xgpu}"
# Preserve this recipe's setting when ray-job.sh resets the generic variable.
export QWEN38_CUDA_DEVICE_MAX_CONNECTIONS="${QWEN38_CUDA_DEVICE_MAX_CONNECTIONS:-1}"
now=$(date "+%Y%m%d-%H%M%S")
export SUBMISSION_ID="${SUBMISSION_ID:-${EXP_NAME}-${now}-$$}"

# Use an exclusive Ray cluster; the operator must prevent concurrent training.
if [[ "${DRY_RUN:-0}" != 1 && -z "${RELAX_ENTRYPOINT_MODE:-}" ]]; then
    : "${RAY_JOBS_ADDRESS:?Set RAY_JOBS_ADDRESS to the Ray Jobs HTTP endpoint}"
    exec bash "${RELAX}/scripts/entrypoint/ray-job.sh" "${BASH_SOURCE[0]}" "$@"
fi

# Explicit driver/actor settings override entrypoint defaults (notably connections=1).
ENV_ARGS=(
    "RELAX_SKIP_NONFINITE_GRAD=${RELAX_SKIP_NONFINITE_GRAD:-0}"
    MCORE_QSA_SPARSE_BACKEND=id_sparse
    TORCHINDUCTOR_COMPILE_THREADS=1
    QWEN38_PLE_CPU_OFFLOAD=1
    "RELAX_HYBRID_EP_COMMIT=${RELAX_HYBRID_EP_COMMIT:-}"
    "LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}"
    "CUDA_HOME=${CUDA_HOME:-}"
    RELAX_PROPAGATE_ENV_VARS=QWEN38_WORKSPACE,SAVE_DIR,RELAX_PROFILE_DIR,TENSORBOARD_DIR,CLEARML_CACHE_DIR,CLEARML_OFFLINE_MODE,PYTHONDONTWRITEBYTECODE,RELAX_HYBRID_EP_COMMIT,LD_LIBRARY_PATH,CUDA_HOME,NVLINK_DOMAIN_SIZE,USE_MNNVL,CUDA_DEVICE_MAX_CONNECTIONS,QWEN38_RUN_DIR,RELAX_SKIP_NONFINITE_GRAD
    NCCL_NVLS_ENABLE=0
    "PYTHONPATH=${RELAX}:${MEGATRON}${PYTHONPATH:+:${PYTHONPATH}}"
    "MEGATRON=${MEGATRON}"
    "RELAX=${RELAX}"
    PYTHONNOUSERSITE=1
    "CUDA_DEVICE_MAX_CONNECTIONS=${QWEN38_CUDA_DEVICE_MAX_CONNECTIONS}"
    "NVTE_USE_CUTLASS_GROUPED_GEMM=${NVTE_USE_CUTLASS_GROUPED_GEMM:-0}"
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,garbage_collection_threshold:0.8
    TOKENIZERS_PARALLELISM=false
    RELAX_QWEN38_QSA_FUSED_ATTENTION=0
    RELAX_QWEN38_RETAIN_QSA_ATTENTION=0
    RELAX_QWEN38_QSA_FIXED_POOL=0
    RELAX_QWEN38_QSA_SPARSE_BLOCKMASK=1
    "RELAX_QWEN38_RECORD_FUNCTION_TAGS=${RELAX_QWEN38_RECORD_FUNCTION_TAGS:-1}"
    RELAX_QWEN38_QSA_LOCAL_QUERY=1
    RELAX_QWEN38_ROUTER_SORTED_REPLAY=1
    RELAX_QWEN38_QSA_ROUTED_QKV=1
    RELAX_QWEN38_QSA_RECOMPUTE_REPLAY=1
    RELAX_QWEN38_QSA_CPU_LAYOUT=0
    RELAX_QWEN38_PLE_STREAMED=0
    RELAX_QWEN38_GDN_OUTPUT_SAFE_CONFIG=1
    RELAX_QWEN38_FLA_CONV_CACHE_REUSE=1
    RELAX_QWEN38_QSA_COMMUNITY_BACKEND=mixed
    "RELAX_QWEN38_QSA_MIXED_LONG_BACKEND=${RELAX_QWEN38_QSA_MIXED_LONG_BACKEND:-grouped}"
    "RELAX_QWEN38_QSA_MIXED_FLASH_MAX_LEN=${RELAX_QWEN38_QSA_MIXED_FLASH_MAX_LEN:--1}"
    "RELAX_QWEN38_QSA_NV_MAX_PACK_TOKENS=${RELAX_QWEN38_QSA_NV_MAX_PACK_TOKENS:-0}"
    "QWEN38_WORKSPACE=${RELAX}"
    "SAVE_DIR=${SAVE_DIR}"
    RELAX_SFT_RESIDENT_PARAM_OVERLAP=0
    RAY_OVERRIDE_JOB_RUNTIME_ENV=1
    "TENSORBOARD_DIR=${SAVE_DIR}/tensorboard"
    "CLEARML_CACHE_DIR=${SAVE_DIR}/clearml"
    "RELAX_PROFILE_DIR=${SAVE_DIR}/traces"
    PYTHONDONTWRITEBYTECODE=1
    RELAX_PROFILE_RANKS=0
    RELAX_PROFILE_GZIP_LEVEL=1
    FLA_CACHE_RESULTS=1
    TRITON_CACHE_AUTOTUNING=1
    TORCHINDUCTOR_FX_GRAPH_CACHE=1
    TORCHINDUCTOR_AUTOTUNE_LOCAL_CACHE=1
    "RELAX_AUTOTUNE_LOG_DIR=${RELAX_AUTOTUNE_LOG_DIR-${SAVE_DIR}/autotune}"
    RELAX_QWEN38_HC_FUSED_MIX=1
    RELAX_SFT_SELECTIVE_LOGITS=1
    RELAX_QWEN38_QSA_TP_SELECTOR=1
    RELAX_QWEN38_QSA_FUSED_SELECTOR=0
    RELAX_QWEN38_QSA_GROUPED_BACKWARD_WARPS=4
    RELAX_QWEN38_QSA_FLEX_SPARSE_MASK=1
    NVLINK_DOMAIN_SIZE=8
    USE_MNNVL=0
    RELAX_QWEN38_HC_FUSED_COMBINE=1
    RELAX_QWEN38_PLE_FUSED_CONV=1
    RELAX_SFT_FUSED_LINEAR_CE=1
    "QWEN38_RUN_DIR=${SAVE_DIR}"
    RELAX_QWEN38_QSA_MIXED_FLEX_MAX_LEN=0
    RELAX_QWEN38_QSA_GROUPED_BACKWARD_STAGES=3
    RELAX_SFT_MTP_ACCEPTANCE_METRICS=0
    RELAX_QWEN38_QSA_GROUPED_BACKWARD_BLOCK_K=32
    RELAX_QWEN38_QSA_SELECTOR_COMPILE=1
)
unset NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN
export "${ENV_ARGS[@]}"
TRAIN_ENV_JSON=$(python3 - "${ENV_ARGS[@]}" <<'PYENV'
import json
import sys
print(json.dumps(dict(item.split("=", 1) for item in sys.argv[1:])))
PYENV
)
BASE_RUNTIME_ENV_JSON="${RUNTIME_ENV_JSON:-}"
if [[ -z "${BASE_RUNTIME_ENV_JSON}" ]]; then BASE_RUNTIME_ENV_JSON='{}'; fi
RUNTIME_ENV_JSON=$(python3 - "${BASE_RUNTIME_ENV_JSON}" "${TRAIN_ENV_JSON}" <<'PYRUNTIME'
import json
import sys
runtime = json.loads(sys.argv[1])
runtime.setdefault("env_vars", {}).pop("NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN", None)
runtime["env_vars"].update(json.loads(sys.argv[2]))
print(json.dumps(runtime))
PYRUNTIME
)

# All 48 decoder layers, vision and MTP are on a single pipeline stage.
# DP = 8 / TP2 / PP1 / CP1 = 4; ETP1 allows EP8 on this eight-GPU node.
PARALLEL_ARGS=(
    --tensor-model-parallel-size 2
    --pipeline-model-parallel-size 1
    --context-parallel-size 1
    --expert-model-parallel-size 8
    --expert-tensor-parallel-size 1
    --sequence-parallel
    --moe-grouped-gemm
    --colocate
    --no-rope-fusion
    --sft-chunked-logits
    --sft-logits-chunk-size 2048
    --recompute-granularity full
    --recompute-method uniform
    --recompute-num-layers 2
    --no-offload-train
    --cuda-graph-impl none
)

MOE_ARGS=()
if [[ "${MOE_BACKEND}" == hybridep ]]; then
    MOE_ARGS+=(
        --moe-token-dispatcher-type flex
        --moe-flex-dispatcher-backend hybridep
        --moe-flex-dispatcher-num-sms 80
        --moe-hybridep-pad-variable-tokens
        --moe-shared-expert-overlap
    )
else
    MOK_CONFIG="${MOK_BACKEND_CONFIG:-}"
    if [[ -z "${MOK_CONFIG}" ]]; then MOK_CONFIG='{"shared_expert_mode":"native"}'; fi
    MOE_ARGS+=(
        --moe-token-dispatcher-type alltoall
        --moe-megakernel-backend mok
        --moe-megakernel-backend-config "${MOK_CONFIG}"
    )
fi

CKPT_ARGS=(
    --hf-checkpoint "${MODEL_DIR}"
    --ref-load "${MODEL_DIR}"
    --megatron-to-hf-mode bridge
    --num-rollout "${NUM_STEPS:-50}"
    --load "${SAVE_DIR}/${EXP_NAME}"
)
case "${SAVE_CHECKPOINTS:-1}" in
    1)
        CKPT_ARGS+=(
            --save "${SAVE_DIR}/${EXP_NAME}"
            --save-interval "${SAVE_INTERVAL:-50}"
            --max-actor-ckpt-to-keep 2
        )
        ;;
    0) ;;
    *) echo "SAVE_CHECKPOINTS must be 0 or 1" >&2; exit 2 ;;
esac

SFT_ARGS=(
    --loss-type sft
    --prompt-data "${DATA_DIR}"
    --eval-size "${EVAL_SIZE:-512}"
    --input-key messages
    --tool-key tools
    --apply-chat-template
    --apply-chat-template-kwargs '{"add_generation_prompt":false,"preserve_thinking":true}'
    --sft-invalid-multimodal-strategy skip
    --sft-oversize-strategy skip
    --calculate-per-token-loss
    --rollout-max-context-len "${SEQ_LENGTH:-8192}"
    --global-batch-size "${GLOBAL_BATCH_SIZE:-32}"
    --micro-batch-size 1
    --seq-length "${SEQ_LENGTH:-8192}"
    --use-dynamic-batch-size
    --max-tokens-per-gpu "${MAX_TOKENS_PER_GPU:-8192}"
    --data-pad-size-multiplier "${DATA_PAD_SIZE_MULTIPLIER}"
    --per-rank-fetch
    --sft-prefetch-num-workers 16
    --sft-train-data-prefetch
    --freeze-vision-model
    --freeze-vision-projection
    --freeze-params-name-list '(^|\.)per_layer_embedding\.|self_attention\.indexer\.'
    --enable-mtp-training
    --mtp-num-layers 1
    --mtp-loss-scaling-factor 0.1
    --mtp-detach-paths none
    --eval-interval "${EVAL_INTERVAL:-100}"
)
SFT_ARGS+=(
    --multimodal-keys '{"image":"images"}'
    --image-max-token-num 128
    --sft-image-preprocess-on-rank
)

OPTIMIZER_ARGS=(
    --optimizer adam
    --lr 2e-5
    --lr-decay-style cosine
    --lr-decay-iters "${LR_DECAY_ITERS:-2023}"
    --min-lr 2e-6
    --lr-warmup-fraction 0.05
    --lr-warmup-init 0
    --weight-decay 0.1
    --adam-beta1 0.9
    --adam-beta2 0.98
    --clip-grad 1.0
    --use-distributed-optimizer
    --use-precision-aware-optimizer
    --exp-avg-dtype "${ADAM_EXP_AVG_DTYPE:-bf16}"
    --exp-avg-sq-dtype "${ADAM_EXP_AVG_SQ_DTYPE:-bf16}"
    # BF16 moment storage and 16-bit master-parameter remainders reduce optimizer memory.
    # Gradient accumulation remains FP32; both MoE backends use the same settings.
    --moe-aux-loss-coeff 0
)

MISC_ARGS=(
    --seed 1234
    --bf16
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --attention-softmax-in-fp32
    --attention-backend flash
    --accumulate-allreduce-grads-in-fp32
    --train-env-vars "${TRAIN_ENV_JSON}"
    --overlap-grad-reduce
    --overlap-param-gather
)
if [[ "${QWEN38_SFT_HEALTH_CHECK:-1}" == 1 ]]; then
    MISC_ARGS+=(--use-health-check)
fi

METRICS_ARGS=(
    --use-clearml
    --use-metrics-service
    --tb-project-name "${PROJECT_NAME}"
    --tb-experiment-name "${EXP_NAME}-${now}"
)

TRAIN_ARGS=(
    --resource '{"sft":[1,0],"actor":[1,8]}'
    --max-global-restart "${MAX_GLOBAL_RESTART:-0}"
    --actor-num-nodes 1
    --actor-num-gpus-per-node 8
    --sft-max-in-flight-steps 4
    --num-data-storage-units 2
    "${MODEL_ARGS[@]}"
    "${CKPT_ARGS[@]}"
    "${SFT_ARGS[@]}"
    "${PARALLEL_ARGS[@]}"
    "${MOE_ARGS[@]}"
    "${OPTIMIZER_ARGS[@]}"
    "${MISC_ARGS[@]}"
    "${METRICS_ARGS[@]}"
    "$@"
)

# Retain the effective argv and worker environment for old/new comparison.
if [[ "${DRY_RUN:-0}" != 1 ]]; then
    mkdir -p "${SAVE_DIR}/logs"
    python3 - "${SAVE_DIR}/logs/${EXP_NAME}-effective-config.json" "${RUNTIME_ENV_JSON}" "${TRAIN_ARGS[@]}" <<'PYCONFIG'
import json
import sys
from pathlib import Path

runtime = json.loads(sys.argv[2])
# Only record explicitly selected training settings, not inherited credentials.
selected = {key: value for key, value in runtime.get("env_vars", {}).items()
            if key.startswith(("RELAX_QWEN38_", "RELAX_SFT_")) or key in
            ("MCORE_QSA_SPARSE_BACKEND", "CUDA_DEVICE_MAX_CONNECTIONS", "RELAX_HYBRID_EP_COMMIT",
             "NVTE_USE_CUTLASS_GROUPED_GEMM")}
Path(sys.argv[1]).write_text(json.dumps({"argv": sys.argv[3:], "env_vars": selected}, indent=2) + "\n")
PYCONFIG
fi

if [[ "${DRY_RUN:-0}" == 1 ]]; then
    python3 - "${RUNTIME_ENV_JSON}" python3 -m relax.entrypoints.train "${TRAIN_ARGS[@]}" <<'PYDRY'
import json
import sys
print(json.dumps({"runtime_env": json.loads(sys.argv[1]), "argv": sys.argv[2:]}, indent=2))
PYDRY
    exit 0
fi

: "${RAY_JOBS_ADDRESS:?Set RAY_JOBS_ADDRESS to the Ray Jobs HTTP endpoint}"
cd "${RELAX}"
mkdir -p "${SCRIPT_DIR}/log"
ray job submit \
    ${RAY_NO_WAIT:+--no-wait} \
    --address="${RAY_JOBS_ADDRESS}" \
    --submission-id "${SUBMISSION_ID}" \
    ${WORKING_DIR:+--working-dir "${WORKING_DIR}"} \
    --runtime-env-json="${RUNTIME_ENV_JSON}" \
    -- python3 -m relax.entrypoints.train \
    "${TRAIN_ARGS[@]}" \
    2>&1 | tee "${SCRIPT_DIR}/log/${EXP_NAME}-${now}.log"
