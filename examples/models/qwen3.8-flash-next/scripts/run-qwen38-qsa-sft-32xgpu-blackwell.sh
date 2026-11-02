#!/usr/bin/env bash
# Copyright (c) 2026 Relax Authors. All Rights Reserved.
# Qwen3.8: 4 nodes x 8 Blackwell GPUs, TP2/PP4/DP4/CP1/EP8, one MTP layer.
# Run this script directly; it delegates setup to scripts/entrypoint/ray-job.sh.
# DRY_RUN=1 prints JSON without Ray setup, cleanup, network calls or GPU imports.
# MODEL_DIR and DATA_DIR are the exact HF checkpoint and prompt-data paths.
# Multimodal F0 recipe: 50 steps and evaluation every 100 steps.
# Keep SAVE_DIR/EXP_NAME stable to resume; save every 50 steps and retain 2.
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
: "${RELAX_HYBRID_EP_COMMIT:?Use the unified Blackwell image with HybridEP installed}"
export RAY_NO_WAIT="${RAY_NO_WAIT-}"
export RELAX_SKIP_CLUSTER_CLEANUP="${RELAX_SKIP_CLUSTER_CLEANUP:-1}"
export EXP_NAME="${EXP_NAME:-qwen38-qsa-32xb300}" # gitleaks:allow -- stable experiment/checkpoint identity
export PROJECT_NAME="${PROJECT_NAME:-Relax/sft/qwen38-qsa}"
now=$(date "+%Y%m%d-%H%M%S")
export SUBMISSION_ID="${SUBMISSION_ID:-${EXP_NAME}-${now}-$$}"

# This branch must precede the standard entrypoint: that helper cleans old jobs.
if [[ "${DRY_RUN:-0}" != 1 && -z "${RELAX_ENTRYPOINT_MODE:-}" ]]; then
    : "${RAY_JOBS_ADDRESS:?Set RAY_JOBS_ADDRESS to the Ray Jobs HTTP endpoint}"
    # Training shares Serve state even when startup cleanup is disabled.
    running_jobs=$(timeout 20s ray job list --address "${RAY_JOBS_ADDRESS}")
    running_jobs=$(printf '%s\n' "${running_jobs}" | grep RUNNING || true)
    [[ -z "${running_jobs}" ]] || { echo "Other Ray Jobs are RUNNING; refusing concurrent training" >&2; exit 2; }
    exec bash "${RELAX}/scripts/entrypoint/ray-job.sh" "${BASH_SOURCE[0]}" "$@"
fi

# Explicit driver/actor settings override entrypoint defaults (notably connections=1).
ENV_ARGS=(
    "RELAX_SKIP_NONFINITE_GRAD=${RELAX_SKIP_NONFINITE_GRAD:-1}"
    MCORE_QSA_SPARSE_BACKEND=id_sparse
    TORCHINDUCTOR_COMPILE_THREADS=1
    QWEN38_PLE_CPU_OFFLOAD=1
    "RELAX_HYBRID_EP_COMMIT=${RELAX_HYBRID_EP_COMMIT}"
    "LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}"
    RELAX_PROPAGATE_ENV_VARS=QWEN38_WORKSPACE,SAVE_DIR,RELAX_PROFILE_DIR,TENSORBOARD_DIR,CLEARML_CACHE_DIR,CLEARML_OFFLINE_MODE,PYTHONDONTWRITEBYTECODE,RELAX_HYBRID_EP_COMMIT,LD_LIBRARY_PATH,NVLINK_DOMAIN_SIZE,USE_MNNVL,CUDA_DEVICE_MAX_CONNECTIONS,QWEN38_RUN_DIR,RELAX_SKIP_NONFINITE_GRAD
    NCCL_NVLS_ENABLE=0
    "PYTHONPATH=${RELAX}:${MEGATRON}"
    "MEGATRON=${MEGATRON}"
    "RELAX=${RELAX}"
    PYTHONNOUSERSITE=1
    CUDA_DEVICE_MAX_CONNECTIONS=32
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,garbage_collection_threshold:0.8
    TOKENIZERS_PARALLELISM=false
    RELAX_QWEN38_QSA_FUSED_ATTENTION=0
    RELAX_QWEN38_RETAIN_QSA_ATTENTION=0
    RELAX_QWEN38_QSA_FIXED_POOL=0
    RELAX_QWEN38_QSA_SPARSE_BLOCKMASK=1
    RELAX_QWEN38_RECORD_FUNCTION_TAGS=1
    RELAX_QWEN38_QSA_LOCAL_QUERY=1
    RELAX_QWEN38_ROUTER_SORTED_REPLAY=1
    RELAX_QWEN38_QSA_ROUTED_QKV=1
    RELAX_QWEN38_QSA_RECOMPUTE_REPLAY=1
    RELAX_QWEN38_QSA_CPU_LAYOUT=0
    RELAX_QWEN38_PLE_STREAMED=0
    RELAX_QWEN38_GDN_OUTPUT_SAFE_CONFIG=1
    RELAX_QWEN38_FLA_CONV_CACHE_REUSE=1
    RELAX_QWEN38_QSA_COMMUNITY_BACKEND=mixed
    "QWEN38_WORKSPACE=${RELAX}"
    "SAVE_DIR=${SAVE_DIR}"
    RELAX_SFT_RESIDENT_PARAM_OVERLAP=0
    RAY_OVERRIDE_JOB_RUNTIME_ENV=1
    "TENSORBOARD_DIR=${SAVE_DIR}/tensorboard"
    "CLEARML_CACHE_DIR=${SAVE_DIR}/clearml"
    "RELAX_PROFILE_DIR=${SAVE_DIR}/traces"
    PYTHONDONTWRITEBYTECODE=1
    RELAX_PROFILE_RANKS=0,8,16,24
    RELAX_PROFILE_GZIP_LEVEL=1
    FLA_CACHE_RESULTS=1
    TRITON_CACHE_AUTOTUNING=1
    TORCHINDUCTOR_FX_GRAPH_CACHE=1
    TORCHINDUCTOR_AUTOTUNE_LOCAL_CACHE=1
    "RELAX_AUTOTUNE_LOG_DIR=${SAVE_DIR}/autotune"
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

# PP stage layers: 12 / 14 / 14 / 8; DP = 32 / TP2 / PP4 / CP1 = 4.
PARALLEL_ARGS=(
    --tensor-model-parallel-size 2
    --pipeline-model-parallel-size 4
    --context-parallel-size 1
    --expert-model-parallel-size 8
    --expert-tensor-parallel-size 1
    --pipeline-model-parallel-layout 'Et*12|t*14|t*14|t*8mL'
    --sequence-parallel
    --moe-token-dispatcher-type flex
    --moe-flex-dispatcher-backend hybridep
    --moe-grouped-gemm
    --colocate
    --no-rope-fusion
    --sft-chunked-logits
    --sft-logits-chunk-size 2048
    --recompute-granularity full
    --recompute-method uniform
    --recompute-num-layers 2
    --no-offload-train
)

CKPT_ARGS=(
    --hf-checkpoint "${MODEL_DIR}"
    --ref-load "${MODEL_DIR}"
    --megatron-to-hf-mode bridge
    --num-rollout "${NUM_STEPS:-50}"
    --load "${SAVE_DIR}/${EXP_NAME}"
    --save "${SAVE_DIR}/${EXP_NAME}"
    --save-interval "${SAVE_INTERVAL:-50}"
    --max-actor-ckpt-to-keep 2
)

SFT_ARGS=(
    --loss-type sft
    --prompt-data "${DATA_DIR}"
    --eval-size 512
    --input-key messages
    --tool-key tools
    --apply-chat-template
    --apply-chat-template-kwargs '{"add_generation_prompt":false,"preserve_thinking":true}'
    --sft-invalid-multimodal-strategy skip
    --sft-oversize-strategy skip
    --calculate-per-token-loss
    --rollout-max-context-len 131072
    --global-batch-size 512
    --micro-batch-size 1
    --seq-length 131072
    --use-dynamic-batch-size
    --max-tokens-per-gpu 131072
    --data-pad-size-multiplier 16384
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
    --eval-interval 100
    --multimodal-keys '{"image":"images"}'
    --image-max-token-num 128
    --sft-image-preprocess-on-rank
)

OPTIMIZER_ARGS=(
    --optimizer adam
    --lr 2e-5
    --lr-decay-style cosine
    --lr-decay-iters 2023
    --min-lr 2e-6
    --lr-warmup-fraction 0.05
    --lr-warmup-init 0
    --weight-decay 0.1
    --adam-beta1 0.9
    --adam-beta2 0.98
    --clip-grad 1.0
    --use-distributed-optimizer
    --use-precision-aware-optimizer
    --no-store-param-remainders
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
    --use-health-check
    --train-env-vars "${TRAIN_ENV_JSON}"
    --moe-flex-dispatcher-num-sms 80
    # Keep the default balanced packing used by this baseline.
    --moe-hybridep-pad-variable-tokens
    --moe-shared-expert-overlap
    --overlap-grad-reduce
    --overlap-param-gather
)

METRICS_ARGS=(
    --use-clearml
    --use-metrics-service
    --tb-project-name "${PROJECT_NAME}"
    --tb-experiment-name "${EXP_NAME}-${now}"
)

TRAIN_ARGS=(
    --resource '{"sft":[1,0],"actor":[1,32]}'
    --actor-num-nodes 4
    --actor-num-gpus-per-node 8
    --sft-max-in-flight-steps 4
    --num-data-storage-units 2
    "${MODEL_ARGS[@]}"
    "${CKPT_ARGS[@]}"
    "${SFT_ARGS[@]}"
    "${PARALLEL_ARGS[@]}"
    "${OPTIMIZER_ARGS[@]}"
    "${MISC_ARGS[@]}"
    "${METRICS_ARGS[@]}"
    "$@"
)

# Retain the effective argv and worker environment for old/new comparison.
if [[ "${DRY_RUN:-0}" != 1 ]]; then
    mkdir -p "${SAVE_DIR}/logs"
    python3 - "${SAVE_DIR}/logs/effective-config.json" "${RUNTIME_ENV_JSON}" "${TRAIN_ARGS[@]}" <<'PYCONFIG'
import json
import sys
from pathlib import Path

runtime = json.loads(sys.argv[2])
# Only record explicitly selected training settings, not inherited credentials.
selected = {key: value for key, value in runtime.get("env_vars", {}).items()
            if key.startswith(("RELAX_QWEN38_", "RELAX_SFT_")) or key in
            ("MCORE_QSA_SPARSE_BACKEND", "CUDA_DEVICE_MAX_CONNECTIONS", "RELAX_HYBRID_EP_COMMIT")}
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
ray job submit ${RAY_NO_WAIT:+--no-wait} --address="${RAY_JOBS_ADDRESS}" \
    --submission-id "${SUBMISSION_ID}" \
    ${WORKING_DIR:+--working-dir "${WORKING_DIR}"} \
    --runtime-env-json="${RUNTIME_ENV_JSON}" \
    -- python3 -m relax.entrypoints.train "${TRAIN_ARGS[@]}" \
    2>&1 | tee "${SCRIPT_DIR}/log/${EXP_NAME}-${now}.log"
