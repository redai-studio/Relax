#!/usr/bin/env bash
# Copyright (c) 2026 Relax Authors. All Rights Reserved.
# Full-length 32K text SFT on 4 x 8 Blackwell GPUs, TP1/PP1/CP1/EP8.
# Export MODEL_DIR, DATA_DIR, SAVE_DIR, MEGATRON and RAY_JOBS_ADDRESS.
# Use the Blackwell Megatron runtime with grouped FlashQLA support.
set -eo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
RELAX_ROOT="$(cd -- "${SCRIPT_DIR}/../../../.." && pwd)"
export RELAX="${RELAX_ROOT}"
source "${RELAX_ROOT}/scripts/models/qwen38-flash-next.sh"
: "${MODEL_DIR:?Set MODEL_DIR to the HF model directory}"
: "${DATA_DIR:?Set DATA_DIR to the prompt-data directory or file}"
: "${SAVE_DIR:?Set SAVE_DIR to a persistent checkpoint root}"
: "${MEGATRON:?Set MEGATRON to the compatible Megatron-LM checkout}"
export RELAX_SKIP_CLUSTER_CLEANUP="${RELAX_SKIP_CLUSTER_CLEANUP:-1}"
export EXP_NAME="${EXP_NAME:-qwen38-32k-mok-muon}" # gitleaks:allow -- stable experiment/checkpoint identity
export PROJECT_NAME="${PROJECT_NAME:-Relax/sft/qwen38-qsa}"
# Preserve this recipe's setting when ray-job.sh resets the generic variable.
export QWEN38_CUDA_DEVICE_MAX_CONNECTIONS="${QWEN38_CUDA_DEVICE_MAX_CONNECTIONS:-32}"
now=$(date "+%Y%m%d-%H%M%S")
export SUBMISSION_ID="${SUBMISSION_ID:-${EXP_NAME}-${now}-$$}"

if [[ "${DRY_RUN:-0}" != 1 ]]; then
    export RAY_API_SERVER_ADDRESS="${RAY_JOBS_ADDRESS:?Set RAY_JOBS_ADDRESS to the Ray Jobs endpoint}"
fi

# Use an exclusive Ray cluster; the operator must prevent concurrent training.
if [[ "${DRY_RUN:-0}" != 1 && -z "${RELAX_ENTRYPOINT_MODE:-}" ]]; then
    exec bash "${RELAX}/scripts/entrypoint/ray-job.sh" "${BASH_SOURCE[0]}" "$@"
fi

# Explicit driver/actor settings override entrypoint defaults (notably connections=1).
export QWEN38_DOSA_INTERVAL_CONFIG=${QWEN38_DOSA_INTERVAL_CONFIG:-'{"boundary_bins":1024}'}
ENV_ARGS=(
    "NVTE_CUTEDSL_FUSED_GROUPED_MLP=${NVTE_CUTEDSL_FUSED_GROUPED_MLP:-0}"
    "RELAX_CP4_FLASH_QLA=${RELAX_CP4_FLASH_QLA:-1}"
    "RELAX_SKIP_NONFINITE_GRAD=${RELAX_SKIP_NONFINITE_GRAD:-0}"
    MCORE_QSA_SPARSE_BACKEND=id_sparse
    TORCHINDUCTOR_COMPILE_THREADS=1
    QWEN38_PLE_CPU_OFFLOAD=1
    "QWEN38_PLE_INPUT_HALO_CP=1"
    "QWEN38_PLE_INPUT_HALO_TRANSPORT=${QWEN38_PLE_INPUT_HALO_TRANSPORT:-adaptive}"
    "LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}"
    RELAX_PROPAGATE_ENV_VARS=QWEN38_WORKSPACE,SAVE_DIR,RELAX_PROFILE_DIR,TENSORBOARD_DIR,CLEARML_CACHE_DIR,CLEARML_OFFLINE_MODE,PYTHONDONTWRITEBYTECODE,LD_LIBRARY_PATH,NVLINK_DOMAIN_SIZE,USE_MNNVL,CUDA_DEVICE_MAX_CONNECTIONS,QWEN38_RUN_DIR,RELAX_SKIP_NONFINITE_GRAD
    NCCL_NVLS_ENABLE=0
    "PYTHONPATH=${RELAX}:${MEGATRON}"
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
    "RELAX_QWEN38_RECORD_FUNCTION_TAGS=${RELAX_QWEN38_RECORD_FUNCTION_TAGS:-0}"
    RELAX_QWEN38_QSA_LOCAL_QUERY=1
    RELAX_QWEN38_ROUTER_SORTED_REPLAY=1
    RELAX_QWEN38_QSA_ROUTED_QKV=1
    RELAX_QWEN38_QSA_RECOMPUTE_REPLAY=1
    RELAX_QWEN38_QSA_CPU_LAYOUT=0
    RELAX_QWEN38_PLE_STREAMED=0
    RELAX_QWEN38_GDN_OUTPUT_SAFE_CONFIG=1
    RELAX_QWEN38_FLA_CONV_CACHE_REUSE=1
    RELAX_QWEN38_QSA_COMMUNITY_BACKEND=mixed
    "RELAX_QWEN38_QSA_MIXED_LONG_BACKEND=${RELAX_QWEN38_QSA_MIXED_LONG_BACKEND:-cute}"
    "RELAX_QWEN38_QSA_MIXED_FLASH_MAX_LEN=${RELAX_QWEN38_QSA_MIXED_FLASH_MAX_LEN:--1}"
    "RELAX_QWEN38_QSA_NV_MAX_PACK_TOKENS=${RELAX_QWEN38_QSA_NV_MAX_PACK_TOKENS:-0}"
    "QWEN38_WORKSPACE=${RELAX}"
    "SAVE_DIR=${SAVE_DIR}"
    RELAX_SFT_RESIDENT_PARAM_OVERLAP=0
    RAY_OVERRIDE_JOB_RUNTIME_ENV=1
    "TENSORBOARD_DIR=${SAVE_DIR}/tensorboard"
    "CLEARML_CACHE_DIR=${SAVE_DIR}/clearml"
    CLEARML_OFFLINE_MODE=0
    "RELAX_PROFILE_DIR=${SAVE_DIR}/traces"
    PYTHONDONTWRITEBYTECODE=1
    RELAX_PROFILE_RANKS=0,8,16,24
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
    "RELAX_LOGPROB_CE_IMPL=native"
    "QWEN38_RUN_DIR=${SAVE_DIR}"
    RELAX_QWEN38_QSA_MIXED_FLEX_MAX_LEN=0
    RELAX_QWEN38_QSA_GROUPED_BACKWARD_STAGES=3
    RELAX_SFT_MTP_ACCEPTANCE_METRICS=0
    RELAX_QWEN38_QSA_GROUPED_BACKWARD_BLOCK_K=32
    RELAX_QWEN38_QSA_SELECTOR_COMPILE=1
    "RELAX_QWEN38_QSA_SELECTOR_BACKEND=${RELAX_QWEN38_QSA_SELECTOR_BACKEND:-dosa}"
    "QWEN38_DOSA_VARIANT=${QWEN38_DOSA_VARIANT:-boundary}"
    "QWEN38_DOSA_INTERVAL_CONFIG=${QWEN38_DOSA_INTERVAL_CONFIG}"
    "RELAX_QWEN38_QSA_CP=${RELAX_QWEN38_QSA_CP:-1}"
    "RELAX_QWEN38_GDN_CP_PREBUILD_ROUTE=${RELAX_QWEN38_GDN_CP_PREBUILD_ROUTE:-0}"
    "RELAX_QWEN38_PLE_CP_FUSED_CONV=${RELAX_QWEN38_PLE_CP_FUSED_CONV:-0}"
    "RELAX_QWEN38_QSA_CP_TP_SELECTOR=${RELAX_QWEN38_QSA_CP_TP_SELECTOR:-0}"
)
export "${ENV_ARGS[@]}"
TRAIN_ENV_JSON=$(python3 - "${RUNTIME_ENV_JSON:-}" "${ENV_ARGS[@]}" <<'PYENV'
import json
import sys
env = dict(item.split("=", 1) for item in sys.argv[2:])
base = json.loads(sys.argv[1] or "{}").get("env_vars", {})
env["LD_LIBRARY_PATH"] = base.get("LD_LIBRARY_PATH", env["LD_LIBRARY_PATH"])
print(json.dumps(env))
PYENV
)
BASE_RUNTIME_ENV_JSON="${RUNTIME_ENV_JSON:-}"
if [[ -z "${BASE_RUNTIME_ENV_JSON}" ]]; then BASE_RUNTIME_ENV_JSON='{}'; fi
RUNTIME_ENV_JSON=$(python3 - "${BASE_RUNTIME_ENV_JSON}" "${TRAIN_ENV_JSON}" <<'PYRUNTIME'
import json
import sys
runtime = json.loads(sys.argv[1])
runtime.setdefault("env_vars", {}).update(json.loads(sys.argv[2]))
print(json.dumps(runtime))
PYRUNTIME
)

PARALLEL_ARGS=(
    --tensor-model-parallel-size 1
    --pipeline-model-parallel-size 1
    --context-parallel-size 1
    --expert-model-parallel-size 8
    --expert-tensor-parallel-size 1
    --pipeline-model-parallel-layout 'Et*48L'
    --linear-cp-mode headwise
    --moe-token-dispatcher-type alltoall
    --cuda-graph-impl none
    --moe-megakernel-backend mok
    --moe-megakernel-backend-config "{\"shared_expert_mode\":\"native\",\"fixed_num_local_tokens\":32768}"
    --moe-grouped-gemm
    --colocate
    --no-rope-fusion
    --sft-chunked-logits
    --sft-logits-chunk-size 2048
    --recompute-granularity full
    --recompute-method uniform
    --recompute-num-layers 2
    --no-offload-train
    --empty-unused-memory-level 0
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
    --custom-dataset-class examples.retool.sft_dataset.AuthorMaskedSFTDataset
    --mtp-num-layers 0
    --input-key messages
    --tool-key tools
    --apply-chat-template
    --apply-chat-template-kwargs '{"add_generation_prompt":false,"preserve_thinking":true}'
    --sft-invalid-multimodal-strategy skip
    --sft-oversize-strategy skip
    --calculate-per-token-loss
    --rollout-max-context-len 32768
    --global-batch-size "${GBS:-128}"
    --rollout-batch-size "${GBS:-128}"
    --n-samples-per-prompt 1
    --micro-batch-size 1
    --seq-length 32768
    --use-dynamic-batch-size
    --max-tokens-per-gpu 32768
    --data-pad-size-multiplier 16384
    --per-rank-fetch
    --sft-prefetch-num-workers 16
    --sft-train-data-prefetch
    --freeze-vision-model
    --freeze-vision-projection
    --freeze-params-name-list '(^|\.)per_layer_embedding\.|self_attention\.indexer\.'
)
OPTIMIZER_ARGS=(
    --optimizer muon
    --muon-momentum 0.95
    --muon-nesterov
    --muon-coefficient-type polar_express
    --muon-num-ns-steps 8
    --muon-scale-mode spectral
    --muon-extra-scale-factor 0.2
    --muon-tp-mode duplicated
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
    --no-use-precision-aware-optimizer
    --moe-aux-loss-coeff 0
)

MISC_ARGS=(
    --seed 1234
    --bf16
    --fp8-format e4m3
    --fp8-recipe mxfp8
    --fp8-param-gather
    --reuse-grad-buf-for-mxfp8-param-ag
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

if [[ "${DRY_RUN:-0}" == 1 ]]; then
    python3 - "${RUNTIME_ENV_JSON}" python3 -m relax.entrypoints.train "${TRAIN_ARGS[@]}" <<'PYDRY'
import json
import sys
print(json.dumps({"runtime_env": json.loads(sys.argv[1]), "argv": sys.argv[2:]}, indent=2))
PYDRY
    exit 0
fi

cd "${RELAX}"
mkdir -p "${SAVE_DIR}/logs"
RAY_SUBMIT_ARGS=(--address="${RAY_JOBS_ADDRESS}")
if [[ "${RAY_NO_WAIT:-0}" == 1 ]]; then RAY_SUBMIT_ARGS+=(--no-wait); fi
ray job submit "${RAY_SUBMIT_ARGS[@]}" \
    --submission-id "${SUBMISSION_ID}" \
    ${WORKING_DIR:+--working-dir "${WORKING_DIR}"} \
    --runtime-env-json="${RUNTIME_ENV_JSON}" \
    -- python3 -m relax.entrypoints.train "${TRAIN_ARGS[@]}" \
    2>&1 | tee "${SAVE_DIR}/logs/${EXP_NAME}-${now}.log"
