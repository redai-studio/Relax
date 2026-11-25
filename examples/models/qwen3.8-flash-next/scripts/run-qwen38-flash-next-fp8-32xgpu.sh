#!/usr/bin/env bash
# Copyright (c) 2026 Relax Authors. All Rights Reserved.
# SM103: train PLE (including its table), vision and MTP; backbone MoE MXFP8.
# Four nodes with a four-stage pipeline; adjust QWEN38_PP_LAYOUT for PLE memory.
# MODEL_DIR/DATA_DIR are exact input paths; SAVE_DIR is a shared checkpoint root.
# PRECISION=bf16 runs the same recipe as a numerical reference.
# PARAM_STORAGE=fp8 keeps backbone expert parameters in MXFP8 with FP32 masters.
set -eo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export RELAX="$(cd -- "${SCRIPT_DIR}/../../../.." && pwd)"
: "${MODEL_DIR:?Set MODEL_DIR to the Qwen3.8 Flash Next HF checkpoint}"
: "${DATA_DIR:?Set DATA_DIR to the SFT JSONL file}"
: "${SAVE_DIR:?Set SAVE_DIR to a shared persistent checkpoint root}"
: "${RAY_JOBS_ADDRESS:?Set RAY_JOBS_ADDRESS to the existing Ray dashboard HTTP address}"
: "${RELAX_HYBRID_EP_COMMIT:?Use the HybridEP revision from the validated cluster image}"
: "${GLOO_SOCKET_IFNAME:?Set GLOO_SOCKET_IFNAME to the verified cross-node network interface}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-${GLOO_SOCKET_IFNAME}}"
export MEGATRON="${MEGATRON:-/root/Megatron-LM}"
export RAY_NO_WAIT="${RAY_NO_WAIT-}"
export RELAX_SKIP_CLUSTER_CLEANUP=1
PRECISION="${PRECISION:-mxfp8}"
PARAM_STORAGE="${PARAM_STORAGE:-bf16}"
PRECISION_LABEL="${PRECISION}"
if [[ "${PARAM_STORAGE}" == fp8 ]]; then PRECISION_LABEL+="-fp8params"; fi
export EXP_NAME="${EXP_NAME:-qwen38-full-sft-${PRECISION_LABEL}-32xgpu}"
export PROJECT_NAME="${PROJECT_NAME:-Relax/sft/qwen38-fp8}"
now=$(date +%Y%m%d-%H%M%S)
export SUBMISSION_ID="${SUBMISSION_ID:-${EXP_NAME}-${now}}"
if [[ "${DRY_RUN:-0}" != 1 && -z "${RELAX_ENTRYPOINT_MODE:-}" ]]; then
    exec bash "${RELAX}/scripts/entrypoint/ray-job.sh" "${BASH_SOURCE[0]}" "$@"
fi
source "${RELAX}/scripts/models/qwen38-flash-next.sh"

# Compile on node-local storage; shared checkpoint filesystems can expose incomplete cache groups.
RELAX_KERNEL_CACHE_LOCAL_DIR="${RELAX_KERNEL_CACHE_LOCAL_DIR:-${MEGATRON}/.relax-kernel-cache/qwen38/${SUBMISSION_ID}}"
ENV_ARGS=(
    "RELAX=${RELAX}"
    "PYTHONPATH=${RELAX}:${MEGATRON}"
    PYTHONDONTWRITEBYTECODE=1
    PYTHONNOUSERSITE=1
    CUDA_DEVICE_MAX_CONNECTIONS=1
    "GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME}"
    "NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME}"
    "NVSHMEM_BOOTSTRAP_UID_SOCK_IFNAME=${NVSHMEM_BOOTSTRAP_UID_SOCK_IFNAME:-${GLOO_SOCKET_IFNAME}}"
    QWEN38_PLE_CPU_OFFLOAD=0
    RELAX_QWEN38_PLE_STREAMED=0
    RELAX_QWEN38_RETAIN_QSA_ATTENTION=0
    RELAX_QWEN38_QSA_COMMUNITY_BACKEND=mixed
    RELAX_QWEN38_QSA_MIXED_LONG_BACKEND=grouped
    RELAX_QWEN38_QSA_MIXED_FLEX_MAX_LEN=0
    RELAX_QWEN38_QSA_FUSED_ATTENTION=0
    RELAX_QWEN38_QSA_FUSED_SELECTOR=0
    MCORE_QSA_SPARSE_BACKEND=id_sparse
    RELAX_QWEN38_GDN_OUTPUT_SAFE_CONFIG=1
    RELAX_SKIP_NONFINITE_GRAD=0
    RELAX_DISABLE_VISION_BRANCH_SYNC=0
    RELAX_SFT_RESIDENT_PARAM_OVERLAP=0
    NVTE_GROUPED_LINEAR_USE_FUSED_GROUPED_GEMM=0
    NVTE_USE_CUTLASS_GROUPED_GEMM=0
    NVLINK_DOMAIN_SIZE=8
    USE_MNNVL=0
    NCCL_NVLS_ENABLE=0
    TOKENIZERS_PARALLELISM=false
    "RELAX_HYBRID_EP_COMMIT=${RELAX_HYBRID_EP_COMMIT}"
    "CLEARML_CACHE_DIR=${SAVE_DIR}/clearml"
    "RELAX_KERNEL_CACHE_LOCAL_DIR=${RELAX_KERNEL_CACHE_LOCAL_DIR}"
    "TORCHINDUCTOR_CACHE_DIR=${RELAX_KERNEL_CACHE_LOCAL_DIR}/inductor"
    "TRITON_CACHE_DIR=${RELAX_KERNEL_CACHE_LOCAL_DIR}/triton"
    "TILELANG_CACHE_DIR=${RELAX_KERNEL_CACHE_LOCAL_DIR}/tilelang"
    "CUDA_CACHE_PATH=${RELAX_KERNEL_CACHE_LOCAL_DIR}/cuda"
    "TORCH_EXTENSIONS_DIR=${RELAX_KERNEL_CACHE_LOCAL_DIR}/extensions"
)
export "${ENV_ARGS[@]}"
TRAIN_ENV_JSON=$(python3 - "${ENV_ARGS[@]}" <<'PY'
import json
import sys
env = dict(value.split("=", 1) for value in sys.argv[1:])
env["CUDA_MODULE_LOADING"] = "EAGER"
print(json.dumps(env))
PY
)
BASE_RUNTIME_ENV_JSON="${RUNTIME_ENV_JSON:-}"
if [[ -z "${BASE_RUNTIME_ENV_JSON}" ]]; then BASE_RUNTIME_ENV_JSON='{}'; fi
RUNTIME_ENV_JSON=$(python3 - "${BASE_RUNTIME_ENV_JSON}" "${TRAIN_ENV_JSON}" <<'PY'
import json
import sys
runtime = json.loads(sys.argv[1])
# Eager CUDA loading is needed before train-actor imports, not by CPU services.
env = {key: value for key, value in json.loads(sys.argv[2]).items() if key != "CUDA_MODULE_LOADING"}
runtime.setdefault("env_vars", {}).update(env)
print(json.dumps(runtime))
PY
)

PRECISION_ARGS=()
PARAM_GATHER_ARGS=()
if [[ "${OVERLAP_PARAM_GATHER:-1}" == 1 ]]; then
    PARAM_GATHER_ARGS=(--overlap-param-gather)
fi
case "${PRECISION}" in
    mxfp8)
        PRECISION_ARGS=(
            --fp8-format e4m3
            --fp8-recipe mxfp8
            --use-precision-aware-optimizer
        )
        ;;
    bf16) ;;
    *) echo "PRECISION must be mxfp8 or bf16" >&2; exit 2 ;;
esac
case "${PARAM_STORAGE}" in
    fp8)
        if [[ "${PRECISION}" != mxfp8 ]]; then
            echo "PARAM_STORAGE=fp8 requires PRECISION=mxfp8" >&2
            exit 2
        fi
        PRECISION_ARGS+=(
            --fp8-param-gather
            --reuse-grad-buf-for-mxfp8-param-ag
        )
        # MCore completes parameter staging/gather after all optimizer groups
        # update, before Relax releases the reused gradient buffers.
        PARAM_GATHER_ARGS=()
        ;;
    bf16) ;;
    *) echo "PARAM_STORAGE must be fp8 or bf16" >&2; exit 2 ;;
esac
PARALLEL_ARGS=(
    --tensor-model-parallel-size 2
    --pipeline-model-parallel-size 4
    --pipeline-model-parallel-layout "${QWEN38_PP_LAYOUT:-Et*12|t*14|t*14|t*8mL}"
    --context-parallel-size 1
    --expert-model-parallel-size 8
    --expert-tensor-parallel-size 1
    --sequence-parallel
    --moe-token-dispatcher-type flex
    --moe-flex-dispatcher-backend hybridep
    --moe-hybridep-pad-variable-tokens
    --moe-grouped-gemm
    --moe-shared-expert-overlap
    --colocate
    --no-offload-train
    --no-rope-fusion
    --recompute-granularity full
    --recompute-method uniform
    --recompute-num-layers 2
)
CKPT_ARGS=(
    --hf-checkpoint "${MODEL_DIR}"
    --megatron-to-hf-mode bridge
    --load "${SAVE_DIR}/${EXP_NAME}"
    --save "${SAVE_DIR}/${EXP_NAME}"
    --save-interval "${SAVE_INTERVAL:-50}"
    --max-actor-ckpt-to-keep 2
    --num-rollout "${NUM_STEPS:-50}"
)
SFT_ARGS=(
    --loss-type sft
    --prompt-data "${DATA_DIR}"
    --input-key messages
    --tool-key tools
    --multimodal-keys '{"image":"images"}'
    --apply-chat-template
    --apply-chat-template-kwargs '{"add_generation_prompt":false,"preserve_thinking":true}'
    --sft-image-preprocess-on-rank
    --image-max-token-num "${IMAGE_MAX_TOKENS:-128}"
    --sft-invalid-multimodal-strategy error
    --sft-oversize-strategy skip
    --calculate-per-token-loss
    --seq-length "${SEQ_LEN:-131072}"
    --rollout-max-context-len "${SEQ_LEN:-131072}"
    --max-tokens-per-gpu "${SEQ_LEN:-131072}"
    --data-pad-size-multiplier 128
    --global-batch-size "${GLOBAL_BATCH_SIZE:-128}"
    --micro-batch-size 1
    --use-dynamic-batch-size
    --per-rank-fetch
    --sft-prefetch-num-workers 4
    --sft-train-data-prefetch
    --sft-chunked-logits
    --sft-logits-chunk-size 1024
    --freeze-params-name-list 'self_attention\.indexer\.'
    --enable-mtp-training
    --mtp-num-layers 1
    --mtp-loss-scaling-factor 0.1
    --mtp-detach-paths none
    --eval-size "${EVAL_SIZE:-8}"
    --eval-interval "${EVAL_INTERVAL:-10}"
)
OPTIMIZER_ARGS=(
    --optimizer adam
    --lr "${LR:-2e-5}"
    --lr-decay-style cosine
    --lr-decay-iters "${LR_DECAY_ITERS:-1000}"
    --min-lr 2e-6
    --lr-warmup-iters 10
    --weight-decay 0.1
    --adam-beta1 0.9
    --adam-beta2 0.98
    --clip-grad 1.0
    --use-distributed-optimizer
    --no-store-param-remainders
    --moe-aux-loss-coeff 0
)
TRAIN_ARGS=(
    --resource '{"sft":[1,0],"actor":[1,32]}'
    --actor-num-nodes 4
    --actor-num-gpus-per-node 8
    --sft-max-in-flight-steps 2
    --num-data-storage-units 2
    "${MODEL_ARGS[@]}"
    "${PARALLEL_ARGS[@]}"
    "${CKPT_ARGS[@]}"
    "${SFT_ARGS[@]}"
    "${OPTIMIZER_ARGS[@]}"
    "${PRECISION_ARGS[@]}"
    --bf16
    --seed 1234
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --attention-softmax-in-fp32
    --attention-backend flash
    --accumulate-allreduce-grads-in-fp32
    --overlap-grad-reduce
    "${PARAM_GATHER_ARGS[@]}"
    --train-env-vars "${TRAIN_ENV_JSON}"
    --use-clearml
    --use-metrics-service
    --tb-project-name "${PROJECT_NAME}"
    --tb-experiment-name "${EXP_NAME}-${now}"
    "$@"
)
if [[ "${DRY_RUN:-0}" == 1 ]]; then
    python3 - "${RUNTIME_ENV_JSON}" "${TRAIN_ARGS[@]}" <<'PY'
import json
import sys
print(json.dumps({"runtime_env": json.loads(sys.argv[1]), "argv": sys.argv[2:]}, indent=2))
PY
    exit 0
fi
cd "${RELAX}"
mkdir -p "${SCRIPT_DIR}/log"
ray job submit \
    ${RAY_NO_WAIT:+--no-wait} \
    --address="${RAY_JOBS_ADDRESS}" \
    --submission-id "${SUBMISSION_ID}" \
    ${WORKING_DIR:+--working-dir "${WORKING_DIR}"} \
    --runtime-env-json="${RUNTIME_ENV_JSON}" \
    -- python3 -m relax.entrypoints.train "${TRAIN_ARGS[@]}" \
    2>&1 | tee "${SCRIPT_DIR}/log/${EXP_NAME}-${now}.log"
