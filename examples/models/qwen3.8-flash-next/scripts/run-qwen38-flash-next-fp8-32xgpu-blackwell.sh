#!/usr/bin/env bash
# Copyright (c) 2026 Relax Authors. All Rights Reserved.
# Text SFT benchmark from q38-final-1123-0820-20261123-013906 (8a1ee4e).
# Historical inputs: BF16 Qwen3.8-Flash-Next and 0820_no_business_identity_x3_relax_v2.
# Both PARAM_STORAGE modes use MXFP8 expert GEMMs. Default bf16 reproduces this benchmark.
# Run with bash <this-script>, or prefix with DRY_RUN=1 to print the Ray and actor config.
# The historical 32-step benchmark starts from HF and omits checkpoint I/O.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export RELAX="$(cd -- "${SCRIPT_DIR}/../../../.." && pwd)"
source "${RELAX}/scripts/models/qwen38-flash-next.sh"
: "${MODEL_DIR:?Set MODEL_DIR to the Qwen3.8-Flash-Next HF checkpoint}"
: "${DATA_DIR:?Set DATA_DIR to the SFT dataset}"
: "${SAVE_DIR:?Set SAVE_DIR to a shared output directory}"
: "${MEGATRON:?Set MEGATRON to the compatible Megatron-LM checkout}"
: "${RELAX_HYBRID_EP_COMMIT:?Use the unified Blackwell image with HybridEP installed}"
export RAY_NO_WAIT="${RAY_NO_WAIT-}"
export RELAX_SKIP_CLUSTER_CLEANUP=1
export EXP_NAME="${EXP_NAME:-qwen38-qsa-mxfp8-32gpu}"
export PROJECT_NAME="${PROJECT_NAME:-Relax/sft/qwen38-qsa}"
export QWEN38_CUDA_DEVICE_MAX_CONNECTIONS="${QWEN38_CUDA_DEVICE_MAX_CONNECTIONS:-32}"
now=$(date +%Y%m%d-%H%M%S)
export SUBMISSION_ID="${SUBMISSION_ID:-${EXP_NAME}-${now}-$$}"

case "${PARAM_STORAGE:-bf16}" in
    bf16)
        PARAM_GATHER_ARGS=(--overlap-param-gather)
        storage_suffix=""
        ;;
    fp8)
        # The MXFP8 parameter gather reuses gradient buffers and must finish synchronously.
        PARAM_GATHER_ARGS=(
            --fp8-param-gather
            --reuse-grad-buf-for-mxfp8-param-ag
        )
        storage_suffix="-fp8params"
        ;;
    *) echo "PARAM_STORAGE must be bf16 or fp8" >&2; exit 2 ;;
esac

if [[ "${DRY_RUN:-0}" != 1 ]]; then
    : "${RAY_JOBS_ADDRESS:?Set RAY_JOBS_ADDRESS to the Ray Jobs HTTP endpoint}"
    if [[ -z "${RELAX_ENTRYPOINT_MODE:-}" ]]; then
        exec bash "${RELAX}/scripts/entrypoint/ray-job.sh" "${BASH_SOURCE[0]}" "$@"
    fi
fi

# The train actors preload cuBLAS 13.3 for Transformer Engine grouped GEMM.
cublas="${RELAX_CUBLAS_133_LIB_DIR:-/opt/relax/cublas13.3/nvidia/cu13/lib}"
if [[ "${DRY_RUN:-0}" != 1 && ( ! -r "${cublas}/libcublasLt.so.13" || ! -r "${cublas}/libcublas.so.13" ) ]]; then
    echo "Missing the cuBLAS 13.3 libraries in ${cublas}" >&2
    exit 2
fi
ENV_ARGS=(
    RELAX_SKIP_NONFINITE_GRAD=0
    MCORE_QSA_SPARSE_BACKEND=id_sparse
    TORCHINDUCTOR_COMPILE_THREADS=1
    QWEN38_PLE_CPU_OFFLOAD=1
    "QWEN38_PLE_INPUT_HALO_CP=${QWEN38_PLE_INPUT_HALO_CP:-0}"
    "QWEN38_PLE_INPUT_HALO_TRANSPORT=${QWEN38_PLE_INPUT_HALO_TRANSPORT:-adaptive}"
    "RELAX_HYBRID_EP_COMMIT=${RELAX_HYBRID_EP_COMMIT}"
    RELAX_PROPAGATE_ENV_VARS=QWEN38_WORKSPACE,SAVE_DIR,RELAX_PROFILE_DIR,TENSORBOARD_DIR,CLEARML_CACHE_DIR,CLEARML_OFFLINE_MODE,PYTHONDONTWRITEBYTECODE,RELAX_HYBRID_EP_COMMIT,LD_LIBRARY_PATH,NVLINK_DOMAIN_SIZE,USE_MNNVL,CUDA_DEVICE_MAX_CONNECTIONS,QWEN38_RUN_DIR,RELAX_SKIP_NONFINITE_GRAD
    NCCL_NVLS_ENABLE=0
    "PYTHONPATH=${RELAX}:${MEGATRON}"
    "MEGATRON=${MEGATRON}"
    "RELAX=${RELAX}"
    PYTHONNOUSERSITE=1
    NVTE_USE_CUTLASS_GROUPED_GEMM=0
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
    RELAX_QWEN38_QSA_MIXED_LONG_BACKEND=cute
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
    "QWEN38_RUN_DIR=${SAVE_DIR}"
    "RELAX_QWEN38_QSA_MIXED_FLEX_MAX_LEN=${RELAX_QWEN38_QSA_MIXED_FLEX_MAX_LEN:-0}"
    RELAX_QWEN38_QSA_GROUPED_BACKWARD_STAGES=3
    RELAX_SFT_MTP_ACCEPTANCE_METRICS=0
    RELAX_QWEN38_QSA_GROUPED_BACKWARD_BLOCK_K=32
    RELAX_QWEN38_QSA_SELECTOR_COMPILE=1
    RELAX_QWEN38_QSA_SELECTOR_BACKEND=radix
    "RELAX_QWEN38_QSA_CP=${RELAX_QWEN38_QSA_CP:-0}"
    "RELAX_QWEN38_GDN_CP_PREBUILD_ROUTE=${RELAX_QWEN38_GDN_CP_PREBUILD_ROUTE:-0}"
    "RELAX_QWEN38_PLE_CP_FUSED_CONV=${RELAX_QWEN38_PLE_CP_FUSED_CONV:-0}"
    "RELAX_QWEN38_QSA_CP_TP_SELECTOR=${RELAX_QWEN38_QSA_CP_TP_SELECTOR:-0}"
    NVTE_GROUPED_LINEAR_USE_FUSED_GROUPED_GEMM=1
    "LD_PRELOAD=${cublas}/libcublasLt.so.13:${cublas}/libcublas.so.13${LD_PRELOAD:+:${LD_PRELOAD}}"
)
for network_var in GLOO_SOCKET_IFNAME NCCL_SOCKET_IFNAME NVSHMEM_BOOTSTRAP_UID_SOCK_IFNAME; do
    if [[ -n "${!network_var:-}" ]]; then ENV_ARGS+=("${network_var}=${!network_var}"); fi
done
unset NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN
if [[ "${DRY_RUN:-0}" != 1 ]]; then export "${ENV_ARGS[@]}"; fi
TRAIN_ENV_JSON=$(python3 - "${RUNTIME_ENV_JSON:-}" "${ENV_ARGS[@]}" <<'PYENV'
import json
import os
import sys

base_env = json.loads(sys.argv[1] or "{}").get("env_vars", {})
train_env = dict(item.split("=", 1) for item in sys.argv[2:])
train_env["LD_LIBRARY_PATH"] = base_env.get("LD_LIBRARY_PATH", os.environ.get("LD_LIBRARY_PATH", ""))
train_env["CUDA_DEVICE_MAX_CONNECTIONS"] = os.environ["QWEN38_CUDA_DEVICE_MAX_CONNECTIONS"]
print(json.dumps(train_env))
PYENV
)
RUNTIME_ENV_JSON=$(python3 - "${RUNTIME_ENV_JSON:-}" "${TRAIN_ENV_JSON}" <<'PYRUNTIME'
import json
import sys

runtime = json.loads(sys.argv[1] or "{}")
env = runtime.setdefault("env_vars", {})
env.pop("NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN", None)
train_env = json.loads(sys.argv[2])
# Keep the entrypoint's driver setting for Megatron TP validation on CPU-only Ray heads.
del train_env["CUDA_DEVICE_MAX_CONNECTIONS"]
env.update(train_env)
env.setdefault("CUDA_DEVICE_MAX_CONNECTIONS", "1")
print(json.dumps(runtime))
PYRUNTIME
)

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
    # MXFP8 expert GEMMs in both parameter storage modes.
    --fp8-format e4m3
    --fp8-recipe mxfp8
)

CKPT_ARGS=(
    --hf-checkpoint "${MODEL_DIR}"
    --megatron-to-hf-mode bridge
    --num-rollout 32
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
)
case "${QWEN38_SFT_MODALITY:-text}" in
    text) ;;
    multimodal)
        SFT_ARGS+=(
            --multimodal-keys '{"image":"images"}'
            --image-max-token-num 128
            --sft-image-preprocess-on-rank
        )
        ;;
    *) echo "QWEN38_SFT_MODALITY must be text or multimodal" >&2; exit 2 ;;
esac
OPTIMIZER_ARGS=(
    --optimizer adam
    --lr 2e-5
    --lr-decay-style cosine
    --lr-decay-iters 2023
    --min-lr 2e-6
    --lr-warmup-fraction 0.05
    --weight-decay 0.1
    --adam-beta1 0.9
    --adam-beta2 0.98
    --clip-grad 1.0
    --use-distributed-optimizer
    --use-precision-aware-optimizer
    --no-store-param-remainders
    --moe-aux-loss-coeff 0
    --lr-warmup-init 0
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
    --moe-flex-dispatcher-num-sms 80
    --moe-hybridep-pad-variable-tokens
    --moe-shared-expert-overlap
    --overlap-grad-reduce
    "${PARAM_GATHER_ARGS[@]}"
    --use-health-check
)

METRICS_ARGS=(
    --use-metrics-service
    --tb-project-name "${PROJECT_NAME}"
    --tb-experiment-name "${EXP_NAME}-${now}"
    --use-clearml
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
    python3 - "${RUNTIME_ENV_JSON}" "${TRAIN_ARGS[@]}" <<'PYDRY'
import json
import sys

print(json.dumps({"runtime_env": json.loads(sys.argv[1]), "argv": sys.argv[2:]}, indent=2))
PYDRY
    exit 0
fi

cd "${RELAX}"
mkdir -p "${SAVE_DIR}/logs"
ray job submit ${RAY_NO_WAIT:+--no-wait} \
    --address="${RAY_JOBS_ADDRESS}" \
    --submission-id "${SUBMISSION_ID}" \
    ${WORKING_DIR:+--working-dir "${WORKING_DIR}"} \
    --runtime-env-json="${RUNTIME_ENV_JSON}" \
    -- python3 -m relax.entrypoints.train "${TRAIN_ARGS[@]}" \
    2>&1 | tee "${SAVE_DIR}/logs/${SUBMISSION_ID}.log"
