#!/usr/bin/env bash
# Copyright (c) 2026 Relax Authors. All Rights Reserved.
# Qwen3.8 full-48L multimodal SFT: 5 x 8 H20, TP4/PP5/DP2/EP4, MTP1.
# Layout: 8/11/11/12/6+MTP; 64K packing, GBS1024, frozen CPU PLE.
# Fresh HF initialization; this recipe does not save checkpoints.
# Usage: MODEL_DIR=... DATA_DIR=... SAVE_DIR=... bash <this-script>
# DATA_DIR is the exact prompt file or a directory containing train.jsonl.
# DRY_RUN=1 prints the command without network, GPU imports or output writes.
set -eo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../launch.sh"
# RELAX_DIR selects the complete checkout, including this recipe and its model config.
TARGET_SCRIPT="${RELAX_DIR}/examples/models/qwen3.8-flash-next/scripts/run-qwen38-qsa-sft-40xh20.sh"
if [[ ! "${BASH_SOURCE[0]}" -ef "${TARGET_SCRIPT}" ]]; then
    exec bash "${TARGET_SCRIPT}" "$@"
fi
RELAX_ROOT="${RELAX_DIR}"
export RELAX="${RELAX_ROOT}"
source "${RELAX_ROOT}/scripts/models/qwen38-flash-next.sh"
for ((i=0; i<${#MODEL_ARGS[@]}; i++)); do
    if [[ "${MODEL_ARGS[i]}" == --moe-aux-loss-coeff ]]; then MODEL_ARGS[i+1]=0; fi
done
: "${MODEL_DIR:?Set MODEL_DIR to the HF model directory}"
: "${DATA_DIR:?Set DATA_DIR to the prompt-data directory or file}"
: "${SAVE_DIR:?Set SAVE_DIR to a persistent run output directory}"
export MEGATRON="${MEGATRON:-/root/Megatron-LM}"
export RAY_JOBS_ADDRESS="${RAY_JOBS_ADDRESS:-http://127.0.0.1:8265}"
export RAY_NO_WAIT="${RAY_NO_WAIT-}"
export RELAX_SKIP_CLUSTER_CLEANUP="${RELAX_SKIP_CLUSTER_CLEANUP:-1}"
now=$(date "+%Y%m%d-%H%M%S")
export SUBMISSION_ID="${SUBMISSION_ID:-${EXP_NAME}-${now}-$$}"

# This branch must precede the standard entrypoint: that helper cleans old jobs.
if [[ "${DRY_RUN:-0}" != 1 && -z "${RELAX_ENTRYPOINT_MODE:-}" ]]; then
    # Admission observes active training jobs without changing other jobs or groups.
    python3 - "${RAY_JOBS_ADDRESS}" <<'PYADMIT'
import sys
from ray.job_submission import JobSubmissionClient
active = [job.submission_id for job in JobSubmissionClient(sys.argv[1]).list_jobs()
          if str(job.status) in ("PENDING", "RUNNING") and "relax.entrypoints.train" in job.entrypoint]
if active:
    raise SystemExit(f"Active Relax training jobs: {active}")
PYADMIT
    exec bash "${RELAX}/scripts/entrypoint/ray-job.sh" "${BASH_SOURCE[0]}" "$@"
fi

# Preserve the library paths supplied by the standard entrypoint for both the
# driver and training actors. The entrypoint may only set them in runtime JSON.
BASE_RUNTIME_ENV_JSON="${RUNTIME_ENV_JSON:-}"
if [[ -z "${BASE_RUNTIME_ENV_JSON}" ]]; then BASE_RUNTIME_ENV_JSON='{}'; fi
ENTRYPOINT_LD_LIBRARY_PATH=$(python3 - "${BASE_RUNTIME_ENV_JSON}" "${LD_LIBRARY_PATH:-}" <<'PYLIBS'
import json
import sys
print(json.loads(sys.argv[1]).get("env_vars", {}).get("LD_LIBRARY_PATH", sys.argv[2]))
PYLIBS
)

# Explicit driver/actor settings override entrypoint defaults (notably connections=1).
ENV_ARGS=(
    RELAX_SKIP_NONFINITE_GRAD=0
    MCORE_QSA_SPARSE_BACKEND=flex
    TORCHINDUCTOR_COMPILE_THREADS=1
    "TORCHINDUCTOR_CACHE_DIR=${TORCHINDUCTOR_CACHE_DIR:-${TMPDIR:-/tmp}/relax-qwen38/inductor}"
    "TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-${TMPDIR:-/tmp}/relax-qwen38/triton}"
    "CUDA_CACHE_PATH=${CUDA_CACHE_PATH:-${TMPDIR:-/tmp}/relax-qwen38/cuda}"
    QWEN38_PLE_CPU_OFFLOAD=1
    "LD_LIBRARY_PATH=${ENTRYPOINT_LD_LIBRARY_PATH}"
    RELAX_PROPAGATE_ENV_VARS=MEGATRON,SAVE_DIR,LD_LIBRARY_PATH
    "PYTHONPATH=${RELAX}:${MEGATRON}"
    "MEGATRON=${MEGATRON}"
    "RELAX=${RELAX}"
    "SAVE_DIR=${SAVE_DIR}"
    PYTHONNOUSERSITE=1
    PYTHONDONTWRITEBYTECODE=1
    CUDA_DEVICE_MAX_CONNECTIONS=1
    CUDA_LAUNCH_BLOCKING=0
    NCCL_P2P_DISABLE=0
    NCCL_NVLS_ENABLE=0
    NCCL_NET_PLUGIN=
    NVSHMEM_DISABLE_NCCL=1
    NVTE_FLASH_ATTN_V4=0
    NVTE_FLASH_ATTN_V3=0
    NVTE_FLASH_ATTN_V2=1
    NVTE_USE_CUTLASS_GROUPED_GEMM=0
    NVTE_CUTLASS_GROUPED_GEMM_WARN_FALLBACK=1
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,garbage_collection_threshold:0.8
    TOKENIZERS_PARALLELISM=false
    RAY_OVERRIDE_JOB_RUNTIME_ENV=1
    OMP_NUM_THREADS=24
    MKL_NUM_THREADS=24
    OPENBLAS_NUM_THREADS=24
    SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=32
    SGLANG_HEALTH_CHECK_TIMEOUT=180
    INDEXER_ROPE_NEOX_STYLE=0
    RELAX_QWEN38_QSA_FUSED_ATTENTION=triton_v2
    RELAX_QWEN38_QSA_FUSED_SELECTOR=1
    RELAX_QWEN38_QSA_SPARSE_BLOCKMASK=1
    RELAX_QWEN38_QSA_LOCAL_QUERY=1
    RELAX_QWEN38_QSA_ROUTED_QKV=1
    RELAX_QWEN38_QSA_RECOMPUTE_REPLAY=1
    RELAX_QWEN38_QSA_CPU_LAYOUT=0
    RELAX_QWEN38_PLE_STREAMED=0
    RELAX_QWEN38_FLA_CONV_CACHE_REUSE=1
    RELAX_QWEN38_QSA_COMMUNITY_BACKEND=
)
export "${ENV_ARGS[@]}"
TRAIN_ENV_JSON=$(python3 - "${ENV_ARGS[@]}" <<'PYENV'
import json
import sys
print(json.dumps(dict(item.split("=", 1) for item in sys.argv[1:])))
PYENV
)
RUNTIME_ENV_JSON=$(python3 - "${BASE_RUNTIME_ENV_JSON}" "${TRAIN_ENV_JSON}" <<'PYRUNTIME'
import json
import sys
runtime = json.loads(sys.argv[1])
runtime.setdefault("env_vars", {})
runtime["env_vars"].update(json.loads(sys.argv[2]))
print(json.dumps(runtime))
PYRUNTIME
)

# PP stage layers: 8 / 11 / 11 / 12 / 6+MTP; dense DP = 40 / TP4 / PP5 = 2.
PARALLEL_ARGS=(
    --tensor-model-parallel-size 4
    --pipeline-model-parallel-size 5
    --context-parallel-size 1
    --expert-model-parallel-size 4
    --expert-tensor-parallel-size 1
    --pipeline-model-parallel-layout 'Et*8|t*11|t*11|t*12|t*6mL'
    --sequence-parallel
    --moe-token-dispatcher-type flex
    --moe-flex-dispatcher-backend deepep
    --moe-grouped-gemm
    --colocate
    --no-rope-fusion
    --sft-chunked-logits
    --sft-logits-chunk-size 2048
    --recompute-granularity full
    --recompute-method uniform
    --recompute-num-layers 1
    --no-offload-train
)

CKPT_ARGS=(
    --hf-checkpoint "${MODEL_DIR}"
    --ref-load "${MODEL_DIR}"
    --megatron-to-hf-mode bridge
    --num-rollout "${NUM_STEPS}"
)

PROMPT_DATA="${DATA_DIR}"
if [[ "${DATA_DIR}" != *.jsonl ]]; then PROMPT_DATA="${DATA_DIR}/train.jsonl"; fi
SFT_ARGS=(
    --loss-type sft
    --prompt-data "${PROMPT_DATA}"
    --input-key messages
    --tool-key tools
    --apply-chat-template
    --apply-chat-template-kwargs '{"add_generation_prompt":false,"preserve_thinking":true}'
    --sft-invalid-multimodal-strategy error
    --sft-oversize-strategy skip
    --calculate-per-token-loss
    --rollout-max-context-len "${SEQ_LENGTH}"
    --global-batch-size "${GBS}"
    --micro-batch-size 1
    --seq-length "${SEQ_LENGTH}"
    --use-dynamic-batch-size
    --max-tokens-per-gpu "${MAX_TOKENS_PER_GPU}"
    --data-pad-size-multiplier 4096
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
    --multimodal-keys '{"image":"images"}'
    --image-max-token-num 16384
    --sft-image-preprocess-on-rank
)

OPTIMIZER_ARGS=(
    --optimizer adam
    --lr "${LR}"
    --lr-decay-style cosine
    --lr-decay-iters "${LR_DECAY_ITERS}"
    --min-lr "${MIN_LR}"
    --lr-warmup-fraction "${LR_WARMUP_FRACTION}"
    --lr-warmup-init 0
    --weight-decay 0.1
    --adam-beta1 0.9
    --adam-beta2 0.98
    --clip-grad 1.0
    --use-distributed-optimizer
    --use-precision-aware-optimizer
    --no-store-param-remainders
)

MISC_ARGS=(
    --seed "${SEED}"
    --rollout-seed "${SEED}"
    --bf16
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --attention-softmax-in-fp32
    --attention-backend flash
    --accumulate-allreduce-grads-in-fp32
    --use-health-check
    --train-env-vars "${TRAIN_ENV_JSON}"
)

METRICS_ARGS=(
    --use-clearml
    --use-metrics-service
    --tb-project-name "${PROJECT_NAME}"
    --tb-experiment-name "${EXP_NAME}-${now}"
)

TRAIN_ARGS=(
    --resource '{"sft":[1,0],"actor":[1,40]}'
    --actor-num-nodes 5
    --actor-num-gpus-per-node 8
    --sft-max-in-flight-steps 2
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
ray job submit ${RAY_NO_WAIT:+--no-wait} --address="${RAY_JOBS_ADDRESS}" \
    --submission-id "${SUBMISSION_ID}" \
    ${WORKING_DIR:+--working-dir "${WORKING_DIR}"} \
    --runtime-env-json="${RUNTIME_ENV_JSON}" \
    -- python3 -m relax.entrypoints.train "${TRAIN_ARGS[@]}" \
    2>&1 | tee "${SAVE_DIR}/logs/${SUBMISSION_ID}.submit.log"
