#!/bin/bash
# Copyright (c) 2026 Relax Authors. All Rights Reserved.
# Full 48-layer multimodal SFT: 5 x 8 H20, TP4/PP5/DP2/EP4, MTP1, 64K packing.
# Export MODEL_DIR (HF checkpoint), DATA_DIR (JSONL or train.jsonl directory), SAVE_DIR, then:
# bash scripts/entrypoint/spmd-multinode.sh examples/models/qwen3.8-flash-next/scripts/run-qwen38-qsa-sft-40xh20.sh
# Existing Ray head: bash examples/models/qwen3.8-flash-next/scripts/run-qwen38-qsa-sft-40xh20.sh
# Default: 100 updates from HF, without saving/resuming training checkpoints.
# DRY_RUN=1 previews arguments without Ray/GPU setup or output writes.
set -eo pipefail
export RAY_NO_WAIT="${RAY_NO_WAIT-}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
RELAX_ROOT="$(cd -- "${SCRIPT_DIR}/../../../.." && pwd)"
export RELAX="${RELAX_ROOT}"
if [[ "${DRY_RUN:-0}" != 1 && -z "${MEGATRON:-}" ]]; then
    MEGATRON=$(python3 -c 'import importlib.util; from pathlib import Path; print(Path(importlib.util.find_spec("megatron.core").origin).resolve().parents[2])')
fi
export MEGATRON="${MEGATRON:-/root/Megatron-LM}"
: "${MODEL_DIR:?Set MODEL_DIR to the HF model directory}"
: "${DATA_DIR:?Set DATA_DIR to the JSONL file or directory}"
: "${SAVE_DIR:?Set SAVE_DIR to a persistent run output directory}"
if [[ "${DRY_RUN:-0}" != 1 && -z "${RELAX_ENTRYPOINT_MODE:-}" ]]; then
    exec bash "${RELAX}/scripts/entrypoint/ray-job.sh" "${BASH_SOURCE[0]}" "$@"
fi
export PYTHONPATH="${RELAX}:${MEGATRON}${PYTHONPATH:+:${PYTHONPATH}}"
source "${RELAX_ROOT}/scripts/models/qwen38-flash-next.sh"
PROJECT_NAME="${PROJECT_NAME:-Relax/sft/qwen38-qsa}"
EXP_NAME="${EXP_NAME:-qwen38-qsa-40xh20}"
now=$(date "+%Y%m%d-%H%M%S")
export SUBMISSION_ID="${SUBMISSION_ID:-${EXP_NAME}-${now}-$$}"

ENV_ARGS=(
    RELAX_SKIP_NONFINITE_GRAD=0
    # Select supervised head rows by default; Hopper linear CE remains opt-in.
    "RELAX_SFT_SELECTIVE_LOGITS=${RELAX_SFT_SELECTIVE_LOGITS:-1}"
    "RELAX_SFT_FUSED_LINEAR_CE=${RELAX_SFT_FUSED_LINEAR_CE:-0}"
    MCORE_QSA_SPARSE_BACKEND=flex
    TORCHINDUCTOR_COMPILE_THREADS=1
    "TORCHINDUCTOR_CACHE_DIR=${TORCHINDUCTOR_CACHE_DIR:-${TMPDIR:-/tmp}/relax-qwen38/inductor}"
    "TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-${TMPDIR:-/tmp}/relax-qwen38/triton}"
    "CUDA_CACHE_PATH=${CUDA_CACHE_PATH:-${TMPDIR:-/tmp}/relax-qwen38/cuda}"
    QWEN38_PLE_CPU_OFFLOAD=1
    "LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}"
    "PYTHONPATH=${PYTHONPATH}"
    "MEGATRON=${MEGATRON}"
    "RELAX=${RELAX}"
    "SAVE_DIR=${SAVE_DIR}"
    PYTHONNOUSERSITE=1
    PYTHONDONTWRITEBYTECODE=1
    CUDA_DEVICE_MAX_CONNECTIONS=1 CUDA_LAUNCH_BLOCKING=0
    NCCL_P2P_DISABLE=0 NCCL_NVLS_ENABLE=0 NCCL_NET_PLUGIN=
    NVSHMEM_DISABLE_NCCL=1
    NVTE_FLASH_ATTN_V4=0 NVTE_FLASH_ATTN_V3=0 NVTE_FLASH_ATTN_V2=1
    NVTE_USE_CUTLASS_GROUPED_GEMM=0
    NVTE_CUTLASS_GROUPED_GEMM_WARN_FALLBACK=1
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,garbage_collection_threshold:0.8
    TOKENIZERS_PARALLELISM=false
    RAY_OVERRIDE_JOB_RUNTIME_ENV=1
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
import os
import sys
env = dict(item.split("=", 1) for item in sys.argv[1:])
runtime = json.loads(os.environ.get("RUNTIME_ENV_JSON") or "{}")
env["LD_LIBRARY_PATH"] = runtime.get("env_vars", {}).get("LD_LIBRARY_PATH", env["LD_LIBRARY_PATH"])
print(json.dumps(env))
PYENV
)

# Preserve the standard entrypoint's library paths, cache settings and hooks.
export RUNTIME_ENV_JSON
RUNTIME_ENV_JSON=$(python3 - "${TRAIN_ENV_JSON}" <<'PYRUNTIME'
import json
import os
import sys
runtime = json.loads(os.environ.get("RUNTIME_ENV_JSON") or "{}")
env = runtime.setdefault("env_vars", {})
env.update(json.loads(sys.argv[1]))
propagated = [*env.get("RELAX_PROPAGATE_ENV_VARS", "").split(","),
              *os.environ.get("RELAX_PROPAGATE_ENV_VARS", "").split(","), *env]
env["RELAX_PROPAGATE_ENV_VARS"] = ",".join(dict.fromkeys(name for name in propagated if name))
print(json.dumps(runtime))
PYRUNTIME
)

CKPT_ARGS=(
    --hf-checkpoint "${MODEL_DIR}"
    --ref-load "${MODEL_DIR}"
    --megatron-to-hf-mode bridge
    --num-rollout "${NUM_STEPS:-100}"
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
    --rollout-max-context-len "${SEQ_LENGTH:-65536}"
    --global-batch-size "${GBS:-1024}"
    --micro-batch-size 1
    --seq-length "${SEQ_LENGTH:-65536}"
    --use-dynamic-batch-size
    --max-tokens-per-gpu "${MAX_TOKENS_PER_GPU:-65536}"
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

MISC_ARGS=(
    --seed "${SEED:-1234}"
    --rollout-seed "${SEED:-1234}"
    --bf16
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --attention-softmax-in-fp32
    --attention-backend flash
    --accumulate-allreduce-grads-in-fp32
    --use-health-check
    --train-env-vars "${TRAIN_ENV_JSON}"
)

OPTIMIZER_ARGS=(
    --optimizer adam
    --lr "${LR:-2e-5}"
    --lr-decay-style cosine
    --lr-decay-iters "${LR_DECAY_ITERS:-2023}"
    --min-lr "${MIN_LR:-2e-6}"
    --lr-warmup-fraction "${LR_WARMUP_FRACTION:-0.05}"
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

TRACKING_ARGS=(
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
    "${TRACKING_ARGS[@]}"
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
mkdir -p "${SCRIPT_DIR}/log"
RAY_JOB_ADDRESS_ARGS=()
if [[ -n "${RAY_JOBS_ADDRESS:-}" ]]; then
    RAY_JOB_ADDRESS_ARGS=(--address "${RAY_JOBS_ADDRESS}")
fi
ray job submit ${RAY_NO_WAIT:+--no-wait} "${RAY_JOB_ADDRESS_ARGS[@]}" \
    --submission-id "${SUBMISSION_ID}" \
    ${WORKING_DIR:+--working-dir "${WORKING_DIR}"} \
    --runtime-env-json="${RUNTIME_ENV_JSON}" \
    -- python3 -m relax.entrypoints.train "${TRAIN_ARGS[@]}" \
    2>&1 | tee "${SCRIPT_DIR}/log/${EXP_NAME}-${now}.log"
