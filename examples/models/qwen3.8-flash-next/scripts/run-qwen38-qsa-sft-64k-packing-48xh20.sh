#!/bin/bash
# Copyright (c) 2026 Relax Authors. All Rights Reserved.
# Text SFT: 6 x 8 H20, TP2/PP3/VPP4/DP8/EP8/CP1, 64K packing.
# Configured context: 256K; keep oversized samples; no MTP.
# Validated sample lengths reached 113,569 tokens; 256K samples are not yet validated.
# Existing 48-GPU Ray head; MODEL_DIR, DATA_DIR and SAVE_DIR are required.
# Default: one epoch, with resumable checkpoints under SAVE_DIR/EXP_NAME.
# NUM_STEPS overrides NUM_EPOCH; SAVE_INTERVAL=epoch rounds up to cover every epoch.
# Integer intervals retain native full-batch counting. Eval flags can be passed as script arguments.
# Short benchmark: NUM_STEPS=10 SAVE_CHECKPOINT=0 LR_DECAY_ITERS=32.
# DRY_RUN=1 prints the effective argv/runtime environment without cluster access.
set -eo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
RELAX_ROOT="$(cd -- "${SCRIPT_DIR}/../../../.." && pwd)"
export RELAX="${RELAX_ROOT}"
: "${MODEL_DIR:?Set MODEL_DIR to the HF model directory}"
: "${DATA_DIR:?Set DATA_DIR to the JSONL file or directory}"
: "${SAVE_DIR:?Set SAVE_DIR to a persistent output and checkpoint directory}"
export MEGATRON="${MEGATRON:-/root/Megatron-LM}"
export RELAX_SKIP_CLUSTER_CLEANUP=1
if [[ "${DRY_RUN:-0}" != 1 && -z "${RELAX_ENTRYPOINT_MODE:-}" ]]; then
    exec bash "${RELAX}/scripts/entrypoint/ray-job.sh" "${BASH_SOURCE[0]}" "$@"
fi
export PYTHONPATH="${RELAX}:${MEGATRON}${RUNTIME_PYTHON:+:${RUNTIME_PYTHON}}${PYTHONPATH:+:${PYTHONPATH}}"
EXP_NAME="${EXP_NAME:-qwen38-sft-48xh20}"
now=$(date "+%Y%m%d-%H%M%S")
export SUBMISSION_ID="${SUBMISSION_ID:-${EXP_NAME}-${now}-$$}"
export RAY_NO_WAIT="${RAY_NO_WAIT-}"
PROMPT_DATA="${DATA_DIR}"
if [[ "${DATA_DIR}" != *.jsonl ]]; then PROMPT_DATA="${DATA_DIR}/train.jsonl"; fi
RUN_ARGS=()
if [[ -n "${NUM_STEPS:-}" ]]; then
    RUN_ARGS+=(--num-rollout "${NUM_STEPS}")
else
    RUN_ARGS+=(--num-epoch "${NUM_EPOCH:-1}")
fi
if [[ -n "${LR_DECAY_ITERS:-}" ]]; then
    RUN_ARGS+=(--lr-decay-iters "${LR_DECAY_ITERS}")
fi
CKPT_ARGS=()
case "${SAVE_CHECKPOINT:-1}" in
    1)
        CKPT_ARGS=(
            --load "${SAVE_DIR}/${EXP_NAME}"
            --save "${SAVE_DIR}/${EXP_NAME}"
            --save-interval "${SAVE_INTERVAL:-100}"
        )
        # An explicitly empty limit keeps all checkpoints; unset retains the recipe default.
        if [[ -n "${MAX_CKPT_TO_KEEP-2}" ]]; then
            CKPT_ARGS+=(--max-actor-ckpt-to-keep "${MAX_CKPT_TO_KEEP-2}")
        fi
        ;;
    0) ;;
    *) echo 'SAVE_CHECKPOINT must be 0 or 1' >&2; exit 2 ;;
esac
source "${RELAX}/scripts/models/qwen38-flash-next.sh"
ENV_ARGS=(
    RELAX_SKIP_NONFINITE_GRAD=0
    RELAX_SFT_SELECTIVE_LOGITS=1
    RELAX_SFT_SELECTIVE_MULTIMODAL_LOGITS=1
    RELAX_SFT_FUSED_LINEAR_CE=1
    MCORE_QSA_SPARSE_BACKEND=id_sparse
    TORCHINDUCTOR_COMPILE_THREADS=1
    "TORCHINDUCTOR_CACHE_DIR=${TORCHINDUCTOR_CACHE_DIR:-${TMPDIR:-/tmp}/relax-qwen38-sft/inductor}"
    "TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-${TMPDIR:-/tmp}/relax-qwen38-sft/triton}"
    "CUDA_CACHE_PATH=${CUDA_CACHE_PATH:-${TMPDIR:-/tmp}/relax-qwen38-sft/cuda}"
    QWEN38_PLE_CPU_OFFLOAD=1
    "LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}"
    "PYTHONPATH=${PYTHONPATH}"
    "MEGATRON=${MEGATRON}"
    "RELAX=${RELAX}"
    "SAVE_DIR=${SAVE_DIR}"
    PYTHONNOUSERSITE=1
    PYTHONDONTWRITEBYTECODE=1
    HF_HUB_OFFLINE=1
    TRANSFORMERS_OFFLINE=1
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
    RELAX_QWEN38_QSA_FUSED_ATTENTION=0
    RELAX_QWEN38_QSA_FUSED_SELECTOR=1
    RELAX_QWEN38_QSA_SPARSE_BLOCKMASK=1
    RELAX_QWEN38_QSA_LOCAL_QUERY=1
    RELAX_QWEN38_QSA_ROUTED_QKV=1
    RELAX_QWEN38_QSA_RECOMPUTE_REPLAY=1
    RELAX_QWEN38_QSA_CPU_LAYOUT=1
    RELAX_QWEN38_PLE_STREAMED=0
    RELAX_QWEN38_FLA_CONV_CACHE_REUSE=1
    RELAX_QWEN38_HC_FUSED_COMBINE=1
    RELAX_QWEN38_HC_FUSED_MIX=1
    RELAX_QWEN38_HC_RECOMPUTE=1
    RELAX_QWEN38_HC_OUTER_BYPASS_MAX_LOCAL_ROWS=0
    RELAX_QWEN38_PLE_FUSED_CONV=1
    RELAX_QWEN38_QSA_COMMUNITY_BACKEND=mixed
    RELAX_QWEN38_QSA_MIXED_PRESERVE_BITS_SELECTOR=1
    RELAX_QWEN38_QSA_MIXED_LONG_BACKEND=grouped
    RELAX_QWEN38_QSA_GROUPED_BACKWARD_BLOCK_K=32
    RELAX_QWEN38_QSA_GROUPED_BACKWARD_WARPS=4
    RELAX_QWEN38_QSA_GROUPED_BACKWARD_STAGES=1
    RELAX_QWEN38_QSA_MIXED_FLASH_MAX_LEN=-1
    RELAX_QWEN38_QSA_MIXED_FLEX_MAX_LEN=0
    RELAX_QWEN38_QSA_FIXED_POOL=0
    RELAX_QWEN38_QSA_TP_SELECTOR=0
    RELAX_QWEN38_QSA_SELECTOR_COMPILE=0
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
export RUNTIME_ENV_JSON
RUNTIME_ENV_JSON=$(python3 - "${TRAIN_ENV_JSON}" <<'PYRUNTIME'
import json
import os
import sys
runtime = json.loads(os.environ.get("RUNTIME_ENV_JSON") or "{}")
env = runtime.setdefault("env_vars", {})
env.update(json.loads(sys.argv[1]))
names = [*env.get("RELAX_PROPAGATE_ENV_VARS", "").split(","), *env]
env["RELAX_PROPAGATE_ENV_VARS"] = ",".join(dict.fromkeys(name for name in names if name))
runtime.setdefault("worker_process_setup_hook", "relax.utils.logging_utils.install_asyncio_noise_filter")
print(json.dumps(runtime))
PYRUNTIME
)
TRAIN_ARGS=(
    --resource '{"sft":[1,0],"actor":[1,48]}'
    --actor-num-nodes 6
    --actor-num-gpus-per-node 8
    --sft-max-in-flight-steps 2
    --num-data-storage-units 2
    "${MODEL_ARGS[@]}"
    --hf-checkpoint "${MODEL_DIR}"
    --ref-load "${MODEL_DIR}"
    --megatron-to-hf-mode bridge
    "${RUN_ARGS[@]}"
    "${CKPT_ARGS[@]}"
    --loss-type sft
    --prompt-data "${PROMPT_DATA}"
    --input-key messages
    --tool-key tools
    --apply-chat-template
    --apply-chat-template-kwargs '{"reasoning_effort":"low"}'
    --sft-mask-reasoning
    --sft-invalid-multimodal-strategy error
    --sft-oversize-strategy keep
    --calculate-per-token-loss
    --global-batch-size 256
    --micro-batch-size 1
    --seq-length 262144
    --use-dynamic-batch-size
    --max-tokens-per-gpu 65536
    --data-pad-size-multiplier 128
    --per-rank-fetch
    --sft-prefetch-num-workers 16
    --sft-prefetch-buffer-size 64
    --sft-prefetch-chunk-size 8
    --sft-train-data-prefetch
    --freeze-vision-model
    --freeze-vision-projection
    --freeze-params-name-list 'self_attention\.indexer\.' 'per_layer_embedding\.ple_embedding\.ngram_embedding\.weight$'
    --tensor-model-parallel-size 2
    --pipeline-model-parallel-size 3
    --context-parallel-size 1
    --expert-model-parallel-size 8
    --expert-tensor-parallel-size 1
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
    --no-offload-train
    --optimizer adam
    --lr 3e-6
    --lr-decay-style cosine
    --min-lr 3e-7
    --lr-warmup-fraction 0.05
    --lr-warmup-init 3e-7
    --adam-eps 1e-8
    --weight-decay 0.1
    --adam-beta1 0.9
    --adam-beta2 0.98
    --clip-grad 1.0
    --use-distributed-optimizer
    --use-precision-aware-optimizer
    --no-store-param-remainders
    --seed 42
    --rollout-seed 42
    --bf16
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --attention-softmax-in-fp32
    --attention-backend flash
    --accumulate-allreduce-grads-in-fp32
    --use-health-check
    --train-env-vars "${TRAIN_ENV_JSON}"
    --use-clearml
    --use-metrics-service
    --tb-project-name "${PROJECT_NAME:-Relax/sft/qwen38}"
    --tb-experiment-name "${EXP_NAME}-${now}"
    --sft-packing-strategy balanced
    --recompute-num-layers 2
    --chunked-optimizer-state-offload
    --optimizer-state-offload-fraction 1.0
    --optimizer-state-offload-chunk-size-mb 512
    --empty-unused-memory-level 1
    --max-global-restart 0
    --num-layers-per-virtual-pipeline-stage 4
    --moe-flex-dispatcher-num-sms 40
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
mkdir -p "${SAVE_DIR}"
RAY_JOB_ADDRESS_ARGS=()
if [[ -n "${RAY_JOBS_ADDRESS:-}" ]]; then RAY_JOB_ADDRESS_ARGS=(--address "${RAY_JOBS_ADDRESS}"); fi
ray job submit ${RAY_NO_WAIT:+--no-wait} "${RAY_JOB_ADDRESS_ARGS[@]}" \
    --submission-id "${SUBMISSION_ID}" \
    --runtime-env-json="${RUNTIME_ENV_JSON}" \
    -- python3 -m relax.entrypoints.train "${TRAIN_ARGS[@]}" \
    2>&1 | tee "${SAVE_DIR}/ray-submit.log"
