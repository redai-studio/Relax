#!/bin/bash
# Copyright (c) 2026 Relax Authors. All Rights Reserved.
#
# GLM-5.3 full 78-layer LoRA SFT on 32 GPUs, retaining all 256 experts.
# Requires the complete HF checkpoint and the patched
# dependencies from docker/Dockerfile or Dockerfile.cu13.
# MODEL_DIR, DATA_DIR and SAVE_DIR must be shared persistent roots.
# PROMPT_DATA selects a JSONL of complete OpenAI messages; use near-40k samples for the stress run.
# OpenThoughts3 complete reasoning conversations, packed without truncation.
# Periodic held-out loss/PPL uses the resident Megatron model; final AIME is separate.
# Default: 1000 optimizer steps at 40k context; save resumable adapters every 50 steps.
# Covers MLA projections, dense MLPs, shared experts and all routed experts.
# KV absorption requires LoRA dropout=0; the discrete indexer remains frozen.
# Submit through scripts/entrypoint/ray-job.sh when using an existing cluster.

set -eo pipefail
export RAY_NO_WAIT="${RAY_NO_WAIT-}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
: "${MODEL_DIR:?Set MODEL_DIR to the model root}"
: "${DATA_DIR:?Set DATA_DIR to the dataset root}"
: "${SAVE_DIR:?Set SAVE_DIR to a persistent checkpoint root}"
MODEL_NAME="${MODEL_NAME:-GLM-5.3}"
PROJECT_NAME="${PROJECT_NAME:-Relax/sft/openthoughts3}"
EXP_NAME="${EXP_NAME:-glm5.3-openthoughts3-sft-lora-40k-32xgpu}"
NUM_ROLLOUT="${NUM_ROLLOUT:-1000}"
SAVE_INTERVAL="${SAVE_INTERVAL:-50}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-4}"
EVAL_INTERVAL="${EVAL_INTERVAL:-50}"
now=$(date "+%Y%m%d-%H%M%S")

if [ -z "${RELAX_ENTRYPOINT_MODE:-}" ]; then
    source "${SCRIPT_DIR}/../../entrypoint/local.sh"
fi
# local.sh starts the dashboard on port 8265; preserve an explicit Jobs URL.
if [[ "${RELAX_ENTRYPOINT_MODE}" == "local" ]]; then
    RAY_JOB_ADDRESS="${RAY_JOB_ADDRESS:-http://${MASTER_ADDR}:8265}"
fi
: "${RAY_JOB_ADDRESS:?Set RAY_JOB_ADDRESS to the existing cluster Ray Jobs HTTP URL}"
source "${SCRIPT_DIR}/../../models/glm5.3-744B-A40B.sh"

CKPT_ARGS=(
    --hf-checkpoint "${MODEL_DIR}/${MODEL_NAME}"
    --ref-load "${MODEL_DIR}/${MODEL_NAME}"
    --megatron-to-hf-mode bridge
    --load "${SAVE_DIR}/${EXP_NAME}"
    --save "${SAVE_DIR}/${EXP_NAME}"
    --save-interval "${SAVE_INTERVAL}"
    --max-actor-ckpt-to-keep 2
    --save-lora-only
)

LORA_ARGS=(
    --lora-rank 16
    --lora-alpha 32
    --lora-dropout 0.0
    --lora-scope language
    --lora-merge-mode
    --lora-target-modules linear_q_down_proj linear_q_up_proj linear_kv_down_proj linear_kv_up_proj linear_proj linear_fc1 linear_fc2
)

SFT_ARGS=(
    --loss-type sft
    --prompt-data "${PROMPT_DATA:-${DATA_DIR}/train-40k.jsonl}"
    --input-key messages
    --num-rollout "${NUM_ROLLOUT}"
    --rollout-batch-size "${GLOBAL_BATCH_SIZE}"
    --global-batch-size "${GLOBAL_BATCH_SIZE}"
    --seq-length 40960
    --sft-oversize-strategy skip
    --sft-chunked-logits
    --sft-logits-chunk-size 1024
    # Chunking alone retains all chunks' vocabulary activations until backward.
    --recompute-loss-function
    --cross-entropy-loss-fusion
    --balance-data
)
if [[ -n "${CUSTOM_DATASET_CLASS_PATH:-}" ]]; then
    SFT_ARGS+=(--custom-dataset-class-path "${CUSTOM_DATASET_CLASS_PATH}")
fi

PERF_ARGS=(
    # Avoid allocator fragmentation and leave space for lazy NCCL buffers.
    --selective-offload
    --train-env-vars '{"PYTORCH_CUDA_ALLOC_CONF":"expandable_segments:True"}'
    --empty-unused-memory-level 1
    --tensor-model-parallel-size 4
    --sequence-parallel
    --pipeline-model-parallel-size 2
    # Stage 2 starts at layer 39, a full indexer in the GLM-5.3 HF config.
    --decoder-first-pipeline-num-layers 38
    --decoder-last-pipeline-num-layers 40
    # CP4 keeps complete 40k samples while reducing MoE/LoRA activation peaks.
    # With TP4/PP2 this uses DP1; GBS4 still provides four microbatches per step.
    --context-parallel-size 4
    --expert-model-parallel-size 16
    --expert-tensor-parallel-size 1
    --recompute-granularity full
    --recompute-method uniform
    --recompute-num-layers 1
    --use-dynamic-batch-size
    --max-tokens-per-gpu 10240
    --data-pad-size-multiplier 256
    --log-probs-chunk-size 1024
    --moe-flex-dispatcher-backend deepep
    --moe-token-dispatcher-type flex
    --calculate-per-token-loss
    # The discrete top-k path has no indexer auxiliary training loss.
    --freeze-params-name-list self_attention.wq_b self_attention.wk self_attention.k_norm self_attention.weights_proj
)

OPTIMIZER_ARGS=(
    --optimizer adam
    --lr "${LEARNING_RATE:-2e-5}"
    --lr-decay-style constant
    --weight-decay 0.1
    --adam-beta1 0.9
    --adam-beta2 0.98
    # Conservative first-run memory budget; benchmark on-device Adam after measuring peaks.
    --optimizer-cpu-offload
    --overlap-cpu-optimizer-d2h-h2d
    --use-precision-aware-optimizer
    --no-pin-cpu-grads
    --no-pin-cpu-params
)
if [[ "${OVERRIDE_OPT_PARAM_SCHEDULER:-0}" == "1" ]]; then
    OPTIMIZER_ARGS+=(--override-opt_param-scheduler)
fi

# Fixed, question-disjoint validation data; pure teacher-forced evaluation.
# Validate 40k eval memory in a short run before starting the long experiment.
RESOURCE='{"actor": [1, 32], "sft": [1, 0]}'
EVAL_ARGS=(
    --eval-interval "${EVAL_INTERVAL}"
    --eval-prompt-data openthoughts3-heldout-seed42 "${EVAL_DATA:-${DATA_DIR}/heldout-40k.jsonl}"
    --eval-input-key messages
)

WANDB_ARGS=(
    --use-clearml
    --use-metrics-service
    --tb-project-name "${PROJECT_NAME}"
    --tb-experiment-name "${EXP_NAME}-${now}"
)

MISC_ARGS=(
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --accumulate-allreduce-grads-in-fp32
    --attention-softmax-in-fp32
    --attention-backend flash
    --skip-eval-before-train
    --update-weight-buffer-size 1073741824
)
if [[ -n "${BEFORE_TRAIN_STEP_HOOK:-}" ]]; then
    MISC_ARGS+=(--custom-megatron-before-train-step-hook-path "${BEFORE_TRAIN_STEP_HOOK}")
fi

mkdir -p "${SCRIPT_DIR}/log"
ray job submit ${RAY_NO_WAIT:+--no-wait} \
    --address="${RAY_JOB_ADDRESS}" \
    ${WORKING_DIR:+--working-dir "${WORKING_DIR}"} \
    --runtime-env-json="${RUNTIME_ENV_JSON}" \
    -- python3 -m relax.entrypoints.train \
    --resource "${RESOURCE}" \
    --max-staleness 0 \
    --num-data-storage-units 2 \
    --colocate \
    "${MODEL_ARGS[@]}" \
    "${CKPT_ARGS[@]}" \
    "${LORA_ARGS[@]}" \
    "${SFT_ARGS[@]}" \
    "${PERF_ARGS[@]}" \
    "${OPTIMIZER_ARGS[@]}" \
    "${WANDB_ARGS[@]}" \
    "${EVAL_ARGS[@]}" \
    "${MISC_ARGS[@]}" 2>&1 | tee "${SCRIPT_DIR}/log/${EXP_NAME}-${now}.log"
