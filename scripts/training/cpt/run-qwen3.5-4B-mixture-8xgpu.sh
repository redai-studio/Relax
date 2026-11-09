#!/bin/bash
# Copyright (c) 2026 Relax Authors. All Rights Reserved.
# Submit through scripts/entrypoint/ray-job.sh on an existing Ray cluster.
set -euo pipefail

: "${HF_CHECKPOINT:?Set HF_CHECKPOINT to a Qwen3.5-4B checkpoint}"
: "${PROMPT_DATA:?Set PROMPT_DATA to a CPT dataset or mixture YAML}"
: "${CPT_ROOT:?Set CPT_ROOT to a writable output directory}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../../models/qwen35-4B.sh"
now=$(date +%Y%m%d-%H%M%S)
PROJECT_NAME="${PROJECT_NAME:-Relax/cpt/dataset-loss}"
RUN_NAME="${RUN_NAME:-qwen3.5-4b-cpt-gpu8-${now}}"
TRAIN_STEPS="${TRAIN_STEPS:-20}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-32}"
MAX_TOKENS_PER_GPU="${MAX_TOKENS_PER_GPU:-2048}"
SEQ_LENGTH="${SEQ_LENGTH:-1024}"

# This HF checkpoint shares its embedding and output weights.
CPT_MODEL_ARGS=()
for arg in "${MODEL_ARGS[@]}"; do
    case "${arg}" in
        --untie-embeddings-and-output-weights|--use-gated-attention) ;;
        *) CPT_MODEL_ARGS+=("${arg}") ;;
    esac
done

WANDB_ARGS=(
    # --use-tensorboard
    --use-metrics-service
    --tb-project-name "${PROJECT_NAME}"
    --tb-experiment-name "${RUN_NAME}"
)
if [[ "${ENABLE_CLEARML:-1}" == 1 ]]; then
    WANDB_ARGS+=(--use-clearml)
fi
if [[ -n "${CLEARML_CONFIG_FILE:-}" ]]; then
    [[ -r "${CLEARML_CONFIG_FILE}" ]] || { echo "ClearML config is not readable" >&2; exit 1; }
    # Pass a shared file path; credential values must never enter runtime-env logs.
    RUNTIME_ENV_JSON=$(python3 -c 'import json,os; e=json.loads(os.environ["RUNTIME_ENV_JSON"]); e.setdefault("env_vars",{})["CLEARML_CONFIG_FILE"]=os.environ["CLEARML_CONFIG_FILE"]; print(json.dumps(e))')
fi
CKPT_ARGS=(--hf-checkpoint "${HF_CHECKPOINT}" --ref-load "${HF_CHECKPOINT}" --megatron-to-hf-mode bridge)
if [[ "${SAVE_CHECKPOINT:-1}" == 1 ]]; then
    CKPT_ARGS+=(--save "${CPT_ROOT}/checkpoints/${RUN_NAME}" --save-interval "${SAVE_INTERVAL:-1000}")
fi

mkdir -p "${CPT_ROOT}/logs"
ray job submit --address="${RAY_DASHBOARD_ADDRESS:-http://127.0.0.1:8265}" --no-wait \
    --submission-id "${RUN_NAME}" --runtime-env-json="${RUNTIME_ENV_JSON}" \
    -- python3 -m relax.entrypoints.train \
    --resource '{"sft": [1, 0], "actor": [1, 8]}' \
    "${CPT_MODEL_ARGS[@]}" "${CKPT_ARGS[@]}" "${WANDB_ARGS[@]}" \
    --loss-type sft --sft-training-mode cpt --sft-cpt-template qwen3_5 \
    --prompt-data "${PROMPT_DATA}" --input-key text --tool-key "" --metadata-key "" \
    --sft-oversize-strategy truncate_right --num-rollout "${TRAIN_STEPS}" --seed 42 \
    --rollout-batch-size "${GLOBAL_BATCH_SIZE}" --global-batch-size "${GLOBAL_BATCH_SIZE}" \
    --micro-batch-size 1 --use-dynamic-batch-size --max-tokens-per-gpu "${MAX_TOKENS_PER_GPU}" \
    --seq-length "${SEQ_LENGTH}" --balance-data --qkv-format thd \
    --sft-max-in-flight-steps "${IN_FLIGHT_STEPS:-2}" --sft-async-prepack --per-rank-fetch \
    --sft-prefetch-num-workers "${PREFETCH_WORKERS:-4}" --sft-prefetch-buffer-size "${PREFETCH_BUFFER:-1024}" \
    --tensor-model-parallel-size 1 --pipeline-model-parallel-size 1 --context-parallel-size 1 \
    --expert-model-parallel-size 1 --expert-tensor-parallel-size 1 \
    --num-data-storage-units 1 --max-staleness 0 --calculate-per-token-loss \
    --bf16 --use-distributed-optimizer --accumulate-allreduce-grads-in-fp32 \
    --mtp-num-layers 0 --mtp-loss-scaling-factor 0 --freeze-vision-model --freeze-vision-projection \
    --sft-chunked-logits --sft-logits-chunk-size "${SFT_LOGITS_CHUNK_SIZE:-512}" --recompute-loss-function \
    --no-rope-fusion --attention-backend flash --attention-softmax-in-fp32 \
    --attention-dropout 0 --hidden-dropout 0 --rollout-temperature 1 \
    --optimizer adam --lr 1e-5 --lr-decay-style constant --weight-decay 0.1 --clip-grad 1
