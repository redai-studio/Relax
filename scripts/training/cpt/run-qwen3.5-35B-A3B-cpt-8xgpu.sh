#!/bin/bash
# Copyright (c) 2026 Relax Authors. All Rights Reserved.
# Qwen3.5-35B-A3B CPT throughput baseline: GBS128, dynamic 32768-token microbatches,
# double TransferQueue buffers, asynchronous CPU prepacking, and shared-expert overlap.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
source "${SCRIPT_DIR}/../../models/qwen35-35B-A3B.sh"

: "${CPT_ROOT:?Set CPT_ROOT to the experiment data and output directory}"
: "${HF_CHECKPOINT:?Set HF_CHECKPOINT to the local Qwen3.5-35B-A3B directory}"
MEGATRON_DIR="${MEGATRON_DIR:-/root/Megatron-LM}"
PROMPT_DATA="${PROMPT_DATA:-${CPT_ROOT}/cpt-gbs128-10step-20260911/train128-repeat4.jsonl}"
TRAIN_STEPS="${TRAIN_STEPS:-10}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-128}"
MAX_TOKENS_PER_GPU="${MAX_TOKENS_PER_GPU:-32768}"
SEQ_LENGTH="${SEQ_LENGTH:-4096}"
LR_DECAY_STEPS="${LR_DECAY_STEPS:-50}"
WARMUP_STEPS="${WARMUP_STEPS:-5}"
RUN_NAME="${RUN_NAME:-qwen35-cpt-perf-32768-$(date +%Y%m%d-%H%M%S)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${CPT_ROOT}/outputs/cpt-perf}"
LOG_DIR="${OUTPUT_ROOT}/${RUN_NAME}"
LOG_FILE="${LOG_DIR}/train.log"

[[ -f "${HF_CHECKPOINT}/config.json" ]] || { echo "Missing model: ${HF_CHECKPOINT}" >&2; exit 1; }
[[ -f "${PROMPT_DATA}" ]] || { echo "Missing dataset: ${PROMPT_DATA}" >&2; exit 1; }
[[ -d "${MEGATRON_DIR}/megatron/core" ]] || { echo "Missing Megatron: ${MEGATRON_DIR}" >&2; exit 1; }
for value in "${TRAIN_STEPS}" "${GLOBAL_BATCH_SIZE}" "${MAX_TOKENS_PER_GPU}" "${SEQ_LENGTH}" "${LR_DECAY_STEPS}"; do
    [[ "${value}" =~ ^[1-9][0-9]*$ ]] || { echo "Batch, token and step values must be positive integers" >&2; exit 1; }
done
[[ "${WARMUP_STEPS}" =~ ^[0-9]+$ ]] || { echo "WARMUP_STEPS must be a non-negative integer" >&2; exit 1; }
[[ "${GLOBAL_BATCH_SIZE}" == 128 ]] || {
    echo "This measured baseline requires GLOBAL_BATCH_SIZE=128; create a separate experiment for other GBS." >&2
    exit 1
}

CPT_MODEL_ARGS=()
for arg in "${MODEL_ARGS[@]}"; do
    [[ "${arg}" == --use-gated-attention ]] || CPT_MODEL_ARGS+=("${arg}")
done

CMD=(
    python3 -m relax.entrypoints.train
    --resource '{"sft": [1, 0], "actor": [1, 8]}'
    "${CPT_MODEL_ARGS[@]}"
    --hf-checkpoint "${HF_CHECKPOINT}" --megatron-to-hf-mode bridge
    --loss-type sft --sft-training-mode cpt --sft-cpt-template qwen3_5
    --tool-key "" --metadata-key ""
    --prompt-data "${PROMPT_DATA}" --input-key auto --sft-oversize-strategy truncate_right
    --num-rollout "${TRAIN_STEPS}" --seed 42
    --rollout-batch-size "${GLOBAL_BATCH_SIZE}" --global-batch-size "${GLOBAL_BATCH_SIZE}"
    --n-samples-per-prompt 1 --num-steps-per-rollout 1 --micro-batch-size 1
    --use-dynamic-batch-size --max-tokens-per-gpu "${MAX_TOKENS_PER_GPU}" --balance-data
    --seq-length "${SEQ_LENGTH}" --qkv-format thd --data-pad-size-multiplier 1
    --sft-max-in-flight-steps 2 --sft-async-prepack --per-rank-fetch
    --num-data-storage-units 1
    --bf16 --tensor-model-parallel-size 4 --pipeline-model-parallel-size 1
    --context-parallel-size 1 --expert-model-parallel-size 8 --expert-tensor-parallel-size 1
    --sequence-parallel --calculate-per-token-loss
    --mtp-num-layers 0 --mtp-loss-scaling-factor 0 --moe-aux-loss-coeff 0
    --moe-router-dtype fp32 --moe-token-dispatcher-type alltoall --moe-shared-expert-overlap
    --recompute-granularity full --recompute-method uniform --recompute-num-layers 1
    --attention-backend flash --no-rope-fusion --cross-entropy-loss-fusion
    --attention-dropout 0 --hidden-dropout 0 --rollout-temperature 1
    --optimizer adam --use-distributed-optimizer --accumulate-allreduce-grads-in-fp32
    --lr 1e-5 --min-lr 1e-6 --lr-decay-style cosine
    --lr-decay-iters "${LR_DECAY_STEPS}" --lr-warmup-iters "${WARMUP_STEPS}"
    --weight-decay 0.1 --adam-beta1 0.9 --adam-beta2 0.98 --adam-eps 1e-8 --clip-grad 1
    --attention-softmax-in-fp32 --freeze-vision-model --freeze-vision-projection
)

mkdir -p "${LOG_DIR}"
printf 'Log: %s\n' "${LOG_FILE}"
printf '%q ' "${CMD[@]}"; printf '\n'
[[ "${DRY_RUN:-0}" == 1 ]] && exit 0

[[ "$(python3 -c 'import torch; print(torch.cuda.device_count())')" == 8 ]] || {
    echo "Exactly 8 visible GPUs are required." >&2; exit 1;
}
if nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | grep -q '[0-9]'; then
    echo "GPU compute processes already exist. Stop the occupation program or other training first." >&2
    exit 1
fi

export RELAX="${REPO_ROOT}"
export MEGATRON="${MEGATRON_DIR}"
export PYTHONPATH="${REPO_ROOT}:${MEGATRON_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export CUDA_DEVICE_MAX_CONNECTIONS=1
export TOKENIZERS_PARALLELISM=true
export TQ_PRE_ALLOC_SAMPLE_NUM="${GLOBAL_BATCH_SIZE}"
export TQ_ZERO_COPY_SERIALIZATION=true
export RELAX_SFT_TQ_SHARDS=1

started_ray=0
if ! ray status >/dev/null 2>&1; then
    ray start --head --num-gpus=8 --dashboard-host=0.0.0.0 \
        --dashboard-port=8265 --disable-usage-stats >"${LOG_DIR}/ray-start.log" 2>&1
    started_ray=1
fi
cleanup_ray() {
    if [[ "${started_ray}" == 1 ]]; then
        ray stop --force >/dev/null 2>&1 || true
    fi
}
trap cleanup_ray EXIT
export RAY_ADDRESS="${RAY_ADDRESS:-auto}"

"${CMD[@]}" 2>&1 | tee "${LOG_FILE}"
