#!/bin/bash
# Copyright (c) 2026 Relax Authors. All Rights Reserved.
# Candidate 8-GPU Dense/MoE SFT experiment. Native GPU validation is pending.
# MODE=A existing length balancing, B FLOPs packing, C FLOPs DP, D both.
# Inspect without Ray/CUDA: TASK16_DRY_RUN=1 ... bash <this-script>.
# Submit through scripts/entrypoint/ray-job.sh after setting the actual Ray address.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
TASK16_REPO="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${TASK16_REPO}"
: "${HF_CHECKPOINT:?Set HF_CHECKPOINT to the pinned local model directory}"
: "${WORKLOAD_DIR:?Set WORKLOAD_DIR to prepare_workload output}"
: "${RUN_DIR:?Set a new RUN_DIR for this independent run}"
TASK16_MODEL="${TASK16_MODEL:-dense}"
MODE="${MODE:-A}"
TP_SIZE="${TP_SIZE:-2}"
PP_SIZE="${PP_SIZE:-2}"
CP_SIZE="${CP_SIZE:-1}"
NUM_GPUS="${NUM_GPUS:-8}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-64}"
if (( NUM_GPUS % (TP_SIZE * PP_SIZE * CP_SIZE) != 0 )); then
    echo "NUM_GPUS must be divisible by TP_SIZE*PP_SIZE*CP_SIZE" >&2
    exit 2
fi
if (( GLOBAL_BATCH_SIZE % (NUM_GPUS / TP_SIZE / PP_SIZE / CP_SIZE) != 0 )); then
    echo "GLOBAL_BATCH_SIZE must be divisible by ordinary DP size" >&2
    exit 2
fi
case "${TASK16_MODEL}" in
    dense) source "${TASK16_REPO}/scripts/models/qwen35-27B.sh"; EP_SIZE=1 ;;
    moe) source "${TASK16_REPO}/scripts/models/qwen35-35B-A3B.sh"; EP_SIZE="${EP_SIZE:-2}" ;;
    *) echo "TASK16_MODEL must be dense or moe" >&2; exit 2 ;;
esac
SCHEDULE_ARGS=()
case "${MODE}" in
    A) ;;
    B) SCHEDULE_ARGS+=(--flops-microbatch-packing) ;;
    C) SCHEDULE_ARGS+=(--flops-dp-balancing) ;;
    D) SCHEDULE_ARGS+=(--flops-microbatch-packing --flops-dp-balancing) ;;
    *) echo "MODE must be A, B, C or D" >&2; exit 2 ;;
esac

TRAIN_ARGS=(
    "${MODEL_ARGS[@]}"
    --resource "{\"sft\": [1, 0], \"actor\": [1, ${NUM_GPUS}]}" --num-data-storage-units 1
    --hf-checkpoint "${HF_CHECKPOINT}" --megatron-to-hf-mode bridge
    # Every run starts from the same HF weights with a fresh optimizer; no resume/save/eval side work.
    --loss-type sft --prompt-data "${WORKLOAD_DIR}/train.jsonl"
    --input-key problem --label-key generated_solution
    --num-rollout "${NUM_ROLLOUT:-30}" --rollout-batch-size "${GLOBAL_BATCH_SIZE}"
    --global-batch-size "${GLOBAL_BATCH_SIZE}" --n-samples-per-prompt 1
    --micro-batch-size 1 --use-dynamic-batch-size --max-tokens-per-gpu "${MAX_TOKENS_PER_GPU:-8192}"
    --balance-data --qkv-format thd --sft-oversize-strategy keep
    --seed 1605 --rollout-seed 1605 --max-staleness 0
    --tensor-model-parallel-size "${TP_SIZE}" --pipeline-model-parallel-size "${PP_SIZE}"
    --context-parallel-size "${CP_SIZE}" --expert-model-parallel-size "${EP_SIZE}"
    --expert-tensor-parallel-size 1 --sequence-parallel
    --use-distributed-optimizer --accumulate-allreduce-grads-in-fp32
    --recompute-granularity full --recompute-method uniform --recompute-num-layers 1
    --calculate-per-token-loss --sft-chunked-logits --sft-logits-chunk-size 1024
    --optimizer adam --lr 1e-5 --lr-decay-style constant --weight-decay 0.1
    # Full FP32 Adam states are large for 27B/35B on eight 80GB GPUs.
    # Use the same CPU offload in all modes; validate CPU RAM/bandwidth on the actual host.
    --optimizer-cpu-offload --optimizer-offload-fraction 1.0
    --use-precision-aware-optimizer
    --adam-beta1 0.9 --adam-beta2 0.98 --clip-grad 1.0
    --attention-dropout 0 --hidden-dropout 0 --attention-softmax-in-fp32
    --attention-backend flash --no-rope-fusion --cross-entropy-loss-fusion
    --mtp-num-layers 0
    # A800 uses the model's standard alltoall dispatcher; DeepEP is not enabled.
    "${SCHEDULE_ARGS[@]}"
)
if [[ "${TASK16_MODEL}" == moe ]]; then
    TRAIN_ARGS+=(--moe-router-load-balancing-type none --moe-aux-loss-coeff 0 --moe-z-loss-coeff 0)
fi
RECORD_ARGS=(--workload "${WORKLOAD_DIR}" --checkpoint "${HF_CHECKPOINT}" --run-dir "${RUN_DIR}" --mode "${MODE}")
if [[ "${TASK16_CAPTURE:-0}" == 1 ]]; then RECORD_ARGS+=(--capture); fi
if [[ "${TASK16_DRY_RUN:-0}" == 1 ]]; then RECORD_ARGS+=(--dry-run); fi
python -m examples.flops_balancing.record_run "${RECORD_ARGS[@]}" -- python3 -m relax.entrypoints.train "${TRAIN_ARGS[@]}"
if [[ "${TASK16_DRY_RUN:-0}" == 1 ]]; then exit 0; fi
if [[ "${RELAX_ENTRYPOINT_MODE:-}" != ray-job ]]; then
    echo "Launch this script through scripts/entrypoint/ray-job.sh" >&2
    exit 2
fi
: "${RAY_ADDRESS:?Set the actual Ray dashboard address}"
# Explicitly disable old capture settings for timing runs as well.
export TASK16_CAPTURE_PATH=""
if [[ "${TASK16_CAPTURE:-0}" == 1 ]]; then export TASK16_CAPTURE_PATH="${RUN_DIR}/capture"; fi
RUNTIME_ENV_JSON="$(python - <<'PY'
import json, os
env = json.loads(os.environ['RUNTIME_ENV_JSON'])
env.setdefault('env_vars', {})['RELAX_FLOPS_CAPTURE_DIR'] = os.environ['TASK16_CAPTURE_PATH']
print(json.dumps(env))
PY
)"
ray job submit --no-wait --address="${RAY_ADDRESS}" \
    ${WORKING_DIR:+--working-dir "${WORKING_DIR}"} --runtime-env-json="${RUNTIME_ENV_JSON}" \
    -- python3 -m relax.entrypoints.train "${TRAIN_ARGS[@]}" 2>&1 | tee "${RUN_DIR}/submit.log"
# --no-wait output is only submission status. Save actual training logs with ray job logs later.
