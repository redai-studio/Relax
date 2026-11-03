#!/bin/bash
# Copyright (c) 2026 Relax Authors. All Rights Reserved.
# Countdown: native RLOO / RLOO+TIS / SC / SC+TIS, eight colocated GPUs.
# USE_SC=0 USE_TIS=1 bash scripts/entrypoint/ray-job.sh examples/algorithms/score-centering/run-qwen3-0.6B-8xgpu.sh
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd -P)"
RELAX_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." &>/dev/null && pwd -P)"
if [[ -z "${RELAX_ENTRYPOINT_MODE:-}" ]]; then
    source "${RELAX_ROOT}/scripts/entrypoint/ray-job.sh"
fi
source "${RELAX_ROOT}/scripts/models/qwen3-0.6B.sh"
: "${MODEL_DIR:?Set MODEL_DIR to the model root}"
: "${DATA_DIR:?Set DATA_DIR to the Countdown JSONL directory}"
: "${SAVE_DIR:?Set SAVE_DIR to a persistent checkpoint root}"
: "${RAY_ADDRESS:?Set RAY_ADDRESS to the Ray Jobs HTTP address}"
USE_SC="${USE_SC:-0}"
USE_TIS="${USE_TIS:-0}"
case "${USE_SC}:${USE_TIS}" in
    0:0) ARM=rloo ;;
    0:1) ARM=rloo_tis ;;
    1:0) ARM=sc ;;
    1:1) ARM=sc_tis ;;
    *) echo "USE_SC and USE_TIS must each be 0 or 1" >&2; exit 2 ;;
esac
PROJECT_NAME="${PROJECT_NAME:-Relax/validation/pr440-native-countdown}"
EXP_NAME="${EXP_NAME:-qwen3-06b-countdown-8gpu-${ARM}-s0}"
now=$(date -u +%Y%m%dT%H%M%SZ)
export RAY_NO_WAIT="${RAY_NO_WAIT-}"

CKPT_ARGS=(
    --hf-checkpoint "${MODEL_DIR}/Qwen3-0.6B-FP8"
    --ref-load "${MODEL_DIR}/Qwen3-0.6B"
    --megatron-to-hf-mode bridge --warm-hf-checkpoint-page-cache
    --load "${SAVE_DIR}/${EXP_NAME}/checkpoint"
    --save "${SAVE_DIR}/${EXP_NAME}/checkpoint"
    --save-interval "${SAVE_INTERVAL:-100}" --max-actor-ckpt-to-keep 1
    --save-hf "${SAVE_DIR}/${EXP_NAME}/hf/{rollout_id}" --save-hf-dtype bf16
)
ROLLOUT_ARGS=(
    --prompt-data "${DATA_DIR}/countdown-train.jsonl"
    --input-key prompt --label-key label --apply-chat-template
    --apply-chat-template-kwargs '{"enable_thinking":false}'
    --custom-rm-path examples.algorithms.score-centering.recipe.signed_reward
    --custom-generate-function-path examples.algorithms.score-centering.recipe.generate_bounded
    --rollout-seed 0 --num-rollout "${NUM_ROLLOUT:-400}"
    --rollout-batch-size 64 --n-samples-per-prompt 8 --global-batch-size 512
    --rollout-max-response-len 512 --rollout-temperature 1 --rollout-top-p 1 --rollout-top-k -1
    --balance-data --reward-num-workers 4 --reward-max-concurrency 8
    --rollout-result-dir "${SAVE_DIR}/${EXP_NAME}/rollouts"
)
EVAL_ARGS=(
    --eval-interval "${EVAL_INTERVAL:-25}"
    --eval-prompt-data dev "${DATA_DIR}/countdown-dev.jsonl"
    --n-samples-per-eval-prompt 8 --eval-temperature 1 --eval-top-p 1 --eval-top-k -1
    --eval-max-response-len 512
)
ALGO_ARGS=(--kl-coef 0 --entropy-coef 0 --clip-grad 0)
if [[ "$USE_SC" == 1 ]]; then
    ALGO_ARGS+=(--advantage-estimator score_centering
        --rollout-top-logprobs-num 128 --use-rollout-logprobs --get-mismatch-metrics)
    if [[ "$USE_TIS" == 1 ]]; then
        ALGO_ARGS+=(--score-centering-is tis --score-centering-tis-clip 2)
    else
        ALGO_ARGS+=(--score-centering-is none)
    fi
else
    ALGO_ARGS+=(--advantage-estimator rloo)
    if [[ "$USE_TIS" == 1 ]]; then
        ALGO_ARGS+=(--use-tis --tis-clip 2 --tis-clip-low 0)
    fi
fi
OPTIMIZER_ARGS=(
    --optimizer sgd --sgd-momentum 0 --lr 0.01 --lr-decay-style constant
    --weight-decay 0 --seed 0 --bf16
)
PERF_ARGS=(
    --tensor-model-parallel-size 1 --pipeline-model-parallel-size 1 --context-parallel-size 1
    --expert-model-parallel-size 1 --expert-tensor-parallel-size 1
    --calculate-per-token-loss --use-dynamic-batch-size
    --max-tokens-per-gpu 4096 --log-probs-max-tokens-per-gpu 4096 --log-probs-chunk-size 4
    --update-weight-buffer-size 134217728 --train-memory-margin-bytes 536870912
    --recompute-loss-function --recompute-granularity full --recompute-method uniform --recompute-num-layers 1
)
SGLANG_ARGS=(
    --rollout-num-gpus-per-engine 1 --sglang-mem-fraction-static 0.3
    --sglang-sampling-backend pytorch --sglang-fp8-gemm-backend triton
    --sglang-kv-cache-dtype fp8_e4m3
)
METRICS_ARGS=(
    --use-clearml --use-metrics-service
    --tb-project-name "$PROJECT_NAME" --tb-experiment-name "${EXP_NAME}-${now}"
)
MISC_ARGS=(
    --attention-dropout 0 --hidden-dropout 0 --accumulate-allreduce-grads-in-fp32
    --attention-softmax-in-fp32 --attention-backend flash
)
# Probabilities must describe the distribution actually sampled, including FP8 effects.
RUNTIME_ENV_JSON=$(python - "$RUNTIME_ENV_JSON" <<'PY'
import json, sys
env = json.loads(sys.argv[1])
env["env_vars"]["SGLANG_RETURN_ORIGINAL_LOGPROB"] = "0"
print(json.dumps(env))
PY
)
mkdir -p "${SCRIPT_DIR}/log"
ray job submit ${RAY_NO_WAIT:+--no-wait} --address="$RAY_ADDRESS" \
    --submission-id "${SUBMISSION_ID:-${EXP_NAME}-${now}}" \
    ${WORKING_DIR:+--working-dir "$WORKING_DIR"} --runtime-env-json="$RUNTIME_ENV_JSON" \
    -- python -m relax.entrypoints.train \
    --resource '{"actor":[1,8],"rollout":[1,8]}' --max-staleness 0 --colocate \
    --num-data-storage-units 1 --use-health-check \
    "${MODEL_ARGS[@]}" "${CKPT_ARGS[@]}" "${ROLLOUT_ARGS[@]}" "${EVAL_ARGS[@]}" \
    "${ALGO_ARGS[@]}" "${OPTIMIZER_ARGS[@]}" "${PERF_ARGS[@]}" "${SGLANG_ARGS[@]}" \
    "${METRICS_ARGS[@]}" "${MISC_ARGS[@]}" "$@" \
    2>&1 | tee "${SCRIPT_DIR}/log/${EXP_NAME}-${now}.log"
