#!/usr/bin/env bash
# Copyright (c) 2026 Relax Authors. All Rights Reserved.
# Submit to an existing Ray cluster: bash scripts/entrypoint/ray-job.sh <this-script>
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
RELAX_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
: "${MODEL_DIR:?Set the initial Hugging Face model directory}"
: "${DATA_DIR:?Set the prepared ReTool data directory}"
: "${SAVE_DIR:?Set a persistent checkpoint root}"
export RELAX="${RELAX_ROOT}" RELAX_SKIP_CLUSTER_CLEANUP=1
export RAY_NO_WAIT="${RAY_NO_WAIT-}"
if [[ -z "${RELAX_ENTRYPOINT_MODE:-}" ]]; then
    source "${RELAX_ROOT}/scripts/entrypoint/ray-job.sh"
fi
source "${RELAX_ROOT}/scripts/models/qwen25-7B.sh"
PROJECT_NAME="${PROJECT_NAME:-Relax/retool}"
EXP_NAME="${EXP_NAME:-qwen25-7b-retool-grpo-8xgpu}"
now=$(date +%Y%m%d-%H%M%S)
: "${SANDBOX_FUSION_URL:?Set the isolated SandboxFusion /run_code endpoint}"
export SANDBOX_FUSION_URL DATA_DIR
RUNTIME_ENV_JSON=$(python3 - <<'ENV'
import json, os
value = json.loads(os.environ["RUNTIME_ENV_JSON"])
value.setdefault("env_vars", {}).update(SANDBOX_FUSION_URL=os.environ["SANDBOX_FUSION_URL"], DATA_DIR=os.environ["DATA_DIR"])
print(json.dumps(value))
ENV
)
START_ROLLOUT_ID=$(python3 -m examples.retool.checkpoint "${SAVE_DIR}/${EXP_NAME}")
ARGS=(
    --disable-s3-model-download --hf-checkpoint "${MODEL_DIR}" --megatron-to-hf-mode bridge
    --load "${SAVE_DIR}/${EXP_NAME}" --save "${SAVE_DIR}/${EXP_NAME}"
    --start-rollout-id "${START_ROLLOUT_ID}"
    --use-checkpoint-opt_param-scheduler
    --save-interval 25 --max-actor-ckpt-to-keep 8 --save-hf "${SAVE_DIR}/${EXP_NAME}/hf_{rollout_id}"
    --prompt-data "${DATA_DIR}/train.jsonl" --input-key prompt --label-key label --metadata-key metadata
    --apply-chat-template --custom-config-path "${RELAX_ROOT}/examples/retool/config.yaml"
    --data-source-path examples.retool.data_source.ReToolDataSource
    --custom-generate-function-path examples.retool.rollout.generate
    --custom-rm-path examples.retool.reward.reward_func --reward-key score --eval-reward-key acc
    --custom-megatron-before-log-prob-hook-path examples.retool.vocab.before_log_prob
    --custom-megatron-before-train-step-hook-path examples.retool.vocab.before_train_step
    --num-rollout 200 --rollout-batch-size 64 --n-samples-per-prompt 16 --global-batch-size 1024
    --num-steps-per-rollout 1 --num-iters-per-train-update 1
    --rollout-max-prompt-len 2048 --rollout-max-response-len 8192 --rollout-max-context-len 10240
    --rollout-temperature 1 --rollout-top-p 1 --rollout-top-k -1 --rollout-seed 20261031
    --advantage-estimator grpo --eps-clip 0.2 --eps-clip-high 0.2 --entropy-coef 0 --kl-coef 0 --kl-loss-coef 0
    --weight-decay 0.01 --micro-batch-size 8 --use-dynamic-batch-size
    --max-tokens-per-gpu 16384 --log-probs-max-tokens-per-gpu 32768 --update-weight-buffer-size 1073741824
    --resource '{"actor":[1,8],"rollout":[1,8]}' --colocate --max-staleness 0 --balance-data
    --selective-offload --rollout-num-gpus-per-engine 1 --sglang-server-concurrency 32
    --sglang-mem-fraction-static 0.5 --sglang-enable-custom-logit-processor --num-data-storage-units 1
    --skip-eval-before-train
    --optimizer adam --lr 1e-6 --lr-decay-style constant
    --adam-beta1 0.9 --adam-beta2 0.999 --adam-eps 1e-8 --clip-grad 1
    --use-distributed-optimizer --bf16
    --tensor-model-parallel-size 2 --pipeline-model-parallel-size 1 --context-parallel-size 1 --sequence-parallel
    --recompute-granularity full --recompute-method uniform --recompute-num-layers 1
    --attention-dropout 0 --hidden-dropout 0 --attention-backend flash
    --use-clearml --use-metrics-service --tb-project-name "${PROJECT_NAME}" --tb-experiment-name "${EXP_NAME}-${now}"
)
if [[ -n "${EVAL_CONFIG:-}" ]]; then
    ARGS+=(--eval-config "${EVAL_CONFIG}" --eval-interval 25 --eval-max-prompt-len 2048 --eval-max-context-len 10240)
fi
mkdir -p "${SCRIPT_DIR}/log"
RAY_JOB_ARGS=()
if [[ -n "${RETOOL_RAY_SUBMISSION_ID:-}" ]]; then
    RAY_JOB_ARGS+=(--submission-id "${RETOOL_RAY_SUBMISSION_ID}")
fi
ray job submit "${RAY_JOB_ARGS[@]}" ${RAY_NO_WAIT:+--no-wait} --address="${RAY_ADDRESS:-http://127.0.0.1:8265}" \
    ${WORKING_DIR:+--working-dir "${WORKING_DIR}"} --runtime-env-json="${RUNTIME_ENV_JSON}" \
    -- python3 -m relax.entrypoints.train "${MODEL_ARGS[@]}" "${ARGS[@]}" "$@" \
    2>&1 | tee "${SCRIPT_DIR}/log/${EXP_NAME}-${now}.log"
