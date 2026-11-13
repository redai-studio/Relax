#!/usr/bin/env bash
# Copyright (c) 2026 Relax Authors. All Rights Reserved.
# Score Centering hybrid async: 6 actor GPUs + 2 rollout GPUs; every launch starts a fresh RL experiment.
# Single node: MODEL_DIR=/path/to/hf DATA_DIR=/path/to/data SAVE_DIR=/path/to/checkpoints bash <this-script>
# Existing cluster: bash scripts/entrypoint/ray-job.sh <this-script>
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
RELAX_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
: "${MODEL_DIR:?Set the initial Hugging Face model directory}"
: "${DATA_DIR:?Set the prepared ReTool data directory}"
: "${SAVE_DIR:?Set a persistent checkpoint root}"
[[ -z "${EVAL_CONFIG:-}" ]] || { echo "This Agentic recipe currently supports training only; omit EVAL_CONFIG" >&2; exit 1; }
[[ -f "${MODEL_DIR}/config.json" ]] || { echo "Missing HF model config: ${MODEL_DIR}/config.json" >&2; exit 1; }
[[ -f "${MODEL_DIR}/tokenizer.json" ]] || { echo "Missing fast tokenizer: ${MODEL_DIR}/tokenizer.json" >&2; exit 1; }
[[ -f "${DATA_DIR}/train.jsonl" ]] || { echo "Missing prepared ReTool data: ${DATA_DIR}/train.jsonl" >&2; exit 1; }
export RELAX="${RELAX_ROOT}" RELAX_SKIP_CLUSTER_CLEANUP=1
export RAY_NO_WAIT="${RAY_NO_WAIT-}"
if [[ -z "${RELAX_ENTRYPOINT_MODE:-}" ]]; then
    # local.sh reads optional environment variables without nounset guards.
    set +u
    source "${RELAX_ROOT}/scripts/entrypoint/local.sh"
    set -u
fi
source "${RELAX_ROOT}/scripts/models/qwen25-7B.sh"
PROJECT_NAME="${PROJECT_NAME:-Relax/retool-agentic}"
EXP_NAME="${EXP_NAME:-qwen25-7b-retool-agentic-sc-8xgpu-hybrid-6a2r}"
now=$(date +%Y%m%d-%H%M%S)
export DATA_DIR SAVE_DIR
export PYTHONPATH="${RELAX_ROOT}:${PYTHONPATH:-}"
RUNTIME_ENV_JSON=$(python3 - <<'ENV'
import json
import os
value = json.loads(os.environ["RUNTIME_ENV_JSON"])
names = ["DATA_DIR", "SAVE_DIR", "SANDBOX_FUSION_URL", "SANDBOX_IMAGE", "RELAX_PROPAGATE_ENV_VARS", "NCCL_NVLS_ENABLE"]
value.setdefault("env_vars", {})["PYTHONPATH"] = os.environ["PYTHONPATH"]
value["env_vars"].update({name: os.environ[name] for name in names if name in os.environ})
value["env_vars"]["SGLANG_RETURN_ORIGINAL_LOGPROB"] = "0"
print(json.dumps(value))
ENV
)
START_ROLLOUT_ID=$(python3 -m examples.retool_agentic.checkpoint "${SAVE_DIR}/${EXP_NAME}" --fresh)
printf -v AGENT_COMMAND '%q -m examples.retool_agentic.agent' "$(command -v python3)"
CKPT_ARGS=(
    --disable-s3-model-download --hf-checkpoint "${MODEL_DIR}" --megatron-to-hf-mode bridge
    --load "${SAVE_DIR}/${EXP_NAME}" --save "${SAVE_DIR}/${EXP_NAME}"
    --start-rollout-id "${START_ROLLOUT_ID}"
    --use-checkpoint-opt_param-scheduler
    --save-interval 25 --max-actor-ckpt-to-keep 8 --save-hf "${SAVE_DIR}/${EXP_NAME}/hf_{rollout_id}"
)
ROLLOUT_ARGS=(
    --prompt-data "${DATA_DIR}/train.jsonl" --input-key prompt --label-key label --metadata-key metadata
    --apply-chat-template --custom-config-path "${RELAX_ROOT}/examples/retool_agentic/config-hybrid.yaml"
    --data-source-path examples.retool_agentic.data_source.ReToolAgenticDataSource
    --use-agentic-rollout --agent-command "${AGENT_COMMAND}"
    --agent-cwd "${RELAX_ROOT}" --agent-env "RETOOL_TOKENIZER_PATH=${MODEL_DIR}/tokenizer.json"
    --agentic-concurrency 16 --agentic-prelaunch --agent-timeout 1800
    --custom-rm-path examples.retool_agentic.reward.reward_func --reward-key score --eval-reward-key acc
    --num-rollout 200 --rollout-batch-size 63 --over-sampling-batch-size 79 --n-samples-per-prompt 16 --global-batch-size 1008
    --num-steps-per-rollout 1 --num-iters-per-train-update 1
    --rollout-max-prompt-len 2048 --rollout-max-response-len 8192 --rollout-max-context-len 10240
    --rollout-temperature 1 --rollout-top-p 1 --rollout-top-k -1 --rollout-seed 20261031
)
ALGORITHM_ARGS=(
    --advantage-estimator score_centering --rollout-top-logprobs-num 128
    --score-centering-is tis --score-centering-tis-clip 2
    --use-rollout-logprobs --get-mismatch-metrics --calculate-per-token-loss
    --entropy-coef 0 --kl-coef 0 --kl-loss-coef 0
)
PERF_ARGS=(
    --weight-decay 0.01 --micro-batch-size 8 --use-dynamic-batch-size
    --max-tokens-per-gpu 16384 --log-probs-max-tokens-per-gpu 32768 --update-weight-buffer-size 1073741824
    --use-distributed-optimizer --bf16
    --tensor-model-parallel-size 2 --pipeline-model-parallel-size 1 --context-parallel-size 1 --sequence-parallel
    --recompute-granularity full --recompute-method uniform --recompute-num-layers 1
    --attention-dropout 0 --hidden-dropout 0 --attention-backend flash
)
SGLANG_ARGS=(
    --resource '{"actor":[1,6],"rollout":[1,2]}' --hybrid --max-staleness 2 --balance-data
    --rollout-num-gpus-per-engine 1 --sglang-server-concurrency 32
    --sglang-mem-fraction-static 0.7 --sglang-router-policy consistent_hashing --num-data-storage-units 1
    # Expected 499 cancellations at weight swaps must not trip the router's worker circuit breakers.
    --sglang-router-disable-circuit-breaker
    --skip-eval-before-train
)
OPTIMIZER_ARGS=(
    --optimizer adam --lr 1e-6 --lr-decay-style constant
    --adam-beta1 0.9 --adam-beta2 0.999 --adam-eps 1e-8 --clip-grad 1
)
TRACKING_ARGS=(
    --use-clearml --use-metrics-service --tb-project-name "${PROJECT_NAME}" --tb-experiment-name "${EXP_NAME}-${now}"
)
mkdir -p "${SCRIPT_DIR}/log"
ray job submit ${RAY_NO_WAIT:+--no-wait} --address="${RAY_ADDRESS:-http://127.0.0.1:8265}" \
    ${WORKING_DIR:+--working-dir "${WORKING_DIR}"} --runtime-env-json="${RUNTIME_ENV_JSON}" \
    -- python3 -m examples.retool_agentic.train "${MODEL_ARGS[@]}" \
    "${CKPT_ARGS[@]}" \
    "${ROLLOUT_ARGS[@]}" \
    "${ALGORITHM_ARGS[@]}" \
    "${OPTIMIZER_ARGS[@]}" \
    "${PERF_ARGS[@]}" \
    "${SGLANG_ARGS[@]}" \
    "${TRACKING_ARGS[@]}" \
    "$@" \
    2>&1 | tee "${SCRIPT_DIR}/log/${EXP_NAME}-${now}.log"
