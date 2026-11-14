#!/usr/bin/env bash
# Copyright (c) 2026 Relax Authors. All Rights Reserved.
set -euo pipefail
MODEL_DIR=$1
OUTPUT=$2
CONFIG=$3
RELAX_ROOT=$(pwd)
if [[ ${RETOOL_SMOKE:-0} == 1 ]]; then
  GPUS=${RETOOL_EVAL_GPUS:-1}
else
  GPUS=${RETOOL_EVAL_GPUS:-8}
fi
source scripts/models/qwen25-7B.sh
RAY_JOB_ARGS=()
if [[ -n "${RETOOL_RAY_SUBMISSION_ID:-}" ]]; then
    RAY_JOB_ARGS+=(--submission-id "${RETOOL_RAY_SUBMISSION_ID}")
fi
ray job submit "${RAY_JOB_ARGS[@]}" --address="$RAY_ADDRESS" --runtime-env-json="$RUNTIME_ENV_JSON" -- \
  python3 -m relax.entrypoints.train "${MODEL_ARGS[@]}" \
  --disable-s3-model-download --hf-checkpoint "$MODEL_DIR" --save "$OUTPUT" \
  --debug-rollout-only --rollout-num-gpus "$GPUS" --rollout-num-gpus-per-engine 1 \
  --bf16 --n-samples-per-prompt 8 --rollout-batch-size 64 --global-batch-size 512 \
  --resource "{\"rollout\":[1,$GPUS]}" --num-rollout 0 \
  --eval-interval 1 --eval-config "$CONFIG" --n-samples-per-eval-prompt 8 --eval-reward-key acc \
  --custom-eval-rollout-log-function-path examples.retool.evaluation.summarize --log-passrate \
  --custom-rm-path examples.retool.reward.reward_func \
  --custom-generate-function-path examples.retool.rollout.generate \
  --prompt-data "$DATA_DIR/train.jsonl" --input-key prompt --label-key label --metadata-key metadata \
  --apply-chat-template --custom-config-path "$RELAX_ROOT/examples/retool/config.yaml" \
  --rollout-max-prompt-len 2048 --rollout-max-response-len 8192 --rollout-max-context-len 10240 \
  --eval-max-prompt-len 2048 --eval-max-context-len 10240 \
  --rollout-temperature 1 --rollout-top-p 1 --rollout-top-k -1 --rollout-seed 20261031 \
  --sglang-server-concurrency 32 --sglang-mem-fraction-static .5 --sglang-enable-custom-logit-processor \
  --save-debug-rollout-data "$OUTPUT/raw/{rollout_id}.pt" \
  --use-tensorboard --no-use-clearml --no-use-wandb --no-use-metrics-service
