#!/bin/bash
# Copyright (c) 2026 Relax Authors. All Rights Reserved.
# Deploy a standalone HF checkpoint and evaluate fixed AIME prompts.
# MODEL_NAME selects the explicitly exported checkpoint (no implicit latest).
# Submit via scripts/entrypoint/ray-job.sh after training has finished.
set -eo pipefail
export RAY_NO_WAIT="${RAY_NO_WAIT-}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
: "${MODEL_DIR:?Set MODEL_DIR to the exported HF model root}"
: "${MODEL_NAME:?Set MODEL_NAME to the fixed checkpoint directory name}"
: "${DATA_DIR:?Set DATA_DIR to the evaluation dataset root}"
: "${SAVE_DIR:?Set SAVE_DIR to a persistent evaluation output root}"
PROJECT_NAME="${PROJECT_NAME:-Relax/sft/dapo-math}"
EXP_NAME="${EXP_NAME:-glm5.3-aime-fixed-checkpoint-32xgpu}"
EVAL_DATA_FILE="${EVAL_DATA_FILE:-aime-2024/aime-2024.jsonl}"
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

EVAL_ARGS=(
    --debug-rollout-only
    --hf-checkpoint "${MODEL_DIR}/${MODEL_NAME}"
    --prompt-data "${DATA_DIR}/${EVAL_DATA_FILE}"
    --input-key prompt
    --label-key label
    --apply-chat-template
    --rm-type dapo
    --reward-key score
    --eval-reward-key score
    --num-rollout 1
    --rollout-batch-size 1
    --n-samples-per-prompt 1
    --global-batch-size 1
    --rollout-max-response-len 1
    --rollout-skip-special-tokens
    --eval-interval 10
    --eval-prompt-data "${EVAL_DATASET_NAME:-aime}" "${DATA_DIR}/${EVAL_DATA_FILE}"
    --n-samples-per-eval-prompt 1
    --eval-temperature 0
    --eval-top-p 1
    --eval-max-response-len "${EVAL_MAX_RESPONSE_LEN:-32768}"
    --rollout-result-dir "${SAVE_DIR}/${EXP_NAME}/results"
)
SGLANG_ARGS=(
    --rollout-num-gpus-per-engine 32
    --sglang-mem-fraction-static "${EVAL_MEM_FRACTION_STATIC:-0.85}"
    --sglang-enable-dp-attention
    --sglang-ep-size 32
    --sglang-dp-size 8
    --sglang-moe-dense-tp-size 1
    --sglang-moe-a2a-backend deepep
    # Unquantized experts must use the BF16 DeepGEMM runner, not the legacy auto path.
    --sglang-moe-runner-backend deep_gemm
    --sglang-deepep-dispatcher-output-dtype bf16
    --sglang-deepep-mode auto
    --sglang-page-size 64
    --sglang-dsa-decode-backend flashmla_sparse
    --sglang-dsa-prefill-backend flashmla_sparse
    --sglang-attention-backend dsa
    --sglang-context-length "${EVAL_CONTEXT_LENGTH:-40960}"
    --sglang-disable-prefill-cuda-graph
    --sglang-cuda-graph-max-bs 4
    --sglang-max-running-requests 32
    --sglang-chunked-prefill-size 8192
    --sglang-watchdog-timeout 3600
)
if [[ -n "${SGLANG_FORWARD_HOOKS:-}" ]]; then
    SGLANG_ARGS+=(--sglang-forward-hooks "${SGLANG_FORWARD_HOOKS}")
fi
MONITOR_ARGS=(
    --use-clearml
    --use-metrics-service
    --tb-project-name "${PROJECT_NAME}"
    --tb-experiment-name "${EXP_NAME}-${now}"
)
# Reject invalid inference numerics instead of scoring silent token-zero loops.
# This leaves finite logits unchanged; do not enable logit sanitization.
RUNTIME_ENV_JSON=$(python - <<'PY'
import json
import os

runtime = json.loads(os.environ["RUNTIME_ENV_JSON"])
runtime.setdefault("env_vars", {}).update(
    SGLANG_ENABLE_ASYNC_ASSERT="1",
    SGLANG_SANITIZE_NAN_LOGITS="0",
    # Isolate valid-token NaNs observed in the BF16 dense MHA prefill branch.
    SGLANG_DSA_PREFILL_DENSE_ATTN_KV_LEN_THRESHOLD=os.environ.get(
        "SGLANG_DSA_PREFILL_DENSE_ATTN_KV_LEN_THRESHOLD", "0"
    ),
)
if module_dir := os.environ.get("SGLANG_FORWARD_HOOK_MODULE_DIR"):
    existing = runtime["env_vars"].get("PYTHONPATH", "")
    runtime["env_vars"]["PYTHONPATH"] = os.pathsep.join(filter(None, [module_dir, existing]))
print(json.dumps(runtime))
PY
)
mkdir -p "${SCRIPT_DIR}/log"
ray job submit ${RAY_NO_WAIT:+--no-wait} \
    --address="${RAY_JOB_ADDRESS}" \
    ${WORKING_DIR:+--working-dir "${WORKING_DIR}"} \
    --runtime-env-json="${RUNTIME_ENV_JSON}" \
    -- python3 -m relax.entrypoints.train \
    --resource '{"rollout": [1, 32]}' \
    "${MODEL_ARGS[@]}" \
    "${EVAL_ARGS[@]}" \
    "${SGLANG_ARGS[@]}" \
    "${MONITOR_ARGS[@]}" 2>&1 | tee "${SCRIPT_DIR}/log/${EXP_NAME}-${now}.log"
