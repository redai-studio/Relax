#!/bin/bash
# Copyright (c) 2026 Relax Authors. All Rights Reserved.
#
# Geometry3K visual geometry, BF16 GLM-5.3-Flash LoRA GRPO, 32 colocated GPUs.
# On the Ray head, set RAY_JOB_ADDRESS to its Jobs HTTP URL, then run:
#   RELAX_SKIP_CLUSTER_CLEANUP=1 bash scripts/entrypoint/ray-job.sh \
#     examples/geometry3k_rl/scripts/run-glm5.3-flash-lora-32xgpu.sh

set -eo pipefail
export RAY_NO_WAIT="${RAY_NO_WAIT-}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
RELAX_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
MODEL_DIR="${MODEL_DIR:-${RELAX_ROOT}/../../ckpt}"
DATA_DIR="${DATA_DIR:-${RELAX_ROOT}/../../dataset/glm53-flash-geometry3k}"
SAVE_DIR="${SAVE_DIR:-${MODEL_DIR}/relax-rl}"
MODEL_NAME="${MODEL_NAME:-relax-sft/glm5.3-flash-opensearch-vl-32xgpu/bf16/model-base}"
HF_CHECKPOINT="${MODEL_DIR}/${MODEL_NAME}"
PROJECT_NAME="${PROJECT_NAME:-Relax/rl/geometry3k}"
EXP_NAME="${EXP_NAME:-glm5.3-flash-geometry3k-lora-32xgpu}"
NUM_ROLLOUT="${NUM_ROLLOUT:-8}"
ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-16}"
SAMPLES_PER_PROMPT="${SAMPLES_PER_PROMPT:-4}"
MAX_RESPONSE_LEN="${MAX_RESPONSE_LEN:-4096}"
EVAL_MAX_RESPONSE_LEN="${EVAL_MAX_RESPONSE_LEN:-${MAX_RESPONSE_LEN}}"
MAX_PROMPT_LEN=1024
MAX_CONTEXT_LEN=$((MAX_PROMPT_LEN + MAX_RESPONSE_LEN))
EVAL_CONTEXT_LEN=$((MAX_PROMPT_LEN + EVAL_MAX_RESPONSE_LEN))
SGLANG_CONTEXT_LEN=$((MAX_CONTEXT_LEN > EVAL_CONTEXT_LEN ? MAX_CONTEXT_LEN : EVAL_CONTEXT_LEN))
EVAL_FILE="${EVAL_FILE:-${DATA_DIR}/validation-pilot.jsonl}"
EVAL_NAME="${EVAL_NAME:-geometry3k_validation_pilot}"
now=$(date "+%Y%m%d-%H%M%S")

for input in "${HF_CHECKPOINT}/config.json" "${DATA_DIR}/train.jsonl" "${EVAL_FILE}"; do
    if [[ ! -s "${input}" ]]; then
        echo "Missing checkpoint or data input: ${input}" >&2
        exit 1
    fi
done

# The Bridge weight export and SGLang's live weight update use BF16.
python3 - "${HF_CHECKPOINT}/config.json" <<'PYCONFIG'
import json
import sys
from pathlib import Path

cfg = json.loads(Path(sys.argv[1]).read_text())
text = cfg.get("text_config", {})
if (
    cfg.get("model_type") != "glm5_next"
    or cfg.get("quantization_config")
    or text.get("quantization_config")
    or text.get("dtype", text.get("torch_dtype")) != "bfloat16"
):
    raise SystemExit("Require a dequantized BF16 GLM-5.3-Flash HF checkpoint")
PYCONFIG

: "${RAY_JOB_ADDRESS:?Set RAY_JOB_ADDRESS to the Ray Jobs HTTP URL}"
if [[ -z "${RELAX_ENTRYPOINT_MODE:-}" ]]; then
    export RELAX_SKIP_CLUSTER_CLEANUP="${RELAX_SKIP_CLUSTER_CLEANUP:-1}"
    exec bash "${RELAX_ROOT}/scripts/entrypoint/ray-job.sh" "${BASH_SOURCE[0]}"
fi
source "${RELAX_ROOT}/scripts/models/glm5.3-flash-320B-A18B.sh"

CKPT_ARGS=(
    --hf-checkpoint "${HF_CHECKPOINT}"
    --megatron-to-hf-mode bridge
    --load "${SAVE_DIR}/${EXP_NAME}"
    --save "${SAVE_DIR}/${EXP_NAME}"
    --save-interval "${SAVE_INTERVAL:-${NUM_ROLLOUT}}"
    --max-actor-ckpt-to-keep 2
    --save-lora-only
)

LORA_ARGS=(
    --lora-rank 16
    --lora-alpha 32
    --lora-dropout 0
    --lora-scope language
    --lora-merge-mode
    --lora-target-modules in_proj beta_proj f_a_proj f_b_proj g_a_proj g_b_proj out_proj linear_q_down_proj linear_q_up_proj linear_kv_down_proj linear_kv_up_proj linear_proj linear_fc1 linear_fc2
    --freeze-vision-model
    --freeze-vision-projection
)

ROLLOUT_ARGS=(
    --prompt-data "${DATA_DIR}/train.jsonl"
    --input-key prompt
    --label-key label
    --multimodal-keys '{"image":"images"}'
    --apply-chat-template
    --apply-chat-template-kwargs '{"reasoning_effort":"low"}'
    --image-max-token-num 256
    --image-min-token-num 1
    --rollout-shuffle
    --rollout-seed 2401
    --rollout-result-dir "${SAVE_DIR}/${EXP_NAME}/results"
    --rm-type dapo
    --custom-rm-path relax.engine.rewards.math_dapo_utils.compute_completed_math_score
    --reward-key score
    --eval-reward-key acc
    --num-rollout "${NUM_ROLLOUT}"
    --rollout-batch-size "${ROLLOUT_BATCH_SIZE}"
    --n-samples-per-prompt "${SAMPLES_PER_PROMPT}"
    --global-batch-size "$((ROLLOUT_BATCH_SIZE * SAMPLES_PER_PROMPT))"
    --rollout-max-prompt-len "${MAX_PROMPT_LEN}"
    --rollout-max-response-len "${MAX_RESPONSE_LEN}"
    --rollout-max-context-len "${MAX_CONTEXT_LEN}"
    --rollout-temperature 1.0
    --rollout-skip-special-tokens
    --balance-data
)

PERF_ARGS=(
    --tensor-model-parallel-size 4
    --pipeline-model-parallel-size 2
    --hybrid-layer-pattern "${flash_hybrid_pattern}"
    --context-parallel-size 4
    --cp-comm-type allgather
    --linear-cp-mode headwise
    --expert-model-parallel-size 16
    --expert-tensor-parallel-size 1
    --sequence-parallel
    --moe-token-dispatcher-type flex
    --moe-flex-dispatcher-backend deepep
    --moe-flex-dispatcher-num-sms 24
    --moe-shared-expert-overlap
    --recompute-granularity full
    --recompute-method uniform
    --recompute-num-layers 2
    --use-dynamic-batch-size
    --max-tokens-per-gpu "${MAX_TOKENS_PER_GPU:-2048}"
    --seq-length "${MAX_CONTEXT_LEN}"
    --mtp-num-layers 0
    --empty-unused-memory-level 1
    --log-probs-chunk-size 1024
    --dsa-kernel-backend "${DSA_KERNEL_BACKEND:-tilelang}"
    --attention-backend flash
    --attention-softmax-in-fp32
    --attention-dropout 0
    --hidden-dropout 0
    --bf16
)

GRPO_ARGS=(
    --advantage-estimator grpo
    --calculate-per-token-loss
    --kl-loss-coef 0
    --entropy-coef 0
    --eps-clip 0.2
    --eps-clip-high 0.28
    --get-mismatch-metrics
    --use-tis
    --custom-tis-function-path relax.backends.megatron.loss.vanilla_tis_function
)

OPTIMIZER_ARGS=(
    --optimizer adam
    --lr "${LEARNING_RATE:-2e-6}"
    --lr-decay-style constant
    --weight-decay 0
    --adam-beta1 0.9
    --adam-beta2 0.98
    --clip-grad 1
    --optimizer-cpu-offload
    --optimizer-offload-fraction 1.0
    --overlap-cpu-optimizer-d2h-h2d
    --use-precision-aware-optimizer
    --main-grads-dtype bf16
)

# Two TP16 rollout engines keep one CUDA IPC handle per global TP rank.
SGLANG_ARGS=(
    --rollout-num-gpus-per-engine 16
    --sglang-pp-size 1
    --sglang-load-format dummy
    --sglang-dtype bfloat16
    --sglang-mem-fraction-static "${SGLANG_MEM_FRACTION_STATIC:-0.82}"
    --sglang-attention-backend dsa
    --sglang-dsa-prefill-backend tilelang
    --sglang-dsa-decode-backend tilelang
    --sglang-kv-cache-dtype bfloat16
    --sglang-page-size 64
    --sglang-context-length "${SGLANG_CONTEXT_LEN}"
    --sglang-chunked-prefill-size 2048
    --sglang-max-running-requests 16
    --sglang-disable-cuda-graph
    --sglang-disable-prefill-cuda-graph
    --sglang-dist-timeout 1800
    --sglang-watchdog-timeout 3600
)

EVAL_ARGS=(
    --eval-prompt-data "${EVAL_NAME}" "${EVAL_FILE}"
    --eval-interval "${EVAL_INTERVAL:-${NUM_ROLLOUT}}"
    --n-samples-per-eval-prompt 1
    --eval-temperature 0
    --eval-top-p 1
    --eval-max-prompt-len "${MAX_PROMPT_LEN}"
    --eval-max-response-len "${EVAL_MAX_RESPONSE_LEN}"
    --eval-max-context-len "${EVAL_CONTEXT_LEN}"
)
if [[ "${SKIP_EVAL_BEFORE_TRAIN:-0}" == 1 ]]; then
    EVAL_ARGS+=(--skip-eval-before-train)
fi

METRICS_ARGS=(
    --use-clearml
    --use-metrics-service
    --tb-project-name "${PROJECT_NAME}"
    --tb-experiment-name "${EXP_NAME}-${now}"
)

mkdir -p "${SCRIPT_DIR}/log"
ray job submit ${RAY_NO_WAIT:+--no-wait} \
    --address="${RAY_JOB_ADDRESS}" \
    ${WORKING_DIR:+--working-dir "${WORKING_DIR}"} \
    --runtime-env-json="${RUNTIME_ENV_JSON}" \
    -- python3 -m relax.entrypoints.train \
    --resource '{"actor":[1,32],"rollout":[1,32]}' \
    --colocate \
    --max-staleness 0 \
    --num-data-storage-units 4 \
    "${MODEL_ARGS[@]}" \
    "${CKPT_ARGS[@]}" \
    "${LORA_ARGS[@]}" \
    "${ROLLOUT_ARGS[@]}" \
    "${PERF_ARGS[@]}" \
    "${GRPO_ARGS[@]}" \
    "${OPTIMIZER_ARGS[@]}" \
    "${SGLANG_ARGS[@]}" \
    "${EVAL_ARGS[@]}" \
    "${METRICS_ARGS[@]}" 2>&1 | tee "${SCRIPT_DIR}/log/${EXP_NAME}-${now}.log"
