#!/bin/bash
# Copyright (c) 2026 Relax Authors. All Rights Reserved.
#
# GLM-5.3 BF16 LoRA GRPO, 32 physical GPUs, colocated Actor/Rollout.
# Baseline: docker/Dockerfile (SGLang 0.5.17) or Dockerfile.cu13 (0.5.21),
# including their Megatron/SGLang patches. GPU validation is still required.
# MODEL_DIR, DATA_DIR and SAVE_DIR must be visible on all nodes.
# Use an explicit BF16 export; the original GLM-5.3 release has FP8 metadata.
# Submit through the established cluster entrypoint, for example:
#   bash scripts/entrypoint/ray-job.sh scripts/training/text/run-glm5.3-744B-A40B-lora-32xgpu.sh

set -eo pipefail
export RAY_NO_WAIT="${RAY_NO_WAIT-}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
: "${MODEL_DIR:?Set MODEL_DIR to the model root}"
: "${DATA_DIR:?Set DATA_DIR to the dataset root}"
: "${SAVE_DIR:?Set SAVE_DIR to a persistent checkpoint root}"
MODEL_NAME="${MODEL_NAME:-glm53-base-dequant-bf16}"
: "${EVAL_DATA:?Set EVAL_DATA to the fixed evaluation JSONL}"
PROJECT_NAME="${PROJECT_NAME:-Relax/rl/dapo-math}"
EXP_NAME="${EXP_NAME:-glm53-dapo-lora-32xgpu-colocate}"
NUM_ROLLOUT="${NUM_ROLLOUT:-200}"
now=$(date "+%Y%m%d-%H%M%S")

if [ -z "${RELAX_ENTRYPOINT_MODE:-}" ]; then
    source "${SCRIPT_DIR}/../../entrypoint/local.sh"
fi
if [ "${RELAX_ENTRYPOINT_MODE}" = local ]; then
    RAY_JOB_ADDRESS="${RAY_JOB_ADDRESS:-http://${MASTER_ADDR}:8265}"
fi
: "${RAY_JOB_ADDRESS:?Set RAY_JOB_ADDRESS to the Ray Jobs HTTP URL}"
source "${SCRIPT_DIR}/../../models/glm5.3-744B-A40B.sh"

CKPT_ARGS=(
    --hf-checkpoint "${MODEL_DIR}/${MODEL_NAME}"
    --ref-load "${MODEL_DIR}/${MODEL_NAME}"
    --megatron-to-hf-mode bridge
    --load "${SAVE_DIR}/${EXP_NAME}"
    --save "${SAVE_DIR}/${EXP_NAME}"
    --save-interval "${SAVE_INTERVAL:-20}"
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

ROLLOUT_ARGS=(
    --prompt-data "${DATA_DIR}/train.jsonl"
    --input-key prompt
    --label-key label
    --apply-chat-template
    --rollout-shuffle
    --rm-type dapo
    --custom-rm-path relax.engine.rewards.math_dapo_utils.compute_completed_math_score
    --reward-key score
    --eval-reward-key acc
    --rollout-seed 20261116
    --rollout-result-dir "${SAVE_DIR}/${EXP_NAME}/results"
    --num-rollout "${NUM_ROLLOUT}"
    --rollout-batch-size "${ROLLOUT_BATCH_SIZE:-32}"
    --n-samples-per-prompt 4
    --rollout-max-response-len "${ROLLOUT_MAX_RESPONSE_LEN:-32768}"
    --rollout-max-prompt-len 4096
    --rollout-max-context-len "${ROLLOUT_MAX_CONTEXT_LEN:-36864}"
    --rollout-temperature 1
    --rollout-skip-special-tokens
    --global-batch-size "${GLOBAL_BATCH_SIZE:-128}"
    --balance-data
)

PERF_ARGS=(
    # Keep full TMS offload: frozen LoRA base weights are outside DDP buffers.
    --tensor-model-parallel-size 4
    --sequence-parallel
    --pipeline-model-parallel-size 2
    # 38/40 layers: stage 2 starts at a full indexer.
    # Each starts on a full indexer; shared top-k never crosses PP boundaries.
    --decoder-first-pipeline-num-layers 38
    --decoder-last-pipeline-num-layers 40
    --context-parallel-size 4
    --expert-model-parallel-size 16
    --expert-tensor-parallel-size 1
    --recompute-granularity full
    --recompute-method uniform
    --recompute-num-layers 1
    --use-dynamic-batch-size
    --max-tokens-per-gpu "${MAX_TOKENS_PER_GPU:-9216}"
    # Release activation cache before lazy NCCL gradient-finalization buffers.
    --empty-unused-memory-level 1
    --data-pad-size-multiplier 256
    --log-probs-chunk-size 1024
    --moe-flex-dispatcher-backend deepep
    --moe-token-dispatcher-type flex
    --calculate-per-token-loss
    # The discrete top-k path has no indexer auxiliary training loss.
    --freeze-params-name-list self_attention.wq_b self_attention.wk self_attention.k_norm self_attention.weights_proj
)

GRPO_ARGS=(
    --advantage-estimator grpo
    --kl-loss-coef 0.0
    --entropy-coef 0.0
    --eps-clip 0.2
    --eps-clip-high 0.28
    --use-tis
)

OPTIMIZER_ARGS=(
    --optimizer adam
    --lr "${LEARNING_RATE:-2e-6}"
    --lr-decay-style constant
    --weight-decay 0.0
    --adam-beta1 0.9
    --adam-beta2 0.98
    --optimizer-cpu-offload
    --overlap-cpu-optimizer-d2h-h2d
    --use-precision-aware-optimizer
    --no-pin-cpu-grads
    --no-pin-cpu-params
)
if [[ "${OVERRIDE_OPT_PARAM_SCHEDULER:-0}" == "1" ]]; then
    OPTIMIZER_ARGS+=(--override-opt_param-scheduler)
fi

# FP8 KV is opt-in and needs its own initial eval; model weights remain BF16.
# Keep BF16 sparse prefill (dequantizing KV), avoiding FP8 query quantization.
SGLANG_KV_CACHE_DTYPE="${SGLANG_KV_CACHE_DTYPE:-bfloat16}"
case "${SGLANG_KV_CACHE_DTYPE}" in
    bfloat16) SGLANG_DSA_DECODE_BACKEND=flashmla_sparse ;;
    fp8_e4m3) SGLANG_DSA_DECODE_BACKEND=flashmla_kv ;;
    *) echo "Unsupported SGLANG_KV_CACHE_DTYPE: ${SGLANG_KV_CACHE_DTYPE}" >&2; exit 1 ;;
esac

SGLANG_ARGS=(
    --rollout-num-gpus-per-engine 32
    --sglang-mem-fraction-static "${SGLANG_MEM_FRACTION_STATIC:-0.88}"
    --sglang-enable-dp-attention
    --sglang-ep-size 32
    --sglang-dp-size 4
    --sglang-moe-dense-tp-size 1
    --sglang-moe-a2a-backend deepep
    --sglang-moe-runner-backend deep_gemm
    --sglang-deepep-dispatcher-output-dtype bf16
    --sglang-deepep-mode auto
    --sglang-load-format dummy
    --sglang-page-size 64
    --sglang-dsa-decode-backend "${SGLANG_DSA_DECODE_BACKEND}"
    --sglang-dsa-prefill-backend flashmla_sparse
    --sglang-attention-backend dsa
    --sglang-kv-cache-dtype "${SGLANG_KV_CACHE_DTYPE}"
    --sglang-context-length 40960
    --sglang-disable-prefill-cuda-graph
    # DP4 over TP32 requires decode graph batches aligned to attention TP8.
    --sglang-cuda-graph-max-bs 8
    --sglang-max-running-requests 32
    # DP attention scales this by 0.3; reserve ratio starts at 0.63 and decays to 0.315.
    --sglang-schedule-conservativeness "${SGLANG_SCHEDULE_CONSERVATIVENESS:-3}"
    # DP4 divides this into 512-token chunks, bounding re-prefill logprob temporaries.
    --sglang-chunked-prefill-size 2048
    --sglang-watchdog-timeout 3600
)
# Validate memory headroom before opting into decode graphs on a new layout.
if [[ "${SGLANG_ENABLE_CUDA_GRAPH:-0}" != "1" ]]; then
    SGLANG_ARGS+=(--sglang-disable-cuda-graph)
fi

# Interval >=2 preserves initial eval/0.jsonl (the first update also has id 0).
EVAL_ARGS=(
    --eval-interval "${EVAL_INTERVAL:-20}"
    --eval-prompt-data "${EVAL_NAME:-validation}" "${EVAL_DATA}"
    --n-samples-per-eval-prompt 1
    --eval-temperature 0
    --eval-top-p 1
    --eval-max-response-len "${EVAL_MAX_RESPONSE_LEN:-32768}"
)
if [[ -n "${EVAL_CONFIG:-}" ]]; then
    EVAL_ARGS+=(--eval-config "${EVAL_CONFIG}")
fi
# Optional: start training immediately while retaining periodic evaluation.
if [[ "${SKIP_EVAL_BEFORE_TRAIN:-0}" == "1" ]]; then
    EVAL_ARGS+=(--skip-eval-before-train)
fi

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
    --update-weight-buffer-size 134217728
)
if [[ -n "${CUSTOM_CONFIG_PATH:-}" ]]; then
    MISC_ARGS+=(--custom-config-path "${CUSTOM_CONFIG_PATH}")
fi

# Propagate validated inference settings to all workers; preserve NaN failures.
RUNTIME_ENV_JSON=$(python - <<'PYENV'
import json
import os
runtime = json.loads(os.environ["RUNTIME_ENV_JSON"])
runtime.setdefault("env_vars", {}).update(
    SGLANG_ENABLE_ASYNC_ASSERT="1",
    SGLANG_SANITIZE_NAN_LOGITS="0",
    SGLANG_DSA_PREFILL_DENSE_ATTN_KV_LEN_THRESHOLD="0",
    # Account for occupied KV plus estimated remaining tokens; retain long-tail headroom.
    SGLANG_INIT_NEW_TOKEN_RATIO="0.7",
    SGLANG_MIN_NEW_TOKEN_RATIO_FACTOR="0.5",
    SGLANG_CLIP_MAX_NEW_TOKENS_ESTIMATION="16384",
    CUDA_HOME=os.environ.get("CUDA_HOME", "/usr/local/cuda"),
)
print(json.dumps(runtime))
PYENV
)
mkdir -p "${SCRIPT_DIR}/log"
ray job submit ${RAY_NO_WAIT:+--no-wait} \
    --address="${RAY_JOB_ADDRESS}" \
    ${WORKING_DIR:+--working-dir "${WORKING_DIR}"} \
    --runtime-env-json="${RUNTIME_ENV_JSON}" \
    -- python3 -m relax.entrypoints.train \
    --resource '{"actor": [1, 32], "rollout": [1, 32]}' \
    --max-staleness 0 \
    --num-data-storage-units 2 \
    --colocate \
    "${MODEL_ARGS[@]}" \
    "${CKPT_ARGS[@]}" \
    "${LORA_ARGS[@]}" \
    "${ROLLOUT_ARGS[@]}" \
    "${PERF_ARGS[@]}" \
    "${GRPO_ARGS[@]}" \
    "${OPTIMIZER_ARGS[@]}" \
    "${SGLANG_ARGS[@]}" \
    "${EVAL_ARGS[@]}" \
    "${WANDB_ARGS[@]}" \
    "${MISC_ARGS[@]}" 2>&1 | tee "${SCRIPT_DIR}/log/${EXP_NAME}-${now}.log"
