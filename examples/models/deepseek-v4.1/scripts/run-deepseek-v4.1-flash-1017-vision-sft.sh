#!/bin/bash
# Copyright (c) 2026 Relax Authors. All Rights Reserved.
# Frozen-vision OpenSearch-VL THD SFT. Set HF_CKPT to the prepared composite BF16 checkpoint.
# A four-layer proxy uses one H800 node by default. The full model defaults to 16 nodes.
# Run on an existing Ray cluster via scripts/entrypoint/ray-job.sh <this script>.
set -eo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
RELAX_ROOT="$(cd -- "${SCRIPT_DIR}/../../../.." && pwd)"
: "${MODEL_DIR:?Set MODEL_DIR to the model root}"
: "${DATA_DIR:?Set DATA_DIR to the prepared data root}"
: "${SAVE_DIR:?Set SAVE_DIR to a persistent checkpoint root}"
HF_CKPT="${HF_CKPT:-${MODEL_DIR}/DeepSeek-V4.1-Flash-1017-bf16-vision}"
PROMPT_DATA="${PROMPT_DATA:-${DATA_DIR}/opensearch_vl/32k/train.jsonl}"
VAL_DATA="${VAL_DATA-$(dirname -- "${PROMPT_DATA}")/val.jsonl}"
layers=$(python3 - "${HF_CKPT}/config.json" <<'PY'
import json
import sys

with open(sys.argv[1]) as config:
    model = json.load(config)
if model.get("model_type") != "deepseek_v41" or not model.get("vision_config") or model.get("quantization_config"):
    raise SystemExit("Expected a composite DeepSeek V4.1 BF16 checkpoint with its original vision tower")
print(model["text_config"]["num_hidden_layers"])
PY
)
case "${layers}" in
    4) source "${RELAX_ROOT}/scripts/models/deepseek-v4.1-4layer.sh"; NUM_NODES="${NUM_NODES:-1}" ;;
    40) source "${RELAX_ROOT}/scripts/models/deepseek-v4.1-flash.sh"; NUM_NODES="${NUM_NODES:-16}" ;;
    *) echo "The 1017 SFT recipe expects 4 or 40 decoder layers, got ${layers}" >&2; exit 1 ;;
esac
TP_SIZE="${TP_SIZE:-2}"
PP_SIZE="${PP_SIZE:-2}"
CP_SIZE="${CP_SIZE:-2}"
GPU_COUNT=$((8 * NUM_NODES))
EP_SIZE="${EP_SIZE:-$((GPU_COUNT / PP_SIZE))}"
SEQ_LENGTH="${SEQ_LENGTH:-32768}"
MAX_TOKENS_PER_GPU="${MAX_TOKENS_PER_GPU:-$((SEQ_LENGTH / CP_SIZE))}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-32}"
EXP_NAME="${EXP_NAME:-deepseek-v41-1017-${layers}l-frozen-vision-sft-${GPU_COUNT}gpu}"
RUN_NAME="${EXP_NAME}-$(date +%Y%m%d-%H%M%S)"
export RAY_NO_WAIT="${RAY_NO_WAIT-}"
if [[ -n "${NUM_STEPS:-}" ]]; then
    LENGTH_ARGS=(--num-rollout "${NUM_STEPS}")
    SAVE_INTERVAL="${SAVE_INTERVAL:-0}"
else
    LENGTH_ARGS=(--num-epoch "${NUM_EPOCHS:-1}")
    SAVE_INTERVAL="${SAVE_INTERVAL:-200}"
fi
SAVE_ARGS=()
if [[ "${SAVE_INTERVAL}" != 0 ]]; then
    SAVE_ARGS=(--save-interval "${SAVE_INTERVAL}")
fi
if [[ "${NUM_NODES}" == 1 ]]; then
    MOE_DISPATCHER="${MOE_DISPATCHER:-deepep}"
else
    MOE_DISPATCHER="${MOE_DISPATCHER:-alltoall}"
fi
case "${MOE_DISPATCHER}" in
    deepep) DISPATCH_ARGS=(
        --moe-token-dispatcher-type flex
        --moe-flex-dispatcher-backend deepep
    ) ;;
    alltoall) DISPATCH_ARGS=(--moe-token-dispatcher-type alltoall) ;;
    *) echo "MOE_DISPATCHER must be deepep or alltoall" >&2; exit 1 ;;
esac

RESUME_ARGS=()
if [[ -f "${SAVE_DIR}/${EXP_NAME}/latest_checkpointed_iteration.txt" ]]; then
    resume_step=$(python3 - "${SAVE_DIR}/${EXP_NAME}" <<'PY'
import sys

from relax.utils.utils import checkpoint_resume_step

print(checkpoint_resume_step(sys.argv[1]))
PY
)
    # Restore the constant LR/WD schedule and its sample counter when extending a short run.
    RESUME_ARGS+=(
        --start-rollout-id "${resume_step}"
        --use-checkpoint-opt-param-scheduler
    )
fi
EVAL_ARGS=()
if [[ -n "${VAL_DATA}" && -f "${VAL_DATA}" ]]; then
    EVAL_ARGS+=(
        --eval-prompt-data opensearch_val "${VAL_DATA}"
        --eval-interval "${EVAL_INTERVAL:-100}"
    )
fi
METRIC_ARGS=()
if [[ "${ENABLE_SWANLAB:-0}" == 1 ]]; then
    METRIC_ARGS+=(--use-tensorboard)
    export SWANLAB_WORKSPACE="${SWANLAB_WORKSPACE:-redai-studio}"
    export SWANLAB_PROJ_NAME="${SWANLAB_PROJ_NAME:-Relax-opensearch-vl}"
    export SWANLAB_EXP_NAME="${RUN_NAME}"
    export SWANLAB_LOG_DIR="${SWANLAB_LOG_DIR:-${SAVE_DIR}/${EXP_NAME}/swanlab}"
fi

TRAIN_ARGS=(
    "${MODEL_ARGS[@]}"
    --hf-checkpoint "${HF_CKPT}"
    --megatron-to-hf-mode bridge
    --load "${SAVE_DIR}/${EXP_NAME}"
    --save "${SAVE_DIR}/${EXP_NAME}"
    "${SAVE_ARGS[@]}"
    --max-actor-ckpt-to-keep "${MAX_ACTOR_CKPT_TO_KEEP:-1}"
    "${RESUME_ARGS[@]}"
    --resource "{\"actor\": [1, ${GPU_COUNT}], \"sft\": [1, 0]}"
    --num-gpus-per-node 8
    --num-data-storage-units "${NUM_NODES}"
    --max-staleness 0
    --loss-type sft
    --prompt-data "${PROMPT_DATA}"
    --input-key messages
    --tool-key tools
    --multimodal-keys '{"image":"images"}'
    --apply-chat-template-kwargs '{"thinking_mode":"thinking","preserve_thinking":true}'
    "${LENGTH_ARGS[@]}"
    --global-batch-size "${GLOBAL_BATCH_SIZE}"
    --seq-length "${SEQ_LENGTH}"
    --max-tokens-per-gpu "${MAX_TOKENS_PER_GPU}"
    --data-pad-size-multiplier "${DATA_PAD_SIZE_MULTIPLIER:-128}"
    --sft-oversize-strategy skip
    --sft-max-in-flight-steps 2
    "${EVAL_ARGS[@]}"
    --balance-data
    --qkv-format thd
    --use-dynamic-batch-size
    --tensor-model-parallel-size "${TP_SIZE}"
    --pipeline-model-parallel-size "${PP_SIZE}"
    --context-parallel-size "${CP_SIZE}"
    --expert-model-parallel-size "${EP_SIZE}"
    --expert-tensor-parallel-size 1
    --sequence-parallel
    --allgather-cp
    --cp-partition-mode contiguous
    --calculate-per-token-loss
    --empty-unused-memory-level 1
    --recompute-granularity full
    --recompute-method uniform
    --recompute-num-layers 1
    --recompute-loss-function
    --log-probs-chunk-size 4096
    "${DISPATCH_ARGS[@]}"
    --optimizer adam
    --lr "${LR:-1e-6}"
    --lr-decay-style constant
    --weight-decay 0.1
    --adam-beta1 0.9
    --adam-beta2 0.98
    --adam-eps 1e-8
    --clip-grad 1.0
    --optimizer-cpu-offload
    --overlap-cpu-optimizer-d2h-h2d
    --use-precision-aware-optimizer
    --bf16
    --transformer-impl transformer_engine
    --attention-backend flash
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --attention-softmax-in-fp32
    --accumulate-allreduce-grads-in-fp32
    --seed "${SEED:-1234}"
    --trust-remote-code
    --log-interval 1
    --use-clearml
    --use-metrics-service
    "${METRIC_ARGS[@]}"
    --tb-project-name "${PROJECT_NAME:-Relax/sft/opensearch-vl}"
    --tb-experiment-name "${RUN_NAME}"
    "$@"
)
if [[ "${DRY_RUN:-0}" == 1 ]]; then
    printf '%q ' python3 -m relax.entrypoints.train "${TRAIN_ARGS[@]}"
    printf '\n'
    exit 0
fi
if [[ -z "${RELAX_ENTRYPOINT_MODE:-}" ]]; then
    source "${RELAX_ROOT}/scripts/entrypoint/local.sh"
fi
# The metrics runtime environment can contain credentials.
set +x
RUNTIME_ENV_JSON=$(python3 - <<'PY'
import json
import os

runtime = json.loads(os.environ.get("RUNTIME_ENV_JSON", "{}"))
env = runtime.setdefault("env_vars", {})
if "SWANLAB_EXP_NAME" in os.environ and os.environ.get("ENABLE_SWANLAB") == "1":
    runtime["worker_process_setup_hook"] = "examples.opensearch_vl.tools.swanlab_hook.install"
    env.update({k: v for k, v in os.environ.items() if k.startswith("SWANLAB_")})
for key, default in (("CUDA_DEVICE_MAX_CONNECTIONS", "1"), ("NCCL_NVLS_ENABLE", "0"),
                     ("RELAX_PROCESSOR_POOL_SIZE", "8"), ("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True"),
                     ("MEGATRON_SYNC_SAVE_BOUNDED_STAGING", "1"), ("MEGATRON_SYNC_SAVE_STAGE_BYTES", "0")):
    env[key] = os.environ.get(key, env.get(key, default))
print(json.dumps(runtime))
PY
)
mkdir -p "${SAVE_DIR}/${EXP_NAME}/logs"
ray job submit \
    ${RAY_NO_WAIT:+--no-wait} \
    --address="${RAY_JOB_ADDRESS:-${RAY_DASHBOARD_ADDRESS:-http://127.0.0.1:8265}}" \
    ${WORKING_DIR:+--working-dir "${WORKING_DIR}"} \
    --runtime-env-json="${RUNTIME_ENV_JSON}" \
    -- python3 -m relax.entrypoints.train \
    "${TRAIN_ARGS[@]}" \
    2>&1 | tee "${SAVE_DIR}/${EXP_NAME}/logs/submit-$(date +%Y%m%d-%H%M%S).log"
