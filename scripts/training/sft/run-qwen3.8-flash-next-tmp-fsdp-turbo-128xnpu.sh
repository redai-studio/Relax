#!/bin/bash

# Copyright (c) 2026 Relax Authors. All Rights Reserved.
# Qwen3.8-Flash-Next SFT with the FSDPTurbo backend.
# Set MASTER_ADDR, HOST_IP and POD_NAME on each of the eight nodes.
# Override EXP_DIR, MODEL_DIR, DATA_DIR, SAVE_DIR and network interfaces as needed.

set -ex
set -o pipefail
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
ulimit -n 65535


now=$(date "+%Y-%m-%d-%H-%M-%S")
echo 当前时间:${now}

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
RELAX_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
TURBO_CONFIG="${TURBO_CONFIG:-${RELAX_ROOT}/configs/turbo_config/qwen3_8_flash_next.yaml}"

export HCCL_SOCKET_IFNAME="${HCCL_SOCKET_IFNAME:-enp23s0f3}"
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-enp23s0f3}"
export TP_SOCKET_IFNAME="${TP_SOCKET_IFNAME:-enp23s0f3}"
export PYTHONUNBUFFERED=1
export CPU_AFFINITY_CONF=${CPU_AFFINITY_CONF:-1}
export TASK_QUEUE_ENABLE=${TASK_QUEUE_ENABLE:-1}
export ASCEND_LAUNCH_BLOCKING=0
export ASCEND_COREDUMP_SIGNAL=none
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15}"
export HCCL_HOST_SOCKET_PORT_RANGE="${HCCL_HOST_SOCKET_PORT_RANGE:-63000-63150}"
export HCCL_NPU_SOCKET_PORT_RANGE="${HCCL_NPU_SOCKET_PORT_RANGE:-22000,22999}"
export ACLNN_CACHE_LIMIT=100000
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export HCCL_CONNECT_TIMEOUT=7200
export RAY_memory_usage_threshold=0.97
export RELAX_SFT_ROPE_FP32=1
export RELAX_SFT_ROUTER_FP32=1

export WORLD_SIZE=8 NUM_NPUS=16

# Auto-source local environment when not launched via an external entrypoint
if [ -z "${RELAX_ENTRYPOINT_MODE:-}" ]; then
    source "${SCRIPT_DIR}/../../entrypoint/local-npu-multinode.sh"
fi

if [ "${RELAX_ENTRYPOINT_MODE:-}" = "ray-job" ]; then
RUNTIME_ENV_JSON=$(python3 - <<'PYENV'
import json
import os

runtime = json.loads(os.environ.get("RUNTIME_ENV_JSON", "{}"))
env = runtime.setdefault("env_vars", {})
for key in ("HCCL_SOCKET_IFNAME", "GLOO_SOCKET_IFNAME", "TP_SOCKET_IFNAME",
            "HCCL_HOST_SOCKET_PORT_RANGE", "HCCL_NPU_SOCKET_PORT_RANGE"):
    if key in os.environ:
        env[key] = os.environ[key]
print(json.dumps(runtime))
PYENV
)
export RUNTIME_ENV_JSON
fi

PROJECT_NAME="${PROJECT_NAME:=Relax/sft/qwen3.8}"
EXP_DIR="${EXP_DIR:-${SCRIPT_DIR}/../../../../exps}"
MODEL_DIR="${MODEL_DIR:-${EXP_DIR}}"
DATA_DIR="${DATA_DIR:-${EXP_DIR}}"
HF_CHECKPOINT="${HF_CHECKPOINT:-${MODEL_DIR}/Qwen3.8-Flash-Next}"
SAVE_DIR="${SAVE_DIR:-${EXP_DIR}/Qwen3.8-Flash-Next-save-sft-fsdp-turbo}"
EXP_NAME="${EXP_NAME:-qwen3.8-flash-next-sft-fsdp-turbo-128xnpu}"
LOG_DIR="${LOG_DIR:-${EXP_DIR}/logs}"
mkdir -p "${LOG_DIR}"
LOG_PATH="${LOG_DIR}/${EXP_NAME}-${now}-${RANK:-0}.log"

echo "==============" 2>&1 | tee "$LOG_PATH"
cat "${BASH_SOURCE[0]}" 2>&1 | tee -a "$LOG_PATH"
echo "==============" 2>&1 | tee -a "$LOG_PATH"
cat "$TURBO_CONFIG" 2>&1 | tee -a "$LOG_PATH"
echo "==============" 2>&1 | tee -a "$LOG_PATH"


TRAIN_FILES=()
for f in "${DATA_DIR}"/*.parquet; do
    TRAIN_FILES+=("'${f}'")
done
PROMPT_DATA="[$(IFS=,; echo "${TRAIN_FILES[*]}")]"

BACKEND_ARGS=(
   --train-backend fsdp_turbo
   --turbo-config "${TURBO_CONFIG}"
   --max-tokens-per-gpu 32768
)

CKPT_ARGS=(
   --hf-checkpoint "${HF_CHECKPOINT}"
   --save "${SAVE_DIR}"
   ${LOAD_PATH:+--load "${LOAD_PATH}"}
   --save-interval "${SAVE_INTERVAL:-5000}"
   # --num-rollout 1000
   --num-epoch 3
)

[ "${FREEZE_INDEXER_VISUAL:-1}" != "0" ] && FREEZE_PARITY_ARGS=(
    --freeze-params-name-list \
    "'(^|\.)self_attn\.indexer\.'" \
    "'(^|\.)visual\.'" \
    "'(^|\.)ngram_embedding\.'"
)

SFT_ARGS=(
   --loss-type sft
   --prompt-data "${PROMPT_DATA}"
   --input-key messages
   --apply-chat-template
   --apply-chat-template-kwargs '{"add_generation_prompt":false,"enable_thinking":false,"preserve_thinking":true,"add_non_thinking_prefix":true}'
   --multimodal-keys '{"image":"images"}'
   --image-max-token-num "${IMAGE_MAX_TOKEN_NUM:-256}"
   --global-batch-size 32
   --sft-invalid-multimodal-strategy skip
   --rollout-max-context-len 16384
   --calculate-per-token-loss
   --seq-length 32768
   --use-dynamic-batch-size
   --sft-oversize-strategy skip
   --balance-data
   --per-rank-fetch
   --sft-prefetch-num-workers 16
   --data-pad-size-multiplier 128
   --sft-train-data-prefetch
   --sft-prefetch-buffer-size 2048
   --seed 1234
   "${FREEZE_PARITY_ARGS[@]}"
)

EVAL_ARGS=(
    #--eval-size 0.1
    #--eval-interval 20
)

MISC_ARGS=(
   --colocate
   # --use-health-check
)

ray job submit ${RAY_NO_WAIT:+--no-wait} --address="http://${MASTER_ADDR}:8265" \
   ${WORKING_DIR:+--working-dir "${WORKING_DIR}"} \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 -m relax.entrypoints.train \
   --resource '{"sft": [1, 0], "actor": [1, 128]}' \
   --nnodes 8 \
   --num-gpus-per-node 16 \
   --sft-max-in-flight-steps 4 \
   --num-data-storage-units 8 \
   "${BACKEND_ARGS[@]}" \
   "${CKPT_ARGS[@]}" \
   "${SFT_ARGS[@]}" \
   "${EVAL_ARGS[@]}" \
   "${MISC_ARGS[@]}"  2>&1 | tee "${LOG_PATH}"
