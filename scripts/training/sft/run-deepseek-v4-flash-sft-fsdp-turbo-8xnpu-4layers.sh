#!/bin/bash

# Copyright (c) 2026 Relax Authors. All Rights Reserved.
#
# DeepSeek-V4-Flash (4L/16E) SFT smoke, 8xNPU / 8xCUDA-GPU single-node
# (auto-detected), ray-submit launch, FSDPTurbo training backend.
#
# Usage (from the Relax repo root):
#   bash scripts/training/sft/run-deepseek-v4-flash-sft-fsdp-turbo-8xnpu-4layers.sh
#
# Structure mirrors run-qwen3.8-flash-next-sft-fsdp-turbo-8xnpu-4layers.sh
# (Relax Ray orchestration).
# Differences:
#   - Model: DeepSeek-V4-Flash 4L/16E — the fsdp_turbo actor routes
#     model_type==deepseek_v4 to build_deepseek_v4_model() (FSDPTurbo
#     adapter) and loads the real released weights when HF_CHECKPOINT
#     points at a safetensors directory (load_weights_into).
#   - fsdp_turbo package: set FSDPTURBO_DIR to inject a source tree via
#     PYTHONPATH ahead of any pip-installed fsdp_turbo.
#   - Turbo plan: configs/turbo_config/deepseek_v4_4l16e.yaml
#     (FSDP=8, EP off in v1, CP=1, recompute on).
#
# Environment overrides:
#   FSDPTURBO_DIR   # fsdp_turbo source tree (optional; the NPU image ships one)
#   TURBO_CONFIG    # turbo yaml (default: deepseek_v4_4l16e.yaml)
#   HF_CHECKPOINT   # DeepSeek-V4-Flash HF checkpoint dir or safetensors dir
#   TRAIN_FILES     # training jsonl paths (messages format, comma-joined)

set -ex
set -o pipefail
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
ulimit -n 65535

now=$(date "+%Y-%m-%d-%H-%M-%S")
echo 当前时间:
# --- Hardware auto-detect (mirrors mjy run.sh L11-19): NPU when the Ascend
# toolkit is present, CUDA GPU otherwise. One script, both platforms. ---
if [[ -f /usr/local/Ascend/ascend-toolkit/set_env.sh ]]; then
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
    export HCCL_SOCKET_IFNAME=enp48s3u1u1
    export GLOO_SOCKET_IFNAME=enp48s3u1u1
    export TP_SOCKET_IFNAME=enp48s3u1u1
    export ASCEND_COREDUMP_SIGNAL=none
    export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
    export HCCL_HOST_SOCKET_PORT_RANGE=63000-63150
    export HCCL_NPU_SOCKET_PORT_RANGE=64000-64150
    # Match the mjy torchrun.sh reference env (see torchrun.sh L18-22):
    # - huge aclnn op-cache for 512 experts + GDN + QSA kernels
    # - expandable segments against NPU HBM fragmentation (needed by cpu_offload)
    # - long HCCL connect timeout: ranks finish from_pretrained at very different
    #   times on cold page cache (observed 15+ min skew), default 120s would abort
    export ACLNN_CACHE_LIMIT=100000
    export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
    export HCCL_CONNECT_TIMEOUT=7200
else
    # CUDA GPU: NCCL/gloo pick their own socket interface on a single node —
    # only export NCCL/GLOO/TP_SOCKET_IFNAME=<nic> yourself for multi-node.
    export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
    # Same expandable-segments rationale as the NPU branch: guards against VRAM
    # fragmentation under fsdp cpu_offload.
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
fi
export RAY_DEDUP_LOGS=0
export PYTHONBUFFERED=1

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"

# --- FSDPTurbo source: inject a source tree only when FSDPTURBO_DIR is set ---
# Exported BEFORE the entrypoint is sourced so local-npu.sh / ray-job-npu.sh
# embed it into the job's RUNTIME_ENV_JSON PYTHONPATH. PYTHONPATH precedes
# site-packages, so this shadows any pip-installed fsdp_turbo. The path must
# exist on every Ray node (shared mount). The NPU image already ships a
# patched fsdp_turbo (docker/Dockerfile.npu), so this stays empty by default.
if [ -n "${FSDPTURBO_DIR:-}" ]; then
    export PYTHONPATH="${FSDPTURBO_DIR}:${PYTHONPATH:-}"
    echo "FSDPTURBO_DIR=$FSDPTURBO_DIR"
fi

# Auto-source local environment when not launched via an external entrypoint
# (local-npu.sh: Ray head with NPU visibility; local.sh: Ray head with
# --num-gpus, auto-counted from CUDA_VISIBLE_DEVICES).
if [ -z "${RELAX_ENTRYPOINT_MODE:-}" ]; then
    if [[ -f /usr/local/Ascend/ascend-toolkit/set_env.sh ]]; then
        source "${SCRIPT_DIR}/../../entrypoint/local-npu.sh"
    else
        source "${SCRIPT_DIR}/../../entrypoint/local.sh"
    fi
fi

PROJECT_NAME="${PROJECT_NAME:=Relax/sft/deepseek-v4-4l16e}"
EXP_NAME=deepseek-v4-4l16e-sft-fsdp-turbo-npu8-smoke

# data files (parquet/jsonl, OpenAI-messages or ShareGPT conversations).
# Override these to point at your dataset / model directories.
DATA_DIR="${DATA_DIR:-/path/to/datasets}"
MODEL_DIR="${MODEL_DIR:-/path/to/models}"
HF_CHECKPOINT=${HF_CHECKPOINT:-${MODEL_DIR}/DeepSeek-V4-Flash}
SAVE_DIR="${SAVE_DIR:-${MODEL_DIR}/DeepSeek-V4-Flash-save-sft-fsdp-turbo}"
TRAIN_FILES=(
    "'${DATA_DIR}/llava_text_10k.jsonl'"
)
PROMPT_DATA="[$(IFS=,; echo "${TRAIN_FILES[*]}")]"

TURBO_CONFIG="${TURBO_CONFIG:-${SCRIPT_DIR}/../../../configs/turbo_config/deepseek_v4_4l16e.yaml}"

BACKEND_ARGS=(
   --train-backend fsdp_turbo
   --turbo-config ${TURBO_CONFIG}
   --max-tokens-per-gpu 8192
   --use-dynamic-batch-size
)

CKPT_ARGS=(
   --hf-checkpoint ${HF_CHECKPOINT}
   --save ${SAVE_DIR}
   # --load ${SAVE_DIR}
   --save-interval 100
   --num-epoch 1
)

SFT_ARGS=(
   --loss-type sft
   --prompt-data "${PROMPT_DATA}"
   --input-key messages
   --global-batch-size 64
   --sft-prefetch-num-workers 16
   --sft-prefetch-buffer-size 512
)

EVAL_ARGS=(
    --eval-size 0.1
    --eval-interval 20
)

MISC_ARGS=(
   --colocate
   --use-health-check
)

mkdir -p log

ray job submit ${RAY_NO_WAIT:+--no-wait} --address="http://${MASTER_ADDR}:8265" \
   ${WORKING_DIR:+--working-dir "${WORKING_DIR}"} \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 -m relax.entrypoints.train \
   --resource '{"sft": [1, 0], "actor": [1, 8]}' \
   --sft-max-in-flight-steps 4 \
   --num-data-storage-units 8 \
   "${BACKEND_ARGS[@]}" \
   "${CKPT_ARGS[@]}" \
   "${SFT_ARGS[@]}" \
   "${EVAL_ARGS[@]}" \
   "${MISC_ARGS[@]}"  2>&1 | tee log/${EXP_NAME}-${now}.log
