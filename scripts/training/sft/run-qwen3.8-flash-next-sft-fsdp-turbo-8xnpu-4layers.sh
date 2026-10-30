#!/bin/bash

# Copyright (c) 2026 Relax Authors. All Rights Reserved.
#
# Qwen3.8-Flash-Next (qwen4_exp) SFT smoke, 8xNPU / 8xCUDA-GPU single-node
# (auto-detected), ray-submit
# launch, FSDPTurbo training backend (mjy/FSDPTurbo source tree).
#
# Usage:
#   bash scripts/training/sft/run-qwen3.8-flash-next-sft-fsdp-turbo-8xnpu-4layers.sh
#
# Structure mirrors sft-fsdp-turbo.sh (Relax Ray orchestration). Differences:
#   - Model: Qwen3.8-Flash-Next — the fsdp_turbo actor detects qwen4_exp
#     configs and applies the full patch chain (QSA triton indexer, GDN/NPU
#     fused ops, MTP head, PLE n-gram CPU offload) before FSDPTurbo wrapping.
#   - fsdp_turbo package: the mjy/FSDPTurbo deployment (qwen3_8 support),
#     injected via PYTHONPATH ahead of any pip-installed fsdp_turbo —
#     override FSDPTURBO_DIR if it lives elsewhere on the cluster.
#   - Turbo plan: configs/turbo_config/qwen3_8_flash_next_4layers_8npu.yaml
#     (FSDP=8 + EP=8, 4-layer smoke, MTP on). For the full 64-NPU run, copy
#     the yaml and raise the parallel sizes / layer count.
#   - HF checkpoint and training data paths are placeholders — fill the TODOs
#     below before the first real run.
#
# Environment overrides:
#   FSDPTURBO_DIR   # mjy/FSDPTurbo deployment (default: shared mount path)
#   TURBO_CONFIG    # turbo yaml (default: qwen3_8_flash_next_4layers_8npu.yaml)
#   HF_CHECKPOINT   # Qwen3.8-Flash-Next HF checkpoint dir
#   TRAIN_FILE_1..N # training parquet/jsonl files (quoted, comma-joined)

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
    # export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
fi
export RAY_DEDUP_LOGS=0
export PYTHONBUFFERED=1
export NGRAM_VOCAB_SIZE_BASE=2000000

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"

# --- FSDPTurbo source: the mjy tree (qwen3_8 / qwen4_exp support) ---
# Exported BEFORE the entrypoint is sourced so local-npu.sh / ray-job-npu.sh
# embed it into the job's RUNTIME_ENV_JSON PYTHONPATH. PYTHONPATH precedes
# site-packages, so this shadows any pip-installed fsdp_turbo. The path must
# exist on every Ray node (shared mount).

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

PROJECT_NAME="${PROJECT_NAME:=Relax/sft/qwen3.8}"
EXP_NAME=qwen3.8-flash-next-sft-fsdp-turbo-npu8-smoke

# data files (parquet/jsonl, OpenAI-messages or ShareGPT conversations).
# Override these to point at your dataset / model directories.
DATA_DIR="${DATA_DIR:-/path/to/datasets}"
MODEL_DIR="${MODEL_DIR:-/path/to/models}"
HF_CHECKPOINT=${HF_CHECKPOINT:-${MODEL_DIR}/Qwen3.8-Flash-Next}
SAVE_DIR="${SAVE_DIR:-${MODEL_DIR}/Qwen3.8-Flash-Next-save-sft-fsdp-turbo}"
TRAIN_FILES=(
    "'${DATA_DIR}/llava_instruct_150k_fsdp_turbo_10k.jsonl'"
)
PROMPT_DATA="[$(IFS=,; echo "${TRAIN_FILES[*]}")]"

TURBO_CONFIG="${TURBO_CONFIG:-${SCRIPT_DIR}/../../../configs/turbo_config/qwen3_8_flash_next_4layers_8npu.yaml}"

BACKEND_ARGS=(
   --train-backend fsdp_turbo
   --turbo-config ${TURBO_CONFIG}
   --max-tokens-per-gpu 8192
)

CKPT_ARGS=(
   --hf-checkpoint ${HF_CHECKPOINT}
   --save ${SAVE_DIR}
   # --load ${SAVE_DIR}
   --save-interval 100
   --num-epoch 10
)

# Token-mean SFT loss (--calculate-per-token-loss), matching every megatron
# SFT recipe under scripts/training/sft: gradients and the logged train/loss
# follow mcore's per-token semantics. Set CALC_PER_TOKEN_LOSS=0 to fall back
# to the legacy sample-mean reduction for A/B comparison.
[ "${CALC_PER_TOKEN_LOSS:-1}" != "0" ] && PER_TOKEN_LOSS_FLAG=(--calculate-per-token-loss)

# Freeze parity with the megatron qwen3.8 recipe (run-4l-pokemon FREEZE_ARGS:
# '.*\.self_attention\.indexer\..*' 'visual\.' + the six PLE projections).
# Dump re-evaluation (loss_mcore/reeval_report.txt) shows the megatron freeze
# DID take effect there: vision tower / aligner ([1152,3,2,16,16] conv,
# merger fc1/fc2) and indexer run no-grad, and neither is in the adamw
# buckets (param census 1.512B vs 1.607B total). On the HF side indexer /
# vision receive no gradient anyway (hard top-k; precomputed
# multimodal_train_inputs), so the ONLY effective trainable-set delta is the
# six PLE projections (megatron freezes them, fsdp_turbo trains them). HF
# names: 'self_attn.indexer.*', 'model.visual.*', '...ple.key_proj.weight'
# etc — passed with re.search semantics like mcore's freeze_model_params;
# the startup log prints the frozen/trainable split to confirm the patterns
# matched. Set FREEZE_INDEXER_VISUAL=0 to keep PLE trainable (old behavior).
[ "${FREEZE_INDEXER_VISUAL:-1}" = "1" ] && FREEZE_PARITY_ARGS=(
   --freeze-params-name-list 'self_attn\.indexer\.' '^model\.visual\.'
   'ple\.key_proj\.weight' 'ple\.value_proj\.weight'
   'ple\.norm_key\.weight' 'ple\.norm_query\.weight'
   'ple\.norm_conv\.weight' 'ple\.conv1d\.weight'
)

SFT_ARGS=(
   --loss-type sft
   --prompt-data "${PROMPT_DATA}"
   --input-key messages
   --multimodal-keys '{"image":"images"}'
   --conversation-key-map '{"from":"role","value":"content","human":"user","gpt":"assistant"}'
   --global-batch-size 64
   --sft-prefetch-num-workers 16
   --sft-prefetch-buffer-size 512
   "${PER_TOKEN_LOSS_FLAG[@]}"
   "${FREEZE_PARITY_ARGS[@]}"
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
