#!/usr/bin/env bash
# Copyright (c) 2026 Relax Authors. All Rights Reserved.

# Defaults to a four-layer image SFT smoke run on 8 GPUs. Use SFT_INPUT=text for text-only JSONL.
# Usage: bash run-mimo-v2.6-flash-vision-sft.sh BF16_CHECKPOINT MESSAGES_JSONL SAVE_DIR [train args...]
set -eo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
HF_CHECKPOINT=${1:?provide the BF16 checkpoint directory}
PROMPT_DATA=${2:?provide a messages dataset}
SAVE_DIR=${3:?provide the checkpoint output directory}
shift 3

source "${MODEL_CONFIG_DIR:-${SCRIPT_DIR}/../../models}/mimo-v2.6-flash.sh"
if [[ "$layers" != "${EXPECTED_LAYERS:-4}" ]]; then
    echo "Expected ${EXPECTED_LAYERS:-4} decoder layers, found $layers in $HF_CHECKPOINT/config.json" >&2
    exit 1
fi

case "${SFT_INPUT:-image}" in
    image)
        SEQ_LENGTH=${SEQ_LENGTH:-40960}
        INPUT_ARGS=(
            --multimodal-keys '{"image":"images"}' --tool-key tools
            --image-max-token-num 1024 --sft-oversize-strategy skip
            --sft-chunked-logits --recompute-loss-function
            --recompute-granularity full --recompute-method uniform --recompute-num-layers 1
        )
        ;;
    text)
        SEQ_LENGTH=${SEQ_LENGTH:-1024}
        INPUT_ARGS=()
        ;;
    *) echo "SFT_INPUT must be image or text" >&2; exit 1 ;;
esac
NODES=${NODES:-1}
NUM_GPUS=${GPUS_PER_NODE:-${NUM_GPUS:-8}}
TP=${TP:-4}
PP=${PP:-1}
EP=${EP:-4}

CKPT_ARGS=(
    --hf-checkpoint "$HF_CHECKPOINT"
    --ref-load "$HF_CHECKPOINT"
    --megatron-to-hf-mode bridge
    --load "$SAVE_DIR"
    --save "$SAVE_DIR"
    --save-interval "${SAVE_INTERVAL:-100}"
)
SFT_ARGS=(
    --loss-type sft
    --prompt-data "$PROMPT_DATA"
    --input-key messages
    --num-rollout "${STEPS:-1}"
    --global-batch-size "${GLOBAL_BATCH_SIZE:-2}"
    --qkv-format thd
    --use-dynamic-batch-size
    --seq-length "$SEQ_LENGTH"
    --max-tokens-per-gpu "${MAX_TOKENS_PER_GPU:-$SEQ_LENGTH}"
    --sft-max-in-flight-steps "${SFT_MAX_IN_FLIGHT_STEPS:-1}"
    --num-data-storage-units "${DATA_STORAGE_UNITS:-1}"
)
PERF_ARGS=(
    --tensor-model-parallel-size "$TP"
    --pipeline-model-parallel-size "$PP"
    --expert-model-parallel-size "$EP"
    --expert-tensor-parallel-size 1
    --context-parallel-size "${CP:-1}"
    --attention-backend fused
    --attention-softmax-in-fp32
)
if (( TP > 1 )); then
    PERF_ARGS+=(--sequence-parallel)
fi
OPTIMIZER_ARGS=(
    --optimizer adam
    --lr 1e-5
    --lr-decay-style "${LR_DECAY_STYLE:-constant}"
    --weight-decay "${WEIGHT_DECAY:-0.01}"
    --adam-beta1 0.9
    --adam-beta2 "${ADAM_BETA2:-0.999}"
    --optimizer-cpu-offload
    --optimizer-offload-fraction 1.0
    --use-precision-aware-optimizer
    --accumulate-allreduce-grads-in-fp32
)
TRAIN_CMD=(
    "${PYTHON:-python3}" -m relax.entrypoints.train
    --resource "{\"sft\":[1,0],\"actor\":[1,$((NODES * NUM_GPUS))]}"
    --num-gpus-per-node "$NUM_GPUS"
    "${MODEL_ARGS[@]}" "${CKPT_ARGS[@]}" "${SFT_ARGS[@]}" "${INPUT_ARGS[@]}" "${PERF_ARGS[@]}" "${OPTIMIZER_ARGS[@]}" "$@"
)

if [[ "${DRY_RUN:-0}" == 1 ]]; then
    printf '%q ' "${TRAIN_CMD[@]}"
    printf '\n'
    exit 0
fi

# A script used as the entrypoint of an existing Ray Job is already on the driver.
if [[ -n "${RAY_JOB_CONFIG_JSON_ENV_VAR:-}" ]]; then
    exec "${TRAIN_CMD[@]}"
fi
if (( NODES > 1 )) && [[ -z "${RELAX_ENTRYPOINT_MODE:-}" ]]; then
    if [[ -z "${RAY_ADDRESS:-}" ]] || ! timeout 5 ray status --address="$RAY_ADDRESS" >/dev/null 2>&1; then
        echo "Multi-node MiMo SFT requires a running Ray cluster. Set RAY_ADDRESS to its GCS address or launch with scripts/entrypoint/spmd-multinode.sh." >&2
        exit 1
    fi
    export RELAX_SKIP_CLUSTER_CLEANUP=${RELAX_SKIP_CLUSTER_CLEANUP:-1}
    exec bash "${SCRIPT_DIR}/../../entrypoint/ray-job.sh" "${BASH_SOURCE[0]}" "$HF_CHECKPOINT" "$PROMPT_DATA" "$SAVE_DIR" "$@"
fi
if (( NODES > 1 )) && [[ "${RELAX_ENTRYPOINT_MODE:-}" == local ]]; then
    echo "Multi-node MiMo SFT requires a multi-node Ray entrypoint." >&2
    exit 1
fi
if [[ -z "${RELAX_ENTRYPOINT_MODE:-}" ]]; then
    source "${SCRIPT_DIR}/../../entrypoint/local.sh"
fi

ray job submit ${RAY_NO_WAIT:+--no-wait} \
    --address="${RAY_DASHBOARD_ADDRESS:-http://127.0.0.1:8265}" \
    ${WORKING_DIR:+--working-dir "${WORKING_DIR}"} \
    --runtime-env-json="${RUNTIME_ENV_JSON}" -- "${TRAIN_CMD[@]}"
