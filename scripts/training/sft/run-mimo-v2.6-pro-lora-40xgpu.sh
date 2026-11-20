#!/usr/bin/env bash
# Copyright (c) 2026 Relax Authors. All Rights Reserved.

# Full-depth MiMo-V2.6-Pro multimodal LoRA SFT on five 8-GPU Ray workers.
# Usage: bash run-mimo-v2.6-pro-lora-40xgpu.sh BF16_HF_DIR TRAIN_JSONL OUTPUT_DIR [TRAIN_ARGS...]
# NUM_EPOCH defaults to 1. Set NUM_STEPS instead for a short run.
# Restores OUTPUT_DIR by default; LOAD_DIR explicitly overrides the checkpoint source.
set -euo pipefail

if (( $# < 3 )); then
    printf 'Usage: %s BF16_HF_DIR TRAIN_JSONL OUTPUT_DIR [TRAIN_ARGS...]\n' "$0" >&2
    exit 2
fi

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
relax_root=$(cd -- "$script_dir/../../.." && pwd)
checkpoint=$(realpath -e -- "$1")
prompt_data=$(realpath -e -- "$2")
save_dir=$(realpath -m -- "$3")
shift 3

python_bin=${PYTHON:-python3}
dry_run=${DRY_RUN:-0}
args=()
if [[ -n ${NUM_STEPS:-} ]]; then
    if [[ -n ${NUM_EPOCH:-} ]]; then
        printf 'Set only one of NUM_STEPS and NUM_EPOCH.\n' >&2
        exit 2
    fi
    args+=(--num-rollout "$NUM_STEPS")
else
    args+=(--num-epoch "${NUM_EPOCH:-1}")
fi
if [[ -n ${LOAD_DIR:-} ]]; then
    LOAD_DIR=$(realpath -e -- "$LOAD_DIR")
    export LOAD_DIR
fi
args+=(--load "${LOAD_DIR:-$save_dir}")
if [[ -n ${SAVE_HF_DIR:-} ]]; then
    SAVE_HF_DIR=$(realpath -m -- "$SAVE_HF_DIR")
    export SAVE_HF_DIR
    args+=(--save-hf "$SAVE_HF_DIR")
fi

test -r "$prompt_data"
if [[ "$dry_run" != 1 ]]; then
    test -r "$checkpoint/model.safetensors.index.json"
fi

source "$relax_root/scripts/models/mimo-v2.6-pro.sh" "$checkpoint" "$python_bin"

train_args=(
    --resource '{"sft":[1,0],"actor":[1,40]}'
    --num-gpus-per-node 8

    --tensor-model-parallel-size 8
    --pipeline-model-parallel-size 5
    --context-parallel-size 1
    --sequence-parallel
    --expert-model-parallel-size 8
    --expert-tensor-parallel-size 1

    --hf-checkpoint "$checkpoint"
    --ref-load "$checkpoint"
    --megatron-to-hf-mode bridge
    "${MODEL_ARGS[@]}"

    --attention-backend fused
    --attention-softmax-in-fp32
    --moe-router-fusion
    --moe-permute-fusion
    --moe-token-dispatcher-type flex
    --moe-flex-dispatcher-backend deepep

    --bf16
    --loss-type sft
    --calculate-per-token-loss
    --prompt-data "$prompt_data"
    --input-key messages
    --tool-key tools
    --multimodal-keys '{"image":"images"}'
    --image-max-token-num 1024
    --freeze-vision-model
    --freeze-vision-projection

    --global-batch-size 64
    --qkv-format thd
    --use-dynamic-batch-size
    --seq-length 32768
    --max-tokens-per-gpu 32768
    --data-pad-size-multiplier 4096
    --sft-oversize-strategy skip
    --sft-invalid-multimodal-strategy skip
    --sft-max-in-flight-steps 2
    --num-data-storage-units 1
    --sft-chunked-logits

    --recompute-granularity full
    --recompute-method uniform
    --recompute-num-layers 1

    --optimizer adam
    --lr 1e-5
    --lr-decay-style constant
    --weight-decay 0.01
    --adam-beta1 0.9
    --adam-beta2 0.999
    --accumulate-allreduce-grads-in-fp32
    --optimizer-cpu-offload
    --optimizer-offload-fraction 1.0
    --use-precision-aware-optimizer
    --overlap-cpu-optimizer-d2h-h2d

    --lora-rank 16
    --lora-alpha 32
    --lora-dropout 0
    --lora-scope all
    --lora-merge-mode
    --lora-target-modules linear_qkv linear_proj linear_fc1 linear_fc2

    --save "$save_dir"
    --save-lora-only
    --ckpt-format torch_dist
    --save-interval "${SAVE_INTERVAL:-50}"
    --max-actor-ckpt-to-keep 3
    --seed "${SEED:-42}"
    "${args[@]}"
    "$@"
)

if [[ "$dry_run" == 1 ]]; then
    printf '%q ' "$python_bin" -m relax.entrypoints.train "${train_args[@]}"
    printf '\n'
    exit 0
fi

export RELAX_MIMO_V2_FUSED_ROUTED_SWIGLU=1
export RELAX_SFT_SELECTIVE_LOGITS=1
export RELAX_LORA_SHARE_EXPERT_ADAPTERS=false
export RELAX_SKIP_CLUSTER_CLEANUP=${RELAX_SKIP_CLUSTER_CLEANUP:-1}
export RELAX_PROPAGATE_ENV_VARS="${RELAX_PROPAGATE_ENV_VARS:+$RELAX_PROPAGATE_ENV_VARS,}RELAX_MIMO_V2_FUSED_ROUTED_SWIGLU,RELAX_SFT_SELECTIVE_LOGITS,RELAX_LORA_SHARE_EXPERT_ADAPTERS"

cd "$relax_root"
if [[ -z ${RELAX_ENTRYPOINT_MODE:-} ]]; then
    exec bash scripts/entrypoint/ray-job.sh \
        "$script_dir/$(basename -- "$0")" "$checkpoint" "$prompt_data" "$save_dir" "$@"
fi
exec "$python_bin" -m relax.entrypoints.train "${train_args[@]}"
