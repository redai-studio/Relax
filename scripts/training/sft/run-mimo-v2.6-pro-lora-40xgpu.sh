#!/usr/bin/env bash
# Copyright (c) 2026 Relax Authors. All Rights Reserved.

# Full-depth MiMo-V2.6-Pro multimodal LoRA SFT on five 8-GPU Ray workers.
# Usage: bash run-mimo-v2.6-pro-lora-40xgpu.sh BF16_HF_DIR TRAIN_JSONL OUTPUT_DIR [TRAIN_ARGS...]
# NUM_EPOCH defaults to 1. Set NUM_STEPS instead for a short run.
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
    args+=(--load "$LOAD_DIR")
fi
if [[ -n ${SAVE_HF_DIR:-} ]]; then
    SAVE_HF_DIR=$(realpath -m -- "$SAVE_HF_DIR")
    export SAVE_HF_DIR
    args+=(--save-hf "$SAVE_HF_DIR")
fi

test -r "$prompt_data"
if [[ "$dry_run" != 1 ]]; then
    test -r "$checkpoint/model.safetensors.index.json"
fi

model_dims=$("$python_bin" - "$checkpoint/config.json" <<'PY'
import json
import sys
from pathlib import Path

config_path = Path(sys.argv[1])
config = json.loads(config_path.read_text())
if (
    config.get("model_type") != "mimo_v2"
    or config.get("num_hidden_layers") != 70
    or config.get("n_routed_experts") != 384
    or config.get("attention_projection_layout", "split") != "split"
    or config.get("quantization_config") is not None
):
    raise SystemExit("Use the full-depth BF16 MiMo-V2.6-Pro checkpoint with split Q/K/V.")
index_path = config_path.with_name("model.safetensors.index.json")
if index_path.is_file():
    weights = json.loads(index_path.read_text())["weight_map"]
    if any(
        f"model.layers.{layer}.self_attn.{part}_proj.weight" not in weights
        for layer in range(70)
        for part in "qkv"
    ):
        raise SystemExit("The checkpoint index must contain split Q/K/V for every decoder layer.")
keys = (
    "hidden_size", "intermediate_size", "num_attention_heads", "num_key_value_heads",
    "head_dim", "v_head_dim", "partial_rotary_factor", "rope_theta", "layernorm_epsilon",
    "vocab_size", "moe_intermediate_size", "num_experts_per_tok",
)
print(*(config[key] for key in keys), json.dumps(config["moe_layer_freq"], separators=(",", ":")))
PY
)
read -r hidden ffn heads groups qk v rotary rope epsilon vocab expert_ffn topk moe_freq <<< "$model_dims"

train_args=(
    --resource '{"sft":[1,0],"actor":[1,40]}' --num-gpus-per-node 8
    --tensor-model-parallel-size 8 --pipeline-model-parallel-size 5
    --context-parallel-size 1 --sequence-parallel
    --expert-model-parallel-size 8 --expert-tensor-parallel-size 1
    --hf-checkpoint "$checkpoint" --ref-load "$checkpoint" --megatron-to-hf-mode bridge
    --num-layers 70 --hidden-size "$hidden" --ffn-hidden-size "$ffn"
    --num-attention-heads "$heads" --group-query-attention --num-query-groups "$groups"
    --kv-channels "$qk" --v-head-dim "$v"
    --position-embedding-type rope --rotary-percent "$rotary" --rotary-base "$rope"
    --normalization RMSNorm --norm-epsilon "$epsilon" --swiglu --disable-bias-linear
    --untie-embeddings-and-output-weights --vocab-size "$vocab"
    --num-experts 384 --moe-layer-freq "$moe_freq" --moe-ffn-hidden-size "$expert_ffn"
    --moe-router-topk "$topk" --moe-router-score-function sigmoid --moe-router-enable-expert-bias
    --moe-router-bias-update-rate 0 --moe-router-load-balancing-type none
    --moe-aux-loss-coeff 0 --moe-router-dtype fp32 --moe-grouped-gemm
    --hidden-dropout 0 --attention-dropout 0
    --attention-backend fused --attention-softmax-in-fp32
    --moe-router-fusion --moe-permute-fusion
    --moe-token-dispatcher-type flex --moe-flex-dispatcher-backend deepep
    --bf16 --loss-type sft --calculate-per-token-loss --prompt-data "$prompt_data"
    --input-key messages --tool-key tools --multimodal-keys '{"image":"images"}'
    --image-max-token-num 1024 --freeze-vision-model --freeze-vision-projection
    --global-batch-size 64 --qkv-format thd --use-dynamic-batch-size
    --seq-length 32768 --max-tokens-per-gpu 32768 --data-pad-size-multiplier 4096
    --sft-oversize-strategy skip --sft-invalid-multimodal-strategy skip
    --sft-max-in-flight-steps 2 --num-data-storage-units 1 --sft-chunked-logits
    --recompute-granularity full --recompute-method uniform --recompute-num-layers 1
    --optimizer adam --lr 1e-5 --lr-decay-style constant --weight-decay 0.01
    --adam-beta1 0.9 --adam-beta2 0.999 --accumulate-allreduce-grads-in-fp32
    --optimizer-cpu-offload --optimizer-offload-fraction 1.0
    --use-precision-aware-optimizer --overlap-cpu-optimizer-d2h-h2d
    --lora-rank 16 --lora-alpha 32 --lora-dropout 0 --lora-scope all --lora-merge-mode
    --lora-target-modules linear_qkv linear_proj linear_fc1 linear_fc2
    --save "$save_dir" --save-lora-only --ckpt-format torch_dist
    --save-interval "${SAVE_INTERVAL:-50}" --max-actor-ckpt-to-keep 3 --seed "${SEED:-42}"
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
    exec bash scripts/entrypoint/ray-job.sh "$script_dir/$(basename -- "$0")" "$checkpoint" "$prompt_data" "$save_dir" "$@"
fi
exec "$python_bin" -m relax.entrypoints.train "${train_args[@]}"
