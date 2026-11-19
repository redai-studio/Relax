#!/usr/bin/env bash
# Copyright (c) 2026 Relax Authors. All Rights Reserved.

# Ray job entrypoint for a MiMo-V2.6-Flash BF16 text checkpoint and messages JSONL.
set -euo pipefail
model=${1:?provide the converted BF16 checkpoint directory}
data=${2:?provide a messages dataset}
save=${3:?provide the checkpoint output directory}
shift 3
python=${PYTHON:-python3}
read -r layers experts moe_freq < <("$python" - "$model/config.json" <<'PY'
import json
import sys

with open(sys.argv[1]) as f:
    config = json.load(f)
print(config["num_hidden_layers"], config["n_routed_experts"], json.dumps(config["moe_layer_freq"], separators=(",", ":")))
PY
)
nodes=${NODES:-1}
gpus=${GPUS_PER_NODE:-4}
exec "$python" -m relax.entrypoints.train \
    --resource "{\"sft\":[1,0],\"actor\":[$nodes,$gpus]}" --num-gpus-per-node "$gpus" \
    --hf-checkpoint "$model" --ref-load "$model" --megatron-to-hf-mode bridge --save "$save" --save-interval "${SAVE_INTERVAL:-100}" \
    --loss-type sft --prompt-data "$data" --input-key messages --num-rollout "${STEPS:-1}" \
    --global-batch-size "${GLOBAL_BATCH_SIZE:-2}" --use-dynamic-batch-size \
    --max-tokens-per-gpu "${MAX_TOKENS_PER_GPU:-1024}" --sft-max-in-flight-steps 1 \
    --num-data-storage-units 1 --tensor-model-parallel-size "${TP:-2}" --sequence-parallel \
    --pipeline-model-parallel-size "${PP:-1}" --expert-model-parallel-size "${EP:-4}" \
    --expert-tensor-parallel-size 1 --attention-backend fused --attention-softmax-in-fp32 \
    --num-layers "$layers" --hidden-size 4096 --ffn-hidden-size 16384 --num-attention-heads 64 \
    --group-query-attention --num-query-groups 4 --kv-channels 192 --v-head-dim 128 \
    --position-embedding-type rope --rotary-percent 0.334 --rotary-base 10000000 --no-rope-fusion \
    --normalization RMSNorm --norm-epsilon 1e-6 --swiglu --disable-bias-linear \
    --untie-embeddings-and-output-weights --vocab-size 152576 --seq-length "${SEQ_LENGTH:-1024}" \
    --num-experts "$experts" --moe-layer-freq "$moe_freq" --moe-ffn-hidden-size 2048 \
    --moe-router-topk 8 --moe-router-score-function sigmoid --moe-router-enable-expert-bias \
    --moe-router-bias-update-rate 0 --moe-router-load-balancing-type none --moe-aux-loss-coeff 0 \
    --moe-router-dtype fp32 --moe-grouped-gemm --moe-token-dispatcher-type alltoall \
    --hidden-dropout 0 --attention-dropout 0 --optimizer adam --lr 1e-5 --lr-decay-style constant \
    --optimizer-cpu-offload --use-precision-aware-optimizer "$@"
