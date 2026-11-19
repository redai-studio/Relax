# Copyright (c) 2026 Relax Authors. All Rights Reserved.

# Read the dense/MoE layout of the converted checkpoint used by this recipe.
read -r layers experts moe_freq < <("${PYTHON:-python3}" - "$HF_CHECKPOINT/config.json" <<'PY'
import json
import sys

with open(sys.argv[1]) as f:
    config = json.load(f)
print(config["num_hidden_layers"], config["n_routed_experts"], json.dumps(config["moe_layer_freq"], separators=(",", ":")))
PY
)

MODEL_ARGS=(
    --num-layers "$layers"
    --hidden-size 4096
    --ffn-hidden-size 16384
    --num-attention-heads 64
    --group-query-attention
    --num-query-groups 4
    --kv-channels 192
    --v-head-dim 128
    --position-embedding-type rope
    --rotary-percent 0.334
    --rotary-base 10000000
    --no-rope-fusion
    --normalization RMSNorm
    --norm-epsilon 1e-6
    --swiglu
    --disable-bias-linear
    --untie-embeddings-and-output-weights
    --vocab-size 152576
    --num-experts "$experts"
    --moe-layer-freq "$moe_freq"
    --moe-ffn-hidden-size 2048
    --moe-router-topk 8
    --moe-router-score-function sigmoid
    --moe-router-enable-expert-bias
    --moe-router-bias-update-rate 0
    --moe-router-load-balancing-type none
    --moe-aux-loss-coeff 0
    --moe-router-dtype fp32
    --moe-grouped-gemm
    --moe-token-dispatcher-type alltoall
    --hidden-dropout 0
    --attention-dropout 0
)
