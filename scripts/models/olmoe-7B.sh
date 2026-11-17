# Copyright (c) 2026 Relax Authors. All Rights Reserved.

MODEL_ARGS=(
    --num-layers 16 --hidden-size 2048 --ffn-hidden-size 1024
    --num-attention-heads 16 --num-query-groups 16 --kv-channels 128
    --disable-bias-linear --qk-layernorm --normalization RMSNorm --norm-epsilon 1e-5
    --position-embedding-type rope --rotary-base 10000 --rotary-percent 1.0
    --swiglu --untie-embeddings-and-output-weights --vocab-size 50304
    --num-experts 64 --moe-ffn-hidden-size 1024 --moe-router-topk 8
    --moe-router-pre-softmax --moe-router-score-function softmax --moe-router-dtype fp32
    --moe-grouped-gemm --moe-permute-fusion --moe-token-dispatcher-type alltoall
)
