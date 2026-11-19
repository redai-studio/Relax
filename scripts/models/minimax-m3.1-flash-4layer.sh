# Copyright (c) 2026 Relax Authors. All Rights Reserved.

MODEL_ARGS=(
   --num-layers 4
   --hidden-size 6144
   --ffn-hidden-size 3072
   --num-attention-heads 64
   --group-query-attention
   --num-query-groups 4
   --kv-channels 128
   --position-embedding-type rope
   --rotary-base 10000000
   --rotary-percent 0.5
   --max-position-embeddings 1048576
   --normalization RMSNorm
   --norm-epsilon 1e-6
   --qk-layernorm
   --disable-bias-linear
   --untie-embeddings-and-output-weights
   --swiglu
   --num-experts 128
   --moe-layer-freq '[0,0,1,1]'
   --moe-router-topk 4
   --moe-ffn-hidden-size 3072
   --moe-shared-expert-intermediate-size 3072
   --moe-router-score-function sigmoid
   --moe-router-enable-expert-bias
   --moe-router-topk-scaling-factor 2
   --moe-router-dtype fp32
   --moe-router-load-balancing-type none
   --moe-aux-loss-coeff 0
   --vocab-size 200064
)
