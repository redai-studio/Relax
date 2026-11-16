# Copyright (c) 2026 Relax Authors. All Rights Reserved.

MODEL_ARGS=(
    --disable-bias-linear
    --qk-layernorm
    --group-query-attention
    --num-attention-heads 8
    --num-query-groups 2
    --kv-channels 256
    --num-layers 24
    --hidden-size 1024
    --ffn-hidden-size 3584
    --use-gated-attention
    --normalization RMSNorm
    --apply-layernorm-1p
    --position-embedding-type mrope
    --mrope-section 11 11 10
    --norm-epsilon 1e-6
    --rotary-percent 0.25
    --swiglu
    --vocab-size "${QWEN35_VOCAB_SIZE:-248320}"
    --rotary-base 10000000
    --attention-output-gate
)
