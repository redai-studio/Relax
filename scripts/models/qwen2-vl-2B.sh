# Copyright (c) 2026 Relax Authors. All Rights Reserved.

# Qwen/Qwen2-VL-2B-Instruct; preserve its tied embedding and original MRoPE.
# The prepared <emb> checkpoint may have a resized vocabulary.
MODEL_ARGS=(
    --swiglu
    --num-layers 28
    --hidden-size 1536
    --ffn-hidden-size 8960
    --num-attention-heads 12
    --group-query-attention
    --num-query-groups 2
    --use-rotary-position-embeddings
    --disable-bias-linear
    --add-qkv-bias
    --normalization RMSNorm
    --norm-epsilon 1e-6
    --rotary-base 1000000
    --vocab-size "${QWEN2_VL_VOCAB_SIZE:-151936}"
    --kv-channels 128
)
