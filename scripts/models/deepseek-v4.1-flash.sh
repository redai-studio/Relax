# Copyright (c) 2026 Relax Authors. All Rights Reserved.
#
# Full DeepSeek-V4.1-Flash text decoder, using a BF16 checkpoint.
# The Bridge reads compression, retrieval, mHC and Engram topology from HF config.
# Vision, MTP and DSpark are not part of this text-training recipe.

MODEL_ARGS=(
    --num-layers 40
    --hidden-size 5120
    --ffn-hidden-size 2304
    --num-attention-heads 64
    --normalization RMSNorm
    --norm-epsilon 1e-20
    --position-embedding-type rope
    --disable-bias-linear
    --swiglu
    --untie-embeddings-and-output-weights
    --vocab-size 129280
    --make-vocab-size-divisible-by 64

    --multi-latent-attention
    --q-lora-rank 1280
    --kv-lora-rank 512
    --qk-head-dim 512
    --qk-pos-emb-head-dim 64
    --v-head-dim 512
    --qk-layernorm
    --rotary-base 10000
    --rotary-scaling-factor 16
    --no-rope-fusion

    --num-experts 384
    --moe-layer-freq '[1]*40'
    --moe-ffn-hidden-size 2304
    --moe-shared-expert-intermediate-size 2304
    --moe-router-topk 6
    --moe-router-score-function sqrtsoftplus
    --moe-router-dtype fp32
    --moe-router-topk-scaling-factor 1.5
    --moe-router-enable-expert-bias
    --moe-router-bias-update-rate 0
    --moe-router-load-balancing-type none
    --moe-aux-loss-coeff 0
    --moe-grouped-gemm
    --enable-experimental
)
