# Copyright (c) 2026 Relax Authors. All Rights Reserved.

# Qwen3.8-Flash-Next published 48-layer text shape. Experimental GDN/QSA/PLE
# details are finalized by the pinned Qwen4Exp Megatron Bridge provider.

# Shared defaults; launchers may override settings for their topology.
export RELAX_QWEN38_QSA_SPARSE_BLOCKMASK="${RELAX_QWEN38_QSA_SPARSE_BLOCKMASK:-1}"
export RELAX_QWEN38_QSA_LOCAL_QUERY="${RELAX_QWEN38_QSA_LOCAL_QUERY:-1}"
export RELAX_QWEN38_QSA_RECOMPUTE_REPLAY="${RELAX_QWEN38_QSA_RECOMPUTE_REPLAY:-1}"
export RELAX_QWEN38_QSA_FUSED_SELECTOR="${RELAX_QWEN38_QSA_FUSED_SELECTOR:-1}"
export RELAX_QWEN38_QSA_FUSED_ATTENTION="${RELAX_QWEN38_QSA_FUSED_ATTENTION:-1}"
export RELAX_QWEN38_QSA_COMMUNITY_BACKEND="${RELAX_QWEN38_QSA_COMMUNITY_BACKEND:-}"
if [[ -n "${RELAX_QWEN38_QSA_COMMUNITY_BACKEND}" ]]; then
    export MCORE_QSA_SPARSE_BACKEND="${MCORE_QSA_SPARSE_BACKEND:-id_sparse}"
    export RELAX_QWEN38_QSA_FUSED_SELECTOR=0
fi
export RELAX_QWEN38_FLA_CONV_CACHE_REUSE="${RELAX_QWEN38_FLA_CONV_CACHE_REUSE:-1}"
export RELAX_QWEN38_PLE_STREAMED="${RELAX_QWEN38_PLE_STREAMED:-1}"

MODEL_ARGS=(
    --num-layers 48
    --hidden-size 2560
    --ffn-hidden-size 640
    --num-attention-heads 24
    --group-query-attention
    --num-query-groups 2
    --kv-channels 256
    --normalization RMSNorm
    --norm-epsilon 1e-6
    --apply-layernorm-1p
    --position-embedding-type rope
    --rotary-percent 0.25
    --rotary-base 10000000
    --disable-bias-linear
    --qk-layernorm
    --swiglu
    --untie-embeddings-and-output-weights
    --vocab-size 248320
    --num-experts 512
    --moe-ffn-hidden-size 640
    --moe-router-topk 10
    --moe-router-load-balancing-type global_aux_loss
    --moe-aux-loss-coeff 0.001
    --moe-router-dtype fp32
    --moe-permute-fusion
    --moe-shared-expert-intermediate-size 640
)
