# Copyright (c) 2026 Relax Authors. All Rights Reserved.

# Qwen3.8-Flash-Next reduced-depth bring-up.
#
# Experimental GDN/QSA/PLE details are supplied by the Qwen4Exp Bridge
# provider. The basic shape still has to be present on the Megatron CLI so
# Megatron's argument validation can finish before that provider is built.
# Defaults below are the published Qwen3.8-Flash-Next text config; environment
# overrides are useful for reduced synthetic smoke checkpoints.

MODEL_ARGS=(
    --num-layers "${NUM_LAYERS:-8}"
    --hidden-size "${HIDDEN_SIZE:-2560}"
    --ffn-hidden-size "${FFN_HIDDEN_SIZE:-640}"
    --num-attention-heads "${NUM_ATTENTION_HEADS:-24}"
    --group-query-attention
    --num-query-groups "${NUM_QUERY_GROUPS:-2}"
    --kv-channels "${KV_CHANNELS:-256}"
    --normalization RMSNorm
    --norm-epsilon "${NORM_EPSILON:-1e-6}"
    --apply-layernorm-1p
    --position-embedding-type rope
    --rotary-percent 0.25
    --rotary-base "${ROTARY_BASE:-10000000}"
    --disable-bias-linear
    --qk-layernorm
    --swiglu
    --untie-embeddings-and-output-weights
    --vocab-size "${VOCAB_SIZE:-248320}"
    --num-experts "${NUM_EXPERTS:-512}"
    --moe-ffn-hidden-size "${MOE_FFN_HIDDEN_SIZE:-640}"
    --moe-router-topk "${MOE_ROUTER_TOPK:-10}"
    --moe-router-load-balancing-type "${MOE_ROUTER_LOAD_BALANCING_TYPE:-global_aux_loss}"
    --moe-aux-loss-coeff "${MOE_AUX_LOSS_COEFF:-0.001}"
    --moe-router-dtype fp32
    --moe-permute-fusion
    --moe-shared-expert-intermediate-size "${MOE_SHARED_EXPERT_INTERMEDIATE_SIZE:-640}"
)
