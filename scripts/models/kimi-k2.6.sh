# Copyright (c) 2026 Relax Authors. All Rights Reserved.

# Kimi K2.6 (KimiK25ForConditionalGeneration) language backbone is identical to
# K2-Thinking. The vision tower + mm_projector are populated by Megatron-Bridge's
# KimiK25VLBridge from the HF text_config / vision_config; no extra MODEL_ARGS are
# required on the Relax side.

NLAYERS=${NLAYERS:-61}
FIRST_K_DENSE_REPLACE=1

NHIDDEN=7168
FFN_HIDDEN=18432
NHEADS=64

MOE_ROUTED_EXPERTS=384
MOE_ACTIVE_ROUTED_EXPERTS=8
MOE_FFN_HIDDEN=2048
MOE_SHARED_EXPERTS=1
MOE_SHARED_EXPERT_INTERMEDIATE_SIZE=$((MOE_FFN_HIDDEN * MOE_SHARED_EXPERTS))

MODEL_ARGS=(
    --num-layers $NLAYERS
    --hidden-size $NHIDDEN
    --ffn-hidden-size $FFN_HIDDEN
    --num-attention-heads $NHEADS
    --kv-channels 64
    --normalization RMSNorm
    --norm-epsilon 1e-5
    --position-embedding-type rope
    --disable-bias-linear
    --swiglu
    --untie-embeddings-and-output-weights
    --vocab-size 163840

    --multi-latent-attention
    --q-lora-rank 1536
    --kv-lora-rank 512
    --qk-head-dim 128
    --qk-pos-emb-head-dim 64
    --v-head-dim 128
    --qk-layernorm
    --rotary-scaling-factor 64.0
    --rotary-base 50000
    --mscale 1.0
    --mscale-all-dim 1.0
    --attention-softmax-in-fp32
    --no-rope-fusion

    --moe-layer-freq [0]*$FIRST_K_DENSE_REPLACE+[1]*$((NLAYERS - FIRST_K_DENSE_REPLACE))
    --num-experts $MOE_ROUTED_EXPERTS
    --moe-ffn-hidden-size $MOE_FFN_HIDDEN
    --moe-router-topk $MOE_ACTIVE_ROUTED_EXPERTS
    --moe-shared-expert-intermediate-size $MOE_SHARED_EXPERT_INTERMEDIATE_SIZE
    --moe-router-pre-softmax
    --moe-router-score-function sigmoid
    --moe-router-enable-expert-bias
    --moe-router-load-balancing-type seq_aux_loss
    --moe-token-dispatcher-type alltoall
    --moe-aux-loss-coeff 0
    --moe-router-bias-update-rate 0
    --moe-router-group-topk 1
    --moe-router-num-groups 1
    --moe-grouped-gemm
    --moe-router-topk-scaling-factor 2.827
    --moe-router-dtype fp32
    --moe-permute-fusion
)
