# Copyright (c) 2026 Relax Authors. All Rights Reserved.

# moonshotai/Kimi-K3, HF revision f831ab66814297da540d832a5235f8e904f29d06.
# Use --megatron-to-hf-mode bridge: the provider supplies KDA/MLA placement,
# SiTU, AttnRes, latent MoE, and the MoonViT-V2 + PatchMergerMLPV2 vision path.
# Fixed CP uses zigzag input partitioning with headwise or chunkwise KDA.
# Dynamic CP, allgather CP, VPP, and MTP are not supported.
# TP>1 also requires --sequence-parallel for K3's replicated parameter gradients.
# Ordinary 1F1B PP can combine CP with --recompute-granularity full
# --recompute-method uniform --recompute-num-layers 1. BF16 optimizer offload
# uses --optimizer-cpu-offload --overlap-cpu-optimizer-d2h-h2d
# --use-precision-aware-optimizer; configure these runtime flags in the training script.
# Choose TP/PP/EP and the GPU allocation in the training script for the target cluster.

MODEL_ARGS=(
    --num-layers 93
    --hidden-size 7168
    --ffn-hidden-size 33792
    --num-attention-heads 96
    --kv-channels 128
    --normalization RMSNorm
    --norm-epsilon 1e-5
    --position-embedding-type none
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
    --attention-softmax-in-fp32
    --no-rope-fusion

    --moe-layer-freq '[0]+[1]*92'
    --num-experts 896
    --moe-latent-size 3584
    --moe-ffn-hidden-size 3072
    --moe-router-topk 16
    --moe-shared-expert-intermediate-size 6144
    --moe-router-pre-softmax
    --moe-router-score-function sigmoid
    --moe-router-enable-expert-bias
    --moe-router-load-balancing-type none
    --moe-token-dispatcher-type alltoall
    --moe-aux-loss-coeff 0
    --moe-router-bias-update-rate 0
    --moe-router-group-topk 1
    --moe-router-num-groups 1
    --moe-router-topk-scaling-factor 1.0
    --moe-router-dtype fp32
    --moe-grouped-gemm
    --moe-permute-fusion
)
