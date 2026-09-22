# Copyright (c) 2026 Relax Authors. All Rights Reserved.

# google/gemma-4-26B-A4B-it -- MoE text path.
#
# Values come from gemma-4-26B-A4B-it/config.json `text_config`, not the
# top-level multimodal config. Needs GEMMA4_CONVERSION_MODE=text (the launch
# script sets it in the Ray runtime env) and megatron-dev >= a8d06a1.

NLAYERS=30
NHIDDEN=2816
NHEADS=16
NUM_QUERY_GROUPS=8            # num_key_value_heads
HEAD_DIM=256                  # head_dim (sliding layers; global layers use 512)
VOCAB=262144

MOE_ROUTED_EXPERTS=128        # num_experts
MOE_ACTIVE_ROUTED_EXPERTS=8   # top_k_experts
MOE_FFN_HIDDEN=704            # moe_intermediate_size
SHARED_EXPERT_FFN_HIDDEN=2112 # intermediate_size (the shared expert, not a dense FFN)

# Most --moe-* flags below are in bridge_keys
# (relax/backends/megatron/model_provider.py): omitting one overwrites the
# checkpoint value with megatron's default instead of keeping it.
MODEL_ARGS=(
   --num-layers ${NLAYERS}
   --hidden-size ${NHIDDEN}
   --ffn-hidden-size ${SHARED_EXPERT_FFN_HIDDEN}
   --num-attention-heads ${NHEADS}
   --group-query-attention
   --num-query-groups ${NUM_QUERY_GROUPS}
   --kv-channels ${HEAD_DIM}
   --vocab-size ${VOCAB}

   --normalization RMSNorm
   --norm-epsilon 1e-6
   --position-embedding-type rope
   --disable-bias-linear
   --qk-layernorm

   --num-experts ${MOE_ROUTED_EXPERTS}
   --moe-router-topk ${MOE_ACTIVE_ROUTED_EXPERTS}
   --moe-ffn-hidden-size ${MOE_FFN_HIDDEN}
   --moe-shared-expert-intermediate-size ${SHARED_EXPERT_FFN_HIDDEN}
   --moe-layer-freq 1
   --moe-grouped-gemm
   --moe-permute-fusion
   --moe-router-pre-softmax
   --moe-router-dtype fp32
   --moe-token-dispatcher-type alltoall

   --moe-aux-loss-coeff 0
   --moe-router-load-balancing-type none

   --no-rope-fusion
)

# Do not add --rotary-base, --swiglu, --untie-embeddings-and-output-weights or
# --make-vocab-size-divisible-by: the provider owns them and some fail
# validation. Expert-parallel sizes belong to the launch script's PERF_ARGS.
