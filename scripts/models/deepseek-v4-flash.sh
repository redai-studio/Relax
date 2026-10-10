# Copyright (c) 2026 Relax Authors. All Rights Reserved.
#
# DeepSeek-V4-Flash-0731 (~290B total / ~6.5B activated) model shape.
#
# NOTE ON BRIDGE MODE
# -------------------
# This model MUST run with `--megatron-to-hf-mode bridge`. Megatron-Bridge's
# DeepSeekV4Bridge (`megatron/bridge/models/deepseek/deepseek_v4_bridge.py`,
# registered for source="DeepseekV4ForCausalLM", model_type="deepseek_v4")
# derives the whole DSv4-hybrid configuration from the HF config.json:
#
#   experimental_attention_variant = "dsv4_hybrid"
#   csa_compress_ratios / csa_window_size / csa_compress_rotary_base
#   o_groups / o_lora_rank                (grouped output projection)
#   enable_hyper_connections / num_residual_streams / mhc_sinkhorn_iterations
#   moe_n_hash_layers                     (Hash-MoE bootstrap layers)
#   dsa_indexer_n_heads / _head_dim / _topk
#   activation_func_clamp_value           (= swiglu_limit)
#   norm_topk_prob / actual_vocab_size
#
# None of those have a Relax CLI flag, and `model_provider.py` only lets args
# in its `bridge_keys` allowlist override the provider. So MODEL_ARGS below is
# deliberately MINIMAL: it carries only (a) shape values that
# `_hf_validate_args` cross-checks against config.json, and (b) fields that are
# in `bridge_keys` AND that we must pin because the bridge gets them wrong or
# leaves them at a bad default.
#
# Do NOT add csa_* / o_groups / moe_n_hash_layers / num_residual_streams flags
# here — they do not exist in the parser and would be silently dropped even if
# they did.

NLAYERS=43
NHIDDEN=4096
NHEADS=64
MOE_ROUTED_EXPERTS=256
MOE_ACTIVE_ROUTED_EXPERTS=6
MOE_SHARED_EXPERTS=1
MOE_FFN_HIDDEN=2048
MOE_SHARED_EXPERT_INTERMEDIATE_SIZE=$((MOE_FFN_HIDDEN * MOE_SHARED_EXPERTS))

# head_dim=512 splits into a 448-dim NoPE part + a 64-dim RoPE part.
# Checkpoint proof: layers.N.attn.wq_b.weight is [32768, 1024] = 64 heads x 512.
# mcore derives the 448 itself (see the MLA block below); it is not passed.
HEAD_DIM=512
QK_POS_EMB_HEAD_DIM=64

MODEL_ARGS=(
    --num-layers $NLAYERS
    --hidden-size $NHIDDEN
    --ffn-hidden-size $MOE_FFN_HIDDEN
    --num-attention-heads $NHEADS
    --kv-channels $HEAD_DIM
    --normalization RMSNorm
    --norm-epsilon 1e-6
    --position-embedding-type rope
    --disable-bias-linear
    --swiglu
    --untie-embeddings-and-output-weights
    --vocab-size 129280
    --make-vocab-size-divisible-by 1280
    --qk-layernorm

    # --- MLA -------------------------------------------------------------
    # Only v_head_dim and qk_pos_emb_head_dim are set. mcore logs
    #   "DSv4 hybrid mode is enabled, deriving qk_head_dim and kv_lora_rank
    #    from v_head_dim and qk_pos_emb_head_dim"
    # at model-build time, so --qk-head-dim / --kv-lora-rank are deliberately
    # NOT passed: DSv4HybridAttention never reads qk_head_dim (it uses
    # q_head_dim = v_head_dim, and nope_dim = v_head_dim - qk_pos_emb_head_dim
    # = 448), and passing them would only look authoritative while being
    # overwritten. Verified on the rebuilt image: the provider reports
    # qk_head_dim=128 (the MLATransformerConfig default) and the model still
    # builds with the correct 448-wide NoPE split.
    --multi-latent-attention
    --q-lora-rank 1024
    --qk-pos-emb-head-dim $QK_POS_EMB_HEAD_DIM
    --v-head-dim $HEAD_DIM

    # --- RoPE / YaRN ------------------------------------------------------
    # rotary-base is the MAIN rotary (10000). The compressed CSA/HCA branch
    # uses compress_rope_theta=160000 with YaRN factor 16 — the bridge reads
    # that from config.json into csa_compress_rotary_base; there is no flag.
    --rotary-base 10000
    --rotary-scaling-factor 16.0

    # --- MoE --------------------------------------------------------------
    # Every layer is MoE (the first 3 are hash-routed, which the bridge
    # expresses via moe_n_hash_layers, not via moe-layer-freq).
    --moe-layer-freq [1]*$NLAYERS
    --num-experts $MOE_ROUTED_EXPERTS
    --moe-ffn-hidden-size $MOE_FFN_HIDDEN
    --moe-router-topk $MOE_ACTIVE_ROUTED_EXPERTS
    --moe-shared-expert-intermediate-size $MOE_SHARED_EXPERT_INTERMEDIATE_SIZE
    --moe-router-score-function sqrtsoftplus
    --moe-router-enable-expert-bias
    --moe-router-bias-update-rate 0
    --moe-router-topk-scaling-factor 1.5
    --moe-router-dtype fp32
    --moe-grouped-gemm
    --moe-permute-fusion

    # --- Lightning Indexer -------------------------------------------------
    # MUST be set explicitly. mcore's field default is Optional[float] = None,
    # and csa.py reads it two inconsistent ways:
    #   csa.py:1795  self.config.dsa_indexer_loss_coeff or 0.0        (None-safe)
    #   csa.py:1964  getattr(self.config, '...', 0.0)                 (NOT None-safe
    #                -- getattr's default only fires when the attribute is absent)
    # The unfused CSA THD path goes through 1964, so leaving it unset reaches
    # dsa.py:365 `kl_div * loss_coeff` with loss_coeff=None and dies with
    # "unsupported operand type(s) for *: 'Tensor' and 'NoneType'" on the first
    # train_one_step (the compute_log_prob forward does not compute this loss,
    # so the failure surfaces one step later than you would expect).
    # The bridge does not set it either -- only the recipes do -- and
    # model_provider.py's bridge_keys loop assigns args values unconditionally,
    # so provider.dsa_indexer_loss_coeff ends up None regardless.
    #
    # 0.0 is what every upstream H100 recipe uses (bridge/recipes/deepseek/h100/
    # deepseek_v4.py:94,174,246,331,401); only the GB200/GB300 *pretrain* recipes
    # use 0.01. The indexer KL loss is the sole gradient path into the Lightning
    # Indexer, so 0.0 freezes the indexer at its pretrained state -- which is what
    # we want for RL post-training.
    #
    # Caveat: 0.0 does NOT skip the work. mcore has no coeff==0 short-circuit --
    # FusedDSAIndexerLoss.apply is called unconditionally (csa.py:2006) and
    # compute_dsa_indexer_loss builds the dense [1, heads, sq, sk] fp32 score
    # tensor before multiplying by the coefficient on the very last line. That is
    # ~1GB per segment at 4K, and is another reason the sequence length has to
    # stay small on Hopper.
    --dsa-indexer-loss-coeff 0.0

    # MANDATORY whenever apply_dsa_kernel_fusion is on. mcore's own guard
    # (transformer_config.py:1656-1666) says the cuDNN Frontend SM90 *dense* DSA
    # kernels "are not reliable for this path" -- but it only fires when
    # dsa_indexer_loss_coeff > 0, while dsa_kernels.py:1550 picks dense vs sparse
    # purely off this flag. At coeff 0.0 we therefore silence the guard and still
    # take the unreliable path. It really is broken: under CP,
    # _dense_score_recompute_varlen slices q_causal_offsets[b:b+1] (int32, so
    # data_ptr = base + 4*b) and hands it to CuTe, which demands 16-byte alignment
    # -> "Tensor data pointer is not aligned to 16 bytes" on the first b % 4 != 0
    # that misses the JIT compile cache. The sparse branch never touches
    # q_causal_offsets, and its `if loss_coeff > 0` skips the backward kernel
    # entirely at coeff 0. Every upstream recipe with fusion on (gb200/gb300 perf)
    # sets this True; the h100 recipes leave it False only because they also force
    # fusion off.
    --dsa-indexer-use-sparse-loss

    # DSv4 hybrid attention + mHC + Hash-MoE all live behind mcore's
    # experimental gate (enable_megatron_core_experimental).
    --enable-experimental
)
