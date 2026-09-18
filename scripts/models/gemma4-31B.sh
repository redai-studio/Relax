# Copyright (c) 2026 Relax Authors. All Rights Reserved.

# google/gemma-4-31B-it -- DENSE text path.
#
# Needs GEMMA4_CONVERSION_MODE=text (the launch script sets it in the Ray
# runtime env) so Gemma4VLBridge drops the vision tower.
#
# Most args here only feed Relax's _hf_validate_args; the real config comes from
# Gemma4DenseProvider. The exception is `bridge_keys`
# (relax/backends/megatron/model_provider.py), which DOES overwrite the
# provider -- num-layers and rotary-base are in that list, so a wrong value
# silently builds the wrong model instead of erroring.

MODEL_ARGS=(
   --num-layers 60
   --hidden-size 5376
   --ffn-hidden-size 21504
   --num-attention-heads 32
   --group-query-attention
   --num-query-groups 16
   --kv-channels 256
   --vocab-size 262144

   --normalization RMSNorm
   --norm-epsilon 1e-6
   --position-embedding-type rope
   --disable-bias-linear
   --qk-layernorm

   --no-rope-fusion
)

# Do not add --swiglu, --untie-embeddings-and-output-weights or --rotary-base:
# the provider owns them, and --rotary-base 1000000 would silently corrupt every
# sliding layer rather than fail.
