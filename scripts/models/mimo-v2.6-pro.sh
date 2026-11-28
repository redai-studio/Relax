#!/usr/bin/env bash
# Copyright (c) 2026 Relax Authors. All Rights Reserved.

# Full-depth BF16 MiMo-V2.6-Pro architecture, with LoRA train-time split Q/K/V.
model_dims=$("${2:-python3}" - "${1:?Provide the converted MiMo Pro checkpoint}/config.json" <<'PY'
import json
import sys
from pathlib import Path

config_path = Path(sys.argv[1])
config = json.loads(config_path.read_text())
if (
    config.get("model_type") != "mimo_v2"
    or config.get("num_hidden_layers") != 70
    or config.get("n_routed_experts") != 384
    or config.get("attention_projection_layout", "split") != "split"
    or config.get("quantization_config") is not None
):
    raise SystemExit("Use the full-depth BF16 MiMo-V2.6-Pro checkpoint with split Q/K/V.")
index_path = config_path.with_name("model.safetensors.index.json")
if index_path.is_file():
    weights = json.loads(index_path.read_text())["weight_map"]
    if any(
        f"model.layers.{layer}.self_attn.{part}_proj.weight" not in weights
        for layer in range(70)
        for part in "qkv"
    ):
        raise SystemExit("The checkpoint index must contain split Q/K/V for every decoder layer.")
keys = (
    "hidden_size", "intermediate_size", "num_attention_heads", "num_key_value_heads",
    "head_dim", "v_head_dim", "partial_rotary_factor", "rope_theta", "layernorm_epsilon",
    "vocab_size", "moe_intermediate_size", "num_experts_per_tok",
)
print(*(config[key] for key in keys), json.dumps(config["moe_layer_freq"], separators=(",", ":")))
PY
)
read -r hidden ffn heads groups qk v rotary rope epsilon vocab expert_ffn topk moe_freq <<< "$model_dims"

MODEL_ARGS=(
    --num-layers 70
    --hidden-size "$hidden"
    --ffn-hidden-size "$ffn"
    --num-attention-heads "$heads"
    --group-query-attention
    --num-query-groups "$groups"
    --kv-channels "$qk"
    --v-head-dim "$v"
    --position-embedding-type rope
    --rotary-percent "$rotary"
    --rotary-base "$rope"

    --normalization RMSNorm
    --norm-epsilon "$epsilon"
    --swiglu
    --disable-bias-linear
    --untie-embeddings-and-output-weights
    --vocab-size "$vocab"

    --num-experts 384
    --moe-layer-freq "$moe_freq"
    --moe-ffn-hidden-size "$expert_ffn"
    --moe-router-topk "$topk"
    --moe-router-score-function sigmoid
    --moe-router-enable-expert-bias
    --moe-router-bias-update-rate 0
    --moe-router-load-balancing-type none
    --moe-aux-loss-coeff 0
    --moe-router-dtype fp32
    --moe-grouped-gemm
    --hidden-dropout 0
    --attention-dropout 0
)
