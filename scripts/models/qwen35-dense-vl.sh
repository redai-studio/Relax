# Copyright (c) 2026 Relax Authors. All Rights Reserved.

# Use the dimensions and embedding layout of the actual stage input.
read -r LAMRA_MODEL_NAME hidden_size ffn_hidden_size num_layers attention_heads query_groups \
    head_dim norm_epsilon rotary_factor rotary_base mrope_t mrope_h mrope_w tied_embeddings < <(
    python3 - "${HF_CHECKPOINT}/config.json" <<'MODEL_CONFIG'
import json
import sys

with open(sys.argv[1]) as stream:
    config = json.load(stream)
text = config["text_config"]
rope = text["rope_parameters"]
sizes = {(24, 1024): "0.8b", (24, 2048): "2b", (32, 2560): "4b", (32, 4096): "9b"}
size = sizes.get((text["num_hidden_layers"], text["hidden_size"]))
if size is None:
    raise SystemExit("Unsupported dense Qwen3.5 LamRA profile")
tied = config.get("tie_word_embeddings", text.get("tie_word_embeddings", False))
print(
    f"qwen35-{size}",
    text["hidden_size"],
    text["intermediate_size"],
    text["num_hidden_layers"],
    text["num_attention_heads"],
    text["num_key_value_heads"],
    text["head_dim"],
    text["rms_norm_eps"],
    rope["partial_rotary_factor"],
    rope["rope_theta"],
    *rope["mrope_section"],
    int(tied),
)
MODEL_CONFIG
)

MODEL_ARGS=(
    --disable-bias-linear
    --qk-layernorm
    --group-query-attention
    --num-attention-heads "$attention_heads"
    --num-query-groups "$query_groups"
    --kv-channels "$head_dim"
    --num-layers "$num_layers"
    --hidden-size "$hidden_size"
    --ffn-hidden-size "$ffn_hidden_size"
    --use-gated-attention
    --normalization RMSNorm
    --apply-layernorm-1p
    --position-embedding-type mrope
    --mrope-section "$mrope_t" "$mrope_h" "$mrope_w"
    --norm-epsilon "$norm_epsilon"
    --rotary-percent "$rotary_factor"
    --swiglu
    --vocab-size "$lamra_vocab_size"
    --rotary-base "$rotary_base"
    --attention-output-gate
)

if [ "$tied_embeddings" = 0 ]; then
    MODEL_ARGS+=(--untie-embeddings-and-output-weights)
fi
