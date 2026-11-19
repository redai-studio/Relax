# Copyright (c) 2026 Relax Authors. All Rights Reserved.

# CLI metadata follows the Hugging Face text config. GLM53FlashBridge derives
# the physical KDA/DSA + FFN layout, mHC, and FP8 import from the checkpoint.
# Read depths and MoE layout from the checkpoint so a subset of real weights
# can run the same recipe for single-node verification.
: "${HF_CHECKPOINT:?Set HF_CHECKPOINT before sourcing the Flash model config}"
flash_layout=$(python3 - "${HF_CHECKPOINT}/config.json" <<'PYCONFIG'
import json
import sys

with open(sys.argv[1]) as config_file:
    config = json.load(config_file)
if config["model_type"] != "glm5_next":
    raise ValueError("Expected a GLM-5.3-Flash checkpoint")
text = config["text_config"]
mlp_types = text["mlp_layer_types"]
if len(mlp_types) != text["num_hidden_layers"] or set(mlp_types) - {"dense", "sparse"}:
    raise ValueError("Invalid Flash MLP layer layout")
frequency = ",".join("1" if mlp == "sparse" else "0" for mlp in mlp_types)
print(text["num_hidden_layers"], text["n_routed_experts"], f"[{frequency}]")
PYCONFIG
)
read -r flash_layers flash_experts flash_frequency <<< "${flash_layout}"

MODEL_ARGS=(
    --num-layers "${flash_layers}"
    --hidden-size 4096
    --ffn-hidden-size 12288
    --num-attention-heads 64
    --vocab-size 154880
    --kv-channels 256
    --position-embedding-type none
    --no-rope-fusion
    --normalization RMSNorm
    --norm-epsilon 1e-5
    --disable-bias-linear
    --swiglu
    --untie-embeddings-and-output-weights

    --multi-latent-attention
    --q-lora-rank 1536
    --kv-lora-rank 512
    --qk-head-dim 256
    --qk-pos-emb-head-dim 0
    --v-head-dim 256

    --num-experts "${flash_experts}"
    --moe-layer-freq "${flash_frequency}"
    --moe-ffn-hidden-size 2048
    --moe-shared-expert-intermediate-size 2048
    --moe-router-topk 8
    --moe-router-score-function sigmoid
    --moe-router-pre-softmax
    --moe-router-enable-expert-bias
    --moe-router-bias-update-rate 0
    --moe-router-topk-scaling-factor 2.5
    --moe-router-dtype fp32
    --moe-router-load-balancing-type none
    --moe-aux-loss-coeff 0
    --moe-grouped-gemm
    --moe-permute-fusion
    --enable-experimental
)
