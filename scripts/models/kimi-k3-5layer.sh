# Copyright (c) 2026 Relax Authors. All Rights Reserved.

# Kimi-K3-5L-128E-AttnRes4-MXFP4: KDA layers 1/2/3/5, MLA layer 4;
# one dense and four MoE layers, with the complete 27-layer vision tower.
# Bridge reads AttnRes block size 4 and layer placement from the HF config.
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)/kimi-k3.sh"

MODEL_ARGS+=(
    --num-layers 5
    --num-experts 128
    --moe-layer-freq '[0]+[1]*4'
)
