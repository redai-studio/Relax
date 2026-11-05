#!/bin/bash

# Copyright (c) 2026 Relax Authors. All Rights Reserved.
# GPU profile: 280G Blackwell per GPU.
#
# Full Kimi K3 multimodal LoRA SFT on 8 x 8 GPUs (280G Blackwell each): reuse the full-model 64-GPU
# base driver (93 layers, 896 experts, TP4/PP4/CP4/EP16, all-to-all dispatcher,
# CPU optimizer offload) and train adapters only. LoRA keeps optimizer state
# negligible, so the full 2.78T model fits on 64 GPUs where full-parameter SFT
# would host-RAM OOM — this is the intended path on this GPU count.
#
# LoRA targets cover the full language backbone — everything EXCEPT the vision
# tower and lm_head — matching the validated 8-GPU reduced-layer recipe with its
# vision_tower.*/mm_projector entries removed: attention (q/k/v/o), KDA gates
# (f_a/f_b/b/g), MLA (q_a/q_b/kv_a/kv_b), dense/expert MLPs (linear_fc1/fc2) and
# the latent MoE projections (fc1/fc2_latent_proj). Because expert MLPs are
# targeted, RELAX_LORA_SHARE_EXPERT_ADAPTERS=false is injected below (independent
# per-expert adapters + Bridge EP grad finalization), as in the 8-GPU recipe.
#
# Match the 64-GPU K3 Think recipe's training optimizations and optimizer
# defaults. Pass trailing command-line arguments to override them.
#
# Usage (submit through the existing cluster entrypoint):
#   bash scripts/entrypoint/ray-job.sh \
#       examples/models/kimi-k3/scripts/run-kimi-k3-pokemon-lora-64xblackwell.sh
# Inspect resolved arguments without starting Ray or submitting:
#   DRY_RUN=1 bash examples/models/kimi-k3/scripts/run-kimi-k3-pokemon-lora-64xblackwell.sh

set -eo pipefail
export RAY_NO_WAIT="${RAY_NO_WAIT-}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
export EXP_NAME="${EXP_NAME:-kimi-k3-full-lora-sft-pokemon-zh-blackwell-280g-gpu64-bridge}"
# Produce an early adapter checkpoint for conversion validation.
export NUM_STEPS="${NUM_STEPS:-1}"
export SAVE_INTERVAL="${SAVE_INTERVAL:-1}"
export MAX_TOKENS_PER_GPU="${MAX_TOKENS_PER_GPU:-32768}"
export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-64}"
export SFT_LOGITS_CHUNK_SIZE="${SFT_LOGITS_CHUNK_SIZE:-1024}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"
export RAY_USAGE_STATS_ENABLED="${RAY_USAGE_STATS_ENABLED:-0}"
export NVSHMEM_REMOTE_TRANSPORT="${NVSHMEM_REMOTE_TRANSPORT:-libfabric}"

# Keep these switches in train-env-vars so they reach the GPU workers.
export TRAIN_ENV_VARS="{\"FLA_TILELANG\":\"${FLA_TILELANG:-0}\",\"PYTORCH_CUDA_ALLOC_CONF\":\"expandable_segments:True\",\"RELAX_KIMI_K3_EXPERT_SPLIT\":\"${RELAX_KIMI_K3_EXPERT_SPLIT:-1}\",\"RELAX_KIMI_K3_FUSED_SITU\":\"${RELAX_KIMI_K3_FUSED_SITU:-1}\",\"RELAX_KIMI_K3_FUSED_ATTN_RES\":\"${RELAX_KIMI_K3_FUSED_ATTN_RES:-1}\"}"

OPTIMIZATION_ARGS=(
    --attention-backend fused
    --image-max-token-num 1024
    --data-pad-size-multiplier 4096
    --sft-max-in-flight-steps 4
    --num-data-storage-units 16
    --lr "${LR:-1e-5}"
    --lr-warmup-fraction 0.05
    --weight-decay "${WEIGHT_DECAY:-0.1}"
    --adam-beta2 0.95
)
if [ "${FREEZE_VISION_TOWER:-1}" = 1 ]; then
    OPTIMIZATION_ARGS+=(--vision-dp-when-tp --freeze-vision-model)
fi

LORA_ARGS=(
    --lora-rank "${LORA_RANK:-16}"
    --lora-alpha "${LORA_ALPHA:-32}"
    --lora-dropout "${LORA_DROPOUT:-0.0}"
    --lora-scope "${LORA_SCOPE:-all}"
    --lora-merge-mode
    --lora-target-modules
    # Full language backbone (everything except vision tower and lm_head).
    # KDA and MLA use separate projections, not linear_qkv/linear_proj.
    q_proj k_proj v_proj o_proj f_a_proj f_b_proj b_proj g_proj
    q_a_proj q_b_proj kv_a_proj_with_mqa kv_b_proj
    # Dense, shared/routed expert and latent MoE projections.
    linear_fc1 linear_fc2 fc1_latent_proj fc2_latent_proj
)
if [ -n "${SAVE_DIR:-}" ]; then
    LORA_ARGS+=(--save-lora-only)
fi

# Expert MLPs (linear_fc1/linear_fc2) are LoRA targets, so use independent
# per-expert adapters — the shared variant isn't wired for Bridge's extra EP
# gradient finalization. Inject the flag into the Ray runtime env so it reaches
# every actor, mirroring the 8-GPU LoRA recipe. Skipped under DRY_RUN, which
# never brings up Ray / builds RUNTIME_ENV_JSON.
if [ "${DRY_RUN:-0}" != 1 ]; then
    if [ -z "${RELAX_ENTRYPOINT_MODE:-}" ]; then
        source "${SCRIPT_DIR}/../../../../scripts/entrypoint/ray-job.sh"
    fi
    RUNTIME_ENV_JSON=$(python3 - <<'PY'
import json
import os

runtime = json.loads(os.environ["RUNTIME_ENV_JSON"])
env = runtime.setdefault("env_vars", {})
env["RELAX_LORA_SHARE_EXPERT_ADAPTERS"] = "false"
worker_env_names = (
    "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
    "RAY_USAGE_STATS_ENABLED", "NVSHMEM_REMOTE_TRANSPORT",
)
env.update({name: os.environ[name] for name in worker_env_names})
propagate = env.get("RELAX_PROPAGATE_ENV_VARS", "").split(",")
env["RELAX_PROPAGATE_ENV_VARS"] = ",".join(
    dict.fromkeys(name for name in [*propagate, "RELAX_LORA_SHARE_EXPERT_ADAPTERS", *worker_env_names] if name)
)
print(json.dumps(runtime))
PY
    )
    export RUNTIME_ENV_JSON
fi

exec bash "${SCRIPT_DIR}/run-kimi-k3-pokemon-64xblackwell.sh" "${LORA_ARGS[@]}" "${OPTIMIZATION_ARGS[@]}" "$@"
