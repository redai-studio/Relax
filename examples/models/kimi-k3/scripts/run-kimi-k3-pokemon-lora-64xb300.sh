#!/bin/bash

# Copyright (c) 2026 Relax Authors. All Rights Reserved.
#
# Full Kimi K3 multimodal LoRA SFT on 8 x 8 B300: reuse the full-model 64-GPU
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
# Inherit the base script learning rate (1e-5). Pass --lr as a trailing
# command-line argument to override it.
#
# Usage (submit through the existing cluster entrypoint):
#   bash scripts/entrypoint/ray-job.sh \
#       examples/models/kimi-k3/scripts/run-kimi-k3-pokemon-lora-64xb300.sh
# Inspect resolved arguments without starting Ray or submitting:
#   DRY_RUN=1 bash examples/models/kimi-k3/scripts/run-kimi-k3-pokemon-lora-64xb300.sh

set -eo pipefail
export RAY_NO_WAIT="${RAY_NO_WAIT-}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
export EXP_NAME="${EXP_NAME:-kimi-k3-full-lora-sft-pokemon-zh-b300-gpu64-bridge}"
# Produce an early adapter checkpoint for conversion validation.
export NUM_STEPS="${NUM_STEPS:-1}"
export SAVE_INTERVAL="${SAVE_INTERVAL:-1}"

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
propagate = env.get("RELAX_PROPAGATE_ENV_VARS", "").split(",")
env["RELAX_PROPAGATE_ENV_VARS"] = ",".join(
    dict.fromkeys(name for name in [*propagate, "RELAX_LORA_SHARE_EXPERT_ADAPTERS"] if name)
)
print(json.dumps(runtime))
PY
    )
    export RUNTIME_ENV_JSON
fi

exec bash "${SCRIPT_DIR}/run-kimi-k3-pokemon-64xb300.sh" "${LORA_ARGS[@]}" "$@"
