#!/usr/bin/env bash
# Variable-length long-context SFT: TP8/PP2/EP8, DP4, GBS64 on 64 Blackwell GPUs.
# MODEL_DIR, DATA_DIR, SAVE_DIR and MEGATRON follow the neighboring recipes.
# Requires the FLA long-offset patch in docker/patch/fla (included in new images).
# Packing is a soft 256k budget; keep retains oversized singleton samples.
# Audit input lengths: this recipe does not truncate or filter records at 512k.
# Defaults reproduce the no-save validation mode; save/resume needs validation.
# DRY_RUN=1 prints argv and runtime environment without contacting Ray.
set -eo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export ACTOR_NUM_NODES=8 TP_SIZE=8 PP_SIZE=2
export PIPELINE_LAYOUT="${PIPELINE_LAYOUT:-Et*24|t*24mL}"
export SEQ_LENGTH="${SEQ_LENGTH:-524288}" MAX_TOKENS_PER_GPU="${MAX_TOKENS_PER_GPU:-262144}"
export GBS="${GBS:-64}" DATA_PAD_SIZE_MULTIPLIER="${DATA_PAD_SIZE_MULTIPLIER:-4096}"
export EMPTY_UNUSED_MEMORY_LEVEL=1 RELAX_MEMORY_RELEASE_GC=1
export RELAX_MEMORY_RELEASE_FIRST_ATTEMPT_ONLY="${RELAX_MEMORY_RELEASE_FIRST_ATTEMPT_ONLY:-0}"
export QWEN48_PP_PAYLOAD_UNBATCHED=1 QWEN38_CUDA_DEVICE_MAX_CONNECTIONS=1
export RELAX_FLA_CONV_NB_MODE=none
export SFT_OVERSIZE_STRATEGY=keep SFT_INVALID_MULTIMODAL_STRATEGY=error
export EVAL_SIZE="${EVAL_SIZE:-none}" SAVE_INTERVAL="${SAVE_INTERVAL:-none}" SEED="${SEED:-9714}"
export EXP_NAME="${EXP_NAME:-qwen38-qsa-sft-512k-64xblackwell}"
export PROJECT_NAME="${PROJECT_NAME:-Relax/sft/qwen38-qsa-512k}"
exec bash "${SCRIPT_DIR}/run-qwen38-qsa-sft-256k-32xgpu-blackwell.sh" --max-global-restart 0 "$@"
