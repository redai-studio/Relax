#!/usr/bin/env bash
# Fresh fixed-shape 512k validation: release memory only on the first attempt.
# DATA_DIR must contain prepared equal-length samples; no dataset is generated.
# This mode has no save/resume acceptance and must use a fresh checkpoint path.
set -eo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
: "${SAVE_DIR:?Set SAVE_DIR to a fresh checkpoint root}"
export EXP_NAME="${EXP_NAME:-qwen38-qsa-sft-512k-fixed-64xblackwell}"
if [[ -e "${SAVE_DIR}/${EXP_NAME}" ]]; then
    echo 'Fixed first-attempt release requires a new SAVE_DIR/EXP_NAME; resume is unsupported.' >&2
    exit 2
fi
for argument in "$@"; do
    if [[ "$argument" == --load || "$argument" == --load=* ]]; then
        echo 'Use a fresh SAVE_DIR/EXP_NAME instead of overriding --load in the fixed-shape recipe.' >&2
        exit 2
    fi
done
export RELAX_MEMORY_RELEASE_FIRST_ATTEMPT_ONLY=1
export MAX_TOKENS_PER_GPU=131072 DATA_PAD_SIZE_MULTIPLIER=16384
export EVAL_SIZE=1 NUM_EPOCH="${NUM_EPOCH:-10}"
exec bash "${SCRIPT_DIR}/run-qwen38-qsa-sft-512k-64xgpu-blackwell.sh" --num-rollout 10 "$@"
