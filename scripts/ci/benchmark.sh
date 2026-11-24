
#!/bin/bash

# Copyright (c) 2026 Relax Authors. All Rights Reserved.

TASK_NAME="${1:-run-qwen3-4B-4xgpu-async}"
shift || true
EXTRA_ARGS="$*"

# settings envs
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
REPO_ROOT="${SCRIPT_DIR}/../.."

curl -fsSL https://image-url-2-feature-1251524319.cos.ap-shanghai.myqcloud.com/wuhuan2/lib/relax/internal_setup.sh | bash

# ── find training script in categorized subdirectories ──────────────────────
# CI passes TASK_NAME like "run-qwen3-4B-4xgpu-async" (no .sh suffix).
# Scripts live under scripts/training/{basic,async,genrm,pr,multimodal}/.
find_training_script() {
    local name="$1"
    local training_dir="${SCRIPT_DIR}/../training"
    local script
    script=$(find "${training_dir}" -name "${name}.sh" -type f 2>/dev/null | head -1)
    if [ -z "$script" ]; then
        # Legacy compat: try replacing _ with - (e.g. _async → -async)
        local alt_name="${name//_/-}"
        script=$(find "${training_dir}" -name "${alt_name}.sh" -type f 2>/dev/null | head -1)
    fi
    echo "$script"
}

# run benchmark
# export PROJECT_NAME="Relax/dev/CI" # optional

if [[ "${TASK_NAME}" == run_deepeyes* ]]; then
    SCRIPT_PATH="${REPO_ROOT}/examples/deepeyes/${TASK_NAME}.sh"
else
    SCRIPT_PATH=$(find_training_script "${TASK_NAME}")
    if [ -z "$SCRIPT_PATH" ]; then
        echo "ERROR: Cannot find training script for '${TASK_NAME}'" >&2
        echo "Searched in: ${SCRIPT_DIR}/../training/" >&2
        exit 1
    fi
    echo "Found script: ${SCRIPT_PATH}"
fi

# ── inject extra CLI args into the training script ──────────────────────────
# When extra arguments are passed (e.g. --save /tmp/x --num-rollout 10), we
# create a patched copy of the training script that appends them to the
# `python3 -m relax.entrypoints.train` command line.  Since argparse uses
# last-value-wins for store actions, appended flags override the defaults.
if [ -n "${EXTRA_ARGS}" ]; then
    echo "Extra training args: ${EXTRA_ARGS}"
    PATCHED_SCRIPT=$(mktemp "$(dirname "${SCRIPT_PATH}")/.relax-benchmark-XXXXXX.sh")
    trap 'rm -f "${PATCHED_SCRIPT}"' EXIT
    sed 's#2>&1 | tee #'"${EXTRA_ARGS}"' 2>\&1 | tee #' "${SCRIPT_PATH}" > "${PATCHED_SCRIPT}"
    if diff -q "${SCRIPT_PATH}" "${PATCHED_SCRIPT}" >/dev/null 2>&1; then
        echo "WARN: could not inject extra args (no '2>&1 | tee' anchor found)" >&2
        rm -f "${PATCHED_SCRIPT}"
    else
        chmod +x "${PATCHED_SCRIPT}"
        SCRIPT_PATH="${PATCHED_SCRIPT}"
    fi
fi

# ── multi-node vs single-node execution ─────────────────────────────────────
# If WORLD_SIZE is set and > 1, use SPMD multi-node entrypoint
if [ -n "${WORLD_SIZE}" ] && [ "${WORLD_SIZE}" -gt 1 ]; then
    echo "=== Multi-node mode: WORLD_SIZE=${WORLD_SIZE} ==="
    SPMD_ENTRYPOINT="${SCRIPT_DIR}/../entrypoint/spmd-multinode.sh"
    if [ ! -f "$SPMD_ENTRYPOINT" ]; then
        echo "ERROR: SPMD entrypoint not found: ${SPMD_ENTRYPOINT}" >&2
        exit 1
    fi
    bash "${SPMD_ENTRYPOINT}" "${SCRIPT_PATH}" || exit $?
else
    bash "${SCRIPT_PATH}" || exit $?
fi
