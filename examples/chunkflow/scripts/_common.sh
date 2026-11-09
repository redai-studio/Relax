#!/bin/bash
# Copyright (c) 2026 Relax Authors. All Rights Reserved.

# Shared entrypoint and submission plumbing; training arguments live in each recipe.
submit_training() {
    local entrypoint="$1"
    shift
    local train_command=(python3 -m relax.entrypoints.train "$@")
    if [ "${DRY_RUN:-0}" = 1 ]; then
        printf '%q ' "${train_command[@]}"
        printf '\n'
        return
    fi

    python3 "${RELAX_ROOT}/examples/chunkflow/setup.py" \
        --relax-root "${RELAX_ROOT}"
    if [ -z "${RELAX_ENTRYPOINT_MODE:-}" ]; then
        local ray_status=0
        timeout 10 ray status --address="${RAY_ADDRESS:-auto}" >/dev/null 2>&1 || ray_status=$?
        if [ "$ray_status" = 0 ]; then
            entrypoint=ray-job
            export RELAX_SKIP_CLUSTER_CLEANUP=1
        elif [ -n "${RAY_ADDRESS:-}${RAY_JOBS_ADDRESS:-}" ] || [ "$ray_status" = 124 ] || pgrep -x raylet >/dev/null; then
            echo "Cannot reach the existing Ray cluster; no local cluster was started." >&2
            return 1
        fi
        export RELAX_RUN_SCRIPT_PATH="${SCRIPT_DIR}/${BASH_SOURCE[1]##*/}"
        if [ "$entrypoint" = spmd-multinode ]; then
            : "${MASTER_ADDR:?Set MASTER_ADDR to the head node POD_NAME}"
            : "${POD_NAME:?Set POD_NAME to the current node hostname}"
            : "${HOST_IP:?Set HOST_IP to the current node reachable IP}"
            export WORLD_SIZE="${WORLD_SIZE:-4}" NUM_GPUS="${NUM_GPUS:-8}"
            if ! [[ "$WORLD_SIZE" =~ ^[1-9][0-9]*$ && "$NUM_GPUS" =~ ^[1-9][0-9]*$ ]] ||
                (( WORLD_SIZE * NUM_GPUS != 32 )); then
                echo "This recipe requires WORLD_SIZE (nodes) * NUM_GPUS (per node) = 32." >&2
                return 1
            fi
        fi
        if [ "$entrypoint" != local ]; then
            # Standard cluster entrypoints re-enter the recipe with their mode set.
            exec bash "${RELAX_ROOT}/scripts/entrypoint/${entrypoint}.sh" "${RELAX_RUN_SCRIPT_PATH}"
        fi
        source "${RELAX_ROOT}/scripts/entrypoint/${entrypoint}.sh"
    fi
    export RAY_NO_WAIT="${RAY_NO_WAIT-}"
    export WORKING_DIR="${WORKING_DIR:-${RELAX_ROOT}}"

    # A submitter's environment is not automatically inherited by Ray workers.
    RUNTIME_ENV_JSON=$(python3 - <<'PY'
import json
import os
import sys

runtime = json.loads(os.environ["RUNTIME_ENV_JSON"])
environment = runtime.setdefault("env_vars", {})
environment.update({key: value for key, value in os.environ.items() if key.startswith(("RELAX_CHUNKFLOW_", "RELAX_QWEN38_", "MCORE_QSA_", "QWEN38_PLE_"))})
sys.stdout.write(json.dumps(runtime))
PY
    )
    export RUNTIME_ENV_JSON
    mkdir -p "${SCRIPT_DIR}/log"
    cd "${RELAX_ROOT}"
    ray job submit ${RAY_NO_WAIT:+--no-wait} \
        --address="${RAY_JOBS_ADDRESS:-${RAY_ADDRESS:-auto}}" \
        ${WORKING_DIR:+--working-dir "${WORKING_DIR}"} \
        --runtime-env-json="${RUNTIME_ENV_JSON}" \
        -- "${train_command[@]}" 2>&1 | tee "${SCRIPT_DIR}/log/${EXP_NAME}-${now}.log"
}
