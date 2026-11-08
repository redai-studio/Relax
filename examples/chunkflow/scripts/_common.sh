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
        --relax-root "${RELAX_ROOT}" \
        --check
    if [ -z "${RELAX_ENTRYPOINT_MODE:-}" ]; then
        if [ -n "${RAY_ADDRESS:-}${RAY_JOBS_ADDRESS:-}" ]; then
            : "${RAY_JOBS_ADDRESS:?Set RAY_JOBS_ADDRESS to the existing Ray Jobs HTTP URL}"
            # An explicit cluster must not fall back to local.sh's node cleanup.
            if ! timeout 5 ray status --address="${RAY_ADDRESS:-auto}"; then
                echo "Cannot reach the configured Ray cluster; no local cluster was started." >&2
                return 1
            fi
            entrypoint=ray-job
        fi
        export RELAX_RUN_SCRIPT_PATH="${SCRIPT_DIR}/${BASH_SOURCE[1]##*/}"
        if [ "$entrypoint" != local ]; then
            # Standard cluster entrypoints re-enter the recipe with their mode set.
            exec bash "${RELAX_ROOT}/scripts/entrypoint/${entrypoint}.sh" "${RELAX_RUN_SCRIPT_PATH}"
        fi
        source "${RELAX_ROOT}/scripts/entrypoint/${entrypoint}.sh"
    fi
    if [[ "${RELAX_ENTRYPOINT_MODE:-}" = local || "${RELAX_ENTRYPOINT_MODE:-}" = spmd-multinode ]]; then
        export RAY_JOBS_ADDRESS="${RAY_JOBS_ADDRESS:-http://127.0.0.1:8265}"
    else
        : "${RAY_JOBS_ADDRESS:?Set RAY_JOBS_ADDRESS to the existing Ray Jobs HTTP URL}"
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
        --address="${RAY_JOBS_ADDRESS}" \
        ${WORKING_DIR:+--working-dir "${WORKING_DIR}"} \
        --runtime-env-json="${RUNTIME_ENV_JSON}" \
        -- "${train_command[@]}" 2>&1 | tee "${SCRIPT_DIR}/log/${EXP_NAME}-${now}.log"
}
