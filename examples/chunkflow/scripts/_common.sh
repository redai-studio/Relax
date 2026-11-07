#!/bin/bash
# Copyright (c) 2026 Relax Authors. All Rights Reserved.

# Shared submission plumbing; model and training arguments live in each recipe.
submit_training() {
    local train_command=(python3 -m relax.entrypoints.train "$@")
    if [ "${DRY_RUN:-0}" = 1 ]; then
        printf '%q ' "${train_command[@]}"
        printf '\n'
        return
    fi

    python3 "${RELAX_ROOT}/examples/chunkflow/setup.py" \
        --relax-root "${RELAX_ROOT}" \
        --check
    : "${RAY_JOBS_ADDRESS:?Set RAY_JOBS_ADDRESS to the existing Ray Jobs HTTP URL}"
    if [ -z "${RELAX_ENTRYPOINT_MODE:-}" ]; then
        source "${RELAX_ROOT}/scripts/entrypoint/ray-job.sh"
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
