#!/bin/bash
# SGLang Judge Service (NPU version): auto-deploys on script start, auto-terminates on exit
# Usage: source "$(dirname "${BASH_SOURCE[0]}")/sglang_judge_service_npu.sh"
# Dependencies: MODEL_DIR and TIMESTAMP must be defined before sourcing

# Set default port, ensuring the variable has an initial value
SGLANG_JUDGE_PORT=${SGLANG_JUDGE_PORT:-30000}
SGLANG_JUDGE_MODEL_PATH="${MODEL_DIR}/Qwen2.5-1.5B-Instruct"
SGLANG_JUDGE_PID=""
LOG_FILE="logs/sglang_judge_${TIMESTAMP}.log"

cleanup_sglang_judge() {
    if [ -n "$SGLANG_JUDGE_PID" ] && kill -0 "$SGLANG_JUDGE_PID" 2>/dev/null; then
        echo "Stopping sglang judge service (PID: $SGLANG_JUDGE_PID)..."
        kill -9 "$SGLANG_JUDGE_PID" 2>/dev/null || true
        echo "Sglang judge service stopped successfully."
    fi
}

trap cleanup_sglang_judge EXIT

# Check if required dependencies are installed
if ! command -v curl &> /dev/null; then
    echo "Error: curl is required but not installed."
    exit 1
fi

# Check if model path exists
if [ ! -d "$SGLANG_JUDGE_MODEL_PATH" ]; then
    echo "Error: Model path not found - $SGLANG_JUDGE_MODEL_PATH"
    exit 1
fi

mkdir -p logs || { echo "Error: Failed to create logs directory"; exit 1; }


# Start SGLang service
echo "Starting sglang judge service on port $SGLANG_JUDGE_PORT..."
echo "Model path: $SGLANG_JUDGE_MODEL_PATH"
echo "Log file: $LOG_FILE"
python -m sglang.launch_server \
    --model-path "$SGLANG_JUDGE_MODEL_PATH" \
    --port "$SGLANG_JUDGE_PORT" \
    --api-key "EMPTY" \
    --device npu \
    --attention-backend ascend \
    --mem-fraction-static 0.1 \
    --disable-cuda-graph \
    > "$LOG_FILE" 2>&1 &

SGLANG_JUDGE_PID=$!
if [ -z "$SGLANG_JUDGE_PID" ]; then
    echo "Error: Failed to get PID for sglang judge service"
    exit 1
fi
echo "Sglang judge service started with PID: $SGLANG_JUDGE_PID"

# Wait for service to be ready
wait_for_sglang_ready() {
    local max_attempts=60       # Maximum wait attempts (total 5 minutes)
    local attempt=1
    local retry_interval=5      # Retry interval (seconds)
    local url="http://127.0.0.1:${SGLANG_JUDGE_PORT}/health"

    echo "Waiting for sglang judge service to be ready (max wait: $((max_attempts * retry_interval)) seconds)..."

    # Health check
    while [ $attempt -le $max_attempts ]; do
        local http_status
        http_status=$(curl -s -o /dev/null -w "%{http_code}" --max-time 2 "$url" 2>/dev/null || echo "000")

        if [ -z "$http_status" ]; then
            http_status="000"
        fi

        if [ "$http_status" = "200" ] || [ "$http_status" = "204" ]; then
            echo "Sglang judge service is ready (health check HTTP 200)."
            return 0
        fi

        # Only print retry info when not on the last attempt
        if [ $attempt -lt $max_attempts ]; then
            echo "  Attempt $attempt/$max_attempts: Waiting for judge model service (HTTP status: $http_status), retrying in ${retry_interval}s..."
        fi

        sleep $retry_interval
        attempt=$((attempt + 1))
    done

    # Timeout handling
    echo "Error: Sglang judge service failed to start within timeout (${max_attempts} attempts)."
    cleanup_sglang_judge
    return 1
}

# Run the wait logic; exit on failure
if ! wait_for_sglang_ready; then
    echo "Aborting due to sglang judge service startup failure."
    exit 1
fi


export DEEPEYES_JUDGE_API_KEY="EMPTY"
export DEEPEYES_JUDGE_BASE_URL="http://127.0.0.1:${SGLANG_JUDGE_PORT}/v1"
export DEEPEYES_JUDGE_MODELS="Qwen2.5-1.5B-Instruct"

if [ -n "${RUNTIME_ENV_JSON:-}" ]; then
    json_escape() {
        local value="${1:-}"
        value=${value//\\/\\\\}
        value=${value//\"/\\\"}
        value=${value//$'\n'/\\n}
        value=${value//$'\r'/\\r}
        value=${value//$'\t'/\\t}
        printf '%s' "$value"
    }

    runtime_env_prefix="${RUNTIME_ENV_JSON%$'\n}\n}'}"
    export RUNTIME_ENV_JSON="${runtime_env_prefix},
   \"DEEPEYES_JUDGE_API_KEY\": \"$(json_escape "${DEEPEYES_JUDGE_API_KEY}")\",
   \"DEEPEYES_JUDGE_BASE_URL\": \"$(json_escape "${DEEPEYES_JUDGE_BASE_URL}")\",
   \"DEEPEYES_JUDGE_MODELS\": \"$(json_escape "${DEEPEYES_JUDGE_MODELS}")\"
}
}"
fi

ray() {
    if [ "$1" = "job" ] && [ "$2" = "submit" ] && [ -n "${RUNTIME_ENV_JSON:-}" ]; then
        local arg=""
        for arg in "$@"; do
            if [ "$arg" = "--runtime-env-json" ] || [[ "$arg" == --runtime-env-json=* ]]; then
                command ray "$@"
                return
            fi
        done
        command ray job submit ${RAY_NO_WAIT:+--no-wait} --runtime-env-json="${RUNTIME_ENV_JSON}" "${@:3}"
        return
    fi
    command ray "$@"
}

echo "Sglang judge service is fully ready for use."
