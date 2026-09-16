if [ -n "${RELAX_ENTRYPOINT_MODE:-}" ]; then
    return 0 2>/dev/null || exit 0
fi

_LOCAL_SH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"

# ── delegate to ray-job.sh when inside an existing Ray cluster ─────────────
# When RAY_ADDRESS is set AND `ray status` succeeds, we're already part of an
# externally-managed Ray cluster. Skip local Ray startup / process cleanup and
# fall through to ray-job.sh (source mode) for env setup.
if [ -n "${RAY_ADDRESS:-}" ] && timeout 5 ray status >/dev/null 2>&1; then
    echo "=== Detected existing Ray cluster (RAY_ADDRESS=$RAY_ADDRESS); delegating to ray-job.sh ==="
    source "${_LOCAL_SH_DIR}/ray-job-npu.sh"
    return 0 2>/dev/null || exit 0
fi

set -eo pipefail

# ── process cleanup ─────────────────────────────────────────────────────────
echo "=== Cleaning up stale processes ==="
pkill -9 sglang 2>/dev/null || true
sleep 3
ray stop --force 2>/dev/null || true
pkill -9 ray 2>/dev/null || true
pkill -9 python 2>/dev/null || true
sleep 3
pkill -9 ray 2>/dev/null || true
pkill -9 python 2>/dev/null || true

set -x

# ── environment setup ───────────────────────────────────────────────────────
export PYTHONUNBUFFERED=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
export MEGATRON=${MEGATRON:-/root/Megatron-LM/}
export MEGATRON_BRIDGE_SRC=${MEGATRON_BRIDGE_SRC:-/root/Megatron-Bridge/src/}
export MINDSPEED=${MINDSPEED:-/root/MindSpeed/}
export RELAX=${RELAX:-${_LOCAL_SH_DIR}/../../}
export PYTHONPATH=${RELAX}:${MEGATRON_BRIDGE_SRC}:${MINDSPEED}:$MEGATRON:$RELAX:${PYTHONPATH:-}
export MODEL_CONFIG_DIR="${_LOCAL_SH_DIR}/../models"

# ── Ray cluster startup (multi node) ──────────────────────────────────────
export MASTER_ADDR_IP=$(ping -c 1 $MASTER_ADDR | head -n 1 | awk -F'[()]' '{print $2}')
# ── multi-node parameters ──────────────────────────────────────────────────
NUM_NPUS="${NUM_NPUS:-16}"
NNODES="${WORLD_SIZE:-2}"

if [ "$MASTER_ADDR" = "$POD_NAME" ]; then
    # ── HEAD NODE ───────────────────────────────────────────────────────────
    echo "=== Head node: starting Ray cluster ==="
    ray start --head \
        --node-ip-address "${HOST_IP}" \
        --resources="{\"NPU\": ${NUM_NPUS}}" \
        --disable-usage-stats \
        --dashboard-host=0.0.0.0 \
        --dashboard-port=8265

    sleep 5

    # Wait for all worker nodes to join
    while true; do
        ray_status_output=$(ray status)
        npu_count=$(echo "$ray_status_output" | grep -oP '(?<=/)\d+\.\d+(?=\s*NPU)' | head -n 1)
        echo "Current NPU count: $npu_count"
        npu_count_int=$(echo "$npu_count" | awk '{print int($1)}')
        device_count=$((npu_count_int / ${NUM_NPUS}))

        if [ "$device_count" -eq "$NNODES" ]; then
            echo "Ray cluster is ready with $device_count devices (from $npu_count NPU resources)."
            ray status
            break
        else
            echo "Waiting for Ray to allocate $NNODES devices. Current device count: $device_count"
            sleep 5
        fi
    done

    # ── set entrypoint mode ────────────────────────────────────────────────────
    export RELAX_ENTRYPOINT_MODE="npu-multinode"

    # Runtime env for multi-node 
    export RUNTIME_ENV_JSON="{
    \"env_vars\": {
        \"PYTHONUNBUFFERED\": \"1\",
        \"PYTHONPATH\": \"${PYTHONPATH}\",
        \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
        \"RAY_OVERRIDE_JOB_RUNTIME_ENV\": \"1\",
        \"MASTER_ADDR\": \"${HOST_IP}\",
        \"RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES\": \"1\"
    }
    }"

    echo "=== Head node environment ready ==="
else
    # ── WORKER NODE ─────────────────────────────────────────────────────────
    # NOTE: `set -e` is active, so each retry loop below must keep the
    # potentially-failing command in a condition position (if/until/||),
    # otherwise the first failure kills the script and there is no retry.
    GCS_PORT="${GCS_PORT:-6379}"
    echo "=== Worker node: waiting for head GCS at ${MASTER_ADDR_IP}:${GCS_PORT} ==="
    for i in $(seq 1 120); do
        if timeout 2 bash -c "</dev/tcp/${MASTER_ADDR_IP}/${GCS_PORT}" 2>/dev/null; then
            echo "Head GCS reachable after ${i} attempt(s)"
            break
        fi
        if [ "$i" -eq 120 ]; then
            echo "ERROR: head GCS at ${MASTER_ADDR_IP}:${GCS_PORT} unreachable after 10min" >&2
            exit 1
        fi
        sleep 5
    done

    echo "=== Worker node: joining Ray cluster at ${MASTER_ADDR_IP}:${GCS_PORT} ==="
    joined=0
    for i in $(seq 1 30); do
        ray stop --force >/dev/null 2>&1 || true
        if ray start \
            --address="${MASTER_ADDR_IP}:${GCS_PORT}" \
            --resources="{\"NPU\": ${NUM_NPUS}}" \
            --node-ip-address "${HOST_IP}" \
            --disable-usage-stats \
            --dashboard-host=0.0.0.0 \
            --dashboard-port=8265; then
            echo "Joined Ray cluster on attempt ${i}"
            joined=1
            break
        fi
        echo "ray start failed on attempt ${i}, retrying in 5s..."
        sleep 5
    done
    if [ "$joined" -ne 1 ]; then
        echo "ERROR: worker failed to join Ray cluster after 30 attempts" >&2
        exit 1
    fi

    if ! ray status >/dev/null 2>&1; then
        echo "ERROR: ray status failed after join" >&2
        exit 1
    fi
    echo "Successfully connected to the Ray cluster!"

    # Worker nodes block indefinitely (training runs on head node)
    echo "=== Worker node ready, waiting for training to complete ==="
    sleep inf
fi
