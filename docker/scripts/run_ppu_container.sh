#!/bin/bash
# Start a PPU image as a long-lived container with all 8 cards attached.
#
# Usage:
#   bash docker/scripts/run_ppu_container.sh          # start
#   docker exec -it relax-ppu bash                    # attach
#   bash docker/scripts/run_ppu_container.sh stop     # stop
set -euo pipefail

IMAGE="${IMAGE:-relax-sail:ppu-v0.1.0}"
CONTAINER="${CONTAINER:-relax-ppu}"
# host suits a dedicated box; bridge keeps the ray ports to yourself.
NETWORK="${NETWORK:-host}"
# Default to the repo this script lives in.
RELAX_SRC="${RELAX_SRC:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
# Extra host binds for weights / datasets; empty means "none".
CPFS_ROOT="${CPFS_ROOT:-}"
# Forwarded only when set: a container-wide value would win over the run
# script's own ${MODEL_DIR:-...} default.
MODEL_DIR="${MODEL_DIR:-}"
DATA_DIR="${DATA_DIR:-}"

if [ "${1:-start}" = "stop" ]; then
    # Idempotent only when the container is absent; a real removal failure must surface.
    if out=$(docker rm -f "${CONTAINER}" 2>&1); then
        echo "container ${CONTAINER} removed"
    elif printf '%s' "${out}" | grep -qi 'no such container'; then
        echo "container ${CONTAINER} not present"
    else
        echo "failed to remove container ${CONTAINER}: ${out}" >&2
        exit 1
    fi
    exit 0
fi

docker rm -f "${CONTAINER}" >/dev/null 2>&1 || true

# PPU device passthrough: 8 cards + the control / separator nodes.
DEVICES=""
for ((idx = 0; idx < 8; idx++)); do
    DEVICES+=" --device=/dev/alixpu_ppu${idx}:/dev/alixpu_ppu${idx}"
done
DEVICES+=" --device=/dev/alixpu:/dev/alixpu"
DEVICES+=" --device=/dev/alixpu_ctl:/dev/alixpu_ctl"
DEVICES+=" --device=/dev/alixpu_sep:/dev/alixpu_sep"

# relax detects PPU by /dev/alixpu and reports the cuda-family triple
# (nccl / CUDA_VISIBLE_DEVICES / GPU), so no extra env is needed.
docker run -d \
    --name "${CONTAINER}" \
    ${DEVICES} \
    --net="${NETWORK}" \
    --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
    --tmpfs /dev/shm:rw,nosuid,nodev,exec,size=64g \
    --ulimit nofile=262144:262144 \
    -v "${RELAX_SRC}:/workspace/Relax" \
    ${CPFS_ROOT:+-v "${CPFS_ROOT}:${CPFS_ROOT}"} \
    -e WORKDIR=/workspace \
    ${MODEL_DIR:+-e MODEL_DIR="${MODEL_DIR}"} \
    ${DATA_DIR:+-e DATA_DIR="${DATA_DIR}"} \
    -e PYTHONPATH=/workspace/Relax:/root/Megatron-LM \
    -w /workspace/Relax \
    "${IMAGE}" \
    sleep infinity

echo "container ${CONTAINER} started from ${IMAGE}"
