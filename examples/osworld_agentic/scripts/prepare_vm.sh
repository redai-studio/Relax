#!/bin/bash
# Copyright (c) 2026 Relax Authors. All Rights Reserved.
#
# Stage the OSWorld Docker VM assets for the agentic pipeline.
#
# Two things need to exist on every host before Relax can spin up sessions:
#
#   1. The `happysixd/osworld-docker` image (KVM-aware QEMU wrapper).
#      Each session's DesktopEnv boots one container from this image.
#
#   2. The Ubuntu.qcow2 disk (~9 GB), which upstream pulls from HuggingFace
#      the first time a session runs. The pull is slow enough that N parallel
#      sessions racing on it can wedge the whole rollout; do it once, up front.
#
# Optional: copy the qcow2 to /dev/shm to cut per-session cold-start from
# ~30-40s down to ~10s. Set OSWORLD_STAGE_TMPFS=1 to enable. Requires enough
# free tmpfs (df -h /dev/shm).
#
# Env inputs:
#   OSWORLD_CACHE_DIR  - where docker_vm_data/ lives; default /tmp/osworld_cache
#   OSWORLD_STAGE_TMPFS - "1" to copy qcow2 to /dev/shm/osworld
#   HF_ENDPOINT        - optional hf-mirror.com override; the vendor manager
#                        rewrites the URL automatically when this contains
#                        "hf-mirror.com".

set -euo pipefail

OSWORLD_CACHE_DIR="${OSWORLD_CACHE_DIR:-/tmp/osworld_cache}"
DOCKER_IMAGE="happysixd/osworld-docker"
QCOW2_URL="${OSWORLD_QCOW2_URL:-https://huggingface.co/datasets/xlangai/ubuntu_osworld/resolve/main/Ubuntu.qcow2.zip}"

if [ -n "${HF_ENDPOINT:-}" ] && [[ "${HF_ENDPOINT}" == *"hf-mirror.com"* ]]; then
    QCOW2_URL="${QCOW2_URL/huggingface.co/hf-mirror.com}"
fi

log() { echo "[$(date +%H:%M:%S)] $*"; }

# ── 1. docker image ────────────────────────────────────────────────────────
log "Checking docker image ${DOCKER_IMAGE}"
if ! docker image inspect "${DOCKER_IMAGE}" > /dev/null 2>&1; then
    log "Pulling ${DOCKER_IMAGE} (this can take a while)"
    docker pull "${DOCKER_IMAGE}"
else
    log "Image already present."
fi

# ── 2. qcow2 disk ──────────────────────────────────────────────────────────
mkdir -p "${OSWORLD_CACHE_DIR}/docker_vm_data"
cd "${OSWORLD_CACHE_DIR}/docker_vm_data"

QCOW2_ZIP="$(basename "${QCOW2_URL}")"
QCOW2_FILE="${QCOW2_ZIP%.zip}"

if [ ! -f "${QCOW2_FILE}" ]; then
    if [ ! -f "${QCOW2_ZIP}" ]; then
        log "Downloading ${QCOW2_URL}"
        # -C - resumes if partially downloaded, --retry buys headroom on flaky nets.
        curl -L --retry 5 --retry-delay 10 -C - -o "${QCOW2_ZIP}" "${QCOW2_URL}"
    fi
    log "Unzipping ${QCOW2_ZIP}"
    unzip -o "${QCOW2_ZIP}"
fi

log "qcow2 ready at $(pwd)/${QCOW2_FILE}"

# ── 3. optional tmpfs staging ──────────────────────────────────────────────
if [ "${OSWORLD_STAGE_TMPFS:-0}" = "1" ]; then
    TMPFS_TARGET="/dev/shm/osworld/${QCOW2_FILE}"
    mkdir -p "$(dirname "${TMPFS_TARGET}")"
    QCOW2_SIZE=$(stat -c %s "${QCOW2_FILE}")
    SHM_FREE=$(df --output=avail -B1 /dev/shm | tail -1)
    if [ "${SHM_FREE}" -lt "${QCOW2_SIZE}" ]; then
        log "WARN: /dev/shm has ${SHM_FREE}B free, qcow2 is ${QCOW2_SIZE}B — skipping tmpfs staging."
    elif [ ! -f "${TMPFS_TARGET}" ] || [ "$(stat -c %s "${TMPFS_TARGET}")" != "${QCOW2_SIZE}" ]; then
        log "Copying qcow2 to tmpfs at ${TMPFS_TARGET}"
        cp --sparse=always "${QCOW2_FILE}" "${TMPFS_TARGET}"
    else
        log "tmpfs copy already up-to-date."
    fi
fi

log "Done."
