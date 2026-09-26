#!/bin/bash

# Copyright (c) 2026 Relax Authors. All Rights Reserved.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"

# REPO_ROOT is used to bind-mount this Relax checkout into the container at
# /opt/relax-integration so recipe-side edits take effect without rebuilding
# the image. It must be the path the DOCKER DAEMON sees (not the caller's
# shell path — those differ inside a dev container).
#
# Default: unset → the container uses the recipe scripts baked into the image
# by the Dockerfile's `COPY examples/nemo_gym_agentic ...`. This is what the
# README's 4-flag invocation relies on and needs no host-side setup.
#
# Opt-in: pass `--repo-dir /daemon/visible/absolute/path` to iterate on the
# recipe without rebuilding the image.
REPO_ROOT=""

usage() {
    echo "Usage: $0 --gym-host HOST [--callback-network CIDR] --data-jsonl PATH \\"
    echo "         [--image IMAGE] [--port-base PORT] [--repo-dir PATH]"
    echo
    echo "Options:"
    echo "  --gym-host HOST         Bare host or IP where the Gym runs (required)."
    echo "  --data-jsonl PATH       Absolute path to the reasoning-gym-cc JSONL (required)."
    echo "  --callback-network CIDR Allowed callback source CIDR (repeatable). Default 10.0.0.0/8."
    echo "                          Override via NEMO_GYM_CALLBACK_ALLOWED_NETWORKS."
    echo "  --image IMAGE           Gym docker image tag. Default relax-nemo-gym:reasoning-gym-cc-dev."
    echo "  --port-base PORT        Base port for gateway/agent/resource/head. Default 29200."
    echo "  --repo-dir PATH         Optional. Bind-mount this Relax checkout at /opt/relax-integration"
    echo "                          so recipe edits take effect without rebuilding the image. Must be"
    echo "                          the DOCKER DAEMON's absolute path (outer host, not dev container)."
}

GYM_HOST=""
RELAX_CALLBACK_NETWORKS="${NEMO_GYM_CALLBACK_ALLOWED_NETWORKS:-}"
REASONING_GYM_CC_DATA_JSONL=""
NEMO_GYM_IMAGE="${NEMO_GYM_IMAGE:-relax-nemo-gym:reasoning-gym-cc-dev}"
NEMO_GYM_CONTAINER="${NEMO_GYM_CONTAINER:-nemo-gym-reasoning-gym-cc}"
REASONING_GYM_CC_PORT_BASE="${REASONING_GYM_CC_PORT_BASE:-29200}"
REASONING_GYM_CC_MAX_CONCURRENCY="${REASONING_GYM_CC_MAX_CONCURRENCY:-64}"
NEMO_GYM_START_TIMEOUT_S="${NEMO_GYM_START_TIMEOUT_S:-600}"

while [ "$#" -gt 0 ]; do
    case "$1" in
        --gym-host) GYM_HOST="${2:-}"; shift 2 ;;
        --data-jsonl) REASONING_GYM_CC_DATA_JSONL="${2:-}"; shift 2 ;;
        --callback-network)
            if [ -z "${2:-}" ] || [[ "${2}" == --* ]]; then
                echo "--callback-network requires a CIDR network" >&2
                exit 2
            fi
            RELAX_CALLBACK_NETWORKS="${RELAX_CALLBACK_NETWORKS:+${RELAX_CALLBACK_NETWORKS},}${2}"
            shift 2
            ;;
        --image) NEMO_GYM_IMAGE="${2:-}"; shift 2 ;;
        --repo-dir) REPO_ROOT="${2:-}"; shift 2 ;;
        --port-base) REASONING_GYM_CC_PORT_BASE="${2:-}"; shift 2 ;;
        --max-concurrency) REASONING_GYM_CC_MAX_CONCURRENCY="${2:-}"; shift 2 ;;
        --container-name) NEMO_GYM_CONTAINER="${2:-}"; shift 2 ;;
        -h | --help) usage; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
    esac
done

RELAX_CALLBACK_NETWORKS="${RELAX_CALLBACK_NETWORKS:-10.0.0.0/8}"

if [ -z "${GYM_HOST}" ] || [ -z "${REASONING_GYM_CC_DATA_JSONL}" ] \
    || [ -z "${RELAX_CALLBACK_NETWORKS}" ]; then
    usage >&2
    exit 2
fi
for host_name in GYM_HOST; do
    host_value="${!host_name}"
    if [[ "${host_value}" == *"://"* ]] || [[ "${host_value}" == *":"* ]] || [[ "${host_value}" == */* ]]; then
        echo "${host_name} must be a bare host or IP" >&2
        exit 2
    fi
done
if [ -n "${REPO_ROOT}" ]; then
    case "${REPO_ROOT}" in /*) ;; *) echo "--repo-dir must be absolute: ${REPO_ROOT}" >&2; exit 2 ;; esac
fi
case "${REASONING_GYM_CC_DATA_JSONL}" in /*) ;;
    *) echo "--data-jsonl must be absolute: ${REASONING_GYM_CC_DATA_JSONL}" >&2; exit 2 ;;
esac
test -s "${REASONING_GYM_CC_DATA_JSONL}"
if [ -n "${RELAX_CALLBACK_NETWORKS}" ]; then
    python3 - "${RELAX_CALLBACK_NETWORKS}" <<'PY'
import ipaddress
import sys

for value in sys.argv[1].split(","):
    try:
        network = ipaddress.ip_network(value.strip(), strict=True)
        if network.prefixlen == 0:
            raise ValueError("default-route networks are not allowed")
    except ValueError as exc:
        raise SystemExit(f"--callback-network must contain valid CIDR networks: {value!r}: {exc}") from None
PY
fi

command -v docker >/dev/null
command -v curl >/dev/null
command -v jq >/dev/null
docker image inspect "${NEMO_GYM_IMAGE}" >/dev/null

# If --repo-dir was passed, verify it is visible to the docker daemon so the
# bind-mount actually surfaces the recipe files (dev-container shell paths
# usually differ from the daemon's paths).
if [ -n "${REPO_ROOT}" ]; then
    if ! docker run --rm \
        --network none \
        --entrypoint bash \
        --mount "type=bind,source=${REPO_ROOT},target=/repo,readonly" \
        "${NEMO_GYM_IMAGE}" \
        -ceu 'test -s /repo/examples/nemo_gym_agentic/recipes/reasoning-gym-cc/start_reasoning_gym_cc_gym.sh'; then
        echo "--repo-dir ${REPO_ROOT} is not visible to the Docker daemon." >&2
        echo "Pass the daemon-visible absolute path (usually the outer host's path), or omit --repo-dir to use the image's baked-in recipe files." >&2
        exit 2
    fi
fi

if docker container inspect "${NEMO_GYM_CONTAINER}" >/dev/null 2>&1; then
    owner="$(docker container inspect --format '{{index .Config.Labels "ai.relax.recipe"}}' "${NEMO_GYM_CONTAINER}")"
    if [ "${owner}" != "reasoning-gym-cc" ]; then
        echo "Refusing to replace ${NEMO_GYM_CONTAINER}: ai.relax.recipe=${owner:-<unset>}" >&2
        exit 2
    fi
    docker rm -f "${NEMO_GYM_CONTAINER}" >/dev/null
fi

data_parent="$(dirname -- "${REASONING_GYM_CC_DATA_JSONL}")"
container_no_proxy="127.0.0.1,localhost,0.0.0.0,${GYM_HOST}"
repo_mount_args=()
if [ -n "${REPO_ROOT}" ]; then
    repo_mount_args+=(--mount "type=bind,source=${REPO_ROOT},target=/opt/relax-integration,readonly")
fi
docker create \
    --name "${NEMO_GYM_CONTAINER}" \
    --label ai.relax.recipe=reasoning-gym-cc \
    --network host \
    --shm-size 12g \
    --cap-add SYS_ADMIN \
    --security-opt seccomp=unconfined \
    "${repo_mount_args[@]}" \
    --mount "type=bind,source=${data_parent},target=${data_parent},readonly" \
    --env GYM_HOST="${GYM_HOST}" \
    --env GYM_BIND_HOST="0.0.0.0" \
    --env NEMO_GYM_CALLBACK_ALLOWED_NETWORKS="${RELAX_CALLBACK_NETWORKS}" \
    --env REASONING_GYM_CC_DATA_JSONL="${REASONING_GYM_CC_DATA_JSONL}" \
    --env REASONING_GYM_CC_PORT_BASE="${REASONING_GYM_CC_PORT_BASE}" \
    --env REASONING_GYM_CC_MAX_CONCURRENCY="${REASONING_GYM_CC_MAX_CONCURRENCY}" \
    --env NO_PROXY="${container_no_proxy}" \
    --env no_proxy="${container_no_proxy}" \
    "${NEMO_GYM_IMAGE}" \
    bash /opt/relax-integration/examples/nemo_gym_agentic/recipes/reasoning-gym-cc/start_reasoning_gym_cc_gym.sh \
    >/dev/null
docker start "${NEMO_GYM_CONTAINER}" >/dev/null

gateway_url="http://${GYM_HOST}:${REASONING_GYM_CC_PORT_BASE}"
expected_gym_commit="a85670eb167ba9b48cc53a36a070eed815e6c40d"
deadline=$((SECONDS + NEMO_GYM_START_TIMEOUT_S))
while [ "${SECONDS}" -lt "${deadline}" ]; do
    if ! docker container inspect --format '{{.State.Running}}' "${NEMO_GYM_CONTAINER}" | grep -q true; then
        docker logs --tail 200 "${NEMO_GYM_CONTAINER}" >&2
        exit 1
    fi
    if ready_json="$(curl --noproxy "*" -fsS "${gateway_url}/readyz" 2>/dev/null)" \
        && printf '%s' "${ready_json}" | jq -e \
            --arg commit "${expected_gym_commit}" \
            '.ready == true and .gym_commit == $commit' >/dev/null; then
        printf '%s\n' "${ready_json}"
        echo "reasoning-gym-cc is ready at ${gateway_url}"
        exit 0
    fi
    sleep 2
done

docker logs --tail 200 "${NEMO_GYM_CONTAINER}" >&2
exit 1
