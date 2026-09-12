#!/bin/bash

# Copyright (c) 2026 Relax Authors. All Rights Reserved.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
EXAMPLE_DIR="$(cd -- "${SCRIPT_DIR}/../.." &>/dev/null && pwd)"

usage() {
    echo "Default: 10.0.0.0/8. Override with --callback-network or NEMO_GYM_CALLBACK_ALLOWED_NETWORKS."
    cat <<'EOF'
Usage:
  start_workplace_assistant_gym_remote.sh \
    --gym-host <routable-local-ip> \
    [--callback-network <relax-callback-cidr>] \
    [--image <nemo-gym-image>] \
    [--repo-dir <docker-host-relax-checkout>] \
    [--port-base <gateway-port>] \
    [--ray-port <gcs-port>] \
    [--max-concurrency <positive-integer>] \
    [--verbose] \
    [--container-name <name>]
EOF
}

GYM_HOST=""
RELAX_CALLBACK_NETWORKS="${NEMO_GYM_CALLBACK_ALLOWED_NETWORKS:-}"
NEMO_GYM_IMAGE="${NEMO_GYM_IMAGE:-relax-nemo-gym:workplace-dev}"
NEMO_GYM_CONTAINER="${NEMO_GYM_CONTAINER:-nemo-gym-workplace}"
RELAX_REPO_ROOT="${RELAX_REPO_ROOT:-$(cd -- "${EXAMPLE_DIR}/../.." &>/dev/null && pwd)}"
WORKPLACE_ASSISTANT_PORT_BASE="${WORKPLACE_ASSISTANT_PORT_BASE:-29000}"
GYM_RAY_PORT="${GYM_RAY_PORT:-6382}"
GYM_DASHBOARD_PORT="${GYM_DASHBOARD_PORT:-28365}"
GYM_RAY_NUM_CPUS="${GYM_RAY_NUM_CPUS:-8}"
WORKPLACE_ASSISTANT_MAX_CONCURRENCY="${WORKPLACE_ASSISTANT_MAX_CONCURRENCY:-64}"
NEMO_GYM_START_TIMEOUT_S="${NEMO_GYM_START_TIMEOUT_S:-600}"
NEMO_GYM_VERBOSE="${NEMO_GYM_VERBOSE:-0}"

while [ "$#" -gt 0 ]; do
    case "$1" in
        --gym-host)
            GYM_HOST="${2:-}"
            shift 2
            ;;
        --callback-network)
            if [ -z "${2:-}" ] || [[ "${2}" == --* ]]; then
                echo "--callback-network requires a CIDR network" >&2
                exit 2
            fi
            RELAX_CALLBACK_NETWORKS="${RELAX_CALLBACK_NETWORKS:+${RELAX_CALLBACK_NETWORKS},}${2}"
            shift 2
            ;;
        --image)
            NEMO_GYM_IMAGE="${2:-}"
            shift 2
            ;;
        --repo-dir)
            RELAX_REPO_ROOT="${2:-}"
            shift 2
            ;;
        --port-base)
            WORKPLACE_ASSISTANT_PORT_BASE="${2:-}"
            shift 2
            ;;
        --ray-port)
            GYM_RAY_PORT="${2:-}"
            shift 2
            ;;
        --max-concurrency)
            WORKPLACE_ASSISTANT_MAX_CONCURRENCY="${2:-}"
            shift 2
            ;;
        --verbose)
            NEMO_GYM_VERBOSE=1
            shift
            ;;
        --container-name)
            NEMO_GYM_CONTAINER="${2:-}"
            shift 2
            ;;
        -h | --help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

RELAX_CALLBACK_NETWORKS="${RELAX_CALLBACK_NETWORKS:-10.0.0.0/8}"

if [ -z "${GYM_HOST}" ] \
    || [ -z "${RELAX_CALLBACK_NETWORKS}" ]; then
    usage >&2
    exit 2
fi
for host_name in GYM_HOST; do
    host_value="${!host_name}"
    if [[ "${host_value}" == *"://"* ]] || [[ "${host_value}" == *":"* ]] || [[ "${host_value}" == */* ]]; then
        echo "${host_name} must be a bare host or IP without scheme or port" >&2
        exit 2
    fi
done
case "${RELAX_REPO_ROOT}" in
    /*) ;;
    *)
        echo "--repo-dir must be an absolute Docker-host path" >&2
        exit 2
        ;;
esac
if ! [[ "${WORKPLACE_ASSISTANT_PORT_BASE}" =~ ^[0-9]+$ ]] \
    || ((10#${WORKPLACE_ASSISTANT_PORT_BASE} < 1 || 10#${WORKPLACE_ASSISTANT_PORT_BASE} > 65532)); then
    echo "--port-base must be an integer between 1 and 65532" >&2
    exit 2
fi
for port_name in GYM_RAY_PORT GYM_DASHBOARD_PORT; do
    port_value="${!port_name}"
    if ! [[ "${port_value}" =~ ^[0-9]+$ ]] || ((10#${port_value} < 1 || 10#${port_value} > 65535)); then
        echo "${port_name} must be an integer between 1 and 65535" >&2
        exit 2
    fi
done
if ! [[ "${WORKPLACE_ASSISTANT_MAX_CONCURRENCY}" =~ ^[1-9][0-9]*$ ]]; then
    echo "--max-concurrency must be a positive integer" >&2
    exit 2
fi
if ! [[ "${NEMO_GYM_START_TIMEOUT_S}" =~ ^[1-9][0-9]*$ ]]; then
    echo "NEMO_GYM_START_TIMEOUT_S must be a positive integer" >&2
    exit 2
fi
if [ "${NEMO_GYM_VERBOSE}" != "0" ] && [ "${NEMO_GYM_VERBOSE}" != "1" ]; then
    echo "NEMO_GYM_VERBOSE must be 0 or 1" >&2
    exit 2
fi

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

command -v docker >/dev/null 2>&1 || {
    echo "docker is required" >&2
    exit 2
}
command -v curl >/dev/null 2>&1 || {
    echo "curl is required" >&2
    exit 2
}
docker image inspect "${NEMO_GYM_IMAGE}" >/dev/null
if ! docker run --rm \
    --network none \
    --mount "type=bind,source=${RELAX_REPO_ROOT},target=/repo,readonly" \
    "${NEMO_GYM_IMAGE}" \
    test -s /repo/examples/nemo_gym_agentic/recipes/workplace-assistant/start_workplace_assistant_gym.sh; then
    echo "Invalid Relax checkout for the Docker host: ${RELAX_REPO_ROOT}" >&2
    exit 2
fi

if docker container inspect "${NEMO_GYM_CONTAINER}" >/dev/null 2>&1; then
    echo "Removing existing Workplace container ${NEMO_GYM_CONTAINER}..."
    docker rm -f "${NEMO_GYM_CONTAINER}" >/dev/null
fi

container_no_proxy="127.0.0.1,localhost,${GYM_HOST}"

echo "Creating remote Workplace Assistant container:"
echo "  container=${NEMO_GYM_CONTAINER}"
echo "  image=${NEMO_GYM_IMAGE}"
echo "  gym_url=http://${GYM_HOST}:${WORKPLACE_ASSISTANT_PORT_BASE}"
echo "  callback_networks=${RELAX_CALLBACK_NETWORKS:-<none>}"
echo "  repo_dir=${RELAX_REPO_ROOT}"
echo "  ray_port=${GYM_RAY_PORT}"
echo "  ray_num_cpus=${GYM_RAY_NUM_CPUS}"
echo "  max_concurrency=${WORKPLACE_ASSISTANT_MAX_CONCURRENCY}"
echo "  verbose=${NEMO_GYM_VERBOSE}"

docker create \
    --name "${NEMO_GYM_CONTAINER}" \
    --network host \
    --shm-size 12g \
    --mount "type=bind,source=${RELAX_REPO_ROOT},target=/opt/relax-integration,readonly" \
    --env GYM_HOST="${GYM_HOST}" \
    --env GYM_BIND_HOST="0.0.0.0" \
    --env GYM_RAY_PORT="${GYM_RAY_PORT}" \
    --env GYM_DASHBOARD_PORT="${GYM_DASHBOARD_PORT}" \
    --env GYM_RAY_NUM_CPUS="${GYM_RAY_NUM_CPUS}" \
    --env NEMO_GYM_CALLBACK_ALLOWED_NETWORKS="${RELAX_CALLBACK_NETWORKS}" \
    --env WORKPLACE_ASSISTANT_PORT_BASE="${WORKPLACE_ASSISTANT_PORT_BASE}" \
    --env WORKPLACE_ASSISTANT_MAX_CONCURRENCY="${WORKPLACE_ASSISTANT_MAX_CONCURRENCY}" \
    --env NEMO_GYM_VERBOSE="${NEMO_GYM_VERBOSE}" \
    --env NO_PROXY="${container_no_proxy}" \
    --env no_proxy="${container_no_proxy}" \
    "${NEMO_GYM_IMAGE}" \
    bash /opt/relax-integration/examples/nemo_gym_agentic/recipes/workplace-assistant/start_workplace_assistant_gym.sh \
    >/dev/null

docker start "${NEMO_GYM_CONTAINER}" >/dev/null

gateway_url="http://${GYM_HOST}:${WORKPLACE_ASSISTANT_PORT_BASE}"
echo "Waiting for ${gateway_url}/readyz ..."
deadline=$((SECONDS + NEMO_GYM_START_TIMEOUT_S))
next_progress=$SECONDS
while [ "${SECONDS}" -lt "${deadline}" ]; do
    if ! docker container inspect --format "{{.State.Running}}" "${NEMO_GYM_CONTAINER}" | grep -q true; then
        echo "Workplace Assistant container exited before becoming ready." >&2
        docker logs --tail 200 "${NEMO_GYM_CONTAINER}" >&2
        exit 1
    fi
    if ready_json="$(curl --noproxy "*" -fsS "${gateway_url}/readyz" 2>/dev/null)" \
        && printf '%s' "${ready_json}" | grep -q '"ready":true'; then
        printf '%s\n' "${ready_json}"
        echo "Workplace Assistant is ready at ${gateway_url}"
        echo "Inspect logs with: docker logs -f ${NEMO_GYM_CONTAINER}"
        exit 0
    fi
    if [ "${SECONDS}" -ge "${next_progress}" ]; then
        echo "Still starting; inspect with: docker logs -f ${NEMO_GYM_CONTAINER}"
        next_progress=$((SECONDS + 30))
    fi
    sleep 2
done

echo "Timed out waiting for Workplace Assistant readiness." >&2
docker logs --tail 200 "${NEMO_GYM_CONTAINER}" >&2
exit 1
