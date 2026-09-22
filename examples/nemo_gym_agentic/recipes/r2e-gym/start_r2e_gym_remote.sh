#!/bin/bash

# Copyright (c) 2026 Relax Authors. All Rights Reserved.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
EXAMPLE_DIR="$(cd -- "${SCRIPT_DIR}/../.." &>/dev/null && pwd)"

usage() {
    cat <<'EOF'
Usage:
  start_r2e_gym_remote.sh \
    --gym-host <routable-local-ip> \
    --data-dir <prepared-r2e-directory> \
    --mode <golden|train> \
    [--sif-dir <docker-host-shared-sif-directory>] \
    [--sif-prefix <filename-prefix>] \
    [--callback-network <relax-callback-cidr>] \
    [--image <nemo-gym-image>] \
    [--repo-dir <docker-host-relax-checkout>] \
    [--proxy <http-proxy-url>] \
    [--callback-proxy <http-proxy-url>] \
    [--callback-timeout-s <positive-integer>] \
    [--port-base <port>] \
    [--max-concurrency <positive-integer>] \
    [--gym-cpus <positive-integer>] \
    [--verbose] \
    [--container-name <name>]

Default: 10.0.0.0/8. Override with --callback-network (repeatable) or NEMO_GYM_CALLBACK_ALLOWED_NETWORKS.
The networks must contain the callback IP, including the golden verifier URL when used.
Ports: gateway=base, agent=base+1, head=base+3. Default base: 28100.
Override via --port-base or R2E_GYM_PORT_BASE; valid range: 1-65532.
EOF
}

GYM_HOST=""
R2E_DATA_DIR=""
R2E_GYM_MODE=""
R2E_GYM_PORT_BASE="${R2E_GYM_PORT_BASE:-28100}"
R2E_GYM_SIF_DIR="${R2E_GYM_SIF_DIR:-}"
R2E_GYM_SIF_PREFIX="${R2E_GYM_SIF_PREFIX:-}"
RELAX_CALLBACK_NETWORKS="${NEMO_GYM_CALLBACK_ALLOWED_NETWORKS:-}"
NEMO_GYM_IMAGE="${NEMO_GYM_IMAGE:-relax-nemo-gym:r2e-dev}"
NEMO_GYM_CONTAINER="${NEMO_GYM_CONTAINER:-nemo-gym-r2e-local}"
NEMO_GYM_HTTP_PROXY="${NEMO_GYM_HTTP_PROXY:-${https_proxy:-${HTTPS_PROXY:-${http_proxy:-${HTTP_PROXY:-}}}}}"
NEMO_GYM_NO_PROXY="${NEMO_GYM_NO_PROXY:-${no_proxy:-${NO_PROXY:-}}}"
NEMO_GYM_CALLBACK_PROXY="${NEMO_GYM_CALLBACK_PROXY:-}"
NEMO_GYM_CALLBACK_TIMEOUT_S="${NEMO_GYM_CALLBACK_TIMEOUT_S:-600}"
NEMO_GYM_START_TIMEOUT_S="${NEMO_GYM_START_TIMEOUT_S:-1800}"
GYM_RAY_NUM_CPUS="${GYM_RAY_NUM_CPUS:-8}"
R2E_GYM_MAX_CONCURRENCY="${R2E_GYM_MAX_CONCURRENCY:-16}"
R2E_GYM_MAX_TURNS="${R2E_GYM_MAX_TURNS:-50}"
NEMO_GYM_VERBOSE="${NEMO_GYM_VERBOSE:-0}"
RELAX_REPO_ROOT="${RELAX_REPO_ROOT:-$(cd -- "${EXAMPLE_DIR}/../.." &>/dev/null && pwd)}"

while [ "$#" -gt 0 ]; do
    case "$1" in
        --gym-host)
            GYM_HOST="${2:-}"
            shift 2
            ;;
        --data-dir)
            R2E_DATA_DIR="${2:-}"
            shift 2
            ;;
        --mode)
            R2E_GYM_MODE="${2:-}"
            shift 2
            ;;
        --sif-dir)
            R2E_GYM_SIF_DIR="${2:-}"
            shift 2
            ;;
        --sif-prefix)
            R2E_GYM_SIF_PREFIX="${2:-}"
            shift 2
            ;;
        --callback-network)
            if [ -z "${2:-}" ] || [[ "${2}" == --* ]]; then
                echo "--callback-network requires a CIDR network" >&2
                exit 2
            fi
            if [ -n "${RELAX_CALLBACK_NETWORKS}" ]; then
                RELAX_CALLBACK_NETWORKS="${RELAX_CALLBACK_NETWORKS},${2:-}"
            else
                RELAX_CALLBACK_NETWORKS="${2:-}"
            fi
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
        --proxy)
            NEMO_GYM_HTTP_PROXY="${2:-}"
            shift 2
            ;;
        --callback-proxy)
            NEMO_GYM_CALLBACK_PROXY="${2:-}"
            shift 2
            ;;
        --callback-timeout-s)
            NEMO_GYM_CALLBACK_TIMEOUT_S="${2:-}"
            shift 2
            ;;
        --port-base)
            if [ -z "${2:-}" ] || [[ "${2}" == --* ]]; then
                echo "--port-base requires an integer between 1 and 65532" >&2
                exit 2
            fi
            R2E_GYM_PORT_BASE="${2}"
            shift 2
            ;;
        --max-concurrency)
            R2E_GYM_MAX_CONCURRENCY="${2:-}"
            shift 2
            ;;
        --gym-cpus)
            GYM_RAY_NUM_CPUS="${2:-}"
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

if [ -z "${GYM_HOST}" ] || [ -z "${R2E_DATA_DIR}" ] || [ -z "${R2E_GYM_MODE}" ]; then
    usage >&2
    exit 2
fi
if [[ "${GYM_HOST}" == *"://"* ]] || [[ "${GYM_HOST}" == *":"* ]]; then
    echo "--gym-host must be a bare routable host or IP without scheme or port" >&2
    exit 2
fi
case "${R2E_DATA_DIR}" in
    /*) ;;
    *)
        echo "--data-dir must be an absolute path" >&2
        exit 2
        ;;
esac
R2E_GYM_SIF_DIR="${R2E_GYM_SIF_DIR:-${R2E_DATA_DIR}/sif}"
case "${R2E_GYM_SIF_DIR}" in
    /*) ;;
    *)
        echo "--sif-dir must be an absolute Docker-host path" >&2
        exit 2
        ;;
esac
if [[ "${R2E_GYM_SIF_PREFIX}" == */* ]]; then
    echo "--sif-prefix must be a filename prefix without '/'" >&2
    exit 2
fi
case "${RELAX_REPO_ROOT}" in
    /*) ;;
    *)
        echo "--repo-dir must be an absolute path" >&2
        exit 2
        ;;
esac
case "${R2E_GYM_MODE}" in
    golden)
        R2E_GYM_VERIFY_GOLDEN_PATCH=1
        ;;
    train)
        R2E_GYM_VERIFY_GOLDEN_PATCH=0
        ;;
    *)
        echo "--mode must be golden or train" >&2
        exit 2
        ;;
esac
: "${RELAX_CALLBACK_NETWORKS:?Set --callback-network or NEMO_GYM_CALLBACK_ALLOWED_NETWORKS}"
if [ -n "${RELAX_CALLBACK_NETWORKS}" ]; then
    command -v python3 >/dev/null 2>&1 || {
        echo "python3 is required to validate --callback-network" >&2
        exit 2
    }
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
if ! [[ "${NEMO_GYM_START_TIMEOUT_S}" =~ ^[1-9][0-9]*$ ]]; then
    echo "NEMO_GYM_START_TIMEOUT_S must be a positive integer" >&2
    exit 2
fi
if ! [[ "${GYM_RAY_NUM_CPUS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "GYM_RAY_NUM_CPUS must be a positive integer" >&2
    exit 2
fi
if ! [[ "${R2E_GYM_PORT_BASE}" =~ ^[1-9][0-9]{0,4}$ ]] || ((R2E_GYM_PORT_BASE > 65532)); then
    echo "--port-base must be an integer between 1 and 65532" >&2
    exit 2
fi
if ! [[ "${R2E_GYM_MAX_CONCURRENCY}" =~ ^[1-9][0-9]*$ ]]; then
    echo "--max-concurrency must be a positive integer" >&2
    exit 2
fi
if [ "${NEMO_GYM_VERBOSE}" != "0" ] && [ "${NEMO_GYM_VERBOSE}" != "1" ]; then
    echo "NEMO_GYM_VERBOSE must be 0 or 1" >&2
    exit 2
fi
if ! [[ "${NEMO_GYM_CALLBACK_TIMEOUT_S}" =~ ^[1-9][0-9]*$ ]]; then
    echo "--callback-timeout-s must be a positive integer" >&2
    exit 2
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
test -s "${R2E_DATA_DIR}/r2e_gym_train.jsonl" || {
    echo "Missing prepared dataset: ${R2E_DATA_DIR}/r2e_gym_train.jsonl" >&2
    exit 2
}
first_sif="$(find -L "${R2E_GYM_SIF_DIR}" -maxdepth 1 -type f \
    -name "${R2E_GYM_SIF_PREFIX}*.sif" -size +0c -print -quit)"
if [ -z "${first_sif}" ]; then
    echo "No prepared ${R2E_GYM_SIF_PREFIX}*.sif found under ${R2E_GYM_SIF_DIR}" >&2
    exit 2
fi

sif_mount_args=()
docker_preflight_mount_args=(
    --mount "type=bind,source=${RELAX_REPO_ROOT},target=/opt/relax-integration,readonly"
    --mount "type=bind,source=${R2E_DATA_DIR},target=${R2E_DATA_DIR},readonly"
)
if [ "${R2E_GYM_SIF_DIR}" != "${R2E_DATA_DIR}/sif" ]; then
    sif_mount_args+=(--volume "${R2E_GYM_SIF_DIR}:${R2E_GYM_SIF_DIR}:ro")
    docker_preflight_mount_args+=(
        --mount "type=bind,source=${R2E_GYM_SIF_DIR},target=${R2E_GYM_SIF_DIR},readonly"
    )
fi

if ! docker run --rm \
    --network none \
    "${docker_preflight_mount_args[@]}" \
    --entrypoint /bin/bash \
    "${NEMO_GYM_IMAGE}" \
    -ceu '
        data_dir="$1"
        sif_dir="$2"
        sif_prefix="$3"
        test -s "${data_dir}/r2e_gym_train.jsonl"
        test -s /opt/relax-integration/examples/nemo_gym_agentic/recipes/r2e-gym/start_r2e_gym_local.sh
        first_sif="$(find -L "${sif_dir}" -maxdepth 1 -type f \
            -name "${sif_prefix}*.sif" -size +0c -print -quit)"
        test -n "${first_sif}"
    ' -- "${R2E_DATA_DIR}" "${R2E_GYM_SIF_DIR}" "${R2E_GYM_SIF_PREFIX}"; then
    echo "ERROR: Docker daemon cannot read the configured Relax checkout, dataset, or SIF directory." >&2
    echo "In Docker-in-Docker, a FUSE mount created only inside the client container is not visible to the" >&2
    echo "outer Docker daemon. Mount --repo-dir, --data-dir, and --sif-dir on the Docker host first." >&2
    echo "The existing ${NEMO_GYM_CONTAINER} container was not changed." >&2
    exit 2
fi

if docker container inspect "${NEMO_GYM_CONTAINER}" >/dev/null 2>&1; then
    echo "Removing existing container ${NEMO_GYM_CONTAINER}..."
    docker rm -f "${NEMO_GYM_CONTAINER}" >/dev/null
fi

container_no_proxy="127.0.0.1,localhost,${GYM_HOST}"
if [ -n "${NEMO_GYM_NO_PROXY}" ]; then
    container_no_proxy="${container_no_proxy},${NEMO_GYM_NO_PROXY}"
fi

r2e_setup_volume="${NEMO_GYM_CONTAINER}-r2e-setup"
openhands_setup_volume="${NEMO_GYM_CONTAINER}-openhands-setup"

docker volume create "${r2e_setup_volume}" >/dev/null
docker volume create "${openhands_setup_volume}" >/dev/null

echo "Creating remote R2E-Gym container:"
echo "  container=${NEMO_GYM_CONTAINER}"
echo "  image=${NEMO_GYM_IMAGE}"
echo "  mode=${R2E_GYM_MODE}"
echo "  gym_url=http://${GYM_HOST}:${R2E_GYM_PORT_BASE}"
echo "  callback_networks=${RELAX_CALLBACK_NETWORKS:-<none>}"
echo "  data_dir=${R2E_DATA_DIR}"
echo "  sif_dir=${R2E_GYM_SIF_DIR}"
echo "  sif_prefix=${R2E_GYM_SIF_PREFIX:-<none>}"
echo "  repo_dir=${RELAX_REPO_ROOT}"
echo "  max_concurrency=${R2E_GYM_MAX_CONCURRENCY}"
echo "  callback_timeout_s=${NEMO_GYM_CALLBACK_TIMEOUT_S}"
if [ -n "${NEMO_GYM_CALLBACK_PROXY}" ]; then
    echo "  callback_proxy=enabled"
else
    echo "  callback_proxy=disabled"
fi
echo "  verbose=${NEMO_GYM_VERBOSE}"

docker create \
    --name "${NEMO_GYM_CONTAINER}" \
    --privileged \
    --network host \
    --shm-size 16g \
    --mount "type=bind,source=${RELAX_REPO_ROOT},target=/opt/relax-integration,readonly" \
    --volume "${R2E_DATA_DIR}:${R2E_DATA_DIR}" \
    "${sif_mount_args[@]}" \
    --volume "${r2e_setup_volume}:/opt/nemo-gym/responses_api_agents/swe_agents/swe_r2e_gym_setup" \
    --volume "${openhands_setup_volume}:/opt/nemo-gym/responses_api_agents/swe_agents/swe_openhands_setup" \
    --env GYM_HOST="${GYM_HOST}" \
    --env GYM_BIND_HOST="${GYM_HOST}" \
    --env GYM_RAY_NUM_CPUS="${GYM_RAY_NUM_CPUS}" \
    --env RAY_CLI="/opt/nemo-gym/.venv/bin/ray" \
    --env R2E_GYM_CLUSTER_PYTHON="/opt/nemo-gym/.venv/bin/python" \
    --env R2E_GYM_MODE="${R2E_GYM_MODE}" \
    --env R2E_GYM_PORT_BASE="${R2E_GYM_PORT_BASE}" \
    --env R2E_GYM_MAX_CONCURRENCY="${R2E_GYM_MAX_CONCURRENCY}" \
    --env R2E_GYM_MAX_TURNS="${R2E_GYM_MAX_TURNS}" \
    --env NEMO_GYM_VERBOSE="${NEMO_GYM_VERBOSE}" \
    --env NEMO_GYM_CALLBACK_ALLOWED_NETWORKS="${RELAX_CALLBACK_NETWORKS}" \
    --env NEMO_GYM_CALLBACK_PROXY="${NEMO_GYM_CALLBACK_PROXY}" \
    --env NEMO_GYM_CALLBACK_TIMEOUT_S="${NEMO_GYM_CALLBACK_TIMEOUT_S}" \
    --env R2E_GYM_DATA="${R2E_DATA_DIR}/r2e_gym_train.jsonl" \
    --env R2E_GYM_SIF_DIR="${R2E_GYM_SIF_DIR}" \
    --env R2E_GYM_SIF_PREFIX="${R2E_GYM_SIF_PREFIX}" \
    --env R2E_GYM_VERIFY_GOLDEN_PATCH="${R2E_GYM_VERIFY_GOLDEN_PATCH}" \
    --env HTTP_PROXY="${NEMO_GYM_HTTP_PROXY}" \
    --env HTTPS_PROXY="${NEMO_GYM_HTTP_PROXY}" \
    --env NO_PROXY="${container_no_proxy}" \
    --env http_proxy="${NEMO_GYM_HTTP_PROXY}" \
    --env https_proxy="${NEMO_GYM_HTTP_PROXY}" \
    --env no_proxy="${container_no_proxy}" \
    "${NEMO_GYM_IMAGE}" \
    bash -lc '
        set -euo pipefail
        /opt/nemo-gym/.venv/bin/ray start \
            --head \
            --node-ip-address="${GYM_HOST}" \
            --port=6381 \
            --num-cpus="${GYM_RAY_NUM_CPUS}" \
            --include-dashboard=true \
            --dashboard-port=28265 \
            --disable-usage-stats \
            --temp-dir=/tmp/ray-nemo-r2e
        export RAY_ADDRESS="${GYM_HOST}:6381"
        local_args=(
            --data-dir "$(dirname "${R2E_GYM_DATA}")"
            --mode "${R2E_GYM_MODE}"
            --port-base "${R2E_GYM_PORT_BASE}"
            --sif-dir "${R2E_GYM_SIF_DIR}"
            --sif-prefix "${R2E_GYM_SIF_PREFIX}"
            --max-concurrency "${R2E_GYM_MAX_CONCURRENCY}"
        )
        if [ "${NEMO_GYM_VERBOSE}" = "1" ]; then
            local_args+=(--verbose)
        fi
        exec bash /opt/relax-integration/examples/nemo_gym_agentic/recipes/r2e-gym/start_r2e_gym_local.sh \
            "${local_args[@]}"
    ' >/dev/null

docker start "${NEMO_GYM_CONTAINER}" >/dev/null

echo "Waiting for http://${GYM_HOST}:${R2E_GYM_PORT_BASE}/readyz ..."
deadline=$((SECONDS + NEMO_GYM_START_TIMEOUT_S))
next_progress=$SECONDS
while [ "${SECONDS}" -lt "${deadline}" ]; do
    if ! docker container inspect --format "{{.State.Running}}" "${NEMO_GYM_CONTAINER}" | grep -q true; then
        echo "R2E-Gym container exited before becoming ready." >&2
        docker logs --tail 200 "${NEMO_GYM_CONTAINER}" >&2
        exit 1
    fi
    if ready_json="$(curl --noproxy "*" -fsS "http://${GYM_HOST}:${R2E_GYM_PORT_BASE}/readyz" 2>/dev/null)" &&
        printf '%s' "${ready_json}" | grep -q '"ready":true'; then
        printf '%s\n' "${ready_json}"
        echo "R2E-Gym is ready at http://${GYM_HOST}:${R2E_GYM_PORT_BASE}"
        exit 0
    fi
    if [ "${SECONDS}" -ge "${next_progress}" ]; then
        echo "Still starting; inspect with: docker logs -f ${NEMO_GYM_CONTAINER}"
        next_progress=$((SECONDS + 30))
    fi
    sleep 2
done

echo "Timed out waiting for R2E-Gym readiness." >&2
docker logs --tail 200 "${NEMO_GYM_CONTAINER}" >&2
exit 1
