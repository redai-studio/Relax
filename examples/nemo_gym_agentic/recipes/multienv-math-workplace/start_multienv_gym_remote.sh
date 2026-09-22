#!/bin/bash

# Copyright (c) 2026 Relax Authors. All Rights Reserved.

set -euo pipefail

# Omit the repository override to use the recipe files baked into the image.
REPO_ROOT="${RELAX_REPO_ROOT:-}"

usage() {
    echo "Default: 10.0.0.0/8. Override with --callback-network or NEMO_GYM_CALLBACK_ALLOWED_NETWORKS."
    echo "Usage: $0 --gym-host HOST [--callback-network CIDR] [--image IMAGE] [--port-base PORT] [--container-name NAME]"
    echo "         [--repo-dir PATH] [--max-concurrency N]"
    echo "  --repo-dir PATH         Optional Docker-daemon-visible absolute Relax checkout path."
    echo "                          Defaults to RELAX_REPO_ROOT, or uses the image's recipe files if unset."
    echo "  --max-concurrency N     Set the concurrency limit for each of math and workplace."
    echo "                          Defaults to MATH_MAX_CONCURRENCY / WORKPLACE_MAX_CONCURRENCY (256 each)."
}

GYM_HOST=""
RELAX_CALLBACK_NETWORKS="${NEMO_GYM_CALLBACK_ALLOWED_NETWORKS:-}"
NEMO_GYM_IMAGE="${NEMO_GYM_IMAGE:-relax-nemo-gym:multienv-dev}"
NEMO_GYM_CONTAINER="${NEMO_GYM_CONTAINER:-nemo-gym-multienv}"
MULTIENV_PORT_BASE="${MULTIENV_PORT_BASE:-29300}"
GYM_RAY_PORT="${GYM_RAY_PORT:-6384}"
GYM_DASHBOARD_PORT="${GYM_DASHBOARD_PORT:-30365}"
GYM_RAY_NUM_CPUS="${GYM_RAY_NUM_CPUS:-16}"
MATH_MAX_CONCURRENCY="${MATH_MAX_CONCURRENCY:-256}"
WORKPLACE_MAX_CONCURRENCY="${WORKPLACE_MAX_CONCURRENCY:-256}"
NEMO_GYM_START_TIMEOUT_S="${NEMO_GYM_START_TIMEOUT_S:-600}"

while [ "$#" -gt 0 ]; do
    case "$1" in
        --gym-host) GYM_HOST="${2:-}"; shift 2 ;;
        --callback-network)
            if [ -z "${2:-}" ] || [[ "${2}" == --* ]]; then
                echo "--callback-network requires a CIDR network" >&2
                exit 2
            fi
            RELAX_CALLBACK_NETWORKS="${RELAX_CALLBACK_NETWORKS:+${RELAX_CALLBACK_NETWORKS},}${2}"
            shift 2
            ;;
        --image) NEMO_GYM_IMAGE="${2:-}"; shift 2 ;;
        --repo-dir)
            if [ -z "${2:-}" ] || [[ "${2}" == --* ]]; then
                echo "--repo-dir requires an absolute path" >&2
                exit 2
            fi
            REPO_ROOT="${2}"
            shift 2
            ;;
        --port-base) MULTIENV_PORT_BASE="${2:-}"; shift 2 ;;
        --max-concurrency)
            if ! [[ "${2:-}" =~ ^[0-9]+$ ]] || ! [[ "${2:-}" =~ [1-9] ]]; then
                echo "--max-concurrency must be a positive integer" >&2
                exit 2
            fi
            MATH_MAX_CONCURRENCY="${2}"
            WORKPLACE_MAX_CONCURRENCY="${2}"
            shift 2
            ;;
        --container-name) NEMO_GYM_CONTAINER="${2:-}"; shift 2 ;;
        -h | --help) usage; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
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
        echo "${host_name} must be a bare host or IP" >&2
        exit 2
    fi
done
if [ -n "${REPO_ROOT}" ]; then
    case "${REPO_ROOT}" in /*) ;; *) echo "--repo-dir must be absolute: ${REPO_ROOT}" >&2; exit 2 ;; esac
fi
if ! [[ "${MULTIENV_PORT_BASE}" =~ ^[0-9]+$ ]] \
    || ((10#${MULTIENV_PORT_BASE} < 1 || 10#${MULTIENV_PORT_BASE} > 65530)); then
    echo "--port-base must be an integer between 1 and 65530" >&2
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

command -v docker >/dev/null
command -v curl >/dev/null
command -v jq >/dev/null
docker image inspect "${NEMO_GYM_IMAGE}" >/dev/null
if [ -n "${REPO_ROOT}" ]; then
    if ! docker run --rm --network none --entrypoint bash \
        --mount "type=bind,source=${REPO_ROOT},target=/repo,readonly" \
        "${NEMO_GYM_IMAGE}" \
        -ceu 'test -s /repo/examples/nemo_gym_agentic/recipes/multienv-math-workplace/start_multienv_gym.sh'; then
        echo "--repo-dir ${REPO_ROOT} is not visible to the Docker daemon." >&2
        echo "Pass the daemon-visible absolute path, or omit --repo-dir to use the image's recipe files." >&2
        exit 2
    fi
fi

if docker container inspect "${NEMO_GYM_CONTAINER}" >/dev/null 2>&1; then
    owner="$(docker container inspect --format '{{index .Config.Labels "ai.relax.recipe"}}' "${NEMO_GYM_CONTAINER}")"
    if [ "${owner}" != "multienv-math-workplace" ]; then
        echo "Refusing to replace ${NEMO_GYM_CONTAINER}: ai.relax.recipe=${owner:-<unset>}" >&2
        exit 2
    fi
    docker rm -f "${NEMO_GYM_CONTAINER}" >/dev/null
fi

container_no_proxy="127.0.0.1,localhost,${GYM_HOST}"
repo_mount_args=()
if [ -n "${REPO_ROOT}" ]; then
    repo_mount_args+=(--mount "type=bind,source=${REPO_ROOT},target=/opt/relax-integration,readonly")
fi
docker create \
    --name "${NEMO_GYM_CONTAINER}" \
    --label ai.relax.recipe=multienv-math-workplace \
    --network host \
    --shm-size 12g \
    "${repo_mount_args[@]}" \
    --env GYM_HOST="${GYM_HOST}" \
    --env GYM_BIND_HOST="0.0.0.0" \
    --env GYM_RAY_PORT="${GYM_RAY_PORT}" \
    --env GYM_DASHBOARD_PORT="${GYM_DASHBOARD_PORT}" \
    --env GYM_RAY_NUM_CPUS="${GYM_RAY_NUM_CPUS}" \
    --env NEMO_GYM_CALLBACK_ALLOWED_NETWORKS="${RELAX_CALLBACK_NETWORKS}" \
    --env MULTIENV_PORT_BASE="${MULTIENV_PORT_BASE}" \
    --env MATH_MAX_CONCURRENCY="${MATH_MAX_CONCURRENCY}" \
    --env WORKPLACE_MAX_CONCURRENCY="${WORKPLACE_MAX_CONCURRENCY}" \
    --env NO_PROXY="${container_no_proxy}" \
    --env no_proxy="${container_no_proxy}" \
    "${NEMO_GYM_IMAGE}" \
    bash /opt/relax-integration/examples/nemo_gym_agentic/recipes/multienv-math-workplace/start_multienv_gym.sh \
    >/dev/null
docker start "${NEMO_GYM_CONTAINER}" >/dev/null

gateway_url="http://${GYM_HOST}:${MULTIENV_PORT_BASE}"
expected_gym_commit="a85670eb167ba9b48cc53a36a070eed815e6c40d"
deadline=$((SECONDS + NEMO_GYM_START_TIMEOUT_S))
while [ "${SECONDS}" -lt "${deadline}" ]; do
    if ! docker container inspect --format "{{.State.Running}}" "${NEMO_GYM_CONTAINER}" | grep -q true; then
        docker logs --tail 200 "${NEMO_GYM_CONTAINER}" >&2
        exit 1
    fi
    if ready_json="$(curl --noproxy "*" -fsS "${gateway_url}/readyz" 2>/dev/null)" \
        && printf '%s' "${ready_json}" | jq -e \
            --arg commit "${expected_gym_commit}" \
            '.ready == true and .gym_commit == $commit and .active_trials == 0' >/dev/null; then
        printf '%s\n' "${ready_json}"
        echo "Multi-environment Gym is ready at ${gateway_url}"
        exit 0
    fi
    sleep 2
done

docker logs --tail 200 "${NEMO_GYM_CONTAINER}" >&2
exit 1
