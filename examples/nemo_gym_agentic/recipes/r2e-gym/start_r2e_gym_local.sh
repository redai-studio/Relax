#!/bin/bash

# Copyright (c) 2026 Relax Authors. All Rights Reserved.

set -euo pipefail

GYM_ROOT="${GYM_ROOT:-/opt/nemo-gym}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXAMPLE_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
RELAX_INTEGRATION_ROOT="${RELAX_INTEGRATION_ROOT:-$(cd "${EXAMPLE_DIR}/../.." && pwd)}"
RAY_CLI="${RAY_CLI:-ray}"
R2E_DATA_DIR=""
R2E_GYM_MODE=""
R2E_GYM_PORT_BASE="${R2E_GYM_PORT_BASE:-28100}"
R2E_GYM_SIF_DIR="${R2E_GYM_SIF_DIR:-}"
R2E_GYM_SIF_PREFIX="${R2E_GYM_SIF_PREFIX:-}"
R2E_GYM_MAX_CONCURRENCY="${R2E_GYM_MAX_CONCURRENCY:-16}"
NEMO_GYM_VERBOSE="${NEMO_GYM_VERBOSE:-0}"
NEMO_GYM_HTTP_PROXY="${NEMO_GYM_HTTP_PROXY:-${https_proxy:-${HTTPS_PROXY:-${http_proxy:-${HTTP_PROXY:-}}}}}"
NEMO_GYM_CALLBACK_PROXY="${NEMO_GYM_CALLBACK_PROXY:-}"
NEMO_GYM_CALLBACK_TIMEOUT_S="${NEMO_GYM_CALLBACK_TIMEOUT_S:-600}"
R2E_GYM_AGENT_TIMEOUT="${R2E_GYM_AGENT_TIMEOUT:-3600}"
R2E_GYM_TEST_TIMEOUT="${R2E_GYM_TEST_TIMEOUT:-1800}"
R2E_GYM_MAX_TURNS="${R2E_GYM_MAX_TURNS:-30}"
R2E_GYM_MEMORY_MB="${R2E_GYM_MEMORY_MB:-32768}"
R2E_GYM_MAX_DEADLINE_S="${R2E_GYM_MAX_DEADLINE_S:-7200}"
GYM_DRY_RUN="${GYM_DRY_RUN:-false}"
OPENHANDS_SETUP_PATCH="${EXAMPLE_DIR}/service/patches/openhands_setup_portability.patch"
OPENHANDS_PLATFORM_PATCH="${EXAMPLE_DIR}/service/patches/openhands_platform_release.patch"
OPENHANDS_R2E_RUNTIME_PATCH="${EXAMPLE_DIR}/service/patches/openhands_r2e_runtime.patch"
OPENHANDS_R2E_RUNTIME_SETUP_PATCH="${EXAMPLE_DIR}/service/patches/openhands_r2e_runtime_setup.patch"
PYTHON_STARTUP_DIR="${EXAMPLE_DIR}/service/python_startup"

usage() {
    cat <<'EOF'
Usage:
  start_r2e_gym_local.sh \
    --data-dir <prepared-r2e-directory> \
    --mode <golden|train> \
    [--sif-dir <shared-sif-directory>] \
    [--sif-prefix <filename-prefix>] \
    [--port-base <port>] \
    [--max-concurrency <positive-integer>] \
    [--proxy <http-proxy-url>] \
    [--callback-proxy <http-proxy-url>] \
    [--callback-timeout-s <positive-integer>] \
    [--verbose]

The launcher reuses the existing local Ray cluster and must run on the Ray head
node. RAY_ADDRESS is optional and defaults to auto. The launcher does not start
Docker or a second Ray cluster.
Ports: gateway=base, agent=base+1, head=base+3. Default base: 28100.
Override via --port-base or R2E_GYM_PORT_BASE; valid range: 1-65532.
EOF
}

while [ "$#" -gt 0 ]; do
    case "$1" in
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
        --verbose)
            NEMO_GYM_VERBOSE=1
            shift
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

if [ -z "${R2E_DATA_DIR}" ] || [ -z "${R2E_GYM_MODE}" ]; then
    usage >&2
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
        echo "--sif-dir must be an absolute path" >&2
        exit 2
        ;;
esac
if [[ "${R2E_GYM_SIF_PREFIX}" == */* ]]; then
    echo "--sif-prefix must be a filename prefix without '/'" >&2
    exit 2
fi
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
if ! [[ "${R2E_GYM_PORT_BASE}" =~ ^[1-9][0-9]{0,4}$ ]] || ((R2E_GYM_PORT_BASE > 65532)); then
    echo "--port-base must be an integer between 1 and 65532" >&2
    exit 2
fi
R2E_GYM_GATEWAY_PORT="${R2E_GYM_PORT_BASE}"
R2E_GYM_AGENT_PORT="$((R2E_GYM_PORT_BASE + 1))"
R2E_GYM_HEAD_PORT="$((R2E_GYM_PORT_BASE + 3))"
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

ray_address="${RAY_ADDRESS:-auto}"
case "${ray_address}" in
    http://* | https://* | ray://* | *://*)
        echo "RAY_ADDRESS must be a Ray GCS address such as <head>:6379 or auto, not ${ray_address}" >&2
        exit 2
        ;;
    auto)
        discovery_ray_address="auto"
        ;;
    *:*)
        discovery_ray_address="${ray_address}"
        ;;
    *)
        discovery_ray_address="${ray_address}:6379"
        ;;
esac

for command_name in "${RAY_CLI}" jq hostname apptainer; do
    command -v "${command_name}" >/dev/null 2>&1 || {
        echo "ERROR: required command not found: ${command_name}" >&2
        exit 2
    }
done

"${RAY_CLI}" status --address="${discovery_ray_address}" >/dev/null
ray_nodes_json="$("${RAY_CLI}" list nodes --address="${discovery_ray_address}" --format json)"
GYM_HOST="$(
    jq -er '
        [
            .[]
            | select(
                .state == "ALIVE"
                and (
                    .is_head_node == true
                    or ((.resources_total // {})["node:__internal_head__"] // 0) > 0
                )
            )
        ]
        | unique_by(.node_id)
        | if length == 1
          then .[0].node_ip
          else error("expected exactly one ALIVE Ray head node, found \(length)")
          end
    ' <<<"${ray_nodes_json}"
)"
if [ -z "${GYM_HOST}" ]; then
    echo "ERROR: Ray head node has an empty node_ip" >&2
    exit 2
fi
case " $(hostname -I 2>/dev/null || true) " in
    *" ${GYM_HOST} "*) ;;
    *)
        echo "ERROR: local R2E-Gym must run on Ray head ${GYM_HOST}; this host does not own that address" >&2
        exit 2
        ;;
esac

if [ "${discovery_ray_address}" = "auto" ]; then
    GYM_RAY_ADDRESS="${GYM_HOST}:6379"
else
    GYM_RAY_ADDRESS="${discovery_ray_address}"
fi
GYM_BIND_HOST="${GYM_BIND_HOST:-0.0.0.0}"

export NEMO_GYM_CALLBACK_ALLOWED_NETWORKS="${NEMO_GYM_CALLBACK_ALLOWED_NETWORKS:-10.0.0.0/8}"

existing_no_proxy="${no_proxy:-${NO_PROXY:-}}"
NO_PROXY="127.0.0.1,localhost,${GYM_HOST}"
if [ -n "${existing_no_proxy}" ]; then
    NO_PROXY="${NO_PROXY},${existing_no_proxy}"
fi
no_proxy="${NO_PROXY}"
export NO_PROXY no_proxy
if [ -n "${NEMO_GYM_HTTP_PROXY}" ]; then
    HTTP_PROXY="${NEMO_GYM_HTTP_PROXY}"
    HTTPS_PROXY="${NEMO_GYM_HTTP_PROXY}"
    http_proxy="${NEMO_GYM_HTTP_PROXY}"
    https_proxy="${NEMO_GYM_HTTP_PROXY}"
    export HTTP_PROXY HTTPS_PROXY http_proxy https_proxy
fi

R2E_GYM_DATA="${R2E_DATA_DIR}/r2e_gym_train.jsonl"
NEMO_GYM_ARTIFACT_ROOT="${NEMO_GYM_ARTIFACT_ROOT:-${R2E_DATA_DIR}/artifacts}"
mkdir -p "${NEMO_GYM_ARTIFACT_ROOT}"
chmod 700 "${NEMO_GYM_ARTIFACT_ROOT}"

echo "Local R2E-Gym startup configuration:"
echo "  gym_root=${GYM_ROOT}"
echo "  gym_host=${GYM_HOST}"
echo "  gym_url=http://${GYM_HOST}:${R2E_GYM_GATEWAY_PORT}"
echo "  bind_host=${GYM_BIND_HOST}"
echo "  ray_address=${GYM_RAY_ADDRESS}"
echo "  callback_networks=${NEMO_GYM_CALLBACK_ALLOWED_NETWORKS:-<none>}"
echo "  data=${R2E_GYM_DATA}"
echo "  sif_dir=${R2E_GYM_SIF_DIR}"
echo "  sif_prefix=${R2E_GYM_SIF_PREFIX:-<none>}"
echo "  artifact_root=${NEMO_GYM_ARTIFACT_ROOT}"
echo "  golden=${R2E_GYM_VERIFY_GOLDEN_PATCH}"
echo "  max_concurrency=${R2E_GYM_MAX_CONCURRENCY}"
echo "  callback_timeout_s=${NEMO_GYM_CALLBACK_TIMEOUT_S}"
if [ -n "${NEMO_GYM_CALLBACK_PROXY}" ]; then
    echo "  callback_proxy=enabled"
else
    echo "  callback_proxy=disabled"
fi
echo "  verbose=${NEMO_GYM_VERBOSE}"

if [ ! -x "${GYM_ROOT}/.venv/bin/gym" ]; then
    echo "ERROR: NeMo Gym executable not found: ${GYM_ROOT}/.venv/bin/gym" >&2
    exit 2
fi
if [ ! -s "${R2E_GYM_DATA}" ]; then
    echo "ERROR: prepared R2E-Gym JSONL is missing or empty on this Ray worker: ${R2E_GYM_DATA}" >&2
    exit 2
fi
if [ ! -d "${R2E_GYM_SIF_DIR}" ]; then
    echo "ERROR: R2E-Gym SIF directory is missing on this Ray worker: ${R2E_GYM_SIF_DIR}" >&2
    exit 2
fi
first_sif="$(find -L "${R2E_GYM_SIF_DIR}" -maxdepth 1 -type f -name "${R2E_GYM_SIF_PREFIX}*.sif" -size +0c -print -quit)"
if [ -z "${first_sif}" ]; then
    echo "ERROR: no non-empty ${R2E_GYM_SIF_PREFIX}*.sif file found in ${R2E_GYM_SIF_DIR}" >&2
    exit 2
fi
sif_manifest="${R2E_DATA_DIR}/r2e_gym_train_sifs.jsonl"
if [ -s "${sif_manifest}" ]; then
    manifest_entry_count=0
    missing_sif_count=0
    while IFS= read -r sif_name; do
        if [ -z "${sif_name}" ]; then
            continue
        fi
        manifest_entry_count=$((manifest_entry_count + 1))
        expected_sif="${R2E_GYM_SIF_DIR}/${R2E_GYM_SIF_PREFIX}${sif_name}"
        if [ ! -s "${expected_sif}" ]; then
            missing_sif_count=$((missing_sif_count + 1))
            if [ "${missing_sif_count}" -le 10 ]; then
                echo "ERROR: manifest SIF is missing or empty: ${expected_sif}" >&2
            fi
        fi
    done < <(jq -r '.sif_name // empty' "${sif_manifest}")
    if [ "${manifest_entry_count}" -eq 0 ]; then
        echo "ERROR: no sif_name entries found in ${sif_manifest}" >&2
        exit 2
    fi
    if [ "${missing_sif_count}" -gt 0 ]; then
        echo "ERROR: ${missing_sif_count}/${manifest_entry_count} manifest SIFs are unavailable." >&2
        exit 2
    fi
    echo "Manifest SIF coverage: ${manifest_entry_count}/${manifest_entry_count}"
fi
if [ "${GYM_DRY_RUN}" != "false" ] && [ "${GYM_DRY_RUN}" != "true" ]; then
    echo "GYM_DRY_RUN must be false or true" >&2
    exit 2
fi

cluster_python="${R2E_GYM_CLUSTER_PYTHON:-/usr/bin/python3}"
if [ ! -x "${cluster_python}" ]; then
    echo "ERROR: cluster Ray Python is not executable: ${cluster_python}" >&2
    exit 2
fi
cluster_ray_version="$("${cluster_python}" -c 'import ray; print(ray.__version__)')"
ray_python_paths=(
    "${GYM_ROOT}/.venv/bin/python"
    "${GYM_ROOT}/responses_api_models/relax_gateway_model/.venv/bin/python"
    "${GYM_ROOT}/responses_api_agents/swe_agents/.venv/bin/python"
)
for ray_python in "${ray_python_paths[@]}"; do
    if [ ! -x "${ray_python}" ]; then
        echo "ERROR: Ray Python is not executable: ${ray_python}" >&2
        exit 2
    fi
    ray_version="$("${ray_python}" -c 'import ray; print(ray.__version__)')"
    if [ "${ray_version}" != "${cluster_ray_version}" ]; then
        uv_cli="${UV_CLI:-uv}"
        command -v "${uv_cli}" >/dev/null 2>&1 || {
            echo "ERROR: Ray ${ray_version} in ${ray_python} must be aligned to ${cluster_ray_version}," >&2
            echo "but the image does not provide the required uv command: ${uv_cli}" >&2
            exit 2
        }
        echo "Aligning Ray in ${ray_python}: ${ray_version} -> ${cluster_ray_version}"
        "${uv_cli}" pip install \
            --python "${ray_python}" \
            "ray[default]==${cluster_ray_version}"
        aligned_ray_version="$("${ray_python}" -c 'import ray; print(ray.__version__)')"
        if [ "${aligned_ray_version}" != "${cluster_ray_version}" ]; then
            echo "ERROR: Ray alignment failed: ${ray_python}=${aligned_ray_version}, expected=${cluster_ray_version}" >&2
            exit 2
        fi
    fi
done
echo "Ray versions aligned with cluster Python: ${cluster_ray_version}"

"${cluster_python}" - "${GYM_ROOT}" "${R2E_GYM_PORT_BASE}" <<'PY'
from pathlib import Path
import sys

import psutil

gym_root = Path(sys.argv[1]).resolve()
port_base = int(sys.argv[2])
expected_cwds = {
    port_base: gym_root / "responses_api_models" / "relax_gateway_model",
    port_base + 1: gym_root / "responses_api_agents" / "swe_agents",
    port_base + 3: gym_root,
}
owned_processes: dict[int, tuple[psutil.Process, set[int]]] = {}
foreign_listeners: list[str] = []

for connection in psutil.net_connections(kind="tcp"):
    if connection.status != psutil.CONN_LISTEN or not connection.laddr:
        continue
    port = connection.laddr.port
    if port not in expected_cwds:
        continue
    if connection.pid is None:
        foreign_listeners.append(f"port={port} pid=<unknown>")
        continue
    try:
        process = psutil.Process(connection.pid)
        cwd = Path(process.cwd()).resolve()
        command = " ".join(process.cmdline())
    except (psutil.AccessDenied, psutil.NoSuchProcess) as exc:
        foreign_listeners.append(f"port={port} pid={connection.pid} error={exc}")
        continue
    if cwd != expected_cwds[port]:
        foreign_listeners.append(f"port={port} pid={connection.pid} cwd={cwd} command={command}")
        continue
    if connection.pid not in owned_processes:
        owned_processes[connection.pid] = (process, set())
    owned_processes[connection.pid][1].add(port)

if foreign_listeners:
    print("ERROR: R2E-Gym ports are occupied by processes not owned by this launcher:", file=sys.stderr)
    for listener in foreign_listeners:
        print(f"  {listener}", file=sys.stderr)
    raise SystemExit(2)

processes = [entry[0] for entry in owned_processes.values()]
for process, ports in owned_processes.values():
    print(f"Stopping existing local R2E-Gym process pid={process.pid} ports={sorted(ports)}")
    process.terminate()

_, alive = psutil.wait_procs(processes, timeout=10)
for process in alive:
    print(f"Existing local R2E-Gym process pid={process.pid} did not terminate; killing it")
    process.kill()
_, still_alive = psutil.wait_procs(alive, timeout=5)
if still_alive:
    pids = ", ".join(str(process.pid) for process in still_alive)
    print(f"ERROR: failed to stop existing local R2E-Gym process(es): {pids}", file=sys.stderr)
    raise SystemExit(2)
PY

for port in "${R2E_GYM_GATEWAY_PORT}" "${R2E_GYM_AGENT_PORT}" "${R2E_GYM_HEAD_PORT}"; do
    if ! port_error="$(
        "${cluster_python}" - "${port}" <<'PY'
import socket
import sys
import time

port = int(sys.argv[1])
last_error = None
for _ in range(20):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        sock.bind(("0.0.0.0", port))
        sock.close()
        break
    except OSError as exc:
        last_error = exc
        sock.close()
        time.sleep(0.5)
else:
    print(last_error)
    raise SystemExit(1) from last_error
PY
    )"; then
        echo "ERROR: local R2E-Gym port ${port} is already in use:" >&2
        echo "${port_error}" >&2
        exit 2
    fi
done

echo "R2E-Gym preflight passed:"
wc -l "${R2E_GYM_DATA}"
printf "  sif_count=%s\n" "$(
    find -L "${R2E_GYM_SIF_DIR}" -maxdepth 1 -type f -name "${R2E_GYM_SIF_PREFIX}*.sif" -size +0c |
        wc -l
)"
printf "  first_sif=%s\n" "${first_sif}"
apptainer version
apptainer exec "${first_sif}" true

cp \
    "${OPENHANDS_R2E_RUNTIME_PATCH}" \
    "${GYM_ROOT}/responses_api_agents/swe_agents/setup_scripts/openhands_r2e_runtime.patch"

for patch_path in \
    "${OPENHANDS_SETUP_PATCH}" \
    "${OPENHANDS_PLATFORM_PATCH}" \
    "${OPENHANDS_R2E_RUNTIME_SETUP_PATCH}"; do
    if [ ! -f "${patch_path}" ]; then
        echo "ERROR: OpenHands setup patch is missing: ${patch_path}" >&2
        exit 2
    fi
    if git -C "${GYM_ROOT}" apply --reverse --check "${patch_path}" >/dev/null 2>&1; then
        echo "OpenHands setup patch is already applied: ${patch_path}"
    elif git -C "${GYM_ROOT}" apply --check "${patch_path}"; then
        echo "Applying OpenHands setup patch: ${patch_path}"
        git -C "${GYM_ROOT}" apply "${patch_path}"
    else
        echo "ERROR: OpenHands setup patch does not apply cleanly: ${patch_path}" >&2
        exit 2
    fi
done

openhands_checkout="${GYM_ROOT}/responses_api_agents/swe_agents/swe_openhands_setup/OpenHands"
if [ -d "${openhands_checkout}/.git" ]; then
    if git -C "${openhands_checkout}" \
        apply --reverse --check "${OPENHANDS_R2E_RUNTIME_PATCH}" >/dev/null 2>&1; then
        echo "OpenHands R2E runtime patch is already applied: ${OPENHANDS_R2E_RUNTIME_PATCH}"
    elif git -C "${openhands_checkout}" apply --check "${OPENHANDS_R2E_RUNTIME_PATCH}"; then
        echo "Applying OpenHands R2E runtime patch: ${OPENHANDS_R2E_RUNTIME_PATCH}"
        git -C "${openhands_checkout}" apply "${OPENHANDS_R2E_RUNTIME_PATCH}"
    else
        echo "ERROR: OpenHands R2E runtime patch does not apply cleanly: ${OPENHANDS_R2E_RUNTIME_PATCH}" >&2
        exit 2
    fi
fi

export NEMO_GYM_SANITIZE_PLATFORM_RELEASE=1
export NEMO_GYM_PYTHON_STARTUP_DIR="${PYTHON_STARTUP_DIR}"
export PYTHONPATH="${PYTHON_STARTUP_DIR}:${RELAX_INTEGRATION_ROOT}:${GYM_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

export GYM_HOST GYM_RAY_ADDRESS NEMO_GYM_CALLBACK_ALLOWED_NETWORKS
export NEMO_GYM_CALLBACK_PROXY
export NEMO_GYM_ARTIFACT_ROOT NEMO_GYM_CALLBACK_TIMEOUT_S NEMO_GYM_VERBOSE
export R2E_GYM_DATA R2E_GYM_SIF_DIR R2E_GYM_SIF_PREFIX R2E_GYM_MAX_CONCURRENCY R2E_GYM_MAX_DEADLINE_S
export R2E_GYM_AGENT_PORT
export NEMO_GYM_GATEWAY_ENVIRONMENTS_JSON
NEMO_GYM_GATEWAY_ENVIRONMENTS_JSON="$(
    "${GYM_ROOT}/.venv/bin/python" - <<'PY'
import json
import os

host = os.environ["GYM_HOST"]
agent_port = int(os.environ["R2E_GYM_AGENT_PORT"])
max_concurrency = int(os.environ["R2E_GYM_MAX_CONCURRENCY"])
max_deadline = int(os.environ["R2E_GYM_MAX_DEADLINE_S"])
print(
    json.dumps(
        {
            "r2e-gym-v1": {
                "environment": "r2e_gym",
                "agent_name": "swe_agents",
                "agent_url": f"http://{host}:{agent_port}",
                "readiness_urls": [f"http://{host}:{agent_port}"],
                "interrupt_policy": "protected",
                "max_concurrency": max_concurrency,
                "queue_capacity": max_concurrency * 2,
                "max_deadline_s": max_deadline,
            }
        },
        separators=(",", ":"),
    )
)
PY
)"

cd "${GYM_ROOT}"
exec "${GYM_ROOT}/.venv/bin/gym" env start \
    "+config_paths=[responses_api_models/relax_gateway_model/configs/relax_gateway_model.yaml,responses_api_agents/swe_agents/configs/swebench_openhands.yaml]" \
    +observability_enabled=true \
    +dry_run="${GYM_DRY_RUN}" \
    +skip_venv_if_present=true \
    +ray_head_node_address="${GYM_RAY_ADDRESS}" \
    +default_host="${GYM_HOST}" \
    ++head_server.host="${GYM_BIND_HOST}" \
    ++head_server.port="${R2E_GYM_HEAD_PORT}" \
    ++policy_model.responses_api_models.relax_gateway_model.host="${GYM_BIND_HOST}" \
    ++policy_model.responses_api_models.relax_gateway_model.port="${R2E_GYM_GATEWAY_PORT}" \
    ++swe_agents.responses_api_agents.swe_agents.host="${GYM_BIND_HOST}" \
    ++swe_agents.responses_api_agents.swe_agents.port="${R2E_GYM_AGENT_PORT}" \
    ++swe_agents.responses_api_agents.swe_agents.dataset_harness=r2e_gym \
    "++swe_agents.responses_api_agents.swe_agents.container_formatter='${R2E_GYM_SIF_DIR}/${R2E_GYM_SIF_PREFIX}{instance_id}.sif'" \
    ++swe_agents.responses_api_agents.swe_agents.dataset_path="${R2E_GYM_DATA}" \
    ++swe_agents.responses_api_agents.swe_agents.concurrency="${R2E_GYM_MAX_CONCURRENCY}" \
    ++swe_agents.responses_api_agents.swe_agents.agent_max_turns="${R2E_GYM_MAX_TURNS}" \
    ++swe_agents.responses_api_agents.swe_agents.verify_golden_patch="${R2E_GYM_VERIFY_GOLDEN_PATCH}" \
    ++swe_agents.responses_api_agents.swe_agents.swebench_agent_timeout="${R2E_GYM_AGENT_TIMEOUT}" \
    ++swe_agents.responses_api_agents.swe_agents.swebench_tests_timeout="${R2E_GYM_TEST_TIMEOUT}" \
    ++swe_agents.responses_api_agents.swe_agents.apptainer_memory_limit_mb="${R2E_GYM_MEMORY_MB}"
