#!/usr/bin/env bash
# Copyright (c) 2026 Relax Authors. All Rights Reserved.
# Own the GAGAR agent server and training process in the Ray job.
set -eo pipefail
if ((BASH_VERSINFO[0] < 5 || (BASH_VERSINFO[0] == 5 && BASH_VERSINFO[1] < 1))); then
   echo "This example requires Bash 5.1 or newer." >&2
   exit 1
fi
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
RELAX_DIR="$(cd "${SCRIPT_DIR}/../../../../.." && pwd)"
cd "${RELAX_DIR}"
AGENT_SERVER_PID=""
TRAIN_PID=""
RAY_STOP=0
cleanup() {
   trap - EXIT
   trap '' TERM INT
   if [[ -n "${TRAIN_PID}" ]]; then
      if [[ "${RAY_STOP}" == 0 ]]; then kill -TERM "${TRAIN_PID}" 2>/dev/null || true; fi
      wait "${TRAIN_PID}" 2>/dev/null || true
   fi
   if [[ -n "${AGENT_SERVER_PID}" ]]; then
      if [[ "${RAY_STOP}" == 0 ]]; then kill -TERM "${AGENT_SERVER_PID}" 2>/dev/null || true; fi
      wait "${AGENT_SERVER_PID}" 2>/dev/null || true
   fi
}
trap cleanup EXIT
# Ray signals the driver AND its descendants; do not signal train.py twice.
trap 'RAY_STOP=1; exit 143' TERM
trap 'RAY_STOP=1; exit 130' INT

# Fail before starting training if another service owns this port.
python3 - <<'PORT'
import os
import socket
with socket.socket() as sock:
    sock.bind((os.environ["AGENT_SERVER_HOST"], int(os.environ["AGENT_SERVER_PORT"])))
PORT
(
   export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
   export MSWEA_CONFIGURED=true MSWEA_SILENT_STARTUP=true MSWEA_MODEL_RETRY_STOP_AFTER_ATTEMPT=1
   exec "${MINI_SWE_AGENT_VENV}/bin/python" "${SCRIPT_DIR}/agent_server.py" \
      --host "${AGENT_SERVER_HOST}" --port "${AGENT_SERVER_PORT}"
) >"${AGENT_SERVER_WORK_DIR}/server.log" 2>&1 &
AGENT_SERVER_PID=$!
ready=0
for ((attempt=0; attempt<120; attempt++)); do
   kill -0 "${AGENT_SERVER_PID}" 2>/dev/null || { cat "${AGENT_SERVER_WORK_DIR}/server.log" >&2; exit 1; }
   # The existing server has no health route; a Flask 404 proves HTTP readiness.
   if [[ "$(curl --noproxy '*' -s -o /dev/null -w '%{http_code}' --max-time 1 "${AGENT_SERVER_URL}/" || true)" == 404 ]]; then
      ready=1
      break
   fi
   sleep 1
done
[[ "${ready}" == 1 ]] || { echo "Agent server startup timed out; see ${AGENT_SERVER_WORK_DIR}/server.log" >&2; exit 1; }
python3 "${RELAX_DIR}/relax/entrypoints/train.py" "$@" &
TRAIN_PID=$!
result=0
wait -n -p finished "${TRAIN_PID}" "${AGENT_SERVER_PID}" || result=$?
if [[ "${finished}" == "${AGENT_SERVER_PID}" ]]; then
   AGENT_SERVER_PID=""
   echo "Agent server exited before training completed; see server.log" >&2
   exit 1
fi
TRAIN_PID=""
exit "${result}"
