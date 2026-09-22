#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 3 ]; then
  echo "Usage: $0 <relax-eval-log> <eval-pid> <retriever-log> [port]" >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RELAX_ROOT="$(cd "${SCRIPT_DIR}/../../../../.." && pwd)"
EVAL_LOG=$1
EVAL_PID=$2
RETRIEVER_LOG=$3
PORT="${4:-${SEARCH_RETRIEVAL_PORT:-8001}}"
INDEX_PATH="${SEARCH_INDEX_PATH:-/root/search_qa_corpus/e5_Flat.index}"
CORPUS_PATH="${SEARCH_CORPUS_PATH:-/root/search_qa_corpus/wiki-18.jsonl}"
CONDA_HOME="${CONDA_HOME:-/root/miniconda3}"
SEARCH_CONDA_ENV="${SEARCH_CONDA_ENV:-relax-opd-search}"
READY_MARKER="${SEARCH_RELAX_READY_MARKER:-Destroyed 6 process groups}"
PID_FILE="${SEARCH_RETRIEVER_PID_FILE:-${RETRIEVER_LOG}.pid}"

while true; do
  if [ -f "${EVAL_LOG}" ] && tail -c 4000000 "${EVAL_LOG}" 2>/dev/null | grep -a -F "${READY_MARKER}" >/dev/null; then
    break
  fi
  if ! kill -0 "${EVAL_PID}" 2>/dev/null; then
    echo "Relax evaluation exited before the model-load marker appeared." >&2
    exit 1
  fi
  sleep 1
done

cd "${RELAX_ROOT}"
RETRIEVER_ARGS=(
  --index_path "${INDEX_PATH}"
  --corpus_path "${CORPUS_PATH}"
  --port "${PORT}"
)
if [ "${SEARCH_RETRIEVER_FAISS_GPU:-1}" = "1" ]; then
  RETRIEVER_ARGS+=(--faiss_gpu)
fi

nohup "${CONDA_HOME}/bin/conda" run --no-capture-output -n "${SEARCH_CONDA_ENV}"   python "${SCRIPT_DIR}/retrieval_server.py" "${RETRIEVER_ARGS[@]}"   > "${RETRIEVER_LOG}" 2>&1 &
RETRIEVER_PID=$!
echo "${RETRIEVER_PID}" > "${PID_FILE}"
echo "Retriever launched: pid=${RETRIEVER_PID}, port=${PORT}"

for _ in $(seq 1 120); do
  if curl --noproxy '*' -sS --max-time 3 -o /dev/null     -H 'Content-Type: application/json'     -d '{"queries":["health check"],"topk":1,"return_scores":true}'     "http://127.0.0.1:${PORT}/retrieve"; then
    echo "Retriever ready: pid=${RETRIEVER_PID}, port=${PORT}"
    exit 0
  fi
  if ! kill -0 "${RETRIEVER_PID}" 2>/dev/null; then
    echo "Retriever exited during startup; see ${RETRIEVER_LOG}." >&2
    exit 1
  fi
  sleep 1
done

echo "Retriever readiness timed out; see ${RETRIEVER_LOG}." >&2
exit 1
