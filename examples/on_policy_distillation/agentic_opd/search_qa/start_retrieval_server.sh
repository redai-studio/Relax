#!/usr/bin/env bash
set -euo pipefail

if [[ "${SEARCH_RETRIEVAL_AUTOSTART:-1}" != "1" ]]; then
  exit 0
fi

retrieval_url="${SEARCH_RETRIEVAL_URL:?SEARCH_RETRIEVAL_URL must be set}"
retrieval_url="${retrieval_url%/}"
if [[ "${retrieval_url}" =~ ^http://(127\.0\.0\.1|localhost):([0-9]+)/retrieve$ ]]; then
  retrieval_port="${BASH_REMATCH[2]}"
else
  echo "Search retrieval autostart skipped for non-local URL: ${retrieval_url}" >&2
  exit 0
fi

health_check() {
  env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY -u all_proxy -u ALL_PROXY \
    curl --noproxy '*' -fsS --max-time "${SEARCH_RETRIEVAL_HEALTH_TIMEOUT_SECONDS:-5}" \
    -H 'Content-Type: application/json' \
    -d '{"queries":["health check"],"topk":1,"return_scores":true}' \
    "${retrieval_url}" >/dev/null 2>&1
}

if health_check; then
  echo "Search retrieval service is already healthy at ${retrieval_url}"
  exit 0
fi

retrieval_python="${SEARCH_RETRIEVAL_PYTHON:-}"
if [[ -z "${retrieval_python}" ]]; then
  retrieval_env="${SEARCH_RETRIEVAL_CONDA_ENV:-relax-opd-webshop}"
  if [[ -n "${CONDA_HOME:-}" && -x "${CONDA_HOME}/envs/${retrieval_env}/bin/python" ]]; then
    retrieval_python="${CONDA_HOME}/envs/${retrieval_env}/bin/python"
  else
    retrieval_python="$(command -v python3 || true)"
  fi
fi
if [[ -z "${retrieval_python}" || ! -x "${retrieval_python}" ]]; then
  echo "Unable to find a Python interpreter for the retrieval service" >&2
  exit 1
fi

corpus_dir="${SEARCH_RETRIEVAL_CORPUS_DIR:-/root/search_qa_corpus}"
index_path="${SEARCH_RETRIEVAL_INDEX_PATH:-${corpus_dir}/e5_Flat.index}"
corpus_path="${SEARCH_RETRIEVAL_CORPUS_PATH:-${corpus_dir}/wiki-18.jsonl}"
if [[ ! -f "${index_path}" || ! -f "${corpus_path}" ]]; then
  echo "Retrieval assets are missing: index=${index_path} corpus=${corpus_path}" >&2
  exit 1
fi

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
retrieval_log="${SEARCH_RETRIEVAL_LOG:-${EXP_DIR:-/tmp}/retrieval-server-${retrieval_port}.log}"
mkdir -p "$(dirname "${retrieval_log}")"

env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY -u all_proxy -u ALL_PROXY \
  CUDA_VISIBLE_DEVICES="${SEARCH_RETRIEVAL_CUDA_VISIBLE_DEVICES:-0}" \
  HF_HOME="${HF_HOME:-${HOME}/.cache/huggingface}" \
  HF_HUB_OFFLINE=1 \
  OPENBLAS_NUM_THREADS="${SEARCH_RETRIEVAL_OPENBLAS_THREADS:-1}" \
  OMP_NUM_THREADS="${SEARCH_RETRIEVAL_OMP_THREADS:-128}" \
  nohup "${retrieval_python}" -u "${script_dir}/retriever/retrieval_server.py" \
  --index_path "${index_path}" \
  --corpus_path "${corpus_path}" \
  --port "${retrieval_port}" \
  --topk "${SEARCH_RETRIEVAL_TOPK:-3}" \
  --retriever_name "${SEARCH_RETRIEVER_NAME:-e5}" \
  --retriever_model "${SEARCH_RETRIEVAL_MODEL:-intfloat/e5-base-v2}" \
  --device "${SEARCH_RETRIEVAL_DEVICE:-cuda}" \
  >"${retrieval_log}" 2>&1 &
retrieval_pid=$!

for _ in $(seq 1 "${SEARCH_RETRIEVAL_STARTUP_TIMEOUT_SECONDS:-180}"); do
  if health_check; then
    echo "Started search retrieval service pid=${retrieval_pid} at ${retrieval_url}; log=${retrieval_log}"
    exit 0
  fi
  if ! kill -0 "${retrieval_pid}" 2>/dev/null; then
    echo "Search retrieval service exited during startup; log=${retrieval_log}" >&2
    tail -80 "${retrieval_log}" >&2 || true
    exit 1
  fi
  sleep 1
done

echo "Search retrieval service did not become healthy within startup timeout; log=${retrieval_log}" >&2
exit 1
