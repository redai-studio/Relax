#!/usr/bin/env bash

# Copyright (c) 2026 Relax Authors. All Rights Reserved.

set -e
set -o pipefail

: "${R2E_DATA_PATH:?}"
: "${R2E_SIF_DIR:?}"
: "${MINI_SWE_AGENT_VENV:?}"

if [ ! -f "${MINI_SWE_AGENT_VENV}/bin/activate" ]; then
   echo "ERROR: mini-swe-agent venv not found: ${MINI_SWE_AGENT_VENV}" >&2
   exit 1
fi

source "${MINI_SWE_AGENT_VENV}/bin/activate"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
REGISTRY="${R2E_IMAGE_REGISTRY:-mirror.ccs.tencentyun.com}"
APPTAINER_BIN="${APPTAINER_BIN:-apptainer}"
MAX_JOBS="${R2E_SETUP_MAX_JOBS:-16}"
LOG_DIR="${R2E_SIF_DIR}/logs"

mkdir -p "${R2E_DATA_PATH}" "${R2E_SIF_DIR}" "${LOG_DIR}"

if ! command -v "${APPTAINER_BIN}" >/dev/null 2>&1; then
   echo "ERROR: ${APPTAINER_BIN} not found on PATH." >&2
   exit 1
fi

if [ "${R2E_SETUP_SKIP_DOWNLOADS:-0}" != "1" ]; then
   mkdir -p "${R2E_DATA_PATH}/wheels"
   uv run --no-project --with pip python -m pip download --no-deps --only-binary=:all: \
      -d "${R2E_DATA_PATH}/wheels" chardet==5.2.0

   hf download R2E-Gym/R2E-Gym-Lite --repo-type dataset --include 'data/train-*.parquet' \
      --local-dir "${R2E_DATA_PATH}/R2E-Gym-Lite"
   hf download R2E-Gym/SWE-Bench-Lite --repo-type dataset --include 'data/test-*.parquet' \
      --local-dir "${R2E_DATA_PATH}/SWE-Bench-Lite"
fi

IMAGES="$(python "${SCRIPT_DIR}/check_r2e_sifs.py" \
   --data-path "${R2E_DATA_PATH}" \
   --sif-dir "${R2E_SIF_DIR}" \
   --output missing-images)"
TOTAL_IMAGES="$(printf '%s\n' "${IMAGES}" | sed '/^$/d' | wc -l | tr -d ' ')"

if [ "${TOTAL_IMAGES}" -eq 0 ]; then
   echo "All required R2E SIF images are already present in ${R2E_SIF_DIR}."
   exit 0
fi

echo "Building ${TOTAL_IMAGES} missing R2E SIF images into ${R2E_SIF_DIR}."

build_image() {
   image="$1"
   [ -n "${image}" ] || return 0
   name="$(printf '%s.sif' "${image}" | sed 's#[/:]#__#g')"
   sif_path="${R2E_SIF_DIR}/${name}"
   log_path="${LOG_DIR}/${name}.log"
   if [ -f "${sif_path}" ]; then
      return 2
   fi

   tmp_path="${sif_path}.${BASHPID}.tmp"
   rm -f "${tmp_path}"
   if APPTAINER_BIN="${APPTAINER_BIN}" TMP_PATH="${tmp_path}" IMAGE_URI="docker://${REGISTRY%/}/${image}" \
      bash -lc '"${APPTAINER_BIN}" build "${TMP_PATH}" "${IMAGE_URI}"' >"${log_path}" 2>&1; then
      mv "${tmp_path}" "${sif_path}"
   else
      rm -f "${tmp_path}"
      return 1
   fi
}

show_progress() {
   printf '\r[%d/%d] succeeded=%d skipped=%d failed=%d running=%d' \
      "${completed}" "${TOTAL_IMAGES}" "${succeeded}" "${skipped}" "${errors}" "${running}"
}

reap() {
   set +e
   wait -n
   status=$?
   set -e
   completed=$((completed + 1))
   case "${status}" in
      0) succeeded=$((succeeded + 1)) ;;
      2) skipped=$((skipped + 1)) ;;
      *) failed=1; errors=$((errors + 1)) ;;
   esac
   running=$((running - 1))
   show_progress
}

failed=0
running=0
completed=0
succeeded=0
skipped=0
errors=0
while IFS= read -r image; do
   [ -n "${image}" ] || continue
   build_image "${image}" &
   running=$((running + 1))
   if [ "${running}" -ge "${MAX_JOBS}" ]; then
      reap
   fi
done <<< "${IMAGES}"

while [ "${running}" -gt 0 ]; do
   reap
done

printf '\n'
exit "${failed}"
